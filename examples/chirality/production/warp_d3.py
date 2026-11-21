"""In-process GPU DFT-D3(BJ) for ASE: nvalchemiops (NVIDIA ALCHEMI toolkit ops) Warp kernels, analytic forces and virial.

Same D3(BJ) form as VASP IVDW=12; r2SCAN parameters s6 1.0, s8 0.7898, a1 0.4948, a2 5.7308 bohr (as torch-dftd
xc="r2scan", used for the ft2 labels and the 137/140 runs). Mirrors nvalchemi.models.dftd3.DFTD3ModelWrapper.forward:
positions and cell in bohr, one "system" per atom for periodic cells, C5-style smoothing from
cutoff*(1 - smoothing_fraction) to cutoff (smoothing_fraction 0 = sharp cutoff, as the Fortran reference).
The CN uses the same neighbour list as the dispersion sum. The kernel computes in float32 internally.

Parameter tensors (rcov, r4r2, c6ab, cn_ref) come from a .pt file made once by
nvalchemi.models.dftd3.extract_dftd3_parameters (Grimme reference sources) and shipped with the package, so the
compute node needs no download.
"""
from __future__ import annotations

import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes

BOHR = 0.529177210544
HARTREE = 27.211386245981
R2SCAN_BJ = {"s6": 1.0, "s8": 0.7898, "a1": 0.4948, "a2": 5.7308}


class WarpD3(Calculator):
    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(self, param_file, cutoff_A=20.0, smoothing_fraction=0.2, device="cuda", damping=R2SCAN_BJ, **kw):
        super().__init__(**kw)
        from nvalchemiops.torch.interactions.dispersion import D3Parameters
        sd = torch.load(param_file, weights_only=True)
        self.dev = torch.device(device)
        self.params = D3Parameters(**{k: sd[k].to(self.dev) for k in ("rcov", "r4r2", "c6ab", "cn_ref")})
        self.cutoff, self.frac, self.damp = float(cutoff_A), float(smoothing_fraction), dict(damping)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        from nvalchemiops.torch.interactions.dispersion import dftd3
        from nvalchemiops.torch.neighbors import neighbor_list
        a = self.atoms
        n = len(a)
        pos = torch.tensor(a.positions, dtype=torch.float64, device=self.dev)
        cell = torch.tensor(np.asarray(a.cell), dtype=torch.float64, device=self.dev).unsqueeze(0)
        pbc = torch.tensor(a.pbc, dtype=torch.bool, device=self.dev)
        nm, _num, shifts = neighbor_list(pos, cutoff=self.cutoff, cell=cell, pbc=pbc)[:3]
        on = self.cutoff * (1.0 - self.frac) if self.frac > 0 else 1e10
        off = self.cutoff if self.frac > 0 else 1e10
        per_atom = torch.arange(n, device=self.dev, dtype=torch.int32)
        e, f, _cn, vir = dftd3(positions=pos / BOHR, numbers=torch.tensor(a.numbers, dtype=torch.int32, device=self.dev),
                               a1=self.damp["a1"], a2=self.damp["a2"], s8=self.damp["s8"], s6=self.damp["s6"],
                               s5_smoothing_on=on / BOHR, s5_smoothing_off=off / BOHR, d3_params=self.params,
                               fill_value=n, batch_idx=per_atom, cell=(cell / BOHR).expand(n, 3, 3).contiguous(),
                               neighbor_matrix=nm.contiguous(), neighbor_matrix_shifts=shifts, compute_virial=True,
                               num_systems=n)
        energy = float(e.to(torch.float64).sum()) * HARTREE
        forces = f.to(torch.float64).cpu().numpy() * (HARTREE / BOHR)
        virial = vir.to(torch.float64).sum(0).cpu().numpy() * HARTREE          # eV
        stress = -virial / a.get_volume()                                        # ASE sign convention
        self.results = {"energy": energy, "free_energy": energy, "forces": forces,
                        "stress": stress[[0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]]}
