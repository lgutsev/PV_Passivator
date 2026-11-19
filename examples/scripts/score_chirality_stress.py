"""Stress criterion of the chirality gate (plan: within 0.3 GPa on D7): ft2 + D3(BJ, r2SCAN) vs VASP total stress
on the D1 fixed-cell statics (package 105) and the D7 strain statics (package 113). CPU, 4 threads.
"""
import os, json, numpy as np, warnings
os.environ.setdefault("OMP_NUM_THREADS", "4")
import torch; torch.set_num_threads(4)
from ase.io import read
from ase.units import GPa
from mace.calculators import MACECalculator
from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
from ase.calculators.mixing import SumCalculator
warnings.simplefilter("ignore")
P105 = "D:/MLIP_Work_Folder/hpc_smoke_tests/105_vasp_pvpass_mba2pbi4_chirality_d0/outputs"
P113 = "D:/MLIP_Work_Folder/hpc_smoke_tests/113_vasp_pvpass_mba2pbi4_chiral_labels_a/outputs"
mace = MACECalculator(model_paths="D:/MLIP_Custom_Models/MACE/mba2pbi4_chirality_ft2.model", device="cpu", default_dtype="float64")
d3 = TorchDFTD3Calculator(damping="bj", xc="r2scan", device="cpu", dtype=torch.float64)
calc = SumCalculator([mace, d3])
out = {}
for sysn in ("R", "rac"):
    jobs = {"ref": f"{P105}/{sysn}_fixcell/static/OUTCAR"}
    for s in ("m2", "m1", "p1", "p2"): jobs[s] = f"{P113}/{sysn}_D7_strain_{s}/static/OUTCAR"
    res = {}
    for k, f in jobs.items():
        a = read(f, format="vasp-out", index=-1)
        sd = a.get_stress(voigt=True) / GPa          # VASP total stress (incl. D3), GPa
        b = a.copy(); b.calc = calc
        sm = b.get_stress(voigt=True) / GPa
        res[k] = {"dft": sd, "mlip": sm}
        print(sysn, k, "DFT", np.round(sd, 3), "MLIP", np.round(sm, 3), flush=True)
    ref = res["ref"]
    for k, r in res.items():
        err_abs = np.abs(r["mlip"] - r["dft"]).max()
        err_rel = np.abs((r["mlip"] - ref["mlip"]) - (r["dft"] - ref["dft"])).max()
        out[f"{sysn}_{k}"] = {"max_abs_err_GPa": round(float(err_abs), 3), "max_err_vs_ref_GPa": round(float(err_rel), 3),
                              "dft_GPa": np.round(r["dft"], 3).tolist(), "mlip_GPa": np.round(r["mlip"], 3).tolist()}
json.dump(out, open("data/cache/chirality_scores/ft2_stress.json", "w"), indent=1)
for k, v in out.items(): print(k, "abs err %.3f  err vs unstrained %.3f GPa" % (v["max_abs_err_GPa"], v["max_err_vs_ref_GPa"]))
