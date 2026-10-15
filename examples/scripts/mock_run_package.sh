#!/bin/bash
# Dry-run a staged cluster package locally with a mock VASP (no compute): exercises the run_*.slurm bodies, run_step's
# DONE logic, the static-needs-relax guard and POTCAR assembly. Usage: mock_run_package.sh <package dir> <script>...
# Each script is run once per array task (SLURM_ARRAY_TASK_ID over its --array range), in the order given.
set -eo pipefail
pkg=$(cd "$1" && pwd); shift
mock=$(mktemp -d)
mkdir -p "$mock/bin" "$mock/pot"
for el in Pb_d I O C H N F Cl Br S; do mkdir -p "$mock/pot/$el"; printf "   TITEL  = PAW_PBE %s 01Jan2000\n" "$el" > "$mock/pot/$el/POTCAR"; done
cat > "$mock/bin/vasp_mock" <<'EOF'
#!/bin/bash
# Mock VASP: copies POSCAR to CONTCAR and writes the completion markers run_step checks for.
nsw=$(awk -F= '/^ *NSW/{gsub(/ /,"",$2); print $2}' INCAR); nsw=${nsw:-0}
nelm=$(awk -F= '/^ *NELM *=/{gsub(/ /,"",$2); print $2}' INCAR)
cp POSCAR CONTCAR
{ echo " running mock VASP"; grep TITEL POTCAR;
  for i in 1 2 3; do echo "      LOOP:  cpu time      1.0: real time      1.0"; done
  if [ "${nelm:-150}" -gt 4 ]; then echo " ------------------------ aborting loop because EDIFF is reached ----------------------------------------"; fi
  [ "$nsw" -gt 0 ] && echo " reached required accuracy - stopping structural energy minimisation"
  echo "  energy  without entropy=     -100.000000  energy(sigma->0) =     -100.000000"; } > OUTCAR
echo "   1 F= -.10000000E+03 E0= -.10000000E+03  d E =-.1E+00" > OSZICAR
printf '<?xml version="1.0"?>\n<modeling>\n</modeling>\n' > vasprun.xml
EOF
cp "$mock/bin/vasp_mock" "$mock/bin/vasp_std"; cp "$mock/bin/vasp_mock" "$mock/bin/vasp_gam"; chmod +x "$mock/bin/"*
export PATH="$mock/bin:$PATH"
work=$(mktemp -d); cp -r "$pkg/." "$work/"; cd "$work"
sed -i "s#/home/lgutsev/pot/potpaw_PBE#$mock/pot#" lib_pvpass.sh
# module and srun as shell functions, exported into the scripts' bash
module() { :; }; srun() { "$@"; }; export -f module srun
for s in "$@"; do
  arr=$(sed -n 's/^#SBATCH --array=\([0-9]*\)-\([0-9]*\).*/\1 \2/p' "$s")
  if [ -n "$arr" ]; then set -- $arr; lo=$1; hi=$2; else lo=0; hi=0; fi
  for i in $(seq "$lo" "$hi"); do
    SLURM_SUBMIT_DIR=$work SLURM_ARRAY_TASK_ID=$i bash "$s" > "logs/mock_${s%.slurm}_$i.out" 2>&1 \
      && echo "OK   $s task $i" || echo "FAIL $s task $i (exit $?): $(tail -1 logs/mock_${s%.slurm}_$i.out)"
  done
done
echo "DONE markers: $(find outputs -name DONE | wc -l)"
echo "work dir: $work"
