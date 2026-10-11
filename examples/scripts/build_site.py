import json
from pathlib import Path
b = json.load(open("data/cache/demo_jobs/PbI2rich_PEAI_pose0/bundle.json"))
t = Path("docs/site/template.html").read_text(encoding="utf-8")
t = t.replace("__DATA__", Path("data/cache/viz/structures.json").read_text())
t = t.replace("__EQUATION__", b["equation"]).replace("__FP_SLAB__", b["jobs"]["complex"]["static"]["fingerprint"])
t = t.replace("__INCAR_RELAX__", Path("data/cache/demo_jobs/PbI2rich_PEAI_pose0/complex/relax/INCAR").read_text().strip())
Path("docs/site/pv_passivator_workbench.html").write_text(t, encoding="utf-8")
print(len(t))
