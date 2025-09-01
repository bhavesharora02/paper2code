# tools/patch_runs_trainer_imports.py
from pathlib import Path

ROOT = Path("runs")
if not ROOT.exists():
    print("No runs/ directory")
    raise SystemExit(1)

for run in sorted([p for p in ROOT.glob("*_*") if p.is_dir()]):
    repo = run / "repo"
    trainer = repo / "trainer.py"
    if not trainer.exists():
        print("Missing trainer.py in", repo)
        continue
    txt = trainer.read_text(encoding="utf-8")
    new = txt.replace("from .model import", "from model import").replace("from .data_loader import", "from data_loader import")
    if new != txt:
        trainer.write_text(new, encoding="utf-8")
        print("Patched", trainer)
    else:
        print("No change for", trainer)
