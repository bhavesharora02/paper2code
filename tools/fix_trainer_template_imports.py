# tools/fix_trainer_template_imports.py
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TPL = ROOT / "core" / "codegen" / "templates" / "trainer.py.jinja"

if not TPL.exists():
    print("Template not found:", TPL)
    raise SystemExit(1)

txt = TPL.read_text(encoding="utf-8")

# Replace relative imports with absolute ones (safe idempotent replacements)
txt2 = txt.replace("from .model import", "from model import")
txt2 = txt2.replace("from .data_loader import", "from data_loader import")

if txt2 != txt:
    TPL.write_text(txt2, encoding="utf-8")
    print("Patched trainer template imports in", TPL)
else:
    print("No changes needed to trainer template.")
