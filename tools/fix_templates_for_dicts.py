# tools/fix_templates_for_dicts.py
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
TPL_DIR = ROOT / "core" / "codegen" / "templates"
FILES = [
    "model.py.jinja",
    "data_loader.py.jinja",
    "trainer.py.jinja",
    "config.yaml.jinja",
    "README.jinja",
]

# Ordered replacements (longer/compound first)
REPLACEMENTS = [
    # mapping replacements
    (r"\bmapping\.imports\b", "mapping.get('imports', [])"),
    (r"\bmapping\.model_constructor\b", "mapping.get('model_constructor')"),
    (r"\bmapping\.loss_constructor\b", "mapping.get('loss_constructor')"),
    (r"\bmapping\.optimizer_constructor\b", "mapping.get('optimizer_constructor')"),
    (r"\bmapping\.get\('imports', \[\]\)\.get", "mapping.get('imports', [])"),  # no-op safety

    # ir -> dict-safe access
    (r"\bir\.dataset\.num_classes\b", "ir.get('dataset', {}).get('num_classes')"),
    (r"\bir\.dataset\b", "ir.get('dataset', {})"),
    (r"\bir\.model\.architecture\b", "ir.get('model', {}).get('architecture')"),
    (r"\bir\.model\.task_type\b", "ir.get('model', {}).get('task_type')"),
    (r"\bir\.model\b", "ir.get('model', {})"),
    (r"\bir\.hyperparameters\.batch_size\b", "ir.get('hyperparameters', {}).get('batch_size')"),
    (r"\bir\.hyperparameters\b", "ir.get('hyperparameters', {})"),
    (r"\bir\.task\b", "ir.get('task')"),
    (r"\bir\.domain\b", "ir.get('domain')"),
    (r"\bir\b", "ir"),  # keep ir as-is (avoid removing)
]

def fix_text(text: str) -> str:
    new = text
    for pat, repl in REPLACEMENTS:
        new = re.sub(pat, repl, new)
    # also fix Jinja for-loop line if it still uses attribute form:
    new = re.sub(r"\{%\s*for\s+(\w+)\s+in\s+mapping\.get\('imports', \[\]\)\s*%}", r"{% for \1 in mapping.get('imports', []) %}", new)
    return new

def main():
    if not TPL_DIR.exists():
        print("Templates dir not found:", TPL_DIR)
        return
    for fname in FILES:
        p = TPL_DIR / fname
        if not p.exists():
            print("missing template:", p)
            continue
        txt = p.read_text(encoding="utf-8")
        newtxt = fix_text(txt)
        if newtxt != txt:
            p.write_text(newtxt, encoding="utf-8")
            print("patched", fname)
        else:
            print("no changes for", fname)

if __name__ == "__main__":
    main()
