# tools/fix_templates_context.py
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TPL_DIR = ROOT / "core" / "codegen" / "templates"
FILES = [
    "model.py.jinja",
    "data_loader.py.jinja",
    "trainer.py.jinja",
    "config.yaml.jinja",
    "README.jinja",
]

# Header to insert that safely defines mapping/ir/header_comment when undefined
SAFE_HEADER = (
    "{% set mapping = mapping if mapping is defined else {} %}\n"
    "{% set ir = ir if ir is defined else {} %}\n"
    "{% set header_comment = header_comment if header_comment is defined else '' %}\n\n"
)

def main():
    if not TPL_DIR.exists():
        print("Templates directory not found:", TPL_DIR)
        return
    for fname in FILES:
        p = TPL_DIR / fname
        if not p.exists():
            print("Skipping missing template:", p)
            continue
        txt = p.read_text(encoding="utf-8")
        if SAFE_HEADER.strip() in txt:
            print("Already patched:", fname)
            continue
        # Insert safe header at top
        new_txt = SAFE_HEADER + txt
        p.write_text(new_txt, encoding="utf-8")
        print("Patched:", fname)

if __name__ == "__main__":
    main()
