# tools/fix_model_try_blocks.py
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
TPL = ROOT / "core" / "codegen" / "templates" / "model.py.jinja"
RUNS = ROOT / "runs"

def fix_template():
    if not TPL.exists():
        print("Template not found:", TPL)
        return
    txt = TPL.read_text(encoding="utf-8")

    # We look for the for-loop pattern that renders imports and ensure
    # we add an "if not mapping.get('imports') ... pass" after the loop
    # so the `try:` always has a body even if imports is empty.
    pattern = r"try:\s*\n\s*\{%\s*for\s+line\s+in\s+mapping\.imports\s*%}[\s\S]*?\{%\s*endfor\s*%}\s*\n\s*except\s+Exception\s+as\s+e:"
    m = re.search(pattern, txt)
    if not m:
        print("No looped import block pattern found in template; checking for simpler pattern.")
        # fallback: after the for-endfor, insert guard if not present
        if "{% for line in mapping.imports %}" in txt and "{% endfor %}" in txt:
            txt = txt.replace("{% endfor %}\nexcept Exception as e:",
                              "{% endfor %}\n{% if not mapping.get('imports') %}\n    pass\n{% endif %}\nexcept Exception as e:")
            TPL.write_text(txt, encoding="utf-8")
            print("Patched template with conditional PASS after for-loop.")
        else:
            print("Template did not match expected structure; no change made.")
        return

    # If matched, perform replacement: keep the for loop, and add a guard that inserts pass when empty.
    new_txt = re.sub(r"(\{%\s*endfor\s*%}\s*\n)\s*except\s+Exception\s+as\s+e:",
                     r"\1{% if not mapping.get('imports') %}\n    pass\n{% endif %}\nexcept Exception as e:",
                     txt)
    if new_txt != txt:
        TPL.write_text(new_txt, encoding="utf-8")
        print("Patched template:", TPL)
    else:
        print("Template looks already patched.")

def fix_generated_models():
    if not RUNS.exists():
        print("No runs/ directory, skipping generated file fixes.")
        return
    pattern_try_except = re.compile(r"try:\s*\n\s*\n\s*except\s+Exception\s+as\s+e:", flags=re.MULTILINE)
    for repo in sorted(RUNS.glob("*_*")):
        model_py = repo / "repo" / "model.py"
        if not model_py.exists():
            continue
        txt = model_py.read_text(encoding="utf-8")
        # if we detect an empty try..except pattern, insert a '    pass' line
        if pattern_try_except.search(txt):
            new_txt = pattern_try_except.sub("try:\n    pass\n\nexcept Exception as e:", txt)
            model_py.write_text(new_txt, encoding="utf-8")
            print("Patched generated model.py:", model_py)
        else:
            # also check for 'try:' followed immediately by 'except' on next non-empty line
            if re.search(r"try:\s*\n\s*except\s+Exception\s+as\s+e:", txt):
                new_txt = re.sub(r"try:\s*\n\s*except\s+Exception\s+as\s+e:", "try:\n    pass\n\nexcept Exception as e:", txt)
                model_py.write_text(new_txt, encoding="utf-8")
                print("Patched generated model.py (simple case):", model_py)
            else:
                print("No empty try/except in", model_py.name)

def main():
    fix_template()
    fix_generated_models()
    print("Done. After this, regenerate repos if you want the template change to apply to fresh runs.")

if __name__ == "__main__":
    main()
