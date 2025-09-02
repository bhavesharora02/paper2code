# tools/fix_try_block_model_template.py
"""
Patch core/codegen/templates/model.py.jinja to ensure 'try:' block has at least one indented statement
(if mapping.imports is empty). Also patch generated runs/*/repo/model.py files with the same safe pattern
if they show an empty try block.

Run:
  $env:PYTHONPATH = (Get-Location).Path    # (Windows PowerShell)
  python tools/fix_try_block_model_template.py
"""
from pathlib import Path
import re

ROOT = Path(__file__).parent.parent
TEMPLATE = ROOT / "core" / "codegen" / "templates" / "model.py.jinja"

def patch_template(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    # Find the try/imports/except pattern that was causing trouble.
    # We'll replace:
    # try:
    # {% for line in mapping.imports %}
    # {{ line }}
    # {% endfor %}
    # except Exception as e:
    # ...
    #
    # with a guarded version that inserts a pass when mapping.imports is empty:
    #
    # try:
    # {% if mapping.imports %}
    # {% for line in mapping.imports %}
    # {{ line }}
    # {% endfor %}
    # {% else %}
    #     pass
    # {% endif %}
    # except Exception as e:
    #
    new_block = (
        "try:\n"
        "{% if mapping.imports %}\n"
        "{% for line in mapping.imports %}\n"
        "{{ line }}\n"
        "{% endfor %}\n"
        "{% else %}\n"
        "    pass\n"
        "{% endif %}\n"
        "except Exception as e:\n"
    )

    # Very tolerant replace: replace the first "try:" up to "except Exception as e:" occurrence.
    m = re.search(r"try:\s*\n\s*\{%\s*for\s+line\s+in\s+mapping\.imports\s*%}[\s\S]*?except\s+Exception\s+as\s+e\s*:", text)
    if m:
        start, end = m.span()
        new_text = text[:start] + new_block + text[end:]
        path.write_text(new_text, encoding="utf-8")
        print(f"[patch_template] Patched template: {path}")
        return True
    else:
        # If pattern not found, attempt to find a simple "try:" followed by immediate "except"
        if re.search(r"try:\s*\n\s*except\s+Exception\s+as\s+e\s*:", text):
            new_text = re.sub(r"try:\s*\n\s*except\s+Exception\s+as\s+e\s*:", new_block, text, count=1)
            path.write_text(new_text, encoding="utf-8")
            print(f"[patch_template] Patched simple try/except in template: {path}")
            return True
    print(f"[patch_template] No matching try-block found in template: {path}")
    return False

def patch_generated_model(file_path: Path) -> bool:
    text = file_path.read_text(encoding="utf-8")
    # Look for "try:" then immediate import lines or blank then "except..."
    # If a try block has no indented body, or only a blank/newline, we insert '    pass' as a safe placeholder.
    # More defensively: if you have 'try:\n\nexcept Exception as e:' -> insert pass
    if re.search(r"try:\s*\n\s*except\s+Exception\s+as\s+e\s*:", text):
        new_text = re.sub(r"try:\s*\n\s*except\s+Exception\s+as\s+e\s*:", "try:\n    pass\nexcept Exception as e:", text, count=1)
        file_path.write_text(new_text, encoding="utf-8")
        print(f"[patch_generated_model] Patched generated model: {file_path}")
        return True
    # Another case: try:\n<maybe comments/whitespace>\n<no indented import lines>\nexcept...
    # We'll try a more tolerant pattern: if there is a try: and the following lines are not indented imports, insert pass.
    m = re.search(r"try:\s*\n((?:[ \t]*\n){0,2})except\s+Exception\s+as\s+e\s*:", text)
    if m:
        new_text = text[:m.start()] + "try:\n    pass\nexcept Exception as e:\n" + text[m.end():]
        file_path.write_text(new_text, encoding="utf-8")
        print(f"[patch_generated_model] Patched generated model (fallback): {file_path}")
        return True
    # If there appears to be a try with some imports but not indented correctly, we don't attempt heavy fixes here.
    return False

def main():
    patched_any = False
    if TEMPLATE.exists():
        patched_any |= patch_template(TEMPLATE)
    else:
        print(f"[patch_template] Template not found: {TEMPLATE}")

    # Patch existing generated model.py files in runs/*/*/repo/model.py
    runs_dir = ROOT / "runs"
    if runs_dir.exists():
        for run_sub in sorted(runs_dir.glob("*_*")):
            repo_model = run_sub / "repo" / "model.py"
            if repo_model.exists():
                patched_any |= patch_generated_model(repo_model)
    else:
        print("[patch_generated_model] runs/ dir not found; skipping generated files patch.")

    if not patched_any:
        print("[fix_try_block_model_template] Nothing patched (maybe already fixed).")
    else:
        print("[fix_try_block_model_template] Done. Please re-run generate_all and run_smoke_all.")

if __name__ == "__main__":
    main()
