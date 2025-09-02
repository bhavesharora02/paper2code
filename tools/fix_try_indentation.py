# tools/fix_try_indentation.py
"""
Fix Jinja template & already-generated model.py files that contain a `try:` followed
by no indented block (causes IndentationError on import).

Usage (PowerShell):
  $env:PYTHONPATH = (Get-Location).Path
  python tools/fix_try_indentation.py

What it does:
 - Patches core/codegen/templates/model.py.jinja to use a guarded jinja `if mapping.imports` block
   so that `try:` always has an indented body (either imports or `pass`).
 - Scans runs/*_* / repo/model.py and:
   - If a `try:` is immediately followed by `except Exception as e:` it inserts an indented `pass`.
   - If a `try:` block contains top-level import lines (no indentation) it indents them by 4 spaces.
 - Backs up changed files to <file>.bak before writing.
"""
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "core" / "codegen" / "templates" / "model.py.jinja"
RUNS_DIR = ROOT / "runs"

def backup(path: Path):
    bak = path.with_suffix(path.suffix + ".bak")
    try:
        shutil.copy2(path, bak)
        print(f"[backup] backed up {path} -> {bak}")
    except Exception as e:
        print(f"[backup] failed to back up {path}: {e}")

def patch_template():
    if not TEMPLATE.exists():
        print(f"[patch_template] template not found: {TEMPLATE}")
        return False
    txt = TEMPLATE.read_text(encoding="utf-8")

    # Looking for the problematic try / for mapping.imports region.
    # We'll replace it with a safe guarded block.
    safe_block = (
        "try:\n"
        "{% if mapping and mapping.imports %}\n"
        "{% for line in mapping.imports %}\n"
        "    {{ line }}\n"
        "{% endfor %}\n"
        "{% else %}\n"
        "    pass\n"
        "{% endif %}\n"
        "except Exception as e:\n"
    )

    # Heuristic replace: find "try:" followed by a Jinja for over mapping.imports until "except Exception as e:"
    pattern = re.compile(r"try:\s*\n\s*\{%\s*for\s+line\s+in\s+mapping\.imports\s*%}[\s\S]*?except\s+Exception\s+as\s+e\s*:", re.M)
    if pattern.search(txt):
        backup(TEMPLATE)
        new = pattern.sub(safe_block, txt, count=1)
        TEMPLATE.write_text(new, encoding="utf-8")
        print(f"[patch_template] patched template: {TEMPLATE}")
        return True

    # If not matched, try simpler pattern: "try:" then nothing then "except Exception as e:"
    simple_pattern = re.compile(r"try:\s*\n\s*except\s+Exception\s+as\s+e\s*:", re.M)
    if simple_pattern.search(txt):
        backup(TEMPLATE)
        new = simple_pattern.sub(safe_block, txt, count=1)
        TEMPLATE.write_text(new, encoding="utf-8")
        print(f"[patch_template] patched simple try/except in template: {TEMPLATE}")
        return True

    print(f"[patch_template] no change (pattern not found) in {TEMPLATE}")
    return False

def indent_block_lines(lines):
    # indent non-empty lines that don't already start with whitespace by 4 spaces
    out = []
    for ln in lines:
        if ln.strip() == "":
            out.append(ln)
        elif re.match(r"^\s", ln):
            out.append(ln)
        else:
            out.append("    " + ln)
    return out

def patch_generated_model(path: Path):
    txt = path.read_text(encoding="utf-8")
    orig = txt

    changed = False

    # Case 1: try:\n <nothing>\n except Exception as e:
    txt, n1 = re.subn(r"try:\s*\n\s*except\s+Exception\s+as\s+e\s*:", "try:\n    pass\nexcept Exception as e:", txt, count=1)
    if n1:
        changed = True

    # Case 2: try:\n<lines that look like imports but not indented>\nexcept ...
    # We will capture the inner block between try: and except and indent non-indented lines.
    m = re.search(r"(try:\s*\n)([\s\S]{0,1000}?)(\nexcept\s+Exception\s+as\s+e\s*:)", txt)
    if m:
        head, body, tail = m.groups()
        # if body already contains an indented line, likely fine. If not, indent anything that looks like
        # top-level import or code by adding 4 spaces.
        body_lines = body.splitlines(keepends=True)
        # detect whether there's at least one properly-indented line (starts with whitespace)
        has_indented = any(re.match(r"^\s", bl) for bl in body_lines if bl.strip() != "")
        if not has_indented:
            new_body_lines = indent_block_lines(body_lines)
            new_body = "".join(new_body_lines)
            new_txt = txt[:m.start()] + head + new_body + tail + txt[m.end():]
            txt = new_txt
            changed = True

    if changed and txt != orig:
        backup(path)
        path.write_text(txt, encoding="utf-8")
        print(f"[patch_generated_model] patched {path}")
    else:
        print(f"[patch_generated_model] no changes needed for {path}")

    return changed

def patch_all_generated():
    if not RUNS_DIR.exists():
        print("[patch_generated_model] runs/ directory not found; skipping generated files")
        return False
    any_changed = False
    for run in sorted(RUNS_DIR.glob("*_*")):
        repo_model = run / "repo" / "model.py"
        if repo_model.exists():
            changed = patch_generated_model(repo_model)
            any_changed = any_changed or changed
    return any_changed

def main():
    print("[fix_try_indentation] Starting patch process...")
    tpatched = patch_template()
    gpatched = patch_all_generated()
    if not (tpatched or gpatched):
        print("[fix_try_indentation] Nothing patched. If you still see IndentationError, open one generated model.py and inspect the try/except block.")
    else:
        print("[fix_try_indentation] Done. Re-run generate_all (if you changed template) and run_smoke_all.")

if __name__ == "__main__":
    main()
