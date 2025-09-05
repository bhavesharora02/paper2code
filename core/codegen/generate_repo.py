# core/codegen/generate_repo.py
from __future__ import annotations
import json
import shutil
from pathlib import Path
from typing import Dict, Any
from jinja2 import Environment, FileSystemLoader, select_autoescape


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def generate_repo(ir_path: Path | str, mapping_path: Path | str, out_dir: Path | str, templates_dir: Path | str):
    """
    Render Jinja2 templates into out_dir using ir.json and mapping.json.

    - ir_path, mapping_path: paths to JSON files (ir.json, mapping.json)
    - out_dir: destination folder for generated repo (a package)
    - templates_dir: folder containing jinja templates (*.jinja)
    """
    ir_path = Path(ir_path)
    mapping_path = Path(mapping_path)
    out_dir = Path(out_dir)
    templates_dir = Path(templates_dir)

    # load inputs (safe dicts)
    ir = _load_json(ir_path)
    mapping = _load_json(mapping_path)

    # create output package dir
    _ensure_dir(out_dir)
    # ensure Python package to allow "python -m runs.<run>.repo.trainer" style
    init_file = out_dir / "__init__.py"
    if not init_file.exists():
        init_file.write_text("# auto-generated package init\n", encoding="utf-8")

    # ensure mapping.json is present in generated repo (tests expect it)
    try:
        # Prefer copying the original mapping file if it's an actual file
        if mapping_path.exists():
            shutil.copy2(str(mapping_path), str(out_dir / "mapping.json"))
        elif mapping:
            # Dump the loaded mapping dict to mapping.json for visibility
            (out_dir / "mapping.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    except Exception:
        # non-fatal: continue generation even if writing mapping fails
        pass

    # prepare jinja environment
    env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=select_autoescape(enabled_extensions=("html", "xml")),
        keep_trailing_newline=True,
    )

    # Build a minimal ctx that templates expect
    # Provide both dict-style and attribute-style access in templates by keeping plain dicts;
    # templates should use dict access (e.g. ir.get('dataset')) or safe filters.
    run_dir = str(out_dir.parent / out_dir.name)  # human friendly
    # run_module: dotted module path to execute trainer: derive from runs dir layout
    try:
        parts = list(out_dir.resolve().parts)
        if "runs" in parts:
            idx = parts.index("runs")
            module_parts = parts[idx:]  # runs, cv_vit_xxx, repo
            run_module = ".".join(module_parts + ["trainer"])
        else:
            run_module = out_dir.name + ".trainer"
    except Exception:
        run_module = out_dir.name + ".trainer"

    ctx = {
        "ir": ir if ir is not None else {},
        "mapping": mapping if mapping is not None else {},
        "run_dir": run_dir,
        "run_module": run_module,
    }

    # Render every .jinja file in templates_dir
    for tpl_path in sorted(templates_dir.glob("*.jinja")):
        tpl = env.get_template(tpl_path.name)
        rendered = tpl.render(**ctx)
        outfile = out_dir / tpl_path.with_suffix("").name  # model.py.jinja -> model.py
        # Ensure parent exists
        _ensure_dir(outfile.parent)
        outfile.write_text(rendered, encoding="utf-8")
        print(f"[generate_repo] wrote {outfile}")

    # Ensure README.md exists: some templates produced README (no ext)
    try:
        readme = out_dir / "README"
        md = out_dir / "README.md"
        if readme.exists() and not md.exists():
            # rename to README.md for tests and usability
            readme.rename(md)
            print(f"[generate_repo] renamed README -> README.md in {out_dir}")
    except Exception:
        pass

    return out_dir
