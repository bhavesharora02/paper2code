# core/codegen/generate_repo.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any
from jinja2 import Environment, FileSystemLoader, select_autoescape


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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
    # If out_dir is runs/<paper_run>/repo, create module path runs.<paper_run>.repo.trainer
    try:
        # attempt to construct module path for readability in README templates
        parts = list(out_dir.resolve().parts)
        # naive approach: find "runs" in path and create a module-like path from there
        if "runs" in parts:
            idx = parts.index("runs")
            module_parts = parts[idx:]  # runs, cv_vit_xxx, repo
            run_module = ".".join(module_parts + ["trainer"])
        else:
            run_module = out_dir.name + ".trainer"
    except Exception:
        run_module = out_dir.name + ".trainer"

    ctx = {
    "ir": ir or {},
    "mapping": mapping or {},
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

    return out_dir
