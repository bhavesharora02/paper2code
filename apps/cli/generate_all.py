# apps/cli/generate_all.py
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

from core.codegen.generate_repo import generate_repo

def _latest_run_dir(runs_root: Path, paper_id: str) -> Path | None:
    # choose the latest directory matching <paper_id>_*
    candidates = sorted([p for p in runs_root.glob(f"{paper_id}_*") if p.is_dir()])
    return candidates[-1] if candidates else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, help='papers.yaml path')
    ap.add_argument('--out', required=True, help='runs/ directory')
    ap.add_argument('--templates', default='core/codegen/templates/', help='templates directory')
    args = ap.parse_args()

    runs_root = Path(args.out)
    templates_dir = Path(args.templates)

    # safe YAML load
    import yaml
    papers_raw = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))

    # Accept older format where file has top-level `papers:` key
    if isinstance(papers_raw, dict) and "papers" in papers_raw:
        papers_raw = papers_raw["papers"]

    if not isinstance(papers_raw, list):
        print("[generate_all] expected a YAML list at top-level in papers.yaml (or a dict with 'papers' key)")
        return

    for item in papers_raw:
        # Accept either string entries (e.g. - cv_vit) or dict entries (e.g. - id: cv_vit)
        if isinstance(item, str):
            paper_id = item
        elif isinstance(item, dict):
            # prefer common keys
            paper_id = item.get('id') or item.get('paper_id') or item.get('name')
            if not paper_id:
                print(f"[generate_all] skipping item without id: {item}")
                continue
        else:
            print(f"[generate_all] unsupported papers.yaml item type: {type(item)} - skipping")
            continue

        run_dir = _latest_run_dir(runs_root, paper_id)
        if not run_dir:
            print(f"[generate_all] no runs found for {paper_id}")
            continue
        ir_path = run_dir / 'ir.json'
        mapping_path = run_dir / 'mapping.json'
        out_dir = run_dir / 'repo'
        # use ASCII arrow to avoid Windows console encoding issues
        print(f"[generate_all] generating repo for {paper_id} -> {out_dir}")
        generate_repo(ir_path, mapping_path, out_dir, templates_dir)

if __name__ == '__main__':
    main()
