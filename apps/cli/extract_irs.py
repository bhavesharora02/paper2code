# apps/cli/extract_irs.py
"""
Usage:
  python -m apps.cli.extract_irs --config papers.yaml --runs_dir runs/ --goldens goldens/

For each paper in papers.yaml:
 - find latest runs/<paper_id>_*/parsed.json
 - call IR extractor (Gemini) -> runs/<...>/ir.json
 - if goldens/<paper_id>.ir.json doesn't exist, save first validated IR there
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional
import yaml

from core.ir.extractor import extract_ir_from_parsed, write_ir
from core.ir.schema import IR


def load_papers(cfg_path: str):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("papers", [])


def find_latest_run_with_parsed(runs_dir: Path, paper_id: str) -> Optional[Path]:
    candidates = sorted(runs_dir.glob(f"{paper_id}_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for c in candidates:
        if (c / "parsed.json").exists():
            return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="papers.yaml path")
    ap.add_argument("--runs_dir", default="runs", help="runs dir")
    ap.add_argument("--goldens", default="goldens", help="goldens dir")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    goldens_dir = Path(args.goldens)
    goldens_dir.mkdir(parents=True, exist_ok=True)

    papers = load_papers(args.config)
    if not papers:
        print("No papers in config.")
        return

    for p in papers:
        pid = p.get("id")
        run = find_latest_run_with_parsed(runs_dir, pid)
        if not run:
            print(f"[extract_irs] No parsed run found for {pid}. Run parse_all first.")
            continue

        parsed_path = run / "parsed.json"
        parsed = json.loads(parsed_path.read_text(encoding="utf-8"))
        try:
            ir_dict = extract_ir_from_parsed(parsed, pid)
        except Exception as e:
            print(f"[extract_irs] IR extraction failed for {pid}: {e}")
            continue

        out_path = write_ir(run, ir_dict)
        print(f"[extract_irs] Wrote IR to {out_path}")

        # Save golden if not exists
        golden_path = goldens_dir / f"{pid}.ir.json"
        if not golden_path.exists():
            golden_path.write_text(json.dumps(ir_dict, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"[extract_irs] Saved golden IR snapshot at {golden_path}")


if __name__ == "__main__":
    main()
