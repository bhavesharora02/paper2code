# apps/cli/parse_all.py
"""
CLI to iterate over papers.yaml and run parser for each paper.
Writes per-run JSON into: <out_dir>/<paperid>_<ts>/parsed.json
Usage:
  python -m apps.cli.parse_all --config papers.yaml --out runs/
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Any

import yaml

from core.parser.parse_pdf import parse_pdf_to_dict


def load_papers(cfg_path: str):
    p = Path(cfg_path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("papers", [])


def run_one(paper: Dict[str, Any], out_base: str):
    paper_id = paper.get("id") or Path(paper.get("pdf", "unknown")).stem
    ts = int(time.time())
    run_dir = Path(out_base) / f"{paper_id}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = paper.get("pdf")
    if not pdf_path:
        raise ValueError(f"No pdf path for paper {paper_id} in config.")
    # If path is relative, make relative to repo root
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        # try relative to current working dir
        pdf_path = Path(os.getcwd()) / pdf_path
    # Parse
    print(f"[parse_all] Parsing paper {paper_id} from {pdf_path}")
    parsed = parse_pdf_to_dict(str(pdf_path))
    out_file = run_dir / "parsed.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(parsed, f, indent=2, ensure_ascii=False)
    print(f"[parse_all] Wrote parsed JSON to {out_file}")
    return str(out_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to papers.yaml")
    parser.add_argument("--out", type=str, default="runs", help="Output base directory")
    args = parser.parse_args()

    papers = load_papers(args.config)
    if not papers:
        print("No papers found in config.")
        return

    for paper in papers:
        try:
            run_one(paper, args.out)
        except Exception as e:
            print(f"[parse_all] ERROR for paper {paper.get('id')}: {e}")


if __name__ == "__main__":
    main()
