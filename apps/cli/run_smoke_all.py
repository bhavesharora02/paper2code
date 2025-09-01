# apps/cli/run_smoke_all.py
from __future__ import annotations
import argparse
from pathlib import Path
import yaml
from core.verify.smoke_runner import run_smoke_for_run_dir

def _latest_run_dir(runs_root: Path, paper_id: str) -> Path | None:
    candidates = sorted([p for p in runs_root.glob(f"{paper_id}_*") if p.is_dir()])
    return candidates[-1] if candidates else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, help='papers.yaml path')
    ap.add_argument('--out', required=True, help='runs/ directory')
    ap.add_argument('--timeout', type=int, default=300, help='timeout (s) per smoke run')
    args = ap.parse_args()

    runs_root = Path(args.out)
    papers_raw = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    # support old structure
    if isinstance(papers_raw, dict) and "papers" in papers_raw:
        papers_raw = papers_raw["papers"]

    for item in papers_raw:
        if isinstance(item, str):
            paper_id = item
        elif isinstance(item, dict):
            paper_id = item.get('id') or item.get('paper_id') or item.get('name')
        else:
            continue
        run_dir = _latest_run_dir(runs_root, paper_id)
        if not run_dir:
            print(f"[run_smoke_all] no run found for {paper_id}, skipping.")
            continue
        print(f"[run_smoke_all] running smoke for {paper_id} -> {run_dir}")
        report = run_smoke_for_run_dir(run_dir, timeout=args.timeout)
        print(f"[run_smoke_all] result for {paper_id} pass={report.get('pass')}, elapsed={report.get('elapsed')}")
    print("[run_smoke_all] done.")

if __name__ == '__main__':
    main()
