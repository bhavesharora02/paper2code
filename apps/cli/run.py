# apps/cli/run.py
"""
CLI that triggers orchestrator flows.
Usage:
  python -m apps.cli.run --paper cv_vit --run runs/cv_vit_1756191425 --metrics --seeds 0,1,2
  python -m apps.cli.run --all --out runs/  # run smoke for all papers in papers.yaml using existing generate_all/generate_repo
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List
from core.orchestrator.flow import orchestrate_paper, _load_registry

def _read_papers_config(cfg: str = "papers.yaml"):
    p = Path(cfg)
    if not p.exists():
        raise FileNotFoundError(cfg)
    return json.loads(p.read_text(encoding="utf-8")) if p.suffix == ".json" else None  # keep minimal

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", help="paper id (as in papers.yaml)")
    ap.add_argument("--run", help="run dir (e.g. runs/cv_vit_1756191425)")
    ap.add_argument("--metrics", action="store_true")
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--all", action="store_true", help="run smoke/metrics for all papers in papers.yaml (best-effort)")
    ap.add_argument("--seeds", default="0,1,2", help="comma-separated seeds for metric runs")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    if args.paper and args.run:
        res = orchestrate_paper(args.paper, args.run, do_generate=args.generate, do_smoke=True, do_metrics=args.metrics, metric_seeds=seeds)
        print(json.dumps(res, indent=2))
        return

    if args.all:
        # Try to read papers.yaml and iterate. Keep tolerant: if file missing, error.
        import yaml, sys
        cfg = Path("papers.yaml")
        if not cfg.exists():
            print("papers.yaml not found; cannot run --all")
            raise SystemExit(1)
        papers = yaml.safe_load(cfg.read_text(encoding="utf-8")).get("papers", [])
        for p in papers:
            pid = p.get("id")
            if not pid:
                continue
            # derive run_dir by looking for existing run subdir or generate a new one naming
            # find the first runs/<id>_* dir if exists, else create runs/<id>_auto
            runs_root = Path("runs")
            candidates = sorted([d for d in runs_root.glob(f"{pid}_*") if d.is_dir()])
            if candidates:
                run_dir = str(candidates[-1])
            else:
                # fallback naming - this will be created by generate step inside orchestrator if you pass generate=True
                run_dir = str(runs_root / f"{pid}_auto")
            print(f"[orchestrator] running {pid} -> {run_dir}")
            res = orchestrate_paper(pid, run_dir, do_generate=False, do_smoke=True, do_metrics=args.metrics, metric_seeds=seeds)
            print(json.dumps(res, indent=2))
        return

    ap.print_help()

if __name__ == "__main__":
    main()
