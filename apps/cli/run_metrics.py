
# apps/cli/run_metrics.py
"""Simple CLI to run metric verification for one or many runs.
Usage examples:
  python -m apps.cli.run_metrics --run runs/cv_vit_1756191425
  python -m apps.cli.run_metrics --dir runs/ --seeds 0,1,2
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
from core.verify.metric_runner import run_metric_verification


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="single run dir (e.g. runs/cv_vit_12345)")
    ap.add_argument("--dir", help="directory containing multiple run dirs (e.g. runs/)")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--timeout", type=int, default=300)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    if args.run:
        report = run_metric_verification(args.run, seeds=seeds, timeout_per_seed=args.timeout)
        print(json.dumps(report, indent=2))
        return

    if args.dir:
        base = Path(args.dir)
        for p in sorted([p for p in base.iterdir() if p.is_dir()]):
            print(f"Running metrics for {p}")
            report = run_metric_verification(str(p), seeds=seeds, timeout_per_seed=args.timeout)
            print(f"Wrote: {p}/artifacts/metric_report.json\n")
        return

    ap.print_help()


if __name__ == "__main__":
    main()
