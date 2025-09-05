# core/verify/metric_runner.py
"""
Parallel multi-seed metric runner.

For each seed we:
 - make a temporary copy of the repo/ into its own temp dir
 - run `python trainer.py` with SEED set in env (and SCALED_RUN/USE_FAKE_DATA if requested)
 - read metrics.json from the temp repo (if produced)
 - return per-seed results

The main function `run_metric_verification` preserves the previous return format but
adds aggregate fields n_success and n_total, and supports parallel workers.
"""
from __future__ import annotations
import json
import subprocess
import statistics
import time
import tempfile
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
import concurrent.futures
import os


def _read_expected_metrics(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Read expected_metrics from artifacts/verification_report.json if present."""
    vr = run_dir / "artifacts" / "verification_report.json"
    if not vr.exists():
        return None
    try:
        obj = json.loads(vr.read_text(encoding="utf-8"))
        if "expected_metrics" in obj:
            return obj["expected_metrics"]
        if "key" in obj and "target" in obj:
            return {"key": obj["key"], "target": obj.get("target"), "tolerance_pct": obj.get("tolerance_pct")}
        return None
    except Exception:
        return None


def _load_metrics_file(repo_dir: Path) -> Optional[Dict[str, Any]]:
    mpath = repo_dir / "metrics.json"
    if not mpath.exists():
        return None
    try:
        return json.loads(mpath.read_text(encoding="utf-8"))
    except Exception:
        return None


def _safe_get_metric_value(metrics: Dict[str, Any], key: str) -> Optional[float]:
    if metrics is None or key not in metrics:
        return None
    try:
        v = metrics[key]
        return float(v)
    except Exception:
        return None


def _run_seed_in_temp_repo(orig_repo: Path, seed: int, timeout_per_seed: int, scaled_run: bool, keep_temp: bool = False) -> Dict[str, Any]:
    """
    Copy orig_repo -> temp_dir, run trainer.py inside temp_dir with env SEED,
    then read metrics.json and return per-seed result. Cleans up temp_dir unless keep_temp True.
    """
    tempdir = Path(tempfile.mkdtemp(prefix=f"mr_seed_{seed}_"))
    try:
        # copy tree (raises on existence, but tempdir is empty)
        tmp_repo = tempdir / "repo"
        shutil.copytree(orig_repo, tmp_repo)
    except Exception as e:
        # copy failed: return error
        if not keep_temp:
            try:
                shutil.rmtree(tempdir)
            except Exception:
                pass
        return {"seed": seed, "ok": False, "returncode": 1, "stdout": "", "stderr": f"copy failed: {e}", "metrics": None, "tempdir": str(tempdir)}

    trainer = tmp_repo / "trainer.py"
    if not trainer.exists():
        if not keep_temp:
            try:
                shutil.rmtree(tempdir)
            except Exception:
                pass
        return {"seed": seed, "ok": False, "returncode": 127, "stdout": "", "stderr": f"{trainer} not found", "metrics": None, "tempdir": str(tempdir)}

    env = dict(os.environ)
    env["SEED"] = str(seed)
    if scaled_run:
        env["SCALED_RUN"] = "1"
        env["USE_FAKE_DATA"] = "1"

    try:
        proc = subprocess.run(
            ["python", str(trainer)],
            cwd=str(tmp_repo),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_per_seed,
        )
        ok = proc.returncode == 0
        stdout = proc.stdout
        stderr = proc.stderr
        metrics = _load_metrics_file(tmp_repo)
        result = {
            "seed": seed,
            "ok": ok,
            "returncode": proc.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "metrics": metrics,
            "tempdir": str(tempdir),
        }
    except subprocess.TimeoutExpired as te:
        result = {"seed": seed, "ok": False, "returncode": 124, "stdout": "", "stderr": f"timeout: {te}", "metrics": None, "tempdir": str(tempdir)}
    except Exception as e:
        result = {"seed": seed, "ok": False, "returncode": 1, "stdout": "", "stderr": f"exception: {e}", "metrics": None, "tempdir": str(tempdir)}

    # cleanup unless user asked to keep temp dirs
    if not os.environ.get("MR_KEEP_TEMP") and not os.environ.get("MR_KEEP_TEMPDIR"):
        try:
            shutil.rmtree(tempdir)
        except Exception:
            pass

    return result


def run_metric_verification(
    run_dir: str,
    seeds: List[int] = (0, 1, 2),
    timeout_per_seed: int = 120,
    scaled_run: bool = False,
    max_workers: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run trainer.py for multiple seeds in parallel (each seed in its own temp repo copy).

    Args:
      - run_dir: path to runs/<run>
      - seeds: list of integer seeds
      - timeout_per_seed: seconds per seed
      - scaled_run: set scaled env flags for small datasets
      - max_workers: how many seeds to run concurrently (defaults to min(len(seeds), 4))

    Returns same structure as before plus n_success/n_total in aggregate.
    """
    run_path = Path(run_dir)
    repo = run_path / "repo"
    artifacts = run_path / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    report_path = artifacts / "metric_report.json"

    expected = _read_expected_metrics(run_path)
    key = expected["key"] if expected and "key" in expected else None
    tolerance = float(expected.get("tolerance_pct", 0.0)) if expected else None
    target = float(expected.get("target")) if expected and expected.get("target") is not None else None

    per_seed_results: List[Dict[str, Any]] = []
    numeric_values: List[float] = []
    notes: List[str] = []

    # determine workers
    if max_workers is None:
        max_workers = min(max(1, len(seeds)), 4)

    # Submit seed jobs in parallel
    futures = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        for sd in seeds:
            seed = int(sd)
            fut = ex.submit(_run_seed_in_temp_repo, repo, seed, timeout_per_seed, scaled_run, False)
            futures[fut] = seed

        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            # remove tempdir from res before saving if present (we keep included for debug)
            per_seed_results.append({k: v for k, v in res.items() if True})  # keep everything
            metrics = res.get("metrics")
            # prioritize explicit expected key
            if metrics is not None:
                if key:
                    val = _safe_get_metric_value(metrics, key)
                    if val is not None:
                        numeric_values.append(val)
                else:
                    for candidate in ("accuracy", "acc", "top1", "loss"):
                        v = _safe_get_metric_value(metrics, candidate)
                        if v is not None:
                            numeric_values.append(v)
                            break

    # compute aggregate
    agg: Dict[str, Any] = {"mean": None, "std": None, "count": 0, "n_success": 0, "n_total": len(seeds)}
    if numeric_values:
        try:
            agg["mean"] = float(statistics.mean(numeric_values))
            agg["std"] = float(statistics.pstdev(numeric_values)) if len(numeric_values) > 1 else 0.0
            agg["count"] = len(numeric_values)
        except Exception:
            agg["mean"] = None
            agg["std"] = None
            agg["count"] = len(numeric_values)
    else:
        notes.append("No numeric per-seed metrics were found; aggregate not computed.")

    # compute n_success: seeds with ok True
    try:
        agg["n_success"] = sum(1 for r in per_seed_results if bool(r.get("ok")))
    except Exception:
        agg["n_success"] = 0

    cmp = None
    if key and target is not None and agg["mean"] is not None:
        try:
            rel = abs(agg["mean"] - target) / max(1e-12, abs(target))
            rel_pct = rel * 100.0
            # determine direction: later we may prefer "higher_is_better" for accuracy-like keys
            direction = "higher_is_better"
            passed = rel_pct <= (tolerance or 0.0)
            cmp = {
                "pass": bool(passed),
                "expected": {"key": key, "target": target, "tolerance_pct": tolerance},
                "actual": {key: agg["mean"]},
                "relative_pct_diff": rel_pct,
                "direction": direction,
            }
            if not passed:
                notes.append(f"{rel_pct:.2f}% away from target (> tol {tolerance}%).")
        except Exception:
            cmp = {"pass": False, "error": "compare failed"}
    elif key and target is not None and agg["mean"] is None:
        cmp = {"pass": False, "notes": "No aggregate metric to compare."}

    out = {"per_seed": per_seed_results, "aggregate": agg, "cmp": cmp, "notes": notes}
    try:
        report_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    except Exception:
        notes.append("Failed to write metric_report.json")

    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--seeds", default="0,1,2", help="comma separated seeds")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--scaled", action="store_true")
    ap.add_argument("--workers", type=int, default=None, help="max concurrent seed runs")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    res = run_metric_verification(args.run, seeds=seeds, timeout_per_seed=args.timeout, scaled_run=args.scaled, max_workers=args.workers)
    print(json.dumps(res, indent=2))
