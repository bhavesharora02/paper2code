# core/verify/metric_runner.py
from __future__ import annotations
import json
import subprocess
import statistics
import time
import tempfile
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------- helpers ----------
def _read_expected_metrics(run_dir: Path) -> Optional[Dict[str, Any]]:
    vr = run_dir / "artifacts" / "verification_report.json"
    if not vr.exists():
        return None
    try:
        obj = json.loads(vr.read_text(encoding="utf-8"))
        if "expected_metrics" in obj and isinstance(obj["expected_metrics"], dict):
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
        return float(metrics[key])
    except Exception:
        return None


def _is_lower_better_metric_name(key: str) -> bool:
    # heuristics: treat losses/errors as lower-is-better
    k = key.lower()
    return ("loss" in k) or ("error" in k) or ("err" in k)


def _compute_aggregate(values: List[float]) -> Dict[str, Any]:
    agg: Dict[str, Any] = {"mean": None, "std": None, "count": 0, "n_success": 0, "n_total": 0}
    if not values:
        return agg
    try:
        agg["mean"] = float(statistics.mean(values))
        agg["std"] = float(statistics.pstdev(values)) if len(values) > 1 else 0.0
        agg["count"] = len(values)
    except Exception:
        agg["mean"] = None
        agg["std"] = None
        agg["count"] = len(values)
    return agg


# ---------- main runner ----------
def run_metric_verification(
    run_dir: str,
    seeds: List[int] = (0, 1, 2),
    timeout_per_seed: int = 120,
    scaled_run: bool = False,
    use_fake_data_flag: bool = True,
) -> Dict[str, Any]:
    """
    Run trainer.py in an isolated copy of runs/<run>/repo for each seed (so seed runs don't clobber files).
    Collect metrics.json per-seed, compute aggregate stats, compare to expected metrics.

    Returns:
      {
        "per_seed": [ { seed, ok, returncode, stdout, stderr, metrics } ],
        "aggregate": { mean, std, count, n_success, n_total },
        "cmp": {...} or None,
        "notes": [...]
      }
    """
    run_path = Path(run_dir)
    repo_src = run_path / "repo"
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

    # Will count successes (ok True and metric available)
    n_success = 0
    n_total = 0

    for sd in seeds:
        seed = int(sd)
        n_total += 1
        # Make isolated copy of repo for this seed
        with tempfile.TemporaryDirectory(prefix=f"mr_seed_{seed}_") as td:
            repo_copy = Path(td) / "repo"
            try:
                # copytree wants destination not to exist
                shutil.copytree(repo_src, repo_copy)
            except Exception as e:
                # fallback: copy files individually
                repo_copy.mkdir(parents=True, exist_ok=True)
                for p in repo_src.glob("**/*"):
                    if p.is_dir():
                        continue
                    rel = p.relative_to(repo_src)
                    dest = repo_copy / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        shutil.copy2(p, dest)
                    except Exception:
                        pass

            trainer = repo_copy / "trainer.py"
            if not trainer.exists():
                per_seed_results.append(
                    {
                        "seed": seed,
                        "ok": False,
                        "returncode": 127,
                        "stdout": "",
                        "stderr": f"{trainer} not found",
                        "metrics": None,
                    }
                )
                continue

            env = dict(**subprocess.os.environ)
            env["SEED"] = str(seed)
            if scaled_run:
                env["SCALED_RUN"] = "1"
            if use_fake_data_flag:
                # trainers can check USE_FAKE_DATA=1 to use tiny datasets for quick verification.
                env["USE_FAKE_DATA"] = "1"

            # run trainer in the isolated repo_copy
            try:
                proc = subprocess.run(
                    ["python", str(trainer)],
                    cwd=str(repo_copy),
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=timeout_per_seed,
                )
                ok = proc.returncode == 0
                stdout = proc.stdout
                stderr = proc.stderr
                # read metrics from copy repo
                metrics = _load_metrics_file(repo_copy)
                per_seed_results.append(
                    {
                        "seed": seed,
                        "ok": ok,
                        "returncode": proc.returncode,
                        "stdout": stdout,
                        "stderr": stderr,
                        "metrics": metrics,
                    }
                )

                # attempt to extract numeric value for aggregate
                extracted = None
                if metrics:
                    if key:
                        extracted = _safe_get_metric_value(metrics, key)
                    else:
                        # try common candidates (prefer accuracy over loss)
                        for c in ("accuracy", "acc", "top1", "loss"):
                            v = _safe_get_metric_value(metrics, c)
                            if v is not None:
                                extracted = v
                                # if this was a loss we still record it, but mark lower-is-better detection later
                                break

                if extracted is not None:
                    numeric_values.append(extracted)
                    if ok:
                        n_success += 1

            except subprocess.TimeoutExpired as te:
                per_seed_results.append(
                    {"seed": seed, "ok": False, "returncode": 124, "stdout": "", "stderr": f"timeout: {te}", "metrics": None}
                )
            except Exception as e:
                per_seed_results.append({"seed": seed, "ok": False, "returncode": 1, "stdout": "", "stderr": f"exception: {e}", "metrics": None})

    agg = _compute_aggregate(numeric_values)
    # add n_success and n_total to aggregate for CI clarity
    agg["n_success"] = n_success
    agg["n_total"] = n_total

    cmp = None
    # if expected provided and we have aggregate mean, compare respecting direction
    if key and target is not None and agg["mean"] is not None:
        try:
            mean_val = agg["mean"]
            lower_is_better = _is_lower_better_metric_name(key)
            if lower_is_better:
                # compute relative distance for lower-is-better: (target - mean)/|target|
                rel = (target - mean_val) / max(1e-12, abs(target))
            else:
                rel = (mean_val - target) / max(1e-12, abs(target))
            rel_pct = abs(rel) * 100.0
            passed = rel_pct <= (tolerance or 0.0)
            cmp = {
                "pass": bool(passed),
                "expected": {"key": key, "target": target, "tolerance_pct": tolerance},
                "actual": {key: mean_val},
                "relative_pct_diff": rel_pct,
                "direction": "lower_is_better" if lower_is_better else "higher_is_better",
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


# ---------- CLI ----------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--seeds", default="0,1,2", help="comma separated seeds")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--scaled", action="store_true")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    res = run_metric_verification(args.run, seeds=seeds, timeout_per_seed=args.timeout, scaled_run=args.scaled)
    print(json.dumps(res, indent=2))
