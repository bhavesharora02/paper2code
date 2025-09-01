# core/verify/smoke_runner.py
"""
Smoke runner for a generated repo.

Responsibilities:
 - launch the generated trainer in smoke mode (or default trainer.py)
 - wait for completion (with timeout)
 - collect metrics.json produced by the trainer
 - compare metrics to expected values from IR (if present)
 - write runs/<run>/artifacts/verification_report.json

This runner prefers to invoke trainer.py directly:
  (cd runs/<run>/repo && python trainer.py --mode smoke)
Trainer must write metrics.json in its repo folder.
"""

from __future__ import annotations
import json
import subprocess
import shlex
import time
from pathlib import Path
from typing import Dict, Any, Optional

DEFAULT_TIMEOUT = 300  # seconds for smoke run (5 minutes)


def _load_json(p: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def compare_metrics(expected: Dict[str, Any], actual: Dict[str, Any]) -> Dict[str, Any]:
    """
    expected: { "key": "...", "target": float, "tolerance_pct": float }
    actual: metric dict (e.g. {"accuracy": 0.9, "loss": 0.3})
    Returns result dict with pass flag and details.
    """
    if not expected:
        return {"pass": True, "notes": "No expected metric in IR; auto-pass (no check performed)."}

    key = expected.get("key")
    if key is None:
        return {"pass": True, "notes": "Expected metric missing 'key' - skipping check."}

    target = expected.get("target")
    tol_pct = expected.get("tolerance_pct", 5.0)

    if key not in actual:
        return {"pass": False, "notes": f"Actual metrics do not contain expected key '{key}'.", "actual": actual}

    actual_val = actual[key]
    # handle possible percent vs fraction confusion: assume both targets and actuals are in [0,1] or [0,100]
    try:
        diff = abs(actual_val - target)
        # relative percent difference
        rel = (diff / target) * 100.0 if target not in (0, None) else (diff * 100.0)
    except Exception:
        rel = float("inf")

    passed = rel <= float(tol_pct)
    return {
        "pass": bool(passed),
        "expected": {"key": key, "target": target, "tolerance_pct": tol_pct},
        "actual": {key: actual_val},
        "relative_pct_diff": rel,
        "notes": None if passed else f"{rel:.2f}% away from target (> tol {tol_pct}%)."
    }


def run_trainer_smoke(repo_dir: Path, timeout: int = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    """
    Run trainer.py from repo_dir and return a result dict:
    {
      "ok": bool, "returncode": int, "elapsed": float, "stdout": str, "stderr": str,
      "metrics": {...} or None, "error": str or None
    }
    """
    repo_dir = Path(repo_dir)
    trainer_py = repo_dir / "trainer.py"
    if not trainer_py.exists():
        return {"ok": False, "error": "trainer.py not found", "metrics": None}

    # run in a subprocess, from the repo directory
    cmd = ["python", "trainer.py"]
    start = time.time()
    try:
        proc = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True, timeout=timeout)
        elapsed = time.time() - start
        stdout = proc.stdout
        stderr = proc.stderr
        rc = proc.returncode
    except subprocess.TimeoutExpired as e:
        return {"ok": False, "error": f"timeout after {timeout}s", "stdout": e.stdout or "", "stderr": e.stderr or ""}

    # Look for metrics.json in repo after run
    metrics_path = repo_dir / "metrics.json"
    metrics = _load_json(metrics_path) if metrics_path.exists() else None

    return {
        "ok": rc == 0,
        "returncode": rc,
        "elapsed": elapsed,
        "stdout": stdout,
        "stderr": stderr,
        "metrics": metrics,
        "metrics_path": str(metrics_path) if metrics else None
    }


def run_smoke_for_run_dir(run_dir: Path, timeout: int = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    """
    run_dir: path to runs/<paperid>_<ts>
    Produces a verification report dict and writes it to runs/.../artifacts/verification_report.json
    """
    run_dir = Path(run_dir)
    repo_dir = run_dir / "repo"
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    ir = _load_json(run_dir / "ir.json") or {}
    expected_metrics = None
    em = ir.get("expected_metrics")
    # expected_metrics can be a list; pick first
    if isinstance(em, list) and em:
        expected_metrics = em[0]
    elif isinstance(em, dict):
        expected_metrics = em

    # Execute trainer (smoke)
    res = run_trainer_smoke(repo_dir, timeout=timeout)

    # Build verification result
    verification = {
        "paper_id": ir.get("paper_id", run_dir.name),
        "run_dir": str(run_dir),
        "ok": res.get("ok", False),
        "returncode": res.get("returncode"),
        "elapsed": res.get("elapsed"),
        "stdout_preview": (res.get("stdout") or "")[:2000],
        "stderr_preview": (res.get("stderr") or "")[:2000],
        "metrics": res.get("metrics"),
        "expected_metrics": expected_metrics,
        "pass": None,
        "notes": []
    }

    if not res.get("ok"):
        verification["pass"] = False
        verification["notes"].append("trainer execution failed or returned non-zero code.")
    else:
        # If we have expected metrics, compare
        if expected_metrics and verification["metrics"]:
            cmp = compare_metrics(expected_metrics, verification["metrics"])
            verification["pass"] = bool(cmp.get("pass"))
            verification["cmp"] = cmp
            if not cmp.get("pass"):
                verification["notes"].append(cmp.get("notes"))
        elif expected_metrics and not verification["metrics"]:
            verification["pass"] = False
            verification["notes"].append("Expected metrics but trainer did not produce metrics.json.")
        else:
            # No expected metric -> treat as pass if trainer succeeded
            verification["pass"] = True
            verification["notes"].append("No expected metric provided; trainer succeeded -> marked pass.")

    # Write verification report
    out = artifacts_dir / "verification_report.json"
    out.write_text(json.dumps(verification, indent=2), encoding="utf-8")
    return verification
