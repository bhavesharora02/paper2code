# core/orchestrator/flow.py
"""
Orchestrator flow for Week 9.

Responsibilities:
 - maintain a small registry of runs (runs/registry.json)
 - create/generate a repo for a paper (calls apps.cli.generate_all)
 - run smoke and metric stages (calls apps.cli.run_smoke_all and core.verify.metric_runner)
 - record stage logs and status under runs/<paper_run>/artifacts/orchestrator.json
 - simple synchronous orchestration (no async workers yet)
"""
from __future__ import annotations
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import sys
import subprocess
from typing import Optional, List, Dict, Any
import sys
import subprocess
from typing import Optional

def start_orchestrator_subprocess(paper_id: str, run_dir: str, no_smoke: bool = False, no_metrics: bool = False, generate: bool = False, timeout: int = 5) -> Dict[str, Any]:
    """
    Start the orchestrator as a short-lived subprocess for tests:
      - spawns `python -m core.orchestrator.flow --paper-id <pid> --run-dir <run_dir> ...`
      - waits up to `timeout` seconds for process to exit (tests use short timeout)
      - returns dict with keys: ok, returncode, stdout, stderr
    """
    cmd = [sys.executable, "-m", "core.orchestrator.flow", "--paper-id", paper_id, "--run-dir", run_dir]
    if no_smoke:
        cmd += ["--no-smoke"]
    if generate:
        cmd += ["--generate"]
    if not no_metrics:
        cmd += ["--metrics"]
    # run in background for test but capture output
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return {"ok": proc.returncode == 0, "returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}
    except subprocess.TimeoutExpired:
        return {"ok": False, "returncode": 124, "stdout": "", "stderr": "timeout"}
    except Exception as e:
        return {"ok": False, "returncode": 1, "stdout": "", "stderr": str(e)}

REG_PATH = Path("runs") / "registry.json"


def _ensure_runs_dir():
    Path("runs").mkdir(exist_ok=True)


def _load_registry() -> Dict[str, Any]:
    _ensure_runs_dir()
    if not REG_PATH.exists():
        return {}
    try:
        return json.loads(REG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_registry(reg: Dict[str, Any]):
    _ensure_runs_dir()
    REG_PATH.write_text(json.dumps(reg, indent=2), encoding="utf-8")


def _write_artifact(run_dir: Path, key: str, data: Any):
    art = run_dir / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    (art / f"orchestrator_{key}.json").write_text(json.dumps(data, indent=2), encoding="utf-8")


def generate_repo_for_paper(ir_path: str, mapping_path: str, out_dir: str, templates_dir: Optional[str] = None) -> Dict[str, Any]:
    """Wrapper to call the existing generate_all (but for single IR) - adapted to your project structure.
    For compatibility we call apps.cli.generate_all to generate all repos; if you already have a single-run generator,
    swap this out or call core.codegen.generate_repo directly.
    """
    out = {"ok": False, "msg": "", "out_dir": out_dir}
    try:
        # You already have apps.cli.generate_all that writes into runs/ directories.
        # We'll call it for convenience — it accepts --config to point to a papers.yaml, but for single-run
        # usage you might prefer to call core.codegen.generate_repo directly. Here we shell out to the existing CLI.
        cmd = ["python", "-m", "apps.cli.generate_all", "--config", ir_path, "--out", out_dir]
        if templates_dir:
            cmd += ["--templates", templates_dir]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        out["returncode"] = proc.returncode
        out["stdout"] = proc.stdout
        out["stderr"] = proc.stderr
        out["ok"] = proc.returncode == 0
    except subprocess.TimeoutExpired as te:
        out["ok"] = False
        out["msg"] = f"generate timeout: {te}"
    except Exception as e:
        out["ok"] = False
        out["msg"] = f"generate failed: {e}"
    return out


def run_smoke_for_run(run_dir: str, timeout: int = 300) -> Dict[str, Any]:
    """Call apps.cli.run_smoke_all for a single run. We call the programmatic CLI in subprocess for isolation."""
    out = {"ok": False, "msg": ""}
    try:
        cmd = ["python", "-m", "apps.cli.run_smoke_all", "--config", "papers.yaml", "--out", run_dir, "--timeout", str(timeout)]
        # Note: run_smoke_all in your project expects config and out; calling it this way will re-generate if needed.
        # Instead to run a single repo you may call apps.cli.run_smoke directly if available. We keep a robust approach.
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 30)
        out["returncode"] = proc.returncode
        out["stdout"] = proc.stdout
        out["stderr"] = proc.stderr
        out["ok"] = proc.returncode == 0
    except subprocess.TimeoutExpired as te:
        out["ok"] = False
        out["msg"] = f"smoke timeout: {te}"
    except Exception as e:
        out["ok"] = False
        out["msg"] = f"smoke failed: {e}"
    return out


def run_metric_verification_for_run(run_dir: str, seeds: List[int] = [0, 1, 2], timeout_per_seed: int = 120, scaled: bool = False) -> Dict[str, Any]:
    """Call the metric runner module programmatically (import) to collect metrics."""
    out = {"ok": False, "report": None, "msg": ""}
    try:
        # import locally so we run in-process and capture return value
        from core.verify.metric_runner import run_metric_verification
        report = run_metric_verification(run_dir, seeds=seeds, timeout_per_seed=timeout_per_seed, scaled_run=scaled)
        out["ok"] = True
        out["report"] = report
    except Exception as e:
        out["ok"] = False
        out["msg"] = f"metric runner exception: {e}"
    return out


def orchestrate_paper(paper_id: str, run_dir: str, do_generate: bool = False, do_smoke: bool = True, do_metrics: bool = False, metric_seeds: List[int] = [0, 1, 2]) -> Dict[str, Any]:
    """
    High-level orchestration for one paper:
      - optional generation step (if do_generate True)
      - run smoke
      - if requested run metrics
      - record registry entry and artifacts
    """
    reg = _load_registry()
    entry = reg.get(paper_id, {})
    entry.setdefault("paper_id", paper_id)
    entry.setdefault("last_updated", int(time.time()))
    entry.setdefault("history", [])
    run_path = Path(run_dir)
    results = {"paper_id": paper_id, "run_dir": run_dir, "stages": {}}

    if do_generate:
        res_gen = generate_repo_for_paper(ir_path=paper_id, mapping_path="mapping.json", out_dir=run_dir)
        results["stages"]["generate"] = res_gen
        entry["history"].append({"ts": int(time.time()), "stage": "generate", "result": res_gen})

    if do_smoke:
        res_smoke = run_smoke_for_run(run_dir, timeout=300)
        results["stages"]["smoke"] = res_smoke
        entry["history"].append({"ts": int(time.time()), "stage": "smoke", "result": res_smoke})
        _write_artifact(run_path, "smoke", res_smoke)

    if do_metrics:
        res_metrics = run_metric_verification_for_run(run_dir, seeds=metric_seeds, timeout_per_seed=120, scaled=False)
        results["stages"]["metrics"] = res_metrics
        entry["history"].append({"ts": int(time.time()), "stage": "metrics", "result": res_metrics})
        _write_artifact(run_path, "metrics", res_metrics)

    entry["last_updated"] = int(time.time())
    entry["last_result"] = results
    reg[paper_id] = entry
    _save_registry(reg)

    # also write final orchestrator artifact
    _write_artifact(run_path, "final", results)
    return results


if __name__ == "__main__":
    # Simple CLI for testing
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-id", required=True, help="paper id (as in papers.yaml / runs folder)")
    ap.add_argument("--run-dir", required=True, help="target runs/<paper_run> path")
    ap.add_argument("--no-smoke", dest="do_smoke", action="store_false")
    ap.add_argument("--metrics", action="store_true")
    ap.add_argument("--generate", action="store_true")
    args = ap.parse_args()
    res = orchestrate_paper(args.paper_id, args.run_dir, do_generate=args.generate, do_smoke=args.do_smoke, do_metrics=args.metrics)
    print(json.dumps(res, indent=2))
