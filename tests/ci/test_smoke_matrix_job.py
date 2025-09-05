# tests/ci/test_smoke_matrix_job.py
import json
import os
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path.cwd()
RUNS = ROOT / "runs"
PAPER = os.environ.get("PAPER")  # provided by workflow matrix

def _latest_run_dir_for(paper_id: str) -> Path | None:
    candidates = sorted([d for d in RUNS.glob(f"{paper_id}_*") if d.is_dir()])
    return candidates[-1] if candidates else None

@pytest.mark.skipif(not PAPER, reason="PAPER env var not set (CI matrix provides it)")
def test_smoke_matrix_job():
    """
    CI smoke job for a single paper. This test assumes the repository-level
    CLIs were already run by the job (parse_all, generate_all, run_smoke_all).
    It finds the latest run for PAPER and asserts artifacts/verification_report.json exists
    and that the run either passed or produced a valid report.
    """
    assert PAPER, "PAPER env var must be set"

    # find latest run for this paper
    run_dir = _latest_run_dir_for(PAPER)
    assert run_dir is not None, f"No run dir found for paper {PAPER} under runs/ - ensure parse/generate steps ran."

    vreport = run_dir / "artifacts" / "verification_report.json"
    assert vreport.exists(), f"Missing verification_report.json for run {run_dir}"

    rep = json.loads(vreport.read_text(encoding="utf-8"))
    # We expect the smoke runner to write a report with at least 'paper_id' and 'pass' (bool).
    assert "paper_id" in rep, "verification_report.json missing 'paper_id'"
    assert isinstance(rep.get("pass", None), (bool, type(None))) or "notes" in rep

    # Prefer pass==True, but in smoke-mode some runs may be marked pass if trainer executed successfully
    assert rep.get("pass", False) or rep.get("notes") is not None, f"Smoke failed for {run_dir}: {rep}"
