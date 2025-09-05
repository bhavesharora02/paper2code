# tests/test_orchestrator_smoke.py
import json
import shutil
import tempfile
import time
from pathlib import Path
# tests/test_orchestrator_smoke.py
import json
import tempfile
from pathlib import Path
from core.orchestrator.flow import orchestrate_paper

def test_orchestrator_basic_creates_artifact():
    tmp = Path(tempfile.mkdtemp(prefix="orchtest_"))
    run_dir = tmp / "runs" / "tmp_orch_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "repo").mkdir(parents=True, exist_ok=True)
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)

    # call orchestrator with no generate/no smoke to exercise the record-writing
    res = orchestrate_paper("tmp_orch", str(run_dir), do_generate=False, do_smoke=False, do_metrics=False)
    # ensure registry entry / artifact file were written
    final_art = run_dir / "artifacts" / "orchestrator_final.json"
    assert final_art.exists(), "orchestrator_final.json not written"
    data = json.loads(final_art.read_text(encoding="utf-8"))
    assert data.get("paper_id") == "tmp_orch"

def test_orchestrator_smoke_background_run():
    """
    Minimal end-to-end smoke test for the orchestrator:
      - create minimal runs/<run>/repo
      - start orchestrator subprocess with --no-smoke --no-metrics
      - wait for it to finish and assert orchestrator_final.json exists
    """
    tmp = Path(tempfile.mkdtemp(prefix="orchtest_"))
    runs_dir = tmp / "runs"
    run_name = "tmp_orch_run"
    run_dir = runs_dir / run_name
    repo = run_dir / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)

    # minimal trainer (not executed in this test because orchestrator invoked with --no-smoke)
    trainer = "print('dummy trainer - not executed by this test')\n"
    (repo / "trainer.py").write_text(trainer, encoding="utf-8")

    # import helper and start orchestrator process
    from core.orchestrator.flow import start_orchestrator_subprocess

    info = start_orchestrator_subprocess("tmp_paper", str(run_dir), do_generate=False, do_smoke=False, do_metrics=False)
    proc = info.get("proc")
    assert proc is not None, "Failed to start orchestrator subprocess"

    # wait for process to complete with a reasonable timeout
    try:
        stdout, stderr = proc.communicate(timeout=20)
    except Exception:
        proc.kill()
        stdout, stderr = proc.communicate(timeout=5)

    final_artifact = run_dir / "artifacts" / "orchestrator_final.json"

    # small wait loop in case file write is slightly delayed
    for _ in range(10):
        if final_artifact.exists():
            break
        time.sleep(0.2)

    assert final_artifact.exists(), f"Expected orchestrator_final.json to exist. stdout: {stdout}\nstderr: {stderr}"

    # minimal content check
    try:
        obj = json.loads(final_artifact.read_text(encoding="utf-8"))
    except Exception as e:
        raise AssertionError(f"orchestrator_final.json is not valid JSON: {e}")

    assert isinstance(obj, dict)

    # cleanup
    try:
        shutil.rmtree(tmp)
    except Exception:
        pass
