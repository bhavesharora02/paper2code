
# tests/test_metric_runner.py
"""Lightweight test for metric_runner.
Creates a temporary run with a repo/trainer.py that writes deterministic metrics based on SEED
and verifies the metric_report.json contains the expected aggregate.
"""
import json
import tempfile
import shutil
from pathlib import Path
from core.verify.metric_runner import run_metric_verification


def test_metric_runner_basic():
    tmp = Path(tempfile.mkdtemp(prefix="mrtest_"))
    run_dir = tmp / "runs" / "tmp_paper_1"
    repo = run_dir / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)

    # trainer: read SEED env var and write metrics.json deterministically
    trainer = r"""import os, json
seed = int(os.environ.get('SEED','0'))
# deterministic metric: accuracy = 0.5 + seed*0.1
acc = 0.5 + seed * 0.1
metrics = {'accuracy': acc, 'epoch': 1}
open('metrics.json','w').write(json.dumps(metrics))
print('wrote metrics', metrics)
"""
    (repo / "trainer.py").write_text(trainer, encoding='utf-8')

    # write a verification_report.json with expected_metrics
    vr = {
        'paper_id': 'tmp_paper',
        'expected_metrics': {'key': 'accuracy', 'target': 0.6, 'tolerance_pct': 10.0}
    }
    (run_dir / 'artifacts' / 'verification_report.json').write_text(json.dumps(vr), encoding='utf-8')

    report = run_metric_verification(str(run_dir), seeds=[0,1,2], timeout_per_seed=5)
    assert 'aggregate' in report
    agg = report['aggregate']
    # mean of [0.5, 0.6, 0.7] -> 0.6
    assert abs(agg['mean'] - 0.6) < 1e-6
    assert agg['n_success'] == 3

    # cleanup
    shutil.rmtree(tmp)