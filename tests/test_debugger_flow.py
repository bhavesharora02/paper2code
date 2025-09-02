"""
Simple test that simulates a failing run and ensures debug_fix_run applies a patch and
that the runner succeeds after the patch.
"""
import json
import tempfile
from pathlib import Path
import shutil


def test_debugger_flow(monkeypatch):
    tmp = Path(tempfile.mkdtemp(prefix='dbgtest_'))
    run_dir = tmp / 'runs' / 'tmp_run'
    repo = run_dir / 'repo'
    repo.mkdir(parents=True, exist_ok=True)

    bad_trainer = """import sys
print('running bad trainer')
sys.exit(1)
"""
    (repo / 'trainer.py').write_text(bad_trainer, encoding='utf-8')

    good_trainer = """import json
print('running good trainer')
metrics = {'epoch': 1, 'loss': 0.1, 'accuracy': 1.0}
open('metrics.json','w').write(json.dumps(metrics))
"""

    patch = {'path': 'repo/trainer.py', 'patch_type': 'full_replace', 'content': good_trainer, 'notes': 'make trainer succeed'}

    import core.debug.debugger as debugger_mod
    monkeypatch.setattr(debugger_mod, 'suggest_patches_for_run', lambda run_dir, stderr, ir=None: [patch])

    import apps.cli.debug_fix as cli_mod
    ok = cli_mod.debug_fix_run(str(run_dir), attempts=2, timeout=5)
    assert ok is True

    metrics_path = repo / 'metrics.json'
    assert metrics_path.exists()
    data = json.loads(metrics_path.read_text(encoding='utf-8'))
    assert data.get('accuracy') == 1.0

    shutil.rmtree(tmp)
