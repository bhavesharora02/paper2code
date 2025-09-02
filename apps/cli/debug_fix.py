# apps/cli/debug_fix.py
"""
CLI to run closed-loop debugging for a single run dir.
Usage:
  python -m apps.cli.debug_fix --run runs/cv_vit_1756191425 --attempts 3

Behavior:
  - Run the repo's trainer script (repo/trainer.py) as a subprocess
  - If it fails (non-zero returncode), call debugger.suggest_patches_for_run to get patches
  - Apply first patch with patch_apply.apply_patch, record to artifacts/debug_history.json
  - Re-run trainer, repeat until pass or attempts exhausted
"""
from __future__ import annotations
import argparse
import subprocess
import json
import time
from pathlib import Path
from typing import Any, Dict

# Import local modules
from core.debug.debugger import suggest_patches_for_run
from core.debug.patch_apply import apply_patch


def _run_trainer(run_dir: Path, timeout: int = 120) -> Dict[str, Any]:
    """
    Run the repo/trainer.py as a subprocess **with the repo directory as cwd**.
    Returns dict with keys: ok (bool), returncode (int), stdout (str), stderr (str)
    """
    repo_trainer = run_dir / 'repo' / 'trainer.py'
    if not repo_trainer.exists():
        return {'ok': False, 'returncode': 127, 'stdout': '', 'stderr': f'{repo_trainer} not found'}

    try:
        # IMPORTANT: set cwd to the repo directory so relative writes (metrics.json) go there
        proc = subprocess.run(
            ['python', str(repo_trainer)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str((run_dir / 'repo').resolve())
        )
    except subprocess.TimeoutExpired as e:
        return {'ok': False, 'returncode': 124, 'stdout': e.stdout or '', 'stderr': 'timeout expired'}
    except Exception as e:
        return {'ok': False, 'returncode': 1, 'stdout': '', 'stderr': f'failed to run trainer: {e}'}

    return {'ok': proc.returncode == 0, 'returncode': proc.returncode, 'stdout': proc.stdout, 'stderr': proc.stderr}


def debug_fix_run(run_dir: str, attempts: int = 3, timeout: int = 120) -> bool:
    """
    Closed-loop debug fixer for a single run directory.
    Returns True if the run eventually succeeds within `attempts`, False otherwise.
    """
    run_path = Path(run_dir)
    art = run_path / 'artifacts'
    art.mkdir(parents=True, exist_ok=True)
    history_path = art / 'debug_history.json'
    history = []

    for attempt in range(1, attempts + 1):
        t0 = time.time()
        res = _run_trainer(run_path, timeout=timeout)
        elapsed = time.time() - t0
        entry = {'timestamp': int(time.time()), 'attempt': attempt, 'result_run': res, 'elapsed': elapsed}
        history.append(entry)
        history_path.write_text(json.dumps(history, indent=2), encoding='utf-8')

        if res.get('ok'):
            print(f"[debug_fix] run succeeded on attempt {attempt}")
            return True

        # otherwise ask for patches
        stderr = res.get('stderr') or ''
        ir = None
        try:
            ir_fp = run_path / 'ir.json'
            if ir_fp.exists():
                ir = json.loads(ir_fp.read_text(encoding='utf-8'))
        except Exception:
            ir = None

        patches = suggest_patches_for_run(str(run_path), stderr, ir=ir)
        if not patches:
            print("[debug_fix] no patches suggested — stopping")
            return False

        # apply first patch
        patch = patches[0]
        applied = apply_patch(str(run_path), patch)
        # record the applied-result as another history entry
        history.append({'timestamp': int(time.time()), 'attempt': attempt, 'result': applied, 'elapsed': time.time() - t0})
        history_path.write_text(json.dumps(history, indent=2), encoding='utf-8')

    print(f"[debug_fix] exhausted {attempts} attempts — still failing")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='Path to run directory, e.g. runs/cv_vit_1756191425')
    ap.add_argument('--attempts', type=int, default=3)
    ap.add_argument('--timeout', type=int, default=120)
    args = ap.parse_args()
    ok = debug_fix_run(args.run, attempts=args.attempts, timeout=args.timeout)
    if not ok:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
