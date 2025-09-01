# tests/test_smoke_matrix.py
from pathlib import Path
import json
import pytest

def test_all_smoke_reports_exist():
    runs = Path('runs')
    assert runs.exists(), "runs/ directory not found; generate runs first."

    failures = []
    for run_dir in sorted([p for p in runs.glob('*_*') if p.is_dir()]):
        art = run_dir / 'artifacts' / 'verification_report.json'
        if not art.exists():
            failures.append(f"{run_dir.name}: missing report")
            continue
        rep = json.loads(art.read_text(encoding='utf-8'))
        if not rep.get('pass', False):
            failures.append(f"{run_dir.name}: FAILED ({rep.get('notes')})")
    if failures:
        pytest.fail("Smoke failures:\n" + "\n".join(failures))
