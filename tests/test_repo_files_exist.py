from pathlib import Path
import json


def test_repo_files_exist():
    runs = Path('runs')
    assert runs.exists(), "runs/ not found — generate runs first"

    # find at least one repo
    any_repo = False
    for run in runs.glob('*_*'):
        repo = run / 'repo'
        if not repo.exists():
            continue
        any_repo = True
        for fname in ['model.py', 'trainer.py', 'data_loader.py', 'config.yaml', 'README.md', '__init__.py']:
            assert (repo / fname).exists(), f"missing {fname} in {repo}"
    assert any_repo, "no generated repo found in any runs/*_*/repo"
