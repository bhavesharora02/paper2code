"""
Safe applier for patch proposals. Currently only supports full-file replacements.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import time
import shutil
import tempfile
import os


def _ensure_artifacts(run_dir: Path) -> Path:
    art = run_dir / 'artifacts'
    art.mkdir(parents=True, exist_ok=True)
    (art / 'backups').mkdir(parents=True, exist_ok=True)
    return art


def apply_patch(run_dir: str, patch: Dict[str, Any]) -> Dict[str, Any]:
    run_path = Path(run_dir)
    art = _ensure_artifacts(run_path)

    path = patch.get('path')
    if not path:
        return {'applied': False, 'msg': 'patch missing path'}
    if patch.get('patch_type') != 'full_replace':
        return {'applied': False, 'msg': 'unsupported patch_type'}

    content = patch.get('content')
    if content is None:
        return {'applied': False, 'msg': 'no content provided'}

    target = run_path / path
    if not target.parent.exists():
        target.parent.mkdir(parents=True, exist_ok=True)

    ts = int(time.time())
    bak = art / 'backups' / (Path(path).name + f'.{ts}.orig')
    try:
        if target.exists():
            shutil.copy2(target, bak)
        else:
            bak.write_text('(created by patch)')
    except Exception as e:
        return {'applied': False, 'msg': f'backup failed: {e}'}

    try:
        fd, tmp = tempfile.mkstemp(suffix='.py', text=True)
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(content)
        with open(tmp, 'r', encoding='utf-8') as f:
            src = f.read()
        compile(src, str(tmp), 'exec')
    except Exception as e:
        try:
            if bak.exists():
                shutil.copy2(bak, target)
        except Exception:
            pass
        return {'applied': False, 'msg': f'syntax check failed: {e}', 'notes': patch.get('notes')}
    finally:
        try:
            os.remove(tmp)
        except Exception:
            pass

    try:
        target.write_text(content, encoding='utf-8')
    except Exception as e:
        return {'applied': False, 'msg': f'write failed: {e}'}

    return {'applied': True, 'file_path': path, 'msg': f'Wrote {path} (backup at {bak})', 'notes': patch.get('notes')}
