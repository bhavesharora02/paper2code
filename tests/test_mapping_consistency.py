# tests/test_mapping_consistency.py
import json
from pathlib import Path
import yaml
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def _latest_run_dir(runs_dir: Path, paper_id: str):
    cands = sorted(Path(runs_dir).glob(f"{paper_id}_*/"), key=lambda p: p.name)
    return cands[-1] if cands else None

def test_mapping_has_essentials():
    papers = yaml.safe_load((REPO_ROOT / "papers.yaml").read_text())["papers"]
    for p in papers:
        pid = p["id"]
        run_dir = _latest_run_dir(REPO_ROOT / "runs", pid)
        assert run_dir is not None, f"No run dir for {pid}"
        mpath = run_dir / "mapping.json"
        assert mpath.exists(), f"mapping.json missing for {pid}"
        m = json.loads(mpath.read_text(encoding="utf-8"))
        assert isinstance(m.get("imports"), list), f"imports not list for {pid}"
        assert m.get("model_constructor"), f"model_constructor missing for {pid}"
        pre = m.get("preprocessing", {})
        assert isinstance(pre, dict), f"preprocessing missing for {pid}"
        assert "image" in pre and "text_tokenizer" in pre, f"preprocessing keys missing for {pid}"
