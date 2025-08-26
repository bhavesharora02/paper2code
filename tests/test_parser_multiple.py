# tests/test_parser_multiple.py
import json
from pathlib import Path
import yaml
import pytest
import glob

ROOT = Path(__file__).resolve().parents[1]


def find_latest_parsed_for_paper(paper_id: str, runs_dir: Path):
    # find folders matching paper_id_*
    candidates = sorted(runs_dir.glob(f"{paper_id}_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        return None
    parsed = candidates[0] / "parsed.json"
    return parsed if parsed.exists() else None


def test_parser_outputs_exist():
    cfg = yaml.safe_load(open(ROOT / "papers.yaml", "r", encoding="utf-8"))
    papers = cfg.get("papers", [])
    runs_dir = ROOT / "runs"
    assert runs_dir.exists(), "runs/ directory not found. Please run the parse_all CLI first."

    missing = []
    for p in papers:
        pid = p.get("id")
        parsed = find_latest_parsed_for_paper(pid, runs_dir)
        if not parsed:
            missing.append(pid)
            continue
        # load and assert keys
        data = json.loads(parsed.read_text(encoding="utf-8"))
        for key in ["title", "abstract", "method", "pseudocode_blocks"]:
            assert key in data, f"Key {key} missing in parsed.json for {pid}"

    assert not missing, f"No parsed outputs found for papers: {missing}. Run `python -m apps.cli.parse_all --config papers.yaml --out runs/`"
