# tests/test_ir_validates.py
import json
from pathlib import Path
from core.ir.schema import IR

ROOT = Path(__file__).resolve().parents[1]

def test_ir_files_validate():
    runs = ROOT / "runs"
    assert runs.exists(), "runs/ not found. First run: python -m apps.cli.extract_irs --config papers.yaml --runs_dir runs/"
    ir_files = list(runs.glob("*_*/ir.json"))
    assert ir_files, "No ir.json found. Run extractor CLI first."

    for ir in ir_files:
        data = json.loads(ir.read_text(encoding="utf-8"))
        obj = IR(**data)  # will raise if invalid
        assert obj.paper_id and obj.title and obj.model.architecture, f"IR missing essentials in {ir}"
