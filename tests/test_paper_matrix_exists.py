# tests/test_paper_matrix_exists.py
import os
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # repo root

def test_paper_files_exist():
    cfg_path = ROOT / "papers.yaml"
    assert cfg_path.exists(), "papers.yaml not found at repo root."

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    papers = cfg.get("papers", [])
    assert papers and isinstance(papers, list), "papers.yaml must contain a 'papers' list."

    missing = []
    for p in papers:
        pdf_path = ROOT / p.get("pdf", "")
        if not pdf_path.exists():
            missing.append(str(pdf_path))

    assert not missing, f"The following PDFs listed in papers.yaml are missing: {missing}\nPlace them under the `samples/` folder or update papers.yaml accordingly."
