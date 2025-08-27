# tests/test_ir_matrix_vs_goldens.py
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TOP_KEYS = {
    "paper_id","title","domain","task","model","hyperparameters","dataset","expected_metrics","notes","uncertain"
}

def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))

def test_ir_keys_against_goldens():
    goldens = ROOT / "goldens"
    runs = ROOT / "runs"
    if not goldens.exists():
        # No goldens yet, skip gracefully
        return

    for g in goldens.glob("*.ir.json"):
        paper_id = g.stem
        # find latest run ir.json for this paper
        candidates = sorted(runs.glob(f"{paper_id}_*/ir.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            continue
        latest_ir = candidates[0]
        golden = load_json(g)
        latest = load_json(latest_ir)

        # Top-level keys presence (not exact equality; allow extra keys)
        assert TOP_KEYS.issubset(set(latest.keys())), f"Missing top keys in {latest_ir}"

        # Ensure architecture field exists in both
        assert latest.get("model", {}).get("architecture"), f"model.architecture missing in {latest_ir}"
        assert golden.get("model", {}).get("architecture"), f"model.architecture missing in golden {g}"

        # Optional: ensure expected_metrics keys exist (not values)
        if golden.get("expected_metrics"):
            assert isinstance(latest.get("expected_metrics", []), list), "expected_metrics must be a list"
