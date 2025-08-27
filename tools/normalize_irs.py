# tools/normalize_irs.py
import json
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
PAPERS_YAML = ROOT / "papers.yaml"
GOLDENS_DIR = ROOT / "goldens"
RUNS_DIR = ROOT / "runs"

def load_papers():
    if not PAPERS_YAML.exists():
        return {}
    return {p["id"]: p for p in yaml.safe_load(PAPERS_YAML.read_text())["papers"]}

def normalize_ir(ir: dict, fallback_name: str = None) -> dict:
    ir = dict(ir)  # shallow copy

    # 1) normalize expected_metrics.target to 0-1
    em = ir.get("expected_metrics", [])
    new_em = []
    for m in em:
        key = m.get("key")
        target = m.get("target")
        tol = m.get("tolerance_pct")
        # normalize target: if >1, assume it's percent -> divide by 100
        if target is not None:
            try:
                t = float(target)
                if t > 1:
                    t = t / 100.0
            except Exception:
                t = None
        else:
            t = None
        # normalize tolerance_pct to percent number (e.g., 2.0)
        parsed_tol = None
        if tol is not None:
            try:
                parsed_tol = float(tol)
                if parsed_tol <= 1.0:
                    parsed_tol = parsed_tol * 100.0
            except Exception:
                parsed_tol = None
        new_em.append({
            "key": key or "unknown",
            "target": t,
            "tolerance_pct": parsed_tol if parsed_tol is not None else 2.0
        })
    ir["expected_metrics"] = new_em

    # 2) normalize dataset.input_size to [C,H,W] if ends with 3
    ds = ir.get("dataset", {}) or {}
    inp = ds.get("input_size")
    if isinstance(inp, list) and len(inp) == 3:
        # heuristics: if last == 3 -> H,W,C -> convert to C,H,W
        if inp[-1] == 3:
            ds["input_size"] = [3, inp[0], inp[1]]
        # else if first is 3 already, leave it
    ir["dataset"] = ds

    # 3) fill dataset.name from fallback if missing
    if not ir.get("dataset", {}).get("name") and fallback_name:
        ir["dataset"]["name"] = fallback_name

    # 4) standardize model.architecture string (strip whitespace)
    if ir.get("model", {}).get("architecture"):
        ir["model"]["architecture"] = str(ir["model"]["architecture"]).strip()

    # 5) ensure uncertain is a list and unique
    unc = ir.get("uncertain") or []
    if not isinstance(unc, list):
        unc = [unc]
    # deduplicate preserving order
    seen = set()
    new_unc = []
    for x in unc:
        if not x:
            continue
        if x not in seen:
            seen.add(x)
            new_unc.append(x)
    ir["uncertain"] = new_unc

    return ir

def normalize_file(path: Path, papers_map):
    ir = json.loads(path.read_text(encoding='utf-8'))
    pid = ir.get("paper_id")
    fallback = None
    if pid in papers_map:
        fallback = papers_map[pid].get("dataset_fallback", {}).get("name")
    new = normalize_ir(ir, fallback_name=fallback)
    path.write_text(json.dumps(new, indent=2, ensure_ascii=False), encoding='utf-8')
    print("Normalized:", path)

def walk_and_normalize():
    papers_map = load_papers()
    # goldens
    if GOLDENS_DIR.exists():
        for f in GOLDENS_DIR.glob("*.ir.json"):
            normalize_file(f, papers_map)
    # runs
    if RUNS_DIR.exists():
        for f in RUNS_DIR.rglob("ir.json"):
            normalize_file(f, papers_map)

if __name__ == "__main__":
    walk_and_normalize()
    print("Normalization complete.")
