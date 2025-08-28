# tools/debug_map.py
import json
from pathlib import Path
from core.mapping.agent import map_ir, write_mapping, LLM_PROMPT

def run_debug(paper_run_dir: str):
    run_dir = Path(paper_run_dir)
    ir_path = run_dir / "ir.json"
    if not ir_path.exists():
        print("Missing IR:", ir_path)
        return
    ir = json.loads(ir_path.read_text(encoding="utf-8"))
    print("Calling map_ir(...) with run_dir =", run_dir)
    try:
        mapping = map_ir(ir, use_llm=True, run_dir=run_dir)
        print("Mapping result keys:", list(mapping.keys()))
        print("Notes:", mapping.get("notes"))
    except Exception as e:
        print("map_ir raised:", repr(e))

    raw_path = run_dir / "raw_llm_mapping.txt"
    print("raw_llm_mapping.txt exists?", raw_path.exists())
    if raw_path.exists():
        text = raw_path.read_text(encoding="utf-8")
        print("--- RAW LLM OUTPUT (first 2000 chars) ---")
        print(text[:2000])
    else:
        print("No raw_llm_mapping.txt found in run folder.")

if __name__ == "__main__":
    # change this path if you want to debug a different paper run
    run_debug("runs/cv_vit_1756191425")
