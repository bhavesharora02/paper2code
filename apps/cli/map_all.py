# apps/cli/map_all.py
import argparse
import json
from pathlib import Path
import yaml
from core.mapping.agent import map_ir, write_mapping

def latest_run_dir(runs_dir: Path, paper_id: str):
    candidates = sorted(runs_dir.glob(f"{paper_id}_*/"), key=lambda p: p.name)
    return candidates[-1] if candidates else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="papers.yaml")
    ap.add_argument("--out", required=True, help="runs/ directory")
    ap.add_argument("--no_llm", action="store_true", help="Use catalog only (no Gemini calls)")
    args = ap.parse_args()

    papers = yaml.safe_load(Path(args.config).read_text())["papers"]
    for p in papers:
        pid = p["id"]
        run_dir = latest_run_dir(Path(args.out), pid)
        if not run_dir:
            print(f"[map_all] No run dir for {pid}. Run IR extraction first.")
            continue
        ir_path = run_dir / "ir.json"
        if not ir_path.exists():
            print(f"[map_all] Missing IR at {ir_path} for {pid}")
            continue

        ir = json.loads(ir_path.read_text(encoding="utf-8"))
        # pass run_dir so agent can log raw LLM responses into the run folder
        mapping = map_ir(ir, use_llm=(not args.no_llm), run_dir=run_dir)
        out_path = write_mapping(run_dir, mapping)
        print(f"[map_all] Wrote mapping to {out_path}")

if __name__ == "__main__":
    main()
