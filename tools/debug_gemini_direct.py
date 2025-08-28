# tools/debug_gemini_direct.py
import json
import traceback
from pathlib import Path
import sys

# ensure repo root on sys.path (in case PYTHONPATH not set)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.llm.gemini_client import GeminiJSON
from core.mapping.agent import LLM_PROMPT

RUN_DIR = Path("runs/cv_vit_1756191425")

def main():
    # load IR
    ir_path = RUN_DIR / "ir.json"
    if not ir_path.exists():
        print("Missing IR:", ir_path)
        return

    ir = json.loads(ir_path.read_text(encoding="utf-8"))
    prompt = LLM_PROMPT.replace("{ir_json}", json.dumps(ir, ensure_ascii=False, indent=2))

    # instantiate Gemini client (this may raise if GEMINI_API_KEY invalid)
    try:
        g = GeminiJSON()
    except Exception as e:
        tb = traceback.format_exc()
        out = RUN_DIR / "raw_llm_gemini_instantiation_error.txt"
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        out.write_text(tb, encoding="utf-8")
        print("GeminiJSON() instantiation failed — wrote traceback to", out)
        print(tb)
        return

    # call json_call and capture whatever happens
    try:
        resp = g.json_call(prompt)
        # try to serialize resp robustly
        try:
            resp_text = json.dumps(resp, ensure_ascii=False, indent=2, default=str)
        except Exception:
            resp_text = str(resp)
        out = RUN_DIR / "raw_llm_direct_response.txt"
        out.write_text(resp_text, encoding="utf-8")
        print("Wrote direct Gemini response to", out)
        print("Response preview (first 1000 chars):")
        print(resp_text[:1000])
    except Exception as e:
        tb = traceback.format_exc()
        out = RUN_DIR / "raw_llm_direct_exception.txt"
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        out.write_text(tb, encoding="utf-8")
        print("json_call raised an exception — wrote traceback to", out)
        print(tb)

if __name__ == '__main__':
    main()
