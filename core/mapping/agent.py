# core/mapping/agent.py
import json
from typing import Dict, Any, Optional
from pathlib import Path

from core.mapping.catalog import choose_by_ir
from core.llm.gemini_client import GeminiJSON  # this is our compatibility alias

LLM_PROMPT = """You are a senior ML engineer. Map the following paper IR to concrete Python constructors.

Return ONLY JSON (no explanatory text). The JSON schema:
{
  "imports": ["<python import lines>"],
  "framework": "pytorch|transformers|xgboost|other",
  "model_constructor": "<python expression to build the model>",
  "loss_constructor": "<python expression or null>",
  "optimizer_constructor": "<python expression or null>",
  "preprocessing": {
    "image": "<python expr or null>",
    "text_tokenizer": "<python expr or null>"
  },
  "notes": ["short notes or uncertainties"]
}

Be conservative: prefer well-known libraries (PyTorch, torchvision, transformers, xgboost). If uncertain, set the field to null and explain in notes.

IR:
{ir_json}
"""

def _ensure_minimum_fields(d: Dict[str, Any]) -> Dict[str, Any]:
    d = dict(d or {})
    d.setdefault("imports", [])
    d.setdefault("framework", None)
    d.setdefault("model_constructor", None)
    d.setdefault("loss_constructor", None)
    d.setdefault("optimizer_constructor", None)
    d.setdefault("preprocessing", {"image": None, "text_tokenizer": None})
    d.setdefault("notes", [])
    return d

def _merge_catalog_and_llm(cat: Optional[Dict[str, Any]], llm: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    base = _ensure_minimum_fields(cat or {})
    llm = _ensure_minimum_fields(llm or {})
    merged = _ensure_minimum_fields({})
    merged["imports"] = list(dict.fromkeys((base["imports"] or []) + (llm["imports"] or [])))
    merged["framework"] = base["framework"] or llm["framework"]
    merged["model_constructor"] = base["model_constructor"] or llm["model_constructor"]
    merged["loss_constructor"] = base["loss_constructor"] or llm["loss_constructor"]
    merged["optimizer_constructor"] = base["optimizer_constructor"] or llm["optimizer_constructor"]
    merged["preprocessing"] = {
        "image": base["preprocessing"].get("image") or llm["preprocessing"].get("image"),
        "text_tokenizer": base["preprocessing"].get("text_tokenizer") or llm["preprocessing"].get("text_tokenizer"),
    }
    merged["notes"] = list(dict.fromkeys((base["notes"] or []) + (llm["notes"] or [])))
    return merged

def map_ir(ir: Dict[str, Any], use_llm: bool = True, gemini: Optional[GeminiJSON] = None) -> Dict[str, Any]:
    cat = choose_by_ir(ir)
    llm_result = None

    if use_llm:
        try:
            g = gemini or GeminiJSON()
            prompt = LLM_PROMPT.format(ir_json=json.dumps(ir, ensure_ascii=False, indent=2))
            llm_result = g.json_call(prompt)
        except Exception as e:
            llm_result = {"imports": [], "notes": [f"LLM mapping failed: {e}"]}
    else:
        llm_result = {"imports": [], "notes": ["LLM disabled (catalog-only run)"]}

    merged = _merge_catalog_and_llm(cat, llm_result)
    if not merged.get("model_constructor"):
        merged["notes"].append("No model_constructor resolved — please refine IR or extend catalog.")
    return merged

def write_mapping(run_dir: Path, mapping: Dict[str, Any]) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "mapping.json"
    out.write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")
    return out
