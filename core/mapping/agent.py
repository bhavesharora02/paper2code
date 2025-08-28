# core/mapping/agent.py (patched)
import json
import time
import random
from typing import Dict, Any, Optional
from pathlib import Path

from core.mapping.catalog import choose_by_ir
from core.llm.gemini_client import GeminiJSON  # compatibility alias

# Prompt with examples (keep as-is)
LLM_PROMPT = """
You are a senior ML engineer. Map the following paper IR to concrete Python constructors.
Return ONLY JSON fenced in ```json ... ``` and nothing else.

The JSON schema (example):
{
  "imports": ["<python import lines>"],
  "framework": "pytorch|transformers|xgboost|other",
  "model_constructor": "<python expression to build the model>",
  "loss_constructor": "<python expression or null>",
  "optimizer_constructor": "<python expression or null>",
  "preprocessing": {"image": "<python expr or null>", "text_tokenizer": "<python expr or null>"},
  "notes": ["short notes or uncertainties"]
}

Example 1 (ViT):
```json
{
  "imports": ["import torch", "from transformers import ViTForImageClassification, ViTImageProcessor"],
  "framework": "transformers",
  "model_constructor": "ViTForImageClassification.from_pretrained(\"google/vit-base-patch16-224\", num_labels=1000)",
  "loss_constructor": "torch.nn.CrossEntropyLoss()",
  "optimizer_constructor": "torch.optim.AdamW(model.parameters(), lr=5e-05, weight_decay=0.0)",
  "preprocessing": {"image": "ViTImageProcessor.from_pretrained(\"google/vit-base-patch16-224\")", "text_tokenizer": null},
  "notes": ["Catalog-style mapping"]
}
```

Example 2 (BERT sequence classification):
```json
{
  "imports": ["import torch", "from transformers import BertForSequenceClassification, AutoTokenizer"],
  "framework": "transformers",
  "model_constructor": "BertForSequenceClassification.from_pretrained(\"bert-base-uncased\", num_labels=2)",
  "loss_constructor": "torch.nn.CrossEntropyLoss()",
  "optimizer_constructor": "torch.optim.AdamW(model.parameters(), lr=2e-05, weight_decay=0.0)",
  "preprocessing": {"image": null, "text_tokenizer": "AutoTokenizer.from_pretrained(\"bert-base-uncased\")"},
  "notes": ["Example mapping"]
}
```

Now map this IR (below). Be conservative: if unsure, set fields to null and explain in notes.
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
    # imports: union preserving order
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


def _normalize_llm_response(resp: Any) -> Optional[Dict[str, Any]]:
    """
    Enhanced normalizer for LLM outputs.
    Tries (in order):
      1) If resp is dict -> return
      2) If resp is list of dicts -> return first
      3) If resp is string-like -> extract fenced JSON or first {...} block
      4) Try json.loads on candidate
      5) Try to fix common issues: smart quotes, single quotes -> double, remove trailing commas
      6) Try ast.literal_eval on candidate (safe eval for python literals)
      7) Try nested wrapper shapes like {"choices": [{"text": "..."}]}
    Returns dict or None.
    """
    import re, ast

    # 1) direct dict
    if isinstance(resp, dict):
        return resp

    # 2) list of dicts
    if isinstance(resp, list) and resp and isinstance(resp[0], dict):
        return resp[0]

    # Get text representation
    if isinstance(resp, str):
        text = resp
    else:
        try:
            text = json.dumps(resp, ensure_ascii=False)
        except Exception:
            try:
                text = str(resp)
            except Exception:
                return None

    # helper to try parse candidate string into dict
    def try_parse(candidate: str) -> Optional[Dict[str, Any]]:
        candidate = candidate.strip()
        if not candidate:
            return None
        # 1) direct json.loads
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        # 2) fix common problems: smart quotes -> normal quotes
        cand = candidate.replace("“", "\"").replace("”", "\"").replace("‘", "'").replace("’", "'")
        # remove control chars
        cand = re.sub(r"[\x00-\x1f]+", " ", cand)
        # 3) convert single-quoted dicts to double quotes (best-effort)
        cand2 = None
        if "'" in cand and '"' not in cand:
            cand2 = cand.replace("'", "\"")
            try:
                parsed = json.loads(cand2)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
        # 4) remove trailing commas (both in objects and arrays)
        cand3 = re.sub(r",\s*([}\]])", r"\1", cand)
        try:
            parsed = json.loads(cand3)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        # 5) try ast.literal_eval (python literal dict)
        try:
            python_obj = ast.literal_eval(candidate)
            if isinstance(python_obj, dict):
                return python_obj
        except Exception:
            pass
        # 6) try on cand2/cand3 variants too
        for candidate_variant in (cand, cand2 if 'cand2' in locals() else None, cand3):
            if not candidate_variant:
                continue
            try:
                python_obj = ast.literal_eval(candidate_variant)
                if isinstance(python_obj, dict):
                    return python_obj
            except Exception:
                pass
        return None

    # 1) fenced ```json ... ```
    m = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text, flags=re.IGNORECASE)
    if m:
        parsed = try_parse(m.group(1))
        if parsed:
            return parsed

    # 2) first {...} block (greedy minimal attempt)
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidate = text[first_brace:last_brace + 1]
        parsed = try_parse(candidate)
        if parsed:
            return parsed

    # 3) wrapper like {"choices":[{"text":"..."}]} or similar JSON string
    try:
        obj = json.loads(text)
    except Exception:
        obj = None

    if isinstance(obj, dict):
        for key in ("choices", "candidates", "outputs", "responses"):
            if key in obj and isinstance(obj[key], list) and obj[key]:
                first = obj[key][0]
                # if this is a dict with 'text' or 'message'
                if isinstance(first, dict):
                    inner_text = first.get("text") or first.get("message") or first.get("content") or ""
                    if inner_text:
                        parsed = _normalize_llm_response(inner_text)
                        if parsed:
                            return parsed
                elif isinstance(first, str):
                    parsed = _normalize_llm_response(first)
                    if parsed:
                        return parsed

    # 4) last-resort: try to extract a {...} using regex and attempt parse on smaller chunks
    for match in re.finditer(r"(\{(?:[^{}]|(?R))*\})", text, flags=re.DOTALL):
        candidate = match.group(1)
        parsed = try_parse(candidate)
        if parsed:
            return parsed

    return None


def _call_llm_with_retries(prompt: str, gemini: Optional[GeminiJSON] = None,
                           max_retries: int = 3, run_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Call the LLM with retries and exponential backoff. Returns parsed JSON dict.
    Always attempts to write a raw trace to run_dir/'raw_llm_mapping.txt' for debugging:
      - if call returns, write the returned object (json.dumps fallback to str)
      - if call raises, write the exception traceback
    """
    import traceback
    g = gemini or GeminiJSON()
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            raw = g.json_call(prompt)
            # Try to serialize the returned object to text
            try:
                raw_txt = json.dumps(raw, ensure_ascii=False, indent=2, default=str)
            except Exception:
                try:
                    raw_txt = str(raw)
                except Exception:
                    raw_txt = "<unserializable raw response>"

            # Robust write (create folder if needed)
            if run_dir is not None:
                try:
                    p = Path(run_dir)
                    p.mkdir(parents=True, exist_ok=True)
                    (p / "raw_llm_mapping.txt").write_text(raw_txt, encoding="utf-8")
                    print(f"[mapping-agent] Wrote raw LLM output to {p/'raw_llm_mapping.txt'}")
                except Exception as write_err:
                    # don't fail the process, but inform
                    print(f"[mapping-agent] Warning: failed to write raw LLM output: {write_err}")

            # Normalize & validate
            parsed = _normalize_llm_response(raw)
            if parsed is None:
                raise ValueError("LLM response could not be normalized into a JSON object/dict")
            if "model_constructor" not in parsed:
                raise ValueError("LLM JSON missing required key 'model_constructor'")
            return parsed

        except Exception as e:
            # On exception, write exception trace to raw file (so we can inspect)
            last_exc = e
            if run_dir is not None:
                try:
                    p = Path(run_dir)
                    p.mkdir(parents=True, exist_ok=True)
                    tb = traceback.format_exc()
                    (p / f"raw_llm_mapping_attempt{attempt}_error.txt").write_text(tb, encoding="utf-8")
                    print(f"[mapping-agent] Wrote LLM exception trace to {p/f'raw_llm_mapping_attempt{attempt}_error.txt'}")
                except Exception as log_err:
                    print(f"[mapping-agent] Warning: failed to write exception trace: {log_err}")

            backoff = 0.8 * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
            print(f"[mapping-agent] LLM attempt {attempt} failed: {e}. Backing off {backoff:.1f}s")
            time.sleep(backoff)
            continue

    raise RuntimeError(f"LLM mapping failed after {max_retries} attempts. Last error: {last_exc}")

def _postprocess_llm_mapping(mapping: Dict[str, Any], ir: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tidy up small LLM artifacts in the mapping:
      - Replace occurrences like 'lr=null' with a reasonable default (or remove),
      - If optimizer string contains 'weight_decay' as float-like, keep as-is,
      - Record any automated fixes in mapping['notes'].
    Returns the cleaned mapping (mutates copy).
    """
    m = dict(mapping or {})
    notes = list(m.get("notes") or [])

    # Helper: replace lr=null with sensible default for transformers (5e-5),
    # for CV models use 1e-4 as an example. If IR hyperparameters specify lr, use that.
    def choose_default_lr():
        # try IR hyperparameters if present
        try:
            lr = ir.get("hyperparameters", {}).get("learning_rate")
            if isinstance(lr, (int, float)) or (isinstance(lr, str) and lr.replace(".", "", 1).isdigit()):
                return lr
        except Exception:
            pass
        # fallback by domain-like hint in IR
        dom = (ir.get("domain") or "").lower()
        if "nlp" in dom or "transformer" in (ir.get("model", {}).get("architecture") or "").lower():
            return 5e-05
        if "cv" in dom:
            return 1e-04
        # generic fallback
        return 5e-05

    # Clean optimizer_constructor string if present
    opt = m.get("optimizer_constructor")
    if isinstance(opt, str):
        if "lr=null" in opt or "lr= None" in opt or "lr=None" in opt:
            default_lr = choose_default_lr()
            # if default is numeric, format as Python literal (float)
            lr_literal = default_lr if isinstance(default_lr, (int, float)) else default_lr
            cleaned = opt.replace("lr=null", f"lr={lr_literal}").replace("lr=None", f"lr={lr_literal}")
            m["optimizer_constructor"] = cleaned
            notes.append(f"Auto-filled optimizer lr with {lr_literal} (was null).")
        # fix common literal null inside strings -> None (python)
        if " null" in opt and "lr=" not in opt:
            m["optimizer_constructor"] = opt.replace(" null", " None")
            notes.append("Converted JSON 'null' inside optimizer string to Python None.")

    # If loss_constructor is a null (JSON null), keep as None (Pydantic/CodeGen will interpret)
    if m.get("loss_constructor") is None:
        # nothing to do, but could add note if desired
        pass

    # ensure notes list is updated
    m["notes"] = list(dict.fromkeys(notes + (m.get("notes") or [])))
    return m


def map_ir(ir: Dict[str, Any], use_llm: bool = True, gemini: Optional[GeminiJSON] = None,
           run_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Main mapping function:
      - Try deterministic catalog mapping first
      - If use_llm True, call LLM (with retries) to fill gaps
      - Merge catalog + LLM results (catalog wins on conflicts)
      - Return final mapping dict
    """
    cat = choose_by_ir(ir)
    llm_result = None

    if use_llm:
        try:
            prompt = LLM_PROMPT.replace("{ir_json}", json.dumps(ir, ensure_ascii=False, indent=2))
            llm_result = _call_llm_with_retries(prompt, gemini, run_dir=run_dir)
        except Exception as e:
            # LLM failed; record note and fall back to catalog-only
            llm_result = {"imports": [], "notes": [f"LLM mapping failed: {e}"]}
    else:
        llm_result = {"imports": [], "notes": ["LLM disabled (catalog-only run)"]}

    try:
        llm_result = _postprocess_llm_mapping(llm_result, ir) if isinstance(llm_result, dict) else llm_result
    except Exception as e:
    # non-fatal: record a note that postprocessing failed
        llm_result = llm_result or {}
        llm_result.setdefault("notes", []).append(f"Postprocessing failed: {e}")

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
