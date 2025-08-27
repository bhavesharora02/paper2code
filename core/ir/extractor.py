# core/ir/extractor.py
"""
IR extractor (robust):
 - Builds a strict JSON prompt from parsed text
 - Calls Gemini via GeminiJSON (compat wrapper)
 - Sanitizes LLM outputs (numeric coercion, percent -> [0,1], etc.)
 - Validates with Pydantic v2 via IR.model_validate()
 - Returns a Python dict and provides helper to write to disk.
"""

import json
import re
import math
from pathlib import Path
from typing import Any, Dict, Optional, List

from core.llm.gemini_client import GeminiJSON  # our compatibility alias
from core.ir.schema import IR


PROMPT_TEMPLATE = """You are an expert ML engineer. Extract a structured IR (Intermediate Representation) from the given paper content.
Return ONLY valid JSON fenced in ```json ... ``` matching this JSON schema roughly:

{{
  "paper_id": "<string>",
  "title": "<string>",
  "domain": "cv|nlp|ml|other",
  "task": "<string or null>",
  "model": {{
    "architecture": "<e.g. ResNet50, ViT-B/16, BERT-base>",
    "task_type": "<e.g. image-classification, sequence-classification>",
    "layers": [{{"name":"...", "type":"...", "params":{{}} }}],
    "loss": "<e.g. CrossEntropyLoss or MSE or BCEWithLogits>",
    "optimizer": "<e.g. AdamW, SGD>"
  }},
  "hyperparameters": {{
    "learning_rate": <float or null>,
    "batch_size": <int or null>,
    "epochs": <int or null>,
    "weight_decay": <float or null>
  }},
  "dataset": {{
    "name": "<string or null>",
    "train_split": "<string or null>",
    "val_split": "<string or null>",
    "test_split": "<string or null>",
    "input_size": [<ints>],
    "num_classes": <int or null>
  }},
  "expected_metrics": [{{"key":"accuracy|top1|f1", "target": <float>, "tolerance_pct": <float>}}],
  "notes": "<string or null>",
  "uncertain": ["list any fields you are not sure about or missing in the paper"]
}}

Rules:
- Put null for missing numeric values and list such fields inside `uncertain`.
- Convert percentage metrics to decimal (77% -> 0.77).
- Prefer the main experimental configuration (the one used to report headline results).
Now extract IR from the content below.

TITLE:
{title}

ABSTRACT:
{abstract}

METHOD / APPROACH:
{method}

EXPERIMENTS / RESULTS:
{experiments}

PSEUDOCODE (if any):
{pseudocode}
"""


def build_prompt(parsed: Dict[str, Any], paper_id: str) -> str:
    title = parsed.get("title", "") or ""
    abstract = parsed.get("abstract", "") or ""
    method = parsed.get("method", "") or parsed.get("sections", {}).get("method", "") or ""
    experiments = parsed.get("experiments", "") or parsed.get("sections", {}).get("experiments", "") or ""
    pseu = "\n\n---\n\n".join(parsed.get("pseudocode_blocks", [])[:2]) if parsed.get("pseudocode_blocks") else ""
    return PROMPT_TEMPLATE.format(
        title=title[:2000],
        abstract=abstract[:4000],
        method=method[:6000],
        experiments=experiments[:6000],
        pseudocode=pseu[:4000],
    )


def _coerce_float(val: Any) -> Optional[float]:
    """
    Try to convert strings like '0.001', '1e-4', '1e-3', '77%' -> 0.77
    If impossible, return None.
    """
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    if not s:
        return None
    # handle percent
    if s.endswith("%"):
        try:
            num = float(s.rstrip("%").strip())
            return num / 100.0
        except Exception:
            return None
    # try plain float parsing (handles scientific notation)
    try:
        return float(s)
    except Exception:
        # try to extract first numeric substring
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
        if m:
            try:
                return float(m.group(0))
            except Exception:
                return None
    return None


def _coerce_int(val: Any) -> Optional[int]:
    if val is None:
        return None
    if isinstance(val, int) and not isinstance(val, bool):
        return int(val)
    s = str(val).strip()
    if not s:
        return None
    try:
        return int(float(s))
    except Exception:
        # extract first integer-like substring
        m = re.search(r"\d+", s)
        if m:
            try:
                return int(m.group(0))
            except Exception:
                return None
    return None


def _sanitize_raw_ir(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make best-effort cleaning of raw LLM output before feeding to Pydantic.
    - coercions for numeric hyperparams and expected_metrics
    - ensures lists exist
    - appends to uncertain if coercion failed
    """
    if raw is None:
        return {}

    raw = dict(raw)  # shallow copy
    uncertain: List[str] = raw.get("uncertain", []) or []

    # hyperparameters cleaning
    hp = raw.get("hyperparameters", {}) or {}
    if "learning_rate" in hp:
        coerced = _coerce_float(hp.get("learning_rate"))
        if coerced is None and hp.get("learning_rate") not in (None, ""):
            uncertain.append("hyperparameters.learning_rate (could not parse numeric)")
        hp["learning_rate"] = coerced
    if "weight_decay" in hp:
        coerced = _coerce_float(hp.get("weight_decay"))
        if coerced is None and hp.get("weight_decay") not in (None, ""):
            uncertain.append("hyperparameters.weight_decay (could not parse numeric)")
        hp["weight_decay"] = coerced
    if "batch_size" in hp:
        coerced = _coerce_int(hp.get("batch_size"))
        if coerced is None and hp.get("batch_size") not in (None, ""):
            uncertain.append("hyperparameters.batch_size (could not parse int)")
        hp["batch_size"] = coerced
    if "epochs" in hp:
        coerced = _coerce_int(hp.get("epochs"))
        if coerced is None and hp.get("epochs") not in (None, ""):
            uncertain.append("hyperparameters.epochs (could not parse int)")
        hp["epochs"] = coerced
    raw["hyperparameters"] = hp

    # expected_metrics cleaning: ensure list of {key,target,tolerance_pct}
    em = raw.get("expected_metrics", []) or []
    cleaned_em = []
    if isinstance(em, dict):
        # allow single-dict case
        em = [em]
    for m in em:
        try:
            key = m.get("key") if isinstance(m, dict) else None
            target_raw = m.get("target") if isinstance(m, dict) else None
            tol_raw = m.get("tolerance_pct", None) if isinstance(m, dict) else None
            target = _coerce_float(target_raw)
            if target is None and target_raw not in (None, ""):
                uncertain.append(f"expected_metrics.target for {key} (could not parse numeric)")
            tol = _coerce_float(tol_raw) if tol_raw is not None else None
            cleaned_em.append({"key": key or "unknown", "target": target, "tolerance_pct": tol or 2.0})
        except Exception:
            continue
    raw["expected_metrics"] = cleaned_em

    # dataset.num_classes coercion
    ds = raw.get("dataset", {}) or {}
    if "num_classes" in ds:
        nc = _coerce_int(ds.get("num_classes"))
        if nc is None and ds.get("num_classes") not in (None, ""):
            uncertain.append("dataset.num_classes (could not parse int)")
        ds["num_classes"] = nc
    raw["dataset"] = ds

    # ensure model exists and has architecture
    model = raw.get("model", {}) or {}
    if "architecture" not in model or not model.get("architecture"):
        uncertain.append("model.architecture (missing or empty)")
    raw["model"] = model

    # attach uncertain list
    # merge duplicates
    raw_unc = list(dict.fromkeys((raw.get("uncertain") or []) + uncertain))
    raw["uncertain"] = raw_unc

    return raw


def extract_ir_from_parsed(parsed: Dict[str, Any], paper_id: str, gemini: Optional[GeminiJSON] = None) -> Dict[str, Any]:
    """
    Build prompt, call Gemini (expecting JSON), sanitize & validate with Pydantic,
    return a plain Python dict representing the validated IR.
    """
    g = gemini or GeminiJSON()
    prompt = build_prompt(parsed, paper_id)

    raw = g.json_call(prompt)  # may raise on API/parse errors

    # override paper_id/title with parsed authoritative values
    raw["paper_id"] = paper_id
    if parsed.get("title"):
        raw["title"] = parsed["title"]

    # sanitize raw LLM output (coerce numbers, percentages, etc.)
    sanitized = _sanitize_raw_ir(raw)

    # Validate using Pydantic v2 model_validate
    ir_obj = IR.model_validate(sanitized)

    # return as plain dict (model_dump gives a dict)
    return ir_obj.model_dump()


def write_ir(run_dir: Path, ir_dict: Dict[str, Any]):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "ir.json"
    out.write_text(json.dumps(ir_dict, indent=2, ensure_ascii=False), encoding="utf-8")
    return out
