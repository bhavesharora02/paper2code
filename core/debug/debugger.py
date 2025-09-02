"""
Simple debugger skeleton for Week 7.
This module exposes `suggest_patches_for_run(run_dir, reason)` which **returns a list of patch
proposals**. Each proposal is a dict with keys:
  - path: relative file path inside run_dir to replace (e.g. "repo/model.py")
  - patch_type: currently only "full_replace"
  - content: full file content to write
  - notes: short note why this patch
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional


# add at top of file if not present
import json
import traceback
from core.llm.gemini_client import GeminiJSON  # your repo's wrapper

def _extract_json_candidate(text: str):
    """Try to extract a JSON-like block from text (fenced ```json, [] or {})."""
    # fenced ```json ... ```
    import re
    m = re.search(r'```json\s*([\s\S]*?)```', text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # first [ ... ] or { ... }
    m = re.search(r'(\[[\s\S]*\]|\{[\s\S]*\})', text)
    if m:
        return m.group(1)
    return None

# at top of core/debug/debugger.py
from core.llm.gemini_client import GeminiClient

def _heuristic_patch_for_num_classes_zero(run_dir: Path, stderr: str) -> Optional[Dict[str, Any]]:
    """
    Many failures arise from num_classes being 0 or missing, causing:
      - randint(..., to=0) errors
      - modulo-by-zero when clipping labels
      - 'Target 0 is out of bounds' in cross-entropy
    Offer a robust data_loader.py that clamps num_classes and clips labels.
    """
    triggers = [
        "random_ expects 'from' to be less than 'to'",
        "integer modulo by zero",
        "Target 0 is out of bounds",
        "Target -1 is out of bounds",
        "randint(0, self.num_classes"
    ]
    if any(t in stderr for t in triggers):
        candidate = Path(run_dir) / 'repo' / 'data_loader.py'
        if candidate.exists():
            safe_data_loader = _safe_data_loader_template()
            return {
                'path': 'repo/data_loader.py',
                'patch_type': 'full_replace',
                'content': safe_data_loader,
                'notes': 'Replace data_loader.py with safe tiny datasets and defensive clamping of num_classes/labels.'
            }
    return None


def _heuristic_patch_for_ir_missing(run_dir: Path, stderr: str) -> Optional[Dict[str, Any]]:
    """
    If generated files refer to `ir` variable (NameError: ir is not defined),
    propose replacing model.py with robust fallback that does not assume `ir` is present.
    """
    if "name 'ir' is not defined" in stderr or "name 'ir' is not defined" in stderr.lower():
        candidate = Path(run_dir) / 'repo' / 'model.py'
        if candidate.exists():
            safe_model = _safe_model_fallback_template()  # reuse the existing safe model generator
            return {
                'path': 'repo/model.py',
                'patch_type': 'full_replace',
                'content': safe_model,
                'notes': "Replace model.py with a safe fallback that doesn't rely on `ir` being defined."
            }
    return None


# Insert these new heuristics into the top-level suggest_patches_for_run flow
# after the existing heuristic checks (i.e. call them and append proposals if found).
# Example: add below your other p = _heuristic_* calls
#
#    p = _heuristic_patch_for_num_classes_zero(run_path, stderr)
#    if p:
#        proposals.append(p)
#    p = _heuristic_patch_for_ir_missing(run_path, stderr)
#    if p:
#        proposals.append(p)
#

def call_llm_for_patch(ir: Dict[str, Any], evidence: str) -> List[Dict[str, Any]]:
    """
    Use GeminiClient.json_call to ask for JSON patch proposals.
    Returns list of proposals or [] on error.
    """
    try:
        cli = GeminiClient()
    except Exception as e:
        print(f"[debugger] Gemini client init failed: {e}")
        return []

    # Build a concise prompt the gemini client expects (string)
    prompt = (
        "You are a code-fixing assistant. Return a JSON list of patch objects.\n"
        "Each object: {path, patch_type='full_replace', content, notes}\n\n"
        "IR (short):\n" + (json.dumps(ir, ensure_ascii=False)[:4000] if ir else "{}") +
        "\n\nEvidence (stderr + excerpt):\n" + (evidence[:8000] if evidence else "") +
        "\n\nReturn JSON only.\n"
    )
    try:
        parsed = cli.json_call(prompt, max_retries=2)
        # parsed may be a list or dict; normalize:
        if isinstance(parsed, dict):
            # guard: some responses may wrap proposals as {"proposals":[...]}
            if "proposals" in parsed and isinstance(parsed["proposals"], list):
                return parsed["proposals"]
            # or when LLM returns object, try to convert to list
            return [parsed]
        if isinstance(parsed, list):
            return parsed
    except Exception as e:
        print(f"[debugger] LLM call failed: {e}")
    return []



def _heuristic_patch_for_indentation(run_dir: Path, stderr: str) -> Optional[Dict[str, Any]]:
    if "expected an indented block after 'try'" in stderr:
        candidate = Path(run_dir) / 'repo' / 'model.py'
        if candidate.exists():
            safe_content = _safe_model_fallback_template()
            return {
                'path': 'repo/model.py',
                'patch_type': 'full_replace',
                'content': safe_content,
                'notes': 'Replace model.py with safe fallback template to avoid try/except indentation issues.'
            }
    return None


def _heuristic_patch_for_missing_xgb(run_dir: Path, stderr: str) -> Optional[Dict[str, Any]]:
    if "No module named 'xgboost'" in stderr or "name 'xgb' is not defined" in stderr:
        candidate = Path(run_dir) / 'repo' / 'model.py'
        if candidate.exists():
            safe_content = _safe_model_fallback_template(tabular=True)
            return {
                'path': 'repo/model.py',
                'patch_type': 'full_replace',
                'content': safe_content,
                'notes': 'Missing xgboost — switch to small torch fallback for tabular example.'
            }
    return None


def suggest_patches_for_run(run_dir: str, stderr: str, ir: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    run_path = Path(run_dir)
    proposals: List[Dict[str, Any]] = []

    p = _heuristic_patch_for_indentation(run_path, stderr)
    if p:
        proposals.append(p)
    p = _heuristic_patch_for_missing_xgb(run_path, stderr)
    if p:
        proposals.append(p)

    if not proposals:
        evidence = stderr
        try:
            model_txt = (run_path / 'repo' / 'model.py').read_text(encoding='utf-8')
            evidence += "\n---model.py---\n" + model_txt[:2000]
        except Exception:
            pass
        llm_proposals = call_llm_for_patch(ir or {}, evidence)
        proposals.extend(llm_proposals)

    return proposals


def _safe_model_fallback_template(tabular: bool = False) -> str:
    if tabular:
        return """# Auto-generated safe fallback model (tabular)
import torch

def _build_model(num_features=16, num_classes=2):
    model = torch.nn.Sequential(
        torch.nn.Linear(num_features, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, num_classes)
    )
    return model

model = _build_model()
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
"""
    return """# Auto-generated safe fallback model (vision/text/tabular)
import torch

class TinyFallback(torch.nn.Module):
    def __init__(self, num_classes: int = 2, hidden: int = 128):
        super().__init__()
        self.fc = torch.nn.Linear(768, hidden)
        self.relu = torch.nn.ReLU()
        self.out = torch.nn.Linear(hidden, num_classes)

    def forward(self, x):
        if isinstance(x, dict):
            x = next(iter(x.values()))
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        x = self.relu(self.fc(x))
        return self.out(x)

model = TinyFallback()
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
"""
def _safe_data_loader_template() -> str:
    # Minimal robust tiny datasets: clamps num_classes and clips labels.
    return r'''# Auto-generated safe data_loader for smoke tests
import torch
from torch.utils.data import Dataset, DataLoader

class TinyTextDataset(Dataset):
    def __init__(self, tokenizer=None, max_length=128, length=64, num_classes=2):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.length = int(length)
        self.num_classes = max(1, int(num_classes or 2))
        self.data = [
            ("good movie great acting", 1),
            ("bad plot boring scenes", 0),
            ("excellent visuals and story", 1),
            ("terrible pacing and sound", 0),
        ] * (self.length // 4 or 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, y = self.data[idx]
        # Defensive label clipping
        y = int(y) % max(1, self.num_classes)
        if self.tokenizer is None:
            x = torch.randn(768)
            return {"inputs": x, "labels": torch.tensor(y, dtype=torch.long)}
        enc = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        enc = {k: v.squeeze(0) for k, v in enc.items()}
        return {"inputs": enc, "labels": torch.tensor(y, dtype=torch.long)}

class TinyImageDataset(Dataset):
    def __init__(self, size=224, num_classes=2, length=32):
        self.size = int(size)
        self.length = int(length)
        self.num_classes = max(1, int(num_classes or 2))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = torch.rand(3, self.size, self.size)
        # randrange defence ensures upper bound > lower bound
        y = 0
        try:
            if self.num_classes > 0:
                y = torch.randint(0, self.num_classes, (1,)).item()
        except Exception:
            y = 0
        y = int(y) % max(1, self.num_classes)
        return {"inputs": x, "labels": torch.tensor(y, dtype=torch.long)}

def build_dataloaders(config_path: str = None):
    # Keep the minimal logic: default to image if no IR info
    batch_size = 4
    try:
        batch_size = int(8)
    except Exception:
        batch_size = 4
    train = TinyImageDataset()
    val = TinyImageDataset()
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size)
    return train_loader, val_loader
'''
