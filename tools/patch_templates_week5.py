# tools/patch_templates_week5.py
"""
Overwrite the Week5 codegen templates with safer, smoke-friendly versions.
Run from repo root:

    python tools/patch_templates_week5.py

This will overwrite the files under core/codegen/templates/ with improved templates
that avoid HTML escaping issues, smaller default image sizes and batch sizes,
and safer tokenizer handling.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TEMPLATES = {
    "core/codegen/templates/model.py.jinja": r"""{{ header_comment }}
{{ imports|join('\n') }}

# Construct the model from mapping (rendered verbatim)
# NOTE: mapping provides a Python expression (e.g. 'ViTForImageClassification.from_pretrained("...", num_labels=2)')
model = {{ model_constructor | safe or 'None' }}

# Optional loss & optimizer placeholders (keep simple defaults if missing)
loss_fn = {{ loss_constructor | safe or 'torch.nn.CrossEntropyLoss()' }}
optimizer = {{ optimizer_constructor | safe or "torch.optim.Adam(model.parameters(), lr=5e-4)" }}
""",

    "core/codegen/templates/data_loader.py.jinja": r"""{{ header_comment }}
import json
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader

try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None

# Optional HF preprocessing/tokenizer from mapping
{{ preprocessing_imports | safe }}

class TinyTextDataset(Dataset):
    def __init__(self, tokenizer, max_length=128, split='train'):
        data = [
            ("good movie great acting", 1),
            ("bad plot boring scenes", 0),
            ("excellent visuals and story", 1),
            ("terrible pacing and sound", 0),
        ]
        self.texts = [t for t, _ in data]
        self.labels = torch.tensor([y for _, y in data], dtype=torch.long)
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        t = self.texts[idx]
        enc = self.tok(
            t,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        enc = {k: v.squeeze(0) for k, v in enc.items()}
        return {"inputs": enc, "labels": self.labels[idx]}


class TinyImageDataset(Dataset):
    def __init__(self, size=(3, 64, 64), num_classes={{ num_classes or 2 }}, length=64):
        self.size = size
        self.length = length
        self.num_classes = num_classes

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = torch.rand(self.size)
        y = torch.randint(0, self.num_classes, (1,)).item()
        return {"inputs": x, "labels": torch.tensor(y, dtype=torch.long)}


def build_dataloaders(config_path: str = None) -> Tuple[DataLoader, DataLoader]:
    # Configure tokenizer / processor
    tokenizer = None
    # Use tokenizer only if mapping provided one or task looks like NLP
    try:
        if '{{ task_type or "" }}'.lower().startswith('sequence') or {{ 'True' if 'text_tokenizer' in preprocessing_imports else 'False' }}:
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased') if AutoTokenizer else None
        else:
            tokenizer = None
    except Exception:
        tokenizer = None

    task_type = "{{ task_type or '' }}"
    batch_size = int({{ batch_size or 8 }})

    if task_type.startswith('sequence') or tokenizer is not None:
        ds_train = TinyTextDataset(tokenizer)
        ds_val = TinyTextDataset(tokenizer)
    else:
        ds_train = TinyImageDataset()
        ds_val = TinyImageDataset()

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=batch_size)
    return train_loader, val_loader
""",

    "core/codegen/templates/config.yaml.jinja": r"""# Auto-generated config for smoke training
paper_id: {{ paper_id }}
framework: {{ framework or 'unknown' }}
model: {{ architecture or 'unknown' }}
num_classes: {{ num_classes or 2 }}
batch_size: {{ batch_size or 8 }}
epochs: {{ epochs or 1 }}
""",

    "core/codegen/templates/README.jinja": r"""# Generated Repo (Smoke Mode)

This folder was generated from IR + mapping.

## Files
- `model.py` — constructs model/loss/optimizer using mapping
- `data_loader.py` — tiny text/image datasets for smoke tests
- `trainer.py` — minimal training/eval loop; writes `metrics.json`
- `config.yaml` — basic config derived from IR

## Run (smoke test)
```bash
cd runs/{{ run_pkg.split('.')[-1] }}/repo
python trainer.py
```
""",

    "core/codegen/templates/trainer.py.jinja": r"""{{ header_comment }}
import os
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

try:
    import torchvision
    from torchvision import transforms
    HAS_TV = True
except Exception:
    HAS_TV = False

# Import model, loss, optimizer constructed in model.py
from model import model, loss_fn, optimizer
from data_loader import build_dataloaders


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        inputs = batch["inputs"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(**inputs) if isinstance(inputs, dict) else model(inputs)
        logits = getattr(outputs, 'logits', outputs)
        loss = loss_fn(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
    return total_loss / max(1, len(loader))


def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(**inputs) if isinstance(inputs, dict) else model(inputs)
            logits = getattr(outputs, 'logits', outputs)
            loss = loss_fn(logits, labels)
            preds = torch.argmax(logits, dim=-1)
            total_correct += (preds == labels).sum().item()
            total += labels.numel()
            total_loss += float(loss.item())
    acc = total_correct / max(1, total)
    return {"loss": total_loss / max(1, len(loader)), "accuracy": acc}


def main(config_path: str = None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train_loader, val_loader = build_dataloaders(config_path)

    epochs = {{ epochs or 1 }}
    metrics = {}
    for ep in range(epochs):
        tr_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val = evaluate(model, val_loader, loss_fn, device)
        print(f"[train] epoch={ep+1} loss={tr_loss:.4f} val_loss={val['loss']:.4f} val_acc={val['accuracy']:.4f}")
        metrics = {"epoch": ep+1, "train_loss": tr_loss, **val}

    # Write metrics.json beside this file
    out = Path(__file__).resolve().parent / 'metrics.json'
    out.write_text(json.dumps(metrics, indent=2))
    print(f"[trainer] wrote {out}")


if __name__ == "__main__":
    main()
""",
}


def main():
    wrote = []
    for rel, content in TEMPLATES.items():
        p = ROOT / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding='utf-8')
        wrote.append(str(p.relative_to(ROOT)))
    print("[patch_templates_week5] overwritten templates:")
    for w in wrote:
        print(" -", w)


if __name__ == "__main__":
    main()
