"""
Usage:
  1) Activate venv
  2) pip install Jinja2
  3) python tools/bootstrap_week5.py   # creates templates + generator + CLI + test
  4) python -m apps.cli.generate_all --config papers.yaml --out runs/
  5) pytest -q tests/test_repo_files_exist.py

This script is idempotent: it only writes files if they don't already exist, unless you pass
--force to overwrite.
"""
from __future__ import annotations
import json
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TEMPLATES = {
    "core/codegen/templates/model.py.jinja": r'''{{ header_comment }}
{{ imports|join('\n') }}

# Construct the model from mapping
model = {{ model_constructor or 'None' }}

# Optional loss & optimizer placeholders
loss_fn = {{ loss_constructor or 'None' }}
optimizer = {{ optimizer_constructor or 'None' }}
''',
    "core/codegen/templates/trainer.py.jinja": r'''{{ header_comment }}
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
from .model import model, loss_fn, optimizer
from .data_loader import build_dataloaders


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
''',
    "core/codegen/templates/data_loader.py.jinja": r'''{{ header_comment }}
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
{{ preprocessing_imports }}

class TinyTextDataset(Dataset):
    def __init__(self, tokenizer, max_length=128, split='train'):
        # Minimal in-memory dataset for smoke tests
        # Two-class toy data
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
        # squeeze batch dim
        enc = {k: v.squeeze(0) for k, v in enc.items()}
        return {"inputs": enc, "labels": self.labels[idx]}


class TinyImageDataset(Dataset):
    def __init__(self, size=(3, {{ input_h or 224 }}, {{ input_w or 224 }}), num_classes={{ num_classes or 2 }}, length=64):
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
    {{ tokenizer_setup }}

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
''',
    "core/codegen/templates/config.yaml.jinja": r'''# Auto-generated config for smoke training
paper_id: {{ paper_id }}
framework: {{ framework or 'unknown' }}
model: {{ architecture or 'unknown' }}
num_classes: {{ num_classes or 2 }}
batch_size: {{ batch_size or 8 }}
epochs: {{ epochs or 1 }}
''',
    "core/codegen/templates/README.jinja": r'''# Generated Repo (Smoke Mode)

This folder was generated from IR + mapping.

## Files
- `model.py` — constructs model/loss/optimizer using mapping
- `data_loader.py` — tiny text/image datasets for smoke tests
- `trainer.py` — minimal training/eval loop; writes `metrics.json`
- `config.yaml` — basic config derived from IR

## Run
```bash
python -m runs.{{ run_pkg }}.repo.trainer
```
''',
}

GENERATE_REPO = r'''from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any

from jinja2 import Environment, FileSystemLoader, select_autoescape


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _read_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding='utf-8'))


def _jinja_env(templates_dir: Path) -> Environment:
    return Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=select_autoescape(enabled_extensions=('jinja',)),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _derive_context(ir: Dict[str, Any], mapping: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    model = ir.get('model', {})
    dataset = ir.get('dataset', {})
    hp = ir.get('hyperparameters', {})

    preprocessing = mapping.get('preprocessing') or {}
    tokenizer_expr = preprocessing.get('text_tokenizer')

    # Tokenizer setup code fragment
    if tokenizer_expr:
        tokenizer_setup = f"tokenizer = {tokenizer_expr}\n"
        preprocessing_imports = ''
    else:
        tokenizer_setup = "tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased') if AutoTokenizer else None\n"
        preprocessing_imports = ''

    # imports are literal lines
    imports = mapping.get('imports') or []

    ctx = {
        'header_comment': f"# Generated from {run_dir.as_posix()} — DO NOT EDIT by hand",
        'imports': imports,
        'model_constructor': mapping.get('model_constructor'),
        'loss_constructor': mapping.get('loss_constructor') or 'torch.nn.CrossEntropyLoss()',
        'optimizer_constructor': mapping.get('optimizer_constructor') or 'torch.optim.Adam(model.parameters(), lr=5e-4)',
        'task_type': model.get('task_type') or ir.get('task'),
        'batch_size': hp.get('batch_size') or 8,
        'epochs': hp.get('epochs') or 1,
        'paper_id': ir.get('paper_id') or 'unknown',
        'framework': mapping.get('framework'),
        'architecture': model.get('architecture'),
        'num_classes': dataset.get('num_classes') or 2,
        'input_h': None,
        'input_w': None,
        'preprocessing_imports': preprocessing_imports,
        'tokenizer_setup': tokenizer_setup,
        'run_pkg': run_dir.as_posix().replace('/', '.').replace('\\', '.'),
    }

    # Try to pick input size if available [C, H, W]
    inp = dataset.get('input_size')
    if isinstance(inp, (list, tuple)) and len(inp) == 3:
        # Common order in your normalized IR is [C, H, W]
        ctx['input_h'] = inp[1]
        ctx['input_w'] = inp[2]

    return ctx


def generate_repo(ir_path: Path, mapping_path: Path, out_dir: Path, templates_dir: Path) -> None:
    ir = _read_json(ir_path)
    mapping = _read_json(mapping_path)
    run_dir = out_dir.parent

    _ensure_dir(out_dir)

    env = _jinja_env(templates_dir)

    ctx = _derive_context(ir, mapping, run_dir)

    # Render files
    for name in ("model.py.jinja", "trainer.py.jinja", "data_loader.py.jinja", "config.yaml.jinja", "README.jinja"):
        tpl = env.get_template(name)
        rendered = tpl.render(**ctx)
        target_name = name.replace('.jinja', '').replace('.yaml', '.yaml')
        # README extension fix
        if target_name == 'README':
            target_name = 'README.md'
        (out_dir / target_name).write_text(rendered, encoding='utf-8')

    # Write a __init__.py to make package importable as runs.<run>.repo
    (out_dir / "__init__.py").write_text("# generated repo", encoding='utf-8')

'''

GENERATE_ALL = r'''from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List

from core.codegen.generate_repo import generate_repo


def _latest_run_dir(runs_root: Path, paper_id: str) -> Path | None:
    # choose the latest directory matching <paper_id>_*
    candidates = sorted(runs_root.glob(f"{paper_id}_*/"))
    return candidates[-1] if candidates else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, help='papers.yaml path')
    ap.add_argument('--out', required=True, help='runs/ directory')
    ap.add_argument('--templates', default='core/codegen/templates/', help='templates directory')
    args = ap.parse_args()

    runs_root = Path(args.out)
    templates_dir = Path(args.templates)

    # papers.yaml is simple YAML (id list). We'll support .yaml without deps using a tiny parser fallback.
    import yaml
    papers = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))

    for item in papers:
        paper_id = item.get('id') or item.get('paper_id')
        if not paper_id:
            print(f"[generate_all] skipping item without id: {item}")
            continue
        run_dir = _latest_run_dir(runs_root, paper_id)
        if not run_dir:
            print(f"[generate_all] no runs found for {paper_id}")
            continue
        ir_path = run_dir / 'ir.json'
        mapping_path = run_dir / 'mapping.json'
        out_dir = run_dir / 'repo'
        print(f"[generate_all] generating repo for {paper_id} → {out_dir}")
        generate_repo(ir_path, mapping_path, out_dir, templates_dir)


if __name__ == '__main__':
    main()
'''

TEST_REPO = r'''from pathlib import Path
import json


def test_repo_files_exist():
    runs = Path('runs')
    assert runs.exists(), "runs/ not found — generate runs first"

    # find at least one repo
    any_repo = False
    for run in runs.glob('*_*'):
        repo = run / 'repo'
        if not repo.exists():
            continue
        any_repo = True
        for fname in ['model.py', 'trainer.py', 'data_loader.py', 'config.yaml', 'README.md', '__init__.py']:
            assert (repo / fname).exists(), f"missing {fname} in {repo}"
    assert any_repo, "no generated repo found in any runs/*_*/repo"
'''

GENERATE_REPO_INIT = "# package: core.codegen\n"


def write_file(path: Path, content: str, force: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        return False
    path.write_text(content, encoding='utf-8')
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true', help='overwrite existing files')
    args = ap.parse_args()

    wrote = []
    for rel, content in TEMPLATES.items():
        p = ROOT / rel
        if write_file(p, content, args.force):
            wrote.append(p)

    # generator module
    gen_pkg_init = ROOT / 'core/codegen/__init__.py'
    write_file(gen_pkg_init, GENERATE_REPO_INIT, args.force)

    gen_mod = ROOT / 'core/codegen/generate_repo.py'
    write_file(gen_mod, GENERATE_REPO, args.force)

    # CLI app
    app = ROOT / 'apps/cli/generate_all.py'
    write_file(app, GENERATE_ALL, args.force)

    # test
    test = ROOT / 'tests/test_repo_files_exist.py'
    write_file(test, TEST_REPO, args.force)

    if wrote:
        print('[bootstrap_week5] wrote:')
        for p in wrote:
            print('  -', p.relative_to(ROOT))
    print('[bootstrap_week5] ensured core/codegen and CLI + test files')


if __name__ == '__main__':
    main()
