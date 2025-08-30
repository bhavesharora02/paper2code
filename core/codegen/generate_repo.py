from __future__ import annotations
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

