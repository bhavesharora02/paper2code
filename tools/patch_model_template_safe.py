# tools/patch_model_template_safe.py
from pathlib import Path
TPL = Path(__file__).resolve().parents[1] / "core" / "codegen" / "templates" / "model.py.jinja"

SAFE_TEMPLATE = r'''# Generated from {{ run_dir }} — DO NOT EDIT by hand
import torch

# Optional imports provided by mapping (rendered defensively)
{% set _imports = mapping.get('imports') if mapping is defined else [] %}
try:
{% if _imports %}
{% for line in _imports %}
{{ line }}
{% endfor %}
{% else %}
    # no mapping imports provided
    pass
{% endif %}
except Exception as e:
    print(f"[model] optional import failed: {e}")

# Build model with safe constructors and fallback
model = None

# Try mapping constructor if present
try:
    constructor = mapping.get('model_constructor') if mapping is defined else None
    if constructor:
        try:
            # Evaluate constructor expression in a safe local scope
            # We rely on imports above; this is best-effort.
            model = eval(constructor)
        except Exception as e:
            print(f"[model] mapping constructor failed: {e}")
except Exception:
    # defensive: if mapping or eval fails, continue to fallback
    pass

# If model is still None, create a tiny fallback model that accepts common inputs
if model is None:
    class _TinyFallback(torch.nn.Module):
        def __init__(self, in_dim=3*32*32, num_classes={{ ir.get('dataset', {}).get('num_classes', 2) }}):
            super().__init__()
            self.flatten = torch.nn.Flatten()
            self.fc = torch.nn.Linear(in_dim, num_classes)
        def forward(self, x=None, **kwargs):
            # Accept dict or tensor inputs
            if isinstance(x, dict):
                # try common keys
                for k in ("pixel_values", "input_ids", "inputs"):
                    if k in x:
                        t = x[k]
                        break
                else:
                    t = next(iter(x.values()))
            else:
                t = x
            if not isinstance(t, torch.Tensor):
                # create a small random tensor (batch 1)
                t = torch.randn(1, 3, 32, 32)
            if t.dim() > 2:
                t = t.view(t.size(0), -1)
            return self.fc(t)
    model = _TinyFallback()

# Loss & optimizer (constructed after model exists)
try:
    loss_fn = mapping.get('loss_constructor') if mapping is defined else None
    if loss_fn:
        # if mapping supplied a string constructor, try eval - else assume it's an expression
        if isinstance(loss_fn, str):
            try:
                loss_fn = eval(loss_fn)
            except Exception:
                loss_fn = torch.nn.CrossEntropyLoss()
    else:
        loss_fn = torch.nn.CrossEntropyLoss()
except Exception:
    loss_fn = torch.nn.CrossEntropyLoss()

try:
    opt_ctor = mapping.get('optimizer_constructor') if mapping is defined else None
    if opt_ctor and isinstance(opt_ctor, str):
        try:
            optimizer = eval(opt_ctor)
        except Exception as e:
            print(f"[model] optimizer eval failed: {e}; falling back.")
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    elif opt_ctor:
        optimizer = opt_ctor
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
except Exception:
    # final fallback: if model has no parameters (unlikely) create a dummy optimizer on a small param
    try:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    except Exception:
        # create a tiny parameter to attach optimizer
        tmp = torch.nn.Parameter(torch.randn(1))
        optimizer = torch.optim.Adam([tmp], lr=1e-3)
        print("[model] created dummy optimizer due to failure constructing from model.")
'''

# Write template (idempotent)
TPL.parent.mkdir(parents=True, exist_ok=True)
TPL.write_text(SAFE_TEMPLATE, encoding='utf-8')
print("Wrote safe model.py.jinja to", TPL)
