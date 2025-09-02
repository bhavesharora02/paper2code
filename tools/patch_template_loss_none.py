# tools/patch_template_loss_none.py
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent
TPL = ROOT / "core" / "codegen" / "templates" / "model.py.jinja"

LOSS_BLOCK = r'''
# Loss & optimizer with graceful fallbacks
{% set _loss_expr = mapping.loss_constructor if mapping is defined else None %}
{% if _loss_expr is not defined or _loss_expr is none or (_loss_expr|string)|lower in ["", "none", "null"] %}
loss_fn = torch.nn.CrossEntropyLoss()
{% else %}
try:
    loss_fn = {{ mapping.loss_constructor }}
except Exception as e:
    print(f"[model] loss construction failed: {e}")
    loss_fn = torch.nn.CrossEntropyLoss()
{% endif %}

# Optimizer: try mapping's optimizer, else build from model params, else fallback to empty SGD
{% set _opt_expr = mapping.optimizer_constructor if mapping is defined else None %}
{% if _opt_expr is not defined or _opt_expr is none or (_opt_expr|string)|lower in ["", "none", "null"] %}
try:
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=1e-3) if params else torch.optim.SGD([], lr=1e-3)
except Exception as e:
    print(f"[model] optimizer construction failed: {e}")
    optimizer = torch.optim.SGD([], lr=1e-3)
{% else %}
try:
    optimizer = {{ mapping.optimizer_constructor }}
except Exception as e:
    print(f"[model] optimizer construction failed: {e}")
    try:
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=1e-3) if params else torch.optim.SGD([], lr=1e-3)
    except Exception:
        optimizer = torch.optim.SGD([], lr=1e-3)
{% endif %}
'''.lstrip()

def main():
    src = TPL.read_text(encoding="utf-8")
    # Replace the existing "Loss & optimizer" section from the template we wrote earlier.
    start = src.find("# Loss & optimizer with graceful fallbacks")
    if start == -1:
        raise SystemExit("Couldn't find loss/optimizer section in template. Open the file and check.")
    # Take everything up to the start, then append our new guarded block (it also contains optimizer).
    head = src[:start]
    new = head + LOSS_BLOCK
    TPL.write_text(new, encoding="utf-8")
    print(f"[patch] updated loss/optimizer guards in {TPL}")

if __name__ == "__main__":
    main()
