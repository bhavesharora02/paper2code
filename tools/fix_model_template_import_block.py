# tools/fix_model_template_import_block.py
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
TPL = ROOT / "core" / "codegen" / "templates" / "model.py.jinja"

if not TPL.exists():
    print("Template not found:", TPL)
    raise SystemExit(1)

txt = TPL.read_text(encoding="utf-8")

# We'll replace the block:
# try:
# {% for line in mapping.imports %}
# {{ line }}
# {% endfor %}
# except Exception as e:
#
# with:
# try:
# {% if mapping.get('imports') %}
# {% for line in mapping.get('imports', []) %}
# {{ line }}
# {% endfor %}
# {% else %}
#     pass
# {% endif %}
# except Exception as e:

pattern = re.compile(r"try:\s*\n\s*\{%\s*for\s+line\s+in\s+mapping(?:\.imports|\.get\('imports'.*?\))\s*%\}[\s\S]*?\{%\s*endfor\s*%}\s*\n\s*except\s+Exception\s+as\s+e:", flags=re.MULTILINE)

replacement = (
    "try:\n"
    "{% if mapping.get('imports') %}\n"
    "{% for line in mapping.get('imports', []) %}\n"
    "{{ line }}\n"
    "{% endfor %}\n"
    "{% else %}\n"
    "    pass\n"
    "{% endif %}\n"
    "except Exception as e:"
)

if pattern.search(txt):
    new_txt = pattern.sub(replacement, txt)
else:
    # fallback: try simple string replace if pattern didn't hit
    if "{% for line in mapping.imports %}" in txt and "{% endfor %}" in txt:
        new_txt = txt.replace("{% for line in mapping.imports %}",
                              "{% if mapping.get('imports') %}\n{% for line in mapping.get('imports', []) %}")
        new_txt = new_txt.replace("{% endfor %}\nexcept Exception as e:",
                                  "{% endfor %}\n{% else %}\n    pass\n{% endif %}\nexcept Exception as e:")
    else:
        print("Could not find the expected import-for block in the template. Please open the template for manual edit:", TPL)
        raise SystemExit(1)

if new_txt == txt:
    print("Template already patched.")
else:
    TPL.write_text(new_txt, encoding="utf-8")
    print("Patched template:", TPL)
