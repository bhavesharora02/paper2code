# tools/auto_fix_goldens.py
"""
Auto-fix goldens:
- If paper title is present in papers.yaml, overwrite goldens/<paperid>.ir.json title with it.
- Apply heuristic spacing fixes to `title` and `abstract` (simple regex rules).
- Back up original goldens to goldens/_bak/<paperid>.ir.json.bak
"""

import re
import json
from pathlib import Path
import yaml
from shutil import copyfile

ROOT = Path(__file__).resolve().parents[1]
PAPERS_YAML = ROOT / "papers.yaml"
GOLDENS = ROOT / "goldens"
BACKUP_DIR = GOLDENS / "_bak"

def load_paper_titles():
    if not PAPERS_YAML.exists():
        return {}
    cfg = yaml.safe_load(PAPERS_YAML.read_text(encoding="utf-8"))
    return {p["id"]: p.get("title") for p in cfg.get("papers", [])}

def repair_spacing(s: str) -> str:
    if not s:
        return s
    # 1) Insert space between lowercase-to-uppercase runs: "Publishedasaconference" -> "Publishedasaconference"
    # But more useful: add space where a lowercase letter is followed by an uppercase letter.
    s = re.sub(r'([a-z])([A-Z])', r'\1 \2', s)
    # 2) Add spaces after punctuation if missing
    s = re.sub(r'([.,;:!?])([A-Za-z0-9])', r'\1 \2', s)
    # 3) Fix multiple spaces
    s = re.sub(r'\s+', ' ', s)
    # 4) Trim
    s = s.strip()
    return s

def main(dry_run: bool = False):
    titles_map = load_paper_titles()
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    for f in GOLDENS.glob("*.ir.json"):
        pid = f.stem
        data = json.loads(f.read_text(encoding="utf-8"))
        # backup
        copyfile(f, BACKUP_DIR / f"{pid}.ir.json.bak")
        changed = False

        # 1) Overwrite title from papers.yaml if present
        canonical_title = titles_map.get(pid)
        if canonical_title and canonical_title.strip():
            if data.get("title", "").strip() != canonical_title.strip():
                print(f"[auto_fix] Setting title for {pid} from papers.yaml")
                data["title"] = canonical_title.strip()
                changed = True

        # 2) heuristic spacing fix for abstract (if exists)
        if "abstract" in data and isinstance(data["abstract"], str) and data["abstract"].strip():
            repaired = repair_spacing(data["abstract"])
            if repaired != data["abstract"]:
                print(f"[auto_fix] Repairing abstract spacing for {pid}")
                data["abstract"] = repaired
                changed = True

        # 3) also repair title spacing/formatting heuristically if papers.yaml didn't have it
        if not canonical_title and "title" in data and isinstance(data["title"], str):
            repaired_title = repair_spacing(data["title"])
            if repaired_title != data["title"]:
                print(f"[auto_fix] Heuristically repairing title for {pid}")
                data["title"] = repaired_title
                changed = True

        if changed:
            if dry_run:
                print(f"[auto_fix] (dry-run) would update {f}")
            else:
                f.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"[auto_fix] Updated {f}")

    print("Auto-fix done. Backups in:", BACKUP_DIR)

if __name__ == "__main__":
    main(dry_run=False)
