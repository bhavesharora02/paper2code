# core/parser/parse_pdf.py
"""
Simple robust PDF parser for Week 2.

Extracts:
 - title (heuristic: first non-empty line / first heading)
 - abstract (search for 'abstract' heading)
 - method / approach / experiments (heuristic section splitting)
 - pseudocode_blocks (heuristic: look for 'Algorithm' or 'Pseudo' / code-like blocks)
 - figures (placeholder: extracts captions by regex)
Outputs a Python dict. Caller should write to JSON file.

Dependencies: pdfplumber (pip install pdfplumber), optionally PyMuPDF / paddleocr later.
"""

import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import pdfplumber


# SECTION HEADINGS we try to find (case-insensitive)
SECTION_KEYS = [
    "abstract",
    "introduction",
    "related work",
    "background",
    "method",
    "approach",
    "methodology",
    "experiments",
    "results",
    "conclusion",
    "discussion",
]


def _clean_text(s: Optional[str]) -> str:
    if not s:
        return ""
    return re.sub(r"\s+\n", "\n", s).strip()


def _split_sections(fulltext: str) -> Dict[str, str]:
    """
    Naive section splitter: finds headings from SECTION_KEYS and splits by them.
    Returns mapping heading -> content (best-effort).
    """
    text = fulltext
    sections = {}
    # Build pattern to find headings like '\nAbstract\n' or '\n1. Introduction\n'
    heading_pattern = r"(^|\n)(\s{0,8}[0-9]{0,2}\.?\s*)?(%s)\s*\n" % "|".join(
        [re.escape(k) for k in SECTION_KEYS]
    )
    # Find all heading matches
    matches = list(re.finditer(heading_pattern, text, flags=re.IGNORECASE | re.MULTILINE))
    # If no matches, return raw
    if not matches:
        return {"raw_text": text}

    # Append a sentinel at end
    spans = []
    for m in matches:
        spans.append((m.start(), m.group(3).lower()))
    spans.append((len(text), None))

    for i in range(len(spans) - 1):
        start_idx = spans[i][0]
        heading = spans[i][1]
        # find next heading index (use matches[i+1].start())
        next_start = spans[i + 1][0]
        # Extract content after the heading line
        # We try to find the line end of current heading
        after_heading = text[start_idx:next_start]
        # Remove the heading keyword itself from content
        # locate first newline after heading phrase
        first_nl = after_heading.find("\n")
        content = after_heading[first_nl + 1 :] if first_nl != -1 else after_heading
        sections[heading] = _clean_text(content)

    return sections


def _extract_pseudocode_blocks(fulltext: str) -> List[str]:
    """
    Heuristic: find blocks starting with 'Algorithm' or 'Pseudo' or lines with 'for i in' etc.
    Return list of strings (blocks).
    """
    blocks = []
    # Look for 'Algorithm N' blocks
    algo_matches = re.findall(r"(Algorithm\s*\d+[:\s].+?)(?=Algorithm\s*\d+[:\s]|\nReferences|\nACKNOWLEDGMENTS|\Z)", fulltext, flags=re.IGNORECASE | re.DOTALL)
    for a in algo_matches:
        blocks.append(_clean_text(a))

    # If none, look for code-like blocks with indentation and colons (python-like pseudocode)
    if not blocks:
        code_like = re.findall(r"((?:\n\s{2,}.*?)+)", fulltext, flags=re.DOTALL)
        for c in code_like:
            # include blocks that look like pseudocode (contain 'for' or 'if' or 'while' or ':' a lot)
            if re.search(r"\b(for|if|while|return|def|=>|->)\b", c):
                blocks.append(_clean_text(c))

    return blocks


def _extract_title(fulltext: str) -> str:
    """
    Heuristic title: the first 1-3 lines that are not empty and not 'abstract'.
    """
    lines = [ln.strip() for ln in fulltext.splitlines() if ln.strip()]
    if not lines:
        return ""
    # If first line is 'title' or looks like wordy, return first non-empty line
    # Some PDFs put title in the first line(s)
    # Return first line unless it's 'abstract' or similar
    for i, ln in enumerate(lines[:6]):
        if re.search(r"abstract", ln, flags=re.IGNORECASE):
            continue
        # skip lines that are just authors/emails (contain '@' or digits with commas or 'et al.')
        if "@" in ln or re.search(r"\b(et al|author|affiliat)", ln, flags=re.IGNORECASE):
            continue
        # return first candidate
        return ln
    return lines[0]


def parse_pdf_to_dict(pdf_path: str) -> Dict:
    """
    Parse PDF and return a dict with extracted fields.
    """
    p = Path(pdf_path)
    if not p.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    pages_text = []
    # Use pdfplumber to extract text
    try:
        with pdfplumber.open(str(p)) as pdf:
            for page in pdf.pages:
                txt = page.extract_text() or ""
                pages_text.append(txt)
    except Exception:
        # fallback: try reading as binary and at least provide empty content
        pages_text = [""]

    full_text = "\n".join(pages_text).strip()
    # Basic fields
    title = _extract_title(full_text)
    sections = _split_sections(full_text)

    abstract = sections.get("abstract", "") or ""
    # choose method/approach/experiments if present
    method = sections.get("method") or sections.get("approach") or sections.get("methodology") or ""
    experiments = sections.get("experiments") or sections.get("results") or ""

    pseudocode_blocks = _extract_pseudocode_blocks(full_text)

    # figures: naive caption extraction (lines that start with 'Figure' or 'Fig.')
    figures = re.findall(r"(Figure\s*\d+[:\.\s].{0,200})", full_text, flags=re.IGNORECASE)
    if not figures:
        figures = re.findall(r"(Fig\.\s*\d+[:\.\s].{0,200})", full_text, flags=re.IGNORECASE)

    parsed = {
        "source_pdf": str(p.resolve()),
        "title": title,
        "abstract": abstract,
        "method": method,
        "experiments": experiments,
        "sections": sections,  # full sections mapping (best-effort)
        "pseudocode_blocks": pseudocode_blocks,
        "figures_captions": figures,
        "raw_text_snippet": full_text[:8000],  # truncated snippet for safety
        "parsing_timestamp": int(time.time()),
    }
    return parsed
