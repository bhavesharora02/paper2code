# apps/ui/streamlit_app.py
"""
Simple Streamlit UI for the Paper2Code orchestrator (Week 9 demo).
- Pick a paper from papers.yaml or upload a PDF (upload flow is minimal stub).
- Call CLI generate/run commands and surface stdout/stderr.
- Show artifacts/*.json from chosen run when available.
"""
from __future__ import annotations
import json
import subprocess
from pathlib import Path
import streamlit as st
import yaml

# Config path (project root)
CFG_PATH = Path("papers.yaml")


def _load_papers_config(cfg_path: Path):
    """Return list of papers (list of dicts) or [] on error."""
    if not cfg_path.exists():
        return []
    text = cfg_path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            return []
        return data.get("papers", []) or []
    except Exception:
        return []


st.set_page_config(page_title="Paper2Code — Orchestrator", layout="wide")

st.title("Paper2Code — Orchestrator")

# Load papers (YAML)
papers = _load_papers_config(CFG_PATH)
paper_ids = [p.get("id") for p in papers if isinstance(p, dict) and p.get("id")]

col_l, col_r = st.columns([1, 2])

with col_l:
    st.header("Select / Upload")
    sel = st.selectbox("Choose a paper (papers.yaml)", ["--upload-pdf--"] + paper_ids)
    uploaded = st.file_uploader("Upload PDF (optional)", type=["pdf"], help="Upload a PDF and then click Start run to generate a new run folder.")
    domain_override = st.text_input("Domain override (optional)")
    run_dir_input = st.text_input("Run directory (optional; e.g. runs/cv_vit_1756...)", value="")
    btn_generate = st.button("Generate repo")
    btn_run = st.button("Start orchestrator (smoke+metrics)")

with col_r:
    st.header("Run status")
    status_box = st.empty()
    if not papers:
        st.info("papers.yaml not found or empty. You can still upload a PDF and generate a run.")
    # show last registry entry if present
    reg_file = Path("runs") / "registry.json"
    if reg_file.exists():
        try:
            reg = json.loads(reg_file.read_text(encoding="utf-8"))
            st.subheader("Registry (latest)")
            st.json(reg)
        except Exception:
            st.text("Registry exists but could not be parsed.")

# Helper to run a shell command and stream output
def _run_cmd(cmd: list[str], cwd: str | None = None, timeout: int | None = None):
    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}
    except subprocess.TimeoutExpired as te:
        return {"returncode": 124, "stdout": "", "stderr": f"timeout: {te}"}
    except Exception as e:
        return {"returncode": 1, "stdout": "", "stderr": str(e)}

# Generate repo action
if btn_generate:
    if sel == "--upload-pdf--":
        st.error("Select a paper from papers.yaml if you want to use generate_all CLI. Upload-only flow is not implemented here.")
    else:
        out_dir = run_dir_input or "runs/"
        st.info(f"Calling generator for paper {sel} -> out {out_dir}")
        # using apps.cli.generate_all as convenience wrapper
        cmd = ["python", "-m", "apps.cli.generate_all", "--config", "papers.yaml", "--out", out_dir]
        res = _run_cmd(cmd, timeout=300)
        st.subheader("Generator output")
        st.text_area("stdout/stderr", res["stdout"] + "\n\n" + res["stderr"], height=300)

# Orchestrator/run action
if btn_run:
    if sel == "--upload-pdf--":
        # if upload, we could write uploaded file to samples/ and then parse/generate; minimal flow:
        if not uploaded:
            st.error("Upload a PDF or select a paper id to run.")
        else:
            # write upload to samples/ and call parse/generate/run in sequence (minimal)
            samples_dir = Path("samples")
            samples_dir.mkdir(exist_ok=True)
            fname = f"uploaded_{sel or 'paper'}_{int(__import__('time').time())}.pdf"
            target = samples_dir / fname
            target.write_bytes(uploaded.getbuffer())
            st.success(f"Saved uploaded PDF to {str(target)}")
            st.info("Now calling parse_all -> generate_all -> run_smoke_all for this upload (may take time).")
            c1 = _run_cmd(["python", "-m", "apps.cli.parse_all", "--config", "papers.yaml", "--out", "runs/"])
            c2 = _run_cmd(["python", "-m", "apps.cli.generate_all", "--config", "papers.yaml", "--out", "runs/"])
            c3 = _run_cmd(["python", "-m", "apps.cli.run_smoke_all", "--config", "papers.yaml", "--out", "runs/", "--timeout", "300"])
            st.text_area("parse stdout/stderr", c1["stdout"] + "\n\n" + c1["stderr"], height=200)
            st.text_area("generate stdout/stderr", c2["stdout"] + "\n\n" + c2["stderr"], height=200)
            st.text_area("run_smoke stdout/stderr", c3["stdout"] + "\n\n" + c3["stderr"], height=300)
    else:
        # select latest run if not provided
        runs_root = Path("runs")
        candidates = sorted([d for d in runs_root.glob(f"{sel}_*") if d.is_dir()])
        if not candidates and not run_dir_input:
            st.error("No generated run found for this paper. Run 'Generate repo' first or provide run dir.")
        else:
            run_dir = run_dir_input or str(candidates[-1])
            st.info(f"Orchestrating {sel} -> {run_dir} (smoke+metrics)")
            cmd = ["python", "-m", "apps.cli.run", "--paper", sel, "--run", run_dir, "--metrics"]
            res = _run_cmd(cmd, timeout=600)
            st.subheader("Orchestrator output")
            st.text_area("stdout/stderr", res["stdout"] + "\n\n" + res["stderr"], height=400)

            art = Path(run_dir) / "artifacts"
            if art.exists():
                st.write("Artifacts in run:")
                files = sorted([p for p in art.glob("*.json")])
                for p in files:
                    st.subheader(p.name)
                    try:
                        obj = json.loads(p.read_text(encoding="utf-8"))
                        st.json(obj)
                    except Exception:
                        st.code(p.read_text(encoding="utf-8")[:2000])
            else:
                st.warning("No artifacts/ directory found in run.")

st.markdown("---")
st.caption("This is a minimal demo Streamlit UI for the Paper2Code orchestrator.")
