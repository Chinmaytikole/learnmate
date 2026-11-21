"""
app/main.py -- LearnMate Streamlit app.

Run from the learnmate/ directory:
     

Tabs: Material Report | Ask (chat) | Batch CSV
Sidebar: upload course files, demo mode, strictness slider.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

# Prevent slow HF hub retries when offline
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import pandas as pd
import streamlit as st

from app.audit import build_material_report
from app.engine import TutorEngine
from app.index import build_course_retriever
from app.ingest import ingest_files, save_uploads
from src.agree import AgreementScorer
from src.calibrate import Calibrator
from src.generate import LLMGenerator
from src.utils import load_config, seed_everything

CALIBRATOR_PATH = _ROOT / "results" / "models" / "calibrator_app.pkl"
PRESETS = {"Lenient (0.30)": 0.30, "Balanced (0.50)": 0.50, "Strict (0.70)": 0.70}

st.set_page_config(page_title="LearnMate", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner=False)
def get_cfg():
    cfg = load_config(str(_ROOT / "configs" / "proposed.yaml"))
    seed_everything(int(cfg.project.seed))
    return cfg


@st.cache_resource(show_spinner=False)
def get_scorer(demo: bool):
    return AgreementScorer.from_config(get_cfg(), is_mock=demo)


def build_engine(ing, demo: bool, top_k: int) -> TutorEngine:
    cfg = get_cfg()
    retriever = build_course_retriever(cfg, ing.chunks, ing.content_hash)
    gen = LLMGenerator.from_config(cfg)
    if demo:
        gen.runtime = "mock"
    calibrator = Calibrator.load(CALIBRATOR_PATH) if CALIBRATOR_PATH.exists() else None
    return TutorEngine(retriever, gen, get_scorer(demo), calibrator=calibrator, top_k=top_k)


def render_result(r):
    if r.answered:
        st.success(r.answer)
    else:
        st.warning("I can't answer this reliably from your course material.")
        st.caption(r.reason)
    c1, c2, c3 = st.columns(3)
    label = "Confidence" if r.scoring_mode == "calibrated" else "Evidence score"
    c1.metric(label, f"{r.confidence:.2f}")
    c2.metric("Answer-evidence agreement", f"{r.agreement:.2f}")
    c3.metric("Top retrieval similarity", f"{r.top_sim:.2f}")
    if r.citations:
        with st.expander(("Sources" if r.answered else "Closest passages") + f" ({len(r.citations)})"):
            for c in r.citations:
                loc = f"{c['doc']} - {c['heading']}" + (f" (p.{c['page']})" if c["page"] else "")
                st.markdown(f"**{loc}**  \n`{c['passage_id']}` | similarity {c['score']:.2f}")
                st.write(c["text"])


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("📚 LearnMate")
    st.caption("Evidence-grounded course tutor that abstains when the evidence is weak.")
    uploads = st.file_uploader("Course material", type=["pdf", "docx", "pptx", "txt", "md"],
                               accept_multiple_files=True)
    demo = st.toggle("Demo mode (no Ollama / no NLI model)", value=False,
                     help="Uses a deterministic mock generator and agreement scorer. "
                          "For UI testing only; answers are not meaningful.")
    top_k = st.slider("Passages retrieved (k)", 1, 10, 5)
    preset = st.radio("Strictness preset", list(PRESETS) + ["Custom"], index=1)
    strict = PRESETS.get(preset) if preset != "Custom" else st.slider("Custom threshold", 0.0, 1.0, 0.5, 0.01)
    st.caption("Higher strictness = fewer answers, fewer unsupported answers.")
    build = st.button("Process material", type="primary", disabled=not uploads)

if build and uploads:
    with st.spinner("Parsing and indexing..."):
        tmp = Path(tempfile.mkdtemp(prefix="learnmate_"))
        paths = save_uploads([(u.name, u.getvalue()) for u in uploads], tmp)
        ck = get_cfg().chunking
        ing = ingest_files(paths, min_words=int(ck.min_words), max_words=int(ck.max_words),
                           overlap=int(ck.overlap_words))
        st.session_state["ing"] = ing
        st.session_state["report"] = build_material_report(ing)
        st.session_state.pop("engine", None)
        st.session_state["chat"] = []
        if ing.chunks:
            try:
                st.session_state["engine"] = build_engine(ing, demo, top_k)
            except Exception as e:
                st.error(f"Could not build index: {e}")

engine: TutorEngine | None = st.session_state.get("engine")
if engine is not None:
    engine.top_k = top_k

tab_report, tab_chat, tab_batch = st.tabs(["Material Report", "Ask", "Batch CSV"])

# ------------------------------------------------------------------ report
with tab_report:
    rep = st.session_state.get("report")
    if rep is None:
        st.info("Upload course files in the sidebar and click **Process material**.")
    else:
        a, b, c, d = st.columns(4)
        a.metric("Files", rep["n_files"])
        b.metric("Chunks", rep["n_chunks"])
        c.metric("Words", f"{rep['n_words']:,}")
        d.metric("Files needing attention", rep["n_problem_files"])
        for w in rep["warnings"]:
            st.warning(w)
        if not rep["warnings"]:
            st.success("No material problems detected.")
        st.dataframe(rep["table"], use_container_width=True, hide_index=True)

# ------------------------------------------------------------------ chat
with tab_chat:
    if engine is None:
        st.info("Process course material first.")
    else:
        if engine.scoring_mode == "heuristic":
            st.caption("Scoring mode: uncalibrated heuristic (0.5 x retrieval + 0.5 x agreement). "
                       "Scores are not probabilities.")
        if demo:
            st.warning("Demo mode is on: answers come from a mock generator.")
        elif getattr(engine.scorer, "is_mock", False):
            st.warning("NLI model could not be loaded; agreement scores use a mock fallback.")
        for turn in st.session_state.get("chat", []):
            with st.chat_message("user"):
                st.write(turn["question"])
            with st.chat_message("assistant"):
                render_result(turn["result"])
        q = st.chat_input("Ask a question about your course material")
        if q:
            with st.chat_message("user"):
                st.write(q)
            with st.chat_message("assistant"):
                try:
                    with st.spinner("Thinking..."):
                        res = engine.ask(q, strict)
                    render_result(res)
                    st.session_state["chat"].append({"question": q, "result": res})
                except Exception as e:
                    st.error(f"Generation failed: {e}. Is Ollama running with the model pulled? "
                             "You can try Demo mode.")

# ------------------------------------------------------------------ batch
with tab_batch:
    if engine is None:
        st.info("Process course material first.")
    else:
        st.write("Upload a CSV with a `question` column.")
        csv_file = st.file_uploader("Questions CSV", type=["csv"], key="batch_csv")
        if csv_file is not None:
            df = pd.read_csv(csv_file)
            if "question" not in df.columns:
                st.error("CSV must contain a `question` column.")
            elif st.button("Run batch"):
                try:
                    with st.spinner(f"Answering {len(df)} questions..."):
                        out = engine.batch(df["question"].astype(str).tolist(), strict)
                    n_ans = int((out["decision"] == "answered").sum())
                    st.write(f"Answered **{n_ans}/{len(out)}** ({n_ans/max(1,len(out)):.0%} coverage) "
                             f"at strictness {strict:.2f}.")
                    st.dataframe(out, use_container_width=True, hide_index=True)
                    st.download_button("Download results CSV", out.to_csv(index=False).encode("utf-8"),
                                       "learnmate_results.csv", "text/csv")
                except Exception as e:
                    st.error(f"Batch failed: {e}")
