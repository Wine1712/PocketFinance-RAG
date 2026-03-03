# app.py
from __future__ import annotations

import sys
import subprocess
from pathlib import Path

import pandas as pd
import streamlit as st

from rag.ingest import build_or_update_index
from rag.qa import answer_with_ollama
from rag.extract import extract_rates_table
from rag.config import RAW_DIR, INDEX_DIR, CLEAN_DIR


# -----------------------------
# Helpers
# -----------------------------
def _run_pdf_to_clean_text() -> str:
    """
    Runs scripts/pdf_to_text_clean.py to convert PDFs in data/raw into data/clean_text.
    Returns stdout for display/logging.
    """
    script_path = Path("scripts/pdf_to_text_clean.py")
    if not script_path.exists():
        raise FileNotFoundError(
            "Missing scripts/pdf_to_text_clean.py. Create it first (scripts/ folder must exist)."
        )

    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
    )
    # Treat non-zero exit as error
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "PDF-to-text conversion failed.")
    return result.stdout.strip()


def _ensure_index_ready() -> dict:
    """
    On app startup:
    1) Convert PDFs -> clean text (if PDFs exist).
    2) Build/update FAISS index using clean_text if available, else raw.
    Returns status info for UI.
    """
    status = {
        "pdf_converted": False,
        "index_built": False,
        "chunks_indexed": 0,
        "using_clean_text": False,
        "notes": [],
    }

    # Convert PDFs to clean text if any PDFs exist in data/raw
    pdfs = list(Path(RAW_DIR).glob("*.pdf")) if Path(RAW_DIR).exists() else []
    if pdfs:
        try:
            _run_pdf_to_clean_text()
            status["pdf_converted"] = True
        except Exception as e:
            status["notes"].append(f"PDF→TXT conversion skipped/failed: {e}")

    # Build/update index (prefer clean_text if it exists and has files)
    try:
        clean_exists = Path(CLEAN_DIR).exists() and any(Path(CLEAN_DIR).rglob("*"))
        status["using_clean_text"] = bool(clean_exists)

        n = build_or_update_index(use_demo=False, use_clean=True)
        status["index_built"] = True
        status["chunks_indexed"] = int(n)
    except Exception as e:
        status["notes"].append(f"Index build failed: {e}")

    return status


# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(page_title="PocketFinance-RAG", page_icon="💳", layout="wide")
st.title("💳 PocketFinance-RAG")
st.caption("Local RAG for finance PDFs: Q&A with citations + structured rate comparison tables.")

# -----------------------------
# Auto-run preprocessing + indexing ON START
# -----------------------------
with st.spinner("Preparing system: converting PDFs → clean text and building index..."):
    startup_status = _ensure_index_ready()

# Status UI
with st.expander("✅ Startup status", expanded=True):
    st.write(f"**RAW_DIR:** `{RAW_DIR}`")
    st.write(f"**CLEAN_DIR:** `{CLEAN_DIR}`")
    st.write(f"**INDEX_DIR:** `{INDEX_DIR}`")
    st.write(f"**PDF → TXT converted:** {startup_status['pdf_converted']}")
    st.write(f"**Using clean text for indexing:** {startup_status['using_clean_text']}")
    st.write(f"**Index built:** {startup_status['index_built']}")
    st.write(f"**Chunks indexed:** {startup_status['chunks_indexed']}")
    if startup_status["notes"]:
        st.warning("Notes:\n- " + "\n- ".join(startup_status["notes"]))

if not startup_status["index_built"]:
    st.error(
        "Index is not ready. Please check the Startup status notes above. "
        "Common fix: ensure your PDFs are in `data/raw/` and required packages are installed."
    )
    st.stop()

# -----------------------------
# Main UI: Tabs
# -----------------------------
tab_compare, tab_qa = st.tabs(["📊 Comparison tables", "💬 Ask questions (Q&A)"])


# -----------------------------
# Tab 1: Structured Comparison
# -----------------------------
with tab_compare:
    st.subheader("📊 Structured Comparison (Rates Table)")
    st.write(
        "Select a product category and generate a structured comparison table extracted from your indexed documents."
    )

    colA, colB, colC = st.columns([1.2, 1, 1])

    with colA:
        compare_mode = st.radio(
            "Comparison mode",
            ["Loan comparison", "Savings comparison"],
            horizontal=True,
        )

    with colB:
        model_name = st.selectbox(
            "Local model (Ollama)",
            ["llama3.1:8b", "phi3:mini"],
            index=0,
            help="If your laptop is slow, choose phi3:mini.",
        )

    with colC:
        top_k = st.slider(
            "Evidence chunks (k)",
            min_value=6,
            max_value=20,
            value=12,
            step=2,
            help="Higher k improves coverage across banks but can add noise.",
        )

    st.divider()

    if compare_mode == "Loan comparison":
        product_type = st.selectbox(
            "Loan type",
            [
                "Home loan (fixed)",
                "Home loan (variable)",
                "Car loan",
                "Personal loan",
                "Credit card interest",
            ],
        )
    else:
        product_type = st.selectbox(
            "Savings type",
            [
                "High interest savings",
                "Bonus saver",
                "Term deposit (6 months)",
                "Term deposit (1 year)",
            ],
        )

    if st.button("Generate comparison table", type="primary"):
        try:
            data = extract_rates_table(product_type=product_type, model=model_name, k=top_k)
            items = data.get("items", [])

            if not items:
                st.warning(
                    "No structured rates found from retrieved documents. "
                    "Add more rate documents (multiple banks) and restart the app to re-index."
                )
            else:
                df = pd.DataFrame(items)

                preferred_cols = [
                    "bank",
                    "product_type",
                    "term_years",
                    "rate",
                    "comparison_rate",
                    "conditions",
                    "notes",
                ]
                cols = [c for c in preferred_cols if c in df.columns]

                st.subheader("Result table")
                st.dataframe(df[cols], use_container_width=True)

                st.subheader("Sources (citations)")
                seen = set()
                for it in items:
                    src = it.get("source", {}) or {}
                    key = (src.get("source"), src.get("page"), src.get("chunk_id"))
                    if key in seen:
                        continue
                    seen.add(key)
                    st.write(f"- {src.get('source')} | page={src.get('page')} | chunk_id={src.get('chunk_id')}")

        except Exception as e:
            st.error(f"Extraction failed: {e}")


# -----------------------------
# Tab 2: Document-grounded Q&A
# -----------------------------
with tab_qa:
    st.subheader("💬 Ask questions about your documents")
    st.write(
        "This system answers only using the indexed documents. "
        "If the answer isn’t present, it should say it doesn’t know."
    )

    question = st.text_input(
        "Your question",
        placeholder="e.g., What is the comparison rate mentioned for fixed home loans? Provide citations.",
    )

    col1, col2, col3 = st.columns([1.2, 1, 1])

    with col1:
        qa_model = st.selectbox(
            "Local model (Ollama) for Q&A",
            ["llama3.1:8b", "phi3:mini"],
            index=0,
        )

    with col2:
        k = st.slider(
            "Retrieved chunks (k)",
            min_value=3,
            max_value=10,
            value=5,
            step=1,
            help="Lower k is usually better for Q&A precision.",
        )

    with col3:
        show_sources = st.checkbox("Show retrieved chunks", value=False)

    if st.button("Ask", type="primary", disabled=not bool(question.strip())):
        try:
            result = answer_with_ollama(question.strip(), k=k, model=qa_model)

            st.subheader("Answer")
            st.write(result["answer"])

            st.subheader("Citations")
            st.code(result["citations"])

            if show_sources:
                st.subheader("Retrieved chunks (debug)")
                for i, d in enumerate(result["sources"], start=1):
                    meta = d.metadata or {}
                    st.markdown(
                        f"**{i}. {meta.get('source','unknown')} | page={meta.get('page')} | chunk_id={meta.get('chunk_id')}**"
                    )
                    st.write(d.page_content)

        except Exception as e:
            st.error(f"Error: {e}")