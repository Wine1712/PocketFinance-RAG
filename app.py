import streamlit as st
from rag.ingest import build_or_update_index
from rag.qa import answer_with_ollama
from rag.config import RAW_DIR

st.set_page_config(page_title="PocketFinance-RAG", page_icon="💳", layout="wide")

st.title("💳 PocketFinance-RAG (Local Document Assistant)")
st.caption("Runs locally. Index your finance PDFs/CSVs and ask questions with citations.")

with st.sidebar:
    st.header("1) Add documents")
    st.write(f"Put your files here (not uploaded anywhere):")
    st.code(str(RAW_DIR))
    st.write("Supported: PDF, CSV, TXT, MD")

    st.header("2) Build / Update Index")
    if st.button("Build index from data/raw"):
        try:
            n = build_or_update_index(use_demo=False)
            st.success(f"Indexed {n} chunks ✅")
        except Exception as e:
            st.error(str(e))

st.divider()

question = st.text_input("Ask a question about your documents:", placeholder="e.g., What are the late fees in my credit card terms?")

col1, col2 = st.columns([2, 1])

with col1:
    if st.button("Ask", type="primary", disabled=not bool(question.strip())):
        try:
            result = answer_with_ollama(question.strip(), k=6)
            st.subheader("Answer")
            st.write(result["answer"])

            st.subheader("Citations")
            st.code(result["citations"])
        except Exception as e:
            st.error(f"Error: {e}")

with col2:
    st.subheader("Tips")
    st.write("- Ask for **fees**, **interest**, **cancellation**, **refund**, **policy rules**.")
    st.write("- Try: “Cite the exact clause” or “Show evidence snippets” (coming next).")