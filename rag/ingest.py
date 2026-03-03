from __future__ import annotations

import os
from pathlib import Path
from typing import List

import pandas as pd
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

from rag.config import (
    RAW_DIR, DEMO_DIR, INDEX_DIR,
    EMBEDDING_MODEL_NAME, CHUNK_SIZE, CHUNK_OVERLAP
)


def _load_pdf(path: Path) -> List[Document]:
    loader = PyPDFLoader(str(path))
    docs = loader.load()
    # Add metadata
    for d in docs:
        d.metadata["source"] = path.name
        d.metadata["path"] = str(path)
        d.metadata.setdefault("page", d.metadata.get("page", None))
        d.metadata["doc_type"] = "pdf"
    return docs


def _load_text(path: Path) -> List[Document]:
    loader = TextLoader(str(path), encoding="utf-8")
    docs = loader.load()
    for d in docs:
        d.metadata["source"] = path.name
        d.metadata["path"] = str(path)
        d.metadata["doc_type"] = "text"
    return docs


def _load_csv(path: Path) -> List[Document]:
    """
    Convert CSV rows into compact documents for retrieval.
    Great for transaction exports or subscriptions lists.
    """
    df = pd.read_csv(path)
    docs: List[Document] = []

    # Keep it robust if columns vary
    cols = list(df.columns)

    for i, row in df.iterrows():
        # Small row-level doc (good for lookup + evidence)
        pairs = [f"{c}: {row[c]}" for c in cols if pd.notna(row[c])]
        content = "\n".join(pairs)

        docs.append(
            Document(
                page_content=content,
                metadata={
                    "source": path.name,
                    "path": str(path),
                    "doc_type": "csv_row",
                    "row_index": int(i),
                },
            )
        )

    return docs


def load_documents(data_dir: Path) -> List[Document]:
    all_docs: List[Document] = []
    if not data_dir.exists():
        return all_docs

    for path in data_dir.rglob("*"):
        if path.is_dir():
            continue
        suffix = path.suffix.lower()

        try:
            if suffix == ".pdf":
                all_docs.extend(_load_pdf(path))
            elif suffix in [".txt", ".md"]:
                all_docs.extend(_load_text(path))
            elif suffix == ".csv":
                all_docs.extend(_load_csv(path))
        except Exception as e:
            print(f"⚠️ Failed to load {path.name}: {e}")

    return all_docs


def build_or_update_index(use_demo: bool = False) -> int:
    data_dir = DEMO_DIR if use_demo else RAW_DIR
    docs = load_documents(data_dir)

    if not docs:
        raise RuntimeError(
            f"No documents found in {data_dir}. Add PDFs/CSVs/TXTs then re-run ingest."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    # Add stable chunk IDs
    for idx, d in enumerate(chunks):
        d.metadata["chunk_id"] = idx

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # If index exists, load and add; else create
    if (INDEX_DIR / "index.faiss").exists():
        db = FAISS.load_local(str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)
        db.add_documents(chunks)
    else:
        db = FAISS.from_documents(chunks, embeddings)

    db.save_local(str(INDEX_DIR))
    return len(chunks)


if __name__ == "__main__":
    # Default: build from data/raw
    n = build_or_update_index(use_demo=False)
    print(f"✅ Indexed {n} chunks into {INDEX_DIR}")