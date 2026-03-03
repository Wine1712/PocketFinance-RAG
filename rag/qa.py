from __future__ import annotations

from typing import Dict, Any, List

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from rag.config import INDEX_DIR, EMBEDDING_MODEL_NAME
from rag.prompts import SYSTEM_PROMPT, USER_PROMPT


def load_vectorstore() -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    db = FAISS.load_local(str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)
    return db


def _truncate(text: str, limit: int = 1200) -> str:
    return text[:limit]


def format_context(docs: List[Document]) -> str:
    parts = []
    for d in docs:
        meta = d.metadata or {}
        source = meta.get("source", "unknown")
        page = meta.get("page", None)
        chunk_id = meta.get("chunk_id", None)

        header = f"[source={source} page={page} chunk_id={chunk_id}]"
        parts.append(header + "\n" + d.page_content)
    return "\n\n---\n\n".join(parts)


def format_citations(docs: List[Document]) -> str:
    cites = []
    for d in docs:
        meta = d.metadata or {}
        cites.append(
            f"- {meta.get('source','unknown')} | page={meta.get('page', None)} | chunk_id={meta.get('chunk_id', None)}"
        )

    seen = set()
    out = []
    for c in cites:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return "\n".join(out)


def answer_with_ollama(question: str, k: int = 5, model: str = "llama3.1:8b") -> Dict[str, Any]:
    from langchain_community.chat_models import ChatOllama
    from langchain_core.messages import SystemMessage, HumanMessage

    db = load_vectorstore()

    docs = db.max_marginal_relevance_search(question, k=k, fetch_k=max(20, k * 4))

    # ✅ truncate BEFORE building context
    for d in docs:
        d.page_content = _truncate(d.page_content, 1200)

    context = format_context(docs)
    citations = format_citations(docs)

    llm = ChatOllama(model=model, temperature=0.1)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=USER_PROMPT.format(question=question, context=context)),
    ]
    resp = llm.invoke(messages)

    return {
        "answer": resp.content.strip(),
        "citations": citations,
        "sources": docs,
    }