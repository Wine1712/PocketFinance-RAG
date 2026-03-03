from __future__ import annotations

import json
from typing import Any, Dict, List

from langchain_community.chat_models import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document

from rag.prompts import EXTRACT_RATES_SYSTEM, EXTRACT_RATES_USER
from rag.qa import load_vectorstore, format_context


def _safe_json_loads(text: str) -> Dict[str, Any]:
    """
    Ollama usually returns clean JSON if prompted strictly.
    This adds a small safety net in case of accidental extra text.
    """
    text = text.strip()

    # Try direct
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try to extract first {...} block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])

    raise ValueError("Model did not return valid JSON.")


def retrieve_docs_for_product(product_type: str, k: int = 10) -> List[Document]:
    db = load_vectorstore()
    query = f"{product_type} interest rate comparison rate fixed variable term years bank"
    docs = db.max_marginal_relevance_search(query, k=k, fetch_k=max(30, k * 4))
    return docs


def extract_rates_table(product_type: str, model: str = "llama3.1:8b", k: int = 12) -> Dict[str, Any]:
    docs = retrieve_docs_for_product(product_type=product_type, k=k)
    context = format_context(docs)

    llm = ChatOllama(model=model, temperature=0.0)

    messages = [
        SystemMessage(content=EXTRACT_RATES_SYSTEM),
        HumanMessage(content=EXTRACT_RATES_USER.format(product_type=product_type, context=context)),
    ]
    resp = llm.invoke(messages)

    data = _safe_json_loads(resp.content)

    # Ensure schema exists
    if "items" not in data or not isinstance(data["items"], list):
        data = {"items": []}

    return data