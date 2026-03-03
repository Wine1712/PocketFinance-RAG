💳 PocketFinance-RAG

A Local, Retrieval-Augmented Financial Product Intelligence System

⸻

🚀 Overview

PocketFinance-RAG is a local Retrieval-Augmented Generation (RAG) system designed to perform document-grounded financial analysis and structured product comparison across public bank disclosures.

The system ingests financial PDFs (home loans, savings accounts, credit cards, car loans), performs semantic retrieval using vector search, and generates:
    •    Citation-backed answers
    •    Structured interest rate comparison tables
    •    Product-level financial extraction

All computation runs locally using open-source LLMs.

This project demonstrates applied GenAI engineering for regulated domains where transparency and document grounding are critical.

⸻

🎯 Objective

The goal of this project is to showcase:
    •    Practical RAG system architecture
    •    Hallucination control via document grounding
    •    Structured financial data extraction from unstructured PDFs
    •    Local-first GenAI deployment (no external API dependency)
    •    Production-style modular design

Unlike generic chatbots, this system only answers based on indexed financial documents and always provides source citations.

⸻

🏗 System Architecture

1️⃣ Document Ingestion Layer
    •    Loads PDF, CSV, and text documents
    •    Applies recursive semantic chunking
    •    Adds metadata (source, page, chunk ID)

2️⃣ Embedding Layer
    •    Sentence-transformers (all-MiniLM-L6-v2)
    •    Converts chunks into dense vectors

3️⃣ Vector Store
    •    FAISS (local)
    •    Supports fast similarity search
    •    Uses Max Marginal Relevance (MMR) for better coverage

4️⃣ Retrieval Layer
    •    Retrieves top-k relevant chunks
    •    Adjustable retrieval depth
    •    Optimized for multi-bank comparisons

5️⃣ Generation Layer
    •    Ollama (local LLM runtime)
    •    Models: llama3.1:8b, phi3:mini
    •    Prompt-engineered for:
    •    Strict context grounding
    •    JSON-only structured extraction
    •    Citation enforcement

6️⃣ UI Layer
    •    Streamlit-based interface
    •    Separate tabs for:
    •    Document-grounded Q&A
    •    Structured loan comparison
    •    Structured savings comparison

⸻

🧠 Core Capabilities

📌 Document-Grounded Q&A

Example queries:
    •    “What is the comparison rate for this home loan?”
    •    “Where is the late payment interest defined?”
    •    “Summarize the exclusions in this product disclosure statement.”

Responses:
    •    Generated only from retrieved document chunks
    •    Include citations (source, page, chunk ID)
    •    No hallucinated external knowledge

⸻

📊 Structured Financial Comparison

Users can select:
    •    Home loan (fixed / variable)
    •    Car loan
    •    Personal loan
    •    Credit card interest
    •    High-interest savings
    •    Term deposits

The system retrieves relevant sections and extracts structured JSON:

Bank    Product Type    Term    Rate    Comparison Rate    Conditions


This demonstrates LLM-powered structured data extraction on top of RAG retrieval.

⸻

⚙️ Technology Stack

Component    Tool
Programming Language    Python
RAG Framework    LangChain
Embeddings    Sentence-Transformers
Vector Database    FAISS
LLM Runtime    Ollama
UI    Streamlit
Data Source    Public financial PDFs


⸻

🔬 GenAI Engineering Concepts Demonstrated
    •    Retrieval-Augmented Generation (RAG)
    •    Vector similarity search
    •    MMR retrieval tuning
    •    Chunking strategy design
    •    Prompt engineering for structured output
    •    JSON schema enforcement
    •    Hallucination mitigation
    •    Local LLM orchestration
    •    Modular AI system design

⸻

🛡 Why Local Deployment?

Financial applications require:
    •    Privacy
    •    Traceability
    •    Deterministic document grounding
    •    Controlled model behavior

This system runs entirely offline:
    •    No API calls
    •    No external data transfer
    •    Full reproducibility

⸻

📈 Production-Relevant Features
    •    Adjustable retrieval depth (k)
    •    Structured comparison tables
    •    Strict JSON extraction prompts
    •    Citation traceability
    •    Extensible architecture (easy to add tools or product types)

⸻

🔮 Future Extensions
    •    Hybrid retrieval (BM25 + vector)
    •    Automatic rate normalization and sorting
    •    Confidence scoring
    •    Multi-document consistency validation
    •    Financial calculation tools (e.g., loan repayment simulation)
    •    Deployment via FastAPI backend

⸻

💡 Summary

PocketFinance-RAG demonstrates how Retrieval-Augmented Generation can be applied to regulated financial documents to enable grounded, structured, and explainable AI-driven product comparison.

This project highlights real-world GenAI engineering practices beyond chatbot prototypes.

