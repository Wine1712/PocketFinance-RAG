from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEMO_DIR = PROJECT_ROOT / "data" / "demo"

INDEX_DIR = PROJECT_ROOT / "index" / "faiss"

# Lightweight local embeddings
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking defaults (good for finance PDFs + policies)
CHUNK_SIZE = 900
CHUNK_OVERLAP = 160