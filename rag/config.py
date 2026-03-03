from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"
DEMO_DIR = BASE_DIR / "data" / "demo"
CLEAN_DIR = BASE_DIR / "data" / "clean_text"
INDEX_DIR = BASE_DIR / "index" / "faiss"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 160