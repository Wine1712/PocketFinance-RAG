from pathlib import Path
import re

from langchain_community.document_loaders import PyPDFLoader

RAW_DIR = Path("data/raw")
CLEAN_DIR = Path("data/clean_text")
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

def clean_text(s: str) -> str:
    s = s.replace("\x00", " ")
    s = re.sub(r"[ \t]+", " ", s)          # collapse spaces
    s = re.sub(r"\n{3,}", "\n\n", s)       # collapse many newlines
    s = re.sub(r"(\w)-\n(\w)", r"\1\2", s) # fix hyphen line breaks
    return s.strip()

for pdf_path in RAW_DIR.glob("*.pdf"):
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()

    full = "\n\n".join([p.page_content for p in pages])
    full = clean_text(full)

    out_path = CLEAN_DIR / (pdf_path.stem + ".txt")
    out_path.write_text(full, encoding="utf-8")
    print(f"✅ Saved: {out_path}")