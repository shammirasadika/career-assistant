# data_prep.py
# Purpose: Build a TF-IDF index from files in data/raw (PDF + TXT supported).
# - Extract text from PDFs (per-page via PyPDF2; fallback to pdfminer.six whole-doc)
# - Paragraph-aware chunking with soft overlap (~900 chars)
# - Stable, repeatable IDs (SHA-1) per chunk
# - Save chunks to JSONL
# - Build TF-IDF index
#
# Run:  python data_prep.py
# Requires: PyPDF2, scikit-learn, joblib, PyYAML
# Optional: pdfminer.six

import os, re, json, glob, hashlib
from typing import List, Dict
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer

# ---- Try to import PDF libs ----
PDFMINER_AVAILABLE = True
try:
    from pdfminer_high_level import extract_text as pdfminer_extract  # type: ignore
except Exception:
    PDFMINER_AVAILABLE = False

from PyPDF2 import PdfReader  # primary per-page extraction


# --------- Utilities ----------
def clean_text(t: str) -> str:
    """Normalise whitespace; preserve paragraph breaks where possible."""
    if not t:
        return ""
    # Normalise line endings
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse spaces but keep blank lines that indicate paragraph breaks
    # First: trim trailing spaces on each line
    t = "\n".join([ln.strip() for ln in t.split("\n")])
    # Collapse 3+ newlines to 2 (blank line separation)
    t = re.sub(r"\n{3,}", "\n\n", t)
    # Inside lines: collapse internal whitespace
    t = re.sub(r"[ \t]+", " ", t)
    return t.strip()


def split_into_paragraphs(page_text: str) -> List[str]:
    """Split on blank lines; then merge tiny paragraphs so chunks are meaningful."""
    if not page_text:
        return []
    paras = [p.strip() for p in re.split(r"\n\s*\n", page_text) if p.strip()]
    merged: List[str] = []
    buf = ""
    for p in paras:
        # Merge small fragments (tune the 250-char threshold if needed)
        if len(p) < 250:
            buf = (buf + " " + p).strip() if buf else p
            if len(buf) >= 250:
                merged.append(buf); buf = ""
        else:
            if buf:
                merged.append(buf); buf = ""
            merged.append(p)
    if buf:
        merged.append(buf)
    return merged


def pack_paragraphs(paragraphs: List[str], target_len: int = 900, overlap: int = 150) -> List[str]:
    """
    Pack paragraphs into ~target_len char chunks, with soft tail overlap to preserve context.
    """
    chunks: List[str] = []
    cur = ""
    for p in paragraphs:
        if not cur:
            cur = p
            continue
        if len(cur) + 2 + len(p) <= target_len:
            cur = f"{cur}\n\n{p}"
        else:
            chunks.append(cur)
            tail = cur[-overlap:] if overlap and len(cur) > overlap else ""
            cur = (tail + "\n\n" + p).strip() if tail else p
    if cur:
        chunks.append(cur)
    # Drop very tiny chunks that are likely noise
    chunks = [c for c in chunks if len(c.strip()) >= 180]
    return chunks


def stable_id(source: str, page: int, idx: int, text: str) -> str:
    """
    Stable, repeatable ID based on source, page, local chunk index, and text length.
    """
    h = hashlib.sha1(f"{source}|{page}|{idx}|{len(text)}".encode("utf-8")).hexdigest()[:12]
    return f"{source}:p{page}:c{idx}:{h}"


# --------- Readers ----------
def read_txt(path: str) -> List[Dict]:
    """Return a single 'page' for a TXT file."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = clean_text(f.read())
    return [{"source": os.path.basename(path), "page": 1, "text": text}]


def read_pdf(path: str) -> List[Dict]:
    """
    Extract text per page with PyPDF2.
    If most pages come back empty, optionally fallback to pdfminer.six (whole doc).
    Returns list of dicts: {source, page, text}
    """
    pages = []
    empty_pages = 0
    try:
        reader = PdfReader(path)
        for i, page in enumerate(reader.pages, start=1):
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            t = clean_text(t)
            if not t:
                empty_pages += 1
            pages.append({"source": os.path.basename(path), "page": i, "text": t})
    except Exception:
        pages = []  # will try fallback below

    # Fallback if many pages were empty
    if (not pages) or (empty_pages >= max(1, len(pages) - 1)):
        if PDFMINER_AVAILABLE:
            try:
                t = clean_text(pdfminer_extract(path) or "")
                if t:
                    # Treat whole doc as page 1; still chunk paragraph-wise below
                    return [{"source": os.path.basename(path), "page": 1, "text": t}]
            except Exception:
                pass
        # If we reach here, we couldn't extract usable text
        return [{"source": os.path.basename(path), "page": 1, "text": ""}]

    return pages


# --------- Corpus & Index builders ----------
def build_corpus(raw_dir: str, out_jsonl: str, target_len: int, overlap: int):
    """
    Scan raw_dir for .pdf and .txt, paragraph-chunk them, and write records to JSONL:
      {"id","source","page","text","section_hint"}
    """
    records: List[Dict] = []
    # Ensure parent dir exists; default to '.' if none
    parent = os.path.dirname(out_jsonl) or "."
    os.makedirs(parent, exist_ok=True)

    files = sorted(glob.glob(os.path.join(raw_dir, "*.pdf")) +
                   glob.glob(os.path.join(raw_dir, "*.txt")))
    if not files:
        print(f"[WARN] No input files found in {raw_dir}. Put PDFs or TXTs there.")
    for fp in files:
        base = os.path.basename(fp)
        if base.lower().endswith(".txt"):
            pages = read_txt(fp)
        else:
            pages = read_pdf(fp)

        for p in pages:
            paragraphs = split_into_paragraphs(p["text"])
            chunks = pack_paragraphs(paragraphs, target_len=target_len, overlap=overlap)

            for idx, c in enumerate(chunks):
                if not c.strip():
                    continue
                rid = stable_id(p["source"], p["page"], idx, c)
                section_hint = c.split("\n", 1)[0][:140]
                records.append({
                    "id": rid,
                    "source": p["source"],
                    "page": p["page"],
                    "text": c,
                    "section_hint": section_hint
                })

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved {len(records)} chunks to {out_jsonl}")

    if len(records) == 0:
        msg = (
            "No text chunks were created.\n"
            f"- Check files in '{raw_dir}' (PDFs/TXTs).\n"
            "- Make sure PDFs have selectable text (not scanned images).\n"
            "- Try pdfminer.six (pip install pdfminer.six) or add a small .txt in data/raw to test."
        )
        raise ValueError(msg)


def build_index(corpus_jsonl: str, index_pkl: str):
    """Load JSONL chunks and build a TF-IDF index."""
    texts, meta = [], []
    if not os.path.exists(corpus_jsonl):
        raise FileNotFoundError(f"Corpus file not found: {corpus_jsonl}")

    with open(corpus_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            txt = (obj.get("text") or "").strip()
            if txt:
                texts.append(txt)
                meta.append({
                    "id": obj["id"],
                    "source": obj["source"],
                    "page": obj["page"],
                    "section_hint": obj.get("section_hint") or ""
                })

    print(f"Loaded {len(texts)} texts from {corpus_jsonl}")
    if not texts:
        raise ValueError("Corpus has 0 non-empty texts. Please re-check your PDF/TXT sources.")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,                 # tolerant; bump to 2 if corpus grows
        max_features=60000,
        token_pattern=r"(?u)\b\w\w+\b"
    )
    try:
        X = vectorizer.fit_transform(texts)
    except ValueError as e:
        raise ValueError(
            f"Failed to build TF-IDF vocabulary: {e}\n"
            "Hint: ensure your PDFs yield real text (not images) or add a TXT file."
        )

    dump({"X": X, "vectorizer": vectorizer, "meta": meta, "texts": texts}, index_pkl)
    print(f"Index saved to {index_pkl}. Shapes: X={X.shape}")


# --------- Main ----------
if __name__ == "__main__":
    import yaml
    # Read config with UTF-8 to avoid Windows cp1252 issues
    with open("config.yml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    retr = cfg.get("retrieval", {})
    # Map your existing keys to the new semantics
    target_len = int(retr.get("chunk_size", 900))     # now interpreted as target chunk length (chars)
    overlap    = int(retr.get("overlap", retr.get("chunk_overlap", 150)))
    corpus_path = retr.get("corpus_path", "data/corpus.jsonl")
    index_path  = retr.get("index_path",  "data/index.pkl")

    raw_dir = "data/raw"  # keep fixed as per your layout

    print(f"Using raw_dir={raw_dir}, target_len={target_len}, overlap={overlap}")
    print(f"Output: corpus={corpus_path}, index={index_path}")
    if not PDFMINER_AVAILABLE:
        print("[INFO] pdfminer.six not installed; using PyPDF2 only. "
              "For better PDF extraction: pip install pdfminer.six")

    build_corpus(raw_dir, corpus_path, target_len, overlap)
    build_index(corpus_path, index_path)  # <-- end cleanly
