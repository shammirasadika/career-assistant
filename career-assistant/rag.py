# rag.py
# Purpose: Lightweight TF-IDF retrieval + citation helpers, with labeled context.

from typing import List, Dict
from joblib import load
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class RAGIndex:
    """
    Simple TF-IDF-based retriever for document snippets.
    Loads a prebuilt index and returns top-k matches for a query.
    """
    def __init__(self, index_path: str, top_k: int = 5):
        blob = load(index_path)
        self.X = blob["X"]
        self.vectorizer = blob["vectorizer"]
        self.meta = blob["meta"]
        self.texts = blob["texts"]
        self.top_k = top_k

    def retrieve(self, query: str, k: int = None) -> List[Dict]:
        """
        Retrieve top-k most relevant document snippets for a given query.
        Returns: list of dicts with score, source, page, text, id.
        """
        if k is None:
            k = self.top_k

        # Transform query into TF-IDF vector
        q = self.vectorizer.transform([query])
        sims = cosine_similarity(q, self.X).ravel()

        # Get indices of top-k similarities
        top_idx = np.argsort(-sims)[:k]
        results = []
        for i in top_idx:
            m = self.meta[i]
            results.append({
                "score": float(sims[i]),
                "id": m.get("id"),
                "source": m.get("source"),
                "page": m.get("page"),
                "text": self.texts[i]
            })
        return results


def build_citations(snippets: List[Dict]) -> str:
    """
    Build a markdown-style citation block that matches app.py output.

    One line per source:
      - With page:  "[n] filename.pdf, p.X"
      - Without:    "[n] filename.pdf,"   (trailing comma as requested)
    """
    if not snippets:
        return "No sources available"

    seen = set()
    lines = []
    n = 1

    for s in snippets:
        src = (s.get("source") or "").strip()
        page = s.get("page")
        key = (src, page if page not in (None, "", 0) else "__NO_PAGE__")
        if not src or key in seen:
            continue
        seen.add(key)

        if page not in (None, "", 0):
            lines.append(f"[{n}] {src}, p.{page}")
        else:
            lines.append(f"[{n}] {src},")
        n += 1

    if not lines:
        return "No sources available"

    return "\n".join(lines)


# --------- NEW: simple snippet labeling to encourage distributed citations ---------

def _label_for(text: str) -> str:
    """
    Heuristic label for a snippet so the model can match claims to snippets.
    Tune keywords to your document’s headings/phrasing if needed.
    """
    t = (text or "").lower()

    # Duties / tasks
    if any(k in t for k in [
        "dutie", "duties", "task", "tasks", "indicative task", "responsibilit", "responsibility"
    ]):
        return "Duties"

    # Qualification / skill level
    if any(k in t for k in [
        "qualification", "skill level", "qualification/skill", "anzsco skill", "educational requirement"
    ]):
        return "Qualification/Skill level"

    # Alternative titles / specialisations
    if any(k in t for k in [
        "alternative title", "also known as", "job title", "specialisation", "specialization", "alias"
    ]):
        return "Alternative titles"

    # Processes / coordination / procedures
    if any(k in t for k in [
        "coordinate", "coordination", "procedure", "procedures", "process", "processes",
        "cross-functional", "quality assurance process", "quality assurance procedure"
    ]):
        return "Coordination/Process"

    return "General"


def make_context(snippets: List[Dict]) -> str:
    """
    Concatenate retrieved text snippets into a labeled context block
    for LLM input, where each snippet is prefixed with [n] and a label.

    Example output:
        [1] (Duties) Software engineers design, develop, test, and maintain systems.

        [2] (Qualification/Skill level) Typically skill level 1 (Bachelor degree) ...
    """
    if not snippets:
        return ""

    parts = []
    for i, s in enumerate(snippets, 1):
        text = (s.get("text") or "").strip()
        if not text:
            continue
        label = _label_for(text)
        parts.append(f"[{i}] ({label}) {text}")

    return "\n\n".join(parts)
