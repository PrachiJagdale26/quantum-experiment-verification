"""BM25 keyword index over knowledge-base chunks (the lexical half of hybrid retrieval)."""

import re
from typing import Any, Dict, List

from rank_bm25 import BM25Okapi

_STOP = set("the a an and or of to in on for with by is are was were be been this that these those we our it its as at "
            "from not can will may into than then also which such there their has have had do does did using use".split())


def tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9][a-z0-9\-]*", (text or "").lower()) if t not in _STOP and len(t) > 1]


class BM25Index:
    """Rebuilt lazily whenever the document list grows; documents are dicts with at least 'text' and 'chunk_id'."""

    def __init__(self, documents: List[Dict[str, Any]]):
        self.documents = documents
        self._n = -1
        self._bm25 = None

    def _ensure(self) -> None:
        if self._n != len(self.documents):
            corpus = [tokenize(d["text"]) or ["_"] for d in self.documents]
            self._bm25 = BM25Okapi(corpus) if corpus else None
            self._n = len(self.documents)

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        self._ensure()
        q = tokenize(query)
        if not q or self._bm25 is None:
            return []
        scores = self._bm25.get_scores(q)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [{**{k: self.documents[i][k] for k in ("chunk_id", "text", "source_filename", "page_number", "title")},
                 "bm25": round(float(scores[i]), 4)} for i in ranked if scores[i] > 0]
