"""
Vector store for the quantum literature knowledge base.

Primary backend: ChromaDB (persistent, cosine distance) with Sentence-Transformers embeddings.
Fallback backend: an in-memory word-count cosine matcher. It is NOT semantic; when it is in use
the knowledge base reports itself as degraded (see `backend`) and logs a warning, and with
KB_REQUIRE_SEMANTIC=1 it refuses to start instead of degrading.

Every chunk keeps its citation metadata (source file, page, chunk id, title).
"""

import logging
import math
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional

from ..document_processing.chunker import DocumentChunk, TraceableChunker
from .starter_corpus import STARTER_DOCUMENTS

logger = logging.getLogger(__name__)

try:
    import chromadb
    HAS_CHROMADB = True
except ImportError:  # pragma: no cover - depends on environment
    HAS_CHROMADB = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:  # pragma: no cover - depends on environment
    HAS_ST = False

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_MODEL_CACHE: Dict[str, Any] = {}   # one embedding model per process, not per knowledge base


class KnowledgeBaseUnavailable(RuntimeError):
    """Raised when semantic retrieval is required but not available."""


def _load_embedding_model(name: str):
    if name not in _MODEL_CACHE:
        _MODEL_CACHE[name] = SentenceTransformer(name)
    return _MODEL_CACHE[name]


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()[:40]


class FallbackVectorStore:
    """
    In-memory word-count cosine matcher (bag of words, no IDF, no semantics).
    Kept as a last-resort backend and as the lexical index; never presented as semantic search.
    """
    def __init__(self):
        self.documents: List[Dict[str, Any]] = []
        self._ids: set = set()

    def add(self, chunk: DocumentChunk, title: str):
        if chunk.chunk_id in self._ids:
            return
        self._ids.add(chunk.chunk_id)
        tokens = self._tokenize(chunk.text)
        self.documents.append({
            "chunk_id": chunk.chunk_id, "text": chunk.text, "source_filename": chunk.source_filename,
            "page_number": chunk.page_number, "title": title, "tokens": tokens, "tf": Counter(tokens),
        })

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\b[a-zA-Z0-9_\-]+\b", text.lower())

    def search(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        q_tf = Counter(self._tokenize(query))
        if not q_tf or not self.documents:
            return []
        norm_q = math.sqrt(sum(v * v for v in q_tf.values()))
        results = []
        for doc in self.documents:
            dot = sum(q_tf[w] * doc["tf"].get(w, 0) for w in q_tf)
            norm_d = math.sqrt(sum(v * v for v in doc["tf"].values()))
            score = (dot / (norm_q * norm_d)) if norm_d > 0 else 0.0
            results.append({k: doc[k] for k in ("chunk_id", "text", "source_filename", "page_number", "title")}
                           | {"score": round(score, 4), "retrieval": "word_match"})
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]


class QuantumKnowledgeBase:
    """Semantic storage and traceable retrieval of quantum literature."""

    def __init__(self, persist_dir: Optional[str] = None, collection_name: str = "quantum_literature",
                 require_semantic: Optional[bool] = None):
        self.persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIRECTORY") or os.path.join(ROOT, "chroma_db")
        self.model_name = os.getenv("EMBEDDING_MODEL_NAME") or "all-MiniLM-L6-v2"
        self.fallback = FallbackVectorStore()
        self.chroma_client = None
        self.collection = None
        self.embedding_model = None
        self.use_chroma = False
        self.degraded_reasons: List[str] = []
        if require_semantic is None:
            require_semantic = os.getenv("KB_REQUIRE_SEMANTIC", "0") == "1"
        # Collections are per embedding model so vectors from different models are never mixed.
        self.collection_name = f"{collection_name}__{_slug(self.model_name)}__cosine"
        self._initialize_backends()
        if require_semantic and not self.is_semantic:
            raise KnowledgeBaseUnavailable("Semantic retrieval required but unavailable: " + "; ".join(self.degraded_reasons))

    # ----------------------------------------------------------------- setup
    def _initialize_backends(self) -> None:
        if not HAS_ST:
            self.degraded_reasons.append("sentence-transformers is not installed")
        else:
            try:
                self.embedding_model = _load_embedding_model(self.model_name)
            except Exception as exc:
                self.degraded_reasons.append(f"embedding model '{self.model_name}' failed to load: {exc}")

        if not HAS_CHROMADB:
            self.degraded_reasons.append("chromadb is not installed")
        elif self.embedding_model is not None:
            try:
                os.makedirs(self.persist_dir, exist_ok=True)
                self.chroma_client = chromadb.PersistentClient(path=self.persist_dir)
                self.collection = self.chroma_client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine", "embedding_model": self.model_name})
                self.use_chroma = True
            except Exception as exc:
                self.degraded_reasons.append(f"ChromaDB failed to open at {self.persist_dir}: {exc}")

        if self.degraded_reasons:
            logger.warning("Knowledge base is DEGRADED (word matching, not semantic): " + "; ".join(self.degraded_reasons))

    @property
    def is_semantic(self) -> bool:
        return self.use_chroma and self.embedding_model is not None

    @property
    def backend(self) -> Dict[str, Any]:
        """What retrieval is actually running on: shown in the UI and recorded with results."""
        return {
            "vector_store": "chromadb (cosine)" if self.use_chroma else "in-memory word match",
            "embedding_model": self.model_name if self.embedding_model is not None else None,
            "semantic": self.is_semantic,
            "persist_dir": self.persist_dir if self.use_chroma else None,
            "collection": self.collection_name if self.use_chroma else None,
            "degraded_reasons": list(self.degraded_reasons),
        }

    def describe(self) -> str:
        b = self.backend
        if b["semantic"]:
            return f"Semantic search: ChromaDB + {b['embedding_model']}"
        return "Word matching only (not semantic): " + "; ".join(b["degraded_reasons"])

    # ------------------------------------------------------------------ write
    def index_chunks(self, chunks: List[DocumentChunk], source_title: str) -> int:
        """Add traceable chunks. Chroma write failures are logged and reported, not swallowed."""
        if not chunks:
            return 0
        for ch in chunks:
            self.fallback.add(ch, source_title)

        if self.is_semantic:
            try:
                texts = [ch.text for ch in chunks]
                embeddings = self.embedding_model.encode(texts, normalize_embeddings=True).tolist()
                self.collection.upsert(
                    ids=[ch.chunk_id for ch in chunks], documents=texts, embeddings=embeddings,
                    metadatas=[{"source_filename": ch.source_filename, "page_number": ch.page_number or 1,
                                "title": source_title, "chunk_id": ch.chunk_id} for ch in chunks])
            except Exception as exc:
                reason = f"ChromaDB write failed: {exc}"
                logger.error(reason)
                if reason not in self.degraded_reasons:
                    self.degraded_reasons.append(reason)
        return len(chunks)

    # ------------------------------------------------------------------- read
    def retrieve(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """Top passages with citation metadata. Each hit says which retrieval produced it."""
        if self.is_semantic and self.collection.count() > 0:
            try:
                q_emb = self.embedding_model.encode([query], normalize_embeddings=True).tolist()
                res = self.collection.query(query_embeddings=q_emb, n_results=min(top_k, self.collection.count()))
                docs = (res.get("documents") or [[]])[0]
                metas = (res.get("metadatas") or [[{}] * len(docs)])[0]
                dists = (res.get("distances") or [[None] * len(docs)])[0]
                return [{
                    "chunk_id": m.get("chunk_id", "unknown"), "text": d,
                    "source_filename": m.get("source_filename"), "page_number": m.get("page_number"),
                    "title": m.get("title"),
                    # cosine distance in [0, 2] -> cosine similarity in [-1, 1], clipped to [0, 1]
                    "score": round(max(0.0, 1.0 - dist), 4) if dist is not None else None,
                    "retrieval": f"semantic:{self.model_name}",
                } for d, m, dist in zip(docs, metas, dists)]
            except Exception as exc:
                logger.error(f"Semantic retrieval failed, using word matching for this query: {exc}")
        return self.fallback.search(query, top_k=top_k)

    def seed_starter_corpus(self) -> int:
        """Index the built-in reference texts (textbook-style summaries, see starter_corpus.py)."""
        chunker = TraceableChunker(chunk_size=500, chunk_overlap=80)
        total = 0
        for doc in STARTER_DOCUMENTS:
            chunks = chunker.chunk_document(full_text="", source_filename=doc["filename"], pages=doc.get("pages", []))
            total += self.index_chunks(chunks, source_title=doc["title"])
        return total

    def count(self) -> int:
        if self.is_semantic:
            try:
                return self.collection.count()
            except Exception as exc:
                logger.error(f"ChromaDB count failed: {exc}")
        return len(self.fallback.documents)
