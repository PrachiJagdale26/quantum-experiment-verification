"""
PaperLibrary: stage 1 of the pipeline. Ingests papers (PDF / DOCX / TXT),
analyses each one into a ResearchPaperAnalysis, and indexes its chunks into the
knowledge base so later stages can retrieve and cite it.
"""

import glob
import os
from typing import Dict, List, Optional, Tuple

from ..document_processing.chunker import TraceableChunker
from ..document_processing.document_extractor import DocumentExtractor
from ..knowledge_base.vector_store import QuantumKnowledgeBase
from ..llm.extractor import ProposalExtractor, guess_title
from ..schemas.research_paper import ResearchPaperAnalysis

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAMPLE_PAPERS_DIR = os.path.join(ROOT, "data", "sample_papers")
SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".txt")


class PaperIngestError(Exception):
    """Raised when a document cannot be turned into a paper analysis."""


class PaperLibrary:
    def __init__(
        self,
        extractor: ProposalExtractor,
        kb: Optional[QuantumKnowledgeBase] = None,
        doc_extractor: Optional[DocumentExtractor] = None,
        chunker: Optional[TraceableChunker] = None,
    ):
        self.extractor = extractor
        self.kb = kb
        self.doc_extractor = doc_extractor or DocumentExtractor()
        self.chunker = chunker or TraceableChunker(chunk_size=500, chunk_overlap=80)
        self._papers: Dict[str, ResearchPaperAnalysis] = {}
        self._counter = 0

    # ------------------------------------------------------------------ ingest
    def add_document(self, data: bytes, filename: str) -> Tuple[str, ResearchPaperAnalysis]:
        ext = os.path.splitext(filename)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise PaperIngestError(f"{ext or 'Files without an extension'} aren't supported. Upload a PDF, DOCX or TXT file.")
        try:
            doc = self.doc_extractor.extract_from_bytes(data, filename=filename)
        except Exception as exc:
            raise PaperIngestError(f"The document couldn't be parsed: {exc}") from exc
        if not doc.full_text.strip():
            raise PaperIngestError("The document has no readable text layer. Upload a DOCX or a text-based PDF.")

        title = guess_title(doc.full_text)
        paper = self.extractor.extract_research_paper(
            doc.full_text, document_title=title or os.path.splitext(filename)[0], extracted_doc=doc)
        if not paper.source_filename:
            paper.source_filename = filename

        if self.kb is not None:
            chunks = self.chunker.chunk_document(doc.full_text, filename, pages=doc.pages)
            self.kb.index_chunks(chunks, source_title=paper.title)

        # Re-adding the same file replaces the earlier analysis instead of duplicating it.
        for pid, existing in list(self._papers.items()):
            if existing.source_filename == filename:
                self._papers[pid] = paper
                return pid, paper
        self._counter += 1
        pid = f"PAPER-{self._counter:02d}"
        self._papers[pid] = paper
        return pid, paper

    def add_file(self, path: str) -> Tuple[str, ResearchPaperAnalysis]:
        with open(path, "rb") as f:
            return self.add_document(f.read(), os.path.basename(path))

    def load_samples(self, directory: str = SAMPLE_PAPERS_DIR) -> List[str]:
        ids = []
        for path in sorted(glob.glob(os.path.join(directory, "*"))):
            if path.lower().endswith(SUPPORTED_EXTENSIONS):
                ids.append(self.add_file(path)[0])
        return ids

    # ------------------------------------------------------------------ access
    def papers(self) -> List[Tuple[str, ResearchPaperAnalysis]]:
        return list(self._papers.items())

    def get(self, paper_id: str) -> Optional[ResearchPaperAnalysis]:
        return self._papers.get(paper_id)

    def remove(self, paper_id: str) -> None:
        self._papers.pop(paper_id, None)

    def clear(self) -> None:
        self._papers.clear()

    def __len__(self) -> int:
        return len(self._papers)
