"""
Paper analyzer: turns a stored publication into a provenance-carrying PaperAnalysisRecord.

1. Source text: the open-access full text if it was fetched, otherwise title + abstract.
   Metadata-only records are not analysed (nothing to ground statements in).
2. LLM extraction (PAPER_DIMENSIONS_PROMPT) at temperature 0. Every item must quote evidence.
3. Each excerpt is verified against the source text (verbatim, or fuzzy >= VERIFY_THRESHOLD to absorb
   PDF artefacts). Kinds: direct / interpretation / unverified (see research_schema).
4. Rule-based statements (the paper's own sentences, found by cue phrases) are always added; they are
   verbatim by construction. Without a live LLM they are the whole analysis, labelled as such.
5. Research dimensions are tagged deterministically with the concept ontology.
6. Everything is stored as ExtractionRecords (replacing the paper's previous analysis).
"""

import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from rapidfuzz import fuzz
from sqlalchemy import delete, select

from ..knowledge_base.ontology import find_concepts
from ..llm.client import ConfigurableLLMClient
from ..llm.extractor import extract_paper_signals, unwrap_pdf_text
from ..llm.json_utils import parse_llm_json
from ..llm.prompts import PAPER_DIMENSIONS_PROMPT
from ..storage import ExtractionRecord, Publication, session_scope
from .research_schema import (STATEMENT_FIELDS, PaperAnalysisRecord, Provenance, ResearchDimensions, Statement)

logger = logging.getLogger(__name__)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VERIFY_THRESHOLD = 92.0       # fuzzy partial match needed to count an excerpt as found in the source
MAX_SOURCE_CHARS = 80_000

_RULE_FIELD_MAP = {"limitations": "limitation", "future_work": "future_work", "open_questions": "unresolved_problem",
                   "results": "result", "claims": "result", "contributions": "objective"}


def normalize_for_match(text: str) -> str:
    import unicodedata
    t = unicodedata.normalize("NFKD", text or "")
    t = "".join(ch for ch in t if not unicodedata.combining(ch)).lower()   # Rényi == Renyi
    t = t.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    t = re.sub(r"[‐-―−]", "-", t)
    t = re.sub(r"(\w)-\s+(\w)", r"\1\2", t)          # line-break hyphenation
    t = re.sub(r"\s+", " ", t)
    return t.strip()


_NUM = re.compile(r"\d+(?:\.\d+)?")
_NEG = re.compile(r"\b(?:not|no|never|cannot|without|neither|nor|none)\b|n't\b")
_WORD = re.compile(r"[a-z]{4,}")


def _same_meaning(ex: str, source_norm: str, start: int, end: int) -> bool:
    """
    A fuzzy match finds WHERE the quote is; this checks it says the same thing there. Measured in Phase 9: fuzzy
    matching alone accepted quotes with a changed number, an inserted "not", or a swapped term. Required:
    every number and every content word of the quote occurs in the aligned passage, and negations agree in count.
    """
    core = source_norm[start:end]
    margin = 40 + len(ex) // 4                       # the alignment can stop short of a long quote's last words
    wide = source_norm[max(0, start - margin):end + margin]
    if not set(_NUM.findall(ex)) <= set(_NUM.findall(wide)):
        return False
    if len(_NEG.findall(ex)) != len(_NEG.findall(core)):
        return False
    unhyph = lambda t: t.replace("-", "")            # PDF line breaks: "meas-urements", "how-ever"
    return set(_WORD.findall(unhyph(ex))) <= set(_WORD.findall(unhyph(wide)))


def verify_excerpt(excerpt: str, source_norm: str, pages_norm: List[Tuple[int, str]]) -> Tuple[bool, float, Optional[int]]:
    """Is the excerpt in the source? Returns (verified, score 0..100, page)."""
    ex = normalize_for_match(excerpt)
    if len(ex) < 12:
        return False, 0.0, None
    if ex in source_norm:
        score = 100.0
        verified = True
    else:
        verified, score = False, 0.0
        if len(source_norm) < 400_000:
            al = fuzz.partial_ratio_alignment(ex, source_norm)
            score = al.score
            verified = score >= VERIFY_THRESHOLD and _same_meaning(ex, source_norm, al.dest_start, al.dest_end)
    page = None
    if verified and pages_norm:
        page = next((p for p, txt in pages_norm if ex in txt), None)
        if page is None:
            best = max(pages_norm, key=lambda pt: fuzz.partial_ratio(ex, pt[1]))
            page = best[0]
    return verified, round(score, 1), page


def strip_references(text: str) -> str:
    """
    Drop the bibliography: cited titles must not be tagged as this paper's own dimensions or statements.
    Handles an explicit heading ("References", "Bibliography"), Nature's "Online content" box that ends the
    main text, and headingless numbered lists ("1. Author ... (2019).").
    """
    floor = len(text) * 0.4          # never cut the first 40%: a bibliography is at the end
    cuts = [m.start() for m in re.finditer(
        r"\n\s*(?:\d+\.?\s*)?(?:references|bibliography|references and notes|online content)\s*\n", text, re.I)
        if m.start() > floor]
    for m in re.finditer(r"(?m)^\s*1\.\s+\S", text):
        if m.start() <= floor:
            continue
        window = text[m.start():m.start() + 3000]
        if len(re.findall(r"(?m)^\s*\d{1,3}\.\s+\S", window)) >= 5 and len(re.findall(r"\((?:19|20)\d{2}\)", window)) >= 3:
            cuts.append(m.start())
            break
    return text[:min(cuts)] if cuts else text


_QUBITS = re.compile(r"\b(\d{1,4})\s+(?:[a-z\-]+\s+){0,2}qubits?\b", re.I)


def qubit_counts(text: str) -> List[int]:
    return sorted({int(n) for n in _QUBITS.findall(text or "") if 0 < int(n) <= 5000})


class PaperAnalyzer:
    def __init__(self, llm: Optional[ConfigurableLLMClient] = None, db_url: Optional[str] = None, doc_extractor=None):
        self.llm = llm
        self.db_url = db_url
        self.doc_extractor = doc_extractor

    @property
    def live(self) -> bool:
        return self.llm is not None and not getattr(self.llm, "demo_mode", True)

    # ---------------------------------------------------------------- source
    def _source(self, pub: Publication) -> Tuple[str, str, List[Tuple[int, str]]]:
        """(kind, text, [(page, text)]) for analysis."""
        path = (pub.raw_metadata or {}).get("fulltext_path")
        if pub.full_text_available and path and os.path.exists(os.path.join(ROOT, path)):
            from ..document_processing.document_extractor import DocumentExtractor
            doc = (self.doc_extractor or DocumentExtractor()).extract_from_file(os.path.join(ROOT, path))
            pages = [(p.get("page_number"), unwrap_pdf_text(p.get("text", ""))) for p in doc.pages]
            text = strip_references(unwrap_pdf_text(doc.full_text))
            return "full_text", text, pages
        return "abstract", f"{pub.title}.\n\n{pub.abstract or ''}", []

    # --------------------------------------------------------------- analyse
    def analyse(self, publication_id: int, use_llm: bool = True) -> PaperAnalysisRecord:
        with session_scope(self.db_url) as s:
            pub = s.get(Publication, publication_id)
            if pub is None:
                raise KeyError(f"Unknown publication {publication_id}")
            title, has_abstract = pub.title, bool(pub.abstract)
            src_kind, source, pages = self._source(pub)
        record = PaperAnalysisRecord(publication_id=publication_id, title=title, analysed_from=src_kind)
        if src_kind == "abstract" and not has_abstract:
            record.warnings.append("Metadata only: no abstract or full text, so nothing could be analysed.")
            return record

        source_norm = normalize_for_match(source)
        pages_norm = [(p, normalize_for_match(t)) for p, t in pages]

        if use_llm and self.live:
            try:
                record.statements += self._llm_statements(publication_id, src_kind, source, source_norm, pages_norm)
                call = getattr(self.llm, "last_call", {}) or {}
                record.llm = f"{call.get('provider')}/{call.get('model')}"
            except Exception as exc:
                record.warnings.append(f"LLM extraction unavailable ({str(exc)[:120]}); rule-based statements only.")
        elif use_llm:
            record.warnings.append("No live LLM configured; rule-based statements only.")

        record.statements += self._rule_statements(publication_id, src_kind, source, record.statements, pages_norm)
        record.dimensions = self._dimensions(title, source, record.statements)
        unverified = sum(1 for st in record.statements if st.kind == "unverified")
        if unverified:
            record.warnings.append(f"{unverified} model statement(s) could not be found in the source and are marked unverified.")
        self._store(record)
        return record

    def _llm_statements(self, pid: int, src_kind: str, source: str, source_norm: str,
                        pages_norm: List[Tuple[int, str]]) -> List[Statement]:
        user = f"<UNTRUSTED_PAPER_TEXT>\n{source[:MAX_SOURCE_CHARS]}\n</UNTRUSTED_PAPER_TEXT>\n\nExtract the evidence-backed items."
        raw = self.llm.complete(PAPER_DIMENSIONS_PROMPT, user, temperature=0.0, fallback_to_demo=False,
                                purpose="paper_dimensions")
        data = parse_llm_json(raw)
        call = getattr(self.llm, "last_call", {}) or {}
        model = f"{call.get('provider', 'llm')}/{call.get('model', '?')}"
        out: List[Statement] = []
        for item in (data.get("items") or [])[:120]:
            field, text, evidence = item.get("field"), (item.get("text") or "").strip(), (item.get("evidence") or "").strip()
            if field not in STATEMENT_FIELDS or not text:
                continue
            verified, score, page = verify_excerpt(evidence, source_norm, pages_norm) if evidence else (False, 0.0, None)
            text_is_quote = verified and (normalize_for_match(text) in source_norm
                                          or fuzz.ratio(normalize_for_match(text), normalize_for_match(evidence)) >= 90)
            kind = "direct" if text_is_quote else ("interpretation" if verified else "unverified")
            conf = 0.9 if kind == "direct" else (round(0.75 * score / 100, 2) if kind == "interpretation" else 0.2)
            out.append(Statement(field=field, text=text[:1000], kind=kind, confidence=conf, extractor=model,
                                 provenance=Provenance(publication_id=pid, source=src_kind, excerpt=(evidence or text)[:600],
                                                       verified=verified, match_score=score, page=page)))
        return out

    def _rule_statements(self, pid: int, src_kind: str, source: str, existing: List[Statement],
                         pages_norm: List[Tuple[int, str]]) -> List[Statement]:
        seen = [normalize_for_match(st.provenance.excerpt) for st in existing if st.provenance and st.provenance.verified]
        signals = extract_paper_signals(source, cap=8)
        here_we = [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", re.sub(r"\s+", " ", source))
                   if re.search(r"(?i)\b(here,? we|in this (?:work|paper|letter|article),? we|we aim to)\b", s) and 40 <= len(s) <= 400][:2]
        candidates = [(f, s) for key, f in _RULE_FIELD_MAP.items() for s in signals.get(key, [])] + [("objective", s) for s in here_we]
        out: List[Statement] = []
        for field, sentence in candidates:
            norm = normalize_for_match(sentence)
            if any(fuzz.partial_ratio(norm, x) >= 90 for x in seen):
                continue  # already covered by a verified model statement
            seen.append(norm)
            page = next((p for p, txt in pages_norm if norm[:80] in txt), None)
            out.append(Statement(field=field, text=sentence, kind="direct", confidence=0.95, extractor="rule-based",
                                 provenance=Provenance(publication_id=pid, source=src_kind, excerpt=sentence, verified=True,
                                                       match_score=100.0, page=page)))
        return out

    @staticmethod
    def _dimensions(title: str, source: str, statements: List[Statement]) -> ResearchDimensions:
        """Tag from the paper's own words only: title + source text, never from unverified model output."""
        text = f"{title}\n{source}"
        found: Dict[str, List[str]] = {"platform": [], "method": [], "noise": [], "phenomenon": [], "metric": []}
        for h in find_concepts(text):
            if h.concept not in found[h.category]:
                found[h.category].append(h.concept)
        return ResearchDimensions(platforms=found["platform"], methods=found["method"], noise=found["noise"],
                                  phenomena=found["phenomenon"], metrics=found["metric"], qubit_counts=qubit_counts(text))

    # ----------------------------------------------------------------- store
    def _store(self, record: PaperAnalysisRecord) -> None:
        with session_scope(self.db_url) as s:
            s.execute(delete(ExtractionRecord).where(ExtractionRecord.publication_id == record.publication_id))
            for st in record.statements:
                p = st.provenance
                s.add(ExtractionRecord(publication_id=record.publication_id, field=st.field, value=st.text, kind=st.kind,
                                       page=p.page if p else None, section=p.section if p else None,
                                       excerpt=p.excerpt if p else None, excerpt_verified=bool(p and p.verified),
                                       confidence=st.confidence, extractor=st.extractor,
                                       live_llm=st.extractor not in ("rule-based", "ontology")))
            dims = record.dimensions
            for cat, values in (("platform", dims.platforms), ("method", dims.methods), ("noise", dims.noise),
                                ("phenomenon", dims.phenomena), ("metric", dims.metrics),
                                ("qubits", [str(q) for q in dims.qubit_counts])):
                for v in values:
                    s.add(ExtractionRecord(publication_id=record.publication_id, field=f"dim:{cat}", value=v, kind="direct",
                                           extractor="ontology", excerpt_verified=True, confidence=1.0))
            pub = s.get(Publication, record.publication_id)
            pub.analysed_at = datetime.now(timezone.utc)
            pub.raw_metadata = {**(pub.raw_metadata or {}), "analysis": {
                "analysed_from": record.analysed_from, "llm": record.llm, "warnings": record.warnings,
                "statements": len(record.statements), "verified_share": round(record.verified_share, 3)}}


def load_analysis(publication_id: int, db_url: Optional[str] = None) -> Optional[PaperAnalysisRecord]:
    """Rebuild a stored analysis from the database (None if the paper was never analysed)."""
    with session_scope(db_url) as s:
        pub = s.get(Publication, publication_id)
        if pub is None or pub.analysed_at is None:
            return None
        info = (pub.raw_metadata or {}).get("analysis", {})
        rec = PaperAnalysisRecord(publication_id=publication_id, title=pub.title,
                                  analysed_from=info.get("analysed_from", "abstract"), llm=info.get("llm"),
                                  warnings=info.get("warnings", []))
        dims: Dict[str, List] = {"platform": [], "method": [], "noise": [], "phenomenon": [], "metric": [], "qubits": []}
        for r in s.scalars(select(ExtractionRecord).where(ExtractionRecord.publication_id == publication_id)
                           .order_by(ExtractionRecord.id)):
            if r.field.startswith("dim:"):
                dims[r.field[4:]].append(int(r.value) if r.field == "dim:qubits" else r.value)
                continue
            rec.statements.append(Statement(
                field=r.field, text=r.value, kind=r.kind, confidence=r.confidence or 0.0, extractor=r.extractor,
                provenance=Provenance(publication_id=publication_id, source=rec.analysed_from, excerpt=r.excerpt or r.value,
                                      verified=r.excerpt_verified, match_score=100.0 if r.excerpt_verified else 0.0,
                                      page=r.page, section=r.section)))
        rec.dimensions = ResearchDimensions(platforms=dims["platform"], methods=dims["method"], noise=dims["noise"],
                                            phenomena=dims["phenomenon"], metrics=dims["metric"], qubit_counts=dims["qubits"])
        return rec
