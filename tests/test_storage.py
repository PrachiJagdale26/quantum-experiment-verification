"""Research database: schema creation, provenance fields, uniqueness and cascades (in-memory SQLite)."""

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from src.storage import (CandidateGap, DimensionComparison, ExtractionRecord, GapEvidence, NoveltyAssessment,
                         Publication, SearchHit, SearchRun, session_scope)

URL = "sqlite:///:memory:"


def _pub(**kw):
    base = dict(title="Measurement-induced entanglement and teleportation on a noisy quantum processor",
                normalized_title="measurement induced entanglement and teleportation on a noisy quantum processor",
                year=2023, source="openalex", doi="10.1038/s41586-023-06505-7")
    base.update(kw)
    return Publication(**base)


def test_publication_with_provenanced_extraction_round_trips(tmp_path):
    url = f"sqlite:///{tmp_path / 'r.db'}"
    with session_scope(url) as s:
        p = _pub()
        p.extractions.append(ExtractionRecord(field="limitation", value="Limited to 70 qubits.", kind="direct",
                                              page=3, section="Discussion", excerpt="up to 70 superconducting qubits",
                                              excerpt_verified=True, extractor="rule-based"))
        s.add(p)
    with session_scope(url) as s:
        p = s.scalars(select(Publication)).one()
        assert p.full_text_available is False and p.is_synthetic is False
        e = p.extractions[0]
        assert (e.kind, e.page, e.excerpt_verified) == ("direct", 3, True)


def test_doi_is_unique(tmp_path):
    url = f"sqlite:///{tmp_path / 'u.db'}"
    with session_scope(url) as s:
        s.add(_pub())
    with pytest.raises(IntegrityError):
        with session_scope(url) as s:
            s.add(_pub(title="dup"))


def test_novelty_assessment_records_searches_and_comparisons(tmp_path):
    url = f"sqlite:///{tmp_path / 'n.db'}"
    with session_scope(url) as s:
        pub = _pub()
        gap = CandidateGap(title="GHZ beyond 3 qubits on transmons", statement="...", origin="scale_limit",
                           generator="deterministic")
        gap.evidence.append(GapEvidence(role="supporting", quote="We only demonstrate GHZ preparation up to 3 qubits"))
        a = NoveltyAssessment(classification="insufficient_search_coverage", confidence=0.2,
                              coverage={"sources_ok": ["arxiv"], "sources_failed": ["openalex"]})
        a.comparisons.append(DimensionComparison(dimension="platform", result="same", method="rule"))
        gap.assessments.append(a)
        s.add_all([pub, gap])
        s.flush()
        run = SearchRun(purpose="novelty", source="arxiv", query="GHZ transmon", novelty_assessment_id=a.id, result_count=1)
        run.hits.append(SearchHit(rank=1, publication_id=pub.id, score=0.8))
        s.add(run)
    with session_scope(url) as s:
        gap = s.scalars(select(CandidateGap)).one()
        assert gap.review_status == "candidate"
        assert gap.assessments[0].comparisons[0].dimension == "platform"
        assert s.scalars(select(SearchRun)).one().hits[0].rank == 1
        s.delete(gap)
    with session_scope(url) as s:
        assert s.scalars(select(NoveltyAssessment)).all() == []
