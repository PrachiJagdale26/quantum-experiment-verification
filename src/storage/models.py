"""
Relational schema for the research-discovery framework (SQLAlchemy 2.0).

Design rules:
- Every scientific object keeps its provenance (source, URL, retrieval time, page, excerpt).
- Content is labelled by kind: a direct statement from a paper, an LLM interpretation,
  or an inference across papers. Generated content is never stored as if it were evidence.
- Search runs and LLM calls are logged so that novelty assessments can be reproduced and audited.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


# --------------------------------------------------------------------------- literature
class Publication(Base):
    """One scholarly work, deduplicated across sources (DOI > arXiv id > OpenAlex id > fuzzy title)."""
    __tablename__ = "publications"

    id: Mapped[int] = mapped_column(primary_key=True)
    doi: Mapped[Optional[str]] = mapped_column(String(255), unique=True, index=True)
    arxiv_id: Mapped[Optional[str]] = mapped_column(String(64), unique=True, index=True)
    openalex_id: Mapped[Optional[str]] = mapped_column(String(64), unique=True, index=True)
    title: Mapped[str] = mapped_column(Text)
    normalized_title: Mapped[str] = mapped_column(String(512), index=True)
    year: Mapped[Optional[int]] = mapped_column(Integer, index=True)
    publication_date: Mapped[Optional[str]] = mapped_column(String(32))
    venue: Mapped[Optional[str]] = mapped_column(String(512))
    authors: Mapped[List[str]] = mapped_column(JSON, default=list)
    abstract: Mapped[Optional[str]] = mapped_column(Text)
    abstract_source: Mapped[Optional[str]] = mapped_column(String(32), doc="Which API supplied the abstract, if any.")
    source: Mapped[str] = mapped_column(String(32), doc="openalex | arxiv | crossref | semantic_scholar | upload")
    source_url: Mapped[Optional[str]] = mapped_column(Text)
    pdf_url: Mapped[Optional[str]] = mapped_column(Text)
    is_open_access: Mapped[bool] = mapped_column(Boolean, default=False)
    full_text_available: Mapped[bool] = mapped_column(Boolean, default=False,
                                                      doc="True only if full text was actually obtained, not merely linked.")
    is_synthetic: Mapped[bool] = mapped_column(Boolean, default=False,
                                               doc="Synthetic demo documents; excluded from evaluation and evidence.")
    curation_status: Mapped[str] = mapped_column(String(16), default="unreviewed",
                                                 doc="unreviewed | included | excluded: a researcher's corpus decision.")
    curation_note: Mapped[Optional[str]] = mapped_column(Text)
    analysed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    discovered_via: Mapped[str] = mapped_column(String(16), default="collection",
                                                doc="collection | novelty | citation: how the work first entered the database. "
                                                    "Only 'collection' works form the researcher's corpus.")
    retrieved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    raw_metadata: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)

    versions: Mapped[List["PublicationVersion"]] = relationship(back_populates="publication", cascade="all, delete-orphan")
    extractions: Mapped[List["ExtractionRecord"]] = relationship(back_populates="publication", cascade="all, delete-orphan")


class PublicationVersion(Base):
    """Versions of a work (e.g. arXiv v1, v2, journal version) with their dates."""
    __tablename__ = "publication_versions"
    __table_args__ = (UniqueConstraint("publication_id", "version_label"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    publication_id: Mapped[int] = mapped_column(ForeignKey("publications.id", ondelete="CASCADE"))
    version_label: Mapped[str] = mapped_column(String(64))
    version_date: Mapped[Optional[str]] = mapped_column(String(32))
    source_url: Mapped[Optional[str]] = mapped_column(Text)
    retrieved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    publication: Mapped[Publication] = relationship(back_populates="versions")


class ExtractionRecord(Base):
    """One extracted research field with provenance and an explicit kind."""
    __tablename__ = "extraction_records"

    id: Mapped[int] = mapped_column(primary_key=True)
    publication_id: Mapped[int] = mapped_column(ForeignKey("publications.id", ondelete="CASCADE"), index=True)
    field: Mapped[str] = mapped_column(String(64), index=True, doc="e.g. limitation, future_work, method, platform")
    value: Mapped[str] = mapped_column(Text)
    kind: Mapped[str] = mapped_column(String(16), doc="direct | interpretation | inference")
    page: Mapped[Optional[int]] = mapped_column(Integer)
    section: Mapped[Optional[str]] = mapped_column(String(128))
    excerpt: Mapped[Optional[str]] = mapped_column(Text, doc="Verbatim supporting text from the source.")
    excerpt_verified: Mapped[bool] = mapped_column(Boolean, default=False,
                                                   doc="True if the excerpt was found verbatim in the source text.")
    confidence: Mapped[Optional[float]] = mapped_column(Float)
    extractor: Mapped[str] = mapped_column(String(128), doc="rule-based, or provider/model for LLM extraction")
    live_llm: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    publication: Mapped[Publication] = relationship(back_populates="extractions")


# ------------------------------------------------------------------------------ searches
class SearchRun(Base):
    """One query sent to one source, for collection or for a novelty investigation."""
    __tablename__ = "search_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    purpose: Mapped[str] = mapped_column(String(32), doc="collection | novelty | citation")
    source: Mapped[str] = mapped_column(String(32))
    query: Mapped[str] = mapped_column(Text)
    query_kind: Mapped[Optional[str]] = mapped_column(String(32), doc="exact | synonym | llm_alternative | method | condition")
    filters: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(16), default="ok", doc="ok | failed | rate_limited | cached")
    error: Mapped[Optional[str]] = mapped_column(Text)
    result_count: Mapped[int] = mapped_column(Integer, default=0)
    total_available: Mapped[Optional[int]] = mapped_column(Integer)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    novelty_assessment_id: Mapped[Optional[int]] = mapped_column(ForeignKey("novelty_assessments.id", ondelete="SET NULL"))

    hits: Mapped[List["SearchHit"]] = relationship(back_populates="search_run", cascade="all, delete-orphan")


class SearchHit(Base):
    __tablename__ = "search_hits"

    id: Mapped[int] = mapped_column(primary_key=True)
    search_run_id: Mapped[int] = mapped_column(ForeignKey("search_runs.id", ondelete="CASCADE"), index=True)
    publication_id: Mapped[Optional[int]] = mapped_column(ForeignKey("publications.id", ondelete="SET NULL"))
    rank: Mapped[int] = mapped_column(Integer)
    score: Mapped[Optional[float]] = mapped_column(Float)

    search_run: Mapped[SearchRun] = relationship(back_populates="hits")


# ---------------------------------------------------------------------------------- gaps
class CandidateGap(Base):
    """A candidate research gap. Never 'validated' without human review."""
    __tablename__ = "candidate_gaps"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(Text)
    statement: Mapped[str] = mapped_column(Text)
    domain: Mapped[str] = mapped_column(String(128), default="quantum physics")
    origin: Mapped[str] = mapped_column(String(48), doc="stated_limitation | future_work | unexplored_combination | ...")
    classification: Mapped[Optional[str]] = mapped_column(String(48), doc="Set by the novelty investigation.")
    review_status: Mapped[str] = mapped_column(String(16), default="candidate", doc="candidate | validated | rejected")
    generator: Mapped[str] = mapped_column(String(128), doc="deterministic, or provider/model")
    payload: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict, doc="Dimensions, research question, rationale, scores.")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    evidence: Mapped[List["GapEvidence"]] = relationship(back_populates="gap", cascade="all, delete-orphan")
    assessments: Mapped[List["NoveltyAssessment"]] = relationship(back_populates="gap", cascade="all, delete-orphan")


class GapEvidence(Base):
    __tablename__ = "gap_evidence"

    id: Mapped[int] = mapped_column(primary_key=True)
    gap_id: Mapped[int] = mapped_column(ForeignKey("candidate_gaps.id", ondelete="CASCADE"), index=True)
    publication_id: Mapped[Optional[int]] = mapped_column(ForeignKey("publications.id", ondelete="SET NULL"))
    role: Mapped[str] = mapped_column(String(16), doc="supporting | contradicting | related")
    quote: Mapped[str] = mapped_column(Text)
    page: Mapped[Optional[int]] = mapped_column(Integer)
    section: Mapped[Optional[str]] = mapped_column(String(128))
    kind: Mapped[str] = mapped_column(String(16), default="direct", doc="direct | inference")

    gap: Mapped[CandidateGap] = relationship(back_populates="evidence")


class NoveltyAssessment(Base):
    """A recorded novelty investigation: queries, coverage, comparisons, rule-based conclusion."""
    __tablename__ = "novelty_assessments"

    id: Mapped[int] = mapped_column(primary_key=True)
    gap_id: Mapped[int] = mapped_column(ForeignKey("candidate_gaps.id", ondelete="CASCADE"), index=True)
    classification: Mapped[str] = mapped_column(String(48))
    confidence: Mapped[float] = mapped_column(Float)
    confidence_basis: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict, doc="Inputs to the confidence formula.")
    coverage: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict, doc="Sources searched/failed, queries, hits screened.")
    findings: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict, doc="Closest papers, similarities, differences, evidence for/against.")
    uncertainty: Mapped[str] = mapped_column(Text, default="")
    searched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    gap: Mapped[CandidateGap] = relationship(back_populates="assessments")
    comparisons: Mapped[List["DimensionComparison"]] = relationship(back_populates="assessment", cascade="all, delete-orphan")


class DimensionComparison(Base):
    """Gap vs. one related paper on one research dimension."""
    __tablename__ = "dimension_comparisons"

    id: Mapped[int] = mapped_column(primary_key=True)
    assessment_id: Mapped[int] = mapped_column(ForeignKey("novelty_assessments.id", ondelete="CASCADE"), index=True)
    publication_id: Mapped[Optional[int]] = mapped_column(ForeignKey("publications.id", ondelete="SET NULL"))
    dimension: Mapped[str] = mapped_column(String(48))
    result: Mapped[str] = mapped_column(String(16), doc="same | different | not_stated")
    evidence_quote: Mapped[Optional[str]] = mapped_column(Text)
    quote_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    method: Mapped[str] = mapped_column(String(32), doc="rule | llm:<model>")

    assessment: Mapped[NoveltyAssessment] = relationship(back_populates="comparisons")


# ------------------------------------------------------------------- plans and decisions
class ExperimentPlanRecord(Base):
    __tablename__ = "experiment_plans"

    id: Mapped[int] = mapped_column(primary_key=True)
    gap_id: Mapped[Optional[int]] = mapped_column(ForeignKey("candidate_gaps.id", ondelete="SET NULL"))
    revision: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(16), default="draft", doc="draft | approved | rejected")
    feasibility_class: Mapped[Optional[str]] = mapped_column(String(32), doc="simulatable | needs_hardware | needs_physical_validation")
    plan: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    generator: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Approval(Base):
    """Human decisions on gaps and plans (the system never approves on its own)."""
    __tablename__ = "approvals"

    id: Mapped[int] = mapped_column(primary_key=True)
    object_type: Mapped[str] = mapped_column(String(32), doc="gap | plan")
    object_id: Mapped[int] = mapped_column(Integer)
    decision: Mapped[str] = mapped_column(String(16), doc="approved | rejected | changes_requested")
    reviewer: Mapped[Optional[str]] = mapped_column(String(128))
    note: Mapped[Optional[str]] = mapped_column(Text)
    decided_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class VerificationRun(Base):
    __tablename__ = "verification_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    plan_id: Mapped[Optional[int]] = mapped_column(ForeignKey("experiment_plans.id", ondelete="SET NULL"))
    verdict: Mapped[str] = mapped_column(String(16), doc="pass | fail | hold")
    simulator: Mapped[str] = mapped_column(String(64), doc="e.g. qiskit 2.6.1 statevector")
    report: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    ran_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ReportRecord(Base):
    __tablename__ = "reports"

    id: Mapped[int] = mapped_column(primary_key=True)
    kind: Mapped[str] = mapped_column(String(32), doc="research_investigation | verification")
    formats: Mapped[List[str]] = mapped_column(JSON, default=list)
    path: Mapped[Optional[str]] = mapped_column(Text)
    payload: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


# ----------------------------------------------------------------------- research runs
class ResearchRunRecord(Base):
    """
    Index of orchestrated research runs. The run's full state lives in the LangGraph SQLite checkpoint
    (keyed by thread_id); this row is what lists, statuses and failures are read from.
    """
    __tablename__ = "research_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    thread_id: Mapped[str] = mapped_column(String(64), unique=True)
    params: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(16), default="running", doc="running | waiting | done | failed")
    current: Mapped[Optional[str]] = mapped_column(String(64), doc="Node that runs next or is waiting.")
    waiting: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, doc="Interrupt payload while waiting for the researcher.")
    error: Mapped[Optional[str]] = mapped_column(Text)
    failures: Mapped[List[Dict[str, Any]]] = mapped_column(JSON, default=list, doc="Failed node attempts (node, error, at).")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


# ----------------------------------------------------------------------------- audit log
class LLMCall(Base):
    """Every LLM call: which model, live or not, success, latency. No prompts or secrets stored."""
    __tablename__ = "llm_calls"

    id: Mapped[int] = mapped_column(primary_key=True)
    purpose: Mapped[str] = mapped_column(String(64))
    provider: Mapped[str] = mapped_column(String(32))
    model: Mapped[str] = mapped_column(String(128))
    live: Mapped[bool] = mapped_column(Boolean)
    success: Mapped[bool] = mapped_column(Boolean)
    error: Mapped[Optional[str]] = mapped_column(Text)
    latency_ms: Mapped[Optional[float]] = mapped_column(Float)
    prompt_sha256: Mapped[Optional[str]] = mapped_column(String(64), doc="Hash only, for reproducibility checks.")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
