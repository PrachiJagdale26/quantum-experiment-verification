"""Persistent research database (publications, searches, gaps, assessments, plans, approvals, audit log)."""

from .db import database_url, get_engine, session_scope
from .models import (Approval, Base, CandidateGap, DimensionComparison, ExperimentPlanRecord, ExtractionRecord,
                     GapEvidence, LLMCall, NoveltyAssessment, Publication, PublicationVersion, ReportRecord,
                     ResearchRunRecord, SearchHit, SearchRun, VerificationRun)

__all__ = [
    "database_url", "get_engine", "session_scope", "Base", "Publication", "PublicationVersion", "ExtractionRecord",
    "SearchRun", "SearchHit", "CandidateGap", "GapEvidence", "NoveltyAssessment", "DimensionComparison",
    "ExperimentPlanRecord", "Approval", "VerificationRun", "ReportRecord", "LLMCall", "ResearchRunRecord",
]
