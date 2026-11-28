"""Module 2 / 4A-B: provenance-carrying paper analysis and cross-paper comparison."""

from .cross_paper_analysis import CorpusAnalysis, analyse_corpus
from .paper_analyzer import PaperAnalyzer, load_analysis, normalize_for_match, verify_excerpt
from .research_schema import FIELD_LABELS, STATEMENT_FIELDS, PaperAnalysisRecord, Provenance, ResearchDimensions, Statement

__all__ = ["analyse_corpus", "CorpusAnalysis", "PaperAnalyzer", "load_analysis", "verify_excerpt", "normalize_for_match", "PaperAnalysisRecord",
           "Statement", "Provenance", "ResearchDimensions", "FIELD_LABELS", "STATEMENT_FIELDS"]
