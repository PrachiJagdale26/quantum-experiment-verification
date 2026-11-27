"""Scholarly API collectors. Each returns normalised PublicationRecords and never invents metadata."""

from .arxiv import ArxivCollector
from .base import BaseCollector, PublicationRecord, RateLimitedError, SearchResult
from .crossref import CrossrefCollector
from .openalex import OpenAlexCollector
from .semantic_scholar import SemanticScholarCollector

COLLECTORS = {
    "openalex": OpenAlexCollector,
    "arxiv": ArxivCollector,
    "crossref": CrossrefCollector,
    "semantic_scholar": SemanticScholarCollector,
}

__all__ = ["BaseCollector", "PublicationRecord", "SearchResult", "RateLimitedError", "COLLECTORS",
           "OpenAlexCollector", "ArxivCollector", "CrossrefCollector", "SemanticScholarCollector"]
