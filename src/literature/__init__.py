"""Module 1: scholarly literature collection (OpenAlex, arXiv, Crossref, Semantic Scholar)."""

from .collectors import COLLECTORS, PublicationRecord, SearchResult
from .ingestion import IngestReport, LiteratureIngestor, list_publications, recent_search_runs
from .topics import TOPICS

__all__ = ["COLLECTORS", "PublicationRecord", "SearchResult", "IngestReport", "LiteratureIngestor",
           "list_publications", "recent_search_runs", "TOPICS"]
