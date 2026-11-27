"""
Semantic Scholar Graph API (optional source). Keyless requests are frequently rate-limited (HTTP 429);
a rate-limited run is recorded as such, never treated as "no results".
"""

import os
from typing import Optional

from ..metadata import normalize_doi, parse_year, split_arxiv_id
from .base import BaseCollector, PublicationRecord, SearchResult

API = "https://api.semanticscholar.org/graph/v1"
FIELDS = "title,year,publicationDate,externalIds,abstract,authors,venue,url,openAccessPdf,citationCount"


class SemanticScholarCollector(BaseCollector):
    name = "semantic_scholar"
    min_interval = 1.1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        if key:
            self.client.headers["x-api-key"] = key

    @property
    def has_key(self) -> bool:
        return bool(os.getenv("SEMANTIC_SCHOLAR_API_KEY"))

    @staticmethod
    def parse(p: dict) -> Optional[PublicationRecord]:
        if not p or not p.get("title"):
            return None
        ext = p.get("externalIds") or {}
        arxiv_id, arxiv_v = split_arxiv_id(ext.get("ArXiv"))
        return PublicationRecord(
            source="semantic_scholar", title=p["title"], doi=normalize_doi(ext.get("DOI")),
            arxiv_id=arxiv_id, arxiv_version=arxiv_v, s2_id=p.get("paperId"),
            year=parse_year(p.get("year")), publication_date=p.get("publicationDate"), venue=p.get("venue") or None,
            authors=[a.get("name") for a in p.get("authors") or [] if a.get("name")],
            abstract=p.get("abstract"), source_url=p.get("url"),
            pdf_url=(p.get("openAccessPdf") or {}).get("url"),
            is_open_access=bool(p.get("openAccessPdf")), cited_by_count=p.get("citationCount"),
        )

    def search(self, query: str, limit: int = 25, year_from: Optional[int] = None,
               year_to: Optional[int] = None) -> SearchResult:
        def run(q, **_):
            year = f"{year_from or ''}-{year_to or ''}" if (year_from or year_to) else None
            params = {k: v for k, v in {"query": q, "limit": min(limit, 100), "fields": FIELDS, "year": year}.items() if v}
            data, cached = self.get(f"{API}/paper/search", params)
            return ([r for r in (self.parse(p) for p in (data or {}).get("data", [])) if r],
                    (data or {}).get("total"), cached, f"{API}/paper/search")
        return self.run_search(query, run)
