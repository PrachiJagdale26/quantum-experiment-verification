"""Crossref REST API: bibliographic search and authoritative DOI metadata."""

import os
from typing import Optional

from ..metadata import clean_abstract, normalize_doi, parse_year
from .base import BaseCollector, PublicationRecord, SearchResult

API = "https://api.crossref.org"
SELECT = "DOI,title,author,issued,published,container-title,abstract,URL,link,type,license"
SCHOLARLY_TYPES = ("journal-article", "proceedings-article", "posted-content", "book-chapter")


class CrossrefCollector(BaseCollector):
    name = "crossref"
    min_interval = 1.0

    def _params(self, **params):
        p = {k: v for k, v in params.items() if v is not None}
        if os.getenv("CROSSREF_MAILTO"):
            p["mailto"] = os.getenv("CROSSREF_MAILTO")   # polite pool; chosen by the user, never inferred
        public = {k: v for k, v in p.items() if k != "mailto"}
        return p, public

    @staticmethod
    def parse(item: dict) -> Optional[PublicationRecord]:
        titles = item.get("title") or []
        if not titles or not item.get("DOI"):
            return None
        date_parts = ((item.get("published") or item.get("issued") or {}).get("date-parts") or [[None]])[0]
        date = "-".join(f"{p:02d}" if i else str(p) for i, p in enumerate(date_parts) if p) or None
        pdf = next((l.get("URL") for l in item.get("link") or [] if l.get("content-type") == "application/pdf"), None)
        return PublicationRecord(
            source="crossref", title=titles[0], doi=normalize_doi(item["DOI"]),
            year=parse_year(date_parts[0] if date_parts else None), publication_date=date,
            venue=(item.get("container-title") or [None])[0],
            authors=[" ".join(filter(None, [a.get("given"), a.get("family")])) for a in item.get("author") or []
                     if a.get("family") or a.get("given")],
            abstract=clean_abstract(item.get("abstract")),
            source_url=item.get("URL") or f"https://doi.org/{item['DOI']}",
            pdf_url=pdf,
            is_open_access=False,   # Crossref does not state OA status reliably; OpenAlex/arXiv do
            raw={"type": item.get("type"), "licenses": [l.get("URL") for l in item.get("license") or []]},
        )

    def search(self, query: str, limit: int = 25, year_from: Optional[int] = None,
               year_to: Optional[int] = None) -> SearchResult:
        def run(q, **_):
            filters = [f for f in (f"from-pub-date:{year_from}" if year_from else None,
                                   f"until-pub-date:{year_to}" if year_to else None) if f]
            # Keyword search otherwise returns peer-review records, theses and components.
            filters += [f"type:{t}" for t in SCHOLARLY_TYPES]
            full, public = self._params(query=q, rows=min(limit, 100), select=SELECT,
                                        filter=",".join(filters) or None)
            data, cached = self.get(f"{API}/works", full, cache_params=public)
            msg = (data or {}).get("message") or {}
            return [r for r in (self.parse(i) for i in msg.get("items", [])) if r], msg.get("total-results"), cached, f"{API}/works"
        return self.run_search(query, run)

    def by_doi(self, doi: str) -> Optional[PublicationRecord]:
        d = normalize_doi(doi)
        if not d:
            return None
        full, public = self._params()
        data, _ = self.get(f"{API}/works/{d}", full, cache_params=public)
        return self.parse((data or {}).get("message") or {}) if data else None
