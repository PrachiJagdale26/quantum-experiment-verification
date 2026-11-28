"""OpenAlex works API: search, DOI lookup, author works, references and citing works."""

import os
from typing import List, Optional, Tuple

from ..metadata import normalize_doi, parse_year, reconstruct_inverted_abstract, split_arxiv_id
from .base import BaseCollector, PublicationRecord, SearchResult

API = "https://api.openalex.org"
SELECT = ("id,doi,display_name,publication_year,publication_date,primary_location,best_oa_location,locations,"
          "open_access,authorships,abstract_inverted_index,referenced_works,cited_by_count,type")


class OpenAlexCollector(BaseCollector):
    name = "openalex"
    min_interval = 0.2   # well under the documented 10 requests/s

    def _params(self, **params):
        """Adds the API key (if configured) to the request, but never to the cache key."""
        public = {k: v for k, v in params.items() if v is not None}
        full = dict(public)
        key = os.getenv("OPENALEX_API_KEY")
        if key:
            full["api_key"] = key
        return full, public

    # ---------------------------------------------------------------- parsing
    @staticmethod
    def parse(work: dict) -> Optional[PublicationRecord]:
        if not work or not work.get("display_name"):
            return None
        arxiv_id = arxiv_v = None
        for loc in work.get("locations") or []:
            arxiv_id, arxiv_v = split_arxiv_id((loc or {}).get("landing_page_url"))
            if arxiv_id:
                break
        primary = work.get("primary_location") or {}
        best_oa = work.get("best_oa_location") or {}
        src = (primary.get("source") or {})
        return PublicationRecord(
            source="openalex",
            title=work["display_name"],
            doi=normalize_doi(work.get("doi")),
            arxiv_id=arxiv_id, arxiv_version=arxiv_v,
            openalex_id=(work.get("id") or "").rsplit("/", 1)[-1] or None,
            year=parse_year(work.get("publication_year")),
            publication_date=work.get("publication_date"),
            venue=src.get("display_name"),
            authors=[(a.get("author") or {}).get("display_name") for a in work.get("authorships") or []
                     if (a.get("author") or {}).get("display_name")],
            abstract=reconstruct_inverted_abstract(work.get("abstract_inverted_index")),
            source_url=primary.get("landing_page_url") or work.get("doi") or work.get("id"),
            pdf_url=best_oa.get("pdf_url"),
            is_open_access=bool((work.get("open_access") or {}).get("is_oa")),
            referenced_works=[r.rsplit("/", 1)[-1] for r in work.get("referenced_works") or []],
            cited_by_count=work.get("cited_by_count"),
            raw={"type": work.get("type"), "openalex_url": work.get("id")},
        )

    def _works(self, params: dict) -> Tuple[List[PublicationRecord], Optional[int], bool, str]:
        full, public = self._params(select=SELECT, **params)
        data, cached = self.get(f"{API}/works", full, cache_params=public)
        records = [r for r in (self.parse(w) for w in (data or {}).get("results", [])) if r]
        return records, ((data or {}).get("meta") or {}).get("count"), cached, f"{API}/works"

    # --------------------------------------------------------------- queries
    def search(self, query: str, limit: int = 25, year_from: Optional[int] = None,
               year_to: Optional[int] = None) -> SearchResult:
        def run(q, **_):
            filters = []
            if year_from:
                filters.append(f"from_publication_date:{year_from}-01-01")
            if year_to:
                filters.append(f"to_publication_date:{year_to}-12-31")
            return self._works({"search": q, "per_page": min(limit, 200), "filter": ",".join(filters) or None})
        return self.run_search(query, run)

    def by_doi(self, doi: str) -> Optional[PublicationRecord]:
        d = normalize_doi(doi)
        if not d:
            return None
        full, public = self._params(select=SELECT)
        data, _ = self.get(f"{API}/works/https://doi.org/{d}", full, cache_params=public)
        return self.parse(data) if data else None

    def by_author(self, author_name: str, limit: int = 25) -> SearchResult:
        def run(q, **_):
            full, public = self._params(search=q, per_page=1)
            authors, _ = self.get(f"{API}/authors", full, cache_params=public)
            top = ((authors or {}).get("results") or [None])[0]
            if not top:
                return [], 0, False, f"{API}/authors"
            return self._works({"filter": f"author.id:{top['id'].rsplit('/', 1)[-1]}", "per_page": min(limit, 200),
                                "sort": "cited_by_count:desc"})
        return self.run_search(author_name, run)

    def by_ids(self, openalex_ids: List[str]) -> SearchResult:
        """Fetch specific works (e.g. a paper's references), up to 50 per request."""
        ids = [i for i in openalex_ids if i][:50]
        return self.run_search(",".join(ids), lambda q, **_: self._works({"filter": "openalex_id:" + "|".join(ids),
                                                                           "per_page": 50}))

    def citing(self, openalex_id: str, limit: int = 25) -> SearchResult:
        """Works that cite the given work (forward citation exploration)."""
        return self.run_search(openalex_id, lambda q, **_: self._works({"filter": f"cites:{q}", "per_page": min(limit, 200),
                                                                         "sort": "cited_by_count:desc"}))
