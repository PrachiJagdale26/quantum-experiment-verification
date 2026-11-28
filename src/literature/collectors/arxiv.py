"""arXiv Atom API (https only; arXiv asks for at least 3 s between requests)."""

import re
import xml.etree.ElementTree as ET
from typing import List, Optional

from ..metadata import normalize_doi, parse_year, split_arxiv_id
from .base import BaseCollector, PublicationRecord, SearchResult

API = "https://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom",
      "os": "http://a9.com/-/spec/opensearch/1.1/"}


def _text(el, path) -> Optional[str]:
    node = el.find(path, NS)
    return re.sub(r"\s+", " ", node.text).strip() if node is not None and node.text else None


class ArxivCollector(BaseCollector):
    name = "arxiv"
    min_interval = 3.1

    @staticmethod
    def build_query(query: str, year_from: Optional[int] = None, year_to: Optional[int] = None) -> str:
        """Every word must appear (AND), quoted phrases kept together; optional submission-date range."""
        phrases = re.findall(r'"([^"]+)"', query)
        rest = re.sub(r'"[^"]+"', " ", query)
        terms = [f'all:"{p}"' for p in phrases] + [f"all:{w}" for w in re.findall(r"[A-Za-z0-9\-]+", rest) if len(w) > 1]
        q = " AND ".join(terms) or f"all:{query}"
        if year_from or year_to:
            q += f" AND submittedDate:[{year_from or 1991}01010000 TO {year_to or 2100}12312359]"
        return q

    @staticmethod
    def parse_feed(xml_text: str) -> tuple:
        root = ET.fromstring(xml_text)
        total = _text(root, "os:totalResults")
        records: List[PublicationRecord] = []
        for e in root.findall("a:entry", NS):
            title = _text(e, "a:title")
            abs_url = _text(e, "a:id")
            if not title or not abs_url or title.lower() == "error":
                continue
            arxiv_id, version = split_arxiv_id(abs_url)
            published, updated = _text(e, "a:published"), _text(e, "a:updated")
            pdf = next((l.get("href") for l in e.findall("a:link", NS) if l.get("title") == "pdf"), None)
            versions = [{"label": "v1", "date": (published or "")[:10]}]
            if version and version != "v1":
                versions.append({"label": version, "date": (updated or "")[:10]})
            records.append(PublicationRecord(
                source="arxiv", title=title, doi=normalize_doi(_text(e, "arxiv:doi")),
                arxiv_id=arxiv_id, arxiv_version=version,
                year=parse_year(published), publication_date=(published or "")[:10] or None,
                venue=_text(e, "arxiv:journal_ref") or "arXiv preprint",
                authors=[_text(a, "a:name") for a in e.findall("a:author", NS) if _text(a, "a:name")],
                abstract=_text(e, "a:summary"),
                source_url=f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else abs_url,
                pdf_url=pdf.replace("http://", "https://") if pdf else None,
                is_open_access=True,
                versions=versions,
                raw={"categories": [c.get("term") for c in e.findall("a:category", NS)],
                     "primary_category": (e.find("arxiv:primary_category", NS).get("term")
                                          if e.find("arxiv:primary_category", NS) is not None else None)},
            ))
        return records, int(total) if total and total.isdigit() else None

    def search(self, query: str, limit: int = 25, year_from: Optional[int] = None,
               year_to: Optional[int] = None) -> SearchResult:
        def run(q, **_):
            params = {"search_query": self.build_query(q, year_from, year_to), "start": 0,
                      "max_results": min(limit, 100), "sortBy": "relevance"}
            xml_text, cached = self.get(API, params, as_json=False)
            records, total = self.parse_feed(xml_text or "<feed xmlns='http://www.w3.org/2005/Atom'/>")
            return records, total, cached, API
        return self.run_search(query, run)

    def by_id(self, arxiv_id: str) -> Optional[PublicationRecord]:
        base, _ = split_arxiv_id(arxiv_id)
        if not base:
            return None
        xml_text, _ = self.get(API, {"id_list": base}, as_json=False)
        records, _ = self.parse_feed(xml_text or "<feed xmlns='http://www.w3.org/2005/Atom'/>")
        return records[0] if records else None
