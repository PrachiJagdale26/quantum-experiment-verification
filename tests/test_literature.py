"""
Module 1 (literature collection), offline. Parsers run on real API responses captured on 2026-11-27
(tests/fixtures/literature); deduplication and ingestion run against a temporary database.
"""

import json
import os

import pytest
from sqlalchemy import func, select

from src.literature.collectors import ArxivCollector, CrossrefCollector, OpenAlexCollector, PublicationRecord, SearchResult
from src.literature.collectors.base import RateLimitedError
from src.literature.deduplication import titles_match
from src.literature.ingestion import LiteratureIngestor, list_publications
from src.literature.metadata import (normalize_doi, normalize_title, parse_year, reconstruct_inverted_abstract,
                                     split_arxiv_id)
from src.storage import Publication, SearchRun, session_scope

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "literature")


# --------------------------------------------------------------------- metadata
@pytest.mark.parametrize("raw,expected", [
    ("https://doi.org/10.1038/S41586-023-06505-7", "10.1038/s41586-023-06505-7"),
    ("doi:10.22331/q-2021-11-26-592.", "10.22331/q-2021-11-26-592"),
    ("not a doi", None), (None, None),
])
def test_normalize_doi(raw, expected):
    assert normalize_doi(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("http://arxiv.org/abs/2110.05389v3", ("2110.05389", "v3")),
    ("arXiv:2303.04792", ("2303.04792", None)),
    ("https://arxiv.org/pdf/quant-ph/0101012v2", ("quant-ph/0101012", "v2")),
    ("https://example.org/paper", (None, None)),
])
def test_split_arxiv_id(raw, expected):
    assert split_arxiv_id(raw) == expected


def test_title_normalisation_and_abstract_reconstruction():
    assert normalize_title("Quantum <i>error</i> mitigation: $\\epsilon$-limits") == "quantum error mitigation limits"
    assert reconstruct_inverted_abstract({"noise": [1], "Quantum": [0], "matters": [2]}) == "Quantum noise matters"
    assert parse_year("2023-09-22") == 2023 and parse_year(None) is None


# ------------------------------------------------------- parsers on real responses
def test_openalex_parser_on_real_response():
    data = json.load(open(os.path.join(FIX, "openalex_search.json"), encoding="utf-8"))
    recs = [OpenAlexCollector.parse(w) for w in data["results"]]
    assert len(recs) == 5 and all(r.title and r.source == "openalex" for r in recs)
    assert any(r.arxiv_id for r in recs) and any(r.abstract for r in recs)
    assert all(r.doi is None or r.doi.startswith("10.") for r in recs)
    assert all(r.openalex_id.startswith("W") for r in recs)


def test_arxiv_parser_on_real_feed():
    recs, total = ArxivCollector.parse_feed(open(os.path.join(FIX, "arxiv_search.xml"), encoding="utf-8").read())
    assert len(recs) == 5 and total and total > 5
    for r in recs:
        assert r.arxiv_id and r.abstract and r.is_open_access
        assert r.source_url == f"https://arxiv.org/abs/{r.arxiv_id}"
        assert r.versions[0]["label"] == "v1"


def test_crossref_parser_on_real_response():
    data = json.load(open(os.path.join(FIX, "crossref_search.json"), encoding="utf-8"))
    recs = [r for r in (CrossrefCollector.parse(i) for i in data["message"]["items"]) if r]
    assert recs and all(r.doi and r.title for r in recs)
    assert all(r.raw["type"] in ("journal-article", "proceedings-article", "posted-content", "book-chapter") for r in recs)
    assert all(r.abstract is None or "<jats:" not in r.abstract for r in recs)


def test_arxiv_query_builder():
    q = ArxivCollector.build_query('"error mitigation" superconducting', 2020, 2023)
    assert 'all:"error mitigation"' in q and "all:superconducting" in q
    assert "submittedDate:[202001010000 TO 202312312359]" in q


# ---------------------------------------------------------------- deduplication
def test_fuzzy_title_match_requires_same_first_author_and_close_year():
    t = "Error mitigation with Clifford quantum-circuit data"
    assert titles_match(t, 2021, ["Piotr Czarnik"], "Error Mitigation with Clifford Quantum-Circuit Data.", 2020, ["P. Czarnik"])
    assert not titles_match(t, 2021, ["Piotr Czarnik"], t, 2021, ["Someone Else"])
    assert not titles_match(t, 2021, [], t, 2015, [])
    assert not titles_match(t, 2021, [], "Error mitigation for variational algorithms", 2021, [])


def _rec(source, **kw):
    base = dict(title="Error mitigation with Clifford quantum-circuit data", year=2021, authors=["Piotr Czarnik"])
    base.update(kw)
    return PublicationRecord(source=source, **base)


class FakeCollector:
    def __init__(self, name, result):
        self.name, self.result = name, result

    def search(self, query, **kw):
        return self.result if isinstance(self.result, SearchResult) else SearchResult(self.name, query, "ok", self.result, len(self.result))


@pytest.fixture
def db(tmp_path):
    return f"sqlite:///{tmp_path / 'lit.db'}"


def test_preprint_and_journal_versions_merge_into_one_publication(db):
    arxiv = _rec("arxiv", arxiv_id="2005.10189", arxiv_version="v3", abstract="We propose...", is_open_access=True,
                 versions=[{"label": "v1", "date": "2020-05-20"}, {"label": "v3", "date": "2021-11-16"}],
                 venue="arXiv preprint", source_url="https://arxiv.org/abs/2005.10189")
    journal = _rec("crossref", doi="10.22331/q-2021-11-26-592", venue="Quantum", publication_date="2021-11-26")
    oa = _rec("openalex", doi="10.22331/q-2021-11-26-592", arxiv_id="2005.10189", openalex_id="W3108",
              referenced_works=["W1", "W2"])
    ing = LiteratureIngestor(db, collectors={"arxiv": FakeCollector("arxiv", [arxiv]),
                                             "crossref": FakeCollector("crossref", [journal]),
                                             "openalex": FakeCollector("openalex", [oa])})
    rep = ing.collect("clifford data", sources=("arxiv", "crossref", "openalex"))
    assert rep.new == 1 and rep.merged == 2 and len(set(rep.publication_ids)) == 1
    with session_scope(db) as s:
        p = s.scalars(select(Publication)).one()
        assert (p.doi, p.arxiv_id, p.openalex_id, p.venue) == ("10.22331/q-2021-11-26-592", "2005.10189", "W3108", "Quantum")
        assert p.abstract == "We propose..." and p.abstract_source == "arxiv"
        assert sorted(p.raw_metadata["sources"]) == ["arxiv", "crossref", "openalex"]
        assert p.raw_metadata["referenced_works"] == ["W1", "W2"]
        assert {v.version_label for v in p.versions} == {"arXiv v1", "arXiv v3", "published version"}


def test_arxiv_datacite_doi_is_replaced_by_journal_doi(db):
    pre = _rec("openalex", doi="10.48550/arxiv.2005.10189", arxiv_id="2005.10189")
    pub = _rec("crossref", doi="10.22331/q-2021-11-26-592")
    ing = LiteratureIngestor(db, collectors={"openalex": FakeCollector("openalex", [pre]),
                                             "crossref": FakeCollector("crossref", [pub])})
    ing.collect("q", sources=("openalex", "crossref"))
    with session_scope(db) as s:
        p = s.scalars(select(Publication)).one()
        assert p.doi == "10.22331/q-2021-11-26-592" and p.raw_metadata["arxiv_doi"] == "10.48550/arxiv.2005.10189"


def test_failures_and_rate_limits_are_recorded_not_hidden(db):
    ing = LiteratureIngestor(db, collectors={
        "semantic_scholar": FakeCollector("semantic_scholar", SearchResult("semantic_scholar", "q", "rate_limited",
                                                                           error="HTTP 429")),
        "arxiv": FakeCollector("arxiv", SearchResult("arxiv", "q", "failed", error="timeout"))})
    rep = ing.collect("q", sources=("semantic_scholar", "arxiv"))
    assert [o.status for o in rep.outcomes] == ["rate_limited", "failed"]
    with session_scope(db) as s:
        runs = s.scalars(select(SearchRun)).all()
        assert {(r.source, r.status) for r in runs} == {("semantic_scholar", "rate_limited"), ("arxiv", "failed")}
        assert s.scalar(select(func.count(Publication.id))) == 0


def test_metadata_only_records_are_kept_and_filterable(db):
    ing = LiteratureIngestor(db, collectors={"crossref": FakeCollector("crossref", [
        _rec("crossref", doi="10.1/a", title="A paper without abstract", authors=[]),
        _rec("crossref", doi="10.1/b", title="A paper with abstract", authors=[], abstract="text")])})
    ing.collect("q", sources=("crossref",))
    assert [p.title for p in list_publications(db, availability="metadata_only")] == ["A paper without abstract"]
    assert [p.title for p in list_publications(db, availability="abstract")] == ["A paper with abstract"]


def test_rate_limit_from_http_layer_becomes_status(monkeypatch):
    c = ArxivCollector(use_cache=False)

    def boom(*a, **k):
        raise RateLimitedError("arxiv rate limit (HTTP 429)")
    monkeypatch.setattr(c, "get", boom)
    res = c.search("anything")
    assert res.status == "rate_limited" and res.records == []


# ------------------------------------------------------------------ full text
class _Resp:
    def __init__(self, status, content):
        self.status_code, self.content = status, content


class _FakeHTTP:
    def __init__(self, content, status=200):
        self.content, self.status, self.urls = content, status, []

    def get(self, url, **kw):
        self.urls.append(url)
        return _Resp(self.status, self.content)


def _tiny_pdf() -> bytes:
    import io
    from reportlab.pdfgen import canvas
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.drawString(72, 720, "Measurement-induced phases on superconducting qubits. Our study is limited to 70 qubits.")
    c.save()
    return buf.getvalue()


def test_full_text_only_from_arxiv_and_flags_availability(db, tmp_path, monkeypatch):
    import src.literature.ingestion as ing_mod
    monkeypatch.setattr(ing_mod, "FULLTEXT_DIR", str(tmp_path / "ft"))
    ing = LiteratureIngestor(db, collectors={"crossref": FakeCollector("crossref", [_rec("crossref", doi="10.1/x", title="Paywalled", authors=[])]),
                                             "arxiv": ArxivCollector(client=_FakeHTTP(_tiny_pdf()), use_cache=False)})
    ing.collector("arxiv").min_interval = 0
    ing.collect("q", sources=("crossref",))
    with session_scope(db) as s:
        paywalled_id = s.scalars(select(Publication)).one().id
    refused = ing.fetch_full_text(paywalled_id)
    assert not refused["ok"] and "arXiv" in refused["reason"]          # no attempt to bypass a paywall

    ing._collectors["openalex"] = FakeCollector("openalex", [_rec("openalex", arxiv_id="2303.04792", title="Measurement-induced entanglement",
                                                                  authors=["Jesse Hoke"], abstract="abs")])
    ing.collect("q", sources=("openalex",))
    with session_scope(db) as s:
        pid = s.scalars(select(Publication).where(Publication.arxiv_id == "2303.04792")).one().id
    res = ing.fetch_full_text(pid)
    assert res["ok"] and res["pages"] == 1
    assert ing.collector("arxiv").client.urls == ["https://arxiv.org/pdf/2303.04792"]
    with session_scope(db) as s:
        p = s.get(Publication, pid)
        assert p.full_text_available and p.raw_metadata["fulltext_pages"] == 1


def test_publication_view_states_availability_and_type(db):
    from src.ui.overlap_view import build_publication_view
    ing = LiteratureIngestor(db, collectors={"openalex": FakeCollector("openalex", [
        _rec("openalex", doi="10.5281/zenodo.123", title="A repository deposit", authors=["A B"], raw={"type": "preprint"}),
        _rec("openalex", doi="10.1103/x", title="A journal paper", authors=["C D"], abstract="abs", raw={"type": "article"})])})
    ing.collect("q", sources=("openalex",))
    views = {v["title"]: v for v in (build_publication_view(p) for p in list_publications(db))}
    assert views["A repository deposit"]["availability"] == "metadata_only" and views["A repository deposit"]["preprint_only"]
    assert views["A journal paper"]["availability"] == "abstract" and not views["A journal paper"]["preprint_only"]
    assert views["A journal paper"]["work_types"] == ["article"]
