"""
Phase 4: gap generation, multi-stage novelty investigation, ranking and the known-work benchmark (offline).
Collectors are fakes returning fixed records, so the rules are tested without the network.
"""

import json

import numpy as np
import pytest

from src.evaluation.novelty_benchmark import run_known_work
from src.gap_discovery import (CoveragePolicy, GapGenerator, NoveltyChecker, load_assessment, load_gaps, save_gaps,
                               score_gap, weighted_total)
from src.gap_discovery.novelty_checker import compare_dimension
from src.literature.collectors.base import PublicationRecord, SearchResult
from src.literature.ingestion import LiteratureIngestor, list_publications
from src.research_analysis import PaperAnalyzer, normalize_for_match
from src.research_analysis.cross_paper_analysis import load_corpus
from src.storage import CandidateGap, Publication, session_scope


def bow_embed(texts):
    vocab = sorted({w for t in texts for w in normalize_for_match(t).split()} | {"_"})
    m = np.array([[normalize_for_match(t).split().count(w) for w in vocab] + [1.0] for t in texts], dtype=float)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


class StableEmbed:
    """Bag of words over a fixed vocabulary, so vectors computed in different calls are comparable."""
    def __init__(self):
        self.vocab = {}

    def __call__(self, texts):
        rows = []
        for t in texts:
            v = {}
            for w in normalize_for_match(t).split():
                j = self.vocab.setdefault(w, len(self.vocab))
                v[j] = v.get(j, 0) + 1
            rows.append(v)
        m = np.zeros((len(texts), 4096))
        for i, v in enumerate(rows):
            for j, c in v.items():
                m[i, j % 4096] += c
        return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)


class FakeLLM:
    demo_mode = False

    def __init__(self, items):
        self.items = items
        self.last_call = {"provider": "fake", "model": "fake-1", "live": True}

    def complete(self, system, user, **kw):
        if "<UNTRUSTED_PAPER_TEXT>" in user and "items" not in system:
            return json.dumps({"items": self.items})
        return json.dumps({"items": self.items})


class FakeCollector:
    def __init__(self, name, records=(), status="ok"):
        self.name, self.records, self.status, self.queries = name, list(records), status, []

    def search(self, query, limit=10, **kw):
        self.queries.append(query)
        if self.status != "ok":
            return SearchResult(self.name, query, self.status, error="simulated outage")
        return SearchResult(self.name, query, "ok", records=list(self.records)[:limit], total_available=len(self.records))

    def citing(self, oid, limit=10):
        return SearchResult(self.name, oid, "ok", records=[])


def rec(source, title, abstract, doi=None, year=2024):
    return PublicationRecord(source=source, title=title, abstract=abstract, doi=doi, year=year)


@pytest.fixture
def db(tmp_path):
    return f"sqlite:///{tmp_path / 'g.db'}"


def _pub(db, title, abstract, **kw):
    with session_scope(db) as s:
        p = Publication(title=title, normalized_title=title.lower(), source="openalex", abstract=abstract, **kw)
        s.add(p)
        s.flush()
        return p.id


LIM = "Our study is limited to shallow circuits with few qubits because dephasing destroys entanglement."


@pytest.fixture
def corpus(db):
    a = _pub(db, "Error mitigation on superconducting qubits",
             "Here we apply error mitigation to entanglement on superconducting qubits. " + LIM)
    b = _pub(db, "Error mitigation on trapped ions",
             "Here we apply error mitigation to entanglement on trapped ions. " + LIM)
    c = _pub(db, "Error correction on superconducting qubits",
             "Here we build error correction with a surface code on superconducting qubits.")
    d = _pub(db, "Error correction on trapped ions", "Here we test error correction with a surface code on trapped ions.")
    e = _pub(db, "Dynamical decoupling on superconducting qubits",
             "Here we use dynamical decoupling on superconducting qubits. Extending this to neutral atoms is left for future work.")
    an = PaperAnalyzer(FakeLLM([{"field": "limitation", "text": "Invented limitation.", "evidence": "not in any paper"}]), db)
    for pid in (a, b, c, d, e):
        an.analyse(pid)
    return {"a": a, "b": b, "c": c, "d": d, "e": e}


# ------------------------------------------------------------------ generation
def test_generator_merges_statements_skips_unverified_and_finds_combinations(db, corpus):
    gaps = GapGenerator(db, embed=bow_embed).generate()
    lim = [g for g in gaps if g.origin == "stated_limitation"]
    assert len(lim) == 1 and {e.publication_id for e in lim[0].evidence} == {corpus["a"], corpus["b"]}
    assert not any("Invented" in g.statement for g in gaps)
    assert lim[0].dimensions["noise"].source == "statement" and "decoherence" in lim[0].dimensions["noise"].values
    fw = [g for g in gaps if g.origin == "future_work"]
    assert fw and fw[0].dimensions["platform"].values == ["neutral atoms"]
    combos = {(g.dimensions["method"].values[0], g.dimensions.get("platform", g.dimensions.get("noise")).values[0])
              for g in gaps if g.origin == "unexplored_combination"}
    assert combos == {("quantum error correction", "decoherence")}   # the only pair both sides of which 2+ papers study
    assert all(e.kind == "inference" for g in gaps if g.origin == "unexplored_combination" for e in g.evidence)


def test_save_gaps_is_idempotent(db, corpus):
    gen = GapGenerator(db, embed=bow_embed)
    first = save_gaps(gen.generate(), db)
    second = save_gaps(gen.generate(), db)
    assert [g.id for g in first] == [g.id for g in second]
    with session_scope(db) as s:
        assert s.query(CandidateGap).count() == len(first)
    assert {g.id for g in load_gaps(db)} == {g.id for g in first}


# --------------------------------------------------------------- novelty rules
def test_compare_dimension_rules():
    text = "We run zero-noise extrapolation on a trapped-ion processor with 12 qubits. Dephasing limits depth."
    assert compare_dimension("method", ["quantum error mitigation"], text)[0] == "same"
    r, quote = compare_dimension("platform", ["superconducting qubits"], text)
    assert r == "different" and "trapped-ion" in quote
    assert compare_dimension("phenomenon", ["entanglement"], text) == ("not_stated", None)
    assert compare_dimension("qubit_scale", ["10"], text)[0] == "same"
    assert compare_dimension("qubit_scale", ["100"], text)[0] == "different"


def _checker(db, collectors, **kw):
    kw.setdefault("embed", StableEmbed())
    return NoveltyChecker(db, ingestor=LiteratureIngestor(db, collectors=collectors), citations=False, **kw)


def _combo_gap(db, corpus, method="quantum error correction", col="decoherence"):
    gaps = save_gaps(GapGenerator(db, embed=bow_embed).generate(), db)
    return next(g for g in gaps if g.origin == "unexplored_combination" and g.dimensions["method"].values == [method]
                and col in str(g.dimensions))


def test_published_combination_is_already_investigated(db, corpus):
    gap = _combo_gap(db, corpus)
    hit = rec("openalex", "Surface code error correction under dephasing",
              "We study quantum error correction with a surface code under dephasing noise.", doi="10.1/qec-deph")
    noise = [rec("openalex", f"Unrelated optics paper {i}", f"A study of lasers number {i}.", doi=f"10.1/x{i}") for i in range(20)]
    cols = {"openalex": FakeCollector("openalex", [hit] + noise), "arxiv": FakeCollector("arxiv", noise[:5]),
            "crossref": FakeCollector("crossref", noise[5:])}
    a = _checker(db, cols, policy=CoveragePolicy(already_similarity=0.3)).check(gap.id)
    assert a["classification"] == "already_investigated"
    top = a["findings"]["closest"][0]
    assert top["title"] == "Surface code error correction under dephasing" and top["dimensions"]["noise"]["result"] == "same"
    assert "formula" in a["confidence_basis"] and 0 <= a["confidence"] <= 1
    kinds = {q["kind"] for q in a["coverage"]["queries"]}
    assert "concepts" in kinds and "exact" not in kinds      # a combination's statement is our template, never a query
    # long keyword queries are not sent to arXiv (it requires every word to match)
    assert all(len(q.split()) <= 8 for q in cols["arxiv"].queries)
    # works found during the check stay out of the researcher's corpus
    assert "Surface code error correction under dephasing" not in {p.title for p in list_publications(db)}
    assert len(load_corpus(db)[0]) == 5 and load_corpus(db)[2] == 0


def test_failed_sources_give_insufficient_coverage_never_novel(db, corpus):
    gap = _combo_gap(db, corpus)
    cols = {n: FakeCollector(n, status="failed") for n in ("openalex", "arxiv", "crossref")}
    a = _checker(db, cols).check(gap.id)
    assert a["classification"] == "insufficient_coverage"
    assert a["coverage"]["sources_failed"] == ["arxiv", "crossref", "openalex"]
    assert "failed" in a["uncertainty"]


def test_no_match_with_good_coverage_is_potentially_unexplored(db, corpus):
    gap = _combo_gap(db, corpus)
    noise = [rec("openalex", f"Unrelated optics paper {i}", f"A study of lasers number {i}.", doi=f"10.1/y{i}") for i in range(20)]
    cols = {"openalex": FakeCollector("openalex", noise), "arxiv": FakeCollector("arxiv", noise[:3]),
            "crossref": FakeCollector("crossref", noise[3:])}
    a = _checker(db, cols).check(gap.id)
    assert a["classification"] == "potentially_unexplored"
    assert a["label"] == "Potentially unexplored in the searched literature"
    assert "never means the question has never been published" in a["uncertainty"]
    assert load_assessment(gap.id, db)["id"] == a["id"]
    with session_scope(db) as s:
        assert s.get(CandidateGap, gap.id).classification == "potentially_unexplored"


def test_supporting_papers_are_not_counted_as_prior_work(db, corpus):
    gaps = save_gaps(GapGenerator(db, embed=bow_embed).generate(), db)
    lim = next(g for g in gaps if g.origin == "stated_limitation")
    cols = {n: FakeCollector(n) for n in ("openalex", "arxiv", "crossref")}
    a = _checker(db, cols).check(lim.id)
    compared = {c["publication_id"] for c in a["findings"]["closest"]}
    assert not compared & {corpus["a"], corpus["b"]}


def test_unverified_model_judgement_is_discarded(db, corpus):
    gap = _combo_gap(db, corpus)
    hit = rec("openalex", "Surface codes under dephasing", "We run quantum error correction under dephasing.", doi="10.1/qd")

    class Judge:
        demo_mode = False
        last_call = {"provider": "fake", "model": "judge"}

        def complete(self, system, user, **kw):
            return json.dumps({"result": "same", "evidence": "a sentence the abstract does not contain"})

    cols = {"openalex": FakeCollector("openalex", [hit]), "arxiv": FakeCollector("arxiv"), "crossref": FakeCollector("crossref")}
    a = _checker(db, cols, llm=Judge(), policy=CoveragePolicy(already_similarity=0.99, partial_similarity=0.99)).check(gap.id)
    judged = [c["research_question"] for c in a["findings"]["closest"] if c["research_question"]]
    assert judged and not judged[0]["verified"]
    assert a["findings"]["judgements"]["discarded_unverified"] >= 1
    assert a["classification"] != "already_investigated"      # an unverified "same" never decides


def test_core_prefers_the_gaps_own_dimensions():
    from src.gap_discovery import Gap, GapDimension
    own = Gap(key="k", title="t", statement="s", origin="stated_limitation", dimensions={
        "phenomenon": GapDimension(values=["entanglement"], source="statement"),
        "qubit_scale": GapDimension(values=["20"], source="statement"),
        "method": GapDimension(values=["quantum teleportation"], source="paper_context"),
        "platform": GapDimension(values=["superconducting qubits"], source="paper_context")})
    assert own.core() == {"phenomenon": ["entanglement"], "qubit_scale": ["20"]}
    thin = own.model_copy(update={"dimensions": {k: v for k, v in own.dimensions.items() if k != "qubit_scale"}})
    assert thin.core() == {"phenomenon": ["entanglement"], "method": ["quantum teleportation"]}   # topped up to two


# --------------------------------------------------------------------- ranking
def test_ranking_has_eight_dimensions_and_uses_classification(db, corpus):
    gap = _combo_gap(db, corpus)
    s0 = score_gap(gap, None)
    assert len(s0) == 8 and all(0 <= v["score"] <= 1 and v["why"] for v in s0.values())
    s1 = score_gap(gap, {"classification": "already_investigated", "confidence": 0.8,
                         "confidence_basis": {"coverage": 0.9}, "coverage": {"sources_ok": [1, 2], "sources_attempted": [1, 2, 3],
                                                                            "queries": [1, 2, 3], "screened": 30}})
    assert s1["novelty"]["score"] < s0["novelty"]["score"]
    assert weighted_total(s1, {"novelty": 0, "coverage": 10}) > weighted_total(s1, {"novelty": 10, "coverage": 0})


# ------------------------------------------------------------------- benchmark
def test_known_work_benchmark_finds_published_question(db, tmp_path):
    abstract = "Here we demonstrate zero-noise extrapolation for entanglement on superconducting qubits with 5 qubits."
    pid = _pub(db, "ZNE for entanglement", abstract, doi="10.1/zne")
    PaperAnalyzer(FakeLLM([{"field": "objective", "text": "Demonstrate zero-noise extrapolation for entanglement",
                            "evidence": "demonstrate zero-noise extrapolation for entanglement on superconducting qubits"}]), db).analyse(pid)
    same = rec("openalex", "ZNE for entanglement", abstract, doi="10.1/zne")
    noise = [rec("crossref", f"Other topic {i}", f"Lasers {i}.", doi=f"10.1/z{i}") for i in range(20)]
    cols = {"openalex": FakeCollector("openalex", [same]), "arxiv": FakeCollector("arxiv"), "crossref": FakeCollector("crossref", noise)}
    r = run_known_work(db, n=5, checker=_checker(db, cols), out_dir=str(tmp_path))
    m = r["metrics"]
    assert m["items"] == 1 and m["false_novelty_rate"] == 0.0 and m["recall_screened"] == 1.0 and m["mrr_compared"] == 1.0
    assert r["items"][0]["classification"] in ("already_investigated", "partially_investigated")
    assert json.load(open(r["path"], encoding="utf-8"))["limitations"]
    assert load_gaps(db) == []                     # benchmark gaps never mix with real candidates
