"""
Phase 3: provenance-carrying paper analysis, cross-paper analysis, hybrid retrieval (offline).
A fake LLM returns a fixed answer that includes a fabricated quote, so verification is tested honestly.
"""

import json

import numpy as np
import pytest
from sqlalchemy import create_engine, inspect, text

from src.knowledge_base.ontology import concepts_by_category, expand_query
from src.research_analysis import PaperAnalyzer, analyse_corpus, load_analysis, normalize_for_match, verify_excerpt
from src.research_analysis.paper_analyzer import qubit_counts, strip_references
from src.storage import Publication, session_scope
from src.storage.db import get_engine

ABSTRACT = ("Here we study measurement-induced phases on up to 70 superconducting qubits. "
            "Our study is limited to shallow circuits because dephasing destroys the long-range Rényi entanglement. "
            "The existence of efficient decoders for arbitrarily large circuits remains an open problem. "
            "Extending the protocol to trapped ions is left for future work.")


class FakeLLM:
    demo_mode = False

    def __init__(self, items):
        self.items, self.calls = items, 0
        self.last_call = {"provider": "fake", "model": "fake-1", "live": True}

    def complete(self, system, user, **kw):
        self.calls += 1
        assert "<UNTRUSTED_PAPER_TEXT>" in user          # injection defence applied
        return "```json\n" + json.dumps({"items": self.items}) + "\n```"


@pytest.fixture
def db(tmp_path):
    return f"sqlite:///{tmp_path / 'ra.db'}"


def _pub(db, title="Measurement-induced phases on a noisy processor", abstract=ABSTRACT, **kw):
    with session_scope(db) as s:
        p = Publication(title=title, normalized_title=title.lower(), source="openalex", abstract=abstract, **kw)
        s.add(p)
        s.flush()
        return p.id


# ----------------------------------------------------------------- verification
def test_verify_excerpt_verbatim_accents_paraphrase_and_page():
    src = normalize_for_match(ABSTRACT)
    assert verify_excerpt("limited to shallow circuits because dephasing destroys", src, [])[0]
    assert verify_excerpt("the long-range Renyi entanglement", src, [])[0]          # accent difference is harmless
    assert not verify_excerpt("the authors could not access trapped-ion hardware", src, [])[0]
    pages = [(1, normalize_for_match("intro text")), (3, normalize_for_match(ABSTRACT))]
    assert verify_excerpt("remains an open problem", src, pages)[2] == 3


def test_strip_references_heading_and_numbered_list():
    body = "Main text. " * 300
    assert strip_references(body + "\nReferences\n1. A. Author, Title (2019).") == body
    refs = "\n".join(f"{i}. Author {i}, Some title. Nat. Phys. {i}, 1 ({2000 + i})." for i in range(1, 9))
    assert strip_references(body + "\nOnline content\nAny methods...\n" + refs).startswith(body[:50])
    assert "Author 3" not in strip_references(body + "\n" + refs)
    early = "\nReferences\n" + body          # a "References" heading early in the text is not the bibliography
    assert strip_references(early) == early


def test_qubit_counts_and_ontology():
    assert qubit_counts(ABSTRACT) == [70]
    tags = concepts_by_category(ABSTRACT)
    assert tags["platform"] == ["superconducting qubits", "trapped ions"]
    assert "decoherence" in tags["noise"]
    variants = dict(expand_query("decoherence in trapped ions"))
    assert variants["decoherence in trapped ions"] == "exact" and "synonym" in variants.values()


# --------------------------------------------------------------------- analyzer
def test_statement_kinds_and_confidence(db):
    pid = _pub(db)
    llm = FakeLLM([
        {"field": "quantum_system", "text": "up to 70 superconducting qubits", "evidence": "up to 70 superconducting qubits"},
        {"field": "limitation", "text": "Dephasing restricts the study to shallow circuits.",
         "evidence": "Our study is limited to shallow circuits because dephasing destroys the long-range Renyi entanglement."},
        {"field": "result", "text": "Fidelity of 99.9% was achieved.", "evidence": "we achieved a fidelity of 99.9%"},  # fabricated
        {"field": "not_a_field", "text": "x", "evidence": "x"},
    ])
    rec = PaperAnalyzer(llm, db).analyse(pid)
    by = {(s.field, s.extractor): s for s in rec.statements}
    assert by[("quantum_system", "fake/fake-1")].kind == "direct"
    lim = by[("limitation", "fake/fake-1")]
    assert lim.kind == "interpretation" and lim.provenance.verified and lim.confidence > 0.6
    fab = by[("result", "fake/fake-1")]
    assert fab.kind == "unverified" and not fab.provenance.verified and fab.confidence <= 0.2
    assert not any(s.field == "not_a_field" for s in rec.statements)
    assert any("could not be found" in w for w in rec.warnings)
    # rule-based statements add the paper's own sentences that the model did not cover
    rule = [s for s in rec.statements if s.extractor == "rule-based"]
    assert any(s.field == "unresolved_problem" for s in rule) and any(s.field == "future_work" for s in rule)
    assert all(s.kind == "direct" and s.provenance.verified for s in rule)
    assert rec.llm == "fake/fake-1" and rec.dimensions.qubit_counts == [70]


def test_analysis_round_trips_through_database(db):
    pid = _pub(db)
    rec = PaperAnalyzer(FakeLLM([]), db).analyse(pid)
    loaded = load_analysis(pid, db)
    assert len(loaded.statements) == len(rec.statements)
    assert loaded.dimensions.platforms == rec.dimensions.platforms
    with session_scope(db) as s:
        assert s.get(Publication, pid).analysed_at is not None


def test_without_llm_and_metadata_only(db):
    rec = PaperAnalyzer(None, db).analyse(_pub(db))
    assert rec.llm is None and rec.statements and all(s.extractor == "rule-based" for s in rec.statements)
    meta_only = PaperAnalyzer(FakeLLM([]), db).analyse(_pub(db, title="No abstract here", abstract=None))
    assert meta_only.statements == [] and "Metadata only" in meta_only.warnings[0]


# ---------------------------------------------------------------- cross-paper
def _bow_embed(texts):
    vocab = sorted({w for t in texts for w in normalize_for_match(t).split()})
    m = np.array([[normalize_for_match(t).split().count(w) for w in vocab] for t in texts], dtype=float)
    return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)


def test_corpus_analysis_scope_recurring_and_untried(db):
    a = _pub(db, "Error mitigation on superconducting qubits",
             "Here we apply error mitigation on superconducting qubits. Our study is limited to shallow circuits with few qubits.")
    b = _pub(db, "Error mitigation on trapped ions",
             "Here we apply error mitigation on trapped ions. Our study is limited to shallow circuits with few qubits.")
    c = _pub(db, "Error correction on superconducting qubits",
             "Here we build error correction with a surface code on superconducting qubits. Leakage was not characterized.")
    d = _pub(db, "Error correction on trapped ions in a second lab",
             "Here we test error correction with a surface code on trapped ions as well.")
    x = _pub(db, "Excluded speculative deposit", "Here we propose a pre-Big-Bang theory of entanglement.",
             curation_status="excluded")
    an = PaperAnalyzer(FakeLLM([{"field": "limitation", "text": "Invented limitation shared by all papers.",
                                 "evidence": "this sentence is not in any paper"}]), db)
    for pid in (a, b, c, d, x):
        an.analyse(pid)
    res = analyse_corpus(db, embed=_bow_embed, theme_threshold=0.5, statement_threshold=0.8)
    assert res.papers == 4 and res.excluded == 1
    assert all(x not in t.publication_ids for t in res.themes)
    lims = [r for r in res.recurring if r.field == "limitation"]
    assert lims and lims[0].papers == 2 and {m["publication_id"] for m in lims[0].members} == {a, b}
    assert not any("Invented" in m["text"] for r in res.recurring for m in r.members)   # unverified never used
    cells = {(cl.row, cl.col): cl.publication_ids for cl in res.method_platform}
    assert cells[("quantum error mitigation", "trapped ions")] == [b]


# ------------------------------------------------------------------ retrieval
def test_hybrid_retrieval_finds_synonym_and_reports_how(tmp_path, monkeypatch):
    monkeypatch.setenv("CHROMA_PERSIST_DIRECTORY", str(tmp_path / "chroma"))
    from src.document_processing.chunker import DocumentChunk
    from src.knowledge_base.vector_store import QuantumKnowledgeBase
    kb = QuantumKnowledgeBase()
    docs = [("c1", "Dephasing times of transmon qubits were measured."), ("c2", "A recipe for sourdough bread."),
            ("c3", "Zero-noise extrapolation reduces bias in expectation values.")]
    kb.index_chunks([DocumentChunk(cid, t, "f.txt", 1, None, 0, len(t)) for cid, t in docs], "T")
    hits = kb.retrieve_hybrid("decoherence in superconducting qubits", top_k=2)
    assert hits[0]["chunk_id"] == "c1"
    assert any(h.startswith("bm25-synonym") or h == "semantic" for h in hits[0]["found_by"])


# ------------------------------------------------------------------ migration
def test_old_database_gains_new_columns(tmp_path):
    path = tmp_path / "old.db"
    eng = create_engine(f"sqlite:///{path}")
    with eng.begin() as conn:
        conn.execute(text("CREATE TABLE publications (id INTEGER PRIMARY KEY, title TEXT, normalized_title VARCHAR(512), "
                          "source VARCHAR(32))"))
        conn.execute(text("INSERT INTO publications (title, normalized_title, source) VALUES ('t', 't', 'arxiv')"))
    cols = {c["name"] for c in inspect(get_engine(f"sqlite:///{path}")).get_columns("publications")}
    assert {"curation_status", "curation_note", "analysed_at"} <= cols
    with session_scope(f"sqlite:///{path}") as s:
        assert s.get(Publication, 1).curation_status == "unreviewed"
