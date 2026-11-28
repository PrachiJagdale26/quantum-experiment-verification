"""Stage 2: research-gap identification on the synthetic sample papers (offline)."""

import pytest

from src.evaluation.gap_benchmark import load_planted_gaps, match_planted
from src.llm.extractor import extract_expected_outcomes, extract_paper_signals
from tests.research_fixtures import build_pipeline


@pytest.fixture(scope="module")
def gaps():
    pipe = build_pipeline()
    return pipe.find_gaps()


def test_paper_signals_extract_authors_wording():
    text = ("Our method is limited to two-qubit states. Extending it to GHZ states is left for future work. "
            "It remains open whether the protocol works on photonic chips. We measured P(00) = 0.5 with 97% fidelity.")
    s = extract_paper_signals(text)
    assert any("limited to two-qubit" in x for x in s["limitations"])
    assert any("future work" in x for x in s["future_work"])
    assert any("remains open" in x for x in s["open_questions"])
    assert any("P(00)" in x for x in s["claims"])


def test_pdf_title_skips_running_header_and_joins_wrapped_lines():
    from src.llm.extractor import guess_title
    text = ("--- Page 1 ---\nNature | Vol 622 | 19 October 2023 | 481\n"
            "ArticleMeasurement-induced entanglement and \nteleportation on a noisy quantum processor\n"
            "Google Quantum AI and Collaborators*\n")
    assert guess_title(text) == "Measurement-induced entanglement and teleportation on a noisy quantum processor"


def test_wrapped_pdf_sentences_are_rejoined():
    text = ("For present-day processors, the realization can be problematic \nbecause of hardware limitations. "
            "Our study is limited to small system sizes or efficiently \nsimulatable Clifford gates.")
    lims = extract_paper_signals(text)["limitations"]
    assert "Our study is limited to small system sizes or efficiently simulatable Clifford gates." in lims


def test_expected_outcomes_parser():
    outs = {o.basis_state: o.expected_probability for o in extract_expected_outcomes("We measured P(000) = 0.5 and P(111) = 0.5.")}
    assert outs == {"000": 0.5, "111": 0.5}


def test_all_planted_gaps_are_found(gaps):
    planted = load_planted_gaps()
    matches = match_planted(gaps, planted)
    missing = [p["name"] for p in planted if not matches[p["id"]]]
    assert not missing, f"Planted gaps not found: {missing}"


def test_every_gap_has_traceable_evidence_and_valid_scores(gaps):
    assert gaps, "no gaps found"
    for g in gaps:
        assert g.evidence and all(e.quote and e.paper_title for e in g.evidence)
        for score in (g.novelty, g.feasibility, g.evidence_strength, g.priority):
            assert 0.0 <= score <= 1.0
    priorities = [g.priority for g in gaps]
    assert priorities == sorted(priorities, reverse=True)
    assert len({g.gap_id for g in gaps}) == len(gaps)


def test_conflict_detected_between_papers_a_and_d(gaps):
    conflicts = [g for g in gaps if g.gap_type == "conflicting_results"]
    assert conflicts
    papers = {e.paper_title for e in conflicts[0].evidence}
    assert len(papers) == 2


def test_cross_paper_gap_merges_evidence(gaps):
    ghz_transmon = next(g for g in gaps if g.hints.get("family") == "ghz" and g.hints.get("target_system") == "Superconducting Transmon")
    assert len({e.paper_title for e in ghz_transmon.evidence}) >= 2


def test_no_papers_means_no_gaps():
    pipe = build_pipeline(load_samples=False)
    assert pipe.find_gaps() == []
