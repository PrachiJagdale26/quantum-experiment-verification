"""
Regression: the agentic extract_document_tool research-paper branch used field names
from an old schema (extracted_doc.text, paper_analysis.quantum_experiments,
ExtractedClaim(claim_text=...)) and crashed on any research paper.
"""

import os

from src.agents.tools import extract_document_tool
from src.llm.client import ConfigurableLLMClient
from src.schemas.proposal import ProposalSpecification

PAPER = os.path.join(os.path.dirname(__file__), "..", "data", "sample_papers", "paper_b_ghz_trapped_ion.txt")

PAPER_TEXT = (
    "Nature Machine Intelligence | Research Article\n"
    "DOI: 10.1038/s42256-025-01153-0\n"
    "Title: Multi-Circuit Benchmarking on Superconducting Processors\n\n"
    "Abstract: We benchmark two distinct quantum circuits on a superconducting processor.\n\n"
    "Experiment 1: Two-qubit Bell State Preparation\n"
    "We allocate 2 qubits and apply H gate on qubit 0, followed by CX on qubits 0, 1.\n"
    "Expected outcomes: P(00) = 0.5, P(11) = 0.5.\n\n"
    "Results: Both experiments verified state fidelity above 95%.\n\n"
    "References\n[1] Nielsen and Chuang."
)


def test_paper_text_branch_builds_valid_proposal():
    res = extract_document_tool(raw_text=PAPER_TEXT, filename="s42256-025-01153-0.pdf",
                                llm_client=ConfigurableLLMClient(demo_mode=True))
    assert res["doc_type"] == "research_paper"
    assert res["paper_analysis"] is not None
    spec = res["proposal_spec"]
    assert isinstance(spec, ProposalSpecification)
    assert spec.num_qubits == 2
    assert spec.circuit is not None and [g.name for g in spec.circuit.gates] == ["H", "CX"]
    assert all(c.statement for c in spec.author_claims)


def test_paper_file_bytes_branch_does_not_crash():
    with open(PAPER, "rb") as f:
        data = f.read()
    res = extract_document_tool(raw_text="", filename="paper_b_ghz_trapped_ion.txt", file_bytes=data,
                                llm_client=ConfigurableLLMClient(demo_mode=True))
    assert res["raw_text"].strip()
    assert res["doc_metadata"]["page_count"] >= 1
    assert isinstance(res["proposal_spec"], ProposalSpecification)
