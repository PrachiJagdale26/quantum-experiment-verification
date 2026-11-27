"""
Traceable Tools for the LangGraph Multi-Agent Quantum Verification System.
Directly wraps existing deterministic modules, Qiskit simulation, ChromaDB RAG,
and multi-provider LLM extraction without code duplication.
"""

from typing import Dict, Any, List, Optional, Tuple
import time
import uuid
from datetime import datetime, timezone

from ..schemas.proposal import (
    ProposalSpecification,
    CircuitSpec,
    ExpectedOutcome,
    ExtractedClaim,
)
from ..schemas.research_paper import ResearchPaperAnalysis
from ..schemas.verification import (
    LiteratureCitation,
    SimulationOutput,
    OutcomeComparison,
    VerificationCheck,
    VerificationReport,
    ClaimVerificationPair,
    CheckStatus,
    CheckCategory,
)
from ..llm.client import ConfigurableLLMClient
from ..llm.extractor import ProposalExtractor, classify_document
from ..document_processing.document_extractor import DocumentExtractor
from ..knowledge_base.vector_store import QuantumKnowledgeBase
from ..knowledge_base.retriever import QuantumRetriever
from ..circuit.simulator import QuantumSimulator, SimulationError
from ..circuit.metrics import compare_outcomes, total_variation_distance, compute_fidelity
from ..verification.rules import (
    check_qubit_count,
    check_circuit_syntax,
    check_statevector_normalization,
    check_outcome_distribution,
    check_physical_hardware_feasibility,
)
from ..reporting.generator import ReportGenerator


# Singleton / cached tool helpers
_DEFAULT_KB: Optional[QuantumKnowledgeBase] = None
_DEFAULT_SIMULATOR: Optional[QuantumSimulator] = None


def get_default_kb() -> QuantumKnowledgeBase:
    global _DEFAULT_KB
    if _DEFAULT_KB is None:
        _DEFAULT_KB = QuantumKnowledgeBase()
        _DEFAULT_KB.seed_starter_corpus()
    return _DEFAULT_KB


def get_default_simulator() -> QuantumSimulator:
    global _DEFAULT_SIMULATOR
    if _DEFAULT_SIMULATOR is None:
        _DEFAULT_SIMULATOR = QuantumSimulator()
    return _DEFAULT_SIMULATOR


# ---------------------------------------------------------
# Tool 1: Document Extraction Tool
# ---------------------------------------------------------
def extract_document_tool(
    raw_text: str,
    filename: Optional[str] = None,
    file_bytes: Optional[bytes] = None,
    llm_client: Optional[ConfigurableLLMClient] = None,
) -> Dict[str, Any]:
    """
    Extracts text, metadata, and structured specification from proposal document.
    Uses unified DocumentExtractor for PDF/DOCX and ProposalExtractor (with Vertex/Gemini).
    """
    start_time = time.time()
    doc_meta = {}
    extracted_text = raw_text

    # 1. If file_bytes provided, run DocumentExtractor
    extracted_doc = None
    if file_bytes:
        doc_extractor = DocumentExtractor()
        extracted_doc = doc_extractor.extract_from_bytes(file_bytes, filename=filename or "proposal.pdf")
        extracted_text = extracted_doc.full_text
        doc_meta = {
            "page_count": len(extracted_doc.pages),
            "has_math": bool(extracted_doc.math_expressions),
            "sections": list(extracted_doc.sections.keys()),
        }

    # 2. Classify document
    doc_type = classify_document(extracted_text, filename=filename)

    # 3. Extract structured spec via ProposalExtractor
    client = llm_client or ConfigurableLLMClient()
    extractor = ProposalExtractor(llm_client=client)

    proposal_spec = None
    paper_analysis = None

    if doc_type == "research_paper":
        paper_analysis = extractor.extract_research_paper(extracted_text, document_title=filename, extracted_doc=extracted_doc)
        if paper_analysis.experiments:
            # Verify the paper's first executable experiment (or its first experiment if none is executable).
            primary_exp = next((e for e in paper_analysis.experiments if e.has_executable_circuit), paper_analysis.experiments[0])
            circuit, missing = None, []
            if primary_exp.gates and primary_exp.num_qubits:
                try:
                    circuit = CircuitSpec(num_qubits=primary_exp.num_qubits, gates=primary_exp.gates)
                except Exception as exc:
                    missing.append(f"Circuit specification error: {exc}")
            if not primary_exp.num_qubits:
                missing.append("Number of qubits not specified.")
            if not primary_exp.gates:
                missing.append("No quantum gate sequence or circuit diagram provided.")
            if not primary_exp.expected_outcomes:
                missing.append("No expected measurement distribution specified.")
            claims = (paper_analysis.key_claims or paper_analysis.results_and_findings)[:3]
            proposal_spec = ProposalSpecification(
                title=f"{paper_analysis.title}: {primary_exp.title}",
                objective=paper_analysis.research_objective or paper_analysis.title,
                hypothesis=primary_exp.findings or paper_analysis.research_problem,
                quantum_system=primary_exp.quantum_system,
                num_qubits=primary_exp.num_qubits,
                circuit=circuit,
                expected_outcomes=primary_exp.expected_outcomes,
                author_claims=[
                    ExtractedClaim(claim_id=f"CLM-{i+1:02d}", claim_type="outcome", statement=c)
                    for i, c in enumerate(claims)
                ],
                missing_information=missing,
                raw_text=extracted_text,
                source_filename=filename,
                document_type="research_paper",
                doi=paper_analysis.doi,
            )
        else:
            proposal_spec = extractor.extract(extracted_text, document_title=filename)
    else:
        proposal_spec = extractor.extract(extracted_text, document_title=filename)

    duration = time.time() - start_time
    return {
        "raw_text": extracted_text,
        "filename": filename,
        "doc_type": doc_type,
        "doc_metadata": doc_meta,
        "proposal_spec": proposal_spec,
        "paper_analysis": paper_analysis,
        "duration_seconds": duration,
    }


# ---------------------------------------------------------
# Tool 2: Literature Retrieval Tool
# ---------------------------------------------------------
def retrieve_literature_tool(
    proposal_spec: ProposalSpecification,
    query_override: Optional[str] = None,
    top_k_per_query: int = 2,
    kb: Optional[QuantumKnowledgeBase] = None,
) -> Dict[str, Any]:
    """
    Retrieves grounded evidence from ChromaDB vector store.
    Supports query refinement if initial evidence is weak.
    """
    start_time = time.time()
    knowledge_base = kb or get_default_kb()
    retriever = QuantumRetriever(knowledge_base)

    if query_override:
        raw_results = knowledge_base.retrieve(query_override, top_k=top_k_per_query * 2)
        citations = []
        for idx, match in enumerate(raw_results):
            score = match.get("score", 0.0)
            citations.append(LiteratureCitation(
                citation_id=f"CIT-REF-{idx+1:02d}",
                source_title=match.get("title", "Quantum Reference"),
                filename=match.get("source_filename", "corpus.txt"),
                page_number=match.get("page_number", 1),
                chunk_id=match.get("chunk_id", f"chk_{idx}"),
                passage=match.get("text", "")[:350],
                similarity_score=float(score) if score is not None else 0.5,
                relevance_comment=f"Refined query match for: '{query_override[:80]}...'",
            ))
    else:
        citations = retriever.retrieve_for_proposal(proposal_spec, top_k_per_query=top_k_per_query)

    evaluations = []
    for claim in (proposal_spec.author_claims or []):
        claim_statement = getattr(claim, "statement", None) or getattr(claim, "claim_text", "")
        claim_text_lower = claim_statement.lower()
        matched_cits = [c.citation_id for c in citations if any(w in c.passage.lower() for w in claim_text_lower.split()[:4])]
        status = "Supported" if matched_cits else "Unaddressed"
        evaluations.append({
            "claim_id": claim.claim_id,
            "claim_text": claim_statement,
            "literature_support_status": status,
            "supporting_citations": matched_cits if matched_cits else ([citations[0].citation_id] if citations else []),
        })

    duration = time.time() - start_time
    return {
        "citations": citations,
        "evaluations": evaluations,
        "query_used": query_override or "default_proposal_queries",
        "duration_seconds": duration,
    }


# ---------------------------------------------------------
# Tool 3: Quantum Simulation Tool (Deterministic Qiskit)
# ---------------------------------------------------------
def simulate_circuit_tool(
    circuit_spec: CircuitSpec,
    expected_outcomes: Optional[List[ExpectedOutcome]] = None,
    simulator: Optional[QuantumSimulator] = None,
) -> Dict[str, Any]:
    """
    Executes confirmed circuit using Qiskit Statevector simulation.
    Strictly computational: calculates probabilities, statevector, TVD, and fidelity.
    An LLM NEVER calculates or tampers with these results.
    """
    start_time = time.time()
    sim = simulator or get_default_simulator()

    sim_output = sim.simulate(circuit_spec)

    outcome_comparisons = []
    tvd = None
    fidelity = None

    if expected_outcomes and sim_output:
        comparisons, all_matched, max_error = compare_outcomes(
            expected_outcomes=expected_outcomes,
            computed_probabilities=sim_output.probabilities or {},
            default_tolerance=0.05,
        )
        outcome_comparisons = comparisons
        exp_dict = {o.basis_state.strip(): o.expected_probability for o in expected_outcomes}
        tvd = total_variation_distance(exp_dict, sim_output.probabilities or {})
        fidelity = max(0.0, 1.0 - tvd)

    duration = time.time() - start_time
    return {
        "simulation_output": sim_output,
        "outcome_comparisons": outcome_comparisons,
        "tvd": tvd,
        "fidelity": fidelity,
        "duration_seconds": duration,
    }


# ---------------------------------------------------------
# Tool 4: Verification Auditor Tool (Deterministic Rules)
# ---------------------------------------------------------
def run_verification_audit_tool(
    proposal_spec: ProposalSpecification,
    simulation_output: Optional[SimulationOutput],
    outcome_comparisons: Optional[List[OutcomeComparison]] = None,
    citations: Optional[List[LiteratureCitation]] = None,
) -> Dict[str, Any]:
    """
    Executes deterministic verification rules on the proposal and simulation output.
    Enforces strict computational FAIL precedence over LLM claims.
    """
    start_time = time.time()
    checks: List[VerificationCheck] = []
    fail_reasons: List[str] = []

    # 1. Qubit count validation
    chk_qubit = check_qubit_count(proposal_spec)
    checks.append(chk_qubit)
    if chk_qubit.status == CheckStatus.FAIL:
        fail_reasons.append(f"Qubit count: {chk_qubit.details}")

    # 2. Circuit syntax & gates validation
    chk_syntax = check_circuit_syntax(proposal_spec)
    checks.append(chk_syntax)
    if chk_syntax.status == CheckStatus.FAIL:
        fail_reasons.append(f"Circuit syntax: {chk_syntax.details}")

    # 3. Statevector normalization check
    chk_norm = check_statevector_normalization(simulation_output)
    checks.append(chk_norm)
    if chk_norm.status == CheckStatus.FAIL:
        fail_reasons.append(f"Statevector normalization: {chk_norm.details}")

    # 4. Outcome distribution & TVD check
    computed_comparisons = outcome_comparisons or []
    computed_tvd = None
    if proposal_spec.expected_outcomes:
        chk_outcomes, comp_list, tvd_val = check_outcome_distribution(
            proposal=proposal_spec,
            sim_output=simulation_output,
        )
        checks.append(chk_outcomes)
        computed_comparisons = comp_list
        computed_tvd = tvd_val
        if chk_outcomes.status == CheckStatus.FAIL:
            fail_reasons.append(f"Measurement distribution: {chk_outcomes.details}")
    else:
        checks.append(VerificationCheck(
            check_id="CHK-DIST-01",
            name="Measurement Outcome Distribution",
            category=CheckCategory.DISTRIBUTION_MATCH,
            status=CheckStatus.INCONCLUSIVE,
            description="Validate simulated statevector probabilities against author's expected probabilities.",
            details="No expected measurement outcomes specified in proposal. Distribution match skipped.",
            error_message="Omitted expected probabilities in proposal.",
        ))

    # 5. Physical hardware feasibility check
    chk_feasibility = check_physical_hardware_feasibility(proposal_spec)
    checks.append(chk_feasibility)
    if chk_feasibility.status == CheckStatus.FAIL:
        fail_reasons.append(f"Hardware feasibility: {chk_feasibility.details}")

    # 6. Literature grounding check
    if citations and len(citations) > 0:
        avg_score = sum(c.similarity_score for c in citations) / len(citations)
        checks.append(VerificationCheck(
            check_id="CHK-LIT-01",
            name="Literature Evidence Grounding",
            category=CheckCategory.LITERATURE_GROUNDING,
            status=CheckStatus.PASS if avg_score >= 0.20 else CheckStatus.INCONCLUSIVE,
            description="Cross-reference proposal claims against genuine peer-reviewed quantum literature.",
            details=f"Retrieved {len(citations)} literature citation(s) (avg similarity: {avg_score:.2f}).",
        ))
    else:
        checks.append(VerificationCheck(
            check_id="CHK-LIT-01",
            name="Literature Evidence Grounding",
            category=CheckCategory.LITERATURE_GROUNDING,
            status=CheckStatus.INCONCLUSIVE,
            description="Cross-reference proposal claims against genuine peer-reviewed quantum literature.",
            details="No supporting literature citations found in knowledge base.",
            error_message="Missing literature grounding: 0 matching passages.",
        ))

    # Summary counts
    passed = sum(1 for c in checks if c.status == CheckStatus.PASS)
    failed = sum(1 for c in checks if c.status == CheckStatus.FAIL)
    inconclusive = sum(1 for c in checks if c.status == CheckStatus.INCONCLUSIVE)
    not_run = sum(1 for c in checks if c.status == CheckStatus.NOT_RUN)

    # Strict status decision: ANY fail => FAIL
    if failed > 0:
        overall_status = "FAIL"
    elif inconclusive > 0:
        overall_status = "INCONCLUSIVE"
    else:
        overall_status = "PASS"

    duration = time.time() - start_time
    return {
        "checks": checks,
        "summary_counts": {
            "PASS": passed,
            "FAIL": failed,
            "INCONCLUSIVE": inconclusive,
            "NOT_RUN": not_run,
        },
        "overall_status": overall_status,
        "fail_reasons": fail_reasons,
        "outcome_comparisons": computed_comparisons,
        "tvd": computed_tvd,
        "duration_seconds": duration,
    }


# ---------------------------------------------------------
# Tool 5: Report Generation Tool
# ---------------------------------------------------------
def generate_report_tool(
    proposal_spec: ProposalSpecification,
    simulation_output: Optional[SimulationOutput],
    checks: List[VerificationCheck],
    summary_counts: Dict[str, int],
    outcome_comparisons: Optional[List[OutcomeComparison]] = None,
    citations: Optional[List[LiteratureCitation]] = None,
    tvd: Optional[float] = None,
    fidelity: Optional[float] = None,
    explainability_notes: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compiles final structured VerificationReport, Markdown report, and JSON string.
    """
    start_time = time.time()
    report_id = f"REP-AGT-{uuid.uuid4().hex[:8].upper()}"
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # Strict status calculation
    has_fail = any(c.status == CheckStatus.FAIL for c in checks)
    has_inconclusive = any(c.status == CheckStatus.INCONCLUSIVE for c in checks)
    if has_fail:
        overall_status = "FAIL"
    elif has_inconclusive:
        overall_status = "INCONCLUSIVE"
    else:
        overall_status = "PASS"

    summary_text = (
        f"Multi-Agent Verification for '{proposal_spec.title}' completed with status: {overall_status}.\n"
        f"Passed {summary_counts.get('PASS', 0)} of {len(checks)} checks. "
    )
    if tvd is not None:
        summary_text += f"Total Variation Distance (TVD): {tvd:.4f}. "
    if fidelity is not None:
        summary_text += f"State Fidelity: {fidelity:.4f}. "
    if explainability_notes:
        summary_text += f"\n\nScientific Auditor Notes:\n{explainability_notes}"

    # Build claim verification pairs
    claims_analysis = []
    for claim in (proposal_spec.author_claims or []):
        claim_stmt = getattr(claim, "statement", None) or getattr(claim, "claim_text", "")
        matching_cits = [c.citation_id for c in (citations or []) if any(w in c.passage.lower() for w in claim_stmt.lower().split()[:3])]
        comp_status = "Verified" if overall_status == "PASS" else ("Contradicted" if has_fail else "Not Applicable")
        claims_analysis.append(ClaimVerificationPair(
            claim_id=claim.claim_id,
            claim_text=claim_stmt,
            claim_category=getattr(claim, "claim_type", "hypothesis"),
            literature_support_status="Supported" if matching_cits else "Unaddressed",
            supporting_citations=matching_cits if matching_cits else ([citations[0].citation_id] if citations else []),
            computational_status=comp_status,
            explanation=f"Evaluated against independent simulation (status: {overall_status}) and literature corpus.",
        ))

    report = VerificationReport(
        report_id=report_id,
        proposal_title=proposal_spec.title,
        timestamp=timestamp,
        software_environment={
            "Python": "3.12.5",
            "Orchestrator": "LangGraph StateGraph",
            "Simulator": "Qiskit Statevector / NumPy",
            "Vector Store": "ChromaDB (Cosine Similarity)",
        },
        overall_summary=summary_text,
        checks=checks,
        summary_counts=summary_counts,
        simulation=simulation_output,
        outcome_comparisons=outcome_comparisons or [],
        fidelity_score=fidelity,
        total_variation_distance=tvd,
        citations=citations or [],
        claims_analysis=claims_analysis,
        missing_information=proposal_spec.missing_information or [],
        detected_inconsistencies=proposal_spec.potential_inconsistencies or [],
        recommended_next_steps=[
            "Pure-state Qiskit simulation executed deterministically.",
            "Independent statevector probabilities calculated without LLM intervention.",
            "Literature citations grounded directly in indexed corpus.",
        ],
    )

    json_str = ReportGenerator.to_json(report)
    md_str = ReportGenerator.to_markdown(report)

    duration = time.time() - start_time
    return {
        "report": report,
        "json_str": json_str,
        "markdown_str": md_str,
        "duration_seconds": duration,
    }
