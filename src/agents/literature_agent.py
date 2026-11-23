"""
Literature Researcher Agent for LangGraph Quantum Verification.
Retrieves grounded evidence from ChromaDB vector store, evaluates claim grounding,
refines queries upon insufficient evidence (bounded retry loop), and preserves citation provenance.
"""

from typing import Dict, Any, List
from datetime import datetime
import logging

from .state import VerificationAgentState
from .tools import retrieve_literature_tool

logger = logging.getLogger(__name__)


def literature_researcher_agent(state: VerificationAgentState) -> Dict[str, Any]:
    """
    Agent 2: Literature Researcher
    Queries knowledge base, assesses relevance, refines search when necessary,
    and structures evidence for verification auditing.
    """
    timestamp = datetime.now().isoformat()
    proposal_spec = state.get("proposal_spec")
    retry_count = state.get("retrieval_retry_count", 0)
    max_retries = state.get("max_retrieval_retries", 2)
    query_override = state.get("retrieval_query")

    if not proposal_spec:
        return {
            "literature_citations": [],
            "literature_evaluations": [],
            "current_agent": "LiteratureResearcherAgent",
            "current_task": "Skipped - No Proposal Spec Available",
            "agent_logs": [{
                "timestamp": timestamp,
                "agent_name": "LiteratureResearcherAgent",
                "task": "Literature Retrieval",
                "tool_invoked": None,
                "input_summary": "No proposal spec found in state",
                "output_summary": "Skipped literature retrieval.",
                "status": "COMPLETED",
            }],
        }

    # Execute literature retrieval tool
    tool_res = retrieve_literature_tool(
        proposal_spec=proposal_spec,
        query_override=query_override,
        top_k_per_query=2,
    )

    citations = tool_res.get("citations", [])
    evaluations = tool_res.get("evaluations", [])

    # Evaluate sufficiency of retrieved evidence
    avg_relevance = (
        sum(c.similarity_score for c in citations) / len(citations)
        if citations else 0.0
    )
    
    needs_refinement = (len(citations) == 0 or avg_relevance < 0.20) and (retry_count < max_retries)
    next_query = None

    if needs_refinement:
        # Formulate refined query based on circuit operations and core physics
        gates_str = ""
        if proposal_spec.circuit and proposal_spec.circuit.gates:
            gates_str = " ".join(set(g.name for g in proposal_spec.circuit.gates))
        next_query = (
            f"Quantum state fidelity and measurement for {proposal_spec.title} {gates_str} "
            f"entanglement state preparation"
        )
        logger.info(
            f"Literature Researcher: evidence weak (avg={avg_relevance:.2f}). "
            f"Refining query (retry {retry_count + 1}/{max_retries}): '{next_query}'"
        )

    log_entry = {
        "timestamp": timestamp,
        "agent_name": "LiteratureResearcherAgent",
        "task": f"Retrieve Literature Evidence (Attempt {retry_count + 1})",
        "tool_invoked": "retrieve_literature_tool",
        "input_summary": f"Query: {tool_res.get('query_used', 'auto')[:80]}",
        "output_summary": (
            f"Retrieved {len(citations)} citations (avg relevance: {avg_relevance:.2f}). "
            f"Evaluated {len(evaluations)} claims. "
            f"Needs refinement: {needs_refinement}"
        ),
        "status": "RUNNING" if needs_refinement else "COMPLETED",
        "details": {
            "retry_count": retry_count,
            "citations": [c.citation_id for c in citations],
            "average_relevance": avg_relevance,
            "next_query": next_query,
        },
    }

    return {
        "literature_citations": citations,
        "literature_evaluations": evaluations,
        "retrieval_query": next_query,
        "retrieval_retry_count": retry_count + (1 if needs_refinement else 0),
        "literature_grounding_passed": avg_relevance >= 0.20 and len(citations) > 0,
        "current_agent": "LiteratureResearcherAgent",
        "current_task": "Literature Retrieval Completed" if not needs_refinement else "Refining Retrieval Query",
        "agent_logs": [log_entry],
    }
