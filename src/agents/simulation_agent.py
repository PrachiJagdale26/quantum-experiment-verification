"""
Quantum Simulation Agent for LangGraph Quantum Verification.
Executes confirmed circuits using Qiskit Statevector simulation and independent metrics.
Calculates basis probabilities, statevector amplitudes, TVD, and state fidelity.
Strictly computational: the LLM never calculates or modifies quantum simulation outputs.
"""

from typing import Dict, Any, List
from datetime import datetime
import logging

from .state import VerificationAgentState
from .tools import simulate_circuit_tool

logger = logging.getLogger(__name__)


def quantum_simulation_agent(state: VerificationAgentState) -> Dict[str, Any]:
    """
    Agent 3: Quantum Simulation Agent
    Simulates the confirmed quantum circuit using Qiskit statevector evolution.
    Computes mathematical metrics (TVD and fidelity) against author expectations.
    """
    timestamp = datetime.now().isoformat()
    proposal_spec = state.get("proposal_spec")

    if not proposal_spec or not proposal_spec.circuit or not proposal_spec.circuit.gates:
        logger.warning("SimulationAgent: No executable circuit specification found.")
        return {
            "simulation_output": None,
            "outcome_comparisons": [],
            "expected_probabilities": {},
            "computed_probabilities": {},
            "tvd_score": None,
            "fidelity_score": None,
            "simulation_error": "No circuit or gate sequence specified in proposal.",
            "current_agent": "SimulationAgent",
            "current_task": "Simulation Aborted - Missing Circuit",
            "agent_logs": [{
                "timestamp": timestamp,
                "agent_name": "SimulationAgent",
                "task": "Circuit Simulation",
                "tool_invoked": None,
                "input_summary": "Missing circuit specification",
                "output_summary": "Simulation aborted due to missing gates.",
                "status": "FAILED",
            }],
        }

    try:
        # Run independent Qiskit simulation tool
        tool_res = simulate_circuit_tool(
            circuit_spec=proposal_spec.circuit,
            expected_outcomes=proposal_spec.expected_outcomes,
        )

        sim_output = tool_res.get("simulation_output")
        outcome_comparisons = tool_res.get("outcome_comparisons", [])
        tvd = tool_res.get("tvd")
        fidelity = tool_res.get("fidelity")

        computed_probs = getattr(sim_output, "probabilities", {}) if sim_output else {}
        expected_probs = {
            o.basis_state: o.expected_probability for o in (proposal_spec.expected_outcomes or [])
        }

        log_entry = {
            "timestamp": timestamp,
            "agent_name": "SimulationAgent",
            "task": "Execute Qiskit Statevector Simulation",
            "tool_invoked": "simulate_circuit_tool",
            "input_summary": (
                f"{proposal_spec.circuit.num_qubits} qubits, "
                f"{len(proposal_spec.circuit.gates)} gates"
            ),
            "output_summary": (
                f"Simulated {len(computed_probs)} states. "
                f"TVD: {f'{tvd:.4f}' if tvd is not None else 'N/A'}, "
                f"Fidelity: {f'{fidelity:.4f}' if fidelity is not None else 'N/A'}"
            ),
            "status": "COMPLETED",
            "details": {
                "num_qubits": proposal_spec.circuit.num_qubits,
                "execution_time_ms": getattr(sim_output, "execution_time_ms", 0.0) if sim_output else 0.0,
                "probabilities": computed_probs,
                "tvd": tvd,
                "fidelity": fidelity,
            },
        }

        return {
            "simulation_output": sim_output,
            "outcome_comparisons": outcome_comparisons,
            "expected_probabilities": expected_probs,
            "computed_probabilities": computed_probs,
            "tvd_score": tvd,
            "fidelity_score": fidelity,
            "simulation_error": None,
            "current_agent": "SimulationAgent",
            "current_task": "Quantum Simulation Completed",
            "agent_logs": [log_entry],
        }

    except Exception as e:
        logger.error(f"SimulationAgent error during execution: {e}", exc_info=True)
        return {
            "simulation_output": None,
            "outcome_comparisons": [],
            "expected_probabilities": {},
            "computed_probabilities": {},
            "tvd_score": None,
            "fidelity_score": None,
            "simulation_error": str(e),
            "current_agent": "SimulationAgent",
            "current_task": f"Simulation Failed: {e}",
            "agent_logs": [{
                "timestamp": timestamp,
                "agent_name": "SimulationAgent",
                "task": "Circuit Simulation",
                "tool_invoked": "simulate_circuit_tool",
                "input_summary": f"Circuit: {proposal_spec.circuit}",
                "output_summary": f"Simulation threw exception: {e}",
                "status": "FAILED",
                "details": {"error": str(e)},
            }],
        }
