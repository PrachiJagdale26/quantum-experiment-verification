"""
Streamlit UI Components for Quantum Experiment Verification.
Renders scientific audit badges, probability distribution charts,
traceable citation cards, and statevector tables.
"""

from typing import List, Dict, Any, Optional
import streamlit as st
import pandas as pd
import altair as alt
from ..schemas.verification import CheckStatus, VerificationCheck, LiteratureCitation, OutcomeComparison, SimulationOutput
from ..schemas.research_paper import ResearchPaperAnalysis
from .theme import status_badge_markdown, status_label, CHART_EXPECTED, CHART_COMPUTED


def render_status_badge(status: CheckStatus) -> str:
    """Colored status badge (Streamlit markdown) in the theme's status colors."""
    return status_badge_markdown(status)


# Row tints for the checklist table, matching the status colors in the theme.
_ROW_TINT = {
    "Pass": "background-color: #DCEBE5",
    "Fail": "background-color: #F3DEDB",
    "Inconclusive": "background-color: #F4E8CF",
}


def render_checklist_table(checks: List[VerificationCheck]):
    """Display the verification checklist in a structured Streamlit table."""
    data = []
    for c in checks:
        data.append({
            "Status": status_label(c.status),
            "Check": c.name,
            "Category": c.category.value,
            "Result": c.details,
            "Problem": c.error_message or "",
            "ID": c.check_id,
        })
    df = pd.DataFrame(data)

    def tint_row(row):
        return [_ROW_TINT.get(row["Status"], "")] * len(row)

    st.dataframe(df.style.apply(tint_row, axis=1), use_container_width=True, hide_index=True)


EXPECTED_LABEL = "Expected by authors"
SIMULATED_LABEL = "Simulated"


def render_probability_bars(expected: Dict[str, float], computed: Dict[str, float], height: int = 260):
    """Side-by-side bars of expected vs. simulated probability per basis state."""
    states = sorted(set(expected) | set(computed))
    if not computed and not expected:
        return
    if len(states) > 16:
        # Large registers: show only states that matter.
        states = [s for s in states if expected.get(s, 0) > 1e-4 or computed.get(s, 0) > 1e-4]
    rows = []
    for s in states:
        ket = f"|{s}⟩"
        if expected:
            rows.append({"Basis state": ket, "Series": EXPECTED_LABEL, "Probability": expected.get(s, 0.0)})
        rows.append({"Basis state": ket, "Series": SIMULATED_LABEL, "Probability": computed.get(s, 0.0)})
    df = pd.DataFrame(rows)
    domain = [EXPECTED_LABEL, SIMULATED_LABEL] if expected else [SIMULATED_LABEL]
    colors = [CHART_EXPECTED, CHART_COMPUTED] if expected else [CHART_COMPUTED]
    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
        .encode(
            x=alt.X("Basis state:N", title=None, sort=[f"|{s}⟩" for s in states],
                    axis=alt.Axis(labelAngle=0, labelFont="STIX Two Text", labelFontSize=14)),
            y=alt.Y("Probability:Q", title="Probability", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Series:N", scale=alt.Scale(domain=domain, range=colors),
                            legend=alt.Legend(title=None, orient="top", direction="horizontal")),
            xOffset=alt.XOffset("Series:N", sort=domain),
            tooltip=["Basis state", "Series", alt.Tooltip("Probability:Q", format=".3f")],
        )
        .properties(height=height)
        .configure(font="Instrument Sans")
        .configure_view(strokeWidth=0)
        .configure_axis(gridColor="#D5DDD9", domainColor="#C9D3CE", labelColor="#1D2240", titleColor="#56607A")
    )
    st.altair_chart(chart, use_container_width=True)


def render_probability_distribution_chart(comparisons: List[OutcomeComparison], computed_probs: Dict[str, float]):
    """Expected (from the proposal) vs. simulated probabilities."""
    expected = {c.basis_state: c.expected_probability for c in comparisons} if comparisons else {}
    computed = dict(computed_probs or {})
    for c in comparisons or []:
        computed.setdefault(c.basis_state, c.computed_probability)
    render_probability_bars(expected, computed)


def render_citation_card(citation: LiteratureCitation, idx: int):
    """One retrieved passage, with where it came from and how closely it matches."""
    page = f", page {citation.page_number}" if citation.page_number else ""
    with st.expander(f"{citation.source_title}{page}", expanded=(idx == 0)):
        st.markdown(f"> {citation.passage}")
        st.caption(
            f"Match score {citation.similarity_score:.2f}. "
            f"From `{citation.filename}`, passage `{citation.chunk_id}`."
        )
        if citation.relevance_comment:
            st.caption(citation.relevance_comment)


def render_statevector_table(sim: SimulationOutput):
    """Render formatted state amplitudes and phases."""
    rows = []
    for c in sim.statevector_components:
        if c.probability > 0.0001 or sim.num_qubits <= 3:
            complex_str = f"{c.real:+.4f} {c.imag:+.4f}j"
            rows.append({
                "Basis state": f"|{c.basis_state}⟩",
                "Complex Amplitude": complex_str,
                "Magnitude |c|": f"{c.magnitude:.4f}",
                "Probability |c|²": f"{c.probability:.4f}",
                "Phase (rad)": f"{c.phase_rad:.2f}"
            })
    st.table(pd.DataFrame(rows))


def render_research_paper_view(paper: ResearchPaperAnalysis):
    """Render comprehensive structured view of an academic research paper."""
    st.markdown(f"### {paper.title}")
    
    col_hdr1, col_hdr2 = st.columns([3, 1])
    with col_hdr1:
        if paper.doi:
            st.markdown(f"**DOI**: [`{paper.doi}`](https://doi.org/{paper.doi})")
        if paper.publication_info:
            st.markdown(f"**Publication**: {paper.publication_info}")
        if paper.authors:
            st.markdown(f"**Authors**: {', '.join(paper.authors)}")
    with col_hdr2:
        st.info("**Document Type**\nAcademic Research Paper")

    if paper.abstract:
        with st.expander("Paper Abstract", expanded=True):
            st.write(paper.abstract)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Research Objective")
        st.write(paper.research_objective)

        st.markdown("#### Research Problem")
        st.write(paper.research_problem)

        st.markdown("#### Quantum Systems Discussed")
        for qs in paper.quantum_systems_discussed:
            st.markdown(f"- `{qs}`")

    with col2:
        st.markdown("#### Methodology")
        st.write(paper.methodology)

        st.markdown("#### Experimental & Simulation Setup")
        st.write(paper.experimental_setup or "Numerical quantum circuit simulation and state evolution.")

        st.markdown("#### Algorithms & Frameworks")
        st.write(", ".join(f"`{af}`" for af in paper.algorithms_and_frameworks) if paper.algorithms_and_frameworks else "Quantum circuit model")

    st.markdown("---")
    st.subheader(f"Experiments & Benchmarks Discussed ({len(paper.experiments)} identified)")
    st.caption("Academic papers typically discuss several experimental evaluations or circuit configurations.")

    for exp in paper.experiments:
        with st.container(border=True):
            exp_col1, exp_col2 = st.columns([3, 1])
            with exp_col1:
                st.markdown(f"**[{exp.experiment_id}] {exp.title}**")
                st.write(f"Architecture: `{exp.quantum_system or 'Ideal Simulation'}` | Qubits: **{exp.num_qubits if exp.num_qubits else 'Unspecified / Flexible'}**")
                if exp.circuit_summary:
                    st.caption(f"Circuit Summary: {exp.circuit_summary}")
                if exp.findings:
                    st.caption(f"Findings: {exp.findings}")
            with exp_col2:
                if exp.has_executable_circuit:
                    st.success("Can be simulated")
                else:
                    st.info("Conceptual, no circuit given")

    st.markdown("---")
    col_rf1, col_rf2 = st.columns(2)
    with col_rf1:
        st.markdown("#### Key Results & Findings")
        for r in paper.results_and_findings:
            st.markdown(f"- {r}")
    with col_rf2:
        st.markdown("#### Stated Limitations & Constraints")
        for lim in paper.limitations:
            st.markdown(f"- {lim}")

