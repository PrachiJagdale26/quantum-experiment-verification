"""
Visual identity for the Streamlit dashboard.

The palette and fonts live in .streamlit/config.toml. This module adds the
pieces the theme can't express: the verdict header (the one prominent element
on the page) and a little CSS for tab grouping and typography.

The verdict header shows |<proposal|simulation>|^2, the fidelity between the
proposal's expected distribution and the simulated one. It is computed once by
src/circuit/metrics.py::distribution_fidelity and stored on the report / agent
state; this module only displays it.
"""

import base64
from html import escape
from math import cos, pi, sin
from typing import Any, Dict, Optional

import streamlit as st

# Status colors: (label, fill, text on the light page, text on the dark verdict print).
# Text values reach at least 5:1 on the page; the "on print" values at least
# 3:1 over the brightest interference fringe (the verdict is large text).
STATUS_STYLE = {
    "PASS": ("Pass", "#2E7D6B", "#24685A", "#8EDDBF"),
    "FAIL": ("Fail", "#B3362C", "#B3362C", "#FFAA9E"),
    "INCONCLUSIVE": ("Inconclusive", "#C98A12", "#8A5A00", "#F3C35C"),
    "NOT_RUN": ("Not run", "#7A8AA6", "#56607A", "#B8C6D8"),
}

# Expected (author) vs. computed (simulation) series, used by every chart.
CHART_EXPECTED = "#1D2240"   # Indigo ink
CHART_COMPUTED = "#1F4E8C"   # Cyanotype

# The verdict header is printed on Prussian blue, like a cyanotype.
PRINT_BLUE = "#183A66"
FRINGE_LIGHT = "#8FB8E8"

# The first six tabs follow the verification sequence; the remaining three are
# tools. CSS draws a divider before the first tool tab.
SEQUENCE_TAB_COUNT = 6


def _double_slit_fringes(center: float = 0.84, fringe_period: float = 0.045,
                         envelope_width: float = 0.42, peak_opacity: float = 0.30,
                         samples: int = 240) -> str:
    """
    Background image for the verdict print: the intensity of a two-slit
    interference pattern, I(x) = cos^2(pi*x/period) * sinc^2(pi*x/width),
    drawn as light bands whose opacity follows I(x). The central (brightest)
    fringe sits behind the verdict. Fringes fade toward the top and bottom
    edges, as on a screen lit through finite-height slits.
    Returns a CSS url() with an inline SVG.
    """
    stops = []
    for i in range(samples + 1):
        x = i / samples
        d = x - center
        u = pi * d / envelope_width
        envelope = 1.0 if abs(u) < 1e-9 else (sin(u) / u) ** 2
        intensity = cos(pi * d / fringe_period) ** 2 * envelope
        stops.append(
            f'<stop offset="{x:.3f}" stop-color="{FRINGE_LIGHT}" '
            f'stop-opacity="{peak_opacity * intensity:.3f}"/>'
        )
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 200" preserveAspectRatio="none">'
        '<defs>'
        f'<linearGradient id="f" x1="0" x2="1" y1="0" y2="0">{"".join(stops)}</linearGradient>'
        '<linearGradient id="v" x1="0" x2="0" y1="0" y2="1">'
        '<stop offset="0" stop-color="#fff" stop-opacity="0.15"/>'
        '<stop offset="0.45" stop-color="#fff" stop-opacity="1"/>'
        '<stop offset="0.7" stop-color="#fff" stop-opacity="1"/>'
        '<stop offset="1" stop-color="#fff" stop-opacity="0.1"/>'
        '</linearGradient>'
        '<mask id="m"><rect width="1000" height="200" fill="url(#v)"/></mask>'
        '</defs>'
        '<rect width="1000" height="200" fill="url(#f)" mask="url(#m)"/>'
        '</svg>'
    )
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f'url("data:image/svg+xml;base64,{encoded}")'


FRINGES = _double_slit_fringes()

CSS = f"""
<style>
  /* The verdict header: a cyanotype print with a double-slit interference pattern. */
  [data-testid="stMarkdownContainer"] .qv-head {{
    margin: 0 0 1.75rem 0;
    padding: 1.6rem 2rem 1.5rem 2rem;
    border-radius: 6px;
    background-color: {PRINT_BLUE};
    background-image: {FRINGES};
    background-size: 100% 100%;
    background-repeat: no-repeat;
    box-shadow: inset 0 0 0 1px rgba(143, 184, 232, 0.18);
  }}
  [data-testid="stMarkdownContainer"] .qv-title {{
    font-family: "STIX Two Text", "Cambria", serif;
    font-size: 1.25rem;
    line-height: 1.3;
    color: #C9D7E7;
    margin: 0 0 0.35rem 0;
    max-width: 60ch;
  }}
  [data-testid="stMarkdownContainer"] .qv-row {{
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    justify-content: space-between;
    gap: 0.25rem 2rem;
  }}
  [data-testid="stMarkdownContainer"] .qv-ket, [data-testid="stMarkdownContainer"] .qv-verdict {{
    font-family: "STIX Two Text", "Cambria Math", "Cambria", serif;
    font-size: clamp(1.75rem, 4.2vw, 3.4rem);
    line-height: 1.1;
    letter-spacing: -0.01em;
    margin: 0;
  }}
  [data-testid="stMarkdownContainer"] .qv-ket {{ color: #F1F5F9; font-weight: 400; }}
  [data-testid="stMarkdownContainer"] .qv-ket .qv-bracket {{ color: #9CB6D4; }}
  [data-testid="stMarkdownContainer"] .qv-ket .qv-label {{ font-style: italic; }}
  [data-testid="stMarkdownContainer"] .qv-ket .qv-value {{ font-variant-numeric: tabular-nums; }}
  [data-testid="stMarkdownContainer"] .qv-ket .qv-eq {{ white-space: nowrap; }}
  [data-testid="stMarkdownContainer"] .qv-ket.qv-empty .qv-value {{ color: #9CB6D4; }}
  [data-testid="stMarkdownContainer"] .qv-verdict {{ font-weight: 600; }}
  [data-testid="stMarkdownContainer"] .qv-note {{
    color: #C9D7E7;
    font-size: 0.95rem;
    line-height: 1.5;
    margin: 0.7rem 0 0 0;
    max-width: 75ch;
  }}
  [data-testid="stMarkdownContainer"] .qv-note .qv-ket-inline {{ font-family: "STIX Two Text", "Cambria Math", serif; }}

  /* Divider between the verification sequence tabs and the tool tabs. */
  [role="tablist"] > [data-testid="stTab"]:nth-child({SEQUENCE_TAB_COUNT + 1}) {{
    margin-left: 1.5rem;
    position: relative;
    overflow: visible;
  }}
  [role="tablist"] > [data-testid="stTab"]:nth-child({SEQUENCE_TAB_COUNT + 1})::before {{
    content: "";
    position: absolute;
    left: -1rem;
    top: 20%;
    bottom: 20%;
    border-left: 1px solid #A7B8C7;
  }}
  /* Nested tab bars (inside a tab panel) are not grouped. */
  [role="tabpanel"] [role="tablist"] > [data-testid="stTab"] {{ margin-left: 0; }}
  [role="tabpanel"] [role="tablist"] > [data-testid="stTab"]::before {{ content: none; }}

  /* Keyboard focus stays clearly visible on every control. */
  button:focus-visible, [data-testid="stTab"]:focus-visible, a:focus-visible,
  input:focus-visible, textarea:focus-visible {{
    outline: 2px solid #1F4E8C !important;
    outline-offset: 2px;
  }}
</style>
"""


def inject_theme_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def status_label(status: Any) -> str:
    """Plain-language label for a status value (enum or string)."""
    key = getattr(status, "value", status)
    key = str(key).upper().replace(" ", "_")
    return STATUS_STYLE.get(key, (str(status).title(), "", "", ""))[0]


def status_badge_markdown(status: Any) -> str:
    """Colored badge in Streamlit markdown, using the theme's status colors."""
    key = str(getattr(status, "value", status)).upper().replace(" ", "_")
    color = {"PASS": "green", "FAIL": "red", "INCONCLUSIVE": "orange"}.get(key, "gray")
    return f":{color}-badge[{status_label(status)}]"


def _verdict_from_counts(counts: Dict[str, int]) -> str:
    if counts.get("FAIL", 0) > 0:
        return "FAIL"
    if counts.get("INCONCLUSIVE", 0) > 0:
        return "INCONCLUSIVE"
    if counts.get("PASS", 0) > 0:
        return "PASS"
    return "NOT_RUN"


def _summary_from_engine(report: Any) -> Dict[str, Any]:
    counts = report.summary_counts or {}
    first_fail = next((c for c in report.checks if getattr(c.status, "value", c.status) == "FAIL"), None)
    return {
        "title": report.proposal_title,
        "status": _verdict_from_counts(counts),
        "comparisons": report.outcome_comparisons or [],
        "fidelity": report.fidelity_score,
        "passed": counts.get("PASS", 0),
        "total": sum(counts.values()),
        "fail_reason": (first_fail.error_message or first_fail.details) if first_fail else None,
        "missing": list(report.missing_information or []),
    }


def _summary_from_agents(state: Dict[str, Any]) -> Dict[str, Any]:
    spec = state.get("proposal_spec")
    checks = state.get("verification_checks") or []
    if state.get("requires_human_review") and not state.get("overall_status"):
        status = "INCONCLUSIVE"
    else:
        status = str(state.get("overall_status") or "NOT_RUN").upper()
    passed = sum(1 for c in checks if getattr(c.status, "value", c.status) == "PASS")
    reasons = state.get("fail_reasons") or []
    missing = list(state.get("missing_parameters") or [])
    if state.get("requires_human_review") and state.get("human_review_reason"):
        missing.insert(0, state["human_review_reason"])
    return {
        "title": getattr(spec, "title", None) or "Untitled proposal",
        "status": status,
        "comparisons": state.get("outcome_comparisons") or [],
        "fidelity": state.get("fidelity_score"),
        "passed": passed,
        "total": len(checks),
        "fail_reason": reasons[0] if reasons else None,
        "missing": missing,
    }


def _shorten(text: str, limit: int = 160) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _note(summary: Dict[str, Any]) -> str:
    comparisons = summary["comparisons"]
    parts = []
    if comparisons:
        worst = max(comparisons, key=lambda c: c.absolute_difference)
        if all(c.is_match for c in comparisons):
            parts.append(
                "Every expected probability matches the simulation to within "
                f"{worst.absolute_difference:.3f} (allowed {worst.tolerance:.3f})."
            )
        else:
            parts.append(
                "Largest gap between expected and simulated probability: "
                f"{worst.absolute_difference:.3f} on "
                f'<span class="qv-ket-inline">|{escape(worst.basis_state)}⟩</span> '
                f"(allowed {worst.tolerance:.3f})."
            )
    elif summary["status"] == "FAIL" and summary["fail_reason"]:
        parts.append(escape(_shorten(summary["fail_reason"])))
    elif summary["missing"]:
        parts.append("Can't compare yet. " + escape(_shorten(summary["missing"][0])))
    else:
        parts.append("The proposal states no expected probabilities, so there is nothing to compare.")
    if summary["total"]:
        parts.append(f'{summary["passed"]} of {summary["total"]} checks passed.')
    return " ".join(parts)


def render_verdict_header(container: Any, report: Any = None, agent_state: Optional[Dict[str, Any]] = None) -> None:
    """Draw the verdict header into a placeholder created at the top of the page."""
    if report is not None:
        summary = _summary_from_engine(report)
    elif agent_state is not None:
        summary = _summary_from_agents(agent_state)
    else:
        summary = None

    if summary is None:
        html = (
            '<header class="qv-head">'
            '<div class="qv-title">Quantum proposal verifier</div>'
            '<div class="qv-row"><div class="qv-ket qv-empty">'
            '<span class="qv-bracket">⟨</span><span class="qv-label">proposal</span>'
            '<span class="qv-bracket">|</span><span class="qv-label">simulation</span>'
            '<span class="qv-bracket">⟩</span>² <span class="qv-eq">= <span class="qv-value">?</span></span></div></div>'
            '<div class="qv-note">Add a proposal under Input and verify it. The proposal is simulated '
            "exactly and compared with what its authors expect, and the verdict appears here.</div>"
            "</header>"
        )
        container.markdown(html, unsafe_allow_html=True)
        return

    label, _fill, _text_on_page, text_color = STATUS_STYLE.get(summary["status"], STATUS_STYLE["NOT_RUN"])
    overlap = summary["fidelity"]
    value = f"{overlap:.3f}" if overlap is not None else "–"
    empty_class = "" if overlap is not None else " qv-empty"

    html = (
        '<header class="qv-head">'
        f'<div class="qv-title">{escape(summary["title"])}</div>'
        '<div class="qv-row">'
        f'<div class="qv-ket{empty_class}" '
        'title="Fidelity between the expected and simulated distributions: (Σ √(p·q))²">'
        '<span class="qv-bracket">⟨</span><span class="qv-label">proposal</span>'
        '<span class="qv-bracket">|</span><span class="qv-label">simulation</span>'
        f'<span class="qv-bracket">⟩</span>² <span class="qv-eq">= <span class="qv-value">{value}</span></span></div>'
        f'<div class="qv-verdict" style="color:{text_color}" role="status">{label}</div>'
        "</div>"
        f'<div class="qv-note">{_note(summary)}</div>'
        "</header>"
    )
    container.markdown(html, unsafe_allow_html=True)
