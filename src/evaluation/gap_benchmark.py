"""
Gap-finding benchmark: runs the full research pipeline on the synthetic sample papers,
whose gaps were planted deliberately (data/sample_papers/planted_gaps.json), and reports:

- planted-gap recall   share of planted gaps that the finder recovered
- precision proxy      share of found gaps that match a planted gap (extra gaps are not
                       necessarily wrong, so this is a lower bound)
- evidence coverage    share of found gaps with at least one verbatim quote
- first-pass rate      share of plans that verify on the first attempt
- after-revision rate  share of plans that verify after at most one revision
"""

import json
import os
from typing import Dict, List

from ..schemas.research_gap import ResearchGap

PLANTED_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                            "data", "sample_papers", "planted_gaps.json")


def load_planted_gaps(path: str = PLANTED_PATH) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)["gaps"]


def gap_matches(gap: ResearchGap, planted: Dict) -> bool:
    if gap.gap_type not in planted["types"]:
        return False
    text = f"{gap.title} {gap.statement}".lower()  # the gap's own wording, not its evidence quotes
    return all(any(alt.lower() in text for alt in group) for group in planted["keywords"])


def match_planted(gaps: List[ResearchGap], planted: List[Dict]) -> Dict[str, List[str]]:
    return {p["id"]: [g.gap_id for g in gaps if gap_matches(g, p)] for p in planted}


def run_gap_benchmark(pipeline) -> Dict[str, float]:
    """Run papers -> gaps -> plans -> verification on an already-loaded pipeline."""
    planted = load_planted_gaps()
    gaps = pipeline.find_gaps()
    matches = match_planted(gaps, planted)
    matched_gap_ids = {gid for ids in matches.values() for gid in ids}

    first_pass = after_revision = 0
    for gap in gaps:
        plan = pipeline.plan(gap.gap_id)
        attempts = pipeline.history[plan.plan_id]
        first_pass += attempts[0][0].verification_status == "pass"
        after_revision += plan.verification_status == "pass"

    n = max(len(gaps), 1)
    return {
        "papers": len(pipeline.library),
        "gaps_found": len(gaps),
        "planted_gaps": len(planted),
        "planted_recall": sum(1 for ids in matches.values() if ids) / len(planted),
        "precision_lower_bound": len(matched_gap_ids) / n,
        "evidence_coverage": sum(1 for g in gaps if g.evidence) / n,
        "plans_first_pass_rate": first_pass / n,
        "plans_pass_after_revision_rate": after_revision / n,
        "matches": matches,
    }
