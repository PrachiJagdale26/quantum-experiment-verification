"""
Gap evaluation.

Automatic (describes the candidate set; says nothing about scientific value):
  - evidence grounding: share of gaps whose supporting evidence is verified quotes, vs. corpus inferences
  - specificity: share of gaps naming at least two core research dimensions of their own (not borrowed context)
  - near-duplicates: pairs of gaps whose statements have embedding cosine similarity >= 0.85
  - novelty outcomes for the gaps that were checked
Gap quality (relevance, evidence support, duplicates as judged by an expert) needs the review sheet; not measured
until rows are labelled.
"""

from collections import Counter
from typing import Any, Callable, Dict, Optional

import numpy as np

from ..gap_discovery import load_assessment, load_gaps
from ..gap_discovery.schema import CORE_DIMENSIONS


def evaluate_gaps(db_url: Optional[str], embed: Optional[Callable] = None, dup_threshold: float = 0.85) -> Dict[str, Any]:
    gaps = load_gaps(db_url)
    if not gaps:
        return {"gaps": 0}
    own_dims = lambda g: [d for d in CORE_DIMENSIONS if d in g.dimensions and g.dimensions[d].values
                          and g.dimensions[d].source != "paper_context"]
    quoted = [g for g in gaps if any(e.kind in ("direct", "interpretation") for e in g.evidence)]
    classes = Counter(load_assessment(g.id, db_url)["classification"] if load_assessment(g.id, db_url) else "not checked"
                      for g in gaps)
    dups = []
    if embed is not None and len(gaps) > 1:
        v = np.asarray(embed([g.statement for g in gaps]))
        sims = v @ v.T
        for i in range(len(gaps)):
            for j in range(i + 1, len(gaps)):
                if sims[i, j] >= dup_threshold:
                    dups.append({"a": f"CG-{gaps[i].id}", "b": f"CG-{gaps[j].id}", "similarity": round(float(sims[i, j]), 3)})
    return {"procedure": __doc__.strip(), "gaps": len(gaps), "by_origin": dict(Counter(g.origin for g in gaps)),
            "evidence_quoted_share": round(len(quoted) / len(gaps), 3),
            "specific_share": round(sum(len(own_dims(g)) >= 2 for g in gaps) / len(gaps), 3),
            "near_duplicate_pairs": dups, "novelty_outcomes": dict(classes),
            "researcher_review": dict(Counter(g.review_status for g in gaps))}
