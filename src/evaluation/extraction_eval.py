"""
Extraction evaluation.

1. Corpus statistics (automatic): statements per paper by kind; the share of model statements whose quoted evidence
   was found in the source; per-field coverage. Describes what the analyser produced, not whether it is right.
2. Verifier stress test (automatic, objective): the excerpt verifier is the safeguard against fabricated quotes.
   For every verified quote in the analysed corpus, it is fed
     genuine variants      the quote itself, with collapsed whitespace, lower-cased, with accents stripped
     fabricated variants   one number changed, a negation inserted, a platform/method term swapped for another,
                           and a quote taken from a different paper
   against the quote's own source text. Detection rate = fabricated variants rejected; false-rejection rate =
   genuine variants rejected.
3. Field accuracy and attribution accuracy need human annotation: see the labelling sheet; not measured until
   rows are labelled.
"""

import random
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

from sqlalchemy import select

from ..knowledge_base.ontology import CONCEPTS, find_concepts
from ..research_analysis.paper_analyzer import PaperAnalyzer, load_analysis, normalize_for_match, verify_excerpt
from ..storage import Publication, session_scope

NEGATE = re.compile(r"\b(is|are|was|were|can|does|do|has|have)\b", re.I)


def _analysed_ids(db_url: Optional[str]) -> List[int]:
    with session_scope(db_url) as s:
        return [p.id for p in s.scalars(select(Publication).where(
            Publication.analysed_at.is_not(None), Publication.is_synthetic.is_(False),
            Publication.discovered_via == "collection", Publication.curation_status != "excluded").order_by(Publication.id))]


def corpus_statistics(db_url: Optional[str]) -> Dict[str, Any]:
    recs = [r for r in (load_analysis(pid, db_url) for pid in _analysed_ids(db_url)) if r]
    kinds, fields, papers_with = Counter(), Counter(), defaultdict(set)
    model_total = model_verified = 0
    for r in recs:
        for st in r.statements:
            kinds[st.kind] += 1
            fields[st.field] += 1
            papers_with[st.field].add(r.publication_id)
            if st.extractor not in ("rule-based", "ontology"):
                model_total += 1
                model_verified += bool(st.provenance and st.provenance.verified)
    return {"papers": len(recs), "from_full_text": sum(r.analysed_from == "full_text" for r in recs),
            "statements": sum(kinds.values()), "by_kind": dict(kinds),
            "model_statements": model_total,
            "model_evidence_verified_share": round(model_verified / model_total, 3) if model_total else None,
            "papers_with_field": {f: len(v) for f, v in sorted(papers_with.items(), key=lambda x: -len(x[1]))}}


def _variants(q: str, other: Optional[str], rng: random.Random) -> List[tuple]:
    out = [("genuine", "as quoted", q), ("genuine", "whitespace collapsed", " ".join(q.split())),
           ("genuine", "lower-cased", q.lower()),
           ("genuine", "accents stripped", "".join(c for c in unicodedata.normalize("NFKD", q) if not unicodedata.combining(c)))]
    nums = list(re.finditer(r"\d+(?:\.\d+)?", q))
    if nums:
        m = nums[0]
        new = str(int(float(m.group()) * 3 + 7)) if "." not in m.group() else f"{float(m.group()) * 3 + 0.7:.2f}"
        out.append(("fabricated", "number changed", q[:m.start()] + new + q[m.end():]))
    m = NEGATE.search(q)
    if m:
        out.append(("fabricated", "negation inserted", q[:m.end()] + " not" + q[m.end():]))
    hits = [h for h in find_concepts(q) if h.category in ("platform", "method", "noise")]
    if hits:
        h = hits[0]
        alt = rng.choice([c for c, (cat, _, _) in CONCEPTS.items() if cat == h.category and c != h.concept])
        out.append(("fabricated", "term swapped", q[:h.start] + alt + q[h.end:]))
    if other:
        out.append(("fabricated", "quote from another paper", other))
    return out


def verifier_stress_test(db_url: Optional[str], seed: int = 7, max_quotes: int = 300) -> Dict[str, Any]:
    rng = random.Random(seed)
    analyzer = PaperAnalyzer(None, db_url)
    quotes = []                                           # (pid, quote)
    for pid in _analysed_ids(db_url):
        rec = load_analysis(pid, db_url)
        for st in rec.statements if rec else []:
            if st.provenance and st.provenance.verified and len(st.provenance.excerpt) >= 40:
                quotes.append((pid, st.provenance.excerpt))
    quotes = quotes[:max_quotes]
    sources: Dict[int, tuple] = {}
    results = defaultdict(lambda: [0, 0])                 # variant -> [accepted, total]
    misses: List[Dict[str, str]] = []
    for i, (pid, q) in enumerate(quotes):
        if pid not in sources:
            with session_scope(db_url) as s:
                kind, text, pages = analyzer._source(s.get(Publication, pid))
            sources[pid] = (normalize_for_match(text), [(n, normalize_for_match(t)) for n, t in pages])
        norm, pages = sources[pid]
        others = [x for p, x in quotes if p != pid]
        for group, name, text in _variants(q, rng.choice(others) if others else None, rng):
            ok = verify_excerpt(text, norm, pages)[0]
            results[(group, name)][0] += ok
            results[(group, name)][1] += 1
            if group == "fabricated" and ok and len(misses) < 12:
                misses.append({"variant": name, "text": text[:220]})
    gen = [v for (g, _), v in results.items() if g == "genuine"]
    fab = [v for (g, _), v in results.items() if g == "fabricated"]
    acc = lambda vs: sum(a for a, _ in vs)
    tot = lambda vs: sum(t for _, t in vs)
    return {"quotes": len(quotes), "papers": len(sources),
            "false_rejection_rate": round(1 - acc(gen) / tot(gen), 3) if tot(gen) else None,
            "detection_rate": round(1 - acc(fab) / tot(fab), 3) if tot(fab) else None,
            "by_variant": {f"{g}: {n}": {"accepted": a, "total": t, "accepted_share": round(a / t, 3)}
                           for (g, n), (a, t) in sorted(results.items())},
            "accepted_fabrications_examples": misses}
