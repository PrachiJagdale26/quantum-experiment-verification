"""
Multi-stage novelty investigation (Module 4D). Novelty is NOT 1 - similarity.

  1. Decompose   the gap's research dimensions (each tagged with where it came from)
  2. Queries     statement keywords, concept query, ontology synonym, method-focused, condition-focused
                 (optionally model-proposed alternatives, labelled llm_alternative); every query is stored
  3. Retrieve    OpenAlex, arXiv, Crossref + the local corpus + works citing the supporting papers; each
                 search is a SearchRun with its status, so failures lower coverage instead of vanishing
  4. Screen      de-duplicated publications (the ingestion merge rules), supporting papers removed,
                 date relative to the gap's evidence, abstract vs metadata-only recorded
  5. Compare     the closest papers per dimension: same / different / not stated, by deterministic ontology
                 and qubit-count matching with the sentence quoted; optionally a model judge on the research
                 question whose quote must be found in the abstract, otherwise the judgement is discarded
  6. Classify    deterministic rules over the comparisons and the coverage (see classify())
  7. Report      classification, confidence with its formula and inputs, coverage, queries, closest papers,
                 evidence for and against, remaining uncertainty. Always "in the searched literature".
"""

import logging
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sqlalchemy import select

from ..knowledge_base.ontology import CONCEPTS, expand_query, find_concepts
from ..literature.ingestion import LiteratureIngestor
from ..llm.json_utils import parse_llm_json
from ..research_analysis.paper_analyzer import normalize_for_match, qubit_counts, verify_excerpt
from ..storage import CandidateGap, DimensionComparison, NoveltyAssessment, Publication, SearchRun, session_scope
from .gap_generator import load_gap
from .schema import CLASSIFICATION_LABELS, CORE_DIMENSIONS, Gap

logger = logging.getLogger(__name__)

STOP = set("""a an the of in on for to and or with without by from at as is are was were be been being this that these those
it its their our we they which who whom whose not no than then into onto over under about between across beyond within
can could may might will would should must do does did done has have had such also only more most less least very
however while whereas although though because since thus therefore here there study studies paper work results result
show shows shown using use used based approach approaches method methods new further future remains remain open""".split())

JUDGE_PROMPT = """You compare a candidate research question with ONE published abstract.
Decide whether the abstract investigates the SAME research question (same object of study and same aim),
a DIFFERENT one, or whether the abstract does not say enough (NOT_STATED).
Quote, verbatim, the shortest span of the abstract that justifies your answer. If you cannot quote, answer NOT_STATED.
Return JSON only: {"result": "same" | "different" | "not_stated", "evidence": "<verbatim span from the abstract>"}"""


@dataclass
class CoveragePolicy:
    min_sources_ok: int = 2         # external sources that answered (ok or cached)
    min_queries: int = 3
    min_screened: int = 15          # distinct publications screened
    already_similarity: float = 0.55
    partial_similarity: float = 0.45


@dataclass
class Candidate:
    publication_id: int
    title: str
    abstract: Optional[str]
    year: Optional[int]
    doi: Optional[str]
    url: Optional[str]
    full_text: bool
    found_by: List[str]


def _sentence_at(text: str, pos: int) -> str:
    start = max(text.rfind(". ", 0, pos), text.rfind("? ", 0, pos), text.rfind("! ", 0, pos))
    start = 0 if start < 0 else start + 2
    ends = [i for i in (text.find(". ", pos), text.find("? ", pos), text.find("! ", pos)) if i >= 0]
    end = (min(ends) + 1) if ends else len(text)
    return text[start:end].strip()


def keywords(text: str, n: int = 12) -> str:
    # trailing digits are citation markers in extracted text ("phenomena26,27"), never part of the term
    words = [re.sub(r"\d+$", "", w) for w in re.findall(r"[A-Za-z][A-Za-z0-9\-]+", text)]
    words = [w for w in words if w.lower() not in STOP and len(w) > 2]
    seen, out = set(), []
    for w in words:
        if w.lower() not in seen:
            seen.add(w.lower())
            out.append(w)
    return " ".join(out[:n])


def compare_dimension(dim: str, gap_values: List[str], text: str) -> Tuple[str, Optional[str]]:
    """(same | different | not_stated, quoted sentence from the text) by deterministic matching."""
    if dim == "qubit_scale":
        counts = qubit_counts(text)
        if not counts:
            return "not_stated", None
        want = [int(v) for v in gap_values if str(v).isdigit()]
        hit = next((q for q in counts for w in want if abs(q - w) <= 0.5 * max(w, 1)), None)
        q = hit if hit is not None else counts[0]
        m = re.search(rf"\b{q}\b", text)
        return ("same" if hit is not None else "different"), (_sentence_at(text, m.start()) if m else None)
    hits = [h for h in find_concepts(text) if h.category == dim]
    match = next((h for h in hits if h.concept in gap_values), None)
    if match:
        return "same", _sentence_at(text, match.start)
    if hits:
        return "different", _sentence_at(text, hits[0].start)
    return "not_stated", None


class NoveltyChecker:
    def __init__(self, db_url: Optional[str] = None, ingestor: Optional[LiteratureIngestor] = None, llm=None,
                 embed: Optional[Callable] = None, sources: Sequence[str] = ("openalex", "arxiv", "crossref"),
                 per_query: int = 10, top_k: int = 8, max_queries: int = 5, judge_top: int = 3,
                 citations: bool = True, use_llm_queries: bool = False, policy: Optional[CoveragePolicy] = None):
        self.db_url = db_url
        self.ingestor = ingestor or LiteratureIngestor(db_url)
        self.llm, self._embed = llm, embed
        self.sources, self.per_query, self.top_k, self.max_queries = tuple(sources), per_query, top_k, max_queries
        self.judge_top, self.citations, self.use_llm_queries = judge_top, citations, use_llm_queries
        self.policy = policy or CoveragePolicy()
        self._vec_cache: Dict[int, np.ndarray] = {}

    @property
    def embed(self):
        if self._embed is None:
            from ..research_analysis.cross_paper_analysis import default_embedder
            self._embed = default_embedder()
        return self._embed

    @property
    def live_llm(self) -> bool:
        return self.llm is not None and not getattr(self.llm, "demo_mode", True)

    # ------------------------------------------------------------- 2. queries
    def build_queries(self, gap: Gap) -> List[Tuple[str, str, Tuple[str, ...]]]:
        """[(query, kind, sources)]. Long keyword queries skip arXiv, whose API requires every word to match."""
        core = gap.core()                                  # query on what defines the gap, not on borrowed context
        first = lambda d: core[d][0] if d in core else None
        names = lambda ds: [v for v in (first(d) for d in ds) if v]
        relevance_ranked = tuple(s for s in self.sources if s != "arxiv")
        out: List[Tuple[str, str, Tuple[str, ...]]] = []
        text = gap.research_question or gap.statement
        concept = " ".join(names(("phenomenon", "method", "platform", "noise")))
        if gap.origin != "unexplored_combination":          # that statement is our template sentence, not the authors' words
            out.append((keywords(text), "exact", relevance_ranked or self.sources))
            lead = first("phenomenon") or first("method")
            if lead and lead.lower() not in text.lower():
                out.append((f"{keywords(text, 6)} {lead}", "statement_context", relevance_ranked or self.sources))
        if concept:
            out.append((concept, "concepts", self.sources))
            syn = next((v for v, k in expand_query(concept) if k == "synonym"), None)
            if syn:
                out.append((syn, "synonym", self.sources))
        if first("method"):
            out.append((" ".join(names(("method", "platform")) or names(("method",))), "method", self.sources))
        cond = names(("phenomenon", "noise", "platform"))       # qubit counts make poor search terms
        if len(cond) >= 2:
            out.append((" ".join(cond), "condition", self.sources))
        if self.use_llm_queries and self.live_llm:
            out += [(q, "llm_alternative", relevance_ranked or self.sources) for q in self._llm_queries(text)]
        seen, uniq = set(), []
        for q, kind, srcs in out:
            if q and q.lower() not in seen:
                seen.add(q.lower())
                uniq.append((q, kind, srcs))
        return uniq[:self.max_queries]

    def _llm_queries(self, text: str) -> List[str]:
        try:
            raw = self.llm.complete("Propose two alternative literature-search queries (3-8 words each) using the "
                                    "terminology other research communities use for the same question. JSON only: "
                                    '{"queries": ["...", "..."]}', text, fallback_to_demo=False, purpose="novelty_queries")
            return [str(q) for q in (parse_llm_json(raw) or {}).get("queries", [])][:2]
        except Exception as exc:
            logger.warning("alternative queries failed: %s", exc)
            return []

    # ----------------------------------------------------------- 3. retrieval
    def _search(self, gap: Gap, aid: int, queries) -> Tuple[Dict[int, List[str]], List[dict]]:
        found: Dict[int, List[str]] = {}
        runs: List[dict] = []

        def record(outcome, label):
            runs.append({"source": outcome.source, "status": outcome.status, "returned": outcome.returned,
                         "error": outcome.error, "label": label})
            if outcome.search_run_id:
                with session_scope(self.db_url) as s:
                    for h in s.get(SearchRun, outcome.search_run_id).hits:
                        if h.publication_id:
                            found.setdefault(h.publication_id, []).append(f"{label} #{h.rank}")

        for q, kind, srcs in queries:
            for src in srcs:
                try:
                    res = self.ingestor.collector(src).search(q, limit=self.per_query)
                except Exception as exc:       # a collector bug must lower coverage, not abort the investigation
                    from ..literature.collectors.base import SearchResult
                    res = SearchResult(src, q, "failed", error=str(exc)[:300])
                record(self.ingestor.store_result(res, "novelty", {"limit": self.per_query}, query_kind=kind, assessment_id=aid),
                       f"{src}:{kind}")
        if self.citations and "openalex" in self.sources and gap.origin != "unexplored_combination":
            with session_scope(self.db_url) as s:
                oa = [p.openalex_id for p in s.scalars(select(Publication).where(Publication.id.in_(gap.supporting_ids)))
                      if p.openalex_id][:2]
            for oid in oa:
                try:
                    res = self.ingestor.collector("openalex").citing(oid, limit=self.per_query)
                except Exception as exc:
                    from ..literature.collectors.base import SearchResult
                    res = SearchResult("openalex", oid, "failed", error=str(exc)[:300])
                record(self.ingestor.store_result(res, "citation", {"cites": oid}, query_kind="citing", assessment_id=aid),
                       "openalex:citing")
        return found, runs

    def _local(self, gap_vec: np.ndarray, exclude: Iterable[int], k: int = 10) -> Dict[int, List[str]]:
        """Closest papers in the researcher's own corpus (collected, not excluded, not synthetic)."""
        exclude = set(exclude)
        with session_scope(self.db_url) as s:
            pubs = [(p.id, f"{p.title}. {p.abstract or ''}") for p in s.scalars(select(Publication).where(
                Publication.discovered_via == "collection", Publication.is_synthetic.is_(False),
                Publication.curation_status != "excluded")) if p.id not in exclude]
        if not pubs:
            return {}
        sims = self._vectors(pubs) @ gap_vec
        order = np.argsort(-sims)[:k]
        return {pubs[i][0]: [f"local corpus #{r + 1}"] for r, i in enumerate(order)}

    def _vectors(self, items: List[Tuple[int, str]]) -> np.ndarray:
        todo = [(pid, t) for pid, t in items if pid not in self._vec_cache]
        if todo:
            for (pid, _), v in zip(todo, self.embed([t for _, t in todo])):
                self._vec_cache[pid] = np.asarray(v)
        return np.vstack([self._vec_cache[pid] for pid, _ in items])

    # ------------------------------------------------------------- 5. compare
    def _judge(self, question: str, abstract: str) -> Optional[dict]:
        try:
            raw = self.llm.complete(JUDGE_PROMPT, f"Research question: {question}\n\n<UNTRUSTED_PAPER_TEXT>\n{abstract}\n"
                                    "</UNTRUSTED_PAPER_TEXT>", fallback_to_demo=False, purpose="novelty_judge")
            data = parse_llm_json(raw) or {}
        except Exception as exc:
            logger.warning("novelty judge failed: %s", exc)
            return None
        result, quote = str(data.get("result", "")).lower(), str(data.get("evidence") or "").strip()
        call = getattr(self.llm, "last_call", {}) or {}
        model = f"llm:{call.get('provider', '?')}/{call.get('model', '?')}"
        ok, score, _ = verify_excerpt(quote, normalize_for_match(abstract), []) if quote else (False, 0.0, None)
        return {"result": result if result in ("same", "different", "not_stated") else "not_stated",
                "quote": quote, "verified": bool(ok), "method": model}

    # ------------------------------------------------------------ 6. classify
    def classify(self, core: Dict[str, List[str]], closest: List[dict], coverage_ok: bool) -> Tuple[str, Optional[dict], str]:
        """
        Rules, in order:
          fewer than 2 core dimensions                                  -> insufficient_coverage (gap too unspecific)
          a paper the same on every core dimension, similarity >= already_similarity, and no verified model
            judgement saying the question differs                       -> already_investigated
          a verified model judgement "same question" with no core dimension different -> already_investigated
          a paper the same on at least half the core dimensions, similarity >= partial_similarity -> partially_investigated
          search coverage below the policy                              -> insufficient_coverage (never "novel")
          otherwise                                                     -> potentially_unexplored (in the searched literature)
        """
        P = self.policy
        if len(core) < 2:
            return "insufficient_coverage", None, "The gap names fewer than two research dimensions, so related work cannot be compared reliably."
        need_half = math.ceil(len(core) / 2)
        for c in closest:
            rq = c.get("research_question")
            rq_same = bool(rq and rq["verified"] and rq["result"] == "same")
            rq_diff = bool(rq and rq["verified"] and rq["result"] == "different")
            if c["same"] == len(core) and c["similarity"] >= P.already_similarity and not rq_diff:
                return "already_investigated", c, "Matches every core dimension of the gap."
            if rq_same and c["different"] == 0:
                return "already_investigated", c, "A verified judgement says this paper addresses the same question."
        for c in closest:
            if c["same"] >= need_half and c["similarity"] >= P.partial_similarity:
                why = (f"similarity {c['similarity']:.2f} is below the {P.already_similarity:.2f} needed for 'already investigated'"
                       if c["same"] == len(core) else f"{c['different']} differ, {c['not_stated']} not stated")
                return "partially_investigated", c, f"The closest paper matches {c['same']} of {len(core)} core dimensions; {why}."
        if not coverage_ok:
            return "insufficient_coverage", None, "Search coverage is below the policy, so absence of matches is not evidence."
        return "potentially_unexplored", None, "No close match among the screened publications."

    # ------------------------------------------------------------------- run
    def check(self, gap_id: int, exclude_local_ids: Iterable[int] = ()) -> dict:
        gap = load_gap(gap_id, self.db_url)
        if gap is None:
            raise ValueError(f"No gap {gap_id}")
        with session_scope(self.db_url) as s:
            a = NoveltyAssessment(gap_id=gap_id, classification="pending", confidence=0.0)
            s.add(a)
            s.flush()
            aid = a.id
        try:
            return self._run(gap, aid, set(exclude_local_ids))
        except Exception:
            with session_scope(self.db_url) as s:
                s.delete(s.get(NoveltyAssessment, aid))
            raise

    def _run(self, gap: Gap, aid: int, exclude_local: set) -> dict:
        P, core = self.policy, gap.core()
        text = gap.research_question or gap.statement
        gap_vec = np.asarray(self.embed([text])[0])
        queries = self.build_queries(gap)
        found, runs = self._search(gap, aid, queries)
        for pid, how in self._local(gap_vec, exclude_local | set(gap.supporting_ids)).items():
            found.setdefault(pid, []).extend(how)

        # ---- 4. screen
        support = set(gap.supporting_ids)
        with session_scope(self.db_url) as s:
            pubs = {p.id: p for p in s.scalars(select(Publication).where(Publication.id.in_(set(found) - support)))}
            evidence_year = max((p.year for p in s.scalars(select(Publication).where(Publication.id.in_(support))) if p.year),
                                default=None)
            cands = [Candidate(p.id, p.title, p.abstract, p.year, p.doi, p.source_url, bool(p.full_text_available), found[p.id])
                     for p in pubs.values()]
        external = [r for r in runs if r["label"].split(":")[1] not in ("citing",)]
        attempted = sorted({r["source"] for r in external})
        ok_sources = sorted({r["source"] for r in external if r["status"] in ("ok", "cached")})
        screened, with_abstract = len(cands), sum(1 for c in cands if c.abstract)
        coverage_ok = (len(ok_sources) >= P.min_sources_ok and len(queries) >= P.min_queries and screened >= P.min_screened)

        # ---- 5. compare the closest
        closest: List[dict] = []
        if cands:
            sims = self._vectors([(c.publication_id, f"{c.title}. {c.abstract or ''}") for c in cands]) @ gap_vec
            scored = []
            for c, sim in zip(cands, sims):
                body = f"{c.title}. {c.abstract or ''}"
                dims = {d: dict(zip(("result", "quote"), compare_dimension(d, v, body))) for d, v in core.items()}
                scored.append((c, float(sim), dims, sum(r["result"] == "same" for r in dims.values())))
            # Closest = most dimensions in common first (similarity breaks ties), plus the top semantic matches the
            # ontology may miss. Similarity alone lets generic titles crowd out the papers that share the question.
            by_dims = sorted(scored, key=lambda x: (-x[3], -x[1]))
            by_sim = sorted(scored, key=lambda x: -x[1])[:3]
            pick, seen = [], set()
            for x in by_dims[:max(1, self.top_k - 3)] + by_sim + by_dims:
                if x[0].publication_id not in seen and len(pick) < self.top_k:
                    seen.add(x[0].publication_id)
                    pick.append(x)
            ranked = sorted(pick, key=lambda x: (-x[3], -x[1]))
            for i, (c, sim, dims, _) in enumerate(ranked):
                entry = {"publication_id": c.publication_id, "title": c.title, "year": c.year, "doi": c.doi, "url": c.url,
                         "availability": "full_text" if c.full_text else ("abstract" if c.abstract else "metadata_only"),
                         "similarity": round(float(sim), 3), "found_by": c.found_by[:6],
                         "after_evidence": bool(evidence_year and c.year and c.year >= evidence_year),
                         "dimensions": dims, "research_question": None,
                         "same": sum(d["result"] == "same" for d in dims.values()),
                         "different": sum(d["result"] == "different" for d in dims.values()),
                         "not_stated": sum(d["result"] == "not_stated" for d in dims.values())}
                if self.live_llm and i < self.judge_top and c.abstract:
                    entry["research_question"] = self._judge(text, c.abstract)
                closest.append(entry)

        # ---- 6. classify
        classification, decided_by, reason = self.classify(core, closest, coverage_ok)

        # ---- 7. confidence (formula stored with its inputs)
        c_sources = len(ok_sources) / max(1, len(attempted) or len(self.sources))
        c_screen = min(1.0, screened / (2 * P.min_screened))
        c_text = (sum(1 for c in closest if c["availability"] != "metadata_only") / len(closest)) if closest else 0.0
        coverage_score = round((c_sources + c_screen + c_text) / 3, 3)
        if classification in ("already_investigated", "partially_investigated") and decided_by:
            e = decided_by["similarity"] * (decided_by["same"] / max(1, len(core)))
        elif classification == "potentially_unexplored":
            e = 1.0 - (closest[0]["similarity"] if closest else 0.0)
        else:
            e = 1.0 - coverage_score
        agreement = 1.0
        if decided_by and decided_by.get("research_question") and decided_by["research_question"]["verified"]:
            j = decided_by["research_question"]["result"]
            agreement = 0.6 if (j == "different" and classification == "already_investigated") else 1.0
        confidence = round(max(0.0, min(1.0, coverage_score * (0.5 + 0.5 * e) * agreement)), 2)

        judged = [c["research_question"] for c in closest if c.get("research_question")]
        coverage = {"sources_attempted": attempted, "sources_ok": ok_sources,
                    "sources_failed": sorted({r["source"] for r in external if r["status"] not in ("ok", "cached")}),
                    "queries": [{"query": q, "kind": k, "sources": list(srcs)} for q, k, srcs in queries],
                    "runs": runs, "screened": screened, "with_abstract": with_abstract,
                    "screened_ids": sorted(c.publication_id for c in cands),
                    "citation_runs": sum(1 for r in runs if r["label"].endswith("citing")),
                    "local_corpus": True, "coverage_ok": coverage_ok, "policy": asdict(P),
                    "excluded_supporting": sorted(support), "excluded_local": sorted(exclude_local)}
        findings = {"closest": closest, "decided_by": decided_by["publication_id"] if decided_by else None, "reason": reason,
                    "core_dimensions": core,
                    "dimension_sources": {d: gap.dimensions[d].source for d in core},
                    "evidence_against_novelty": [f"{c['title']} ({c['year'] or 'n.d.'}): same on {c['same']} of {len(core)} dimensions, "
                                                 f"similarity {c['similarity']:.2f}" for c in closest if c["same"] >= max(1, len(core) - 1)][:5],
                    "evidence_for_novelty": ([f"No screened paper matches more than {max((c['same'] for c in closest), default=0)} "
                                              f"of {len(core)} core dimensions."] if classification == "potentially_unexplored" else []),
                    "judgements": {"made": len(judged), "discarded_unverified": sum(1 for j in judged if not j["verified"])}}
        uncertainty = self._uncertainty(classification, coverage, closest, gap)
        basis = {"formula": "confidence = coverage x (0.5 + 0.5 x evidence) x agreement;  coverage = mean(sources answered / "
                            "attempted, min(1, screened / (2 x min_screened)), share of compared papers with an abstract)",
                 "coverage": coverage_score, "sources_term": round(c_sources, 3), "screen_term": round(c_screen, 3),
                 "text_term": round(c_text, 3), "evidence": round(float(e), 3), "agreement": agreement}

        with session_scope(self.db_url) as s:
            a = s.get(NoveltyAssessment, aid)
            a.classification, a.confidence, a.confidence_basis = classification, confidence, basis
            a.coverage, a.findings, a.uncertainty = coverage, findings, uncertainty
            a.searched_at = datetime.now(timezone.utc)
            for c in closest:
                for d, r in c["dimensions"].items():
                    a.comparisons.append(DimensionComparison(publication_id=c["publication_id"], dimension=d, result=r["result"],
                                                             evidence_quote=r["quote"], quote_verified=r["quote"] is not None,
                                                             method="rule"))
                j = c.get("research_question")
                if j:
                    a.comparisons.append(DimensionComparison(publication_id=c["publication_id"], dimension="research_question",
                                                             result=j["result"] if j["verified"] else "not_stated",
                                                             evidence_quote=j["quote"], quote_verified=j["verified"], method=j["method"]))
            s.get(CandidateGap, gap.id).classification = classification
        return load_assessment(gap.id, self.db_url)

    @staticmethod
    def _uncertainty(classification: str, coverage: dict, closest: List[dict], gap: Gap) -> str:
        notes = ["Searched literature only: OpenAlex, arXiv and Crossref relevance search plus this corpus; "
                 "a result here never means the question has never been published."]
        if coverage["sources_failed"]:
            notes.append(f"Sources that failed or were rate-limited: {', '.join(coverage['sources_failed'])}.")
        notes.append("Comparisons use titles and abstracts, not full texts; details only in a paper body are missed.")
        if any(c["availability"] == "metadata_only" for c in closest):
            notes.append("Some close papers have no abstract and were compared on their title only.")
        ctx = [d for d, v in gap.dimensions.items() if v.source == "paper_context" and d in CORE_DIMENSIONS]
        if ctx:
            notes.append(f"Dimensions taken from the source paper rather than the gap statement: {', '.join(ctx)}.")
        if not any(c.get("research_question") for c in closest):
            notes.append("No model judged the research question; dimension matching is by ontology terms only.")
        return " ".join(notes)


def load_assessment(gap_id: int, db_url: Optional[str] = None) -> Optional[dict]:
    with session_scope(db_url) as s:
        a = s.scalars(select(NoveltyAssessment).where(NoveltyAssessment.gap_id == gap_id,
                                                      NoveltyAssessment.classification != "pending")
                      .order_by(NoveltyAssessment.searched_at.desc(), NoveltyAssessment.id.desc())).first()
        if a is None:
            return None
        return {"id": a.id, "gap_id": gap_id, "classification": a.classification,
                "label": CLASSIFICATION_LABELS.get(a.classification, a.classification), "confidence": a.confidence,
                "confidence_basis": a.confidence_basis, "coverage": a.coverage, "findings": a.findings,
                "uncertainty": a.uncertainty, "searched_at": a.searched_at.isoformat() if a.searched_at else None}
