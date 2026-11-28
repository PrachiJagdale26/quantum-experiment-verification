"""
GapFinder: stage 2 of the pipeline. Finds research gaps across a set of analysed papers.

Hybrid design so it works with or without an LLM:
  1. Explicit signals   - each paper's stated limitations, future work and open questions.
  2. Coverage matrix    - circuit families x hardware platforms: combinations no paper has tried.
  3. Scale limits       - the largest register any paper reached for a scalable circuit family.
  4. Conflicts          - papers reporting different outcome distributions for the same circuit.
  5. LLM synthesis      - optional: rewrites / merges candidates into crisp statements (never invents evidence).
  6. Scoring            - novelty (vs. the knowledge base), evidence strength, feasibility -> priority.
"""

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from ..knowledge_base.vector_store import QuantumKnowledgeBase
from ..llm.client import ConfigurableLLMClient
from ..llm.json_utils import parse_llm_json
from ..llm.prompts import GAP_IDENTIFICATION_PROMPT
from ..schemas.research_gap import GapEvidence, ResearchGap
from ..schemas.research_paper import ResearchPaperAnalysis
from .domain import (FAMILY_LABELS, TEMPLATE_FAMILIES, experiment_family, families_in,
                     paper_family, paper_system, qubits_in, systems_in)

logger = logging.getLogger(__name__)

TYPE_PRECEDENCE = ["conflicting_results", "scale_limit", "unexplored_combination", "future_work", "stated_limitation"]
SCALABLE_FAMILIES = {"ghz", "qft", "grover", "qaoa"}
_STOP = set("the a an and or of to in on for with by is are was were be been this that these those we our it its as at from "
            "not any all can will may into than then same only also which such there their has have had does did".split())
_SCALE_CUE = r"\b(up to \d+ qubits|beyond \d+ qubits|\d+ or more qubits|limited to (?:\w+-)?(?:two|three|four|\d+)[- ]qubit|larger registers?|scal(?:e|ing) (?:to|beyond))"


@dataclass
class _Candidate:
    gap_type: str
    title: str
    statement: str
    evidence: List[GapEvidence]
    family: Optional[str] = None
    systems: Set[str] = field(default_factory=set)
    qubits: Optional[int] = None
    hints: Dict[str, str] = field(default_factory=dict)
    cid: str = ""


def _tokens(text: str) -> Set[str]:
    return {w[:6] for w in re.findall(r"[a-z]+", (text or "").lower()) if len(w) > 3 and w not in _STOP}


def _jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    return len(ta & tb) / len(ta | tb) if ta and tb else 0.0


def _clean_title(sentence: str) -> str:
    s = re.sub(r"^(it (?:remains|is) (?:open|unknown|unclear) (?:whether|how|if)|we (?:only )?(?:demonstrate|did not|do not|have not)|"
               r"our [a-z]+ is|all runs are|the [a-z]+ (?:is|are|does not))\s+", "", sentence.strip(), flags=re.I)
    s = re.sub(r"\s+(?:is|are) left for future work\.?$|\s+will be (?:explored|investigated|addressed) in future work\.?$", "", s, flags=re.I)
    s = s.rstrip(". ")
    s = s[0].upper() + s[1:] if s else sentence
    return s if len(s) <= 90 else s[:87].rsplit(" ", 1)[0] + "…"


class GapFinder:
    def __init__(self, llm_client: Optional[ConfigurableLLMClient] = None, kb: Optional[QuantumKnowledgeBase] = None):
        self.llm = llm_client
        self.kb = kb

    # ================================================================== public
    def find(self, papers: Sequence[ResearchPaperAnalysis], max_gaps: int = 12) -> List[ResearchGap]:
        papers = [p for p in papers if p is not None]
        if not papers:
            return []
        # Deterministic detectors first so their clear titles survive merging with raw sentences.
        cands = (self._conflicts(papers) + self._scale(papers) + self._coverage(papers) + self._explicit(papers))
        cands = self._merge(cands)
        for i, c in enumerate(cands, 1):
            c.cid = f"C-{i:02d}"
        cands = self._llm_refine(cands, papers)
        gaps = [self._score(c, papers) for c in cands]
        gaps.sort(key=lambda g: g.priority, reverse=True)
        gaps = gaps[:max_gaps]
        for i, g in enumerate(gaps, 1):
            g.gap_id = f"GAP-{i:02d}"
        return gaps

    # ============================================================ 1. explicit
    def _explicit(self, papers: Sequence[ResearchPaperAnalysis]) -> List[_Candidate]:
        out: List[_Candidate] = []
        for paper in papers:
            p_fam, p_sys = paper_family(paper), paper_system(paper)
            for field_name, base_type in (("future_work", "future_work"), ("open_questions", "future_work"),
                                          ("limitations", "stated_limitation")):
                for sentence in getattr(paper, field_name, []) or []:
                    fams = families_in(sentence)
                    systems = set(systems_in(sentence))
                    qs = qubits_in(sentence)
                    fam = fams[0] if fams else p_fam
                    gtype = base_type
                    qubits = max(qs) if qs else None
                    converted = False
                    if re.search(_SCALE_CUE, sentence, re.I):
                        if fam == "bell" and p_sys:
                            # "Limited to two-qubit states" on a platform = multi-qubit entanglement missing there.
                            fam, gtype, systems, qubits, converted = "ghz", "unexplored_combination", {p_sys}, 3, True
                        else:
                            gtype = "scale_limit"
                    # Target platform: a system named in the sentence other than the paper's own, or the
                    # paper's own device when the sentence is about a different circuit family.
                    other = systems - ({p_sys} if p_sys else set())
                    if gtype != "scale_limit" and not converted:
                        if other:
                            systems = other
                        elif fams and fam != p_fam and p_sys:
                            systems = {p_sys}
                        else:
                            systems = set()
                    hints = {"source_paper": paper.title, **({"system": p_sys} if p_sys else {})}
                    if systems and gtype != "scale_limit":
                        in_order = [s for s in systems_in(sentence) if s in systems]
                        hints["target_system"] = (in_order or sorted(systems))[0]
                    out.append(_Candidate(
                        gap_type=gtype, title=_clean_title(sentence), statement=sentence,
                        evidence=[self._evidence(paper, sentence)], family=fam if (fams or gtype != "stated_limitation") else None,
                        systems=systems, qubits=qubits, hints=hints,
                    ))
        return out

    # ============================================================ 2. coverage
    def _coverage(self, papers: Sequence[ResearchPaperAnalysis]) -> List[_Candidate]:
        tried: Dict[str, Dict[str, List[ResearchPaperAnalysis]]] = defaultdict(lambda: defaultdict(list))
        corpus_systems: Dict[str, List[ResearchPaperAnalysis]] = defaultdict(list)
        for paper in papers:
            sys_ = paper_system(paper)
            if not sys_:
                continue
            corpus_systems[sys_].append(paper)
            fams = {experiment_family(e, paper) for e in paper.experiments} - {None} or ({paper_family(paper)} - {None})
            for fam in fams:
                tried[fam][sys_].append(paper)
        out = []
        for fam, by_sys in tried.items():
            for sys_, users in corpus_systems.items():
                if sys_ in by_sys:
                    continue
                demo_paper = next(iter(by_sys.values()))[0]
                label = FAMILY_LABELS.get(fam, fam)
                evidence = [self._evidence(p, f"{label} demonstrated on {s}: {p.title}") for s, ps in by_sys.items() for p in ps[:1]]
                evidence.append(self._evidence(users[0], f"{sys_} hardware is used in: {users[0].title}"))
                out.append(_Candidate(
                    gap_type="unexplored_combination",
                    title=f"{label} on {sys_.lower()} hardware",
                    statement=(f"{label} preparation has been reported on {', '.join(sorted(by_sys))} but not on {sys_} "
                               f"hardware, even though {sys_} devices are used elsewhere in this literature."),
                    evidence=evidence, family=fam, systems={sys_},
                    hints={"source_paper": demo_paper.title, "target_system": sys_,
                           **({"system": paper_system(demo_paper)} if paper_system(demo_paper) else {})},
                ))
        return out[:8]

    # =============================================================== 3. scale
    def _scale(self, papers: Sequence[ResearchPaperAnalysis]) -> List[_Candidate]:
        best: Dict[str, Tuple[int, ResearchPaperAnalysis]] = {}
        for paper in papers:
            for exp in paper.experiments:
                fam = experiment_family(exp, paper)
                if fam in SCALABLE_FAMILIES and exp.num_qubits:
                    if fam not in best or exp.num_qubits > best[fam][0]:
                        best[fam] = (exp.num_qubits, paper)
        out = []
        for fam, (n, paper) in best.items():
            if n >= 10:
                continue
            label = FAMILY_LABELS.get(fam, fam)
            out.append(_Candidate(
                gap_type="scale_limit", title=f"{label} beyond {n} qubits",
                statement=f"The largest {label} reported in this literature uses {n} qubits; behaviour at {n + 1} or more qubits is untested.",
                evidence=[self._evidence(paper, f"Largest reported {label}: {n} qubits ({paper.title})")],
                family=fam, qubits=n, hints={"source_paper": paper.title, "max_qubits": str(n),
                                            **({"system": paper_system(paper)} if paper_system(paper) else {})},
            ))
        return out

    # =========================================================== 4. conflicts
    def _conflicts(self, papers: Sequence[ResearchPaperAnalysis]) -> List[_Candidate]:
        groups: Dict[Tuple, List[Tuple[ResearchPaperAnalysis, Dict[str, float]]]] = defaultdict(list)
        for paper in papers:
            for exp in paper.experiments:
                if exp.gates and exp.expected_outcomes:
                    sig = tuple((g.name.upper(), tuple(g.target_qubits)) for g in exp.gates)
                    groups[sig].append((paper, {o.basis_state: o.expected_probability for o in exp.expected_outcomes}))
        out = []
        for sig, reports in groups.items():
            for i in range(len(reports)):
                for j in range(i + 1, len(reports)):
                    (pa, da), (pb, db) = reports[i], reports[j]
                    if pa is pb:
                        continue
                    keys = set(da) | set(db)
                    tvd = 0.5 * sum(abs(da.get(k, 0) - db.get(k, 0)) for k in keys)
                    if tvd <= 0.1:
                        continue
                    circuit = ", ".join(f"{n} {' '.join(map(str, q))}" for n, q in sig)
                    fmt = lambda d: ", ".join(f"P({k}) = {v:.2f}" for k, v in sorted(d.items()))
                    fam = experiment_family(next(e for e in pa.experiments if e.gates), pa)
                    label = FAMILY_LABELS.get(fam, "circuit")
                    out.append(_Candidate(
                        gap_type="conflicting_results",
                        title=f"Conflicting {label} statistics for the same circuit",
                        statement=(f"Two papers run the same circuit ({circuit}) but report different outcomes: "
                                   f"{fmt(da)} versus {fmt(db)} (TVD {tvd:.2f}). Which result reflects the circuit and which reflects device error is unresolved."),
                        evidence=[self._evidence(pa, self._claim_quote(pa, da)), self._evidence(pb, self._claim_quote(pb, db))],
                        family=fam, hints={"source_paper": pa.title, "other_paper": pb.title, "circuit": circuit,
                                           **({"system": paper_system(pb)} if paper_system(pb) else {})},
                    ))
        return out

    # =============================================================== merging
    def _merge(self, cands: List[_Candidate]) -> List[_Candidate]:
        merged: List[_Candidate] = []
        for c in cands:
            target = next((m for m in merged if self._same_gap(m, c)), None)
            if target is None:
                merged.append(c)
                continue
            quotes = {e.quote for e in target.evidence}
            target.evidence += [e for e in c.evidence if e.quote not in quotes]
            target.systems |= c.systems
            if TYPE_PRECEDENCE.index(c.gap_type) < TYPE_PRECEDENCE.index(target.gap_type):
                target.gap_type, target.title, target.statement = c.gap_type, c.title, c.statement
            target.qubits = max(filter(None, [target.qubits, c.qubits]), default=None)
            for k, v in c.hints.items():
                target.hints.setdefault(k, v)
        return merged

    @staticmethod
    def _same_gap(a: _Candidate, b: _Candidate) -> bool:
        if "conflicting_results" in (a.gap_type, b.gap_type):
            return a.gap_type == b.gap_type and a.hints.get("circuit") == b.hints.get("circuit")
        if a.family and a.family == b.family:
            a_scale, b_scale = a.gap_type == "scale_limit", b.gap_type == "scale_limit"
            if a_scale and b_scale:
                return True
            if not a_scale and not b_scale and a.systems and b.systems and (a.systems & b.systems):
                return True
        if a.family and b.family and a.family != b.family:
            return False  # different circuit families are different gaps, however similar the wording
        if not (a.systems or b.systems) and _jaccard(a.statement, b.statement) >= 0.3:
            return True
        return _jaccard(a.statement, b.statement) >= 0.5

    # ================================================================== LLM
    def _llm_refine(self, cands: List[_Candidate], papers: Sequence[ResearchPaperAnalysis]) -> List[_Candidate]:
        if not cands or self.llm is None or getattr(self.llm, "demo_mode", True):
            return cands
        payload = [{"id": c.cid, "type": c.gap_type, "statement": c.statement,
                    "evidence": [e.quote for e in c.evidence][:3]} for c in cands]
        summaries = "\n".join(f"- {p.title}: {p.research_objective}" for p in papers)
        user = (f"<UNTRUSTED_PAPER_TEXT>\n{summaries}\n</UNTRUSTED_PAPER_TEXT>\n\n"
                f"<CANDIDATE_GAPS>\n{json.dumps(payload, ensure_ascii=False)}\n</CANDIDATE_GAPS>\n\n"
                f"Merge duplicates and rewrite each gap as a clear research gap statement.")
        try:
            raw = self.llm.complete(system_prompt=GAP_IDENTIFICATION_PROMPT, user_prompt=user, temperature=0.0,
                                    fallback_to_demo=False, purpose="gap_refinement")
            data = parse_llm_json(raw)
            by_id = {c.cid: c for c in cands}
            refined: List[_Candidate] = []
            used: Set[str] = set()
            for g in data.get("gaps", []):
                ids = [i for i in g.get("candidate_ids", []) if i in by_id and i not in used]
                if not ids:
                    continue  # never accept a gap without deterministic evidence behind it
                base = by_id[ids[0]]
                for other in ids[1:]:
                    base = self._merge([base, by_id[other]])[0] if self._same_gap(base, by_id[other]) else base
                    base.evidence += [e for e in by_id[other].evidence if e not in base.evidence]
                used.update(ids)
                base.title = (g.get("title") or base.title)[:100]
                base.statement = g.get("statement") or base.statement
                if g.get("gap_type") in TYPE_PRECEDENCE:
                    base.gap_type = g["gap_type"]
                base.hints["generated_by"] = "llm"
                refined.append(base)
            refined += [c for c in cands if c.cid not in used]
            return refined
        except Exception as exc:
            logger.warning(f"LLM gap refinement failed ({exc}); using deterministic candidates.")
            return cands

    # =============================================================== scoring
    def _score(self, c: _Candidate, papers: Sequence[ResearchPaperAnalysis]) -> ResearchGap:
        n_papers = len({e.paper_title for e in c.evidence})
        evidence_strength = min(1.0, 0.35 + 0.25 * (n_papers - 1) + 0.1 * (len(c.evidence) - 1))

        family = c.family if c.family in TEMPLATE_FAMILIES else None
        has_circuit = any(e.gates for p in papers if p.title == c.hints.get("source_paper") for e in p.experiments)
        target_q = (c.qubits + 1) if (c.gap_type == "scale_limit" and c.qubits) else (c.qubits or 3)
        testable = bool(family or has_circuit) and target_q <= 10
        feasibility = 0.3 if not testable else (0.9 if target_q <= 5 else 0.7)

        novelty = self._novelty(c)
        priority = round(0.4 * novelty + 0.35 * evidence_strength + 0.25 * feasibility, 3)

        hints = dict(c.hints)
        if c.family:
            hints["family"] = c.family
        if c.systems and "target_system" not in hints:
            hints["target_system"] = sorted(c.systems)[0]
        if c.qubits:
            hints["qubits"] = str(c.qubits)
        return ResearchGap(
            gap_id=c.cid, title=c.title, statement=c.statement, gap_type=c.gap_type, evidence=c.evidence,
            novelty=round(novelty, 3), feasibility=round(feasibility, 3), evidence_strength=round(evidence_strength, 3),
            priority=priority, testable_on_simulator=testable, hints=hints,
        )

    def _novelty(self, c: _Candidate) -> float:
        """1 - similarity of the gap to literature *other than* the papers that reveal it."""
        if self.kb is None:
            return 0.5
        try:
            own = {e.source_filename for e in c.evidence if e.source_filename} | {e.paper_title for e in c.evidence}
            hits = self.kb.retrieve(c.statement, top_k=8)
            scores = [h.get("score", 0.0) for h in hits
                      if h.get("source_filename") not in own and h.get("title") not in own]
            return max(0.0, min(1.0, 1.0 - (max(scores) if scores else 0.0)))
        except Exception as exc:
            logger.warning(f"Novelty scoring failed: {exc}")
            return 0.5

    # ============================================================== helpers
    @staticmethod
    def _evidence(paper: ResearchPaperAnalysis, quote: str) -> GapEvidence:
        page = None
        probe = re.sub(r"\s+", " ", quote)[:50].lower()
        for pg in paper.source_pages or []:
            if probe and probe in re.sub(r"\s+", " ", str(pg.get("text", ""))).lower():
                page = pg.get("page_number")
                break
        return GapEvidence(paper_title=paper.title, source_filename=paper.source_filename, page_number=page, quote=quote)

    @staticmethod
    def _claim_quote(paper: ResearchPaperAnalysis, dist: Dict[str, float]) -> str:
        first = sorted(dist)[0]
        for claim in paper.key_claims:
            if re.search(rf"P\(\s*\|?{first}", claim):
                return claim
        return "Reported " + ", ".join(f"P({k}) = {v:.2f}" for k, v in sorted(dist.items()))
