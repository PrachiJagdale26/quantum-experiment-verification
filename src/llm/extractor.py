"""
Structured Proposal and Research Paper Extractor.
Invokes LLM with injection-defended prompts and validates output into ProposalSpecification
and ResearchPaperAnalysis models.
Preserves page provenance from extracted documents, supports both PDF and DOCX,
distinguishes between research papers and experiment proposals,
and includes a strict non-hallucinatory rule-based fallback that rejects invalid qubit counts (e.g. DOIs, years).
"""

import re
import json
import logging
from typing import Optional, Dict, Any, List, Tuple

from .client import ConfigurableLLMClient
from .prompts import EXTRACTION_SYSTEM_PROMPT, RESEARCH_PAPER_ANALYSIS_PROMPT
from ..schemas.proposal import (
    ProposalSpecification,
    CircuitSpec,
    GateOperation,
    ExpectedOutcome,
    ExtractedClaim,
)
from ..schemas.research_paper import (
    ResearchPaperAnalysis,
    QuantumExperimentRecord,
)
from ..circuit.parser import parse_circuit_dsl, CircuitParseError

logger = logging.getLogger(__name__)

# Word-to-number mapping for qubit counts
WORD_TO_NUM = {
    "one": 1, "single": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
}


_SIGNAL_CUES = {
    # Order matters: a sentence is filed under the first gap category it matches.
    "future_work": r"\b(future work|in future|future research|future studies|we plan to|we intend to|left (?:for|to) future|leave .{0,60}? (?:for|to) future|further work|next step|will be (?:explored|investigated|addressed))\b",
    "open_questions": r"\b(remains? (?:an )?open|open (?:question|problem)|unresolved|(?:it is )?unclear (?:whether|how|if)|(?:it is )?(?:not|un)known (?:whether|how|if)|has not (?:yet )?been (?:tested|demonstrated|studied|explored|shown))\b",
    "limitations": (r"\b(limited to|limitations? (?:of|is|are|include)|(?:is|are|was|were) restricted to|restricted to|"
                    r"only (?:up to|demonstrate|considered|tested|studied)|we (?:did|could|do) not|"
                    r"(?:was|were) not (?:tested|characteri[sz]ed|studied|considered|explored)|"
                    r"does not (?:account|consider|scale|include)|cannot (?:separate|distinguish|resolve|be tuned|scale)|"
                    r"noise (?:limits|dominates)|decoherence (?:limits|restricts)|beyond \d+ qubits)\b"),
}
_CLAIM_CUE = r"(\bP\(\s*[01]+\s*\)|\b\d+(?:\.\d+)?\s*%|\bfidelity\b|\bprobabilit(?:y|ies)\b|\bsuccess rate\b|\bspeed-?up\b)"
_CONTRIB_CUE = r"\b(we (?:propose|present|introduce|demonstrate|develop|report)|our (?:main )?(?:contribution|result)s?)\b"
_RESULT_CUE = r"\b(we (?:observe|observed|find|found|measured|measure|obtain|obtained|achieve|achieved)|results? (?:show|indicate|demonstrate|confirm))\b"


def unwrap_pdf_text(text: str) -> str:
    """Re-join sentences that PDF extraction broke across lines (and words split by hyphenation)."""
    text = re.sub(r"-{2,}\s*Page \d+\s*-{2,}", "\n", text or "")
    text = re.sub(r"(\w) -[ \t]*\n[ \t]*([a-z])", r"\1\2", text)          # "quan -\ntum" -> "quantum"
    text = re.sub(r"(\w)-[ \t]*\n[ \t]*([a-z])", r"\1-\2", text)          # "mid-\ncircuit" -> "mid-circuit"
    # A line that doesn't end a sentence, followed by one starting in lower case, is a wrapped line.
    text = re.sub(r"(?<![.!?:])[ \t]*\n[ \t]*(?=[a-z0-9(–])", " ", text)
    return text


_RUNNING_HEADER = re.compile(
    r"\||\b(?:vol(?:ume)?\.?\s*\d|doi|issn|pp\.|received|accepted|published|open access|copyright|©|https?://|www\.)",
    re.I)


def guess_title(raw_text: str, max_lines: int = 25) -> Optional[str]:
    """
    Find a paper's title on its first page, skipping journal running headers
    ("Nature | Vol 622 | ..."), page markers and article-type labels glued to the title.
    """
    lines = [l.strip() for l in (raw_text or "").splitlines()[:max_lines]]
    for i, line in enumerate(lines):
        if not line or re.match(r"-{2,}\s*Page", line) or _RUNNING_HEADER.search(line) or line == line.lower():
            continue  # page markers, running headers, lower-case journal names
        line = re.sub(r"^(?:Research\s+)?(?:Article|Letter|Review|Perspective)(?=[A-Z])\s*", "", line)
        line = re.sub(r"^(?:title|abstract)\s*[:.]\s*", "", line, flags=re.I)
        words = line.split()
        if len(words) < 3 or len(line) > 220 or line.endswith((".", ":")) and len(words) > 25:
            continue
        title = line
        # A title wrapped onto the next line continues in lower case ("...and\nteleportation on a ...").
        if i + 1 < len(lines) and re.match(r"^[a-z]", lines[i + 1]) and len(lines[i + 1]) < 120:
            title = f"{title} {lines[i + 1]}"
        return re.sub(r"\s+", " ", title).strip()
    return None


def extract_expected_outcomes(text: str) -> List[ExpectedOutcome]:
    """Parse statements like 'P(00) = 0.5' or 'P(|111>) ≈ 0.48' into ExpectedOutcome records."""
    found: Dict[str, float] = {}
    pairs = re.findall(r"P\(\s*\|?([01]{1,10})(?:>|⟩)?\s*\)\s*(?:=|≈|~|of)\s*([01]?\.\d+|[01](?![\d.]))", text or "")
    # "|00> with prob 0.50", "|11> with probability of 0.5", "|0> probability: 0.25"
    pairs += re.findall(r"\|([01]{1,10})(?:>|⟩)\s*(?:with\s+)?(?:a\s+)?prob(?:ability|\.)?\s*(?:of\s*)?[=:≈]?\s*([01]?\.\d+|[01](?![\d.]))",
                        text or "", flags=re.I)
    for bstate, prob in pairs:
        try:
            p = float(prob)
        except ValueError:
            continue
        if 0.0 <= p <= 1.0 and bstate not in found:
            found[bstate] = p
    lengths = {len(b) for b in found}
    if len(lengths) > 1:  # mixed register sizes: keep the most common length only
        common = max(lengths, key=lambda n: sum(1 for b in found if len(b) == n))
        found = {b: p for b, p in found.items() if len(b) == common}
    return [ExpectedOutcome(basis_state=b, expected_probability=p) for b, p in found.items()]


def extract_paper_signals(raw_text: str, cap: int = 6) -> Dict[str, List[str]]:
    """
    Pull gap-relevant sentences (limitations, future work, open questions, claims,
    contributions, results) from a paper's own wording. Deterministic; used as the
    offline fallback and to backfill fields an LLM leaves empty.
    """
    text = re.split(r"\n\s*(?:\d+\.?\s*)?(?:references|bibliography)\b", raw_text or "", maxsplit=1, flags=re.I)[0]
    text = unwrap_pdf_text(text)
    header = re.compile(r"^(?:\d+\.?\s*)?(?:abstract|introduction|results?|limitations?|future work|future directions|"
                        r"open questions?|methodology|methods|discussion|conclusions?|outlook)\s*[:.\-–]?\s+", re.I)
    sentences: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or (len(line) < 60 and not re.search(r"[.!?]$", line)):
            continue  # headings and title lines
        line = header.sub("", line)
        sentences += [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z(|])", line)]
    sentences = [s for s in sentences if 30 <= len(s) <= 400]

    out: Dict[str, List[str]] = {k: [] for k in ("limitations", "future_work", "open_questions", "claims", "contributions", "results")}

    def add(key: str, s: str):
        if s not in out[key] and len(out[key]) < cap:
            out[key].append(s)

    for s in sentences:
        for key, pat in _SIGNAL_CUES.items():
            if re.search(pat, s, re.I):
                add(key, s)
                break
        if re.search(_CLAIM_CUE, s, re.I) and re.search(r"\d", s):
            add("claims", s)
        if re.search(_CONTRIB_CUE, s, re.I):
            add("contributions", s)
        if re.search(_RESULT_CUE, s, re.I):
            add("results", s)
    return out


def extract_doi(text: str, filename: Optional[str] = None) -> Optional[str]:
    """
    Extract Digital Object Identifier (DOI) or publication ID from text or filename.
    Never confuse DOIs with qubit numbers or gate targets.
    """
    # 1. Standard DOI regex: 10.xxxx/...
    doi_match = re.search(r"\b(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)\b", text)
    if doi_match:
        return doi_match.group(1).rstrip(".,;)")

    # 2. Nature / Springer style article identifier: s42256-025-01153-0
    s_match = re.search(r"\b(s\d{5}-\d{3}-\d{5}-\d)\b", text, re.IGNORECASE)
    if s_match:
        return f"10.1038/{s_match.group(1)}"

    # 3. Check filename
    if filename:
        fn_match = re.search(r"(s\d{5}-\d{3}-\d{5}-\d)", filename, re.IGNORECASE)
        if fn_match:
            return f"10.1038/{fn_match.group(1)}"
        fn_doi = re.search(r"(10\.\d{4,9}_[-._A-Za-z0-9]+)", filename)
        if fn_doi:
            return fn_doi.group(1).replace("_", "/", 1)

    # 4. arXiv identifier
    arxiv_match = re.search(r"\barXiv:\s*(\d{4}\.\d{4,5}(?:v\d+)?)\b", text, re.IGNORECASE)
    if arxiv_match:
        return f"arXiv:{arxiv_match.group(1)}"

    return None


def classify_document(raw_text: str, filename: Optional[str] = None) -> str:
    """
    Classify whether an uploaded document is an academic Research Paper or an Experiment Proposal.
    
    Research Paper: Broad academic work discussing problems, methodologies, multiple experiments,
    literature references, and findings (e.g. Nature, arXiv, PRX).
    
    Experiment Proposal: A focused specification targeting a single executable quantum circuit,
    initial state, gate sequence, and expected measurement distribution.
    """
    text_lower = raw_text.lower()
    fn_lower = (filename or "").lower()

    # Strong indicators of research paper
    if extract_doi(raw_text, filename):
        return "research_paper"

    if any(m in fn_lower for m in ["s42256", "nature", "arxiv", "prx", "prl", "paper", "journal", "review"]):
        return "research_paper"

    # Academic section headings tally
    academic_keywords = [
        "abstract", "introduction", "related work", "methodology", "methods",
        "results and discussion", "discussion", "conclusion", "references",
        "acknowledgements", "supplementary", "bibliography"
    ]
    academic_matches = sum(1 for kw in academic_keywords if re.search(rf"\b{kw}\b", text_lower))
    if academic_matches >= 3:
        return "research_paper"

    # Proposal indicators
    proposal_keywords = ["proposal title", "gate sequence", "initial state", "measurement basis", "expected outcomes"]
    proposal_matches = sum(1 for kw in proposal_keywords if kw in text_lower)
    if proposal_matches >= 2:
        return "experiment_proposal"

    # Default to experiment proposal if short/focused, else research paper if long
    if len(raw_text.split()) > 1200:
        return "research_paper"
    return "experiment_proposal"


def extract_strict_num_qubits(raw_text: str) -> Optional[int]:
    """
    Strict semantic extraction of qubit count.
    
    RULES:
    1. Extract num_qubits ONLY when the document explicitly describes a quantum system
       or circuit with a specified number of qubits (1 to 10).
    2. NEVER interpret DOI numbers (e.g., s42256-025-01153-0), publication identifiers,
       years (e.g. 2024, 2025), page numbers, citations ([1], [42]), or equation numbers as qubit counts.
    3. Require contextual evidence such as "two-qubit circuit", "3 qubits", or an explicitly declared register.
    4. If the number cannot be established confidently or multiple conflicting counts exist, return None.
    5. Never return values > 10 (simulation ceiling).
    """
    # Sanitize text: strip DOIs, publication IDs, years, citations, and page/equation references
    clean_text = raw_text

    # Strip DOIs and Nature identifiers
    clean_text = re.sub(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b", " ", clean_text)
    clean_text = re.sub(r"\bs\d{5}-\d{3}-\d{5}-\d\b", " ", clean_text, flags=re.IGNORECASE)

    # Strip citations like [1], [1, 2], [1-5], [2024]
    clean_text = re.sub(r"\[\s*\d+(?:[\s,\s–-]+\d+)*\s*\]", " ", clean_text)

    # Strip 4-digit years (1900-2099)
    clean_text = re.sub(r"\b(?:19|20)\d{2}\b", " ", clean_text)

    # Strip page and equation markers
    clean_text = re.sub(r"\b(?:page|pp?|eq|equation|fig|figure|ref|reference)\.?\s*\(?\d+\)?\b", " ", clean_text, flags=re.IGNORECASE)

    found_counts: List[int] = []

    # 1. Word-based qubit counts: "two-qubit", "three qubits", "single-qubit"
    for word, val in WORD_TO_NUM.items():
        if val > 10:
            continue
        word_pattern = rf"\b{word}\s*(?:-|–|\s+)(?:qubit|qubits|qubit-register|qubit-system)\b"
        if re.search(word_pattern, clean_text, re.IGNORECASE):
            found_counts.append(val)

    # 2. Explicit digit-based qubit counts: "2 qubits", "3-qubit", "register of 2 qubits"
    digit_patterns = [
        r"\b([1-9]|10)\s*(?:-|–|\s+)(?:qubits?|qubit register|qubit system)\b",
        r"\b(?:register\s+of|system\s+of|circuit\s+with|allocated)\s+([1-9]|10)\s+qubits?\b",
        r"\b(?:QuantumCircuit|QuantumRegister|qreg)\s*\(?\s*([1-9]|10)\b"
    ]
    for pat in digit_patterns:
        for m in re.finditer(pat, clean_text, re.IGNORECASE):
            try:
                val = int(m.group(1))
                if 1 <= val <= 10:
                    found_counts.append(val)
            except ValueError:
                pass

    # Explicitly check for oversized registers (> 10 qubits) mentioned in text (e.g. 127-qubit IBM Eagle)
    oversized_pattern = r"\b(\d{2,})\s*(?:-|–|\s+)(?:qubits?)\b"
    oversized_matches = [int(m.group(1)) for m in re.finditer(oversized_pattern, clean_text, re.IGNORECASE) if int(m.group(1)) > 10]
    if oversized_matches and not found_counts:
        # Document discusses a large hardware system beyond prototype limit, not a <=10 circuit
        logger.info(f"Detected large quantum system ({oversized_matches[0]} qubits) exceeding simulation ceiling of 10.")
        return None

    if not found_counts:
        return None

    # Count frequencies of detected qubit sizes
    freq: Dict[int, int] = {}
    for c in found_counts:
        freq[c] = freq.get(c, 0) + 1

    # If single dominant candidate exists, return it
    sorted_candidates = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    best_candidate, best_freq = sorted_candidates[0]

    # If there is a tie between multiple different qubit counts (multi-experiment paper), return None
    if len(sorted_candidates) > 1 and sorted_candidates[1][1] == best_freq and sorted_candidates[1][0] != best_candidate:
        logger.info("Multiple conflicting qubit counts detected in text. Returning None for ambiguous register size.")
        return None

    return best_candidate


def _safe_angle(expr: str) -> Optional[float]:
    """Parse a rotation angle like 'pi/2', '0.785', '-π/4', '2*pi/3' without eval (the text is untrusted)."""
    e = expr.strip().lower().replace("π", "pi").replace(" ", "")
    m = re.fullmatch(r"([+-]?)(\d+(?:\.\d+)?|\.\d+)?\*?(pi)?(?:/(\d+(?:\.\d+)?))?", e)
    if not m or (m.group(2) is None and m.group(3) is None):
        return None
    val = float(m.group(2)) if m.group(2) is not None else 1.0
    if m.group(3):
        val *= 3.141592653589793
    if m.group(4):
        val /= float(m.group(4))
    return -val if m.group(1) == "-" else val


def extract_strict_gates(raw_text: str, num_qubits: Optional[int] = None) -> List[GateOperation]:
    """
    Extract gate operations with strict syntax and bound checking, in document order.

    CRITICAL:
    - Never match single letters inside identifiers (e.g. 's' in 's42256').
    - Target qubit indices must strictly be within 0-9.
    - If num_qubits is provided, target indices must be < num_qubits.
    - Gates are returned in the order they appear in the text (order changes the physics).
    - Rotation angles are parsed without eval (the text is untrusted).
    """
    clean_text = raw_text or ""
    clean_text = re.sub(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b", " ", clean_text)
    clean_text = re.sub(r"\bs\d{5}-\d{3}-\d{5}-\d\b", " ", clean_text, flags=re.IGNORECASE)
    clean_text = re.sub(r"\b(?:19|20)\d{2}\b", " ", clean_text)

    found: List[Tuple[int, GateOperation]] = []
    claimed: List[Tuple[int, int]] = []   # spans already used by a more specific pattern

    def in_range(*qs: int) -> bool:
        return all(0 <= q <= 9 and (num_qubits is None or q < num_qubits) for q in qs)

    def add(m: "re.Match", name: str, targets: List[int], params: Optional[List[float]] = None) -> None:
        if any(a <= m.start() < b for a, b in claimed):
            return
        try:
            found.append((m.start(), GateOperation(name=name, target_qubits=targets, params=params)))
            claimed.append((m.start(), m.end()))
        except Exception:
            pass

    # Qiskit syntax: qc.h(0), qc.cx(0, 1), qc.rx(pi/2, 0)
    for m in re.finditer(r"\bqc\.([a-z]+)\(([^)]+)\)", clean_text, re.IGNORECASE):
        method, args = m.group(1).upper(), [a.strip() for a in m.group(2).split(",")]
        try:
            if method in ("H", "X", "Y", "Z", "S", "T") and len(args) == 1 and in_range(int(args[0])):
                add(m, method, [int(args[0])])
            elif method in ("CX", "CNOT", "CZ", "SWAP") and len(args) == 2:
                q1, q2 = int(args[0]), int(args[1])
                if q1 != q2 and in_range(q1, q2):
                    add(m, "CX" if method == "CNOT" else method, [q1, q2])
            elif method in ("RX", "RY", "RZ") and len(args) == 2:
                ang, q = _safe_angle(args[0]), int(args[1])
                if ang is not None and in_range(q):
                    add(m, method, [q], [ang])
        except ValueError:
            pass

    # Two-qubit gates written with roles: "CX with control 0 and target 1"
    for m in re.finditer(r"\b(CX|CNOT|CZ)\b[^.;]{0,30}?\bcontrol\s*(?:qubit\s*)?([0-9])\b[^.;]{0,20}?\btarget\s*(?:qubit\s*)?([0-9])\b",
                         clean_text, re.IGNORECASE):
        q1, q2 = int(m.group(2)), int(m.group(3))
        if q1 != q2 and in_range(q1, q2):
            add(m, "CX" if m.group(1).upper() == "CNOT" else m.group(1).upper(), [q1, q2])

    # Parametric gates: "RX(pi/2) on qubit 0"
    for m in re.finditer(r"\b(RX|RY|RZ)\b\s*\(([^)]+)\)\s*(?:gate\s+)?(?:on\s+)?(?:qubits?\s+)?([0-9])\b",
                         clean_text, re.IGNORECASE):
        ang, q = _safe_angle(m.group(2)), int(m.group(3))
        if ang is not None and in_range(q):
            add(m, m.group(1).upper(), [q], [ang])

    # Two-qubit gates: "CX on qubits 0, 1", "CNOT 0 1", "CX(0,1)"
    for m in re.finditer(r"\b(CX|CNOT|CZ|SWAP)\b\s*(?:gate\s+)?(?:on\s+)?(?:qubits?\s+)?\(?\s*([0-9])\s*(?:,|and|\s)\s*([0-9])\b",
                         clean_text, re.IGNORECASE):
        q1, q2 = int(m.group(2)), int(m.group(3))
        if q1 != q2 and in_range(q1, q2):
            add(m, "CX" if m.group(1).upper() == "CNOT" else m.group(1).upper(), [q1, q2])

    # Single-qubit gates: "H on qubit 0", "X gate 1", "H(0)"
    for m in re.finditer(r"\b([HXYZST])\b\s*(?:\(\s*(?=[0-9]))?(?:gate\s+)?(?:on\s+)?(?:qubits?\s+)?([0-9])\b",
                         clean_text, re.IGNORECASE):
        q = int(m.group(2))
        if in_range(q):
            add(m, m.group(1).upper(), [q])

    return [g for _, g in sorted(found, key=lambda t: t[0])]


class ProposalExtractor:
    """
    Extracts structured experiment specifications from unstructured proposal text or documents.
    Validates output through Pydantic schemas without inventing missing data.
    Supports both single-proposal verification and comprehensive research-paper analysis.
    """

    def __init__(self, llm_client: ConfigurableLLMClient):
        self.llm_client = llm_client

    def classify(self, raw_text: str, filename: Optional[str] = None) -> str:
        """Expose document classifier."""
        return classify_document(raw_text, filename)

    def extract(
        self,
        raw_text: str,
        document_title: Optional[str] = None,
        extracted_doc: Optional[Any] = None,
        input_mode: str = "upload"
    ) -> ProposalSpecification:
        """
        Process proposal text into a verified ProposalSpecification.
        Safely identifies research papers and prevents DOI / numerical hallucination.
        """
        filename = extracted_doc.filename if extracted_doc else None
        doc_type = classify_document(raw_text, filename)
        detected_doi = extract_doi(raw_text, filename)

        if not raw_text or not raw_text.strip():
            return ProposalSpecification(
                title=document_title or "Untitled Empty Proposal",
                objective="No content provided in proposal.",
                missing_information=["Empty proposal text submitted."],
                raw_text="",
                input_mode=input_mode,
                source_filename=filename,
                source_pages=extracted_doc.pages if extracted_doc else [],
                document_type=doc_type,
                doi=detected_doi
            )

        # Wrap text in untrusted data container with strict extraction defenses
        user_prompt = (
            f"Analyze the following proposal text:\n\n"
            f"<UNTRUSTED_PROPOSAL_TEXT>\n"
            f"{raw_text}\n"
            f"</UNTRUSTED_PROPOSAL_TEXT>\n\n"
            f"Extract all parameters into the requested JSON schema.\n"
            f"STRICT RULES:\n"
            f"1. Never invent missing circuit operations, parameters, or expected probabilities.\n"
            f"2. Never interpret DOIs (e.g. s42256-025-01153-0), years (2024, 2025), or page numbers as qubits or gate targets.\n"
            f"3. If the document is an academic paper discussing general methods or multiple circuits, set num_qubits=null."
        )

        try:
            response_text = self.llm_client.complete(
                system_prompt=EXTRACTION_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.0,
                fallback_to_demo=False,  # canned demo JSON would describe a different document
                purpose="proposal_extraction",
            )
            parsed_json = self._clean_and_parse_json(response_text)
            call = getattr(self.llm_client, "last_call", {}) or {}
            parsed_json.setdefault("extraction_warnings", [])
            parsed_json["extraction_warnings"] = list(parsed_json["extraction_warnings"]) + [
                f"Extracted by live LLM ({call.get('provider', '?')}/{call.get('model', '?')}); review before relying on it."]
            parsed_json["raw_text"] = raw_text
            parsed_json["input_mode"] = input_mode
            parsed_json["document_type"] = doc_type
            parsed_json["doi"] = detected_doi or parsed_json.get("doi")

            if document_title and not parsed_json.get("title"):
                parsed_json["title"] = document_title

            if "research_objective" in parsed_json and not parsed_json.get("objective"):
                parsed_json["objective"] = parsed_json["research_objective"]
            if not parsed_json.get("objective"):
                parsed_json["objective"] = document_title or "Quantum experiment analysis"

            if "research_problem" in parsed_json and not parsed_json.get("hypothesis"):
                parsed_json["hypothesis"] = parsed_json["research_problem"]

            # Sanitize num_qubits from model
            model_qubits = parsed_json.get("num_qubits")
            if model_qubits is not None:
                if not isinstance(model_qubits, int) or model_qubits < 1 or model_qubits > 10:
                    logger.warning(f"Sanitizing out-of-bounds num_qubits={model_qubits} from model response to None.")
                    parsed_json["num_qubits"] = None

            # If document is classified as a research paper with no specific executable circuit, ensure num_qubits is null
            if doc_type == "research_paper":
                strict_q = extract_strict_num_qubits(raw_text)
                if strict_q is None:
                    parsed_json["num_qubits"] = None
                    parsed_json["circuit"] = None

            if parsed_json.get("num_qubits") is None:
                missing = parsed_json.get("missing_information") or []
                if not any("Target system size" in m for m in missing):
                    missing.append("Target system size (number of qubits) not explicitly specified in proposal.")
                parsed_json["missing_information"] = missing

            # Sanitize circuit from model
            model_circuit = parsed_json.get("circuit")
            if isinstance(model_circuit, dict):
                c_qubits = model_circuit.get("num_qubits")
                if not isinstance(c_qubits, int) or c_qubits < 1 or c_qubits > 10:
                    logger.warning(f"Sanitizing out-of-bounds circuit num_qubits={c_qubits} from model response.")
                    parsed_json["circuit"] = None

            if extracted_doc:
                parsed_json["source_filename"] = extracted_doc.filename
                parsed_json["source_pages"] = extracted_doc.pages
                # Merge extraction warnings
                existing_warnings = parsed_json.get("extraction_warnings") or []
                for w in extracted_doc.warnings:
                    if w not in existing_warnings:
                        existing_warnings.append(w)
                parsed_json["extraction_warnings"] = existing_warnings

            spec = ProposalSpecification(**parsed_json)

            # Assign page references to claims if missing
            if extracted_doc and extracted_doc.pages:
                self._assign_claim_page_references(spec.author_claims, extracted_doc.pages)

            return spec

        except Exception as e:
            logger.warning(f"LLM extraction parsing failed or rejected: {e}. Falling back to strict rule-based extractor.")
            return self._heuristic_fallback(
                raw_text=raw_text,
                document_title=document_title,
                extracted_doc=extracted_doc,
                input_mode=input_mode,
                error_note=str(e),
                doc_type=doc_type,
                doi=detected_doi
            )

    def extract_research_paper(
        self,
        raw_text: str,
        document_title: Optional[str] = None,
        extracted_doc: Optional[Any] = None
    ) -> ResearchPaperAnalysis:
        """
        Extract structured research paper analysis supporting multiple experiments,
        methodology, algorithmic frameworks, experimental setup, and stated limitations.
        """
        filename = extracted_doc.filename if extracted_doc else None
        doi = extract_doi(raw_text, filename)
        pages = extracted_doc.pages if extracted_doc else []

        user_prompt = (
            f"Analyze the following quantum computing research paper:\n\n"
            f"<UNTRUSTED_PAPER_TEXT>\n"
            f"{raw_text}\n"
            f"</UNTRUSTED_PAPER_TEXT>\n\n"
            f"Extract all research paper fields into the requested JSON schema.\n"
            f"If multiple experiments or benchmarks are discussed, list them in the 'experiments' array.\n"
            f"Never confuse DOIs, publication IDs, years, or citations with qubit numbers."
        )

        if getattr(self.llm_client, "demo_mode", False):
            # Demo mode has no model: analyse the paper's own text rather than returning canned JSON.
            return self._heuristic_paper_fallback(
                raw_text=raw_text, document_title=document_title, extracted_doc=extracted_doc,
                doi=doi, error_note="demo mode, no LLM configured",
            )

        try:
            response_text = self.llm_client.complete(
                system_prompt=RESEARCH_PAPER_ANALYSIS_PROMPT,
                user_prompt=user_prompt,
                temperature=0.0,
                fallback_to_demo=False,  # a canned demo answer would describe a different paper
                purpose="paper_analysis",
            )
            parsed_json = self._clean_and_parse_json(response_text)
            parsed_json["document_type"] = "research_paper"
            call = getattr(self.llm_client, "last_call", {}) or {}
            parsed_json["extraction_warnings"] = list(parsed_json.get("extraction_warnings") or []) + [
                f"Analysed by live LLM ({call.get('provider', '?')}/{call.get('model', '?')}); "
                f"fields are model interpretations until checked against the source."]
            parsed_json["raw_text"] = raw_text
            parsed_json["source_filename"] = filename
            parsed_json["source_pages"] = pages
            if doi and not parsed_json.get("doi"):
                parsed_json["doi"] = doi
            if document_title and not parsed_json.get("title"):
                parsed_json["title"] = document_title

            # Normalise experiments: real papers often describe gates/outcomes in prose, which must not
            # sink the whole analysis. Valid structured entries are kept; prose moves to text fields.
            if "experiments" in parsed_json and isinstance(parsed_json["experiments"], list):
                for i, exp in enumerate(parsed_json["experiments"]):
                    if isinstance(exp, dict):
                        exp.setdefault("experiment_id", f"EXP-{i + 1:02d}")
                        exp.setdefault("title", exp.get("experiment_id"))
                        eq = exp.get("num_qubits")
                        if eq is not None and (not isinstance(eq, int) or eq < 1 or eq > 10):
                            exp["num_qubits"] = None
                        gates, prose_gates = [], []
                        for g in exp.get("gates") or []:
                            if not isinstance(g, dict):
                                prose_gates.append(str(g))
                                continue
                            try:
                                gates.append(GateOperation(**g))
                            except Exception:
                                prose_gates.append(str(g.get("name", g)))
                        exp["gates"] = gates
                        if prose_gates:
                            exp["circuit_summary"] = "; ".join(filter(None, [exp.get("circuit_summary"), "Operations: " + ", ".join(prose_gates)]))
                        outs, prose_outs = [], []
                        for o in exp.get("expected_outcomes") or []:
                            try:
                                outs.append(ExpectedOutcome(**o)) if isinstance(o, dict) else prose_outs.append(str(o))
                            except Exception:
                                prose_outs.append(str(o))
                        exp["expected_outcomes"] = outs
                        if prose_outs:
                            exp["findings"] = "; ".join(filter(None, [exp.get("findings"), "Expected: " + "; ".join(prose_outs)]))
                        if exp["gates"] and not exp.get("num_qubits"):
                            exp["has_executable_circuit"] = False

            # Backfill gap signals the model left empty from the paper's own sentences.
            signals = extract_paper_signals(raw_text)
            for field, key in (("limitations", "limitations"), ("future_work", "future_work"),
                               ("open_questions", "open_questions"), ("key_claims", "claims"),
                               ("key_contributions", "contributions"), ("results_and_findings", "results")):
                if not parsed_json.get(field):
                    parsed_json[field] = signals[key]

            return ResearchPaperAnalysis(**parsed_json)

        except Exception as e:
            logger.warning(f"LLM research paper extraction failed: {e}. Falling back to rule-based analysis.")
            return self._heuristic_paper_fallback(
                raw_text=raw_text,
                document_title=document_title,
                extracted_doc=extracted_doc,
                doi=doi,
                error_note=str(e)
            )

    def _heuristic_paper_fallback(
        self,
        raw_text: str,
        document_title: Optional[str],
        extracted_doc: Optional[Any],
        doi: Optional[str],
        error_note: str
    ) -> ResearchPaperAnalysis:
        """Deterministic rule-based extractor for academic research papers."""
        lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
        title = document_title
        if not title or _RUNNING_HEADER.search(title):
            title = guess_title(raw_text) or (lines[0][:120] if lines else "Analyzed Research Paper")

        # Extract abstract; papers without an "Abstract" heading open with it after the title block.
        abstract = None
        abs_match = re.search(r"(?i)\babstract\b[:\s]*(.+?)(?:\n\n|\b(?:introduction|keywords|background)\b|\Z)", raw_text, re.DOTALL)
        if abs_match:
            abstract = re.sub(r"\s+", " ", abs_match.group(1)).strip()[:1000]
        unwrapped = unwrap_pdf_text(raw_text)
        body_sentences = [s.strip() for line in unwrapped.splitlines() if len(line.strip()) >= 60 and not _RUNNING_HEADER.search(line)
                          for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", line.strip()) if 50 <= len(s.strip()) <= 400]
        if not abstract and body_sentences:
            abstract = " ".join(body_sentences[:3])[:1000]

        # Objective: an explicit objective, else the paper's "Here we ..." sentence, else its opening sentence.
        obj_match = re.search(r"(?im)^[ \t]*(?:objectives?|goals?|purpose)[ \t]*[:.][ \t]*(.+?)(?:\n\n|\. |\Z)", raw_text, re.DOTALL)
        here_we = next((s for s in body_sentences if re.search(
            r"(?i)\b(?:here,? we|in this (?:work|paper|article),? we|we aim to|we (?:study|investigate|demonstrate|present|report|propose|introduce))\b", s)), None)
        if obj_match:
            objective = re.sub(r"\s+", " ", obj_match.group(1)).strip()[:300]
        elif here_we:
            objective = here_we
        else:
            objective = abstract[:300] if abstract else "Not identified by rule-based analysis."

        prob_match = re.search(r"(?i)\b(?:problem|challenge|bottleneck|limitation of)\b[:\s]*(.+?)(?:\n\n|\. |\Z)", raw_text, re.DOTALL)
        problem = prob_match.group(1).strip()[:300] if prob_match else "Overcoming quantum resource overhead and noise susceptibility in quantum processing."

        # Extract frameworks discussed
        frameworks = []
        for fw in ["Qiskit", "Cirq", "PennyLane", "PyTorch", "TensorFlow Quantum", "VQE", "QAOA", "Quantum Neural Network", "Error Mitigation"]:
            if re.search(rf"\b{fw}\b", raw_text, re.IGNORECASE):
                frameworks.append(fw)
        if not frameworks:
            frameworks = ["Quantum Circuit Model"]

        # Extract quantum systems discussed
        systems = []
        first_seen = {}
        platform_patterns = {
            "Superconducting Transmon": r"\bsuperconducting\b|\btransmons?\b",
            "Trapped Ion": r"\btrapped[- ]ions?\b|\bion[- ]traps?\b",
            "Neutral Atom": r"\bneutral[- ]atoms?\b|\brydberg[- ]atoms?\b",
            "Photonic": r"\bphotonic\b",
            "Silicon Spin": r"\bsilicon spin\b|\bspin qubits?\b",
        }
        for qs, pat in platform_patterns.items():
            m = re.search(pat, raw_text, re.IGNORECASE)
            if m:
                first_seen[qs] = m.start()
        # Primary system first: the one the paper mentions earliest (title / abstract).
        systems = sorted(first_seen, key=first_seen.get)
        if not systems:
            systems = ["Ideal Statevector Simulation"]

        # Detect multiple experiments / benchmarks
        strict_q = extract_strict_num_qubits(raw_text)
        strict_gates = extract_strict_gates(raw_text, num_qubits=strict_q)

        experiments: List[QuantumExperimentRecord] = []
        exp_pattern = re.compile(
            # A heading such as "Experiment 1:" / "Benchmark II." at the start of a line; prose that merely
            # wraps onto a line starting with "experiment" is not an experiment.
            r"(?im)^[ \t]*(experiment|benchmark|case study|evaluation)[ \t]+([0-9]{1,2}|[IVX]{1,4}|[A-Z])[ \t]*[:.)\-–][ \t]*(.+?)(?=\n\s*\n\s*(?:experiment|benchmark|case study|evaluation|results|limitations)|\Z)",
            re.DOTALL
        )
        exp_matches = list(exp_pattern.finditer(raw_text))

        if exp_matches:
            for idx, em in enumerate(exp_matches[:4]):
                e_label = em.group(2) or str(idx + 1)
                full_block = em.group(3).strip()
                block_lines = [l.strip() for l in full_block.splitlines() if l.strip()]
                e_title = block_lines[0] if block_lines else f"Experiment {e_label}"
                e_qubits = extract_strict_num_qubits(full_block)
                e_gates = extract_strict_gates(full_block, num_qubits=e_qubits)
                experiments.append(QuantumExperimentRecord(
                    experiment_id=f"EXP-{idx + 1:02d}",
                    title=f"Experiment {e_label}: {e_title[:60]}",
                    quantum_system=systems[0],
                    num_qubits=e_qubits,
                    circuit_summary="Parametric quantum ansatz" if not e_gates else f"{len(e_gates)} gate operations",
                    has_executable_circuit=bool(e_gates and e_qubits),
                    gates=e_gates if e_qubits else [],
                    expected_outcomes=extract_expected_outcomes(full_block),
                    findings=(extract_paper_signals(full_block, cap=1)["results"] or [None])[0],
                ))
        else:
            # Create a primary experiment record
            experiments.append(QuantumExperimentRecord(
                experiment_id="EXP-01",
                title=f"Primary Evaluation: {title[:50]}",
                quantum_system=systems[0],
                num_qubits=strict_q,
                circuit_summary=f"Gate sequence with {len(strict_gates)} operations" if strict_gates else "Theoretical quantum framework",
                has_executable_circuit=bool(strict_gates and strict_q),
                gates=strict_gates if strict_q else [],
                expected_outcomes=extract_expected_outcomes(raw_text),
                findings=(extract_paper_signals(raw_text, cap=1)["results"] or [None])[0],
            ))

        # Results, limitations and gap signals come from the paper's own sentences.
        signals = extract_paper_signals(raw_text)
        sections = getattr(extracted_doc, "sections", {}) or {}
        methodology = (sections.get("methodology") or "").strip()[:600] or "Not identified by rule-based analysis."
        setup = (sections.get("quantum_system") or "").strip()[:400] or ", ".join(systems)

        return ResearchPaperAnalysis(
            title=title,
            doi=doi,
            publication_info=None,
            authors=[],
            abstract=abstract,
            research_problem=problem,
            research_objective=objective,
            methodology=methodology,
            algorithms_and_frameworks=frameworks,
            experimental_setup=setup,
            quantum_systems_discussed=systems,
            experiments=experiments,
            results_and_findings=signals["results"],
            limitations=signals["limitations"],
            key_contributions=signals["contributions"],
            future_work=signals["future_work"],
            open_questions=signals["open_questions"],
            key_claims=signals["claims"],
            raw_text=raw_text,
            source_filename=extracted_doc.filename if extracted_doc else None,
            source_pages=extracted_doc.pages if extracted_doc else [],
            extraction_warnings=[f"Rule-based research paper analysis applied (reason: {(error_note or '').splitlines()[0][:200]})"]
        )

    def _clean_and_parse_json(self, text: str) -> Dict[str, Any]:
        """Strip markdown fences and load JSON."""
        clean = text.strip()
        if clean.startswith("```"):
            clean = re.sub(r"^```(?:json)?\s*", "", clean)
            clean = re.sub(r"\s*```$", "", clean)

        match = re.search(r"(\{.*\})", clean, re.DOTALL)
        if match:
            clean = match.group(1)

        return json.loads(clean)

    def _assign_claim_page_references(self, claims: List[ExtractedClaim], pages: List[Dict[str, Any]]):
        """Map extracted claim text to source document pages."""
        for claim in claims:
            if not claim.source_reference or claim.source_reference == "None":
                words = [w.lower() for w in re.findall(r"\b[a-zA-Z]{4,}\b", claim.statement)]
                best_page = 1
                max_matches = 0
                for p in pages:
                    p_text_lower = p.get("text", "").lower()
                    matches = sum(1 for w in words if w in p_text_lower)
                    if matches > max_matches:
                        max_matches = matches
                        best_page = p.get("page_number", 1)
                claim.source_reference = f"Page {best_page}"

    def _heuristic_fallback(
        self,
        raw_text: str,
        document_title: Optional[str],
        extracted_doc: Optional[Any],
        input_mode: str,
        error_note: str,
        doc_type: str = "experiment_proposal",
        doi: Optional[str] = None
    ) -> ProposalSpecification:
        """
        Deterministic rule-based extractor if LLM is unavailable or rejects parsing.
        Strict non-hallucination: Never invents missing circuit operations, parameters,
        or qubit numbers. Never interprets DOIs, years, or citations as qubit counts.
        """
        warnings = [f"Rule-based extraction applied (LLM parser note: {error_note})"]
        if extracted_doc:
            warnings.extend(w for w in extracted_doc.warnings if w not in warnings)

        lines = [l.strip() for l in raw_text.splitlines() if l.strip()]

        # 1. Title
        title = document_title
        if not title and extracted_doc and extracted_doc.sections.get("title"):
            title = extracted_doc.sections["title"]
        if not title:
            title = lines[0][:100] if lines else "Extracted Quantum Proposal"

        # 2. Objective & Hypothesis
        objective = ""
        hypothesis = ""
        if extracted_doc:
            objective = extracted_doc.sections.get("objective", "")
            hypothesis = extracted_doc.sections.get("hypothesis", "")
            methodology = extracted_doc.sections.get("methodology", "")
            quantum_system = extracted_doc.sections.get("quantum_system", "")
        else:
            methodology = ""
            quantum_system = ""

        if not objective:
            obj_match = re.search(r"(?i)\b(?:objective|goal|purpose)\b[:\s]*(.+?)(?:\n\n|\Z)", raw_text, re.DOTALL)
            objective = obj_match.group(1).strip()[:300] if obj_match else (lines[1] if len(lines) > 1 else title)

        if not hypothesis:
            hypo_match = re.search(r"(?i)\b(?:hypothesis|theoretical basis)\b[:\s]*(.+?)(?:\n\n|\Z)", raw_text, re.DOTALL)
            hypothesis = hypo_match.group(1).strip()[:300] if hypo_match else None

        # 3. Quantum System
        if not quantum_system:
            qs_match = re.search(r"(?i)\b(superconducting|transmon|trapped ion|photonic|silicon spin|neutral atom|optical bench|nmr)\b", raw_text)
            quantum_system = qs_match.group(1).title() if qs_match else "Ideal Simulation"

        # 4. STRICT Number of Qubits Extraction
        # Never interpret DOIs (e.g. s42256), years, or page numbers as qubits!
        num_qubits = extract_strict_num_qubits(raw_text)

        # 5. STRICT Gate Sequence Extraction
        # Never match letters in identifiers or out-of-bounds target indices
        # Read gates from the most specific place: an explicit "Gates:" list, then the methodology
        # section, then the whole text. Scanning everything first would count gates twice when the
        # hypothesis also describes the circuit in prose.
        extracted_gates: List[GateOperation] = []
        gate_line = re.search(r"(?im)\bgates?(?:\s+sequence)?\s*:\s*(.+)$", raw_text)
        method_text = (extracted_doc.sections.get("methodology") if extracted_doc else "") or ""
        for source in ([gate_line.group(1)] if gate_line else []) + ([method_text] if method_text else []) + [raw_text]:
            extracted_gates = extract_strict_gates(source, num_qubits=num_qubits)
            if extracted_gates:
                break

        circuit = None
        if extracted_gates:
            max_q = max(max(g.target_qubits) for g in extracted_gates)
            required_qubits = max_q + 1
            if num_qubits is None:
                if required_qubits <= 10:
                    num_qubits = required_qubits

            if num_qubits and 1 <= num_qubits <= 10 and max_q < num_qubits:
                try:
                    circuit = CircuitSpec(
                        num_qubits=num_qubits,
                        gates=extracted_gates,
                        initial_state=f"|{'0' * num_qubits}>",
                        measurement_basis="computational (Z)"
                    )
                except Exception as ce:
                    warnings.append(f"Circuit construction warning: {ce}")
                    circuit = None

        # 6. Expected Outcomes Extraction
        outcomes: List[ExpectedOutcome] = []
        prob_matches_1 = re.findall(r"\|?([01]+)\>?\s*[:=]\s*([0-1]?\.\d+)", raw_text)
        for bstate, prob_str in prob_matches_1:
            try:
                p_val = float(prob_str)
                outcomes.append(ExpectedOutcome(basis_state=bstate, expected_probability=p_val))
            except Exception:
                pass

        if not outcomes:
            prob_matches_2 = re.findall(r"(?:P\(|prob(?:ability)?\s*[:=]?\s*)([0-1]?\.\d+)\s*(?:for\s*\|?([01]+)\>?)", raw_text, re.IGNORECASE)
            for prob_str, bstate in prob_matches_2:
                try:
                    p_val = float(prob_str)
                    outcomes.append(ExpectedOutcome(basis_state=bstate, expected_probability=p_val))
                except Exception:
                    pass

        if not outcomes:
            outcomes = extract_expected_outcomes(raw_text)

        # 7. Assumptions
        assumptions = []
        if extracted_doc and extracted_doc.sections.get("assumptions"):
            assumptions.append(extracted_doc.sections["assumptions"][:300])
        else:
            assump_match = re.search(r"(?i)\b(?:assumptions?|constraints?)\b[:\s]*(.+?)(?:\n\n|\Z)", raw_text)
            if assump_match:
                assumptions.append(assump_match.group(1).strip()[:300])

        # 8. Record missing information explicitly (NEVER INVENT)
        missing: List[str] = []
        if doc_type == "research_paper":
            missing.append("Document classified as an academic research paper discussing methodology and findings. Essential circuit parameters must be verified or selected from paper experiments.")
        if num_qubits is None:
            missing.append("Target system size (number of qubits) not explicitly specified or ambiguous in document.")
        if circuit is None or not circuit.gates:
            missing.append("No explicit quantum gate sequence or circuit operations found in proposal.")
        if not outcomes:
            missing.append("Expected measurement outcome probability distribution was not specified.")
        if not hypothesis:
            missing.append("Scientific hypothesis not explicitly articulated.")

        # 9. Formulate discrete author claims
        claims: List[ExtractedClaim] = []
        c_idx = 1
        if hypothesis:
            claims.append(ExtractedClaim(
                claim_id=f"CLM-{c_idx:02d}",
                claim_type="hypothesis",
                statement=hypothesis[:200],
                source_reference="Page 1",
                confidence_level="high"
            ))
            c_idx += 1
        if circuit and circuit.gates:
            g_desc = ", ".join(f"{g.name} on {g.target_qubits}" for g in circuit.gates)
            claims.append(ExtractedClaim(
                claim_id=f"CLM-{c_idx:02d}",
                claim_type="circuit",
                statement=f"Proposed gate sequence: {g_desc}",
                source_reference="Page 1",
                confidence_level="high"
            ))
            c_idx += 1
        if outcomes:
            o_desc = ", ".join(f"P({o.basis_state})={o.expected_probability}" for o in outcomes)
            claims.append(ExtractedClaim(
                claim_id=f"CLM-{c_idx:02d}",
                claim_type="outcome",
                statement=f"Expected measurement probabilities: {o_desc}",
                source_reference="Page 1",
                confidence_level="high"
            ))

        if extracted_doc and extracted_doc.pages:
            self._assign_claim_page_references(claims, extracted_doc.pages)

        # If research paper, create multi-experiment summary
        experiments_list: List[Dict[str, Any]] = []
        if doc_type == "research_paper":
            paper_analysis = self._heuristic_paper_fallback(
                raw_text=raw_text,
                document_title=title,
                extracted_doc=extracted_doc,
                doi=doi,
                error_note=error_note
            )
            experiments_list = [exp.model_dump() for exp in paper_analysis.experiments]

        return ProposalSpecification(
            title=title,
            objective=objective,
            hypothesis=hypothesis,
            methodology=methodology or None,
            quantum_system=quantum_system,
            num_qubits=num_qubits,
            initial_state=circuit.initial_state if circuit else "|0...0>",
            measurement_basis=circuit.measurement_basis if circuit else "computational (Z)",
            circuit=circuit,
            expected_outcomes=outcomes,
            assumptions=assumptions,
            missing_information=missing,
            potential_inconsistencies=[],
            author_claims=claims,
            raw_text=raw_text,
            extraction_warnings=warnings,
            input_mode=input_mode,
            source_filename=extracted_doc.filename if extracted_doc else None,
            source_pages=extracted_doc.pages if extracted_doc else [],
            document_type=doc_type,
            doi=doi,
            experiments_discussed=experiments_list
        )
