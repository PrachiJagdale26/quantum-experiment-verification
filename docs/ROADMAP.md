# Implementation roadmap: LLM-based research gap identification and experiment planning (quantum physics)

Status: **approved 2026-11-27.** Phase 1 (foundations) and Phase 2 (literature collection) **done 2026-11-27**. Phase 3 (research analysis), Phase 4 (gap discovery and novelty) and Phase 5 (planning with edit and approval) and Phase 6 (verification of approved plans) **done 2026-11-28**. Next: Phase 7 (orchestration).

Phase 6 results:
- **Adapter (`src/verification/plan_verifier.py`):** an approved plan revision becomes a `ProposalSpecification`. Assumed parameters are listed as assumptions; a missing circuit stays missing, so the plan is held with the missing items listed, never filled in.
- **Checks.** The existing engine runs checks 1–7 (structure, qubit indices, gates, circuit, normalisation, predicted vs simulated, TVD). New:
  - **8 State fidelity:** against the plan's expected state (amplitudes, including phase), threshold 0.99. It catches a GHZ-minus target, whose measurement probabilities are identical to GHZ-plus.
  - **9a Noise parameters:** probabilities in [0, 1], positive times, T2 ≤ 2·T1. An impossible combination fails.
  - **9b Predictions under noise:** checked against the plan's noisy predictions when it gives them. Otherwise the effect is reported (TVD to ideal, fidelity, whether the ideal prediction still holds) without a judgement.
  - **Assumptions:** lists any simulated value resting on an assumption, including the default gate time used for T1/T2.
  - **Hardware status:** states that simulation is not hardware validation, using the plan's feasibility class.
  - The verdict uses the existing fixed rules; any failure fails it, and no model output can change it.
- **Noise (`src/quantum/noise.py`):** exact density-matrix simulation with qiskit-aer 0.18.4 (depolarizing, thermal relaxation from T1/T2 and gate times); readout error is applied analytically. Deterministic, with no shot noise.
  - Plan parameters map onto noise by name; a range runs at both ends.
  - When two parameters set the same quantity, the researcher's beats the literature's beats an assumption, and the conflict is reported.
- **Stored runs:** each run is a `VerificationRun` on the exact revision. Only an approved revision can be verified; after an edit, the old run is shown as covering an earlier revision.
- **UI:** a proposal has Plan, Verification, Checks and Report tabs.
  - The Verification tab reuses the probability disks, circuit drawing, state-overlap and TVD panels, plus a noise table showing each value's source.
  - "Verify revision N" appears only on an approved plan; runs are listed in the audit trail.
- **Live check (scratch copy):** Gemini's error-correction plan predicted that a 3-qubit GHZ circuit gives `000` with probability 1.
  - Verification failed it ("Predicted P(000) = 1.00; the circuit gives 0.50", TVD 0.50).
  - In the browser: corrected the prediction and added the target state → revision 2 (draft, the old run marked as revision 1) → approved → verified. It passed with state fidelity 1.000; under the stated depolarizing range the fidelity is 0.998 to 0.898.
- **Known limitations:**
  - Noise parameters are recognised by name (depolarizing / error rate, readout, T1, T2, gate time); other noise is not modelled.
  - A range runs at its two ends, not swept.
  - The literature-grounding check still uses the starter knowledge base, not the research corpus.
- **Tests:** 9 new offline tests covering the noise physics (readout flips, relaxation towards |0⟩, T2 > 2·T1), parameter mapping and precedence, a wrong GHZ prediction, fidelity including the phase case, noisy predictions with flagged assumptions, impossible noise, an unspecified plan, and approval gating. 157 pass; the 3 Vertex tests are deselected because billing is disabled.

Phase 5 results:
- **Research proposal (`src/experiment_planning/schema.py`):** all 20 sections Module 6 lists, plus a feasibility class (simulatable, requires quantum hardware, or requires additional physical validation) with its reason, an optional circuit with theoretical predictions, missing information, warnings, and who wrote it.
- **Grounding (`proposal_planner.py`).** The model may cite only the gap's evidence papers and the closest papers from its novelty investigation, by id. Citations to anything else are dropped.
  - Every quote is checked against the stored title, abstract and verified statements. An unverified justification is kept but flagged.
  - A "literature" parameter whose quote is not found becomes an explicit assumption that says so.
  - An invalid circuit is removed with a warning (gate qubits outside the register, bitstrings of the wrong length).
  - Feasibility is decided by a deterministic rule; the model may raise it but never lower it.
  - The prompt passes the gap's novelty status and forbids novelty claims.
- **Without a live model,** the template fills only what the gap determines (problem, justification from verified evidence and closest prior work, system, techniques, baselines). It lists the required sections still empty under missing information instead of inventing them.
- **Edits and approval (`plan_store.py`).** Each edit is stored as a new revision recording the fields changed and the researcher's note. Changed parameters become "set by the researcher"; unchanged ones keep their source.
  - Approval needs every required section (research question, hypothesis, methodology, dependent variables, metrics). Rejecting or requesting changes needs a note.
  - A decision is an `Approval` row on the exact revision. Editing an approved plan creates a new draft, so approval never carries over.
- **UI:** the Plans step lists proposals with status, revision and feasibility. Each proposal view shows:
  - its sections, missing information and warnings;
  - literature justifications with DOI links and labels: "Quote verified", "Quote not found in the source", or "Inference, no quote";
  - a parameter table with sources, and the circuit;
  - an edit form (text, lists, parameters, circuit as JSON, feasibility, note), the decision panel and the audit trail.
  - "Plan an experiment" on a corpus gap drafts a proposal. Uploaded-paper gaps keep the older plan-and-verify flow.
- **Live check (scratch copy):** for "efficient decoders for measurement-induced phases", Gemini wrote a proposal citing the neural-network decoder paper the novelty check found, with a verified quote. It listed the circuit as missing instead of inventing one.
  - For "error correction under depolarizing noise", its circuit predicted that a 3-qubit GHZ state gives `000` with probability 1 (the correct result is ½ `000` + ½ `111`). Phase 6 verification is there to catch exactly this.
  - In the browser: edit → revision 1 with changed fields; approval recorded on revision 1; a rejection with a short note is refused.
- **Known limitations:**
  - Plan ids skip numbers (RP-1, RP-2, RP-4) because the id is the first revision's row.
  - Plans for gaps without a novelty check have fewer sources to cite.
  - Plan quality is not yet rated (Phase 9 rubric).
- **Tests:** 5 new offline tests (feasibility rules, a template that invents nothing, grounding against a fabricating model, the edit/approve/audit trail, circuit validation). 148 pass; the 3 Vertex tests are deselected because billing is disabled.

Phase 4 results:
- **Gap generation (`src/gap_discovery/gap_generator.py`).** Gaps come only from verified statements (limitations, unresolved problems, future work); statements several papers share are merged into one gap. Untried method × platform and method × noise pairs become "untried in this corpus" gaps, labelled as inference.
  - Each dimension records its source: the gap's own text, the source paper, or corpus coverage. An optional model pass rewrites gaps as research questions; it is labelled and leaves the evidence untouched.
  - Re-running updates gaps by a stable key and never touches assessments or your review.
- **Novelty investigation (`novelty_checker.py`).** It follows the methodology in section 3:
  - Queries: the gap's own words, those words plus the main phenomenon, a concept query, an ontology synonym, and method- and condition-focused queries. Long queries skip arXiv, which requires every word to match.
  - Sources: OpenAlex, arXiv and Crossref, the works citing the supporting papers, and your corpus. Every search is a stored `SearchRun`.
  - Supporting papers are removed from the comparison. Each closest paper is compared dimension by dimension (same, different or not stated) with the sentence quoted. An optional model judge is used only when its quote is found in the abstract.
  - The classification follows fixed rules. Low coverage gives "insufficient search coverage", never "novel". Confidence is stored with its formula and inputs.
- **Problems the live checks exposed, fixed before measuring:**
  1. Dimensions borrowed from a long source paper over-specified gaps, which pushes towards false novelty. A gap's own dimensions now define it; borrowed ones only top up to two.
  2. Embedding similarity let generic titles crowd out the papers that share the question: about 40 MIPT papers were screened but none were compared. The comparison set now favours shared dimensions.
  3. Template words and citation numbers leaked into queries.
  4. Generic values ("entanglement") matched any paper and are now dropped when a dimension names something specific.
- **Ranking (`gap_ranker.py`):** eight separate scores, each with a rubric and a one-line reason. Novelty comes from the classification, not from similarity. The weighted total uses weights you see and can change.
- **Corpus separation:** works found while checking novelty are marked `discovered_via = novelty` and never enter the Literature list, corpus analysis or "Analyse next".
- **UI:** the Gaps step lists corpus gaps by weighted total with origin and classification. Each gap shows its evidence, the dimensions it is compared on, and the full investigation: coverage, queries, the closest papers with per-dimension results and the deciding paper, evidence for and against, and uncertainty.
  - Ranking weights can be edited; there are review buttons (candidate, validated, rejected) and "Plan an experiment", which feeds the existing planner.
  - Checked in the browser on a copy of the corpus.
- **Benchmark (`python evaluate.py --novelty`).** It runs on a copy of the database; the report is `data/benchmark/novelty_known_work_20261127_1949.json`. On 15 published questions from the entanglement/decoherence corpus:
  - **False-novelty rate 0/15** (95% CI 0–0.20). All 15 were classified as already investigated.
  - The source paper was screened in 93% of cases and compared in 87% (MRR 0.78); it decided the verdict 11 times.
  - **Caveat:** the 15 questions have only 4 distinct dimension sets, mostly entanglement + decoherence (the corpus topic). On its own, this set cannot separate a good checker from one that always says "already investigated".
  - **Specificity control:** 5 of 5 implausible combinations were not called already investigated (2 potentially unexplored, 3 partially). The positive control (error correction on trapped ions under depolarizing noise) was detected.
  - Not yet measured: partially-investigated accuracy and the future-work set (both need human labels), and runs with the model judge.
- **Known limitations:**
  - Ontology terms have no word sense: "leakage" matches privacy leakage, and "integrated photonics" in a trapped-ion paper counts as photonic.
  - Comparisons use abstracts, not full text.
  - Conflicting findings are not detected yet; the class exists but no rule produces it.
  - Side-by-side comparison of gaps is deferred to Phase 8.
  - A stronger benchmark needs the planned 3 topics × 40 papers.
- **Tests:** 11 new offline tests use fake collectors. They cover merging, idempotent saving, the dimension rules, already / insufficient / unexplored outcomes, supporting-paper exclusion, discarding an unverified model judgement, the own-dimension rule, ranking, and the benchmark. All 143 tests pass (offline plus live Gemini); the 3 Vertex tests are deselected because billing is disabled.

Phase 3 results:
- **Paper analysis (`src/research_analysis/paper_analyzer.py`).** Every statement (objective, method, result, limitation, unresolved problem, future work, assumption, and so on) carries provenance:
  - `direct` means a verbatim quote, `interpretation` means a paraphrase whose quoted evidence was found in the paper, `unverified` means the evidence could not be found (confidence ≤ 0.2, never used downstream), and `inference` is reserved for cross-paper conclusions.
  - Evidence is checked against the paper text after accent and whitespace normalisation (fuzzy partial match ≥ 92), and the page is recorded when full text exists.
  - The reference list is stripped before analysis (a heading, an "Online content" block, or a trailing numbered bibliography), so cited titles are no longer tagged as the paper's own work.
  - Rule-based sentence extraction adds the paper's own limitation, open-problem and future-work sentences that the model missed; it also runs alone when no LLM is available.
  - LLM JSON containing LaTeX (`\rho`, `\alpha`) is parsed safely (`src/llm/json_utils.py`).
- **Research dimensions:** ontology tagging (`src/knowledge_base/ontology.py`) of methods, phenomena, platforms, noise and qubit counts, stored as `dim:*` extraction records.
- **Cross-paper analysis (`cross_paper_analysis.py`).** It covers only analysed papers that are not excluded and not synthetic.
  - Themes come from average-linkage clustering of embeddings and are labelled by distinctive concepts, with distinctive title words as a fallback.
  - Method × platform and method × noise coverage matrices; "untried in this corpus" pairs are explicitly labelled as not establishing novelty.
  - Recurring limitations, open problems, future work and assumptions are built from verified statements only.
- **Hybrid retrieval:** BM25 combined with semantic search by reciprocal rank fusion, with ontology query variants (exact 1.0, synonym 0.6, related 0.3); every hit reports how it was found.
- **Curation:** papers can be included, excluded or left undecided, and excluded papers leave all corpus analysis. Old databases gain the new columns by migration.
- **UI:** an analysis panel with kind badges and supporting passages, curation buttons, "Analyse next 5", and a Corpus analysis view, all checked in the browser on the real database.
- **Live check (scratch copy of the database):** 22 papers analysed in 53 s, with 97% of the model's statements verified against the text.
  - Themes separated sensibly after the switch from single to average linkage, which had chained everything into one cluster.
  - No limitation recurred across papers: the highest cross-paper similarity was 0.42, below the 0.62 threshold. Most records are abstracts only; full text should be fetched before this is read as a finding.
- **Known limitation:** the verified share varies between runs of the same paper (71% vs 93% seen at temperature 0), because the model's evidence quotes vary. Unverified statements are kept but labelled and excluded, so this affects coverage, not correctness.
- **Tests:** 119 offline tests pass (9 analysis tests and 5 JSON-parsing tests are new). A fake model returns a fabricated quote, so the verification step is tested honestly.

Phase 2 results:
- **`src/literature/`.** Collectors for OpenAlex, arXiv, Crossref and Semantic Scholar share one base: a per-source rate limiter (arXiv 3.1 s between requests), retries on 429/5xx, and an on-disk response cache for a week, with API keys excluded from cache keys.
  - OpenAlex supports search, DOI, author, by-id and citing-works lookups.
  - Crossref keyword search is restricted to journal articles, proceedings, posted content and chapters; it otherwise returns peer-review records and theses.
- **Metadata normalisation:** DOI, versioned arXiv ids, titles (LaTeX, HTML and accents removed), OpenAlex inverted abstracts and JATS markup.
- **De-duplication:** DOI → arXiv id → OpenAlex id → fuzzy title (year within one, same first author). Records with different DOIs or arXiv ids are never merged.
  - The journal DOI is preferred over an arXiv DataCite DOI.
  - arXiv and published versions are recorded with dates.
  - Disagreeing identifiers are kept and reported, never overwritten.
- **Ingestion:** every search is a `SearchRun` with its status (ok, cached, rate-limited, failed), counts and the total available at the source; hits are stored with their ranks.
  - Metadata-only records are kept and flagged.
  - Abstracts are indexed into the knowledge base.
  - Full text is fetched only from arXiv (open access), one paper at a time on request.
- **UI:** a Literature step (first in the flow) with topic presets, years, sources, results per source and DOI lookup, a per-source collection report and search history.
  - The rail filters by text, availability, source and year.
  - The detail view shows identifiers, retrieval times, versions, availability and work type, plus a "Preprint or repository deposit" marker.
- **Live check:** "quantum error mitigation" since 2019, 15 per source: 45 results became 41 unique publications in 6.4 s, with 4 cross-source merges, 3 metadata-only records and no identifier conflicts.
  - Semantic Scholar is rate-limited without a key, and recorded as such.
  - "Entanglement and decoherence" since 2020, 10 per source, gave 27 unique publications. Its top OpenAlex hit was a speculative Zenodo deposit, which is now marked as a preprint rather than filtered out. Corpus curation is a researcher decision (Phase 3).
- **Tests:** 21 literature tests on captured real API responses (offline); 105 offline tests pass in total.

Phase 1 results:
- **Dependencies installed:** chromadb 1.5.9, sentence-transformers 6.3.0, pymupdf 2.2, qiskit-aer 0.18.4, rank-bm25, rapidfuzz, reportlab, langgraph-checkpoint-sqlite.
- **Research database:** `src/storage/` (SQLite via SQLAlchemy). Tables for publications, versions, provenanced extraction records, search runs and hits, candidate gaps and their evidence, novelty assessments, dimension comparisons, plans, approvals, verification runs, reports, and an LLM call log.
- **Knowledge base:** semantic (ChromaDB, cosine, one collection per embedding model). It reports its backend and refuses degraded mode when `KB_REQUIRE_SEMANTIC=1`. The embedding model loads once per process.
- **LLM:** Gemini API is primary. Configurable fallback models; overloaded models cool down; every call is labelled live or demo and logged without prompt text. No path returns canned demo JSON as if it were analysis.
- **Correctness fixes found on the way:**
  - the gate parser now keeps document order (it used to group gates by type);
  - `eval` on untrusted rotation angles is replaced by a safe parser;
  - "control/target" phrasing is supported;
  - prose gates and outcomes from real papers no longer discard the whole LLM analysis.
- **Tests:** 84 offline tests pass, and the 8 live Gemini integration tests pass. The 8 Vertex tests are excluded because billing is disabled.
Audit date: 2026-11-27. Every finding below was checked on this machine; nothing is assumed.

**Research question:** can an LLM-based framework identify potentially unexplored research directions from scientific literature, assess their novelty against existing publications, and turn them into verifiable experiment plans?

**Primary contribution:** a novelty-aware gap discovery pipeline whose false-novelty rate is measured on a labelled benchmark.

---

## 1. Audit: what exists today

### 1.1 Working and reusable

| Area | Where | State | Reuse |
|---|---|---|---|
| Document extraction (PDF/DOCX/TXT) | `src/document_processing/` | Works. Page provenance, section regexes, PDF line re-joining (`unwrap_pdf_text`), title detection (`guess_title`). Uses PyPDF2. | Keep. Upgrade to PyMuPDF for better layout. |
| Chunking with provenance | `src/document_processing/chunker.py` | Works (chunk id, page, char span). | Keep. |
| LLM client | `src/llm/client.py` | Vertex / Gemini / OpenAI / Anthropic / demo. Now has `fallback_to_demo=False` and a circuit breaker for auth failures. | Keep. Add structured-output helper, retries with backoff, per-call logging, model fallback list. |
| Paper analysis schema + extractor | `src/schemas/research_paper.py`, `src/llm/extractor.py` | LLM + rule-based fallback. Limitations, future work, open questions, claims, experiments. | Extend into the full research-dimension schema with per-field provenance. |
| Prompt-injection defence | `src/llm/prompts.py` | `<UNTRUSTED_*>` wrapping on every prompt. | Keep the pattern for all new prompts. |
| Qiskit verification | `src/circuit/`, `src/verification/` | Statevector simulation, TVD, fidelity, 6 rule checks, PASS/FAIL/INCONCLUSIVE/NOT_RUN. The LLM cannot override a computational FAIL. | Keep as Module 7. Add a noise model (qiskit-aer) and an "unsupported assumption" check. |
| Gap finder (v1) | `src/research/gap_finder.py` | Five deterministic detectors with verbatim evidence and merging. Runs offline. | Keep as the **candidate generator** front-end. Replace its novelty scoring (see 1.3). |
| Experiment planner (v1) | `src/research/planner.py` | LLM or theory-based templates (Bell/GHZ/VQE). One revision loop after verification. | Keep. Extend the plan schema and add an edit/approve step. |
| LangGraph verification graph | `src/agents/` | 5 agents, conditional routing, bounded retries, human-review node. The paper-branch crash is fixed. | Keep for verification. Add a research graph. |
| UI | `overlap_app.py`, `src/ui/overlap/`, `app.py` | Overlap app (Papers → Gaps → Plans) plus the themed full dashboard. | Extend the Overlap sections. Keep the dashboard. |
| Tests | `tests/` | 75 offline tests passing. 16 live-API tests deselected. | Keep. Extend. |

### 1.2 Missing entirely

- **Module 1, literature collection.** No OpenAlex, arXiv or Crossref client. No publication database. No deduplication, version tracking or retrieval timestamps.
- **Persistent store for research objects.** Papers, gaps, assessments and approvals live only in Streamlit session memory and are lost on restart.
- **Module 4E/F, novelty investigation.** No external search, no query expansion, no dimension-level comparison, no novelty classification or coverage reporting.
- **Module 5, multi-dimension ranking** (8 separate dimensions with explanations).
- **Module 6, editable and approvable plans.** No human approval state is persisted.
- **Module 8, PDF report export.** No PDF library is installed.
- **Module 9, evaluation benchmarks** for retrieval, extraction, gap classification and novelty.

### 1.3 Problems found that affect scientific validity (fix first)

1. **The vector knowledge base was never semantic.** `chromadb`, `sentence-transformers` and `pymupdf` are listed in `requirements.txt` but **not installed** in `.venv`. Every retrieval so far has used `FallbackVectorStore`, which does word-count matching (its docstring wrongly says TF-IDF). The README claims ChromaDB. Fix: install them, report the active backend in the UI and logs, and fail loudly instead of degrading silently.
2. **Novelty is currently `1 − max similarity`** (`GapFinder._novelty`). That's exactly the weak definition this project must not use. It is replaced by the multi-stage investigation in §3.
3. **The sample papers are synthetic.** They are fine for demos and unit tests but are **not valid evidence** and must never enter the evaluation benchmark or reported results.
4. **Direct statements vs. interpretations are not separated.** Extracted fields don't record whether they are a verbatim quote, an LLM paraphrase or a cross-paper inference.
5. **The canned demo output used to masquerade as real analysis.** That's fixed for the research path. The remaining legacy callers (proposal extraction, analyzer) still default to `fallback_to_demo=True` and must be migrated, so outputs are labelled live or demo per call.

### 1.4 External services: verified 2026-11-27 from this machine

| Service | Result | Needed from you |
|---|---|---|
| OpenAlex | Works without a key, but keyless requests share a small daily budget per IP. | A free personal API key from openalex.org/settings/api, put in `.env` as `OPENALEX_API_KEY`. **I won't request it on your behalf**, because that needs your email. |
| arXiv API | Works over **https only**. | Nothing. Rate limit: 1 request per 3 s. |
| Crossref | Works and returns real metadata. | Optional `CROSSREF_MAILTO` for the polite pool (your choice of address). |
| Semantic Scholar | **429 rate-limited** without a key. | Optional key (apply on their site). Treated as an optional source. |
| Gemini API (your key) | **Authenticates.** `gemini-3.8-flash` returned 503 "high demand"; `gemini-flash-lite-latest` answered. Embedding model `gemini-embedding-001` is available. | Switch `LLM_PROVIDER=gemini` (Vertex is blocked by billing), or enable Vertex billing. |
| Vertex AI | 403, billing disabled on the project. | Enable billing, or use the Gemini API. |

### 1.5 Dependencies to install

`chromadb`, `sentence-transformers` (pulls PyTorch, about 1–2 GB), `pymupdf`, `qiskit-aer` (noise simulation), `rank-bm25` (keyword half of hybrid retrieval), `rapidfuzz` (fuzzy title dedup), `reportlab` (PDF export), `langgraph-checkpoint-sqlite` (persistent checkpoints).
Already present: `sqlalchemy`, `diskcache`, `tenacity`, `httpx`, `scikit-learn`, `markdown`, `langgraph`.

---

## 2. Target architecture

New packages follow the suggested layout. Existing modules are reused, and nothing that works is deleted. `src/research/` is split into the new packages, with re-exports kept so current imports and tests keep working.

```
src/
  storage/                      NEW  SQLite (SQLAlchemy) single source of truth
    db.py                            engine/session, migrations (create_all)
    models.py                        Publication, PublicationVersion, Author, Chunk, ExtractionRecord,
                                     ResearchDimension, CandidateGap, GapEvidence, NoveltyAssessment,
                                     SearchRun, SearchHit, DimensionComparison, ExperimentPlan, Approval,
                                     VerificationRun, Report
  literature/                   NEW  Module 1
    collectors/base.py               Collector protocol, rate limiter, retry (tenacity), disk cache (diskcache)
    collectors/openalex.py           search, by DOI, by author, referenced_works / cited_by
    collectors/arxiv.py              Atom API over https, versions (v1, v2…), PDF link for open access
    collectors/crossref.py           DOI metadata, search
    collectors/semantic_scholar.py   optional, key-gated
    metadata.py                      normalisation, DOI/arXiv-id validation, open-access flags
    deduplication.py                 DOI → arXiv id → fuzzy title + year + first author (rapidfuzz)
    ingestion.py                     query → collect → dedup → store → (optional OA PDF) → extract → index
  document_processing/          EXISTS  + section_parser.py (PyMuPDF layout-aware sections)
  knowledge_base/               EXISTS  rework
    embeddings.py               NEW  sentence-transformers (SPECTER2 / all-MiniLM baseline) or gemini-embedding-001
    vector_store.py                  Chroma, persistent; backend reported, no silent fallback
    lexical.py                  NEW  BM25 over the same chunks
    retriever.py                     hybrid (reciprocal-rank fusion), metadata filters, citations per hit
    ontology.py                 NEW  curated quantum concept map (decoherence ↔ dephasing ↔ T1/T2 ↔ noise …)
  research_analysis/            NEW  Module 2 / 4A / 4B
    research_schema.py               ResearchDimensions + Provenance{paper, page, section, excerpt,
                                     kind: direct|interpretation|inference, confidence}
    paper_analyzer.py                LLM structured extraction (Pydantic, temperature 0), evidence-span check
    cross_paper_analysis.py          coverage matrix, repeated limitations, conflicts, untested assumptions, themes
  gap_discovery/                NEW  Module 4 C–F, Module 5
    gap_generator.py                 deterministic detectors (from research/gap_finder.py) + LLM synthesis
                                     constrained to cited evidence
    gap_classifier.py                7 classes, rule-based over novelty findings
    gap_comparator.py                duplicate / overlapping gap detection
    novelty_checker.py               the multi-stage novelty investigation (§3)
    gap_ranker.py                    8 separately scored dimensions + transparent prioritisation
  experiment_planning/          NEW  Module 6 (from research/planner.py)
    question_generator.py, hypothesis_generator.py, experiment_planner.py
                                     simulatable / needs hardware / needs physical validation labels
  quantum/                      EXISTS as src/circuit + src/verification  + noise.py (qiskit-aer), assumption checks
  agents/                       EXISTS  + research_graph.py (Module 6 optional agentic layer)
  reporting/                    EXISTS  + research_report.py (Markdown, JSON, PDF via reportlab)
  evaluation/                   EXISTS  + retrieval_eval.py, extraction_eval.py, novelty_eval.py, gap_eval.py
data/
  corpus/                       retrieved metadata snapshots (JSON, timestamped)
  benchmark/                    labelled sets (see §5), each item with labeller + review status
reports/                        generated reports
```

---

## 3. Novelty assessment methodology (core contribution)

Novelty is **not** a similarity score. It is a recorded investigation with a rule-based conclusion and explicit uncertainty.

1. **Decompose the gap** into a `CoreQuestion` with research dimensions: concept, method/algorithm, platform, qubit scale, noise condition, measurement, metric, objective. Each dimension is tagged with its source (gap evidence, or LLM interpretation).
2. **Expand the queries.** Exact phrase, synonym expansion from the curated ontology (deterministic), LLM-proposed alternative terminology (labelled as such), a method-focused query and a condition-focused query. Every query is stored.
3. **Retrieve from multiple sources.** OpenAlex, arXiv and Crossref (plus Semantic Scholar if keyed) and the local hybrid knowledge base. **Citation exploration:** references and citing works of the supporting papers via OpenAlex. Each source records success or failure, hit count and timestamp.
4. **Screen.** Deduplicate (DOI, arXiv id, fuzzy title). Record publication date relative to the supporting evidence. Distinguish abstract-only from full-text records.
5. **Compare dimensions** for the top related papers. For each dimension, the result is `same / different / not stated`:
   - deterministic matchers where possible: ontology normalisation, qubit ranges, platform names;
   - an LLM judge otherwise. It must quote the abstract or text span, and the quote is checked to actually occur in the source. A judgement whose quote can't be found is discarded.
6. **Classify with deterministic rules over the comparisons:**
   - a paper matching **all core dimensions** with verified evidence → *Already investigated*;
   - matching most dimensions but differing in at least one key dimension → *Partially investigated*;
   - opposing findings on matched dimensions → *Conflicting research findings*;
   - no close match **and** search coverage above the threshold → *Potentially unexplored in the searched literature*;
   - coverage below the threshold (a source failed, few queries, abstract-only) → *Insufficient search coverage*. This is **never** labelled novel.
   - Plus *Methodological limitation* and *Unresolved future work* from the gap's origin.
7. **Report:** classification, confidence (derived from coverage, agreement and evidence type, with the formula shown), databases, queries, date, counts, closest papers with similarities and differences, evidence for and against novelty, remaining uncertainty. The wording is always "in the searched literature"; it never claims "never published".

**Ranking (Module 5):** eight dimensions, each scored 0–1 by a stated rubric with a one-line justification: novelty (from the classification, not similarity), significance, evidence strength, feasibility, testability, tool availability, potential contribution, and literature coverage. No hidden composite: the UI shows all eight, a researcher-adjustable weighting with its weights displayed, and a side-by-side comparison of selected gaps.

---

## 4. LangGraph research graph (only where it adds reliability)

`collect → analyse → discover gaps → [human: select gap] → novelty investigation (bounded re-query ≤ 2 if coverage is low) → plan → [human: edit/approve] → verify (existing verification graph as a subgraph; revise ≤ 1) → report`.
Typed state, SQLite checkpointing (so a run resumes after a restart), execution log per node with tool calls and timings. Deterministic steps stay plain functions called by nodes; no agent just exchanges text.

---

## 5. Evaluation plan: measured, never invented

| Benchmark | How ground truth is obtained | Metrics |
|---|---|---|
| **Known-work novelty set** (main) | Take real papers retrieved from OpenAlex/arXiv. Turn each paper's own research question into a "candidate gap", **remove that paper from the local corpus**, and run the novelty check. Ground truth is objective: it *is* published. | **False-novelty rate** (labelled novel although published), recall of the source paper, MRR of the source paper |
| Partially investigated set | Known questions with one dimension changed (platform, scale, noise). **Labels need human review.** I draft candidates; you or an expert confirm. | Classification accuracy, confusion matrix |
| Future-work set | Future-work sentences from papers 3+ years old. A human checks whether later work addressed them. | Agreement with human labels |
| Literature retrieval | The known-work set's queries plus a small topic set with relevance judgements. | Precision@K, Recall@K, MRR |
| Extraction | 15–20 real open-access papers, fields annotated by a human. I can pre-fill drafts, clearly marked unreviewed until checked. | Field accuracy, evidence attribution accuracy, missing-field detection |
| Gap quality | Expert review sheet: relevance, evidence support, duplicates. | Mean ratings, inter-rater agreement if 2+ reviewers |
| Experiment plans | Rubric: consistency, completeness, feasibility, alignment with the gap. Human-rated. | Rubric scores |
| Quantum verification | Existing cases (Bell, GHZ, wrong probabilities, invalid index, incomplete) plus an unsupported-assumption case and noise cases. | False-pass / false-fail rates, simulation correctness |

All results are reported with dataset size, procedure, date, model versions and limitations. Synthetic sample papers are excluded.

**Corpus scope** (per your guidance): start with 3 topics × about 40 papers (about 120, metadata plus abstracts; open-access arXiv full text where available), e.g. quantum error mitigation, variational algorithms, entanglement/decoherence. Scale only after the evaluation holds up.

---

## 6. Phases: file by file

Each phase ends with tests run and failures reported.

| Phase | Deliverables (files) | Done when |
|---|---|---|
| **1. Foundations** | Install deps; `src/storage/{db,models}.py`; knowledge base reports its backend; Chroma + sentence-transformers live; `LLM_PROVIDER=gemini` path with model fallback and retries; remaining legacy `fallback_to_demo` callers migrated; README correction about Chroma | Tests green; the knowledge base reports "chroma + all-MiniLM" (or SPECTER2); a live Gemini extraction succeeds |
| **2. Literature collection** | `src/literature/collectors/{base,openalex,arxiv,crossref,semantic_scholar}.py`, `metadata.py`, `deduplication.py`, `ingestion.py`; Literature Explorer section | A real topic query stores N deduplicated records with source URLs and timestamps; metadata-only vs full text is flagged; collector unit tests use recorded responses |
| **3. Research analysis** | `research_analysis/{research_schema,paper_analyzer,cross_paper_analysis}.py`; `knowledge_base/{embeddings,lexical,ontology}.py`, hybrid `retriever.py` | Every extracted field carries provenance and kind (direct, interpretation or inference); quotes are checked to occur in the source |
| **4. Gap discovery + novelty** | `gap_discovery/{gap_generator,gap_classifier,gap_comparator,novelty_checker,gap_ranker}.py`; Gaps and Novelty UI | Known-work benchmark runs and the **false-novelty rate is measured and reported** |
| **5. Planning** | `experiment_planning/*`; editable plan form plus approval stored in the database | A plan can be edited, approved and audited; parameters show their source or assumption |
| **6. Verification** | Plan → `ProposalSpecification` adapter; `quantum/noise.py` (qiskit-aer); assumption checks | Approved plans verify; noise cases are tested |
| **7. Orchestration** | `agents/research_graph.py` with SQLite checkpoints and human interrupts | An end-to-end run resumes after a restart; the execution log is shown |
| **8. Reports + UI** | `reporting/research_report.py` (MD/JSON/PDF); Overview, Novelty, Verification and Reports sections; generated vs verified content visually distinct | All 17 report sections are present with DOI links |
| **9. Evaluation** | `evaluation/{retrieval,extraction,novelty,gap}_eval.py`, `evaluate.py --all` | Real measured numbers with procedure and limitations |
| **10. Documentation** | README rewritten per your outline | Every section is backed by the implemented behaviour |

---

## 7. Risks and honest limits

- **Novelty can only be relative to the searched corpus.** Paywalled full text isn't accessed, so many comparisons are abstract-only; the confidence score reflects this.
- **Rate limits** (OpenAlex budget, arXiv 3 s, S2 429) bound corpus size per session. Caching makes runs reproducible.
- **Human labels are the bottleneck.** Only the known-work set has fully objective ground truth. The other benchmarks need your or an expert's review time; I'll provide the labelling sheets.
- **LLM availability varies** (the 503s seen today). Every LLM step records model, version and whether it ran live, and none fall back to canned content.
- **PyTorch** for sentence-transformers is a large install. `gemini-embedding-001` is an API alternative, but it adds cost and rate limits.
