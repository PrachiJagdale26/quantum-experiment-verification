# Implementation roadmap: LLM-based research gap identification and experiment planning (quantum physics)

Status: **approved 2026-11-27.** Phase 1 (foundations) and Phase 2 (literature collection) **done 2026-11-27**. Next: Phase 3 (research analysis).

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
