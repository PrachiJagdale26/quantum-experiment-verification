# Overlap: LLM-Based Research Gap Identification and Experiment Planning for Scientific Literature

## 1. Project overview

Overlap is a research assistant for quantum physics literature. It collects publications from scholarly databases, analyses them with every extracted statement tied to a quote it has verified in the source, and finds candidate research gaps across papers. It then checks each gap's novelty against the wider literature, drafts an experiment plan for a gap the researcher chooses, and verifies the approved plan by Qiskit simulation. The researcher makes every decision: which papers count, which gap to pursue, and whether a plan is approved. The result is a 17-section report in which every statement is labelled by where it came from.

```
 Literature ──► Analysis ──► Gaps ──► Novelty check ──► Plan ──► [researcher approves] ──► Verification ──► Report
 OpenAlex,      quotes       stated     searches,          20-section     edit / approve /         Qiskit statevector    17 sections,
 arXiv,         verified     limits,    compares,          proposal,      reject, per revision     + density-matrix      provenance
 Crossref       against      future     classifies         parameters                              noise                 labels, DOIs
                the source   work,      by rule            with sources
                             untried
                             combinations
```

The same steps can run as a **guided run**: a checkpointed workflow that stops for the researcher's gap choice and plan approval, and resumes after a restart.

## 2. Research motivation

Finding what has not been done yet is slow, and LLMs make it look easy: they produce plausible "open problems", "novel ideas" and citations without evidence, and similarity scores are often presented as novelty. In quantum computing the cost of a wrong plan is concrete: circuits and predicted outcomes can be wrong in ways a reader does not notice. The motivation is a system whose outputs can be checked: claims trace to quotes, novelty is a recorded search rather than a score, and plans are executed rather than trusted.

## 3. Research problem

Can an LLM-based framework identify potentially unexplored research directions in scientific literature, assess their novelty against existing publications, and turn them into verifiable experiment plans, **without fabricating evidence and without presenting semantic similarity as scientific novelty**?

## 4. Objectives

1. Collect and de-duplicate publications from OpenAlex, arXiv and Crossref with full provenance (source, identifiers, retrieval time).
2. Extract research problems, methods, systems, results, limitations and future work, with every model statement checked against the source text.
3. Generate candidate gaps only from verified statements and from measurable absences in the corpus.
4. Assess novelty with a recorded, multi-source investigation and deterministic classification rules, including an explicit "insufficient search coverage" outcome.
5. Draft experiment plans whose parameters state their source, and keep the researcher's approval as a required gate.
6. Verify approved plans with deterministic Qiskit simulation, including state fidelity and noise conditions.
7. Report everything with provenance, and measure the framework's error rates instead of asserting them.

## 5. Literature review workflow

- **Collection** (`src/literature/`): queries OpenAlex, arXiv and Crossref (Semantic Scholar optional) with per-source rate limits, retries and a response cache. Every search is stored as a `SearchRun` with its status (ok, cached, rate-limited, failed) and ranked hits.
- **De-duplication**: by DOI, then arXiv id, then OpenAlex id, then fuzzy title (with the year and first author). Records with different DOIs or arXiv ids are never merged. arXiv and journal versions are kept with their dates.
- **Curation**: the researcher includes or excludes papers; excluded papers never feed gaps or themes. Papers found during novelty searches are kept out of the corpus.
- **Analysis** (`src/research_analysis/paper_analyzer.py`): the model returns statements with verbatim evidence. Each is labelled:
  - `direct`: the text itself, found in the source;
  - `interpretation`: a paraphrase whose evidence was found in the source;
  - `unverified`: the evidence was not found. Kept, shown and flagged, never used as evidence.

  Verification requires the matched passage to agree on numbers, negations and content words, so an altered quote fails. The paper's own limitation and future-work sentences are added by rule, and research dimensions (platform, method, noise, phenomenon, metric, qubit counts) are tagged from a curated ontology.
- **Cross-paper analysis**: themes (average-linkage clustering of embeddings), method × platform and method × noise coverage, untried pairs, and limitations raised by several papers.

## 6. Research gap identification methodology

`src/gap_discovery/gap_generator.py`. Candidates come from:
- verified **limitations, unresolved problems and future work**, merged when several papers state the same thing;
- **untried combinations**: a method and a platform (or noise type) each studied by two or more papers, but never together. These are labelled as inferences about this corpus, not evidence of absence.

Each gap records its research dimensions and whether each came from the gap's own text, the source paper or corpus coverage. A model may rewrite a gap as a research question; the rewrite is labelled with the model's name and the evidence is left untouched. Gaps are **candidates**; only the researcher marks one validated. Ranking (`gap_ranker.py`) scores eight separate dimensions, each with a one-line reason: novelty (from the classification), significance, evidence strength, feasibility, testability, tool availability, potential contribution and literature coverage. They are combined with weights the researcher sees and can change.

## 7. Novelty assessment methodology

Novelty is not 1 − similarity (`src/gap_discovery/novelty_checker.py`):
1. **Decompose** the gap into its own research dimensions; context borrowed from the source paper only tops up to two.
2. **Query**: the gap's own words, those words plus the main phenomenon, a concept query, an ontology synonym, and method- and condition-focused queries. Long queries skip arXiv, whose API requires every word to match.
3. **Retrieve** from OpenAlex, arXiv and Crossref, the works citing the supporting papers, and the local corpus. Every query and its status is stored.
4. **Screen**: de-duplicate, remove the gap's own supporting papers, and record dates relative to the evidence and whether an abstract exists.
5. **Compare** the closest papers, chosen by shared dimensions before similarity: each dimension is the same, different or not stated, with the sentence quoted. An optional model judge is kept only when its quote is found in the abstract.
6. **Classify by fixed rules**: already investigated, partially investigated, potentially unexplored *in the searched literature*, or insufficient search coverage. Low coverage is never reported as novel. Confidence is stored with its formula and inputs.

## 8. Experiment planning methodology

`src/experiment_planning/`. For a chosen gap, the planner drafts a proposal with the 20 sections the project brief lists (gap, problem, question, objectives, hypothesis, motivation, literature justification, methodology, system, techniques, variables, controls, parameters, expected observations, metrics, baselines, challenges, limitations, contribution, and why it addresses the gap), plus a feasibility class: **simulatable**, **requires quantum hardware** or **requires additional physical validation**.

- The model may cite only the gap's evidence and the closest papers from its novelty check. Every quote is verified; a literature parameter whose quote is not found becomes an explicit **assumption**.
- Feasibility is set by a rule; the model may raise it, never lower it.
- Without a model, the template fills only what the gap determines and lists everything else as missing.
- Every edit is a new revision recording the changed fields and a note. Approval requires the core sections and applies to one exact revision. Rejecting or requesting changes requires a note. Editing an approved plan makes a new draft.

## 9. Quantum verification methodology

`src/verification/plan_verifier.py`, `src/quantum/noise.py`, `src/verification/engine.py`. Only an approved revision is verified.
- The plan becomes a circuit specification; a plan without a circuit is held with the missing items listed, never filled in.
- Checks: structure, qubit indices, gate validity, circuit simulation (Qiskit statevector), normalisation, predicted vs simulated probabilities and total variation distance, **state fidelity** against an expected state (amplitudes, including phase), **noise-parameter validity** (T2 ≤ 2·T1, probabilities in [0, 1]), **predictions under the stated noise** (exact density-matrix simulation with qiskit-aer, analytic readout error; a stated range runs at both ends), the assumptions the result depends on, and a statement that simulation is not hardware validation.
- The verdict (pass, fail or hold) comes from fixed rules; no model output can override a failed computation. Each run is stored against its revision.

## 10. Architecture

| Layer | Code | Role |
|---|---|---|
| UI | `overlap_app.py`, `src/ui/overlap/index.html`, `src/ui/overlap_view.py` | Streamlit app with one custom component: Literature, Papers, Gaps, Plans, Guided run, Reports, Check proposal; the original dashboard is `app.py` |
| Literature | `src/literature/` | Collectors, metadata normalisation, de-duplication, ingestion |
| Analysis | `src/research_analysis/` | Per-paper analysis with verified provenance; cross-paper analysis |
| Knowledge base | `src/knowledge_base/` | ChromaDB + sentence-transformers, BM25, reciprocal-rank fusion, ontology |
| Gaps and novelty | `src/gap_discovery/` | Generation, novelty investigation, ranking |
| Planning | `src/experiment_planning/` | Proposal drafting, revisions, approvals |
| Verification | `src/verification/`, `src/circuit/`, `src/quantum/` | Rules, Qiskit simulation, noise |
| Orchestration | `src/agents/research_graph.py` | LangGraph run with SQLite checkpoints and human pauses |
| Reports | `src/reporting/research_report.py` | 17-section report: Markdown, JSON, PDF |
| Storage | `src/storage/` | SQLite via SQLAlchemy: publications, searches, statements, gaps, assessments, plans, approvals, runs, reports, LLM call log |
| LLM | `src/llm/` | Gemini (primary), Vertex AI, OpenAI, Anthropic; fallback models; prompt-injection delimiters; calls logged without prompt text |
| Evaluation | `src/evaluation/`, `evaluate.py` | Benchmarks and measured results |

## 11. Technology stack

Python 3.12 · Streamlit 1.66 · Pydantic 2 · SQLAlchemy 2 (SQLite) · Qiskit 2.6 and qiskit-aer 0.18 · LangGraph 1.2 with `langgraph-checkpoint-sqlite` · ChromaDB 1.5 and sentence-transformers (`all-MiniLM-L6-v2`) · rank-bm25 · rapidfuzz · scikit-learn · httpx and diskcache · PyMuPDF and python-docx · reportlab · google-genai (Gemini). Versions are those measured in [docs/EVALUATION.md](docs/EVALUATION.md); minimums are in `requirements.txt`.

## 12. Installation

Windows (PowerShell), Python 3.12:

```powershell
python -m venv .venv
```
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```
```powershell
Copy-Item .env.example .env
```

The first run downloads the embedding model (about 90 MB) from Hugging Face.

## 13. API configuration

Set these in `.env` (never commit it; `.gitignore` excludes it):

| Variable | Purpose |
|---|---|
| `LLM_PROVIDER` | `gemini` (default), `vertex`, `openai`, `anthropic`, or `none` for rule-based mode |
| `GEMINI_API_KEY`, `GEMINI_MODEL` | Google AI Studio key and model |
| `GEMINI_FALLBACK_MODELS`, `GEMINI_COOLDOWN_SECONDS` | Models tried in order when the main one is overloaded or rate-limited, and how long an overloaded one is skipped |
| `GOOGLE_APPLICATION_CREDENTIALS`, `VERTEX_PROJECT_ID`, `VERTEX_LOCATION`, `VERTEX_MODEL` | Vertex AI (needs billing enabled on the project) |
| `OPENAI_API_KEY` / `OPENAI_MODEL`, `ANTHROPIC_API_KEY` / `ANTHROPIC_MODEL` | Other providers |
| `OPENALEX_API_KEY` | Free OpenAlex key (keyless requests share a small daily budget) |
| `CROSSREF_MAILTO` | Address for Crossref's polite pool |
| `SEMANTIC_SCHOLAR_API_KEY` | Optional; keyless requests are often rate-limited |
| `RESEARCH_DB_URL` | Research database, default `sqlite:///data/research.db` |
| `RESEARCH_CHECKPOINTS` | Guided-run checkpoint file, default `data/checkpoints.db` |
| `EMBEDDING_MODEL_NAME`, `CHROMA_PERSIST_DIRECTORY`, `KB_REQUIRE_SEMANTIC` | Knowledge base; `KB_REQUIRE_SEMANTIC=1` refuses to start without semantic retrieval |
| `LLM_CALL_LOGGING`, `LITERATURE_CACHE_DIR` | Call log on/off; HTTP cache location |

Without a working model, every step falls back to rule-based behaviour and the app says so; nothing generated is presented as model output.

## 14. Usage

```powershell
.\.venv\Scripts\streamlit.exe run overlap_app.py --server.port 8502
```

Open http://localhost:8502.
1. **Literature**: collect papers by research area or query (years, sources, results per source) or by DOI. Open a paper to include or exclude it, **Analyse paper**, or fetch arXiv full text. **Corpus analysis** shows themes, coverage and recurring statements, and offers **Analyse next 5**.
2. **Gaps**: **Find gaps in the corpus**. Open a gap to see its evidence and dimensions, **Check novelty** (queries, coverage, closest papers per dimension, uncertainty), adjust ranking weights, mark it candidate, validated or rejected, then **Plan an experiment**.
3. **Plans**: read the proposal and its justifications, **Edit plan** (text, parameters, circuit as JSON, feasibility), then **Approve**, **Request changes** or **Reject**. On an approved revision, **Verify** shows the probability disks, circuit, checks, noise results and the Markdown report.
4. **Guided run**: start a run (optionally collecting and analysing papers first). It pauses for your gap choice and your plan decision, survives restarts, and shows an execution log.
5. **Reports**: an overview of where the investigation stands, then per plan **Build report** and download PDF, Markdown or JSON (also saved in `data/reports/`).
6. **Check proposal** verifies a single uploaded experiment proposal; **Dashboard** opens the original dashboard (`app.py`).

## 15. Evaluation methodology

```powershell
.\.venv\Scripts\python.exe evaluate.py --novelty --n 15
```
```powershell
.\.venv\Scripts\python.exe evaluate.py --all
```

`--novelty` runs the known-work benchmark live on a copy of the database. Each analysed paper's own question becomes a gap; the paper is removed from the local corpus and the full novelty check runs. Labelling that question novel is a **false novelty**. A specificity control of implausible combinations checks that the checker can also say "not investigated".

`--all` runs everything else on a copy of the database and writes [docs/EVALUATION.md](docs/EVALUATION.md):
- **retrieval:** Recall@k and MRR of the known source paper, from the stored searches;
- **extraction:** statistics, plus a stress test that feeds the quote verifier genuine quotes and fabricated variants (changed number, inserted negation, swapped term, quote from another paper);
- **gaps:** grounding, specificity and near-duplicates;
- **verification:** false-pass and false-fail rates on the 6 hand-written proposal cases and 12 plan cases with textbook answers.

Metrics that need human judgement (field accuracy, gap quality, the plan rubric, the future-work and partially-investigated sets) come with labelling sheets in `data/benchmark/labels/` and are reported as "not measured" until labelled. `python evaluate.py` runs the original 6-case proposal benchmark; `--gaps` runs the planted-gap benchmark on the 4 synthetic sample papers.

Tests, as GitHub Actions runs them on every push and pull request (`.github/workflows/tests.yml`, Python 3.11 and 3.12):

```powershell
.\.venv\Scripts\python.exe -m pytest -q -m "not live"
```

Without API keys, 176 pass and 2 skip. Tests marked `live` need real Vertex AI or Gemini credentials and network access; run them with `-m live`.

## 16. Results

Measured 2026-11-27 on a working copy of the research database (22 analysed papers on entanglement and decoherence, 1 from full text); details and procedures are in [docs/EVALUATION.md](docs/EVALUATION.md).

| Area | Result |
|---|---|
| Novelty, known-work set (15 published questions) | False novelty 0/15 (95% CI 0–0.20). The 15 questions have only 4 distinct shapes, so this alone cannot show discrimination; the specificity control called 5 of 5 implausible combinations something other than "already investigated" |
| Retrieval | Recall@10 0.93 and MRR 0.79, almost entirely from OpenAlex and the gap's own-words query; arXiv 0.00, Crossref 0.13 |
| Extraction | 0.957 of model statements had their evidence found in the source. Quote verifier: detection 0.996, false rejection 0.000 on 124 real quotes (before the Phase 9 fix detection was 0.52) |
| Gaps | 0.765 backed by verified quotes; 0.294 name two or more dimensions of their own |
| Verification | 6/6 proposal cases and 12/12 textbook plan cases correct; no false passes or false fails |
| Synthetic sample papers (`--gaps`) | Planted-gap recall 1.00 (5 of 5); lower-bound precision 0.50 |

Field accuracy, gap quality and plan quality are **not measured** yet: they need the labelling sheets filled in.

## 17. Limitations

- The novelty verdict holds **in the searched literature only**. Comparisons mostly use abstracts, and retrieval depends heavily on OpenAlex.
- The evaluated corpus is small and on one topic; the rates above have wide uncertainty.
- The ontology has no word sense ("leakage" also matches privacy leakage), and "conflicting findings" is not yet detected.
- Gaps often borrow dimensions from their source paper; only 29% are fully self-specified.
- Verification is simulation: ideal or modelled noise on at most 10 qubits in a plan circuit, with noise parameters recognised by name. It is not hardware validation.
- Guided-run steps run inside the page request, so long collections block the screen until they finish.
- Reports are per plan, and their corpus sections reflect the corpus at build time.

## 18. Future scope

- Full-text ingestion at scale (open-access PDFs beyond arXiv) and a larger benchmark corpus (three topics × about 40 papers).
- Human-labelled evaluation: extraction accuracy, gap quality, the plan rubric, and the future-work and partially-investigated sets, with inter-rater agreement.
- Precision@k with relevance judgements for a topic set, and scientific embeddings (for example SPECTER2) compared with MiniLM.
- Detecting conflicting findings, comparing gaps side by side, and word-sense-aware ontology matching.
- Background execution for guided runs, and per-run reports.
- Optional execution on real quantum hardware as a separate, clearly labelled validation step.

---

Further documentation: [docs/ROADMAP.md](docs/ROADMAP.md) (audit, design and per-phase results), [docs/EVALUATION.md](docs/EVALUATION.md) (measured results), [docs/VERIFICATION_ENGINE.md](docs/VERIFICATION_ENGINE.md) (reference for the original proposal-verification engine).

## Contributors

- Dhyeykumar Patel
- Prachi Jagdale
- Balaji Mali
