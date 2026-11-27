# Overlap: LLM-Based Research Gap Identification and Experiment Planning for Scientific Literature

Overlap reads quantum-computing papers, finds **research gaps** across them, designs **experiments** that would fill those gaps, and **verifies** each experiment plan by simulating its circuit with Qiskit and auditing it against scientific rules.

Every gap is backed by quotes from the papers, and every plan carries an executable circuit with predicted outcomes, so the pipeline's output can be checked rather than taken on trust.

## How it works

```
 1. PAPERS                2. GAPS                       3. PLANS                     4. VERIFY
 upload PDF/DOCX/TXT  →   compare all papers for:   →   for a chosen gap:        →   simulate the circuit (Qiskit)
 extract limitations,     • stated limitations          • hypothesis + rationale     audit 6 rules (qubits, gates,
 future work, open        • future work / open qs       • method, variables          normalization, distribution,
 questions, claims,       • untried circuit × platform  • baseline, metrics, risks   literature, hardware)
 experiments; index       • qubit-scale limits          • executable circuit +       revise the plan once if it
 into the knowledge base  • conflicting results          theoretical prediction      fails, then re-verify
                          rank by novelty, evidence,
                          feasibility
```

| Stage | Code | What it does |
|---|---|---|
| 1. Literature | `src/research/library.py` (`PaperLibrary`) | Extracts text (`DocumentExtractor`), analyses each paper into a `ResearchPaperAnalysis` (limitations, future work, open questions, claims, experiments), and indexes chunks into the knowledge base. |
| 2. Gaps | `src/research/gap_finder.py` (`GapFinder`) | Five detectors build candidate gaps with verbatim evidence, then merge, score and rank them. **Novelty** = 1 − similarity to the rest of the literature in the knowledge base. **Evidence** = number of papers and quotes. **Feasibility** = whether the gap can be tested with a ≤10-qubit circuit. |
| 3. Plans | `src/research/planner.py` (`ExperimentPlanner`) | Writes an `ExperimentPlan` with an executable `ProposalSpecification`. Template predictions come from textbook theory, not from the simulator, so verification is a genuine check. |
| 4. Verify | `src/verification/engine.py` (`VerificationEngine`), `src/research/pipeline.py` | Simulates and audits each plan. If a plan fails, it is revised once using the failed checks as feedback, then re-verified. |
| Report | `ResearchPipeline.agenda_markdown()` | A research agenda listing the ranked gaps with their evidence and every plan with its verification status. |

**LLM and rules.** With a working LLM (Vertex AI / Gemini / OpenAI / Anthropic), the model analyses papers, rewrites gap statements and writes plans. Its output is schema-validated, and it can never add a gap without deterministic evidence behind it. Without one, or when the provider fails (for example, billing disabled), every stage falls back to a rule-based path that works offline, and the app shows which mode is active.

## Quick start

```powershell
.\.venv\Scripts\streamlit.exe run overlap_app.py --server.port 8502
```

Open http://localhost:8502 → **Load 4 sample papers** → **Find research gaps** → open a gap → **Plan an experiment** → **Download research agenda**.

- The sample papers in `data/sample_papers/` are **synthetic**, written to demonstrate the method. Each has deliberately planted gaps (answer key in `planted_gaps.json`).
- **Check a proposal** verifies a single experiment proposal directly (stage 4 on its own).
- **Full dashboard** opens the original research dashboard with all 9 tabs (knowledge base, benchmarks, agentic workflow and more).

## Evaluation

```powershell
.\.venv\Scripts\python.exe evaluate.py --gaps
```

Current results on the synthetic papers (offline, rule-based mode):

| Metric | Value |
|---|---|
| Planted-gap recall | 1.00 (5 of 5) |
| Precision, lower bound (found gaps matching a planted gap) | 0.50 (the extra gaps are real limitations stated in the papers) |
| Gaps with quoted evidence | 1.00 |
| Plans verified first time | 1.00 |

`python evaluate.py` (no flag) still runs the original proposal-verification benchmark.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q --deselect tests/test_gemini_integration.py --deselect tests/test_vertex_ai_integration.py
```

New tests cover the research pipeline: `test_gap_finder.py`, `test_planner.py`, `test_research_pipeline.py`, `test_agent_paper_branch.py` and `test_llm_fallback.py`. The two deselected files call live LLM APIs.

---

# Reference: the verification engine (stage 4)

The sections below document the verification engine and the original research dashboard (`app.py`) that stage 4 is built on.

## 1. System Architecture

```
AI-Based Verification of Quantum Experiment Proposals/
│
├── .streamlit/                   # Local Streamlit configuration (headless, ports, telemetry disabled)
│   ├── config.toml
│   └── credentials.toml
├── src/
│   ├── agents/                   # LangGraph Multi-Agent Architecture
│   │   ├── state.py              # Typed shared state (VerificationAgentState) & execution log reducers
│   │   ├── tools.py              # Structured, traceable agent tool wrappers (RAG, Qiskit, Rules, Reporting)
│   │   ├── document_agent.py     # Agent 1: Document Analyst (PDF/DOCX extraction & missing parameter detection)
│   │   ├── literature_agent.py   # Agent 2: Literature Researcher (ChromaDB query refinement & claim grounding)
│   │   ├── simulation_agent.py   # Agent 3: Quantum Simulation Agent (Deterministic Qiskit statevector simulation)
│   │   ├── verification_agent.py # Agent 4: Verification Auditor (Deterministic rules & computational FAIL precedence)
│   │   ├── reporting_agent.py    # Agent 5: Explainability & Reporting Agent (Scientific synthesis & report artifacts)
│   │   └── orchestrator.py       # LangGraph StateGraph, conditional edges, retry bounds & human-in-the-loop controls
│   ├── schemas/                  # Pydantic v2 data models
│   │   ├── proposal.py           # ProposalSpecification, CircuitSpec, GateOperation, ExpectedOutcome
│   │   └── verification.py       # VerificationCheck, CheckStatus, VerificationReport, LiteratureCitation
│   ├── document_processing/      # Non-hallucinatory document ingestion
│   │   ├── document_extractor.py # Unified polymorphic document extractor (PDF, DOCX, TXT)
│   │   ├── pdf_extractor.py      # PDF parsing, section segmentation, Dirac bra-ket math extraction
│   │   ├── docx_extractor.py     # Pure-Python OpenXML DOCX parser with explicit page-break detection
│   │   └── chunker.py            # Traceable overlapping chunker with page & chunk ID provenance
│   ├── knowledge_base/           # Literature vector store & RAG
│   │   ├── starter_corpus.py     # Verified reference literature (Nielsen & Chuang, Bell 1964, Grover 1996, Preskill 2018)
│   │   ├── vector_store.py       # ChromaDB (cosine) + Sentence-Transformers; word-match fallback reported as degraded
│   │   └── retriever.py          # Semantic & lexical query retriever producing traceable citations
│   ├── circuit/                  # Computational quantum circuit simulation
│   │   ├── parser.py             # Circuit DSL parser, gate arity, and qubit bounds validation
│   │   ├── simulator.py          # Qiskit Statevector engine (with exact NumPy tensor fallback)
│   │   └── metrics.py            # TVD (Total Variation Distance), phase-invariant fidelity, outcome comparison
│   ├── llm/                      # Multi-provider LLM integration & prompt defense
│   │   ├── client.py             # Configurable client (OpenAI, Anthropic, Gemini, OpenAI-compatible, & Demo Mode)
│   │   ├── prompts.py            # Injection-defended system prompts (<UNTRUSTED_PROPOSAL_TEXT>)
│   │   ├── extractor.py          # Structured proposal parameter extraction into Pydantic models
│   │   └── analyzer.py           # Claim cross-examiner enforcing computational precedence
│   ├── verification/             # Independent verification engine
│   │   ├── rules.py              # Modular check rules (Qubit count, syntax, normalization, distributions, feasibility)
│   │   └── engine.py             # End-to-end verification pipeline (PASS, FAIL, INCONCLUSIVE, NOT RUN)
│   ├── reporting/                # Explainable audit report generation
│   │   └── generator.py          # Export to JSON schema and publication-ready Markdown
│   ├── ui/                       # Streamlit UI visual components
│   │   └── components.py         # Status badges, probability bar charts, citation cards, statevector tables
│   └── evaluation/               # Scientific evaluation & benchmarking
│       ├── benchmark.py          # Labelled benchmark test cases loader
│       └── evaluator.py          # Multi-mode precision, recall, FPR, FAR, and simulation agreement evaluator
├── data/
│   ├── benchmark_cases/          # Labelled benchmark test cases (JSON)
│   └── sample_proposals/         # Sample proposals (Bell State PDF, Bell State DOCX, Incomplete DOCX)
├── tests/                        # Automated Pytest test suite (35 unit & integration tests)
├── app.py                        # Streamlit interactive research dashboard
├── evaluate.py                   # Standalone evaluation & benchmark CLI runner
├── requirements.txt              # Production dependency specifications
├── .env.example                  # Environment credentials template
└── .gitignore                    # Git exclusion rules
```

---

## 2. LangGraph Multi-Agent Architecture

The system features an autonomous, multi-agent verification architecture orchestrated via **LangGraph StateGraph**. It coordinates five specialized agents, maintaining a shared typed state (`VerificationAgentState`), dynamic conditional routing, bounded retry loops, and human-in-the-loop review.

```
                    [START: Document Ingestion / Input]
                                   │
                                   ▼
                       [1. Document Analyst Agent]
                                   │
                         (route_after_document)
                        ╱                      ╲
             [Critical Params Missing]     [Spec Valid & Confirmed]
                      ╱                          ╲
                     ▼                            ▼
         [Human Review / Pause Node]    [2. Literature Researcher Agent]
                     │                            │
            (Researcher Confirms)       (Refine Query / Retry <= 2)
                     │                            ▼
                     └─────────────────► [3. Quantum Simulation Agent]
                                                  │
                                                  ▼
                                      [4. Verification Auditor Agent]
                                                  │
                                         (route_after_auditor)
                                        ╱                     ╲
                       [Discrepancy Detected]             [Audited PASS/FAIL]
                                ╱                               ╲
                               ▼                                 ▼
                   [Re-Extraction Review Node]        [5. Reporting Agent]
                               │                                 │
                               └────────────────────────►      [END]
```

### The Five Specialized Agents

| Agent | Module | Primary Responsibility | Reused Tool / Engine |
|---|---|---|---|
| **Agent 1: Document Analyst** | `src/agents/document_agent.py` | Extracts parameters from PDF/DOCX or manual input; flags missing qubit count, gates, or outcomes; strictly isolates non-qubit numbers (DOIs, page numbers, years). | `extract_document_tool` (`ProposalExtractor` via Vertex AI / Gemini) |
| **Agent 2: Literature Researcher** | `src/agents/literature_agent.py` | Grounding retrieval in ChromaDB; evaluates support (`Supported`, `Contradicted`, `Unaddressed`); refines queries when evidence is insufficient (bounded retry <= 2). | `retrieve_literature_tool` (`QuantumRetriever` over ChromaDB) |
| **Agent 3: Quantum Simulation Agent** | `src/agents/simulation_agent.py` | Executes confirmed circuits on Qiskit Statevector simulator; calculates probabilities, TVD, and fidelity. **LLM never calculates or tampers with simulation data.** | `simulate_circuit_tool` (`QuantumSimulator`, `compare_outcomes`, `TVD`) |
| **Agent 4: Verification Auditor** | `src/agents/verification_agent.py` | Executes deterministic verification rules; checks normalization and bounds. **Strict Rule: An LLM can NEVER overturn a computational FAIL.** | `run_verification_audit_tool` (`rules.py`) |
| **Agent 5: Explainability & Reporting** | `src/agents/reporting_agent.py` | Synthesizes grounded scientific explanations; distinguishes author claims, citations, and simulation results; generates JSON and Markdown report artifacts. | `generate_report_tool` (`ReportGenerator`) |

### Core Conditional Workflows Supported
1. **Workflow A (Missing Parameter)**: Document Analyst flags missing qubits/gates $\rightarrow$ pauses at Human Review Node $\rightarrow$ researcher confirms $\rightarrow$ resumes.
2. **Workflow B (Literature Investigation)**: Literature Researcher retrieves passages $\rightarrow$ detects weak evidence $\rightarrow$ refines query with gate terms $\rightarrow$ retries (bounded at $\le 2$ iterations).
3. **Workflow C (Computational Discrepancy)**: Simulation Agent runs circuit $\rightarrow$ Auditor flags probability mismatch $\rightarrow$ triggers single re-extraction review if ambiguity exists $\rightarrow$ confirms and retains `FAIL`.
4. **Workflow D (Successful Verification)**: End-to-end multi-agent pass yielding full explainability artifacts and verification metrics.

---

## 3. Mathematical & Quantum Conventions

### Qubit Ordering & Bitstring Representation
- **Register Indexing**: Qubits are zero-indexed (`q_0, q_1, ..., q_{n-1}`).
- **Basis Bitstrings**: Standard computational basis little-endian convention matching Qiskit: $|q_{n-1} q_{n-2} \dots q_1 q_0\rangle$.
  - For 2 qubits, state $|01\rangle$ indicates qubit $0$ is in state $|1\rangle$ and qubit $1$ is in state $|0\rangle$.
  - In symmetric Bell state $|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$, both $|00\rangle$ and $|11\rangle$ have probability $0.50$ regardless of endianness.

### Supported Gate Vocabulary
- **Single-Qubit Gates**: `H`, `X`, `Y`, `Z`, `S`, `T`, `SDG`, `TDG`, `I`
- **Continuous Rotation Gates**: `RX(theta)`, `RY(theta)`, `RZ(theta)` (angles in radians)
- **Two-Qubit Entangling Gates**: `CX` (CNOT), `CZ`, `SWAP`

### Verification Metrics
- **Statevector Normalization**: $\sum_{i=0}^{2^n-1} |c_i|^2 = 1.0$ (numerical tolerance $\epsilon \le 10^{-6}$).
- **Total Variation Distance (TVD)**:
  $$\text{TVD}(P_{\text{exp}}, P_{\text{comp}}) = \frac{1}{2} \sum_{x} |P_{\text{exp}}(x) - P_{\text{comp}}(x)| \in [0.0, 1.0]$$
- **Phase-Invariant State Fidelity**:
  $$F(|\psi\rangle, |\phi\rangle) = |\langle\psi|\phi\rangle|^2 \in [0.0, 1.0]$$

---

## 3. Mandatory Demonstration Cases

The prototype includes built-in reproducible demonstration cases accessible directly from the Streamlit UI dropdown and benchmark suite:

| Case ID | Proposal Description | Circuit / Gates | Expected Behavior | Verification Status |
|---|---|---|---|---|
| **Case 1** | Two-qubit Bell State Preparation ($|\Phi^+\rangle$) | `H 0`<br>`CX 0 1` | Ideal state $\frac{1}{\sqrt{2}}(\|00\rangle + \|11\rangle)$, $P(00)=0.5, P(11)=0.5$ | **PASS** |
| **Case 2** | Bell State with Deliberately Incorrect Probabilities | `H 0`<br>`CX 0 1` | Hypothesizes $P(00)=0.8, P(11)=0.2$. Discrepancy detected ($0.30 > 0.05$). | **FAIL** |
| **Case 3** | Invalid Qubit Index (Out-of-Bounds Operation) | `H 0`<br>`CX 0 2` (on 2 qubits) | Qubit index 2 exceeds register size (0..1). Rejected with structural error. | **FAIL** |
| **Case 4** | Incomplete Proposal (Missing Parameters) | None | Missing qubit count, circuit gates, and expected distributions. | **INCONCLUSIVE** |
| **Case 5** | Three-Qubit GHZ State Preparation | `H 0`<br>`CX 0 1`<br>`CX 1 2` | Tripartite entanglement $\frac{1}{\sqrt{2}}(\|000\rangle + \|111\rangle)$, $P(000)=0.5, P(111)=0.5$ | **PASS** |
| **Case 6** | Unavailable LLM API Key | Any | Safe fallback to explicitly labelled **DEMO MODE** using fixed benchmarks. | **PASS (Demo Mode)** |

---

## 4. Setup and Installation (Windows)

### Prerequisites
- Python 3.11 or Python 3.12 (64-bit)
- PowerShell (Windows Terminal)

### Step 1: Create Virtual Environment
```powershell
python -m venv --system-site-packages .venv
```

### Step 2: Install Dependencies
```powershell
.\.venv\Scripts\pip.exe install -r requirements.txt
```

### Step 3: Configure Environment Variables
Configure your credentials in the local `.env` file (excluded from git):
```env
# LLM: Gemini API (Google AI Studio key) is the primary provider
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.8-flash
# Tried in order when GEMINI_MODEL is overloaded (503) or rate-limited (429)
GEMINI_FALLBACK_MODELS=gemini-flash-latest,gemini-flash-lite-latest,gemini-3.1-flash-lite

# Optional: Google Cloud Vertex AI instead (LLM_PROVIDER=vertex; needs billing enabled on the project)
GOOGLE_APPLICATION_CREDENTIALS=path\to\service-account.json
VERTEX_PROJECT_ID=your-project-id
VERTEX_LOCATION=global
VERTEX_MODEL=gemini-3.8-flash

# Knowledge base (ChromaDB + Sentence-Transformers)
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
CHROMA_PERSIST_DIRECTORY=./chroma_db
KB_REQUIRE_SEMANTIC=0          # 1 = refuse to start if semantic retrieval is unavailable

# Research database (publications, searches, gaps, assessments, approvals, LLM call log)
RESEARCH_DB_URL=sqlite:///data/research.db
```

Supported LLM providers:
- **Google Gemini API** (primary): official `google-genai` SDK with an AI Studio key, with model fallback and short-term cool-down of overloaded models.
- **Google Cloud Vertex AI**: the same SDK with a service account (requires billing on the project).
- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `LLM_API_KEY` + `LLM_BASE_URL` (OpenAI-compatible).

**Provenance of LLM output.** Every call records which provider and model actually answered and whether the output was live or demo (`ConfigurableLLMClient.last_call`). The call is also logged in the research database's `llm_calls` table, with model, live flag, success and latency, plus a prompt hash; no prompt text or keys are stored. Paper and proposal analysis never use canned demo answers. When no live model is available they fall back to rule-based extraction from the document itself and say so in `extraction_warnings`.

**Knowledge base status.** Retrieval runs on ChromaDB (cosine distance) with Sentence-Transformers embeddings, using one collection per embedding model. If either library is missing, the knowledge base degrades to word matching. It then says so in the UI and logs, and with `KB_REQUIRE_SEMANTIC=1` it refuses to start. *Correction:* before Phase 1 (2026-11-27), ChromaDB and Sentence-Transformers were listed but not installed, so all earlier retrieval used word matching, not semantic search.

---

## 5. Running the Application

### Launch Streamlit Research Dashboard
Run the following command in PowerShell:
```powershell
$env:PYTHONUTF8="1"; $env:PYTHONIOENCODING="utf-8"; $env:USERPROFILE="$PWD"; .\.venv\Scripts\python.exe -m streamlit run app.py
```
Open your browser at: `http://localhost:8501`

### Dual Proposal Input Modes & Confirmation Workflow:
1. **📝 Proposal Input Modes**:
   - **Mode 1: Upload Existing Research Proposal**: Upload academic proposals in **PDF** or Word (**DOCX**) format. Extracts sections, methodology, hypothesis, quantum system, initial state, gate sequences, measurement basis, expected outcomes, and assumptions. Uses the configured LLM (or robust rule-based parser) to populate a Pydantic `ProposalSpecification` with source page references, without inventing missing parameters.
   - **Mode 2: Manual Structured Input**: Enter title, objective, hypothesis, methodology, quantum system, qubit count, initial state, circuit DSL, measurement basis, expected outcomes, and assumptions directly. Includes preloaded 1-click demonstration presets (Bell State, Biased Fail, Out-of-Bounds Fail, GHZ State).
2. **🛡️ Editable Researcher Confirmation Screen**:
   - Both input modes normalize into the identical structured Pydantic schema.
   - Displays source document metadata, full original text in an expander, and any missing information / ambiguity warnings.
   - Allows researchers to inspect and edit every extracted field before computational execution.
   - **Mandatory Confirmation**: Explicit researcher confirmation (`Confirm & Verify Proposal`) is required before sending the specification to Qiskit simulation and the verification engine.
3. **⚛️ Quantum Simulation**: Inspect Qiskit circuit diagrams, statevector components, and measurement probabilities.
4. **📊 Verification Engine**: View audited verification checks with exact PASS, FAIL, or INCONCLUSIVE statuses.
5. **📚 Evidence & Citations**: Inspect retrieved literature passages with provenance (source file, page, chunk ID).
6. **📑 Explainable Report**: Download verification reports as JSON or formatted Markdown.
7. **🗂️ Knowledge Base**: Search literature passages or upload custom research papers.
8. **🧪 Benchmark Evaluation**: Run the automated evaluation suite directly from the UI.

---

## 5. Dual Workflow: Research Paper Analysis vs Proposal Verification

The application distinguishes between two document categories:

### A. Academic Research Paper Analysis (`ResearchPaperAnalysis`)
- **Purpose**: Ingests broad academic literature (e.g. Nature, arXiv, PRX publications).
- **Extracted Fields**:
  - Publication metadata: Title, DOI, Authors, Publication/Journal details.
  - Core Academic Breakdown: Abstract, Research Problem, Research Objective, Methodology.
  - Architecture: Algorithms and Frameworks (Qiskit, PennyLane, VQE, QAOA), Experimental Platform/Setup, Quantum Systems Discussed.
  - Multi-Experiment Support: Preserves discrete `QuantumExperimentRecord` items when papers evaluate multiple benchmarks or circuit configurations.
  - Findings & Limitations: Key scientific results and stated hardware constraints (e.g. $T_2$ decoherence, two-qubit gate error limits).
- **Interactive Verification**: Researchers can inspect the full breakdown in Tab 2 and click `"Load [EXP-ID] for Verification"` to send any individual experiment directly to the Qiskit simulation pipeline.

### B. Quantum Experiment Proposal Verification (`ProposalSpecification`)
- **Purpose**: Target an individual executable quantum experiment.
- **Extracted Parameters**: Initial state, allocated qubit register ($1..10$), gate sequence DSL, measurement basis, expected computational basis probability distribution, and physical assumptions.
- **Verification Workflow**:
  - **Editable Confirmation Screen**: Researchers review and adjust parameters.
  - **Qiskit Statevector Simulation**: Computes exact quantum amplitudes and probability distributions.
  - **Independent Verification Engine**: Evaluates structural bounds, state normalization unity ($\sum |c_i|^2 = 1.0$), probability matching (TVD & fidelity), and literature grounding.
  - **INCONCLUSIVE Status Handling**: If the qubit count or gate sequence is omitted, simulation is safely bypassed without crashing, and the report is transparently marked `INCONCLUSIVE`.

---

## 6. Investigation & Fix: Qubit Count Extraction Bug

### Problem Statement
When uploading academic paper `s42256-025-01153-0.pdf`, the system failed with:
`1 validation error for CircuitSpec: num_qubits Input should be less than or equal to 10 [input_value=42257]`

### Root Cause Analysis
In the heuristic regex fallback in `src/llm/extractor.py`, the gate pattern was defined as:
```python
gate_pattern = re.compile(
    r"\b(H|X|Y|Z|S|T|RX|RY|RZ|CX|CNOT|CZ|SWAP)\s*(\([^\)]+\))?\s*(?:on\s*)?(?:qubit[s]?\s*)?([q\d\s,\[\]]+)",
    re.IGNORECASE
)
```
When encountering the Nature article identifier `s42256-025-01153-0`:
1. `\b(S)` matched the leading letter `s` because `\b` preceded the group but was missing at the end of single-letter gate names.
2. `([q\d\s,\[\]]+)` matched the trailing numerical identifier `42256`.
3. The extractor erroneously generated an `S` gate on target qubit `42256`.
4. Downstream logic derived `num_qubits = max_q + 1 = 42257`, triggering Pydantic validation failure (`num_qubits <= 10`).

### Resolution & Corrective Architecture
1. **Strict Semantic Extraction (`extract_strict_num_qubits`)**:
   - Strips DOIs (`10.xxxx/...`, `s42256-...`), 4-digit years (`2024, 2025`), citations (`[1], [42]`), page numbers, and equation numbers before qubit extraction.
   - Requires explicit contextual evidence (e.g. "two-qubit circuit", "3 qubits", "QuantumRegister(2)").
   - Maps written numerals ("two" $\to 2$, "three" $\to 3$).
   - Rejects registers $> 10$ qubits (respecting simulation prototype bounds).
   - Returns `num_qubits = null` when ambiguous or unspecified.
2. **Strict Gate Matching (`extract_strict_gates`)**:
   - Enforces word boundaries on BOTH sides of gate names (`\b([HXYZST])\b`).
   - Requires explicit gate context (e.g. `H on qubit 0`, `CX on qubits 0, 1`).
   - Enforces target qubit bounds strictly to single digits $0..9$.
3. **Pydantic Bounds & Null Safety**:
   - `CircuitSpec` requires `num_qubits <= 10` and all target qubits $< \text{num\_qubits}$.
   - Downstream simulation engine checks for null `num_qubits` and bypasses execution gracefully, setting status to `INCONCLUSIVE`.

---

## 7. Running Automated Tests

Run the complete Pytest suite (42 unit, regression, and integration tests):
```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

Verifying across the pipeline:
- **Document Classification & Qubit Extraction (Regression Suite)**:
  - Paper with DOI `10.1038/s42256-025-01153-0` does NOT extract 42257
  - Explicit two-qubit document extracts 2
  - Missing qubit information safely returns `num_qubits = null`
  - Multi-experiment papers preserve discrete `QuantumExperimentRecord` items
  - Publication years (2024, 2025) and citations are never confused with qubits
  - Incomplete proposals do not crash and mark status as `INCONCLUSIVE`
  - Valid Bell-state proposal cleanly passes full simulation and verification
- **Google Gemini 3.8 Flash Integration**:
  - Live Gemini model catalog availability (`models/gemini-3.8-flash`)
  - Official `google-genai` SDK connectivity probe and latency diagnostics
  - Missing API key detection without hallucinated AI responses
  - Invalid API key security and error isolation (never exposing credentials)
  - Structured quantum proposal extraction into validated Pydantic models
  - Graceful fallback for malformed model responses
  - RAG context integration and citation provenance preservation
  - Inviolable computational precedence over LLM claims
- **Dual Input Modes & Confirmation**:
  - PDF proposal parsing with section & page-level metadata
  - Pure-Python DOCX parsing with explicit and rendered page-break extraction
  - Manual structured input mode with all experimental parameters
  - Schema compatibility across both input modes
  - Non-hallucination of missing operations/probabilities on incomplete proposals (yielding `INCONCLUSIVE`)
  - Editable researcher review/confirmation flow updating downstream simulation and verification
- **Quantum Circuit Simulation**:
  - Ideal Bell state simulation ($|\Phi^+\rangle$) with $P(00)=0.5, P(11)=0.5$
  - Deliberately incorrect probability rejection
  - Out-of-bounds qubit index handling
  - Malformed gate syntax and angle parameter validation
  - Three-qubit GHZ state tripartite entanglement
  - Normalization unity ($\sum |c_i|^2 = 1.0$)
  - Phase-invariant fidelity & TVD metrics
- **Document Processing & Knowledge Base**:
  - PDF section segmentation and Dirac bra-ket math extraction
  - Document chunking with page-level provenance
  - Anti-hallucination warning for scanned documents
- **Verification Engine Rules**:
  - Verification engine rule precedence (computational failure cannot be overridden by LLM)
  - Physical NISQ hardware feasibility disclaimers
  - Missing parameters producing `INCONCLUSIVE` instead of invented details
- **Reporting & Evaluation**:
  - JSON and Markdown report export completeness
  - Benchmark evaluation harness and CSV export
- **LangGraph Multi-Agent Architecture (`tests/test_agentic_workflow.py`)**:
  - `test_successful_bell_state_agentic_workflow`: End-to-end multi-agent PASS
  - `test_incorrect_bell_state_probabilities_handled_by_auditor`: Flawed distribution caught by Auditor (FAIL)
  - `test_ghz_tripartite_state_agentic_workflow`: Multi-qubit tripartite state synthesis & verification
  - `test_missing_qubit_count_routes_to_human_review`: Incomplete proposal paused at human review node
  - `test_missing_expected_probabilities_handling`: INCONCLUSIVE distribution match without guessing
  - `test_irrelevant_literature_triggers_retrieval_refinement`: Semantic query refinement on weak evidence
  - `test_vertex_ai_failure_fallback`: Graceful fallback to deterministic parser on API connection timeout
  - `test_agent_retry_limits_prevent_infinite_loops`: Bounded retry enforcement (<= 2 iterations)
  - `test_human_approval_and_resume`: Workflow resumes seamlessly upon researcher confirmation
  - `test_computational_fail_cannot_be_overridden_by_llm`: Strict scientific rule enforcement

---

## 9. Running the Multi-Agent & Before/After Benchmark Evaluation

### Run Agentic Tests:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_workflow.py -v
```

### Run Before-and-After Benchmark Comparison:
```powershell
.\.venv\Scripts\python.exe evaluate_agents.py
```

### Empirical Before-and-After Benchmark Results:
```
=============================================================================================================================
Pipeline                                  | Total Cases | Accuracy | Precision | Recall | FPR   | FAR  | Human-Review | Latency
-----------------------------------------------------------------------------------------------------------------------------
Agentic Pipeline (LangGraph Multi-Agent)  | 6           | 100.0%*  | 1.0000    | 1.0000 | 0.0%  | 0.0% | 50.0%        | 15.34 ms
Original Pipeline (Single-Pass Engine)    | 6           | 100.0%   | 1.0000    | 1.0000 | 0.0%  | 0.0% | 0.0%         | 2.79 ms
=============================================================================================================================
*Note: The Agentic pipeline correctly triggers human-in-the-loop review on 50% of cases (the 3 cases with missing or
out-of-bounds parameters), halting execution rather than guessing missing values. Zero false-pass rate (FPR = 0.0%)
is guaranteed across both architectures.
```

---

## 10. Security & Scientific Integrity Principles

1. **Inviolable Computational Precedence**:
   Computational simulation failure (e.g., probability mismatch or out-of-bounds gate) **CANNOT** be overridden by an LLM or literature argument. If simulation fails, the check status is strictly `FAIL`.
2. **Prompt Injection Defense**:
   Uploaded proposal text is strictly isolated in `<UNTRUSTED_PROPOSAL_TEXT>` tags and processed as passive data. System prompts instruct the model never to execute embedded directives.
3. **Traceable Citations (Zero Hallucination)**:
   Every citation contains the exact source filename, page number, chunk identifier, and similarity score. Passages are retrieved exclusively from indexed documents; citations are never fabricated.
4. **Physical NISQ Feasibility Disclaimer**:
   Passing an ideal statevector simulation does **not** certify physical experiment execution. All reports include mandatory disclaimers regarding $T_1/T_2^*$ decoherence, two-qubit gate fidelities, crosstalk, and pulse calibration overhead on real hardware.
