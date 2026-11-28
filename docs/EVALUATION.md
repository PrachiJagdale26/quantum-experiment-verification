# Evaluation results

Measured 2026-11-27 on a copy of the research database (file `phase4_live.db`; 22 analysed papers, 1 from full text). Software: python 3.12.5, qiskit 2.6.1, qiskit-aer 0.18.4, langgraph 1.2.24, sentence-transformers 6.3.0, chromadb 1.5.9, rapidfuzz 3.14.6, embedding_model all-MiniLM-L6-v2.
Every number below was measured by `python evaluate.py --all` (novelty and retrieval come from the stored `python evaluate.py --novelty` run). Nothing is estimated; metrics that need human labels say so.

## 1. Novelty assessment (known-work set)

Procedure: each analysed paper's own verified question became a candidate gap; the paper was removed from the local corpus and the full novelty check ran against OpenAlex, arXiv and Crossref (2026-11-27, model judge: none).

| Metric | Value |
|---|---|
| Items | 15 |
| False-novelty rate | 0.0 (95% CI [0.0, 0.204]) |
| Detected as already / partially investigated | 1.0 |
| Insufficient coverage | 0.0 |
| Source paper among screened / compared | 0.933 / 0.867 |
| Distinct question shapes | 4 of 15 |
| Specificity control: implausible combinations not called “already investigated” | 5 of 5 |
| Positive control detected | True |

Limitations: The gap text is the paper's own sentence, so the exact-keyword query shares wording with the target; this makes retrieval easier than for a gap phrased independently. Recall is therefore an upper bound. Dimensions missing from the sentence are filled from the source paper's own tags (as for real gaps). The source paper stays in the external indexes; only the local corpus excludes it. Comparisons use titles and abstracts; papers without abstracts are compared on titles. Thresholds in CoveragePolicy were set before this run; tuning them on this set would overfit. A small set: rates have wide uncertainty (Wilson 95% interval reported). If most questions share the corpus topic (see distinct_core_dimension_sets), the set cannot separate a good checker from one that always answers 'already investigated'; the specificity control is reported for that reason. Ontology surface forms are matched without word sense: 'leakage' also matches privacy leakage, and 'integrated photonics' marks a trapped-ion device as photonic.

## 2. Literature retrieval (known-work set)

Procedure: rank of the known source paper in the stored results of every query (15 gaps; 0 failed searches). Up to 10 results per query, so Recall@10 is the ceiling.

| Pool | n | Recall@1 | Recall@3 | Recall@10 | MRR |
|---|---|---|---|---|---|
| All queries, all sources | 15 | 0.733 | 0.8 | 0.933 | 0.787 |
| Source: arxiv | 15 | 0.0 | 0.0 | 0.0 | 0.0 |
| Source: crossref | 15 | 0.067 | 0.067 | 0.133 | 0.075 |
| Source: openalex | 15 | 0.733 | 0.8 | 0.933 | 0.787 |
| Query: concepts | 15 | 0.0 | 0.0 | 0.067 | 0.008 |
| Query: exact | 15 | 0.667 | 0.8 | 0.933 | 0.753 |
| Query: method | 2 | 0.0 | 0.0 | 0.0 | 0.0 |
| Query: statement_context | 3 | 1.0 | 1.0 | 1.0 | 1.0 |
| Query: synonym | 15 | 0.0 | 0.0 | 0.0 | 0.0 |

Limitations: one known relevant paper per query, so precision is not measured; the gap text is the paper's own sentence, which favours the own-words query.

## 3. Extraction

Corpus: 129 statements from 22 papers ({'interpretation': 71, 'unverified': 5, 'direct': 53}); 0.957 of model statements had their quoted evidence found in the source.

Verifier stress test on 124 verified quotes from 21 papers: detection of fabricated quotes **0.996**, false rejection of genuine quotes **0.0**.

| Variant | Accepted / total |
|---|---|
| fabricated: negation inserted | 0 / 54 |
| fabricated: number changed | 0 / 32 |
| fabricated: quote from another paper | 0 / 124 |
| fabricated: term swapped | 1 / 59 |
| genuine: accents stripped | 124 / 124 |
| genuine: as quoted | 124 / 124 |
| genuine: lower-cased | 124 / 124 |
| genuine: whitespace collapsed | 124 / 124 |

Phase 9 found that fuzzy matching alone accepted every changed number and inserted negation and 73% of swapped terms (detection 0.52). The verifier now also requires numbers, negations and content words to agree with the matched passage; the table shows the result after that change.

Field accuracy: not measured (0 of 94 rows labelled). Attribution accuracy: not measured (0 labelled).

## 4. Gaps

17 candidate gaps ({'stated_limitation': 10, 'unresolved_problem': 1, 'future_work': 2, 'unexplored_combination': 4}). Supporting evidence is a verified quote for 0.765; 0.294 name two or more research dimensions of their own; 1 near-duplicate pair(s) at cosine ≥ 0.85; novelty outcomes {'partially_investigated': 4, 'not checked': 8, 'already_investigated': 5}.

Expert ratings: relevance not measured, evidence support not measured (0 of 17 labelled).

## 5. Experiment plans

Rubric means (1-5): consistency not measured (0 of 5 labelled), completeness not measured (0 of 5 labelled), feasibility not measured (0 of 5 labelled), alignment not measured (0 of 5 labelled).

## 6. Quantum verification

| Set | Cases | Correct | False pass | False fail |
|---|---|---|---|---|
| Proposal benchmark | 6 | 6 | 0 of 4 | 0 of 2 |
| Plan cases (textbook ground truth) | 12 | 12 | 0 of 7 | 0 of 5 |

Plan cases: bell → pass; ghz → pass; ghz wrong populations → fail; ghz-minus target (phase) → fail; ry(pi/3) → pass; ry(pi/3) wrong prediction → fail; x then h → pass; impossible noise T2 > 2T1 → fail; noisy prediction survives weak noise → pass; noisy prediction fails 10% readout → fail; no circuit → hold; circuit without predictions → hold.

## 7. Future-work and partially-investigated sets

Future-work statements addressed later: not measured (0 of 1 labelled). Partially-investigated variants labelled: 0 of 3.

## How to add the human-labelled numbers

Fill the `label_*` columns in `data/benchmark/labels/*.csv` and run `python evaluate.py --all` again. Existing sheets are never overwritten.

## Limitations

- A small corpus on one topic (entanglement and decoherence); rates have wide uncertainty.
- Most papers were analysed from abstracts only.
- Novelty and retrieval ground truth is one known published paper per question.
- Verification ground truth covers small textbook circuits and the 6 hand-written benchmark cases.
- Human-rated metrics are not measured until labelled.
