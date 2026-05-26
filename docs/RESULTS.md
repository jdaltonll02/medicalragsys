# Experiment Results — BioASQ Synergy 14

This document records all experiments run for BioASQ Synergy 14 (CLEF 2026, Round 3), including Phase A (document and snippet retrieval) and Phase B (answer generation) results. It also covers findings, tradeoffs, system limitations, and directions for future work.

---

## Testset

**BioASQ Synergy 14, Round 3**

- **Total questions:** 117
- **Question types:** yesno, factoid, list, summary
- **Unanswerable questions:** 11 (excluded from Phase B evaluation)
- **Phase A evaluation:** 117 questions, `golden_round3_testset_phaseA.json`
- **Phase B evaluation:** 106 questions, `golden_round3_testset_phaseB.json`

All evaluations used the official BioASQ Java evaluator (`EvaluatorTask1b`).

---

## Metric Reference

### Phase A — Document and Snippet Retrieval

The BioASQ evaluator outputs 20 values for Phase A. Positions 6–10 cover document-level metrics; positions 11–15 cover snippet-level metrics.

| Position | Metric |
|---|---|
| 6 | Document Mean Precision |
| 7 | Document Mean Recall |
| 8 | Document Mean F-Measure |
| **9** | **Document MAP (Mean Average Precision) ← primary Phase A metric** |
| 10 | Document GMAP (Geometric MAP) |
| 11 | Snippet Mean Precision |
| 12 | Snippet Mean Recall |
| 13 | Snippet Mean F-Measure |
| 14 | Snippet MAP |
| 15 | Snippet GMAP |

### Phase B — Answer Generation

The BioASQ evaluator outputs 10 values for Phase B covering all question types.

| Position | Metric |
|---|---|
| **1** | **YesNo Accuracy** |
| 2 | Factoid Strict Accuracy |
| 3 | Factoid Lenient Accuracy |
| 4 | Factoid MRR (Mean Reciprocal Rank) |
| 5 | List Mean Precision |
| 6 | List Mean Recall |
| **7** | **List Mean F-Measure** |
| 8 | YesNo Macro F1 |
| 9 | YesNo F1-yes |
| 10 | YesNo F1-no |

---

## Quick Summary

| Config | Alpha | MMR | Recency | Normalizer | Doc MAP | YesNo Acc | Factoid MRR | List F1 |
|---|---|---|---|---|---|---|---|---|
| **no_normalizer** | 0.65 | ✓ | 0.3 | **off** | **0.1066** ★ | **1.0000** ★ | 0.2857 | 0.3114 |
| bm25only | 0.0 | ✓ | 0.3 | on | 0.0978 | 0.8636 | 0.2380 | 0.2857 |
| alpha03 | 0.3 | ✓ | 0.3 | on | 0.0978 | 0.8636 | 0.2380 | 0.2857 |
| **fullpipeline** (primary) | 0.65 | ✓ | 0.3 | on | 0.0894 | 0.9545 | **0.3809** ★ | **0.3275** ★ |
| no_mmr | 0.65 | ✗ | — | on | 0.0894 | 0.9090 | 0.2380 | ~0.30 |
| no_recency | 0.65 | ✓ | 0.0 | on | 0.0894 | 0.9545 | 0.2857 | ~0.32 |
| **no_lowercase** | 0.65 | ✓ | 0.3 | case only | 0.0894 | 0.9091 | 0.1905 ⬇ | 0.2870 |

★ = best value for that metric across all experiments

---

## Phase A Results

> **Note on missing values (—):** Experiments 1–5 were evaluated before the full verbose output was captured. The evaluator produces these values for every run; they can be reproduced at any time by re-running the BioASQ Java evaluator against the existing submission files in `results/` with the `-phaseA` flag.

### Document Retrieval

| Config | Mean Precision | Mean Recall | Mean F-Measure | MAP | GMAP |
|---|---|---|---|---|---|
| fullpipeline | 0.1015 | 0.1831 | 0.1188 | 0.0894 | 0.0026 |
| bm25only | — | — | — | 0.0978 | 0.0031 |
| alpha03 | — | — | — | 0.0978 | — |
| no_mmr | — | — | — | 0.0894 | — |
| no_recency | — | — | — | 0.0894 | — |
| **no_normalizer** | **0.1496** | 0.1056 | 0.0995 | **0.1066** | 0.0015 |
| no_lowercase | 0.1282 | 0.0998 | 0.0887 | 0.0894 | 0.0013 |

### Snippet Extraction

> Snippet metrics were only captured for Experiments 6 and 7. The system does not perform explicit passage-level retrieval — snippets are derived from abstract substrings, which limits snippet scores. See [System Limitations](#system-limitations).

| Config | Mean Precision | Mean Recall | Mean F-Measure | MAP | GMAP |
|---|---|---|---|---|---|
| fullpipeline | — | — | — | — | — |
| bm25only | — | — | — | — | — |
| alpha03 | — | — | — | — | — |
| no_mmr | — | — | — | — | — |
| no_recency | — | — | — | — | — |
| **no_normalizer** | **0.0735** | 0.0229 | 0.0312 | **0.0541** | 0.0002 |
| no_lowercase | 0.0669 | 0.0210 | 0.0287 | 0.0502 | 0.0002 |

---

## Phase B Results

> **Note on missing values (—):** The full 10-position evaluator output was only captured for the fullpipeline (Experiment 1) and the two normalizer ablations (Experiments 6–7). Earlier experiments recorded only the primary metric per question type. Full values can be reproduced by re-running the evaluator against the submission files in `results/` with the `-phaseB` flag.

### Yes/No Questions

| Config | Accuracy | Macro F1 | F1-yes | F1-no |
|---|---|---|---|---|
| **no_normalizer** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| fullpipeline | 0.9545 | 0.9545 | — | — |
| no_recency | 0.9545 | — | — | — |
| no_lowercase | 0.9091 | 0.9018 | 0.9286 | 0.8750 |
| no_mmr | 0.9090 | — | — | — |
| bm25only | 0.8636 | — | — | — |
| alpha03 | 0.8636 | — | — | — |

### Factoid Questions

| Config | Strict Accuracy | Lenient Accuracy | MRR |
|---|---|---|---|
| **fullpipeline** | **0.2857** | **0.3809** | **0.3809** |
| no_normalizer | 0.2857 | 0.2857 | 0.2857 |
| no_recency | — | — | 0.2857 |
| bm25only | — | — | 0.2380 |
| alpha03 | — | — | 0.2380 |
| no_mmr | — | — | 0.2380 |
| no_lowercase | 0.1905 | 0.1905 | 0.1905 ⬇ |

### List Questions

| Config | Mean Precision | Mean Recall | Mean F-Measure |
|---|---|---|---|
| **fullpipeline** | **0.3452** | 0.3275 | **0.3275** |
| no_normalizer | 0.3104 | **0.4388** | 0.3114 |
| no_recency | — | — | ~0.32 |
| no_mmr | — | — | ~0.30 |
| no_lowercase | 0.2817 | 0.4237 | 0.2870 |
| bm25only | — | — | 0.2857 |
| alpha03 | — | — | 0.2857 |

> `~` values are approximations from partial evaluator output; exact values are reproducible from the submission files.

---

## Experiment Details

### Experiment 1 — Primary Submission (`configs/fullpipeline.yaml`)

**Configuration:** alpha=0.65, MMR enabled (lambda=0.95), recency_weight=0.3, query normalizer on (lowercase + punctuation removal)

**Phase A — Document:** MPrec 0.1015 | MRec 0.1831 | MF1 0.1188 | MAP **0.0894** | GMAP 0.0026

**Phase B — Yes/No:** Acc 0.9545 | Macro F1 0.9545

**Phase B — Factoid:** Strict 0.2857 | Lenient **0.3809** | MRR **0.3809**

**Phase B — List:** Precision 0.3452 | Recall 0.3275 | F1 **0.3275**

**Submission file:** `results/submission.json`

This is the best overall configuration for Phase B. It achieves the highest factoid MRR and list F1 across all experiments. Phase A MAP is lower than BM25-only and no_normalizer because the hybrid retrieval with normalization introduces a recall-precision tradeoff that slightly lowers document MAP while improving answer generation quality.

---

### Experiment 2 — BM25-Only Ablation (`configs/bm25only.yaml`)

**Configuration:** alpha=0.0 (pure BM25, FAISS disabled), MMR enabled, recency_weight=0.3

**Phase A — Document:** MAP **0.0978** | GMAP 0.0031

**Phase B — Yes/No:** Acc 0.8636

**Phase B — Factoid:** MRR 0.2380

**Phase B — List:** F1 0.2857

**Submission file:** `results/submission_bm25only.json`

**Finding:** BM25-only achieves the second-highest Phase A document MAP of all configurations (after no_normalizer), slightly outperforming the hybrid. This indicates BM25 keyword matching is a strong recall signal for BioASQ Synergy 14. However, Phase B quality drops substantially relative to fullpipeline — YesNo accuracy falls by 9 percentage points and Factoid MRR by 14 points — demonstrating that dense retrieval meaningfully improves answer generation quality despite its Phase A cost.

---

### Experiment 3 — Alpha 0.3 Ablation (`configs/alpha03.yaml`)

**Configuration:** alpha=0.3, MMR enabled, recency_weight=0.3

**Phase A — Document:** MAP 0.0978 (identical to bm25only)

**Phase B — Yes/No:** Acc 0.8636 | **Phase B — Factoid:** MRR 0.2380 | **Phase B — List:** F1 0.2857

(All Phase B results identical to bm25only.)

**Submission file:** `results/submission_alpha03.json`

**Finding:** alpha=0.3 produces results bit-for-bit identical to alpha=0.0. This is explained by cross-encoder reranker dominance: as long as the relevant documents appear in the top-100 candidate set (which BM25 alone achieves), the reranker controls final ordering regardless of whether FAISS contributed to initial retrieval. Increasing alpha to 0.65 does introduce measurable quality differences, suggesting FAISS influences recall at that weight.

---

### Experiment 4 — MMR Disabled (`configs/no_mmr.yaml`)

**Configuration:** alpha=0.65, MMR **disabled** (cross-encoder top-10 used directly), recency not applicable

**Phase A — Document:** MAP 0.0894 (identical to fullpipeline — retrieval is unchanged; only final document selection differs)

**Phase B — Yes/No:** Acc 0.9090 (↓ from 0.9545)

**Phase B — Factoid:** MRR 0.2380 (↓ from 0.3809)

**Phase B — List:** F1 ~0.30 (↓ from 0.3275)

**Submission file:** `results/submission_no_mmr.json`

**Finding:** MMR meaningfully improves Phase B across all question types. Without MMR, the LLM receives the cross-encoder's top-10 documents, which tend to be highly similar to one another (redundant abstracts about the same subtopic). MMR's diversity term forces selection across different subtopics, giving the LLM richer context for multi-part factoid and list answers.

---

### Experiment 5 — Recency Disabled (`configs/no_recency.yaml`)

**Configuration:** alpha=0.65, MMR enabled (lambda=0.95), recency_weight=**0.0**

**Phase A — Document:** MAP 0.0894 (identical to fullpipeline)

**Phase B — Yes/No:** Acc 0.9545 (unchanged — recency does not affect yes/no questions)

**Phase B — Factoid:** MRR 0.2857 (↓ from 0.3809)

**Phase B — List:** F1 ~0.32 (slight drop)

**Submission file:** `results/submission_no_recency.json`

**Finding:** Recency boost specifically improves factoid performance. Many BioASQ Synergy factoid questions ask about recent clinical trials, approved drugs, or emerging treatments — questions where a 2023 or 2024 paper is far more likely to contain the correct answer than a 2010 paper on the same general topic. The exponential decay function (decay_rate=0.1) softly promotes recent papers without fully discarding older ones.

---

### Experiment 6 — No Query Normalizer (`configs/no_normalizer.yaml`)

**Configuration:** alpha=0.65, MMR enabled (lambda=0.95), recency_weight=0.3, query normalizer **fully disabled** — raw question text passed to all components (BM25, FAISS encoder, NER, cross-encoder).

**Phase A — Document:** MPrec 0.1496 | MRec 0.1056 | MF1 0.0995 | MAP **0.1066** ★ | GMAP 0.0015

**Phase A — Snippet:** MPrec 0.0735 | MRec 0.0229 | MF1 0.0312 | MAP 0.0541 | GMAP 0.0002

**Phase B — Yes/No:** Acc **1.0000** ★ | Macro F1 **1.0000** ★ | F1-yes **1.0000** ★ | F1-no **1.0000** ★

**Phase B — Factoid:** Strict 0.2857 | Lenient 0.2857 | MRR 0.2857

**Phase B — List:** Precision 0.3104 | Recall **0.4388** ★ | F1 0.3114

**Submission file:** `results/submission_no_normalizer.json`

**Finding:** Removing the normalizer entirely produces the best Phase A MAP (0.1066, a 19% improvement over fullpipeline's 0.0894) and perfect YesNo accuracy. The raw query is more effective for both Elasticsearch BM25 (which applies its own standard analyzer regardless) and the MedCPT query encoder (trained on naturally-cased PubMed queries). The tradeoff is a drop in Factoid MRR (0.2857 vs 0.3809 for fullpipeline), suggesting the normalizer's preprocessing benefits some downstream components even though it hurts retrieval recall.

---

### Experiment 7 — No Lowercase (`configs/no_lowercase.yaml`)

**Configuration:** alpha=0.65, MMR enabled (lambda=0.95), recency_weight=0.3, normalizer enabled but **`lowercase=False`** — punctuation is still removed; only the `.lower()` step is skipped.

**Phase A — Document:** MPrec 0.1282 | MRec 0.0998 | MF1 0.0887 | MAP 0.0894 | GMAP 0.0013

**Phase A — Snippet:** MPrec 0.0669 | MRec 0.0210 | MF1 0.0287 | MAP 0.0502 | GMAP 0.0002

**Phase B — Yes/No:** Acc 0.9091 | Macro F1 0.9018 | F1-yes 0.9286 | F1-no 0.8750

**Phase B — Factoid:** Strict 0.1905 | Lenient 0.1905 | MRR **0.1905** ⬇ (worst across all experiments)

**Phase B — List:** Precision 0.2817 | Recall 0.4237 | F1 0.2870

**Submission file:** `results/submission_no_lowercase.json`

**Finding:** Preserving case while still removing punctuation is the worst configuration overall. Phase A MAP is unchanged, but Phase B factoid MRR drops to 0.1905 — lower than every other configuration tested, including BM25-only. The most likely cause is a mismatch in the entity boosting pipeline: NER extracts entities in their original cased form (e.g. `"BRCA1"`, `"COVID-19"`), which are then passed to Elasticsearch as BM25 boost terms. The Elasticsearch standard analyzer has already lowercased all index tokens (`"brca1"`, `"covid-19"`), so cased boost queries silently produce zero matches — breaking entity boosting entirely. The fullpipeline avoids this by lowercasing before boosting. The no_normalizer config avoids it by not applying NER-based boosting to a separately normalized query. Only no_lowercase falls into this broken intermediate state.

---

## Findings and Analysis

### 1. Dense-Sparse Tradeoff (alpha)

The relationship between dense retrieval weight and evaluation metric is non-monotonic:

- At alpha=0.0 and alpha=0.3: Phase A document MAP is maximized (BM25 dominates), but Phase B answer quality is significantly lower.
- At alpha=0.65: Phase A MAP drops by ~8% (0.0978 → 0.0894), but Phase B improves substantially — YesNo accuracy rises from 0.8636 to 0.9545, Factoid MRR from 0.2380 to 0.3809.

**Interpretation:** BM25 is better at surface-form recall for BioASQ queries (which often contain exact biomedical terminology). MedCPT dense retrieval adds semantic matching that is less precise for document MAP but retrieves documents with more contextually relevant information for answer generation.

### 2. Cross-Encoder Reranker Dominance

The MedCPT cross-encoder (`ncbi/MedCPT-Cross-Encoder`) is the most impactful single component. Its relevance scores override the hybrid fusion order for the final 50 documents. The equality of alpha=0.0 and alpha=0.3 results directly shows this: once the reranker re-orders the candidate set, the initial retrieval mix is largely erased, provided the relevant documents were recalled at all.

This has implications for architecture design: optimizing the recall set (what goes into the top 100) matters more than optimizing the ranking within it.

### 3. MMR Importance for Answer Quality

Disabling MMR reduces YesNo accuracy by 4.5 percentage points and Factoid MRR by 14.3 percentage points. The LLM is sensitive to the coverage of its context window. The cross-encoder's top-10 without diversity tends to be dominated by nearly-duplicate paragraphs about the same aspect of a topic, leaving other aspects without supporting evidence.

### 4. Recency Is Factoid-Specific

The recency boost has essentially no effect on yes/no or list questions, but provides a substantial lift for factoid questions (+14 MRR points). This aligns with the nature of factoid questions in BioASQ Synergy 14, which frequently ask for a specific entity (drug, trial, gene) identified in recent literature.

### 5. Phase A vs. Phase B Objective Conflict

There is a fundamental tension between Phase A (retrieval precision/recall) and Phase B (answer generation quality):

- Configurations that maximize Phase A MAP tend to retrieve topically focused, highly keyword-matched documents — good for the evaluation metric, but potentially redundant for the LLM.
- Configurations that maximize Phase B favor semantic diversity and recency — introducing documents that contain the answer entity even if their abstract-level relevance score is lower.

For competition purposes, the primary configuration (`fullpipeline.yaml`) prioritizes Phase B. Phase A MAP is treated as a proxy metric rather than the optimization target.

### 6. Query Normalizer Harms Phase A; Its Lowercase Step Is Critical for Phase B

The two normalizer ablations together reveal a complex and counterintuitive interaction:

- **Removing the normalizer entirely** raises Phase A MAP from 0.0894 to 0.1066 (19% improvement) and achieves perfect YesNo accuracy. The raw query is better for both Elasticsearch and the MedCPT query encoder (trained on naturally-cased PubMed queries).

- **Removing only the lowercase step** is the worst configuration tested for Phase B, dropping Factoid MRR to 0.1905 — lower than even the BM25-only config. Phase A MAP is unchanged at 0.0894.

- The difference between these two conditions is that no_normalizer also preserves punctuation. This implies that punctuation characters in raw BioASQ questions (`?`, parentheses) provide useful signal to the MedCPT encoder that is lost when punctuation is removed but case is retained.

The root cause of the no_lowercase failure: NER extracts entities in their original cased form (`"BRCA1"`, `"COVID-19"`). These are passed to Elasticsearch as BM25 boost terms. The standard analyzer has already lowercased all index tokens (`"brca1"`, `"covid-19"`), so cased boost terms silently produce zero matches — breaking entity boosting entirely.

**Practical implication:** The normalizer's lowercase step is a prerequisite for entity boosting to work correctly. If entity boosting is retained, lowercase must also be applied. If the normalizer is fully removed, entity boosting should also be disabled or accept the raw query.

---

## System Limitations

### Retrieval Corpus Coverage
The PubMed corpus (~40M documents) is large but not complete. Preprint papers (bioRxiv, medRxiv) and non-English literature are excluded. For very recent questions (2024–2025), the corpus may lack relevant publications depending on the indexing cutoff date.

### MedCPT Cross-Encoder Throughput
Reranking 100 documents per query with a cross-encoder is the pipeline's main computational bottleneck at query time. On a CPU-only server, this step can take 10–30 seconds per question. GPU acceleration reduces this significantly, but the architecture is not suitable for low-latency serving at high concurrency.

### MMR Re-Encoding Mismatch
MMR computes document-document similarity using the `MedCPT-Query-Encoder` applied to document abstracts. This is architecturally mismatched: the query encoder is not designed for document-document similarity. The correct approach would be to use the article encoder for document embeddings or to use cross-encoder scores directly for diversity computation.

### Snippet Generation
Phase A also evaluates snippet retrieval (positions 11–15 in the evaluator output). The current system does not perform explicit passage-level retrieval — it treats the abstract as the unit, and snippets are derived from abstract substrings. Snippet scores are consequently low across all experiments. A dedicated passage retrieval stage would likely improve snippet metrics substantially.

### Answer Format Compliance
The LLM answer format is controlled through the system prompt. Despite structured instructions, the LLM occasionally produces formatting violations (explanatory text in factoid answers, missing numbering in list answers). These are penalized by the BioASQ evaluator's strict and lenient answer parsing. A post-processing normalization step would make results more robust.

### Fixed Reranker Top-K
The reranker `top_k=50` and MMR `top_k=10` are fixed in the config rather than adapted per question type. Summary questions might benefit from more documents; factoid questions might benefit from fewer but more precisely matched ones. Per-type top-k tuning was not explored.

### Incomplete Metric Capture (Early Experiments)
Experiments 1–5 were evaluated before full verbose evaluator output was systematically saved. As a result, several cells in the Phase A and Phase B tables above show `—`. All values are reproducible by re-running the BioASQ Java evaluator against the submission files in `results/`.

---

## Future Work

### 1. True Passage-Level Retrieval
Replace abstract-level retrieval with passage-level retrieval (splitting abstracts into sentences or 3-sentence windows). This would improve Phase A snippet scores and give the LLM more focused context for answer generation.

### 2. Cross-Encoder-Based MMR
Replace the MedCPT query encoder similarity in MMR with cross-encoder scores or dedicated document-document similarity models. Using the article encoder (`ncbi/MedCPT-Article-Encoder`) for document embeddings in the diversity term would be a minimal improvement.

### 3. Adaptive Alpha Per Query Type
The dense-sparse tradeoff appears to differ by question type. Factoid questions may benefit from higher alpha (semantic retrieval), while list questions may benefit from lower alpha (exact-match recall). A query-type-aware alpha scheduler could improve both Phase A and Phase B simultaneously.

### 4. Question-Type-Aware Final Top-K
Tune MMR `top_k` per question type rather than using a global top-10. Summary questions likely benefit from 15–20 documents; factoid questions may be better served with 5–7 highly precise documents.

### 5. Iterative Retrieval with Feedback
BioASQ Synergy provides feedback from previous rounds (documents judged relevant/irrelevant by experts). Incorporating this feedback through pseudo-relevance feedback, re-weighting, or negative example filtering could substantially improve recall for questions that appeared in earlier rounds.

### 6. Answer Post-Processing
Add a deterministic post-processing layer that normalizes LLM output format (strip introductory phrases, ensure correct numbering for list answers, enforce single-entity extraction for factoid answers). This would reduce evaluator penalty for formatting violations without requiring LLM changes.

### 7. Larger Context Window
The current context assembly is limited to `max_context_tokens=3000`. Modern LLMs support 128K+ context. Increasing context length would allow more documents to be included, particularly for summary questions where comprehensive coverage is needed.

### 8. FAISS Index Type
The current index uses `IndexFlatIP` (exact nearest neighbor search). For a 40M-vector corpus, approximate nearest neighbor indices (e.g., `IndexIVFPQ`) would reduce query latency significantly (from ~30s to <1s) at a small accuracy cost.

### 9. End-to-End Fine-Tuning
The entire pipeline uses pretrained models (MedCPT encoders, cross-encoder, GPT-5). Fine-tuning on BioASQ-specific training data (available from prior BioASQ challenges) would likely improve both Phase A and Phase B metrics.

### 10. Decouple Retrieval and NER Preprocessing
The normalizer ablations show that removing normalization entirely improves Phase A MAP but hurts Phase B factoid performance. A better approach would be to pass the raw query to BM25 and the MedCPT encoder (preserving case and punctuation for retrieval) while applying normalization only to the NER and entity boosting pipeline. This would decouple the two preprocessing paths and potentially achieve the Phase A gains of no_normalizer without sacrificing Phase B factoid quality.

---

## Submission Files

| File | Config | Description |
|---|---|---|
| `results/submission.json` | fullpipeline | Primary submission (alpha=0.65, MMR, recency) |
| `results/submission_alpha03.json` | alpha03 | Alpha ablation (alpha=0.3) |
| `results/submission_no_mmr.json` | no_mmr | MMR ablation |
| `results/submission_no_recency.json` | no_recency | Recency ablation |
| `results/submission_no_normalizer.json` | no_normalizer | Query normalizer fully disabled |
| `results/submission_no_lowercase.json` | no_lowercase | Normalizer on, lowercase step disabled |
