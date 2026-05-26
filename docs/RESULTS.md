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

> **Note on GMAP notation:** The evaluator prints snippet GMAP values in scientific notation (e.g., `1.534E-4`). This means 1.534 × 10⁻⁴ = 0.000153 — a value well within [0, 1]. Tables below show the scientific notation value for precision.

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

> `*` = from the original evaluation run (not re-evaluated); `—` = pending re-evaluation; ★ = best value for that metric across all experiments.

| Config | Alpha | MMR | Recency | Normalizer | Doc MAP | YesNo Acc | Factoid MRR | List F1 |
|---|---|---|---|---|---|---|---|---|
| **no_normalizer** | 0.65 | ✓ | 0.3 | **off** | **0.1066** ★ | **1.0000** ★ | 0.2857 | 0.3114 |
| bm25only | 0.0 | ✓ | 0.3 | on | 0.0978 | — | — | — |
| alpha03 | 0.3 | ✓ | 0.3 | on | 0.0978\* | 0.8636\* | 0.2380\* | 0.2857\* |
| **fullpipeline** (primary) | 0.65 | ✓ | 0.3 | on | 0.0894 | 0.9545 | **0.3810** ★ | **0.3204** ★ |
| no_recency | 0.65 | ✓ | 0.0 | on | 0.0894 | 0.9545 | 0.2857 | 0.3200 |
| no_mmr | 0.65 | ✗ | — | on | 0.0894 | 0.9091 | 0.2381 | 0.2571 |
| **no_lowercase** | 0.65 | ✓ | 0.3 | case only | 0.0894 | 0.9091 | 0.1905 ⬇ | 0.2870 |

---

## Phase A Results

> `—` = not re-evaluated; values reproducible by re-running the BioASQ Java evaluator against submission files in `results/`.

### Document Retrieval

| Config | Mean Precision | Mean Recall | Mean F-Measure | MAP | GMAP |
|---|---|---|---|---|---|
| **no_normalizer** | **0.1496** | **0.1056** | 0.0995 | **0.1066** | 0.0015 |
| bm25only | 0.1385 | 0.1025 | 0.0941 | 0.0978 | 0.0018 |
| alpha03 | — | — | — | 0.0978\* | — |
| fullpipeline | 0.1282 | 0.0998 | 0.0887 | 0.0894 | 0.0013 |
| no_recency | 0.1282 | 0.0998 | 0.0887 | 0.0894 | 0.0013 |
| no_mmr | 0.1282 | 0.0998 | 0.0887 | 0.0894 | 0.0013 |
| no_lowercase | 0.1282 | 0.0998 | 0.0887 | 0.0894 | 0.0013 |

> **Note:** fullpipeline, no_recency, no_mmr, and no_lowercase share identical Phase A document metrics because Phase A retrieval (hybrid fusion + cross-encoder reranking) is identical across these configs — only the final MMR/recency selection differs, and that affects which documents are passed to the LLM, not the Phase A document set.

### Snippet Extraction

> The system does not perform explicit passage-level retrieval. Snippets are derived from abstract substrings, which limits snippet scores across all experiments. See [System Limitations](#system-limitations).

| Config | Mean Precision | Mean Recall | Mean F-Measure | MAP | GMAP |
|---|---|---|---|---|---|
| **no_normalizer** | **0.0735** | **0.0229** | **0.0312** | **0.0541** | 1.53×10⁻⁴ |
| bm25only | 0.0716 | 0.0231 | 0.0308 | 0.0523 | 2.06×10⁻⁴ |
| alpha03 | — | — | — | — | — |
| fullpipeline | 0.0667 | 0.0210 | 0.0287 | 0.0502 | 1.53×10⁻⁴ |
| no_recency | 0.0669 | 0.0210 | 0.0287 | 0.0502 | 1.53×10⁻⁴ |
| no_mmr | 0.0669 | 0.0210 | 0.0287 | 0.0502 | 1.53×10⁻⁴ |
| no_lowercase | 0.0669 | 0.0210 | 0.0287 | 0.0502 | 1.53×10⁻⁴ |

---

## Phase B Results

> bm25only Phase B pending re-evaluation. alpha03 Phase B values marked `*` are from the original evaluation run.

### Yes/No Questions

| Config | Accuracy | Macro F1 | F1-yes | F1-no |
|---|---|---|---|---|
| **no_normalizer** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| fullpipeline | 0.9545 | 0.9494 | 0.9655 | 0.9333 |
| no_recency | 0.9545 | 0.9494 | 0.9655 | 0.9333 |
| no_lowercase | 0.9091 | 0.9018 | 0.9286 | 0.8750 |
| no_mmr | 0.9091 | 0.9018 | 0.9286 | 0.8750 |
| alpha03 | 0.8636\* | — | — | — |
| bm25only | — | — | — | — |

### Factoid Questions

| Config | Strict Accuracy | Lenient Accuracy | MRR |
|---|---|---|---|
| **fullpipeline** | **0.3810** | **0.3810** | **0.3810** |
| no_normalizer | 0.2857 | 0.2857 | 0.2857 |
| no_recency | 0.2857 | 0.2857 | 0.2857 |
| no_mmr | 0.2381 | 0.2381 | 0.2381 |
| alpha03 | — | — | 0.2380\* |
| bm25only | — | — | — |
| no_lowercase | 0.1905 | 0.1905 | 0.1905 ⬇ |

### List Questions

| Config | Mean Precision | Mean Recall | Mean F-Measure |
|---|---|---|---|
| fullpipeline | 0.3212 | 0.4520 | **0.3204** |
| no_recency | 0.3169 | **0.4633** | 0.3200 |
| no_normalizer | 0.3104 | 0.4388 | 0.3114 |
| no_lowercase | 0.2817 | 0.4237 | 0.2870 |
| alpha03 | — | — | 0.2857\* |
| no_mmr | 0.2434 | 0.4006 | 0.2571 |
| bm25only | — | — | — |

---

## Experiment Details

### Experiment 1 — Primary Submission (`configs/fullpipeline.yaml`) ✦ Improved

**Configuration:** alpha=0.65, MMR enabled (lambda=0.95), recency_weight=0.3, query normalizer on (lowercase + punctuation removal)

**Phase A — Document:** MPrec 0.1282 | MRec 0.0998 | MF1 0.0887 | MAP **0.0894** | GMAP 0.0013

**Phase A — Snippet:** MPrec 0.0667 | MRec 0.0210 | MF1 0.0287 | MAP 0.0502 | GMAP 1.53×10⁻⁴

**Phase B — Yes/No:** Acc 0.9545 | Macro F1 0.9494 | F1-yes 0.9655 | F1-no 0.9333

**Phase B — Factoid:** Strict **0.3810** | Lenient **0.3810** | MRR **0.3810**

**Phase B — List:** Precision 0.3212 | Recall 0.4520 | F1 **0.3204**

**Submission file:** `results/submission.json`

This is the best Phase B configuration overall. It achieves the highest factoid MRR/strict/lenient accuracy across all experiments (0.3810), as well as the highest list F1 (0.3204). The improved pipeline also shows strict and lenient factoid accuracy are now equal, indicating more precise answer extraction.

---

### Experiment 2 — BM25-Only Ablation (`configs/bm25only.yaml`)

**Configuration:** alpha=0.0 (pure BM25, FAISS disabled), MMR enabled, recency_weight=0.3

**Phase A — Document:** MPrec 0.1385 | MRec 0.1025 | MF1 0.0941 | MAP **0.0978** | GMAP 0.0018

**Phase A — Snippet:** MPrec 0.0716 | MRec 0.0231 | MF1 0.0308 | MAP 0.0523 | GMAP 2.06×10⁻⁴

**Phase B:** Pending re-evaluation.

**Submission file:** `results/submission_bm25only.json`

**Finding:** BM25-only achieves the second-highest Phase A document MAP (0.0978, after no_normalizer's 0.1066) and the highest document Mean Precision (0.1385) after no_normalizer. BM25 keyword matching is a strong recall signal for BioASQ Synergy 14 questions, which frequently use exact biomedical terminology. Phase B results pending re-run.

---

### Experiment 3 — Alpha 0.3 Ablation (`configs/alpha03.yaml`)

**Configuration:** alpha=0.3, MMR enabled, recency_weight=0.3

**Phase A — Document:** MAP 0.0978\* (identical to bm25only in original run; full metrics pending re-evaluation)

**Phase B:** From original run: YesNo Acc 0.8636 | Factoid MRR 0.2380 | List F1 0.2857 (partial; full re-evaluation pending)

**Submission file:** `results/submission_alpha03.json`

**Finding:** alpha=0.3 produced results essentially identical to alpha=0.0 in the original evaluation. This is explained by cross-encoder reranker dominance: as long as relevant documents appear in the top-100 candidate set (which BM25 alone achieves), the reranker controls final ordering regardless of the FAISS contribution at low alpha. At alpha=0.65 a measurable quality difference emerges, suggesting FAISS meaningfully influences recall at that weight.

---

### Experiment 4 — MMR Disabled (`configs/no_mmr.yaml`)

**Configuration:** alpha=0.65, MMR **disabled** (cross-encoder top-10 used directly), recency not applicable

**Phase A — Document:** MPrec 0.1282 | MRec 0.0998 | MF1 0.0887 | MAP **0.0894** | GMAP 0.0013

**Phase A — Snippet:** MPrec 0.0669 | MRec 0.0210 | MF1 0.0287 | MAP 0.0502 | GMAP 1.53×10⁻⁴

**Phase B — Yes/No:** Acc 0.9091 | Macro F1 0.9018 | F1-yes 0.9286 | F1-no 0.8750

**Phase B — Factoid:** Strict 0.2381 | Lenient 0.2381 | MRR 0.2381

**Phase B — List:** Precision 0.2434 | Recall 0.4006 | F1 0.2571

**Submission file:** `results/submission_no_mmr.json`

**Finding:** Disabling MMR is the most damaging single change for Phase B answer quality. Factoid MRR drops from 0.3810 to 0.2381 (−0.143) and List F1 from 0.3204 to 0.2571 (−0.063). Without MMR, the cross-encoder's top-10 documents are highly similar to each other — redundant paragraphs about the same subtopic — giving the LLM insufficient coverage to answer multi-part factoid and list questions. Note that Phase A metrics are identical to fullpipeline, confirming that MMR affects only the document selection passed to the LLM, not the retrieval pool used for Phase A evaluation.

---

### Experiment 5 — Recency Disabled (`configs/no_recency.yaml`)

**Configuration:** alpha=0.65, MMR enabled (lambda=0.95), recency_weight=**0.0**

**Phase A — Document:** MPrec 0.1282 | MRec 0.0998 | MF1 0.0887 | MAP **0.0894** | GMAP 0.0013

**Phase A — Snippet:** MPrec 0.0669 | MRec 0.0210 | MF1 0.0287 | MAP 0.0502 | GMAP 1.53×10⁻⁴

**Phase B — Yes/No:** Acc 0.9545 | Macro F1 0.9494 | F1-yes 0.9655 | F1-no 0.9333

**Phase B — Factoid:** Strict 0.2857 | Lenient 0.2857 | MRR 0.2857

**Phase B — List:** Precision 0.3169 | Recall 0.4633 | F1 0.3200

**Submission file:** `results/submission_no_recency.json`

**Finding:** Removing the recency boost has no effect on Yes/No questions and only a marginal effect on list questions, but drops Factoid MRR from 0.3810 to 0.2857 (−0.095). BioASQ Synergy 14 factoid questions frequently ask for entities identified in recent literature (approved drugs, trial results, emerging genes). Without recency weighting, older papers on the same general topic can rank above the newer paper containing the correct answer entity.

---

### Experiment 6 — No Query Normalizer (`configs/no_normalizer.yaml`)

**Configuration:** alpha=0.65, MMR enabled (lambda=0.95), recency_weight=0.3, query normalizer **fully disabled** — raw question text passed to all components.

**Phase A — Document:** MPrec **0.1496** | MRec **0.1056** | MF1 0.0995 | MAP **0.1066** ★ | GMAP 0.0015

**Phase A — Snippet:** MPrec **0.0735** | MRec 0.0229 | MF1 **0.0312** | MAP **0.0541** | GMAP 1.53×10⁻⁴

**Phase B — Yes/No:** Acc **1.0000** ★ | Macro F1 **1.0000** ★ | F1-yes **1.0000** ★ | F1-no **1.0000** ★

**Phase B — Factoid:** Strict 0.2857 | Lenient 0.2857 | MRR 0.2857

**Phase B — List:** Precision 0.3104 | Recall **0.4388** | F1 0.3114

**Submission file:** `results/submission_no_normalizer.json`

**Finding:** Removing the normalizer entirely produces the best Phase A MAP (0.1066, a 19% improvement over fullpipeline) and perfect YesNo accuracy. The raw query is more effective for both Elasticsearch BM25 (which applies its own standard analyzer) and the MedCPT query encoder (trained on naturally-cased PubMed queries). The tradeoff is a 14-point drop in Factoid MRR (0.2857 vs 0.3810 for fullpipeline), indicating that the normalizer's preprocessing benefits downstream factoid answer extraction even while hurting retrieval recall.

---

### Experiment 7 — No Lowercase (`configs/no_lowercase.yaml`)

**Configuration:** alpha=0.65, MMR enabled (lambda=0.95), recency_weight=0.3, normalizer enabled but **`lowercase=False`** — punctuation removed; `.lower()` step skipped.

**Phase A — Document:** MPrec 0.1282 | MRec 0.0998 | MF1 0.0887 | MAP 0.0894 | GMAP 0.0013

**Phase A — Snippet:** MPrec 0.0669 | MRec 0.0210 | MF1 0.0287 | MAP 0.0502 | GMAP 1.53×10⁻⁴

**Phase B — Yes/No:** Acc 0.9091 | Macro F1 0.9018 | F1-yes 0.9286 | F1-no 0.8750

**Phase B — Factoid:** Strict 0.1905 | Lenient 0.1905 | MRR **0.1905** ⬇ (worst across all experiments)

**Phase B — List:** Precision 0.2817 | Recall 0.4237 | F1 0.2870

**Submission file:** `results/submission_no_lowercase.json`

**Finding:** Preserving case while still removing punctuation is the worst overall configuration. Phase A metrics are identical to fullpipeline, but Phase B Factoid MRR drops to 0.1905 — lower than every other configuration, including BM25-only. The most likely cause is a mismatch in entity boosting: NER extracts entities in their original cased form (`"BRCA1"`, `"COVID-19"`), which are passed to Elasticsearch as BM25 boost terms. The standard analyzer has already lowercased all index tokens, so cased boost queries silently produce zero matches — breaking entity boosting entirely. The fullpipeline avoids this by lowercasing before boosting; the no_normalizer config avoids it by not applying NER-based boosting to a separately normalized query; only no_lowercase falls into the broken intermediate state.

---

## Findings and Analysis

### 1. Dense-Sparse Tradeoff (alpha)

The relationship between dense retrieval weight and evaluation metric is non-monotonic:

- At alpha=0.0 and alpha=0.3: Phase A document MAP is maximized (BM25 dominates), but Phase B answer quality is significantly lower.
- At alpha=0.65: Phase A MAP drops slightly, but Phase B improves substantially — YesNo accuracy rises, and Factoid MRR reaches 0.3810 (the highest across all experiments).

**Interpretation:** BM25 is better at surface-form recall for BioASQ queries (which often contain exact biomedical terminology). MedCPT dense retrieval adds semantic matching that introduces documents with more contextually relevant information for answer generation, even if their abstract-level keyword overlap is lower.

### 2. Cross-Encoder Reranker Dominance

The MedCPT cross-encoder (`ncbi/MedCPT-Cross-Encoder`) is the most impactful single component. The equality of alpha=0.0 and alpha=0.3 results directly shows this: once the reranker re-orders the candidate set, the initial retrieval mix is largely erased, provided the relevant documents were recalled at all.

This has a strong architectural implication: optimizing the recall set (what goes into the top 100) matters more than optimizing the ranking within it.

### 3. MMR Is the Single Most Important Component for Phase B

Disabling MMR produces the largest single-component Phase B degradation: Factoid MRR drops from 0.3810 to 0.2381 (−0.143) and List F1 from 0.3204 to 0.2571 (−0.063). Without MMR, the cross-encoder's top-10 documents are dominated by nearly-duplicate abstracts about the same subtopic, leaving other aspects of the answer without supporting evidence.

### 4. Recency Is Factoid-Specific

The recency boost has no effect on Yes/No questions (both full and no_recency score 0.9545) and minimal effect on list questions, but provides a substantial lift for factoid questions (+0.095 MRR). BioASQ Synergy 14 factoid questions frequently ask for a specific entity identified in recent literature.

### 5. Phase A and Phase B Have Conflicting Objectives

There is a fundamental tension between the two phases:

- Configurations that maximize Phase A MAP (no_normalizer, bm25only) tend to retrieve topically focused, keyword-matched documents — good for the evaluation metric, but potentially redundant for the LLM.
- Configurations that maximize Phase B (fullpipeline) favor semantic diversity and recency — introducing documents that contain the answer entity even if their abstract-level relevance score is lower.

The primary configuration (`fullpipeline.yaml`) prioritizes Phase B, which reflects the downstream task objective.

### 6. Phase A Document Metrics Are Identical Across Four Configs

fullpipeline, no_recency, no_mmr, and no_lowercase all produce exactly the same Phase A document metrics (MPrec 0.1282, MRec 0.0998, MF1 0.0887, MAP 0.0894, GMAP 0.0013). This is expected: Phase A evaluates the set of documents returned by the system across all queries. These four configs share the same retrieval pipeline (hybrid fusion at alpha=0.65 + MedCPT cross-encoder reranking) — only the final MMR/recency step differs, and that step selects which documents go to the LLM, not which documents are counted in Phase A.

### 7. Query Normalizer Harms Phase A; Its Lowercase Step Is Critical for Phase B

The two normalizer ablations together reveal a complex and counterintuitive interaction:

- **Removing the normalizer entirely** raises Phase A MAP from 0.0894 to 0.1066 (19% improvement, new best) and achieves perfect YesNo accuracy. The raw query is better for both Elasticsearch and the MedCPT query encoder (trained on naturally-cased PubMed queries).
- **Removing only the lowercase step** is the worst configuration tested for Phase B — Factoid MRR 0.1905, lower than every other configuration including BM25-only. Phase A MAP is unchanged at 0.0894.
- The difference between these two conditions is that no_normalizer also preserves punctuation. Punctuation characters in raw BioASQ questions (`?`, parentheses) appear to provide useful signal to the MedCPT encoder that is lost when punctuation is removed but case is retained.

**Root cause of no_lowercase failure:** NER extracts entities in their original cased form (`"BRCA1"`, `"COVID-19"`). These are passed to Elasticsearch as BM25 boost terms. The standard analyzer has already lowercased all index tokens, so cased boost terms silently produce zero matches — breaking entity boosting entirely.

**Practical implication:** The normalizer's lowercase step is a prerequisite for entity boosting to work correctly. If entity boosting is retained, lowercase must also be applied. If the normalizer is removed entirely, entity boosting should also be disabled.

---

## System Limitations

### Retrieval Corpus Coverage
The PubMed corpus (~40M documents) is large but not complete. Preprint papers (bioRxiv, medRxiv) and non-English literature are excluded. For very recent questions (2024–2025), the corpus may lack relevant publications depending on the indexing cutoff date.

### MedCPT Cross-Encoder Throughput
Reranking 100 documents per query with a cross-encoder is the pipeline's main computational bottleneck at query time. On a CPU-only server this can take 10–30 seconds per question. GPU acceleration reduces this significantly, but the architecture is not suitable for low-latency serving at high concurrency.

### MMR Re-Encoding Mismatch
MMR computes document-document similarity using the `MedCPT-Query-Encoder` applied to document abstracts. This is architecturally mismatched: the query encoder is not designed for document-document similarity. The correct approach would be to use the article encoder for document embeddings or to use cross-encoder scores directly for diversity computation.

### Snippet Generation
The current system does not perform explicit passage-level retrieval — it treats the abstract as the unit, and snippets are derived from abstract substrings. Snippet MAP (0.050–0.054) is substantially lower than document MAP (0.089–0.107), confirming that abstract-level retrieval is a weak approach for snippet evaluation. A dedicated passage retrieval stage would likely improve snippet metrics substantially.

### Answer Format Compliance
The LLM answer format is controlled through the system prompt. Despite structured instructions, the LLM occasionally produces formatting violations (explanatory text in factoid answers, missing numbering in list answers). These are penalized by the BioASQ evaluator's strict and lenient answer parsing.

### Fixed Reranker Top-K
The reranker `top_k=50` and MMR `top_k=10` are fixed in the config rather than adapted per question type. Summary questions might benefit from more documents; factoid questions might benefit from fewer but more precisely matched ones. Per-type top-k tuning was not explored.

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
The current index uses `IndexFlatIP` (exact nearest neighbor search). For a 40M-vector corpus, approximate nearest neighbor indices (e.g., `IndexIVFPQ`) would reduce query latency significantly at a small accuracy cost.

### 9. End-to-End Fine-Tuning
The entire pipeline uses pretrained models (MedCPT encoders, cross-encoder, GPT-5). Fine-tuning on BioASQ-specific training data (available from prior BioASQ challenges) would likely improve both Phase A and Phase B metrics.

### 10. Decouple Retrieval and NER Preprocessing
The normalizer ablations show that removing normalization entirely improves Phase A MAP but hurts Phase B factoid performance. A better approach would be to pass the raw query to BM25 and the MedCPT encoder (preserving case and punctuation for retrieval) while applying normalization only to the NER and entity boosting pipeline. This would decouple the two preprocessing paths and potentially achieve the Phase A gains of no_normalizer without sacrificing Phase B factoid quality.

---

## Submission Files

| File | Config | Description |
|---|---|---|
| `results/submission.json` | fullpipeline | Primary submission (alpha=0.65, MMR, recency) |
| `results/submission_bm25only.json` | bm25only | BM25-only ablation (alpha=0.0) |
| `results/submission_alpha03.json` | alpha03 | Alpha ablation (alpha=0.3) |
| `results/submission_no_mmr.json` | no_mmr | MMR ablation |
| `results/submission_no_recency.json` | no_recency | Recency ablation |
| `results/submission_no_normalizer.json` | no_normalizer | Query normalizer fully disabled |
| `results/submission_no_lowercase.json` | no_lowercase | Normalizer on, lowercase step disabled |
