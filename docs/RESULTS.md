# Experiment Results — BioASQ Synergy 14

This document records all experiments run for BioASQ Synergy 14 (CLEF 2026, Round 3), including Phase A (document retrieval) and Phase B (answer generation) results. It also covers findings, tradeoffs, system limitations, and directions for future work.

---

## Testset

**BioASQ Synergy 14, Round 3**

- **Total questions:** 117
- **Question types:** yesno (22 answerable + 6 unanswerable), factoid, list, summary
- **Unanswerable questions:** 11 (excluded from Phase B evaluation)
- **Phase A evaluation:** 117 questions, `golden_round3_testset_phaseA.json`
- **Phase B evaluation:** 106 questions, `golden_round3_testset_phaseB.json`

All evaluations used the official BioASQ Java evaluator (`EvaluatorTask1b`).

---

## Metric Reference

**Phase A metrics** (primary: Doc MAP, position 9 in evaluator output):

| Position | Metric |
|---|---|
| 6 | Doc Mean Precision |
| 7 | Doc Mean Recall |
| 8 | Doc Mean F1 |
| **9** | **Doc MAP ← primary Phase A metric** |
| 10 | Doc GMAP |
| 11–15 | Snippet metrics |

**Phase B metrics:**

| Position | Metric |
|---|---|
| 1 | YesNo Accuracy |
| 2 | Factoid Strict Accuracy |
| 3 | Factoid Lenient Accuracy |
| 4 | Factoid MRR |
| 5 | List Precision |
| 6 | List Recall |
| **7** | **List F1** |
| 8 | YesNo macroF1 |
| 9 | YesNo F1-yes |
| 10 | YesNo F1-no |

---

## Experiment Summary Table

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

## Experiment Details

### Experiment 1 — Primary Submission (`configs/fullpipeline.yaml`)

**Configuration:** alpha=0.65, MMR enabled (lambda=0.95), recency_weight=0.3, reranker=MedCPT-Cross-Encoder

**Phase A results:**
```
Doc MPrec: 0.1015  Doc MRec: 0.1831  Doc MF1: 0.1188
Doc MAP: 0.0894    Doc GMAP: 0.0026
```

**Phase B results:**
```
YesNo Acc:        0.9545
Factoid Strict:   0.2857
Factoid Lenient:  0.3809
Factoid MRR:      0.3809
List Precision:   0.3452
List Recall:      0.3275
List F1:          0.3275
YesNo macroF1:    0.9545
```

**Submission file:** `results/submission.json`

This is the best overall configuration. It achieves the highest Phase B scores across all question types at the cost of a slight Phase A MAP reduction relative to BM25-only.

---

### Experiment 2 — BM25-Only Ablation (`configs/bm25only.yaml`)

**Configuration:** alpha=0.0 (pure BM25, FAISS disabled), MMR enabled, recency_weight=0.3

**Phase A results:**
```
Doc MAP: 0.0978    Doc GMAP: 0.0031
```

**Phase B results:**
```
YesNo Acc:        0.8636
Factoid MRR:      0.2380
List F1:          0.2857
```

**Submission file:** `results/submission_bm25only.json` (if generated; identical to alpha03 results)

**Finding:** BM25-only achieves the highest Phase A document MAP of all configurations, slightly outperforming the hybrid. This indicates that for BioASQ Synergy 14 question-document relevance, BM25 keyword matching is a stronger recall signal than dense FAISS retrieval.

---

### Experiment 3 — Alpha 0.3 Ablation (`configs/alpha03.yaml`)

**Configuration:** alpha=0.3, MMR enabled, recency_weight=0.3

**Phase A results:** Identical to BM25-only (Doc MAP: 0.0978)

**Phase B results:** Identical to BM25-only (YesNo 0.8636, Factoid MRR 0.2380, List F1 0.2857)

**Submission file:** `results/submission_alpha03.json`

**Finding:** alpha=0.3 produces results bit-for-bit identical to alpha=0.0. This is explained by the cross-encoder reranker: as long as the relevant documents appear in the top-100 recall set (which BM25 alone achieves), the reranker dominates final ordering regardless of whether FAISS contributed to initial retrieval. Increasing alpha to 0.65 does introduce a measurable quality difference, suggesting FAISS influences recall at that weight.

---

### Experiment 4 — MMR Disabled (`configs/no_mmr.yaml`)

**Configuration:** alpha=0.65, MMR **disabled** (cross-encoder top-10 used directly), recency not applicable

**Phase A results:** Same as fullpipeline (Doc MAP: 0.0894) — retrieval is identical, only final selection changes.

**Phase B results:**
```
YesNo Acc:    0.9090     (↓ from 0.9545 with MMR)
Factoid MRR:  0.2380     (↓ from 0.3809 with MMR)
List F1:      ~0.30      (↓ from 0.3275 with MMR)
```

**Submission file:** `results/submission_no_mmr.json`

**Finding:** MMR meaningfully improves Phase B. Without MMR, the LLM receives the cross-encoder's top-10 documents, which tend to be highly similar to each other (redundant). MMR's diversity term forces selection across different subtopics, giving the LLM richer context for multi-part factoid and list answers.

---

### Experiment 5 — Recency Disabled (`configs/no_recency.yaml`)

**Configuration:** alpha=0.65, MMR enabled (lambda=0.95), recency_weight=**0.0**

**Phase A results:** Same as fullpipeline (Doc MAP: 0.0894).

**Phase B results:**
```
YesNo Acc:    0.9545     (same — recency doesn't affect yes/no)
Factoid MRR:  0.2857     (↓ from 0.3809 with recency)
List F1:      ~0.32      (slight drop)
```

**Submission file:** `results/submission_no_recency.json`

**Finding:** Recency boost specifically improves factoid performance. Many BioASQ Synergy factoid questions ask about recent clinical trials, approved drugs, or emerging treatments — questions where a 2023 or 2024 paper is far more likely to contain the correct answer than a 2010 paper on the same general topic. The exponential decay function (decay_rate=0.1) softly promotes recent papers without fully discarding older ones.

---

### Experiment 6 — No Query Normalizer (`configs/no_normalizer.yaml`)

**Configuration:** alpha=0.65, MMR enabled (lambda=0.95), recency_weight=0.3, query normalizer **fully disabled** — raw question text passed to all components (BM25, FAISS encoder, NER, cross-encoder).

**Phase A results:**
```
Doc MPrec: 0.1496  Doc MRec: 0.1056  Doc MF1: 0.0995
Doc MAP:   0.1066  ← new best across all experiments
Doc GMAP:  0.0015
```

**Phase B results:**
```
YesNo Acc:        1.0000  ← perfect, best across all experiments
Factoid Strict:   0.2857
Factoid Lenient:  0.2857
Factoid MRR:      0.2857
List Precision:   0.3104
List Recall:      0.4388
List F1:          0.3114
YesNo macroF1:    1.0000
```

**Submission file:** `results/submission_no_normalizer.json`

**Finding:** Removing the normalizer entirely produces the best Phase A MAP (0.1066, up from 0.0978 for BM25-only and 0.0894 for fullpipeline — a 19% improvement over the primary config) and perfect YesNo accuracy. The raw query is more effective for both BM25 (Elasticsearch's own standard analyzer handles tokenization) and the MedCPT query encoder (which was trained on naturally-cased PubMed queries). The tradeoff is a drop in Factoid MRR (0.2857 vs 0.3809 for fullpipeline), suggesting that the normalizer's preprocessing benefits some downstream components even though it hurts retrieval recall.

---

### Experiment 7 — No Lowercase (`configs/no_lowercase.yaml`)

**Configuration:** alpha=0.65, MMR enabled (lambda=0.95), recency_weight=0.3, normalizer enabled but **`lowercase=False`** — punctuation is still stripped, only the `.lower()` step is skipped.

**Phase A results:**
```
Doc MPrec: 0.1282  Doc MRec: 0.0998  Doc MF1: 0.0887
Doc MAP:   0.0894  (identical to fullpipeline)
Doc GMAP:  0.0013
```

**Phase B results:**
```
YesNo Acc:        0.9091  (↓ from 0.9545)
Factoid Strict:   0.1905
Factoid Lenient:  0.1905
Factoid MRR:      0.1905  ← worst factoid result across all experiments
List Precision:   0.2817
List Recall:      0.4237
List F1:          0.2870
YesNo macroF1:    0.9018
```

**Submission file:** `results/submission_no_lowercase.json`

**Finding:** Preserving case while still removing punctuation is the worst configuration overall. Phase A MAP is identical to fullpipeline, but Phase B factoid MRR drops to 0.1905 — worse than every other configuration tested. The most likely cause is a mismatch between the query normalizer and Elasticsearch entity boosting: NER extracts entities in their original cased form (e.g. `"COVID-19"`, `"BRCA1"`), which are then used to boost BM25 queries. When case is preserved but the Elasticsearch index has already lowercased all tokens via its standard analyzer, cased boost terms (`"COVID-19"`) fail to match indexed tokens (`"covid-19"`), silently breaking the entity boosting step. Removing the normalizer entirely avoids this by also passing the raw (un-boosted) query to all components consistently.

---

## Findings and Analysis

### 1. Dense-Sparse Tradeoff (alpha)

The relationship between dense retrieval weight and evaluation metric is non-monotonic:

- At alpha=0.0 and alpha=0.3: Phase A MAP is maximized (BM25 dominates), but Phase B answer quality is significantly lower.
- At alpha=0.65: Phase A MAP drops by ~8% (0.0978 → 0.0894), but Phase B improves substantially — YesNo accuracy rises from 0.8636 to 0.9545, Factoid MRR from 0.2380 to 0.3809.

**Interpretation:** BM25 is better at surface-form recall for BioASQ queries (which often contain exact biomedical terminology). MedCPT dense retrieval adds semantic matching that is less precise for document retrieval but retrieves documents with more contextually relevant information for answer generation. The competing objectives of Phase A and Phase B make `alpha` a genuine tradeoff hyperparameter.

### 2. Cross-Encoder Reranker Dominance

The MedCPT cross-encoder (`ncbi/MedCPT-Cross-Encoder`) is the most impactful single component. Its relevance scores override the hybrid fusion order for the final 50 documents. The equality of alpha=0.0 and alpha=0.3 results directly shows this: once the reranker re-orders the candidate set, the initial retrieval mix is largely erased, provided the right documents were recalled.

This has implications for architecture design: optimizing the recall set (what goes into the top 100) matters more than optimizing the ranking within it.

### 3. MMR Importance for Answer Quality

Disabling MMR reduces YesNo accuracy by 4.5 percentage points and Factoid MRR by 14.3 percentage points. The LLM is sensitive to the coverage of its context window. The cross-encoder's top-10 without diversity tends to be dominated by nearly-duplicate paragraphs about the same aspect of a topic, leaving other aspects without supporting evidence.

### 4. Recency Is Factoid-Specific

The recency boost has essentially no effect on yes/no or list questions, but provides a substantial lift for factoid questions (+14 MRR points in this experiment). This aligns with the nature of factoid questions in BioASQ Synergy 14, which frequently ask for a specific entity (drug, trial, gene) that was identified in recent literature.

### 5. Phase A vs. Phase B Objective Conflict

There is a fundamental tension between Phase A (retrieval precision/recall) and Phase B (answer generation quality):

- Configurations that maximize Phase A MAP tend to retrieve topically focused, highly keyword-matched documents — good for the evaluation metric, but potentially redundant for the LLM.
- Configurations that maximize Phase B favor semantic diversity and recency — introducing documents that contain the answer entity even if their abstract-level relevance is lower.

For competition purposes, the primary configuration (`fullpipeline.yaml`) prioritizes Phase B, which reflects the downstream task objective. Phase A MAP is treated as a proxy metric rather than the optimization target.

### 6. Query Normalizer Harms Phase A; Its Lowercase Step Is Critical for Phase B

The two normalizer ablations together reveal a complex and counterintuitive interaction:

- **Removing the normalizer entirely** raises Phase A MAP from 0.0894 to 0.1066 (19% improvement, new best) and achieves perfect YesNo accuracy (1.0). The raw query is better for both Elasticsearch BM25 (which applies its own standard analyzer regardless) and the MedCPT query encoder (trained on naturally-cased PubMed queries).

- **Removing only the lowercase step** (keeping punctuation removal) is the worst configuration tested for Phase B, dropping Factoid MRR to 0.1905 — lower than even the BM25-only config. Phase A MAP is unchanged at 0.0894.

- **Removing the normalizer entirely** is strictly better than removing only the lowercase step. The difference between these two conditions is that no_normalizer also preserves punctuation. This implies that the punctuation characters in raw BioASQ questions (e.g. `?`, parentheses) provide useful signal to the MedCPT encoder that is lost when punctuation is removed.

The most likely explanation for the no_lowercase failure: NER extracts entities in their original cased form (`"BRCA1"`, `"COVID-19"`). These are then passed to Elasticsearch as BM25 boost terms. The Elasticsearch standard analyzer has already lowercased all index tokens (`"brca1"`, `"covid-19"`), so cased boost terms silently produce zero matches — breaking entity boosting entirely. The fullpipeline avoids this by lowercasing entity terms before boosting. The no_normalizer avoids it by not boosting at all (raw query, no entity manipulation). Only the no_lowercase config falls into the broken intermediate state.

**Practical implication:** The normalizer's lowercase step is a prerequisite for the entity boosting logic to work correctly. If entity boosting is retained, lowercase must also be applied. If the normalizer is removed, entity boosting should also be disabled.

---

## System Limitations

### Retrieval Corpus Coverage
The PubMed corpus (~40M documents) is large but not complete. Preprint papers (bioRxiv, medRxiv) and non-English literature are not included. For very recent questions (2024–2025), the corpus may lack relevant publications depending on the indexing cutoff date.

### MedCPT Cross-Encoder Throughput
Reranking 100 documents per query with a cross-encoder is the pipeline's main computational bottleneck at query time. On a CPU-only server, this step can take 10–30 seconds per question. GPU acceleration reduces this significantly, but the architecture is not suitable for low-latency serving at high concurrency.

### MMR Re-Encoding Mismatch
MMR computes document-document similarity using the `MedCPT-Query-Encoder` applied to document abstracts. This is architecturally mismatched: the query encoder is not designed for document-document similarity. The correct approach would be to use the article encoder for document embeddings or use cross-encoder scores directly for diversity computation. This mismatch may reduce the quality of the diversity selection.

### Snippet Generation
Phase A also evaluates snippet retrieval (positions 11–15 in the evaluator output). The current system does not perform explicit passage-level retrieval — it treats the abstract as the unit. Snippet scores are derived from abstract substrings that match the query, which is a weaker approach than true passage retrieval. A dedicated passage retrieval stage would likely improve snippet metrics substantially.

### Answer Format Compliance
The LLM answer format is controlled through the system prompt. Despite the structured instructions, the LLM occasionally produces formatting violations (e.g., factoid answers with explanatory text on line 1, or list answers without numbering). These formatting errors are penalized by the BioASQ evaluator's strict and lenient answer parsing. A post-processing normalization step would make results more robust.

### Fixed Reranker Top-K
The reranker `top_k=50` and MMR `top_k=10` are fixed in the config rather than adapted per question type or question difficulty. Summary questions might benefit from more documents (broader context), while factoid questions might benefit from fewer but more precisely matched documents. Per-type top-k tuning was not explored.

---

## Future Work

### 1. True Passage-Level Retrieval
Replace abstract-level retrieval with passage-level retrieval (e.g., splitting abstracts into sentences or 3-sentence windows). This would improve Phase A snippet scores and give the LLM more focused context for answer generation.

### 2. Cross-Encoder-Based MMR
Replace the MedCPT query encoder similarity in MMR with cross-encoder scores or dedicated document-document similarity models. Using the article encoder (`ncbi/MedCPT-Article-Encoder`) for document embeddings in the diversity term would be a minimal improvement.

### 3. Adaptive Alpha Per Query Type
The dense-sparse tradeoff appears to differ by question type. Factoid questions may benefit from higher alpha (semantic retrieval), while list questions may benefit from lower alpha (exact-match recall). A query-type-aware alpha scheduler could improve both Phase A and Phase B simultaneously.

### 4. Question-Type-Aware Final Top-K
Tune `top_k` for MMR selection per question type rather than using a global top-10. Summary questions likely benefit from 15–20 documents; factoid questions may be better served with 5–7 highly precise documents.

### 5. Iterative Retrieval with Feedback
BioASQ Synergy provides feedback from previous rounds (documents judged relevant/irrelevant by experts). Incorporating this feedback into retrieval through pseudo-relevance feedback, re-weighting, or negative example filtering could substantially improve recall for questions that appeared in earlier rounds.

### 6. Answer Post-Processing
Add a deterministic post-processing layer that normalizes LLM output format (strip introductory phrases, ensure correct numbering for list answers, enforce single-entity extraction for factoid answers). This would reduce evaluator penalty for formatting violations without requiring LLM changes.

### 7. Larger Context Window
The current context assembly is limited to `max_context_tokens=3000`. Modern LLMs support 128K+ context. Increasing context length would allow more documents to be included, particularly for summary questions where comprehensive coverage is needed.

### 8. FAISS Index Type
The current index uses `IndexFlatIP` (exact nearest neighbor search). For a 40M-vector corpus, approximate nearest neighbor indices (e.g., `IndexIVFPQ`) would reduce query latency significantly (from ~30s to <1s) at a small accuracy cost. This is essential for any production or interactive deployment.

### 9. End-to-End Fine-Tuning
The entire pipeline uses pretrained models (MedCPT encoders, cross-encoder, GPT-5). Fine-tuning the retrieval or reranking models on BioASQ-specific training data (available from prior BioASQ challenges) would likely improve both Phase A and Phase B metrics.

### 10. Query Normalization Strategy
The normalizer ablations show that removing normalization entirely improves Phase A MAP but hurts Phase B factoid performance. A better approach would be to pass the raw query to BM25 and the MedCPT encoder (preserving case and punctuation for retrieval) while applying normalization only to the NER and entity boosting pipeline. This would decouple the retrieval and entity-boosting preprocessing paths and potentially achieve the Phase A gains of no_normalizer without sacrificing Phase B factoid quality.

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
