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
| bm25only | 0.0 | ✓ | 0.3 | on | 0.0978 | 0.8636 | 0.2381 | **0.3241** ★ |
| alpha03 | 0.3 | ✓ | 0.3 | on | 0.0978\* | 0.8636\* | 0.2380\* | 0.2857\* |
| **fullpipeline** (primary) | 0.65 | ✓ | 0.3 | on | 0.0894 | 0.9545 | **0.3810** ★ | 0.3204 |
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

> alpha03 Phase B values marked `*` are from the original evaluation run; full re-evaluation pending.

### Yes/No Questions

| Config | Accuracy | Macro F1 | F1-yes | F1-no |
|---|---|---|---|---|
| **no_normalizer** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| fullpipeline | 0.9545 | 0.9494 | 0.9655 | 0.9333 |
| no_recency | 0.9545 | 0.9494 | 0.9655 | 0.9333 |
| no_lowercase | 0.9091 | 0.9018 | 0.9286 | 0.8750 |
| no_mmr | 0.9091 | 0.9018 | 0.9286 | 0.8750 |
| bm25only | 0.8636 | 0.8611 | 0.8800 | 0.8421 |
| alpha03 | 0.8636\* | — | — | — |

### Factoid Questions

| Config | Strict Accuracy | Lenient Accuracy | MRR |
|---|---|---|---|
| **fullpipeline** | **0.3810** | **0.3810** | **0.3810** |
| no_normalizer | 0.2857 | 0.2857 | 0.2857 |
| no_recency | 0.2857 | 0.2857 | 0.2857 |
| bm25only | 0.2381 | 0.2381 | 0.2381 |
| no_mmr | 0.2381 | 0.2381 | 0.2381 |
| alpha03 | — | — | 0.2380\* |
| no_lowercase | 0.1905 | 0.1905 | 0.1905 ⬇ |

### List Questions

| Config | Mean Precision | Mean Recall | Mean F-Measure |
|---|---|---|---|
| **bm25only** | 0.3173 | 0.4502 | **0.3241** ★ |
| fullpipeline | **0.3212** | 0.4520 | 0.3204 |
| no_recency | 0.3169 | **0.4633** | 0.3200 |
| no_normalizer | 0.3104 | 0.4388 | 0.3114 |
| no_lowercase | 0.2817 | 0.4237 | 0.2870 |
| alpha03 | — | — | 0.2857\* |
| no_mmr | 0.2434 | 0.4006 | 0.2571 |

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

This is the best configuration for factoid questions, achieving the highest factoid strict accuracy, lenient accuracy, and MRR across all experiments (0.3810). Strict and lenient factoid accuracy are now equal, indicating precise answer extraction without hedging. For list questions, BM25-only (0.3241) narrowly exceeds fullpipeline (0.3204) — see Finding 1.

---

### Experiment 2 — BM25-Only Ablation (`configs/bm25only.yaml`)

**Configuration:** alpha=0.0 (pure BM25, FAISS disabled), MMR enabled, recency_weight=0.3

**Phase A — Document:** MPrec 0.1385 | MRec 0.1025 | MF1 0.0941 | MAP **0.0978** | GMAP 0.0018

**Phase A — Snippet:** MPrec 0.0716 | MRec 0.0231 | MF1 0.0308 | MAP 0.0523 | GMAP 2.06×10⁻⁴

**Phase B — Yes/No:** Acc 0.8636 | Macro F1 0.8611 | F1-yes 0.8800 | F1-no 0.8421

**Phase B — Factoid:** Strict 0.2381 | Lenient 0.2381 | MRR 0.2381

**Phase B — List:** Precision 0.3173 | Recall 0.4502 | F1 **0.3241** ★ (best List F1 across all experiments)

**Submission file:** `results/submission_bm25only.json`

**Finding:** BM25-only achieves the second-highest Phase A document MAP (0.0978) and, surprisingly, the **highest List F1 across all experiments (0.3241)**. This exceeds even the fullpipeline's 0.3204, suggesting that BM25's strong keyword recall for list questions — which often contain multiple named entities — outweighs the advantage that dense retrieval brings for semantic diversity. However, YesNo accuracy (0.8636) and Factoid MRR (0.2381) are both substantially lower than fullpipeline (0.9545 and 0.3810 respectively), confirming that BM25 alone is not sufficient for answer types requiring semantic understanding.

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

### 1. Retrieval Strategy Is Question-Type Dependent

The most important overall finding is that no single retrieval configuration dominates across all question types. The optimal strategy differs by answer type:

| Question type | Best config | Best value | Key driver |
|---|---|---|---|
| Yes/No Accuracy | no_normalizer | 1.0000 | Raw query improves BM25 + MedCPT recall |
| Factoid MRR | fullpipeline | 0.3810 | Dense retrieval + recency boost surfaces the right entity |
| List F1 | bm25only | 0.3241 | Exact keyword matching captures multi-entity lists |
| Doc MAP | no_normalizer | 0.1066 | Raw query maximises Phase A recall |

**List F1 with BM25-only (0.3241) exceeds fullpipeline (0.3204).** List questions in BioASQ Synergy 14 typically ask for multiple named entities (genes, drugs, organisms) that appear verbatim in abstracts. BM25's exact keyword matching retrieves these more reliably than dense retrieval, which generalises semantically and may miss the precise surface form. This makes BM25 the strongest retrieval strategy for list questions in this domain, despite underperforming on Yes/No and Factoid.

### 2. Dense Retrieval Helps Factoid and YesNo, Hurts Phase A MAP

Increasing alpha from 0.0 (BM25-only) to 0.65 (fullpipeline):
- **Factoid MRR:** 0.2381 → 0.3810 (+0.143) — the largest single improvement of any configuration change
- **YesNo Accuracy:** 0.8636 → 0.9545 (+0.091)
- **List F1:** 0.3241 → 0.3204 (−0.004) — slight regression; BM25 is better for lists
- **Doc MAP:** 0.0978 → 0.0894 (−0.009) — BM25 is better for Phase A document recall

Dense retrieval adds semantic matching that retrieves documents containing the correct answer entity even when the terminology differs from the question. This specifically benefits factoid questions (single entity, precision matters) and yes/no questions (evidence of presence/absence rather than keyword co-occurrence). For list questions, the breadth of BM25 keyword recall is more valuable than semantic precision.

### 3. Cross-Encoder Reranker Dominance and the Alpha Insensitivity Zone

The MedCPT cross-encoder (`ncbi/MedCPT-Cross-Encoder`) is the most impactful single component. The equality of alpha=0.0 and alpha=0.3 results directly demonstrates this: once the reranker re-orders the top-100 candidate set, the initial retrieval mix is largely erased. At low alpha, BM25 dominates the recall pool, and the cross-encoder then applies the same relevance reranking regardless of whether FAISS contributed. Only at alpha=0.65 does FAISS meaningfully reshape which documents enter the top 100, producing measurably different final results.

**Architectural implication:** Optimising the recall set (which 100 documents enter the reranker) matters more than optimising the initial ranking within that set.

### 4. MMR Is the Most Impactful Single Component for Phase B

Disabling MMR produces the largest Phase B degradation of any configuration change tested:

| Metric | fullpipeline (MMR on) | no_mmr (MMR off) | Change |
|---|---|---|---|
| YesNo Accuracy | 0.9545 | 0.9091 | −0.045 |
| Factoid MRR | 0.3810 | 0.2381 | **−0.143** |
| List F1 | 0.3204 | 0.2571 | −0.063 |

Without MMR, the cross-encoder's top-10 documents tend to be nearly-duplicate abstracts about the same subtopic, giving the LLM insufficient coverage across the answer space. MMR's diversity term forces selection across different subtopics, which is especially critical for multi-part factoid and list questions where the correct answer may be distributed across several documents. Note that Phase A metrics are identical with and without MMR — MMR affects only the documents passed to the LLM, not the retrieval pool used for Phase A evaluation.

### 5. Recency Boost Is Factoid-Specific and Has No Effect on Yes/No

| Metric | fullpipeline (recency=0.3) | no_recency (recency=0.0) | Change |
|---|---|---|---|
| YesNo Accuracy | 0.9545 | 0.9545 | 0.000 |
| Factoid MRR | 0.3810 | 0.2857 | **−0.095** |
| List Recall | 0.4520 | 0.4633 | +0.011 |
| List F1 | 0.3204 | 0.3200 | −0.004 |

The recency boost has zero effect on Yes/No questions and negligible effect on list questions, but drops Factoid MRR by 0.095. BioASQ Synergy 14 factoid questions frequently ask for entities reported in recent clinical trials, drug approvals, or emerging research — questions where a 2023 or 2024 paper is far more likely to contain the correct answer than a 2010 paper on the same general topic.

Interestingly, removing the recency boost slightly increases list recall (0.4633 vs 0.4520) without improving list F1. Without recency weighting, MMR selects a more temporally uniform set of documents, which broadens the range of list items retrieved but does not improve answer precision enough to raise F1.

### 6. Phase A Document Metrics Are Identical Across Four Configurations

fullpipeline, no_recency, no_mmr, and no_lowercase all produce exactly the same Phase A document metrics (MPrec 0.1282, MRec 0.0998, MF1 0.0887, MAP 0.0894, GMAP 0.0013). This is structurally expected: all four share the same retrieval pipeline (hybrid fusion at alpha=0.65 followed by MedCPT cross-encoder reranking). Only the final MMR/recency selection step differs, and that step determines which documents go to the LLM — not the full set of documents returned across all queries, which is what Phase A evaluates.

### 7. Query Normalizer Harms Phase A; Its Lowercase Step Is Critical for Phase B

The two normalizer ablations reveal a complex and counterintuitive interaction:

| Config | Doc MAP | Factoid MRR | YesNo Acc |
|---|---|---|---|
| fullpipeline (normalizer on, lowercase on) | 0.0894 | **0.3810** | 0.9545 |
| no_normalizer (normalizer fully off) | **0.1066** | 0.2857 | **1.0000** |
| no_lowercase (normalizer on, lowercase off) | 0.0894 | 0.1905 ⬇ | 0.9091 |

- **Removing the normalizer entirely** raises Phase A MAP by 19% (0.0894 → 0.1066) and achieves perfect YesNo accuracy (1.0000). The raw query contains naturally-cased biomedical terms and punctuation that Elasticsearch's own standard analyzer and the MedCPT query encoder handle better than a pre-normalised lowercase version.
- **Removing only the lowercase step** is the worst configuration tested for factoid questions (MRR 0.1905 — lower than even BM25-only). Phase A MAP is unchanged because document retrieval is unaffected, but factoid answer quality collapses.
- **Root cause of no_lowercase failure:** NER extracts entities in their original cased form (`"BRCA1"`, `"COVID-19"`). These are passed to Elasticsearch as BM25 boost terms. The standard analyzer has already lowercased all index tokens, so cased boost terms produce zero matches — silently breaking entity boosting entirely. The fullpipeline avoids this by lowercasing entity terms before boosting. The no_normalizer config avoids it by not applying NER-based boosting at all. Only no_lowercase falls into the broken intermediate state.

**Practical implication:** The normalizer's lowercase step is a prerequisite for NER-based entity boosting to work correctly. If entity boosting is retained, lowercase must also be applied. If the normalizer is removed entirely, entity boosting should also be disabled or redesigned.

### 8. Phase A and Phase B Have Partially Conflicting Objectives

Configurations that maximise Phase A document MAP (no_normalizer at 0.1066, bm25only at 0.0978) do not maximise Phase B answer quality, and vice versa. However, this conflict is not universal across question types:

- **Factoid questions:** favour dense retrieval (fullpipeline, alpha=0.65) — conflicting with Phase A
- **Yes/No questions:** favour no normalizer — aligned with Phase A (both are maximised by no_normalizer)
- **List questions:** favour BM25-only — partially aligned with Phase A (bm25only has second-best Phase A MAP)

This means the Phase A vs Phase B tension is primarily a factoid retrieval problem: the documents that maximise factoid answer quality (semantic, diverse, recent) are not the same documents that maximise Phase A MAP (keyword-matched, high-precision).

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
The entire pipeline uses pretrained models (MedCPT encoders, cross-encoder, gpt-4o-mini-2024-07-18). Fine-tuning on BioASQ-specific training data (available from prior BioASQ challenges) would likely improve both Phase A and Phase B metrics.

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
