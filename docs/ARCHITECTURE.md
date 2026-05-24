# System Architecture

This document describes the architecture of the BioASQ Synergy 14 RAG system — component design, data flow, configuration, and the key engineering decisions made during development.

---

## Overview

The system is a hybrid Retrieval-Augmented Generation (RAG) pipeline for biomedical question answering, built for the BioASQ Synergy 14 challenge (CLEF 2026). Given a natural-language biomedical question and a large PubMed corpus, it retrieves the most relevant documents, reranks them, selects a diverse final set, and synthesizes an answer via a large language model.

The pipeline has four major stages:

```
Query
  └─> [NER + Query Normalization]
        └─> [Hybrid Retrieval: MedCPT FAISS + Elasticsearch BM25]
              └─> [Cross-Encoder Reranking]
                    └─> [MMR + Recency Selection]
                          └─> [LLM Answer Generation]
                                └─> BioASQ Submission JSON
```

---

## Component Map

```
medicalrag_synergy14/
├── src/
│   ├── core/           — normalizer, MMR, recency, utilities
│   ├── ner/            — biomedical NER service
│   ├── encoder/        — MedCPT query and article encoders
│   ├── retrieval/      — FAISS index, BM25 retriever, hybrid score fusion
│   ├── reranker/       — MedCPT cross-encoder reranker
│   ├── llm/            — OpenAI-compatible and stub LLM clients
│   ├── pipeline/       — end-to-end orchestration
│   ├── evaluation/     — retrieval and answer quality metrics
│   └── api/            — FastAPI service (interactive use)
├── scripts/
│   ├── encode_documents.py     — corpus encoding (GPU-accelerated)
│   ├── build_faiss_index.py    — FAISS index construction
│   ├── ingest_elastic.py       — Elasticsearch ingestion
│   ├── run_hybrid_pipeline.py  — main competition runner
│   ├── evaluate_bioasq.py      — BioASQ metric evaluation
│   ├── validate_submission.py  — submission format check
│   └── diagnose_retrieval.py   — retrieval debugging utility
├── configs/            — YAML experiment configurations
├── test_data/          — BioASQ testsets and golden files
└── results/            — submission JSON and evaluation outputs
```

---

## Stage 1 — Query Understanding

**File:** `src/ner/ner_service.py`, `src/core/normalizer.py`

The raw question is first text-normalized (lowercasing, whitespace cleaning, medical abbreviation expansion) and then passed through a biomedical NER model (`d4data/biomedical-ner-all` via Hugging Face) to extract named entities such as drugs, diseases, genes, and organisms.

The extracted entities serve two purposes:
- **Query augmentation for FAISS**: the entity surface forms are appended to the normalized query before encoding, improving dense retrieval signal for rare biomedical terms.
- **BM25 entity boost**: entity terms are promoted with a configurable weight (`bm25_entity_boost`) to increase sparse recall for exact term matches.

---

## Stage 2 — Hybrid Retrieval

**Files:** `src/retrieval/faiss_index.py`, `src/retrieval/bm25_retriever.py`, `src/retrieval/hybrid_medcpt_retriever.py`

The system uses an asymmetric biencoder retrieval setup:

### Dense Retrieval (FAISS)
- **Query encoder:** `ncbi/MedCPT-Query-Encoder` — encodes natural-language questions.
- **Article encoder:** `ncbi/MedCPT-Article-Encoder` — encodes `[CLS] title [SEP] abstract [SEP]` pairs during corpus indexing.
- **Index type:** `IndexFlatIP` (inner-product similarity over L2-normalized vectors = cosine similarity).
- **Corpus:** ~40 million PubMed documents encoded as float16 (≈61 GB on disk); cast to float32 during FAISS construction (≈122 GB index).
- **Retrieval:** top-200 nearest neighbors per query.

### Sparse Retrieval (BM25 / Elasticsearch)
- **Backend:** Elasticsearch with BM25 (k1=1.5, b=0.75).
- **Index:** `medical_docs_3` — titles and abstracts tokenized by Elasticsearch's default analyzer.
- **Entity boost:** named entities extracted in Stage 1 can be given extra weight via term-level boosting.
- **Retrieval:** top-200 results per query.

### Score Fusion
The hybrid retriever (`src/retrieval/hybrid_medcpt_retriever.py`) merges dense and sparse result sets:

1. Dense scores are **min-max normalized** across the FAISS result set.
2. Sparse scores are **max normalized** (divided by the highest BM25 score in the result set).
3. Combined score: `alpha * dense_normalized + (1 - alpha) * sparse_normalized`
4. Documents appearing in only one result set receive `0.0` for the missing signal.

The `alpha` hyperparameter controls the dense-sparse balance. This is the single most impactful configuration choice (see [RESULTS.md](RESULTS.md)).

---

## Stage 3 — Cross-Encoder Reranking

**File:** `src/reranker/cross_encoder.py`

After hybrid retrieval produces the top-100 candidates, a cross-encoder reranker rescores all pairs `(query, document)` jointly:

- **Model:** `ncbi/MedCPT-Cross-Encoder` — a PubMed-trained cross-encoder.
- **Input:** normalized query + document title and abstract.
- **Output:** a relevance score replacing the hybrid fusion score.
- **Selection:** top-50 documents by cross-encoder score.

The cross-encoder is significantly stronger than the biencoder at fine-grained relevance discrimination, making it the dominant signal for final document ordering. As a consequence, differences in `alpha` (dense-sparse balance) have a smaller effect on final ranking than on initial recall — what matters most is whether the relevant document appears in the top-100 recall set.

---

## Stage 4 — MMR and Recency Selection

**File:** `src/core/mmr.py`

From the reranker's top-50, a final 10 documents are selected using Maximal Marginal Relevance (MMR) with an optional recency boost.

### MMR
MMR selects documents that are both relevant to the query **and** diverse from already-selected documents:

```
score(d) = lambda * cosine_sim(query, d) - (1 - lambda) * max_cosine_sim(d, selected)
```

- `lambda_param = 0.95` (heavily weighted toward relevance, with mild diversity control).
- Embeddings are re-computed using `MedCPT-Query-Encoder` on the document abstracts. This is a weaker signal than the cross-encoder, but provides the similarity geometry needed for the diversity term.

### Recency Boost
If `recency_boost` is enabled:

```
relevance(d) = (1 - recency_weight) * cosine_sim + recency_weight * recency_score
```

- `recency_score` uses exponential decay from the most recent publication date in the candidate set: `exp(-decay_rate * days_from_most_recent)`.
- `recency_weight = 0.3` by default. This penalizes older papers softly, improving performance on factoid questions that ask about recent trials or drug approvals.

---

## Stage 5 — LLM Answer Generation

**File:** `src/llm/openai_client.py`, `src/pipeline/med_rag_hybrid_medcpt.py`

The final 10 documents are passed as context to an LLM with a structured system prompt. The prompt instructs the model on BioASQ answer format:
- **yes/no questions:** single `yes`/`no` on line 1 followed by a brief explanation.
- **factoid questions:** the entity answer on line 1 (no filler text), then evidence.
- **list questions:** numbered list of items, then a summary sentence.
- **summary questions:** a single paragraph, max 200 words.

The LLM backend is OpenAI-compatible (configured to the CMU AI Gateway; model `gpt-5`). A stub LLM is available for offline testing by setting `llm.provider: stub` or `LLM_PROVIDER=stub`.

---

## Corpus Indexing Pipeline

The corpus is indexed in two offline steps before any queries are processed.

### Step 1 — Encode Documents

```bash
python scripts/encode_documents.py \
  --config configs/fullpipeline.yaml \
  --output-dir <output_dir> \
  --batch-size 256
```

- Reads `data.docs_path` (a JSONL file of PubMed documents with `doc_id`, `title`, `abstract`, `pub_date`).
- Passes `[title, abstract]` pairs to `MedCPT-Article-Encoder`, matching the model's `[CLS] title [SEP] abstract [SEP]` training format.
- Writes `embeddings.npy` (float16, ~61 GB for 40M docs), `doc_ids.json`, and `embeddings_manifest.json`.
- Uses `wc -l` for fast JSONL line counting; falls back to Python iteration if unavailable.
- Writes directly to the output directory (no intermediate tmp staging).

### Step 2 — Build FAISS Index

```bash
python scripts/build_faiss_index.py \
  --config configs/fullpipeline.yaml \
  --gpu
```

- Reads `embeddings.npy` in chunks (100K vectors at a time) to stay within RAM budget.
- Casts float16 chunks to float32 before adding to FAISS (index remains full-precision).
- L2-normalizes each chunk before adding to `IndexFlatIP` (inner product = cosine similarity).
- Optionally builds on GPU for faster `add()`, then converts to CPU index for portable save.
- Saves `faiss.index` (~122 GB for 40M × 768-dim float32 vectors).

### Step 3 — Ingest Elasticsearch

```bash
python scripts/ingest_elastic.py --config configs/fullpipeline.yaml
```

- Bulk-indexes documents into Elasticsearch for BM25 retrieval.

---

## Configuration System

All pipeline behavior is controlled through YAML files in `configs/`. The key settings are:

| Section | Key field | Description |
|---|---|---|
| `retrieval` | `alpha` | Dense-sparse weight (0.0 = pure BM25, 1.0 = pure FAISS) |
| `retrieval` | `top_k_dense`, `top_k_sparse` | Initial recall depth per modality |
| `retrieval` | `top_k_final` | Size of merged set passed to reranker |
| `reranker` | `model` | Cross-encoder model name |
| `reranker` | `top_k` | Number of documents after reranking |
| `mmr` | `enabled` | Enable/disable MMR (if false, takes cross-encoder top-10 directly) |
| `mmr` | `lambda_param` | Relevance vs. diversity tradeoff |
| `mmr` | `recency_weight` | Weight for publication-date recency boost |
| `encoder` | `model` | Query encoder |
| `encoder` | `article_model` | Article encoder (must match encoding step) |
| `llm` | `provider` | `openai`, `gemini`, or `stub` |
| `llm` | `model` | LLM model name |

### Experiment Configurations

| File | `alpha` | MMR | Recency weight | Purpose |
|---|---|---|---|---|
| `configs/fullpipeline.yaml` | 0.65 | enabled | 0.3 | Primary submission |
| `configs/bm25only.yaml` | 0.0 | enabled | 0.3 | BM25-only ablation |
| `configs/alpha03.yaml` | 0.3 | enabled | 0.3 | Alpha ablation |
| `configs/no_mmr.yaml` | 0.65 | disabled | — | MMR ablation |
| `configs/no_recency.yaml` | 0.65 | enabled | 0.0 | Recency ablation |

---

## Main Runner Script

```bash
python scripts/run_hybrid_pipeline.py --config configs/fullpipeline.yaml
```

This script:
1. Loads the BioASQ Synergy 14 testset from `test_data/round_3/`.
2. Fetches or loads cached PubMed document metadata.
3. Initializes the hybrid pipeline from config.
4. Runs `process_query()` for each question.
5. Formats Phase A (documents + snippets) and Phase B (answers) output.
6. Writes `results/submission.json`.

---

## Evaluation

### BioASQ Java Evaluator

The official BioASQ evaluator is a Java JAR:

```bash
# Phase A (document + snippet retrieval):
java -Xmx10G -cp ./flat/BioASQEvaluation/dist/BioASQEvaluation.jar \
     evaluation.EvaluatorTask1b -phaseA -e 5 \
     test_data/round_3/golden_round3_testset_phaseA.json \
     results/submission.json

# Phase B (answer generation):
java -Xmx10G -cp ./flat/BioASQEvaluation/dist/BioASQEvaluation.jar \
     evaluation.EvaluatorTask1b -phaseB -e 5 \
     test_data/round_3/golden_round3_testset_phaseB.json \
     results/submission.json
```

**Phase A output** (20 space-separated values): positions 1–5 concepts (deprecated), 6=Doc MPrec, 7=Doc MRec, 8=Doc MF1, **9=Doc MAP** (the primary Phase A metric), 10=Doc GMAP, 11–15 snippets, 16–20 triples (deprecated).

**Phase B output** (10 values): YesNo Acc, Factoid Strict Acc, Factoid Lenient Acc, Factoid MRR, List Prec, List Rec, List F1, YesNo macroF1, YesNo F1-yes, YesNo F1-no.

### Golden Files

| File | Questions | Use |
|---|---|---|
| `test_data/round_3/golden_round3_testset.json` | 117 | Original (has null exact_answers, do not use directly) |
| `test_data/round_3/golden_round3_testset_phaseA.json` | 117 | `-phaseA` evaluation (11 unanswerable questions have dummy answers to prevent Java crash) |
| `test_data/round_3/golden_round3_testset_phaseB.json` | 106 | `-phaseB` evaluation (11 unanswerable questions excluded entirely) |

### Python Evaluation

```bash
python scripts/evaluate_bioasq.py --predictions results/submission.json \
                                   --golden test_data/round_3/golden_round3_testset_phaseB.json
```

```bash
bash scripts/evaluate_and_resubmit.sh results/submission.json
```

---

## Source Module Reference

| Module | Role |
|---|---|
| `src/core/normalizer.py` | Query text normalization |
| `src/core/mmr.py` | MMR selection and recency scoring |
| `src/core/utils.py` | Random seed, general utilities |
| `src/ner/ner_service.py` | Biomedical named entity recognition |
| `src/encoder/medcpt_encoder.py` | MedCPT query and article encoding |
| `src/retrieval/faiss_index.py` | FAISS vector index wrapper |
| `src/retrieval/bm25_retriever.py` | Elasticsearch BM25 wrapper |
| `src/retrieval/hybrid_medcpt_retriever.py` | Dense + sparse score fusion |
| `src/reranker/cross_encoder.py` | MedCPT cross-encoder reranker |
| `src/llm/openai_client.py` | OpenAI-compatible LLM client |
| `src/llm/stub_llm.py` | Offline stub LLM for testing |
| `src/pipeline/med_rag_hybrid_medcpt.py` | Main end-to-end pipeline orchestration |
| `src/api/app.py` | FastAPI interactive query service |
