# MedicalRAG — BioASQ Synergy 14

Hybrid Retrieval-Augmented Generation system for BioASQ Synergy 14 (CLEF 2026), built by the CMU Language Technology Institute team. The system combines MedCPT dense retrieval, Elasticsearch BM25 sparse retrieval, cross-encoder reranking, and MMR-based diversity selection to answer biomedical questions over a ~40M-document PubMed corpus.

---

## Quick Links

- [Architecture](docs/ARCHITECTURE.md) — component design, data flow, config reference, indexing pipeline
- [Experiment Results](docs/RESULTS.md) — Phase A/B numbers, findings, tradeoffs, limitations, future work

---

## System at a Glance

```
Query
  └─> NER + Query Normalization
        └─> Hybrid Retrieval (MedCPT FAISS + Elasticsearch BM25)
              └─> MedCPT Cross-Encoder Reranking (top-50)
                    └─> MMR + Recency Selection (top-10)
                          └─> LLM Answer Generation (GPT-5)
                                └─> BioASQ Submission JSON
```

**Best results across 7 experiments (metrics are not from the same config — see [RESULTS.md](docs/RESULTS.md)):**

| Metric | Best value | Config |
|---|---|---|
| Phase A Doc MAP | **0.1066** | no_normalizer |
| YesNo Accuracy | **1.0000** | no_normalizer |
| Factoid MRR | **0.3809** | fullpipeline |
| List F1 | **0.3275** | fullpipeline |

**Primary submission config (alpha=0.65, MMR, recency_weight=0.3, normalizer on):** Phase A MAP 0.0894 · YesNo 0.9545 · Factoid MRR 0.3809 · List F1 0.3275

---

## Repository Layout

```
medicalrag_synergy14/
├── src/                        — all Python source code
│   ├── core/                   — normalizer, MMR, utilities
│   ├── ner/                    — biomedical NER (d4data/biomedical-ner-all)
│   ├── encoder/                — MedCPT encoder wrappers
│   ├── retrieval/              — FAISS, BM25, hybrid score fusion
│   ├── reranker/               — MedCPT cross-encoder reranker
│   ├── llm/                    — OpenAI-compatible + stub LLM clients
│   ├── pipeline/               — end-to-end pipeline orchestration
│   ├── evaluation/             — metric computation
│   └── api/                    — FastAPI query service
├── scripts/
│   ├── encode_documents.py     — batch corpus encoding (GPU)
│   ├── build_faiss_index.py    — FAISS index construction
│   ├── ingest_elastic.py       — Elasticsearch ingestion
│   ├── run_hybrid_pipeline.py  — main competition runner
│   ├── evaluate_bioasq.py      — BioASQ metric evaluation
│   ├── evaluate_and_resubmit.sh — submission sanity check + eval commands
│   ├── validate_submission.py  — submission format check
│   └── diagnose_retrieval.py   — retrieval debugging
├── configs/
│   ├── fullpipeline.yaml       — primary submission config
│   ├── bm25only.yaml           — BM25-only ablation (alpha=0.0)
│   ├── alpha03.yaml            — alpha=0.3 ablation
│   ├── no_mmr.yaml             — MMR-disabled ablation
│   ├── no_recency.yaml         — recency-disabled ablation
│   ├── no_normalizer.yaml      — query normalizer fully disabled
│   └── no_lowercase.yaml       — normalizer on, lowercase step disabled
├── test_data/round_3/
│   ├── golden_round3_testset_phaseA.json  — 117 questions (Phase A eval)
│   └── golden_round3_testset_phaseB.json  — 106 questions (Phase B eval)
├── results/
│   ├── submission.json              — primary submission
│   ├── submission_alpha03.json
│   ├── submission_no_mmr.json
│   ├── submission_no_recency.json
│   ├── submission_no_normalizer.json
│   └── submission_no_lowercase.json
├── docs/
│   ├── ARCHITECTURE.md         — full system documentation
│   └── RESULTS.md              — all experiment results and analysis
└── tests/                      — unit tests
```

---

## Running the Pipeline

### Prerequisites

- Python 3.10+, FAISS, Elasticsearch 8.x
- Pre-built FAISS index and Elasticsearch index (see [Architecture: Corpus Indexing](docs/ARCHITECTURE.md#corpus-indexing-pipeline))
- OpenAI-compatible API key (or set `llm.provider: stub` for offline testing)

### Run Primary Pipeline

```bash
python scripts/run_hybrid_pipeline.py --config configs/fullpipeline.yaml
```

### Run an Ablation

```bash
python scripts/run_hybrid_pipeline.py --config configs/bm25only.yaml
```

### Evaluate Results

```bash
# Submission sanity check + print evaluation commands:
bash scripts/evaluate_and_resubmit.sh results/submission.json

# Phase A (requires BioASQ Java evaluator):
java -Xmx10G -cp ./flat/BioASQEvaluation/dist/BioASQEvaluation.jar \
     evaluation.EvaluatorTask1b -phaseA -e 5 \
     test_data/round_3/golden_round3_testset_phaseA.json \
     results/submission.json

# Phase B:
java -Xmx10G -cp ./flat/BioASQEvaluation/dist/BioASQEvaluation.jar \
     evaluation.EvaluatorTask1b -phaseB -e 5 \
     test_data/round_3/golden_round3_testset_phaseB.json \
     results/submission.json
```

---

## Key Configuration Parameters

| Parameter | Default | Effect |
|---|---|---|
| `retrieval.alpha` | 0.65 | Dense-sparse weight (0.0 = BM25 only) |
| `mmr.enabled` | true | Use MMR for final doc selection |
| `mmr.lambda_param` | 0.95 | Relevance vs. diversity balance |
| `mmr.recency_weight` | 0.3 | Publication-date recency boost |
| `reranker.top_k` | 50 | Docs after cross-encoder reranking |
| `query_normalization.enabled` | true | Apply query normalizer (false = raw query) |
| `query_normalization.lowercase` | true | Lowercase query before retrieval |
| `query_normalization.remove_punctuation` | true | Strip punctuation from query |
| `llm.provider` | openai | `openai`, `gemini`, or `stub` |

See [Architecture](docs/ARCHITECTURE.md) for the full configuration reference.

---

## Authors

- **Professor Eric Nyberg** — Language Technology Institute, Carnegie Mellon University
- **John Dalton Gibson** — MSECE, Carnegie Mellon University
