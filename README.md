# MedicalRAG — BioASQ Synergy 14

Hybrid Retrieval-Augmented Generation system for BioASQ Synergy 14 (CLEF 2026), built by the CMU Language Technology Institute team. The system combines MedCPT dense retrieval, Elasticsearch BM25 sparse retrieval, cross-encoder reranking, and MMR-based diversity selection to answer biomedical questions over a ~40M-document PubMed corpus.

---

## Quick Links

- [Architecture](docs/ARCHITECTURE.md) — component design, data flow, config reference, indexing pipeline
- [Experiment Results](docs/RESULTS.md) — Phase A/B numbers, findings, tradeoffs, limitations, future work
- [Reproducibility Paths](#reproducibility-paths) — download pre-built corpus, embeddings, or FAISS index from Hugging Face instead of building from scratch

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

## Environment Setup

### Requirements

- Python 3.10 or later
- Elasticsearch 8.x running locally (default: `localhost:9200`)
- A CUDA-capable GPU is strongly recommended for encoding and reranking; the pipeline falls back to CPU automatically but will be significantly slower

### Installation

```bash
# 1. Clone the repository
git clone git@github.com:jdaltonll02/medicalragsys.git
cd medicalragsys

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install the package in editable mode plus all dependencies
pip install -e .
pip install -r requirements.txt

# 4. (Optional) Install GPU FAISS if you have a CUDA GPU
#    Replace faiss-cpu with the GPU build:
pip uninstall faiss-cpu
pip install faiss-gpu
```

### Elasticsearch

Elasticsearch must be running before any pipeline execution that uses BM25.

```bash
# Using Docker (quickest):
docker run -d --name es \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  -p 9200:9200 \
  docker.elastic.co/elasticsearch/elasticsearch:8.11.0

# Verify it is up:
curl http://localhost:9200
```

The host, port, and index name are controlled by the `bm25` section of each config YAML. The defaults (`localhost:9200`, index `medical_docs_3`) match all configs in this repo.

### Corpus Data

The pipeline expects a PubMed corpus in JSONL format at the path set by `data.docs_path` in the config. Each line is one document:

```json
{"doc_id": "12345678", "title": "...", "abstract": "...", "pub_date": "2023-06-01"}
```

The BioASQ testsets used for evaluation are already included in `test_data/round_3/`.

Don't have the raw corpus, the MedCPT embeddings, or a built FAISS index yet? See [Reproducibility Paths](#reproducibility-paths) below — all three are published on Hugging Face so you don't have to re-encode 40M documents from scratch.

### LLM API Key

Set your OpenAI-compatible API key as an environment variable:

```bash
export OPENAI_API_KEY="sk-..."

# Optional — for a custom endpoint (e.g. CMU AI Gateway):
export OPENAI_BASE_URL="https://ai-gateway.andrew.cmu.edu"
export OPENAI_PROJECT_ID="your-project-id"
```

Alternatively, set `llm.api_key` directly in the config YAML. To run the pipeline without any LLM (retrieval only, no answer generation), set `llm.provider: stub` in the config or pass `LLM_PROVIDER=stub` in the environment.

---

## Reproducibility Paths

Corpus indexing (Steps 1–2 of [Reproducing All Experiments](#reproducing-all-experiments)) is the most expensive part of this system — encoding 40M PubMed documents takes GPU-hours, and the outputs (61.7GB of embeddings, 123.5GB FAISS index) are too large to check into git. So that reproducing results doesn't require everyone to re-encode the whole corpus, the raw corpus, the embeddings, and the built FAISS index are all published at:

**[huggingface.co/datasets/jdaltonII02/bioasq-synergy14-medcpt-index](https://huggingface.co/datasets/jdaltonII02/bioasq-synergy14-medcpt-index)** (~227GB total)

Pick the path that matches what you're trying to verify — each skips a different amount of the pipeline:

| Path | Download | Then run | Compute needed | Use when |
|---|---|---|---|---|
| **A — Full rebuild** | `corpus/pubmed_corpus.jsonl` (41.6GB) | [Step 1](#step-1--encode-the-corpus-faiss) (encode) → [Step 2](#step-2--build-the-faiss-index) (build index) | GPU recommended for encoding (hours for 40M docs); CPU or GPU for index build | You want to verify the whole pipeline end-to-end, or re-encode with a different model |
| **B — Reuse embeddings** | `embeddings.npy` + `doc_ids.json` + `embeddings_manifest.json` (~62.2GB) | [Step 2](#step-2--build-the-faiss-index) only | CPU-only, ~10–20 min | You trust the published MedCPT embeddings but want your own FAISS index (different `index_type`, GPU vs. CPU build, etc.) |
| **C — Reuse index** | `faiss.index` + `doc_ids.json` (~124GB) | Nothing — point `faiss.save_path` at it | None | You just want to run the pipeline or evaluation immediately |

### Downloading from Hugging Face

```bash
pip install -U huggingface_hub

# Point the HF cache at a disk with enough free space — this dataset is ~227GB
# total and the default ~/.cache/huggingface can silently fill a small $HOME.
export HF_HOME=/path/to/large/disk/hf_cache

# Path A — raw corpus only
hf download jdaltonII02/bioasq-synergy14-medcpt-index \
  corpus/pubmed_corpus.jsonl --repo-type dataset --local-dir ./data

# Path B — embeddings + doc_ids + manifest (skip corpus + faiss.index)
hf download jdaltonII02/bioasq-synergy14-medcpt-index \
  embeddings.npy doc_ids.json embeddings_manifest.json \
  --repo-type dataset --local-dir ./data

# Path C — prebuilt FAISS index + doc_ids (skip everything else)
hf download jdaltonII02/bioasq-synergy14-medcpt-index \
  faiss.index doc_ids.json --repo-type dataset --local-dir ./data
```

Then point your config at wherever you downloaded to:

```yaml
data:
  docs_path: ./data/pubmed_corpus.jsonl        # Path A only — input to Step 1
  embeddings_path: ./data/embeddings.npy       # Paths A (after Step 1) and B — input to Step 2
faiss:
  save_path: ./data/faiss.index                # Path C, or the output of Step 2 for A/B
```

`doc_ids.json` must live in the same directory as whichever of `embeddings.npy` or `faiss.index` you're using — both `build_faiss_index.py` and the retriever look for it at `<embeddings_dir>/doc_ids.json`.

### If you don't have a PubMed corpus at all

To build a corpus independent of the exact snapshot published above:

- **Small, targeted sets** (a few hundred to a few thousand specific PMIDs): use `src/core/pubmed_fetcher.py`, which wraps the NCBI Entrez `efetch` API. It's rate-limited to 3 requests/sec without an API key (NCBI requires an `email`; pass `api_key` for higher limits), so it isn't practical at corpus scale (millions of documents).
- **Full corpus rebuild**: NCBI publishes the complete PubMed baseline as gzipped XML at [ftp.ncbi.nlm.nih.gov/pubmed/baseline/](https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/) (plus daily `updatefiles/` deltas). This repo does not include an XML→JSONL converter for the baseline dump — you'd need to parse `PubmedArticle` records into the `{doc_id, title, abstract, pub_date}` schema shown in [Corpus Data](#corpus-data) yourself. For most purposes, downloading `corpus/pubmed_corpus.jsonl` from the HF dataset (Path A above) is far faster than rebuilding from the NCBI baseline.

---

## Quick Start

Once the environment is set up and the corpus indices are built (see [Environment Setup](#environment-setup) and [Reproducing All Experiments](#reproducing-all-experiments) Steps 1–3), a single run looks like:

```bash
python scripts/run_hybrid_pipeline.py \
  --config configs/fullpipeline.yaml \
  --testset test_data/round_3/golden_round3_testset_phaseA.json \
  --output results/submission.json
```

Swap `--config` for any file in `configs/` to run an ablation. See [Reproducing All Experiments](#reproducing-all-experiments) for the full list of commands and evaluation steps.

---

## Reproducing All Experiments

All seven experiments can be reproduced with the Python CLI. Steps 1–3 (corpus indexing) only need to be run once; steps 4 onwards are per-experiment runs that share the same pre-built indices.

> **Skip Step 1, 2, or both:** see [Reproducibility Paths](#reproducibility-paths) to download pre-computed embeddings or a pre-built FAISS index from Hugging Face instead of running these yourself.

### Step 1 — Encode the corpus (FAISS)

*Skip this step entirely if you downloaded `embeddings.npy` (Path B) or `faiss.index` (Path C) — see [Reproducibility Paths](#reproducibility-paths).*

```bash
python scripts/encode_documents.py \
  --config configs/fullpipeline.yaml \
  --output-dir /path/to/output/dir \
  --batch-size 32          # increase to 128-256 if you have a GPU
```

Produces `embeddings.npy` (~61 GB for 40M docs, float16), `doc_ids.json`, and `embeddings_manifest.json` in the output directory. Update `data.embeddings_path` and `faiss.save_path` in your config to point there.

### Step 2 — Build the FAISS index

*Skip this step entirely if you downloaded `faiss.index` (Path C) — see [Reproducibility Paths](#reproducibility-paths).*

```bash
python scripts/build_faiss_index.py \
  --config configs/fullpipeline.yaml
  # add --gpu if a CUDA GPU is available (significantly faster for large corpora)
```

Produces `faiss.index` at the path set by `faiss.save_path` in the config.

### Step 3 — Ingest Elasticsearch (BM25)

```bash
python scripts/ingest_elastic.py \
  --config configs/fullpipeline.yaml \
  --force    # recreates the index if it already exists
```

Steps 1–3 are shared across all experiments. Once the indices are built, each experiment below only runs the pipeline.

### Step 4 — Run each experiment

All runs use the same testset and pre-built indices. Only the `--config` and `--output` flags change.

```bash
TESTSET=test_data/round_3/golden_round3_testset_phaseA.json

# Experiment 1 — Primary submission (alpha=0.65, MMR, recency=0.3)
python scripts/run_hybrid_pipeline.py \
  --config configs/fullpipeline.yaml \
  --testset $TESTSET \
  --output results/submission.json

# Experiment 2 — BM25-only (alpha=0.0)
python scripts/run_hybrid_pipeline.py \
  --config configs/bm25only.yaml \
  --testset $TESTSET \
  --output results/submission_bm25only.json

# Experiment 3 — Alpha 0.3
python scripts/run_hybrid_pipeline.py \
  --config configs/alpha03.yaml \
  --testset $TESTSET \
  --output results/submission_alpha03.json

# Experiment 4 — MMR disabled
python scripts/run_hybrid_pipeline.py \
  --config configs/no_mmr.yaml \
  --testset $TESTSET \
  --output results/submission_no_mmr.json

# Experiment 5 — Recency disabled
python scripts/run_hybrid_pipeline.py \
  --config configs/no_recency.yaml \
  --testset $TESTSET \
  --output results/submission_no_recency.json

# Experiment 6 — Query normalizer fully disabled
python scripts/run_hybrid_pipeline.py \
  --config configs/no_normalizer.yaml \
  --testset $TESTSET \
  --output results/submission_no_normalizer.json

# Experiment 7 — Normalizer on, lowercase disabled
python scripts/run_hybrid_pipeline.py \
  --config configs/no_lowercase.yaml \
  --testset $TESTSET \
  --output results/submission_no_lowercase.json
```

### Step 5 — Evaluate each submission

Evaluation requires the official BioASQ Java evaluator JAR (`BioASQEvaluation.jar`). Replace `/path/to` with your local JAR location.

```bash
JAR=/path/to/flat/BioASQEvaluation/dist/BioASQEvaluation.jar
GOLDEN_A=test_data/round_3/golden_round3_testset_phaseA.json
GOLDEN_B=test_data/round_3/golden_round3_testset_phaseB.json

# Phase A — document retrieval (run for each submission file)
java -Xmx10G -cp $JAR evaluation.EvaluatorTask1b -phaseA -e 5 \
  $GOLDEN_A results/submission.json

# Phase B — answer generation (run for each submission file)
java -Xmx10G -cp $JAR evaluation.EvaluatorTask1b -phaseB -e 5 \
  $GOLDEN_B results/submission.json
```

The key metrics to read from the output:
- **Phase A:** position 9 = Doc MAP (primary retrieval metric)
- **Phase B:** position 1 = YesNo Accuracy, position 4 = Factoid MRR, position 7 = List F1

> **Note on golden files:** `golden_round3_testset_phaseA.json` contains all 117 questions including 11 unanswerable ones (which have dummy `exact_answer` values to prevent the Java evaluator from crashing). `golden_round3_testset_phaseB.json` excludes those 11 questions entirely. Always use the `-phaseA` file with the `-phaseA` flag and the `-phaseB` file with the `-phaseB` flag.

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

## Mobile App Integration

The `mobile_app/` directory contains a Flutter application that submits biomedical questions to this system and displays the answers. Authentication and user profiles are handled by Firebase on the client side. The only live connection between the app and this server is the query endpoint.

### 1 — Start the FastAPI server

```bash
# From the project root, with the virtual environment active:
uvicorn src.api.app:app --host 0.0.0.0 --port 8000

# To use a different config (e.g. BM25-only for faster startup):
RAG_CONFIG=configs/bm25only.yaml uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

The server exposes the following endpoints consumed by the app:

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/query/submit` | Submit a question; returns answer + sources |
| `GET` | `/api/v1/query/history` | Query history (returns empty list — app uses local Hive cache) |
| `GET` | `/api/v1/health` | Liveness check |
| `POST` | `/query` | Legacy internal endpoint (unchanged) |

### 2 — Point the app at the server

Edit one line in [mobile_app/lib/core/constants/api_constants.dart](mobile_app/lib/core/constants/api_constants.dart):

```dart
// Same WiFi network — use the server machine's local IP:
static const String baseUrl = 'http://192.168.x.x:8000/api/v1';

// Android emulator on the same machine as the server:
static const String baseUrl = 'http://10.0.2.2:8000/api/v1';

// iOS simulator on the same machine as the server:
static const String baseUrl = 'http://127.0.0.1:8000/api/v1';

// Production with a domain and TLS:
static const String baseUrl = 'https://your-domain.com/api/v1';
```

To find the server machine's local IP:
```bash
# Linux / macOS:
ip addr show | grep "inet " | grep -v 127.0.0.1

# Windows:
ipconfig | findstr "IPv4"
```

### 3 — Build and run the Flutter app

```bash
cd mobile_app
flutter pub get
flutter run          # connects to a plugged-in device or running emulator
```

### Request / Response contract

**POST /api/v1/query/submit**

```jsonc
// Request
{
  "question": "What is the mechanism of action of metformin?",
  "topK": 5,           // number of source documents (1–20, default 5)
  "includeSources": true,
  "sessionId": "optional-string"
}

// Response
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "question": "What is the mechanism of action of metformin?",
  "answer": "Metformin primarily works by...",
  "sources": [
    {
      "title": "Metformin: mechanisms in human obesity and weight loss",
      "url": "https://pubmed.ncbi.nlm.nih.gov/31477015/",
      "page": "Metformin activates AMP-activated protein kinase (AMPK)...",
      "confidence": 0.9312
    }
  ],
  "createdAt": "2026-05-24T14:30:00Z",
  "traceId": "uuid-of-pipeline-run",
  "latencyMs": 4821
}
```

The `Authorization: Bearer <Firebase-JWT>` header sent by the app is accepted but not validated server-side — all queries are processed regardless of token. Firebase token validation via `firebase-admin` can be added later without changing the mobile app.

---

## Authors

- **Professor Eric Nyberg** — Language Technology Institute, Carnegie Mellon University
- **John Dalton Gibson** — MSECE, Carnegie Mellon University

---

## Citation

The system description paper for this work is in preparation as a camera-ready submission to the CLEF 2026 BioASQ Synergy 14 working notes and is not yet published. Until it is, please cite the software directly:

```bibtex
@software{gibson2026medicalrag,
  author = {Gibson, John Dalton and Nyberg, Eric},
  title  = {MedicalRAG: A Hybrid Retrieval-Augmented Generation System for BioASQ Synergy 14},
  year   = {2026},
  url    = {https://github.com/jdaltonll02/medicalragsys},
  note   = {CLEF 2026 BioASQ Synergy 14 working notes paper forthcoming}
}
```

> **TODO:** replace this entry with the full working-notes citation (CEUR-WS volume, pages, DOI) once the camera-ready paper is accepted and published.
