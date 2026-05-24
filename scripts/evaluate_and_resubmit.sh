#!/bin/bash
# ============================================================
# After a SLURM pipeline job finishes, this script:
#   1. Shows a quick sanity check of the submission
#   2. Prints the commands to run on your LOCAL machine for evaluation
#   3. Recommends next SLURM jobs if MAP is bad
#
# Usage:  bash scripts/evaluate_and_resubmit.sh [submission.json]
#
# Evaluation (run on LOCAL machine — needs BioASQ JAR):
#   cd ~/Dalton/bioasq/Evaluation-Measures
#   java -Xmx10G -cp ./flat/BioASQEvaluation/dist/BioASQEvaluation.jar \
#        evaluation.EvaluatorTask1b -phaseA -e 5 \
#        <golden_testset.json> <submission.json>
# ============================================================
set -euo pipefail
cd "$(dirname "$0")/.."

SUBMISSION="${1:-results/submission.json}"

echo "========================================================"
echo "  Submission sanity check: $SUBMISSION"
echo "========================================================"

if [[ ! -f "$SUBMISSION" ]]; then
  echo "ERROR: $SUBMISSION not found!"
  exit 1
fi

python3 - <<PYEOF
import json, sys
with open("$SUBMISSION") as f:
    sub = json.load(f)

qs = sub.get("questions", [])
print(f"  Questions in submission: {len(qs)}")

# Check document format
all_docs = [d for q in qs for d in q.get("documents", [])]
print(f"  Total document entries: {len(all_docs)}")

# URL vs bare PMID check
url_count = sum(1 for d in all_docs if "pubmed" in str(d).lower() or d.startswith("http"))
bare_count = len(all_docs) - url_count
print(f"  Bare PMID format: {bare_count}  (URLs: {url_count})")
if url_count > 0:
    print("  ⚠  WARNING: URLs found — should be bare PMIDs like '12345678'")
else:
    print("  ✓  All documents are bare PMIDs")

# Sample
if qs:
    q0 = qs[0]
    print(f"\n  Sample Q: {q0.get('body','')[:80]}")
    print(f"  Sample docs (first 3): {q0.get('documents',[])[:3]}")
PYEOF

echo ""
echo "========================================================"
echo "  HOW TO EVALUATE (run on your LOCAL machine)"
echo "========================================================"
echo ""
echo "  # Phase A (documents + snippets) — 117 questions:"
echo "  cd ~/Dalton/bioasq/Evaluation-Measures"
echo "  java -Xmx10G -cp ./flat/BioASQEvaluation/dist/BioASQEvaluation.jar \\"
echo "       evaluation.EvaluatorTask1b -phaseA -e 5 \\"
echo "       golden_round3_testset_phaseA.json submission.json"
echo "  # golden_round3_testset_phaseA.json: all 117 questions; 11 unanswerable"
echo "  # questions have dummy exact_answers so the Java parser does not crash."
echo "  # The dummy values do not affect Phase A (doc/snippet) scoring."
echo ""
echo "  # Output format (20 space-separated values):"
echo "  #  1-5:  Concepts (deprecated, always 0)"
echo "  #  6:    Doc MAP  ← KEY METRIC"
echo "  #  7-10: Doc GMAP, AvgP, P@5, P@10"
echo "  # 11-15: Snippet P@1, MRR, Avg, P@5, P@10"
echo "  # 16-20: RDF Triples (deprecated, always 0)"
echo ""
echo "  # Phase B (answers) — 106 questions (excludes 11 unanswerable):"
echo "  java -Xmx10G -cp ./flat/BioASQEvaluation/dist/BioASQEvaluation.jar \\"
echo "       evaluation.EvaluatorTask1b -phaseB -e 5 \\"
echo "       golden_round3_testset_phaseB.json submission.json"
echo ""
echo "========================================================"
echo "  NEXT SLURM JOBS (if Doc MAP ≈ 0)"
echo "========================================================"
echo ""
echo "  # BM25-only run (pure keyword retrieval, no broken FAISS):"
echo "  sbatch --export=ALL,CONFIG=configs/bm25only.yaml slurm/run_pipeline.slurm"
echo ""
echo "  # FAISS rebuild (title+abstract encoding — run once, takes ~1-2h):"
echo "  sbatch slurm/rebuild_faiss_index.slurm"
echo ""
echo "  # After FAISS rebuild, use normal config with alpha=0.65:"
echo "  sbatch slurm/run_pipeline.slurm"
