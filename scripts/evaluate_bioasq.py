#!/usr/bin/env python3
"""
Evaluate BioASQ-style retriever + answer generation output.

Inputs:
- predictions: BioASQ submission JSON (with questions[].id, documents, exact_answer, ideal_answer)
- golden: BioASQ golden JSON (with questions[].id, documents, exact_answer, ideal_answer, type, body)

Outputs:
- Aggregated metrics (retrieval + answer-type metrics) in JSON
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from evaluation.evaluation_QA_system.RAG_evaluator import RAGEvaluator


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_prediction(pred_q: Dict[str, Any]) -> Dict[str, Any]:
    """Convert BioASQ submission question to evaluator prediction format."""
    docs = pred_q.get("documents", []) or []
    retrieved_documents = [{"doc_id": str(d)} for d in docs]

    # Prefer exact_answer when present; else ideal_answer
    answer = pred_q.get("exact_answer")
    if answer is None:
        answer = pred_q.get("ideal_answer")

    return {
        "question_id": pred_q.get("id"),
        "retrieved_documents": retrieved_documents,
        "answer": answer,
    }


def normalize_ground_truth(gold_q: Dict[str, Any]) -> Dict[str, Any]:
    """Convert BioASQ golden question to evaluator ground-truth format."""
    relevant_docs = gold_q.get("documents") or gold_q.get("relevant_docs") or []
    return {
        "question_id": gold_q.get("id"),
        "relevant_docs": [str(d) for d in relevant_docs],
        "exact_answer": gold_q.get("exact_answer"),
        "ideal_answer": gold_q.get("ideal_answer"),
        "type": gold_q.get("type"),
        "question_text": gold_q.get("body") or gold_q.get("question"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BioASQ predictions with BioASQ-style metrics")
    parser.add_argument("--pred", required=True, help="Path to BioASQ submission JSON")
    parser.add_argument("--gold", required=True, help="Path to BioASQ golden JSON")
    parser.add_argument("--output", help="Optional output JSON path for metrics")
    parser.add_argument("--use-llm-judge", action="store_true", help="Use LLM judge for answer quality")
    parser.add_argument("--llm-judge-model", default="gpt-4o-mini-2024-07-18", help="LLM judge model")
    args = parser.parse_args()

    pred_path = Path(args.pred)
    gold_path = Path(args.gold)

    pred_data = load_json(pred_path)
    gold_data = load_json(gold_path)

    pred_questions = pred_data.get("questions", [])
    gold_questions = gold_data.get("questions", [])

    pred_by_id = {q.get("id"): q for q in pred_questions if q.get("id")}
    gold_by_id = {q.get("id"): q for q in gold_questions if q.get("id")}

    common_ids = [qid for qid in pred_by_id.keys() if qid in gold_by_id]
    if not common_ids:
        raise SystemExit("No overlapping question IDs between predictions and golden file.")

    predictions: List[Dict[str, Any]] = []
    ground_truth: List[Dict[str, Any]] = []
    missing_pred = 0
    missing_gold = 0

    for qid in common_ids:
        pred_q = pred_by_id.get(qid)
        gold_q = gold_by_id.get(qid)
        if pred_q is None:
            missing_pred += 1
            continue
        if gold_q is None:
            missing_gold += 1
            continue
        predictions.append(normalize_prediction(pred_q))
        ground_truth.append(normalize_ground_truth(gold_q))

    evaluator = RAGEvaluator(use_llm_judge=args.use_llm_judge, llm_judge_model=args.llm_judge_model)
    metrics = evaluator.evaluate_batch(predictions, ground_truth)

    metrics.update({
        "num_questions_evaluated": len(predictions),
        "missing_predictions": missing_pred,
        "missing_golden": missing_gold,
        "pred_file": str(pred_path),
        "gold_file": str(gold_path),
    })

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"[OK] Metrics saved to {out_path}")
    else:
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
