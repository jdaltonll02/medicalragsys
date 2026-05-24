#!/usr/bin/env python3
"""
BioASQ Submission Format Validator

Validates that a BioASQ results JSON file meets all submission requirements:
- Every question has id, body, type, documents, snippets
- Every question has either exact_answer OR ideal_answer (based on type)
- Answer formats are correct for each question type
- Document/snippet counts are within limits
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

def validate_submission(results_file: str) -> Tuple[bool, List[str]]:
    """
    Validate a BioASQ results JSON file for submission compliance.
    
    Args:
        results_file: Path to the results JSON file
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    try:
        with open(results_file) as f:
            results = json.load(f)
    except Exception as e:
        return False, [f"Failed to load JSON file: {e}"]
    
    if "questions" not in results:
        return False, ["Missing 'questions' field in submission"]
    
    questions = results["questions"]
    if not isinstance(questions, list):
        return False, ["'questions' field must be a list"]
    
    if len(questions) == 0:
        return False, ["No questions in submission"]
    
    # Validate each question
    for idx, q in enumerate(questions):
        q_id = q.get("id", f"<missing at index {idx}>")
        q_type = q.get("type", "<missing>")
        
        # Required fields
        required_fields = ["id", "body", "type", "documents", "snippets"]
        for field in required_fields:
            if field not in q:
                errors.append(f"Q{q_id}: Missing required field '{field}'")
        
        # Validate question type
        valid_types = ["yesno", "factoid", "list", "summary"]
        if q_type not in valid_types:
            errors.append(f"Q{q_id}: Invalid type '{q_type}'. Must be one of {valid_types}")
        
        # Check for answer fields
        has_exact = q.get("exact_answer") is not None and q.get("exact_answer") != "" and q.get("exact_answer") != []
        has_ideal = q.get("ideal_answer") is not None and q.get("ideal_answer") != ""
        
        if not has_exact and not has_ideal:
            errors.append(f"Q{q_id}: Missing both 'exact_answer' and 'ideal_answer'")
        
        # Validate answer formats based on type
        if q_type == "yesno":
            if has_exact:
                exact = q.get("exact_answer")
                if exact not in ["yes", "no"]:
                    errors.append(f"Q{q_id}: yesno exact_answer must be 'yes' or 'no', got '{exact}'")
        
        elif q_type == "factoid":
            if has_exact:
                exact = q.get("exact_answer")
                if not isinstance(exact, list):
                    errors.append(f"Q{q_id}: factoid exact_answer must be a list, got {type(exact).__name__}")
                elif len(exact) > 5:
                    errors.append(f"Q{q_id}: factoid exact_answer can have max 5 items, got {len(exact)}")
                else:
                    for item in exact:
                        if not isinstance(item, list) or len(item) == 0:
                            errors.append(f"Q{q_id}: factoid exact_answer items must be lists, got {item}")
        
        elif q_type == "list":
            if has_exact:
                exact = q.get("exact_answer")
                if not isinstance(exact, list):
                    errors.append(f"Q{q_id}: list exact_answer must be a list, got {type(exact).__name__}")
                elif len(exact) > 100:
                    errors.append(f"Q{q_id}: list exact_answer can have max 100 items, got {len(exact)}")
                else:
                    for item in exact:
                        if not isinstance(item, list) or len(item) == 0:
                            errors.append(f"Q{q_id}: list exact_answer items must be lists, got {item}")
        
        elif q_type == "summary":
            if has_exact:
                errors.append(f"Q{q_id}: summary questions should NOT have exact_answer")
        
        # Check documents (limit 10)
        if "documents" in q:
            docs = q["documents"]
            if not isinstance(docs, list):
                errors.append(f"Q{q_id}: 'documents' must be a list")
            elif len(docs) > 10:
                errors.append(f"Q{q_id}: Max 10 documents allowed, found {len(docs)}")
        
        # Check snippets (limit 10)
        if "snippets" in q:
            snips = q["snippets"]
            if not isinstance(snips, list):
                errors.append(f"Q{q_id}: 'snippets' must be a list")
            elif len(snips) > 10:
                errors.append(f"Q{q_id}: Max 10 snippets allowed, found {len(snips)}")
        
        # Check ideal_answer length (max 200 words)
        if has_ideal:
            ideal = q.get("ideal_answer", "")
            word_count = len(ideal.split())
            if word_count > 200:
                errors.append(f"Q{q_id}: ideal_answer exceeds 200 words ({word_count} words)")
    
    is_valid = len(errors) == 0
    return is_valid, errors

def print_report(results_file: str, is_valid: bool, errors: List[str]):
    """Print validation report."""
    print("=" * 80)
    print(f"BioASQ Submission Validation Report")
    print("=" * 80)
    print(f"File: {results_file}")
    
    # Count questions
    try:
        with open(results_file) as f:
            results = json.load(f)
            num_questions = len(results.get("questions", []))
            print(f"Questions: {num_questions}")
    except:
        pass
    
    print()
    
    if is_valid:
        print("✓ VALID - All requirements met!")
    else:
        print(f"✗ INVALID - {len(errors)} errors found:")
        print()
        
        # Group errors by question
        errors_by_q = {}
        for error in errors:
            if error.startswith("Q"):
                q_id = error.split(":")[0]
                if q_id not in errors_by_q:
                    errors_by_q[q_id] = []
                errors_by_q[q_id].append(error[len(q_id)+2:].strip())
            else:
                if "General" not in errors_by_q:
                    errors_by_q["General"] = []
                errors_by_q["General"].append(error)
        
        for q_id in sorted(errors_by_q.keys()):
            print(f"{q_id}:")
            for err in errors_by_q[q_id]:
                print(f"  - {err}")
            print()
    
    print("=" * 80)
    return 0 if is_valid else 1

def main():
    parser = argparse.ArgumentParser(description="Validate BioASQ submission format")
    parser.add_argument("results_file", help="Path to results JSON file")
    parser.add_argument("--fix", action="store_true", help="Auto-fix missing answers (fallback mode)")
    
    args = parser.parse_args()
    
    is_valid, errors = validate_submission(args.results_file)
    
    if not is_valid and args.fix:
        print("Attempting to fix missing answers...")
        
        with open(args.results_file) as f:
            results = json.load(f)
        
        fixed_count = 0
        for q in results.get("questions", []):
            q_type = q.get("type", "")
            has_exact = q.get("exact_answer") is not None and q.get("exact_answer") != "" and q.get("exact_answer") != []
            has_ideal = q.get("ideal_answer") is not None and q.get("ideal_answer") != ""
            
            if not has_exact and not has_ideal:
                fixed_count += 1
                
                if q_type == "yesno":
                    q["exact_answer"] = "no"
                    q["ideal_answer"] = "Insufficient evidence available."
                elif q_type == "factoid":
                    q["exact_answer"] = [["Information not available"]]
                    q["ideal_answer"] = "The requested information could not be identified."
                elif q_type == "list":
                    q["exact_answer"] = [["No specific items identified"]]
                    q["ideal_answer"] = "Insufficient evidence available."
                else:  # summary
                    q["ideal_answer"] = "The requested information could not be addressed."
        
        with open(args.results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Fixed {fixed_count} questions")
        is_valid, errors = validate_submission(args.results_file)
    
    exit_code = print_report(args.results_file, is_valid, errors)
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
