"""
BioASQ Answer Evaluation Judge Prompt System

This module provides prompts and utilities for using an LLM (ChatGPT) as an expert 
judge to evaluate BioASQ question-answer pairs and provide detailed scoring.
"""

JUDGE_SYSTEM_PROMPT = """You are an expert biomedical researcher and BioASQ evaluation specialist. 
Your role is to assess the quality and correctness of answers to biomedical questions by comparing 
them against reference answers and evaluating against established criteria.

You have deep expertise in:
- Biomedical literature and concepts (genes, proteins, drugs, diseases, clinical trials)
- BioASQ evaluation standards and metrics (accuracy, F-measure, MRR, etc.)
- Answer quality assessment (completeness, relevance, accuracy)
- Different question types (yes/no, factoid, list, summary)

You will evaluate answers based on:
1. ACCURACY: Is the answer factually correct and supported by biomedical knowledge?
2. COMPLETENESS: Does it answer all aspects of the question?
3. RELEVANCE: Are all provided facts directly relevant to the question?
4. CLARITY: Is the answer clear and well-structured?
5. CONCISENESS: Does it avoid unnecessary information while remaining complete?

For each evaluation, provide:
- A score (0-100) for each criterion
- An overall score (0-100)
- Detailed feedback explaining the scores
- Specific issues or areas for improvement
- Comparison to reference answer(s) when available

Be critical but fair. Deduct points for:
- Factual errors or hallucinations
- Missing key information
- Irrelevant information
- Poor answer structure or clarity
- Incorrect entity names or concentrations

Award points for:
- Accurate and complete information
- Clear, well-organized presentation
- Proper use of biomedical terminology
- Conciseness without losing completeness
- Direct answer to the question asked"""

JUDGE_EVALUATION_PROMPT_TEMPLATE = """Evaluate the following BioASQ answer and provide detailed scoring.

QUESTION: {question}
QUESTION TYPE: {question_type}

CANDIDATE ANSWER:
{candidate_answer}

{reference_section}

Provide your evaluation in the following JSON format:
{{
    "accuracy_score": <0-100>,
    "completeness_score": <0-100>,
    "relevance_score": <0-100>,
    "clarity_score": <0-100>,
    "conciseness_score": <0-100>,
    "overall_score": <0-100>,
    "feedback": "<detailed explanation of scores>",
    "strengths": ["<strength1>", "<strength2>", ...],
    "weaknesses": ["<weakness1>", "<weakness2>", ...],
    "improvements": "<suggestions for improvement>",
    "comparison_to_reference": "<how it compares to reference answer(s)>"
}}

Evaluate thoroughly and fairly. The overall_score should be a weighted average:
- For Yes/No questions: Mostly accuracy (weight 60%), clarity (20%), other (20%)
- For Factoid questions: Accuracy (50%), completeness (30%), clarity (20%)
- For List questions: Completeness (40%), accuracy (40%), clarity (20%)
- For Summary questions: Accuracy (40%), completeness (35%), clarity (25%)"""

JUDGE_BATCH_EVALUATION_PROMPT = """Evaluate the following batch of BioASQ answers. For each answer, provide:
1. A 0-100 overall score
2. Key feedback (1-2 sentences max)
3. Major issues if any (null if none)

Question batch:
{batch_json}

Return a JSON array with format:
[
    {{
        "question_id": "<id>",
        "overall_score": <0-100>,
        "feedback": "<brief feedback>",
        "major_issues": null or "<issue description>"
    }},
    ...
]

Be efficient but accurate. Focus on major quality issues."""

COMPARATIVE_EVALUATION_PROMPT = """Compare two BioASQ answers to the same question and determine which is better.

QUESTION: {question}
QUESTION TYPE: {question_type}

ANSWER A:
{answer_a}

ANSWER B:
{answer_b}

{reference_section}

Provide comparison analysis in JSON format:
{{
    "winner": "A" or "B",
    "score_a": <0-100>,
    "score_b": <0-100>,
    "rationale": "<explain why one is better>",
    "accuracy_comparison": "<which is more accurate>",
    "completeness_comparison": "<which is more complete>",
    "clarity_comparison": "<which is clearer>",
    "key_differences": ["<difference1>", "<difference2>", ...],
    "tie_explanation": null or "<if neither clearly better>"
}}

Be objective. Consider all quality dimensions."""

QUESTION_TYPE_SPECIFIC_GUIDANCE = {
    "yesno": """For YES/NO questions, scoring should emphasize:
- ACCURACY (60%): Is the yes/no answer correct?
- EXPLANATION QUALITY (25%): Does it properly explain the answer?
- BREVITY (15%): Is it concise without losing clarity?

Critical issues (major deductions):
- Wrong yes/no answer: -40 points
- Contradiction between answer and explanation: -30 points
- No supporting explanation: -20 points""",

    "factoid": """For FACTOID questions, scoring should emphasize:
- ACCURACY (50%): Are the named entities/facts correct?
- COMPLETENESS (30%): Are all reasonable answers included?
- CLARITY (20%): Are answers clear and properly formatted?

Critical issues (major deductions):
- Wrong entity or fact: -30 points per error
- Missing major entity: -20 points per missing answer
- Incorrect entity type: -15 points""",

    "list": """For LIST questions, scoring should emphasize:
- COMPLETENESS (40%): Are all major items included?
- ACCURACY (40%): Are all items correct and relevant?
- ORGANIZATION (20%): Is the list well-organized and clear?

Critical issues (major deductions):
- Missing critical items: -5 to -20 points each
- Incorrect items included: -10 to -25 points each
- Poor organization: -10 points""",

    "summary": """For SUMMARY questions, scoring should emphasize:
- ACCURACY (40%): Is all information correct?
- COMPLETENESS (35%): Does it cover key aspects?
- CLARITY (25%): Is it well-written and coherent?

Critical issues (major deductions):
- Factual error: -20 to -40 points
- Missing key point: -15 points each
- Too long/verbose: -10 points
- Unclear writing: -10 to -20 points"""
}

def get_judge_evaluation_prompt(question, question_type, candidate_answer, 
                                 reference_answers=None):
    """
    Generate a complete evaluation prompt for a single answer.
    
    Args:
        question: The BioASQ question text
        question_type: Type of question (yesno, factoid, list, summary)
        candidate_answer: The answer to evaluate
        reference_answers: List of reference/ideal answers (optional)
    
    Returns:
        str: Complete evaluation prompt
    """
    reference_section = ""
    if reference_answers:
        ref_text = "\n".join([f"- {ref}" for ref in reference_answers])
        reference_section = f"REFERENCE ANSWERS:\n{ref_text}\n"
    
    prompt = JUDGE_EVALUATION_PROMPT_TEMPLATE.format(
        question=question,
        question_type=question_type,
        candidate_answer=candidate_answer,
        reference_section=reference_section
    )
    
    # Add question-type-specific guidance
    if question_type in QUESTION_TYPE_SPECIFIC_GUIDANCE:
        prompt += f"\n\n{QUESTION_TYPE_SPECIFIC_GUIDANCE[question_type]}"
    
    return prompt

def get_comparative_evaluation_prompt(question, question_type, answer_a, answer_b,
                                      reference_answers=None):
    """
    Generate a prompt to compare two answers.
    
    Args:
        question: The BioASQ question text
        question_type: Type of question (yesno, factoid, list, summary)
        answer_a: First answer
        answer_b: Second answer  
        reference_answers: List of reference/ideal answers (optional)
    
    Returns:
        str: Complete comparative evaluation prompt
    """
    reference_section = ""
    if reference_answers:
        ref_text = "\n".join([f"- {ref}" for ref in reference_answers])
        reference_section = f"REFERENCE ANSWERS:\n{ref_text}\n"
    
    prompt = COMPARATIVE_EVALUATION_PROMPT.format(
        question=question,
        question_type=question_type,
        answer_a=answer_a,
        answer_b=answer_b,
        reference_section=reference_section
    )
    
    return prompt

def get_batch_evaluation_prompt(questions_with_answers):
    """
    Generate a prompt for batch evaluation of multiple answers.
    
    Args:
        questions_with_answers: List of dicts with keys:
            - question_id
            - question
            - question_type  
            - answer
    
    Returns:
        str: Complete batch evaluation prompt
    """
    import json
    batch_json = json.dumps(questions_with_answers, indent=2)
    return JUDGE_BATCH_EVALUATION_PROMPT.format(batch_json=batch_json)

def create_evaluation_report(question, question_type, candidate_answer, 
                            reference_answers, judge_response):
    """
    Create a formatted evaluation report from judge response.
    
    Args:
        question: The BioASQ question
        question_type: Type of question
        candidate_answer: The answer evaluated
        reference_answers: List of reference answers
        judge_response: JSON response from judge LLM
    
    Returns:
        dict: Structured evaluation report
    """
    import json
    
    # Parse judge response if it's a string
    if isinstance(judge_response, str):
        try:
            judge_data = json.loads(judge_response)
        except json.JSONDecodeError:
            judge_data = {"raw_response": judge_response}
    else:
        judge_data = judge_response
    
    report = {
        "question": question,
        "question_type": question_type,
        "candidate_answer": candidate_answer,
        "reference_answers": reference_answers,
        "evaluation": judge_data,
        "timestamp": __import__('datetime').datetime.now().isoformat()
    }
    
    return report

if __name__ == "__main__":
    # Example usage
    sample_question = "What is the role of BRCA1 in DNA repair?"
    sample_question_type = "factoid"
    sample_answer = "BRCA1 is a tumor suppressor protein involved in homologous recombination DNA repair, helping to fix double-strand breaks. Mutations in BRCA1 increase cancer risk, particularly breast and ovarian cancer."
    sample_references = [
        "BRCA1 encodes a nuclear phosphoprotein that plays a key role in maintaining genomic stability",
        "BRCA1 is involved in homologous recombination repair and transcriptional regulation"
    ]
    
    prompt = get_judge_evaluation_prompt(
        sample_question,
        sample_question_type,
        sample_answer,
        sample_references
    )
    
    print("=" * 80)
    print("SAMPLE JUDGE EVALUATION PROMPT")
    print("=" * 80)
    print(prompt)
