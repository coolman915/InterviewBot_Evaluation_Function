"""
InterviewBot - Candidate Answer Evaluation Function

Evaluates a candidate's interview answer using a rubric-based approach.
Scores across multiple dimensions (relevance, depth, JD alignment, communication)
and produces structured JSON output with scores and rationale.

Design decisions:
- Rubric-based, NOT golden-answer comparison (avoids penalizing valid alternative perspectives)
- Rubric is generated dynamically per competency area from the JD (supports adaptive interviewing)
- Structured JSON output enforced via response_format (constrains hallucination)
- All context (CV, JD, question, answer) passed in every prompt (grounding)

Author: Michael
"""

import json
from openai import OpenAI

# ---------------------------------------------------------------------------
# 1. Rubric generator – creates scoring criteria from the JD competency area
# ---------------------------------------------------------------------------

RUBRIC_SYSTEM_PROMPT = """You are an expert technical interviewer.
Given a competency area and its requirements from a Job Description,
generate a scoring rubric with 4 dimensions: relevance, depth, jd_alignment, communication.

For each dimension, describe what a score of 1, 3, and 5 looks like.

Respond ONLY with valid JSON matching this schema:
{
  "competency": "<area>",
  "dimensions": {
    "<dimension_name>": {
      "description": "<what this dimension measures>",
      "1": "<what a poor answer looks like>",
      "3": "<what an adequate answer looks like>",
      "5": "<what an excellent answer looks like>"
    }
  }
}"""


def generate_rubric(
    client: OpenAI,
    competency_area: str,
    jd_requirements: str,
    model: str = "gpt-4o-mini",
) -> dict:
    """Generate a scoring rubric for a specific competency area.

    Args:
        client: OpenAI client instance.
        competency_area: The skill/competency being evaluated (e.g. "Python").
        jd_requirements: The relevant section from the Job Description.
        model: OpenAI model to use.

    Returns:
        A dict containing dimension definitions with score anchors.
    """
    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": RUBRIC_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Competency area: {competency_area}\n\n"
                    f"JD Requirements:\n{jd_requirements}"
                ),
            },
        ],
    )
    return json.loads(response.choices[0].message.content)


# ---------------------------------------------------------------------------
# 2. Answer evaluator – scores one answer against the rubric
# ---------------------------------------------------------------------------

EVAL_SYSTEM_PROMPT = """You are a fair and consistent interview evaluator.
You will evaluate a candidate's answer using the provided rubric.

Rules:
- ONLY use information from the provided context (CV, JD, question, answer).
- Do NOT infer or assume facts not present in the documents.
- If the answer is ambiguous, score conservatively and explain why.
- Be specific in rationale — reference what the candidate said or failed to say.

Respond ONLY with valid JSON matching this schema:
{
  "competency": "<area evaluated>",
  "scores": {
    "relevance": {"score": <1-5>, "rationale": "<why>"},
    "depth": {"score": <1-5>, "rationale": "<why>"},
    "jd_alignment": {"score": <1-5>, "rationale": "<why>"},
    "communication": {"score": <1-5>, "rationale": "<why>"}
  },
  "overall_score": <1.0-5.0>,
  "strengths": ["<strength1>", "<strength2>"],
  "gaps": ["<gap1>", "<gap2>"],
  "summary": "<2-3 sentence overall assessment>"
}"""


def evaluate_answer(
    client: OpenAI,
    competency_area: str,
    jd_requirements: str,
    cv_section: str,
    question: str,
    candidate_answer: str,
    rubric: dict,
    model: str = "gpt-4o-mini",
) -> dict:
    """Evaluate a single candidate answer against the rubric.

    Args:
        client: OpenAI client instance.
        competency_area: The skill/competency being evaluated.
        jd_requirements: Relevant JD section for this competency.
        cv_section: Relevant CV section for this competency.
        question: The interview question that was asked.
        candidate_answer: The candidate's response.
        rubric: The scoring rubric (from generate_rubric).
        model: OpenAI model to use.

    Returns:
        A dict with per-dimension scores, rationale, strengths, and gaps.
    """
    context = (
        f"## Competency Area\n{competency_area}\n\n"
        f"## Job Description Requirements\n{jd_requirements}\n\n"
        f"## Candidate CV Section\n{cv_section}\n\n"
        f"## Interview Question\n{question}\n\n"
        f"## Candidate Answer\n{candidate_answer}\n\n"
        f"## Scoring Rubric\n{json.dumps(rubric, indent=2)}"
    )

    response = client.chat.completions.create(
        model=model,
        temperature=0.1,  # Low temp for consistent scoring
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": EVAL_SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ],
    )

    result = json.loads(response.choices[0].message.content)

    # Validate: ensure all scores are within range
    for dim in result.get("scores", {}).values():
        if not 1 <= dim.get("score", 0) <= 5:
            raise ValueError(f"Score out of range: {dim}")

    return result


# ---------------------------------------------------------------------------
# 3. Final report generator – aggregates all 5 question evaluations
# ---------------------------------------------------------------------------

REPORT_SYSTEM_PROMPT = """You are a hiring evaluation specialist.
Given evaluation results from 5 interview questions, produce a final report.

Rules:
- Base your recommendation ONLY on the evaluation data provided.
- Be specific about strengths and concerns.
- The recommendation must be "proceed" or "reject" with clear justification.

Respond ONLY with valid JSON matching this schema:
{
  "candidate_summary": "<2-3 sentence overview>",
  "competency_scores": {
    "<area>": {"average_score": <float>, "assessment": "<1 sentence>"}
  },
  "overall_score": <1.0-5.0>,
  "strengths": ["<top strength>"],
  "concerns": ["<top concern>"],
  "recommendation": "proceed" | "reject",
  "recommendation_rationale": "<clear explanation of why>"
}"""


def generate_final_report(
    client: OpenAI,
    evaluations: list[dict],
    model: str = "gpt-4o-mini",
) -> dict:
    """Generate the final hiring recommendation report.

    Args:
        client: OpenAI client instance.
        evaluations: List of evaluation results from evaluate_answer().
        model: OpenAI model to use.

    Returns:
        A dict with aggregated scores and a proceed/reject recommendation.
    """
    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": REPORT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Here are the evaluation results from 5 interview questions:\n\n"
                    + json.dumps(evaluations, indent=2)
                ),
            },
        ],
    )
    return json.loads(response.choices[0].message.content)


# ---------------------------------------------------------------------------
# 4. Orchestrator – runs the full evaluation pipeline for one answer
# ---------------------------------------------------------------------------


def evaluate_interview_answer(
    api_key: str,
    competency_area: str,
    jd_requirements: str,
    cv_section: str,
    question: str,
    candidate_answer: str,
    model: str = "gpt-4o-mini",
) -> dict:
    """End-to-end evaluation: generates rubric, then scores the answer.

    This is the main entry point for per-question evaluation.

    Args:
        api_key: OpenAI API key.
        competency_area: The skill being tested (e.g. "System Design").
        jd_requirements: JD requirements for this competency.
        cv_section: Candidate's CV section for this competency.
        question: The question asked.
        candidate_answer: What the candidate said.
        model: OpenAI model to use.

    Returns:
        A dict with rubric, scores, rationale, strengths, and gaps.
    """
    client = OpenAI(api_key=api_key)

    # Step 1: Generate rubric dynamically from JD
    rubric = generate_rubric(client, competency_area, jd_requirements, model)

    # Step 2: Evaluate answer against rubric
    evaluation = evaluate_answer(
        client, competency_area, jd_requirements, cv_section,
        question, candidate_answer, rubric, model,
    )

    return {"rubric": rubric, "evaluation": evaluation}
