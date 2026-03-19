# InterviewBot - Evaluation Function

Candidate answer evaluation function for the InterviewBot system design exercise.

## Architecture

The evaluation pipeline has 3 stages:

```
JD Requirements ──→ [generate_rubric] ──→ Rubric
                                            │
CV Section ─────┐                           ▼
Question ───────┼──→ [evaluate_answer] ──→ Per-Question Score (1-5) + Rationale
Answer ─────────┘
                                            │  (×5 questions)
                                            ▼
                     [generate_final_report] ──→ Proceed / Reject + Explanation
```

## Design Decisions

| Decision | Why |
|---|---|
| **Rubric-based** (not golden-answer) | Open-ended questions have multiple valid answers. Rubrics evaluate quality of reasoning, not similarity to a predefined text. |
| **Dynamic rubric generation** | Supports adaptive interviewing — rubrics are created per competency from the JD, so they work even for dynamically generated questions. |
| **Structured JSON output** (`response_format`) | Constrains the LLM, prevents hallucination and off-topic responses. Output is validated before returning. |
| **Full context in every prompt** | Grounding — the LLM never guesses. CV, JD, question, and answer are all provided explicitly. |
| **Low temperature** (0.1-0.2) | Scoring consistency across candidates. |
| **4 dimensions** (relevance, depth, jd_alignment, communication) | Discussed in the system design interview as the core evaluation axes. |

## Quick Start

```bash
pip install openai
echo 'OPENAI_API_KEY="sk-..."' > .env
python demo.py
```

You can also set `OPENAI_API_KEY` directly in your shell instead of using `.env`.

## Files

- `evaluate.py` — Core evaluation functions (the deliverable)
- `demo.py` — Runs the full pipeline with sample data

## Swapping to Azure OpenAI

Replace the client initialization:

```python
from openai import AzureOpenAI

client = AzureOpenAI(
    azure_endpoint="https://<your-resource>.openai.azure.com",
    api_version="2024-10-21",
    api_key="<your-azure-key>",
)
```

Everything else stays the same — the function signatures and prompts are provider-agnostic.
