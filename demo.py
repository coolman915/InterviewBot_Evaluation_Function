"""
Demo: Run the evaluation pipeline with sample interview data.

Usage:
    Put your key in a local .env file:
        OPENAI_API_KEY="sk-..."

    python demo.py
"""

import json
import os
from pathlib import Path

from evaluate import evaluate_interview_answer, generate_final_report
from openai import OpenAI

# ---------------------------------------------------------------------------
# Sample data (realistic interview scenario)
# ---------------------------------------------------------------------------

JD_REQUIREMENTS = """
Senior Backend Engineer - Requirements:
- 5+ years of experience with Python in production systems
- Strong understanding of distributed systems and microservices
- Experience with cloud platforms (Azure preferred)
- Ability to design and implement RESTful APIs at scale
- Experience with message queues (Kafka, RabbitMQ, or Service Bus)
"""

CV_SECTION = """
Technical Skills:
- Python (7 years): Django, FastAPI, asyncio, Celery
- Built event-driven microservices handling 50K+ requests/sec
- Azure: AKS, Service Bus, Cosmos DB, Functions
- Designed REST APIs serving 200+ internal consumers
"""

# Five interview questions with candidate answers
INTERVIEW_QA = [
    {
        "competency": "Python proficiency",
        "jd": "5+ years of experience with Python in production systems",
        "cv": "Python (7 years): Django, FastAPI, asyncio, Celery",
        "question": (
            "Tell me about a time you used Python's asyncio in production. "
            "What problem did it solve and what tradeoffs did you encounter?"
        ),
        "answer": (
            "At my previous company, we had a data ingestion service that was "
            "bottlenecked on I/O — it was pulling from 15 different APIs sequentially. "
            "I rewrote it using asyncio with aiohttp, which brought the total fetch time "
            "from 45 seconds down to about 6 seconds. The main tradeoff was debugging — "
            "async stack traces are harder to read, and we had to be careful with "
            "connection pooling to avoid exhausting file descriptors. We also added "
            "semaphores to rate-limit concurrent requests per API."
        ),
    },
    {
        "competency": "Distributed systems",
        "jd": "Strong understanding of distributed systems and microservices",
        "cv": "Built event-driven microservices handling 50K+ requests/sec",
        "question": (
            "How would you handle a situation where one microservice in your "
            "pipeline becomes a bottleneck and starts backing up messages?"
        ),
        "answer": (
            "I'd first look at the metrics to understand if it's a throughput issue "
            "or a latency issue. If it's throughput, I'd scale the consumer horizontally. "
            "But the immediate fix would be to make sure we have a dead letter queue "
            "so failed messages don't block the main queue. I've done this with Azure "
            "Service Bus where we set up auto-scaling based on queue depth."
        ),
    },
    {
        "competency": "Cloud platform experience",
        "jd": "Experience with cloud platforms (Azure preferred)",
        "cv": "Azure: AKS, Service Bus, Cosmos DB, Functions",
        "question": (
            "Describe how you'd architect a new microservice on Azure. "
            "Walk me through your infrastructure choices."
        ),
        "answer": (
            "I'd containerize with Docker, deploy on AKS for orchestration. "
            "For async messaging, Azure Service Bus. For the database, it depends "
            "on the access pattern — Cosmos DB for high-throughput document storage, "
            "or Azure SQL for relational needs. I'd use Azure Key Vault for secrets "
            "and Application Insights for observability. Infrastructure as code "
            "with Terraform so it's reproducible across environments."
        ),
    },
    {
        "competency": "API design",
        "jd": "Ability to design and implement RESTful APIs at scale",
        "cv": "Designed REST APIs serving 200+ internal consumers",
        "question": (
            "You're designing an API that 200 internal teams will consume. "
            "How do you handle versioning and breaking changes?"
        ),
        "answer": (
            "I use URL-based versioning like /v1/resources because it's the most "
            "explicit. For breaking changes, I run both versions in parallel with a "
            "deprecation timeline — usually 3 months. I'd communicate through an API "
            "changelog and set up usage tracking so I know which teams are still on "
            "the old version before sunsetting it."
        ),
    },
    {
        "competency": "Message queues",
        "jd": "Experience with message queues (Kafka, RabbitMQ, or Service Bus)",
        "cv": "Built event-driven microservices handling 50K+ requests/sec",
        "question": (
            "When would you choose Kafka over Azure Service Bus, and why?"
        ),
        "answer": (
            "Kafka is better when you need event streaming with replay — like "
            "an event sourcing pattern where consumers might need to re-read history. "
            "Service Bus is better for traditional message queuing with features like "
            "sessions, dead lettering, and scheduled delivery built in. For our use "
            "case at my last job, we went with Service Bus because we needed guaranteed "
            "ordered processing per customer and the session feature handled that natively."
        ),
    },
]


def load_local_env(env_path: str = ".env") -> None:
    """Load simple KEY=VALUE pairs from a local .env file into os.environ."""
    path = Path(env_path)
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')

        # Keep real environment variables as the highest-priority source.
        if key and key not in os.environ:
            os.environ[key] = value


def main():
    load_local_env()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY in your environment or .env file first.")

    print("=" * 60)
    print("InterviewBot - Candidate Evaluation Demo")
    print("=" * 60)

    evaluations = []

    for i, qa in enumerate(INTERVIEW_QA, 1):
        print(f"\n--- Question {i}/5: {qa['competency']} ---")
        print(f"Question: {qa['question']}")
        print(f"Answer: {qa['answer']}")

        result = evaluate_interview_answer(
            api_key=api_key,
            competency_area=qa["competency"],
            jd_requirements=qa["jd"],
            cv_section=qa["cv"],
            question=qa["question"],
            candidate_answer=qa["answer"],
        )

        eval_data = result["evaluation"]
        print(f"Overall score: {eval_data.get('overall_score', 'N/A')}/5")
        print(f"Summary: {eval_data.get('summary', 'N/A')}")

        evaluations.append(eval_data)

    # Generate final report
    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)

    client = OpenAI(api_key=api_key)
    report = generate_final_report(client, evaluations)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
