import argparse
import json
import os
from typing import Dict

import requests
from openai import OpenAI


BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

SYSTEM_PROMPT = """
You are a database migration expert.
You will receive the current schema, target schema, last SQL result, downtime,
and schema match progress. Return exactly one SQL statement ending with a semicolon.
Rules:
- Never use DROP TABLE, TRUNCATE, or DELETE without a WHERE clause
- Prefer safe schema changes and minimal locking
- Output SQL only, with no explanation
""".strip()


def run_episode(task_id: str, seed: int = 42) -> float:
    reset_response = requests.post(f"{BASE_URL}/reset", json={"task_id": task_id, "seed": seed}, timeout=30)
    reset_response.raise_for_status()
    observation = reset_response.json()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for _ in range(15):
        prompt = (
            f"Current schema:\n{observation.get('current_schema_ddl', '')}\n\n"
            f"Target schema:\n{observation.get('target_schema_ddl', '')}\n\n"
            f"Last result: {observation.get('last_sql_result', 'RESET')}\n"
            f"Downtime: {observation.get('downtime_pct', 0.0):.1%}\n"
            f"Schema match: {observation.get('schema_match_pct', 0.0):.1%}\n"
            "Write the next SQL statement."
        )
        messages.append({"role": "user", "content": prompt})
        completion = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.0,
            max_tokens=200,
        )
        sql = (completion.choices[0].message.content or "").strip()
        messages.append({"role": "assistant", "content": sql})

        step_response = requests.post(
            f"{BASE_URL}/step",
            json={"sql": sql, "task_id": task_id, "execute_mode": "transaction"},
            timeout=30,
        )
        step_response.raise_for_status()
        payload = step_response.json()
        observation = payload.get("observation", {})
        if payload.get("done"):
            break

    grader_response = requests.post(f"{BASE_URL}/grader", json={"task_id": task_id}, timeout=30)
    grader_response.raise_for_status()
    return float(grader_response.json().get("score", 0.0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-tasks", action="store_true")
    parser.add_argument("--task", default="easy_add_column")
    args = parser.parse_args()

    if args.all_tasks:
        results: Dict[str, float] = {}
        for task_id in ["easy_add_column", "medium_rename_fk", "hard_repartition"]:
            score = run_episode(task_id)
            results[task_id] = score
            print(f"{task_id}: {score:.4f}", flush=True)
        print(json.dumps(results))
    else:
        print(json.dumps({args.task: run_episode(args.task)}))


if __name__ == "__main__":
    main()
