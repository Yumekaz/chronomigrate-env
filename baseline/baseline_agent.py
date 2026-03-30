import argparse
import json
import os
import sys
from typing import Dict

import requests
from openai import OpenAI


BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")
MODEL = "gpt-4o-mini"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
client = None

SYSTEM_PROMPT = """
You are a database migration expert. You will be given:
1. current_schema_ddl: the current database schema
2. target_schema_ddl: the target schema you must achieve
3. last_sql_result: result of your last SQL command
4. downtime_pct: percentage of queries that failed due to your last action

Your goal: write SQL commands to migrate from current to target schema.

Rules:
- NEVER use DROP TABLE or TRUNCATE (destroys data, instant zero score)
- Prefer CONCURRENTLY for index creation
- For column renames, drop FK constraints first
- Output ONLY a single SQL statement per turn, no explanations
- Output format: just the SQL statement, ending with semicolon
""".strip()


def _get_client() -> OpenAI:
    global client
    if client is None:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is required to run the baseline agent.")
        client = OpenAI(api_key=OPENAI_API_KEY)
    return client


def _parse_final_json(stdout: str) -> Dict[str, float]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        return {}
    for line in reversed(lines):
        if line.startswith("{") and line.endswith("}"):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
    return {}


def _normalize_sql(sql: str) -> str:
    cleaned = (sql or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("sql"):
            cleaned = cleaned[3:].strip()
    cleaned = " ".join(line.strip() for line in cleaned.splitlines() if line.strip())
    if cleaned and not cleaned.endswith(";"):
        cleaned += ";"
    return cleaned


def run_episode(task_id: str, seed: int = 42) -> float:
    reset_response = requests.post(
        f"{BASE_URL}/reset",
        json={"task_id": task_id, "seed": seed},
        timeout=30,
    )
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
        completion = _get_client().chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.0,
            max_tokens=200,
        )
        sql = _normalize_sql(completion.choices[0].message.content or "")
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
    try:
        main()
    except Exception as exc:
        print(json.dumps({"error": str(exc)}))
        sys.exit(1)
