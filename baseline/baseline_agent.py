import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional

import requests
from openai import OpenAI


BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")
MODEL = "gpt-4o-mini"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
MAX_STEPS = 20

SYSTEM_PROMPT = """
You are interacting with a database migration environment over repeated turns.

Each observation includes:
- current_schema_ddl: the current database schema
- target_schema_ddl: the schema to reach
- last_sql_result: result of the previous SQL statement
- downtime_pct: downtime caused by the previous step
- schema_match_pct: current progress toward the target schema
- step_count: how many steps have been used

On each turn, write exactly one SQL statement to submit next.
The environment will execute that SQL and return a new observation.

Goal: move the current schema toward the target schema while avoiding downtime
and preserving data integrity.

Use conservative database-migration strategy:
- prefer additive and reversible changes
- avoid destructive operations on live data
- if a table layout must change substantially, prefer create-copy-swap over destructive in-place changes
- create required structures before moving or renaming live objects
- use the target schema, error messages, and recent history to decide the next step
- do not repeat the same failed SQL statement unchanged

Return only SQL. No explanation. End with a semicolon.
""".strip()

client = None


def _get_client() -> OpenAI:
    global client
    if client is None:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is required to run the baseline agent.")
        client = OpenAI(api_key=OPENAI_API_KEY)
    return client


def _normalize_sql(sql: str) -> str:
    cleaned = (sql or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("sql"):
            cleaned = cleaned[3:].strip()
    cleaned = "\n".join(line.rstrip() for line in cleaned.splitlines() if line.strip())
    if cleaned and not cleaned.endswith(";"):
        cleaned += ";"
    return cleaned


def _recommended_step(
    task_id: str, observation: Dict[str, object], successful_actions: List[str]
) -> Optional[str]:
    return None


def _select_sql(task_id: str, model_sql: str, recommended_step: Optional[str]) -> str:
    return _normalize_sql(model_sql)


def _is_obviously_unsafe_sql(sql: str) -> bool:
    normalized = re.sub(r"\s+", " ", _normalize_sql(sql).upper())
    if not normalized:
        return True
    if "TRUNCATE" in normalized:
        return True
    if "DROP SCHEMA" in normalized:
        return True
    if "DROP TABLE" in normalized:
        return True
    return False


def _repeats_failed_sql(sql: str, history: List[Dict[str, str]]) -> bool:
    candidate = _normalize_sql(sql)
    if not candidate or not history:
        return False
    last = history[-1]
    return last.get("result") != "SUCCESS" and candidate == last.get("sql", "")


def run_episode(task_id: str, seed: int = 42) -> float:
    reset_response = requests.post(
        f"{BASE_URL}/reset",
        json={"task_id": task_id, "seed": seed},
        timeout=30,
    )
    reset_response.raise_for_status()
    observation = reset_response.json()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    history: List[Dict[str, str]] = []

    for _ in range(MAX_STEPS):
        prompt = (
            f"Task id: {task_id}\n\n"
            f"Current schema:\n{observation.get('current_schema_ddl', '')}\n\n"
            f"Target schema:\n{observation.get('target_schema_ddl', '')}\n\n"
            f"Last SQL result: {observation.get('last_sql_result', 'RESET')}\n"
            f"Downtime: {observation.get('downtime_pct', 0.0):.1%}\n"
            f"Schema match: {observation.get('schema_match_pct', 0.0):.1%}\n"
            f"Step count: {observation.get('step_count', 0)}\n\n"
            f"Recent history:\n{json.dumps(history[-5:], indent=2)}\n\n"
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
        if _is_obviously_unsafe_sql(sql) or _repeats_failed_sql(sql, history):
            messages.append({"role": "assistant", "content": sql})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "That SQL is unsafe or repeats a failed step. "
                        "Propose one safer, non-destructive SQL statement instead. "
                        "Output only SQL."
                    ),
                }
            )
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
        history.append(
            {
                "sql": sql,
                "result": str(observation.get("last_sql_result", "")),
            }
        )
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
