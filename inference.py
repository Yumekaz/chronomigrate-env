import argparse
import json
import os
import re
import sys
import time
from typing import Dict, List

import requests
from requests import exceptions as requests_exceptions
from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError


BASE_URL = os.environ.get("ENV_BASE_URL", "http://127.0.0.1:7860")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
API_KEY = (
    os.getenv("HF_TOKEN")
    or os.getenv("API_KEY")
    or os.getenv("OPENAI_API_KEY", "")
)
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
MAX_STEPS = 20
OPENAI_RETRY_ATTEMPTS = int(os.getenv("OPENAI_RETRY_ATTEMPTS", "4"))
ENV_REQUEST_CONNECT_TIMEOUT_SECONDS = float(
    os.getenv("ENV_REQUEST_CONNECT_TIMEOUT_SECONDS", "5")
)
ENV_REQUEST_READ_TIMEOUT_SECONDS = float(
    os.getenv("ENV_REQUEST_READ_TIMEOUT_SECONDS", "90")
)
ENV_REQUEST_RETRY_ATTEMPTS = int(os.getenv("ENV_REQUEST_RETRY_ATTEMPTS", "2"))

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
- when replacing a live table, create the replacement under a temporary name first
- create any required child tables or partitions on the replacement before copying data
- copy data before renaming or dropping anything
- rename/swap only after the replacement structure exists and has been populated
- cleanup old tables last, and only when they are no longer needed
- use the target schema, error messages, and recent history to decide the next step
- do not repeat the same failed SQL statement unchanged

When current and target schemas differ significantly, take the smallest safe step
toward the target schema rather than attempting the whole migration in one SQL statement.

Return only SQL. No explanation. End with a semicolon.
""".strip()


def _get_client() -> OpenAI:
    if not API_KEY:
        raise RuntimeError("HF_TOKEN, API_KEY, or OPENAI_API_KEY is required.")
    return OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


def _generate_sql(client: OpenAI, messages: List[Dict[str, str]], seed: int) -> str:
    for attempt in range(OPENAI_RETRY_ATTEMPTS):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.0,
                seed=seed,
                max_tokens=200,
            )
            return _normalize_sql(completion.choices[0].message.content or "")
        except (RateLimitError, APIConnectionError, APITimeoutError) as exc:
            if attempt == OPENAI_RETRY_ATTEMPTS - 1:
                raise
            time.sleep(_retry_delay_seconds(exc, attempt))

    raise RuntimeError("OpenAI request retry loop exited unexpectedly.")


def _retry_delay_seconds(exc: Exception, attempt: int) -> float:
    message = str(exc)
    match = re.search(r"Please try again in ([0-9.]+)(ms|s)", message, re.IGNORECASE)
    if match:
        value = float(match.group(1))
        unit = match.group(2).lower()
        delay = value / 1000.0 if unit == "ms" else value
        return max(1.0, min(10.0, delay + 0.5))
    return min(10.0, 2.0 * (attempt + 1))


def _env_post(path: str, payload: Dict[str, object], timeout: tuple[float, float] | None = None):
    timeout = timeout or (
        ENV_REQUEST_CONNECT_TIMEOUT_SECONDS,
        ENV_REQUEST_READ_TIMEOUT_SECONDS,
    )
    last_exc: Exception | None = None
    for attempt in range(ENV_REQUEST_RETRY_ATTEMPTS):
        try:
            response = requests.post(
                f"{BASE_URL}{path}",
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            return response
        except (requests_exceptions.ConnectionError, requests_exceptions.ReadTimeout) as exc:
            last_exc = exc
            if attempt == ENV_REQUEST_RETRY_ATTEMPTS - 1:
                raise
            time.sleep(min(5.0, attempt + 1))
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"Unexpected env request failure for {path}")


def _task_guidance(task_id: str) -> str:
    if task_id == "easy_add_column":
        return (
            "This task is usually solved with additive ALTER TABLE ADD COLUMN statements. "
            "Prefer safe additive changes."
        )
    if task_id == "medium_rename_fk":
        return (
            "Do not rename users.id while foreign keys still reference it. "
            "Drop the referencing constraint first, then rename the column, then recreate the foreign key."
        )
    if task_id == "hard_repartition":
        return (
            "Do not repartition the live events table in place. Use create-copy-swap: "
            "create a new partitioned parent table matching the target, create all eight hash partitions, "
            "backfill rows in batches, rename the old table to events_old, rename the new table to events, "
            "and drop events_old only after the swap succeeds."
        )
    return ""


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
    reset_response = _env_post("/reset", {"task_id": task_id, "seed": seed})
    observation = reset_response.json()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    history: List[Dict[str, str]] = []
    client = _get_client()
    attempt_budget = MAX_STEPS * 3
    attempts = 0

    while (
        attempts < attempt_budget
        and not bool(observation.get("done", False))
        and int(observation.get("step_count", 0)) < MAX_STEPS
    ):
        attempts += 1
        prompt = (
            f"Task id: {task_id}\n\n"
            f"Task-specific guidance:\n{_task_guidance(task_id)}\n\n"
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
        sql = _generate_sql(client, messages, seed)
        messages.append({"role": "assistant", "content": sql})

        if _is_obviously_unsafe_sql(sql):
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "That SQL is unsafe. Propose one safer, non-destructive SQL "
                        "statement instead. Output only SQL."
                    ),
                }
            )
            sql = _generate_sql(client, messages, seed)
            messages.append({"role": "assistant", "content": sql})
            if _is_obviously_unsafe_sql(sql):
                messages.append(
                    {
                        "role": "system",
                        "content": "The unsafe SQL was skipped. Propose a different safe SQL step next turn.",
                    }
                )
                continue

        if _repeats_failed_sql(sql, history):
            messages.append(
                {
                    "role": "system",
                    "content": "That SQL repeated a failed step and was skipped. Try a different SQL statement next turn.",
                }
            )
            continue

        step_response = _env_post(
            "/step",
            {"sql": sql, "task_id": task_id, "execute_mode": "transaction"},
        )
        payload = step_response.json()
        observation = payload.get("observation", {})
        history.append(
            {
                "sql": sql,
                "result": str(observation.get("last_sql_result", "")),
                "schema_match": str(observation.get("schema_match_pct", 0.0)),
            }
        )
        if payload.get("done", False):
            break

    grader_response = _env_post("/grader", {"task_id": task_id})
    return float(grader_response.json().get("score", 0.0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-tasks", action="store_true")
    parser.add_argument("--task", default="easy_add_column")
    args = parser.parse_args()

    if args.all_tasks:
        results: Dict[str, float] = {}
        for task_id in ["easy_add_column", "medium_rename_fk", "hard_repartition"]:
            score = run_episode(task_id, seed=42)
            results[task_id] = score
            print(f"{task_id}: {score:.4f}", flush=True)
        print(json.dumps(results))
    else:
        print(json.dumps({args.task: run_episode(args.task, seed=42)}))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"error": str(exc)}))
        sys.exit(1)
