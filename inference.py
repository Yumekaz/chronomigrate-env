import argparse
import json
import os
import re
import time
from typing import Dict, List

import requests
from requests import exceptions as requests_exceptions
from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError
from server.tasks import normalize_task_score
from server.main import app as app


BASE_URL = (
    os.environ.get("SPACE_URL")
    or os.environ.get("ENV_BASE_URL")
    or "http://localhost:7860"
)
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
API_KEY = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY", "")
MAX_STEPS = 20
OPENAI_RETRY_ATTEMPTS = int(os.getenv("OPENAI_RETRY_ATTEMPTS", "4"))
ENV_REQUEST_CONNECT_TIMEOUT_SECONDS = float(
    os.getenv("ENV_REQUEST_CONNECT_TIMEOUT_SECONDS", "5")
)
ENV_REQUEST_READ_TIMEOUT_SECONDS = float(
    os.getenv("ENV_REQUEST_READ_TIMEOUT_SECONDS", "90")
)
ENV_REQUEST_RETRY_ATTEMPTS = int(os.getenv("ENV_REQUEST_RETRY_ATTEMPTS", "2"))
ENV_STARTUP_WAIT_SECONDS = float(os.getenv("ENV_STARTUP_WAIT_SECONDS", "90"))

ENV_NAME = "chronomigrate-env"
SUCCESS_SCORE_THRESHOLD = 0.9

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


def _format_score(value: float) -> str:
    return f"{normalize_task_score(value):.3f}"


def _emit_start_log(task_id: str) -> None:
    print(
        f"[START] task={task_id} env={ENV_NAME} model={MODEL_NAME}",
        flush=True,
    )


def _emit_step_log(
    *,
    step: int,
    action_payload: Dict[str, object],
    reward: float,
    done: bool,
    error: str | None,
) -> None:
    action_str = json.dumps(action_payload, separators=(",", ":"), ensure_ascii=True)
    done_str = "true" if done else "false"
    error_str = error if error else "null"
    print(
        f"[STEP] step={step} action={action_str} reward={_format_score(reward)} "
        f"done={done_str} error={error_str}",
        flush=True,
    )


def _emit_end_log(
    *,
    task_id: str,
    step_count: int,
    score: float,
    rewards: List[float],
) -> None:
    success_str = "true" if score >= SUCCESS_SCORE_THRESHOLD else "false"
    rewards_str = ",".join(_format_score(reward) for reward in rewards)
    print(
        f"[END] task={task_id} success={success_str} steps={step_count} "
        f"score={_format_score(score)} rewards={rewards_str}",
        flush=True,
    )


def _get_client() -> OpenAI:
    if not API_KEY:
        raise RuntimeError("API_KEY or OPENAI_API_KEY is required.")
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


def _wait_for_env_ready() -> None:
    deadline = time.time() + ENV_STARTUP_WAIT_SECONDS
    last_exc: Exception | None = None

    while time.time() < deadline:
        try:
            response = requests.get(
                f"{BASE_URL}/health",
                timeout=(ENV_REQUEST_CONNECT_TIMEOUT_SECONDS, 5.0),
            )
            response.raise_for_status()
            payload = response.json()
            if payload.get("status") == "healthy":
                return
            last_exc = RuntimeError(f"Unexpected health payload: {payload}")
        except (
            requests_exceptions.ConnectionError,
            requests_exceptions.ReadTimeout,
            requests_exceptions.HTTPError,
            ValueError,
        ) as exc:
            last_exc = exc
        time.sleep(2.0)

    if last_exc is not None:
        raise RuntimeError(
            f"Environment did not become ready at {BASE_URL} within "
            f"{ENV_STARTUP_WAIT_SECONDS:.0f}s: {last_exc}"
        ) from last_exc
    raise RuntimeError(
        f"Environment did not become ready at {BASE_URL} within "
        f"{ENV_STARTUP_WAIT_SECONDS:.0f}s"
    )


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
        cleanup_patterns = (
            r"^DROP TABLE EVENTS_OLD;$",
            r"^DROP TABLE EVENTS_OLD CASCADE;$",
            r"^DROP TABLE IF EXISTS EVENTS_OLD;$",
            r"^DROP TABLE IF EXISTS EVENTS_OLD CASCADE;$",
        )
        return not any(re.fullmatch(pattern, normalized) for pattern in cleanup_patterns)
    return False


def _repeats_failed_sql(sql: str, history: List[Dict[str, str]]) -> bool:
    candidate = _normalize_sql(sql)
    if not candidate or not history:
        return False
    last = history[-1]
    return last.get("result") != "SUCCESS" and candidate == last.get("sql", "")


def _fallback_sql(task_id: str) -> str:
    if task_id == "easy_add_column":
        return "SELECT 1;"
    if task_id == "medium_rename_fk":
        return "SELECT 1;"
    if task_id == "hard_repartition":
        return "SELECT 1;"
    return "SELECT 1;"


def _build_action_payload(task_id: str, sql: str) -> Dict[str, object]:
    normalized_sql = _normalize_sql(sql) or "SELECT 1;"
    if normalized_sql == "SELECT 1;":
        return {"commands": [{"action_type": "wait"}]}
    return {
        "commands": [
            {
                "action_type": "execute_sql",
                "sql": normalized_sql,
                "execute_mode": "transaction",
                "task_id": task_id,
            }
        ]
    }


def run_episode(task_id: str, seed: int = 42) -> float:
    _emit_start_log(task_id)
    _wait_for_env_ready()
    reset_response = _env_post("/reset", {"task_id": task_id, "seed": seed})
    reset_payload = reset_response.json()
    observation = reset_payload.get("observation", reset_payload)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    history: List[Dict[str, str]] = []
    client = _get_client()
    attempt_budget = MAX_STEPS * 3
    attempts = 0
    final_score = normalize_task_score(float(observation.get("schema_match_pct", 0.0)))
    reward_history: List[float] = []
    executed_steps = 0

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
        try:
            sql = _generate_sql(client, messages, seed)
        except Exception:
            sql = _fallback_sql(task_id)
        messages.append({"role": "assistant", "content": sql})

        if _is_obviously_unsafe_sql(sql):
            sql = _fallback_sql(task_id)

        if _repeats_failed_sql(sql, history):
            sql = _fallback_sql(task_id)

        action_payload = _build_action_payload(task_id, sql)
        step_response = _env_post(
            "/step",
            action_payload,
        )
        payload = step_response.json()
        observation = payload.get("observation", {})
        info = payload.get("info", {})
        reward = normalize_task_score(
            float(payload.get("reward", observation.get("reward", 0.0)))
        )
        reward_history.append(reward)
        executed_steps += 1
        final_score = normalize_task_score(
            float(info.get("score", observation.get("schema_match_pct", final_score)))
        )
        history.append(
            {
                "sql": _normalize_sql(sql),
                "result": str(observation.get("last_sql_result", "")),
                "schema_match": str(observation.get("schema_match_pct", 0.0)),
            }
        )
        _emit_step_log(
            step=executed_steps,
            action_payload=action_payload,
            reward=reward,
            done=bool(payload.get("done", False)),
            error=str(info.get("last_error")) if info.get("last_error") else None,
        )
        if payload.get("done", False):
            break

    episode_score = normalize_task_score(
        sum(reward_history) / len(reward_history) if reward_history else 0.0
    )
    _emit_end_log(
        task_id=task_id,
        step_count=executed_steps,
        score=episode_score,
        rewards=reward_history,
    )
    return episode_score


def _safe_run_episode(task_id: str, seed: int = 42) -> float:
    try:
        return run_episode(task_id, seed=seed)
    except Exception as exc:
        score = normalize_task_score(0.0)
        _emit_end_log(task_id=task_id, step_count=0, score=score, rewards=[score])
        return score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-tasks", action="store_true")
    parser.add_argument("--task", default="easy_add_column")
    args = parser.parse_args()

    if args.all_tasks:
        results: Dict[str, float] = {}
        for task_id in ["easy_add_column", "medium_rename_fk", "hard_repartition"]:
            score = _safe_run_episode(task_id, seed=42)
            results[task_id] = score
        print(json.dumps(results))
    else:
        print(json.dumps({args.task: _safe_run_episode(args.task, seed=42)}))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            json.dumps(
                {
                    "easy_add_column": normalize_task_score(0.0),
                    "medium_rename_fk": normalize_task_score(0.0),
                    "hard_repartition": normalize_task_score(0.0),
                }
            )
        )
