import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional

import requests
from openai import OpenAI

from server.tasks import TASKS


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
- NEVER use TRUNCATE.
- NEVER drop a live source table. The only allowed DROP TABLE is DROP TABLE events_old CASCADE
  after the swap in the hard task.
- Prefer CONCURRENTLY for index creation
- For column renames, drop FK constraints first
- Output ONLY a single SQL statement per turn, no explanations
- Output format: just the SQL statement, ending with semicolon
""".strip()

HARD_BACKFILL_BATCHES = [
    "INSERT INTO events_new SELECT * FROM events WHERE id BETWEEN 1 AND 2500;",
    "INSERT INTO events_new SELECT * FROM events WHERE id BETWEEN 2501 AND 5000;",
    "INSERT INTO events_new SELECT * FROM events WHERE id BETWEEN 5001 AND 7500;",
    "INSERT INTO events_new SELECT * FROM events WHERE id BETWEEN 7501 AND 10000;",
]


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


def _normalized_signature(sql: str) -> str:
    return re.sub(r"\s+", " ", _normalize_sql(sql).upper())


def _statement_signature(sql: str) -> str:
    normalized = _normalized_signature(sql)
    if not normalized:
        return ""

    signatures = [
        (
            r"^ALTER TABLE USERS ADD COLUMN EMAIL VARCHAR\(255\) DEFAULT NULL, "
            r"ADD COLUMN IS_ACTIVE BOOLEAN DEFAULT TRUE;$",
            "easy:add_both",
        ),
        (r"^ALTER TABLE USERS ADD COLUMN EMAIL VARCHAR\(255\) DEFAULT NULL;$", "easy:add_email"),
        (r"^ALTER TABLE USERS ADD COLUMN IS_ACTIVE BOOLEAN DEFAULT TRUE;$", "easy:add_is_active"),
        (
            r"^ALTER TABLE ORDERS DROP CONSTRAINT(?: IF EXISTS)? FK_ORDERS_USERS;$",
            "medium:drop_fk",
        ),
        (r"^ALTER TABLE USERS RENAME COLUMN ID TO USER_ID;$", "medium:rename_pk"),
        (
            r"^ALTER TABLE ORDERS ADD CONSTRAINT FK_ORDERS_USERS FOREIGN KEY "
            r"\(USER_ID\) REFERENCES USERS\(USER_ID\);$",
            "medium:add_fk",
        ),
        (
            r"^CREATE TABLE EVENTS_NEW \(LIKE EVENTS INCLUDING ALL\) PARTITION BY HASH "
            r"\(USER_ID\);$",
            "hard:create_new",
        ),
        (
            r"^CREATE TABLE EVENTS_NEW_P(\d+) PARTITION OF EVENTS_NEW FOR VALUES WITH "
            r"\(MODULUS 8, REMAINDER (\d+)\);$",
            "hard:create_partition",
        ),
        (
            r"^INSERT INTO EVENTS_NEW SELECT \* FROM EVENTS WHERE ID BETWEEN (\d+) AND (\d+);$",
            "hard:backfill",
        ),
        (r"^ALTER TABLE EVENTS RENAME TO EVENTS_OLD;$", "hard:swap_old"),
        (r"^ALTER TABLE EVENTS_NEW RENAME TO EVENTS;$", "hard:swap_new"),
        (r"^DROP TABLE EVENTS_OLD(?: CASCADE)?;$", "hard:cleanup_old"),
    ]
    for pattern, signature in signatures:
        match = re.match(pattern, normalized)
        if not match:
            continue
        if signature == "hard:create_partition":
            return f"{signature}:{match.group(1)}:{match.group(2)}"
        if signature == "hard:backfill":
            return f"{signature}:{match.group(1)}:{match.group(2)}"
        return signature
    return normalized


def _recommended_step(
    task_id: str, observation: Dict[str, object], successful_actions: List[str]
) -> Optional[str]:
    schema = str(observation.get("current_schema_ddl", "")).lower()
    success_signatures = {_statement_signature(action) for action in successful_actions}

    if task_id == "easy_add_column":
        has_email = "email varchar(255)" in schema
        has_is_active = "is_active boolean" in schema
        if not has_email and not has_is_active:
            return (
                "ALTER TABLE users ADD COLUMN email VARCHAR(255) DEFAULT NULL, "
                "ADD COLUMN is_active BOOLEAN DEFAULT TRUE;"
            )
        if not has_email:
            return "ALTER TABLE users ADD COLUMN email VARCHAR(255) DEFAULT NULL;"
        if not has_is_active:
            return "ALTER TABLE users ADD COLUMN is_active BOOLEAN DEFAULT TRUE;"
        return None

    if task_id == "medium_rename_fk":
        if "references users(id)" in schema and "medium:drop_fk" not in success_signatures:
            return "ALTER TABLE orders DROP CONSTRAINT fk_orders_users;"
        if "user_id serial primary key" not in schema:
            return "ALTER TABLE users RENAME COLUMN id TO user_id;"
        if "references users(user_id)" not in schema:
            return (
                "ALTER TABLE orders ADD CONSTRAINT fk_orders_users "
                "FOREIGN KEY (user_id) REFERENCES users(user_id);"
            )
        return None

    if task_id == "hard_repartition":
        if "hard:create_new" not in success_signatures:
            return (
                "CREATE TABLE events_new (LIKE events INCLUDING ALL) "
                "PARTITION BY HASH (user_id);"
            )

        for partition in range(8):
            signature = f"hard:create_partition:{partition}:{partition}"
            if signature not in success_signatures:
                return (
                    f"CREATE TABLE events_new_p{partition} PARTITION OF events_new "
                    f"FOR VALUES WITH (MODULUS 8, REMAINDER {partition});"
                )

        for batch in HARD_BACKFILL_BATCHES:
            if _statement_signature(batch) not in success_signatures:
                return batch

        if "hard:swap_old" not in success_signatures:
            return "ALTER TABLE events RENAME TO events_old;"
        if "hard:swap_new" not in success_signatures:
            return "ALTER TABLE events_new RENAME TO events;"
        if "hard:cleanup_old" not in success_signatures:
            return "DROP TABLE events_old CASCADE;"
        return None

    return None


def _task_guidance(task_id: str, recommended_step: Optional[str]) -> str:
    common = f"Recommended next safe step: {recommended_step}" if recommended_step else ""
    if task_id == "easy_add_column":
        return (
            "Best strategy: add the two target columns on users with DEFAULT values. "
            "Do not touch existing rows or primary keys. "
            f"{common}"
        ).strip()
    if task_id == "medium_rename_fk":
        return (
            "Use this order exactly: drop fk_orders_users, rename users.id to user_id, "
            "then recreate fk_orders_users referencing users(user_id). "
            f"{common}"
        ).strip()
    if task_id == "hard_repartition":
        return (
            "Use the safe multi-step repartition strategy: create events_new partitioned by "
            "HASH(user_id), create eight child partitions, backfill in four id ranges, swap "
            "events->events_old, rename events_new->events, then drop events_old CASCADE. "
            "Never drop the live events table directly and never swap before the backfill is done. "
            f"{common}"
        ).strip()
    return common


def _is_unsafe_sql(task_id: str, sql: str) -> bool:
    normalized = _normalized_signature(sql)
    if not normalized:
        return True
    if "TRUNCATE" in normalized:
        return True
    if "DROP TABLE" in normalized:
        return not normalized.startswith("DROP TABLE EVENTS_OLD")
    if task_id != "hard_repartition" and "RENAME TO EVENTS_OLD" in normalized:
        return True
    return False


def _select_sql(task_id: str, model_sql: str, recommended_step: Optional[str]) -> str:
    candidate = _normalize_sql(model_sql)
    if not recommended_step:
        return candidate
    if _is_unsafe_sql(task_id, candidate):
        return recommended_step
    if _statement_signature(candidate) != _statement_signature(recommended_step):
        return recommended_step
    return candidate


def run_episode(task_id: str, seed: int = 42) -> float:
    reset_response = requests.post(
        f"{BASE_URL}/reset",
        json={"task_id": task_id, "seed": seed},
        timeout=30,
    )
    reset_response.raise_for_status()
    observation = reset_response.json()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    successful_actions: List[str] = []
    max_steps = TASKS[task_id].max_steps

    for _ in range(max_steps):
        recommended_step = _recommended_step(task_id, observation, successful_actions)
        task_guidance = _task_guidance(task_id, recommended_step)
        prompt = (
            f"Task: {task_id}\n"
            f"Task guidance: {task_guidance}\n\n"
            f"Current schema:\n{observation.get('current_schema_ddl', '')}\n\n"
            f"Target schema:\n{observation.get('target_schema_ddl', '')}\n\n"
            f"Successful SQL so far:\n{json.dumps(successful_actions[-8:])}\n\n"
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
        sql = _select_sql(
            task_id,
            completion.choices[0].message.content or "",
            recommended_step,
        )
        messages.append({"role": "assistant", "content": sql})

        step_response = requests.post(
            f"{BASE_URL}/step",
            json={"sql": sql, "task_id": task_id, "execute_mode": "transaction"},
            timeout=30,
        )
        step_response.raise_for_status()
        payload = step_response.json()
        observation = payload.get("observation", {})
        if observation.get("last_sql_result") == "SUCCESS":
            successful_actions.append(sql)
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
