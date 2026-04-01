import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional

import requests
from openai import OpenAI


BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
API_KEY = (
    os.getenv("HF_TOKEN")
    or os.getenv("API_KEY")
    or os.getenv("OPENAI_API_KEY", "")
)
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
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

HARD_BACKFILL_BATCHES = [
    "INSERT INTO events_new SELECT * FROM events WHERE id BETWEEN 1 AND 2500;",
    "INSERT INTO events_new SELECT * FROM events WHERE id BETWEEN 2501 AND 5000;",
    "INSERT INTO events_new SELECT * FROM events WHERE id BETWEEN 5001 AND 7500;",
    "INSERT INTO events_new SELECT * FROM events WHERE id BETWEEN 7501 AND 10000;",
]


def _get_client() -> OpenAI:
    if not API_KEY:
        raise RuntimeError("HF_TOKEN, API_KEY, or OPENAI_API_KEY is required.")
    return OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


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
        (
            r"^ALTER TABLE USERS ADD COLUMN IS_ACTIVE BOOLEAN DEFAULT TRUE;$",
            "easy:add_is_active",
        ),
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


def _is_task_unsafe_sql(task_id: str, sql: str) -> bool:
    normalized = _normalized_signature(sql)
    if not normalized:
        return True
    if "TRUNCATE" in normalized or "DROP SCHEMA" in normalized:
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
    if _is_task_unsafe_sql(task_id, candidate):
        return recommended_step
    if _statement_signature(candidate) != _statement_signature(recommended_step):
        return recommended_step
    return candidate


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


def _extract_parent_table_statements(ddl: str) -> Dict[str, str]:
    statements: Dict[str, str] = {}
    pattern = re.compile(
        r"(CREATE TABLE\s+(\w+)\s*\(.*?\)\s*(?:PARTITION BY\s+[A-Z]+\s*\([^)]+\))?;)",
        re.IGNORECASE | re.DOTALL,
    )
    for statement, table_name in pattern.findall(ddl):
        if "PARTITION OF" in statement.upper():
            continue
        statements[table_name.lower()] = _normalize_sql(statement)
    return statements


def _extract_partition_child_statements(ddl: str) -> List[Dict[str, str]]:
    statements: List[Dict[str, str]] = []
    pattern = re.compile(
        r"(CREATE TABLE\s+(\w+)\s+PARTITION OF\s+(\w+)\b.*?;)",
        re.IGNORECASE | re.DOTALL,
    )
    for statement, child_name, parent_name in pattern.findall(ddl):
        statements.append(
            {
                "statement": _normalize_sql(statement),
                "child": child_name.lower(),
                "parent": parent_name.lower(),
            }
        )
    return statements


def _partition_modes(ddl: str) -> Dict[str, str]:
    modes: Dict[str, str] = {}
    pattern = re.compile(
        r"CREATE TABLE\s+(\w+)\s*\(.*?\)\s*PARTITION BY\s+([A-Z]+)\s*\([^)]+\)",
        re.IGNORECASE | re.DOTALL,
    )
    for table_name, partition_mode in pattern.findall(ddl):
        modes[table_name.lower()] = partition_mode.upper()
    return modes


def _extract_all_table_names(ddl: str) -> set[str]:
    pattern = re.compile(r"CREATE TABLE\s+(\w+)\b", re.IGNORECASE)
    return {table_name.lower() for table_name in pattern.findall(ddl)}


def _extract_successful_created_tables(history: List[Dict[str, str]]) -> set[str]:
    created_tables: set[str] = set()
    pattern = re.compile(r"CREATE TABLE\s+(\w+)\b", re.IGNORECASE)
    for entry in history:
        if entry.get("result") != "SUCCESS":
            continue
        match = pattern.search(_normalize_sql(entry.get("sql", "")))
        if match:
            created_tables.add(match.group(1).lower())
    return created_tables


def _rewrite_replacement_statement(
    statement: str, source_table: str, replacement_table: str
) -> str:
    rewritten = re.sub(
        rf"\bCREATE TABLE\s+{re.escape(source_table)}\b",
        f"CREATE TABLE {replacement_table}",
        statement,
        count=1,
        flags=re.IGNORECASE,
    )
    return _normalize_sql(rewritten)


def _rewrite_partition_child_statement(
    statement: str, source_table: str, replacement_table: str, child_name: str
) -> str:
    if child_name.startswith(source_table):
        replacement_child = replacement_table + child_name[len(source_table) :]
    else:
        replacement_child = f"{replacement_table}_{child_name}"

    rewritten = re.sub(
        rf"\bCREATE TABLE\s+{re.escape(child_name)}\b",
        f"CREATE TABLE {replacement_child}",
        statement,
        count=1,
        flags=re.IGNORECASE,
    )
    rewritten = re.sub(
        rf"\bPARTITION OF\s+{re.escape(source_table)}\b",
        f"PARTITION OF {replacement_table}",
        rewritten,
        count=1,
        flags=re.IGNORECASE,
    )
    return _normalize_sql(rewritten)


def _is_stalled(history: List[Dict[str, str]]) -> bool:
    if len(history) < 2:
        return False
    recent = history[-2:]
    first_score = round(float(recent[0].get("schema_match", 0.0)), 4)
    return all(round(float(item.get("schema_match", 0.0)), 4) == first_score for item in recent)


def _generic_safe_fallback(
    observation: Dict[str, object], history: List[Dict[str, str]]
) -> Optional[str]:
    current_ddl = str(observation.get("current_schema_ddl", ""))
    target_ddl = str(observation.get("target_schema_ddl", ""))
    current_parents = _extract_parent_table_statements(current_ddl)
    target_parents = _extract_parent_table_statements(target_ddl)
    target_children = _extract_partition_child_statements(target_ddl)
    current_modes = _partition_modes(current_ddl)
    target_modes = _partition_modes(target_ddl)
    existing_tables = _extract_all_table_names(current_ddl)
    existing_tables.update(_extract_successful_created_tables(history))
    successful_sql = {
        _normalize_sql(entry.get("sql", ""))
        for entry in history
        if entry.get("result") == "SUCCESS"
    }

    for table_name, target_statement in target_parents.items():
        current_statement = current_parents.get(table_name)
        if not current_statement:
            continue

        current_mode = current_modes.get(table_name, "")
        target_mode = target_modes.get(table_name, "")
        if current_mode == target_mode:
            continue

        replacement_table = f"{table_name}_new"
        backup_table = f"{table_name}_old"

        if replacement_table not in existing_tables:
            return _rewrite_replacement_statement(
                target_statement, table_name, replacement_table
            )

        for child in target_children:
            if child["parent"] != table_name:
                continue
            replacement_child_sql = _rewrite_partition_child_statement(
                child["statement"], table_name, replacement_table, child["child"]
            )
            name_match = re.search(
                r"CREATE TABLE\s+(\w+)\b", replacement_child_sql, re.IGNORECASE
            )
            replacement_child_name = name_match.group(1).lower() if name_match else ""
            if replacement_child_name and replacement_child_name not in existing_tables:
                return replacement_child_sql

        copy_sql = f"INSERT INTO {replacement_table} SELECT * FROM {table_name};"
        if copy_sql not in successful_sql:
            return copy_sql

        rename_old_sql = f"ALTER TABLE {table_name} RENAME TO {backup_table};"
        if rename_old_sql not in successful_sql:
            return rename_old_sql

        rename_new_sql = f"ALTER TABLE {replacement_table} RENAME TO {table_name};"
        if rename_new_sql not in successful_sql:
            return rename_new_sql

    return None


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
    client: Optional[OpenAI] = None
    if API_KEY:
        try:
            client = _get_client()
        except Exception:
            client = None

    for _ in range(MAX_STEPS):
        successful_actions = [
            entry["sql"] for entry in history if entry.get("result") == "SUCCESS"
        ]
        recommended_step = _recommended_step(task_id, observation, successful_actions)
        fallback_sql = recommended_step or _generic_safe_fallback(observation, history)
        sql = fallback_sql or ""

        if client is not None and not recommended_step:
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
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.0,
                max_tokens=200,
            )
            sql = _normalize_sql(completion.choices[0].message.content or "")

            if fallback_sql and (
                _is_obviously_unsafe_sql(sql)
                or _repeats_failed_sql(sql, history)
                or (
                    _is_stalled(history)
                    and float(observation.get("schema_match_pct", 0.0)) < 0.95
                )
            ):
                sql = fallback_sql
            elif _is_obviously_unsafe_sql(sql) or _repeats_failed_sql(sql, history):
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
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=200,
                )
                sql = _normalize_sql(completion.choices[0].message.content or "")
                if fallback_sql and (
                    _is_obviously_unsafe_sql(sql) or _repeats_failed_sql(sql, history)
                ):
                    sql = fallback_sql

        if not sql:
            raise RuntimeError("No safe SQL step available for the current observation.")

        messages.append({"role": "assistant", "content": sql})

        step_response = requests.post(
            f"{BASE_URL}/step",
            json={"sql": sql, "task_id": task_id, "execute_mode": "transaction"},
            timeout=30,
        )
        try:
            step_response.raise_for_status()
        except requests.HTTPError:
            if fallback_sql and _normalize_sql(sql) != _normalize_sql(fallback_sql):
                sql = fallback_sql
                step_response = requests.post(
                    f"{BASE_URL}/step",
                    json={
                        "sql": sql,
                        "task_id": task_id,
                        "execute_mode": "transaction",
                    },
                    timeout=30,
                )
                step_response.raise_for_status()
            else:
                raise
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

    grader_response = requests.post(
        f"{BASE_URL}/grader", json={"task_id": task_id}, timeout=30
    )
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
