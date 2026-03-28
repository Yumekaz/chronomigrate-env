import json
import re
from datetime import datetime, timedelta

from server.tasks import TaskDefinition


STARTING_SCHEMA = """
CREATE TABLE events (
    id BIGSERIAL,
    user_id INTEGER NOT NULL,
    event_type VARCHAR(50),
    payload JSONB,
    created_at TIMESTAMP NOT NULL
) PARTITION BY RANGE (created_at);

CREATE TABLE events_2025_q1 PARTITION OF events
FOR VALUES FROM ('2025-01-01') TO ('2025-04-01');

CREATE TABLE events_2025_q2 PARTITION OF events
FOR VALUES FROM ('2025-04-01') TO ('2025-07-01');
"""

TARGET_SCHEMA = """
CREATE TABLE events (
    id BIGSERIAL,
    user_id INTEGER NOT NULL,
    event_type VARCHAR(50),
    payload JSONB,
    created_at TIMESTAMP NOT NULL
) PARTITION BY HASH (user_id);

CREATE TABLE events_p0 PARTITION OF events FOR VALUES WITH (MODULUS 8, REMAINDER 0);
CREATE TABLE events_p1 PARTITION OF events FOR VALUES WITH (MODULUS 8, REMAINDER 1);
CREATE TABLE events_p2 PARTITION OF events FOR VALUES WITH (MODULUS 8, REMAINDER 2);
CREATE TABLE events_p3 PARTITION OF events FOR VALUES WITH (MODULUS 8, REMAINDER 3);
CREATE TABLE events_p4 PARTITION OF events FOR VALUES WITH (MODULUS 8, REMAINDER 4);
CREATE TABLE events_p5 PARTITION OF events FOR VALUES WITH (MODULUS 8, REMAINDER 5);
CREATE TABLE events_p6 PARTITION OF events FOR VALUES WITH (MODULUS 8, REMAINDER 6);
CREATE TABLE events_p7 PARTITION OF events FOR VALUES WITH (MODULUS 8, REMAINDER 7);
"""

def _format_insert_batches(table: str, columns: str, rows: list[str], batch_size: int = 400) -> str:
    statements = []
    for index in range(0, len(rows), batch_size):
        batch = rows[index : index + batch_size]
        statements.append(
            f"INSERT INTO {table} ({columns}) VALUES\n" + ",\n".join(batch) + ";"
        )
    return "\n\n".join(statements)


def _escape_sql_text(value: str) -> str:
    return value.replace("'", "''")


def _build_seed_data() -> str:
    base = datetime(2025, 1, 1, 0, 0, 0)
    event_types = ["login", "purchase", "logout", "view", "click", "checkout"]
    rows = []
    for index in range(1, 10001):
        user_id = ((index - 1) % 2048) + 1
        event_type = event_types[(index - 1) % len(event_types)]
        created_at = base + timedelta(minutes=index - 1)
        payload = json.dumps(
            {
                "batch": (index - 1) // 500,
                "kind": event_type,
                "seq": index,
            },
            separators=(",", ":"),
        )
        rows.append(
            f"({index}, {user_id}, '{event_type}', '{_escape_sql_text(payload)}', "
            f"'{created_at:%Y-%m-%d %H:%M:%S}')"
        )
    return _format_insert_batches(
        "events",
        "id, user_id, event_type, payload, created_at",
        rows,
        batch_size=500,
    )


SEED_DATA = _build_seed_data()


def grade_hard(
    current_schema_ddl: str,
    target_schema_ddl: str,
    data_hash_before: str,
    data_hash_after: str,
    availability_pct: float,
    action_history=None,
    steps_used: int = 0,
    **_: object,
) -> float:
    from server.schema_grader import compute_schema_match

    schema_match = compute_schema_match(current_schema_ddl, target_schema_ddl)
    data_integrity = 1.0 if data_hash_before == data_hash_after else 0.0
    action_history = action_history or []

    current_tables = {
        match.group(1).lower()
        for match in re.finditer(r"CREATE TABLE\s+(\w+)\b", current_schema_ddl, re.IGNORECASE)
    }
    target_tables = {
        match.group(1).lower()
        for match in re.finditer(r"CREATE TABLE\s+(\w+)\b", target_schema_ddl, re.IGNORECASE)
    }
    target_partitions = {
        match.group(1)
        for match in re.finditer(r"CREATE TABLE\s+events_p(\d+)\b", target_schema_ddl, re.IGNORECASE)
    }
    current_partitions = {
        match.group(1)
        for match in re.finditer(r"CREATE TABLE\s+events(?:_new)?_p(\d+)\b", current_schema_ddl, re.IGNORECASE)
    }

    table_alignment = (
        len(current_tables & target_tables) / len(target_tables) if target_tables else 1.0
    )
    partition_alignment = (
        len(current_partitions & target_partitions) / len(target_partitions)
        if target_partitions
        else 1.0
    )
    partition_mode = 1.0 if re.search(r"PARTITION BY\s+HASH\s*\(\s*user_id\s*\)", current_schema_ddl, re.IGNORECASE) else 0.0

    has_create_alongside = any("CREATE TABLE EVENTS_NEW" in action.upper() for action in action_history)
    created_partitions = sum("PARTITION OF EVENTS_NEW" in action.upper() for action in action_history)
    backfill_batches = sum(
        "INSERT INTO EVENTS_NEW SELECT * FROM EVENTS" in action.upper() and "BETWEEN" in action.upper()
        for action in action_history
    )
    atomic_swap = any("ALTER TABLE EVENTS RENAME TO EVENTS_OLD" in action.upper() for action in action_history) and any(
        "ALTER TABLE EVENTS_NEW RENAME TO EVENTS" in action.upper() for action in action_history
    )
    cleanup_old = any("DROP TABLE EVENTS_OLD" in action.upper() for action in action_history)
    safe_pattern_score = 0.0
    safe_pattern_score += 0.1 if has_create_alongside else 0.0
    safe_pattern_score += min(0.2, created_partitions / 8 * 0.2)
    safe_pattern_score += min(0.2, backfill_batches / 4 * 0.2)
    safe_pattern_score += 0.1 if atomic_swap else 0.0
    safe_pattern_score += 0.05 if cleanup_old else 0.0
    safe_pattern_score += 0.05 if steps_used >= 8 else 0.0

    structural_score = (table_alignment + partition_alignment + partition_mode) / 3.0
    strategy_quality = min(1.0, 0.35 + 0.30 * structural_score + safe_pattern_score)
    return min(1.0, schema_match * data_integrity * availability_pct * strategy_quality)


TASK = TaskDefinition(
    task_id="hard_repartition",
    description="Repartition a large table under load using a safe multi-step migration.",
    difficulty="hard",
    load_level=500,
    max_steps=20,
    starting_schema_sql=STARTING_SCHEMA.strip(),
    target_schema_ddl=TARGET_SCHEMA.strip(),
    seed_data_sql=SEED_DATA.strip(),
    grade_fn=grade_hard,
)
