import json
from datetime import datetime, timedelta

from server.schema_grader import compute_schema_match
from server.tasks import TaskDefinition, coerce_grader_inputs, normalize_task_score


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
    for index in range(1, 2001):
        user_id = ((index - 1) % 512) + 1
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
    **_: object,
) -> float:
    schema_match = compute_schema_match(current_schema_ddl, target_schema_ddl)
    data_integrity = 1.0 if data_hash_before == data_hash_after else 0.0
    return normalize_task_score(schema_match * data_integrity * availability_pct)


def hard_grader(*args: object, **kwargs: object) -> float:
    payload, is_complete = coerce_grader_inputs(*args, **kwargs)
    if not is_complete:
        return normalize_task_score(0.0)
    return grade_hard(**payload)


hard_grader.grade = hard_grader


class HardGrader(float):
    def __new__(cls, *args: object, **kwargs: object):
        value = hard_grader(*args, **kwargs) if (args or kwargs) else hard_grader()
        return float.__new__(cls, value)

    @staticmethod
    def grade(*args: object, **kwargs: object) -> float:
        return hard_grader(*args, **kwargs)

    def __call__(self, *args: object, **kwargs: object) -> float:
        return self.grade(*args, **kwargs)


TASK = TaskDefinition(
    task_id="hard_repartition",
    description="Repartition a large table under load using a safe multi-step migration.",
    difficulty="hard",
    load_level=160,
    max_steps=20,
    starting_schema_sql=STARTING_SCHEMA.strip(),
    target_schema_ddl=TARGET_SCHEMA.strip(),
    seed_data_sql=SEED_DATA.strip(),
    grade_fn=hard_grader,
)
