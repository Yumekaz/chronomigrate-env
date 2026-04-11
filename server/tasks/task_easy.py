from datetime import datetime, timedelta

from server.schema_grader import compute_schema_match
from server.tasks import TaskDefinition, coerce_grader_inputs, normalize_task_score


STARTING_SCHEMA = """
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT '2024-01-01 00:00:00'
);
"""

TARGET_SCHEMA = """
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT '2024-01-01 00:00:00',
    email VARCHAR(255) DEFAULT NULL,
    is_active BOOLEAN DEFAULT TRUE
);
"""

def _format_insert_batches(table: str, columns: str, rows: list[str], batch_size: int = 200) -> str:
    statements = []
    for index in range(0, len(rows), batch_size):
        batch = rows[index : index + batch_size]
        statements.append(
            f"INSERT INTO {table} ({columns}) VALUES\n" + ",\n".join(batch) + ";"
        )
    return "\n\n".join(statements)


def _build_seed_data() -> str:
    base = datetime(2025, 1, 1, 0, 0, 0)
    rows = []
    for index in range(1, 1001):
        created_at = base + timedelta(seconds=index - 1)
        rows.append(
            f"({index}, 'user_{index:04d}', '{created_at:%Y-%m-%d %H:%M:%S}')"
        )
    return _format_insert_batches("users", "id, username, created_at", rows, batch_size=250)


SEED_DATA = _build_seed_data()


def grade_easy(
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


def easy_grader(*args: object, **kwargs: object) -> float:
    payload, is_complete = coerce_grader_inputs(*args, **kwargs)
    if not is_complete:
        return normalize_task_score(0.0)
    return grade_easy(**payload)


easy_grader.grade = easy_grader


class EasyGrader(float):
    def __new__(cls, *args: object, **kwargs: object):
        value = easy_grader(*args, **kwargs) if (args or kwargs) else easy_grader()
        return float.__new__(cls, value)

    @staticmethod
    def grade(*args: object, **kwargs: object) -> float:
        return easy_grader(*args, **kwargs)

    def __call__(self, *args: object, **kwargs: object) -> float:
        return self.grade(*args, **kwargs)


TASK = TaskDefinition(
    task_id="easy_add_column",
    description="Add two defaulted columns without causing downtime.",
    difficulty="easy",
    load_level=100,
    max_steps=5,
    starting_schema_sql=STARTING_SCHEMA.strip(),
    target_schema_ddl=TARGET_SCHEMA.strip(),
    seed_data_sql=SEED_DATA.strip(),
    grade_fn=easy_grader,
)
