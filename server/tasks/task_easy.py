from server.schema_grader import compute_schema_match
from server.tasks import TaskDefinition


STARTING_SCHEMA = """
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);
"""

TARGET_SCHEMA = """
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    email VARCHAR(255) DEFAULT NULL,
    is_active BOOLEAN DEFAULT TRUE
);
"""

SEED_DATA = """
INSERT INTO users (id, username, created_at) VALUES
(1, 'user_001', '2025-01-01 00:00:00'),
(2, 'user_002', '2025-01-01 00:00:01'),
(3, 'user_003', '2025-01-01 00:00:02'),
(4, 'user_004', '2025-01-01 00:00:03'),
(5, 'user_005', '2025-01-01 00:00:04');
"""


def grade_easy(
    current_schema_ddl: str,
    target_schema_ddl: str,
    data_hash_before: str,
    data_hash_after: str,
    availability_pct: float,
) -> float:
    schema_match = compute_schema_match(current_schema_ddl, target_schema_ddl)
    data_integrity = 1.0 if data_hash_before == data_hash_after else 0.0
    return min(1.0, schema_match * data_integrity * availability_pct)


TASK = TaskDefinition(
    task_id="easy_add_column",
    description="Add two defaulted columns without causing downtime.",
    difficulty="easy",
    load_level=100,
    max_steps=8,
    starting_schema_sql=STARTING_SCHEMA.strip(),
    target_schema_ddl=TARGET_SCHEMA.strip(),
    seed_data_sql=SEED_DATA.strip(),
    grade_fn=grade_easy,
)
