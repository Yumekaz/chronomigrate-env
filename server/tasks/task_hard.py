from server.schema_grader import compute_schema_match
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

SEED_DATA = """
INSERT INTO events (id, user_id, event_type, payload, created_at) VALUES
(1, 10, 'login', '{"ip":"127.0.0.1"}', '2025-01-02 08:00:00'),
(2, 11, 'purchase', '{"item":"book"}', '2025-02-10 09:00:00'),
(3, 12, 'logout', '{"source":"web"}', '2025-04-11 10:00:00');
"""


def grade_hard(
    current_schema_ddl: str,
    target_schema_ddl: str,
    data_hash_before: str,
    data_hash_after: str,
    availability_pct: float,
) -> float:
    schema_match = compute_schema_match(current_schema_ddl, target_schema_ddl)
    data_integrity = 1.0 if data_hash_before == data_hash_after else 0.0
    migration_bonus = 0.15 if "events_old" in current_schema_ddl or "events_new" in current_schema_ddl else 0.0
    return min(1.0, schema_match * data_integrity * availability_pct + migration_bonus)


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
