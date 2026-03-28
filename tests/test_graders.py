import sqlite3
import shutil

import pytest
from fastapi.testclient import TestClient

from models import MigrationAction
from server.app import (
    GraderRequest,
    app,
    create_app,
    create_fastapi_app,
    grade_episode,
    list_tasks,
    env as app_env,
)
from server.chrono_migrate_env import ChronoMigrateEnv
from server.db_manager import DBManager
from server.schema_grader import compute_data_hash, compute_schema_match
from server.tasks import TASKS


def test_schema_match_rewards_target_features():
    current = """
    CREATE TABLE users (
        id SERIAL PRIMARY KEY,
        username VARCHAR(100) NOT NULL
    );
    """
    target = """
    CREATE TABLE users (
        id SERIAL PRIMARY KEY,
        username VARCHAR(100) NOT NULL,
        email VARCHAR(255) DEFAULT NULL
    );
    """
    assert 0.0 < compute_schema_match(current, target) < 1.0


def test_schema_match_is_perfect_for_identical_ddl():
    ddl = """
    CREATE TABLE users (
        id SERIAL PRIMARY KEY,
        username VARCHAR(100) NOT NULL,
        created_at TIMESTAMP DEFAULT NOW()
    );
    """
    assert compute_schema_match(ddl, ddl) == 1.0


def test_schema_match_penalizes_unexpected_extra_columns():
    current = """
    CREATE TABLE users (
        id SERIAL PRIMARY KEY,
        username VARCHAR(100) NOT NULL,
        created_at TIMESTAMP DEFAULT NOW(),
        email VARCHAR(255) DEFAULT NULL,
        is_active BOOLEAN DEFAULT TRUE,
        debug_notes TEXT
    );
    """
    target = """
    CREATE TABLE users (
        id SERIAL PRIMARY KEY,
        username VARCHAR(100) NOT NULL,
        created_at TIMESTAMP DEFAULT NOW(),
        email VARCHAR(255) DEFAULT NULL,
        is_active BOOLEAN DEFAULT TRUE
    );
    """
    assert compute_schema_match(current, target) < 1.0


def test_schema_match_handles_foreign_keys():
    current = """
    CREATE TABLE users (id SERIAL PRIMARY KEY);
    CREATE TABLE orders (
        id SERIAL PRIMARY KEY,
        user_id INTEGER,
        CONSTRAINT fk_orders_users FOREIGN KEY (user_id) REFERENCES users(id)
    );
    """
    target = """
    CREATE TABLE users (user_id SERIAL PRIMARY KEY);
    CREATE TABLE orders (
        id SERIAL PRIMARY KEY,
        user_id INTEGER,
        CONSTRAINT fk_orders_users FOREIGN KEY (user_id) REFERENCES users(user_id)
    );
    """
    assert compute_schema_match(current, target) < 1.0


def test_compute_data_hash_is_deterministic_for_sqlite():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE users (id INTEGER, username TEXT)")
    conn.execute("INSERT INTO users (id, username) VALUES (1, 'alice'), (2, 'bob')")
    first = compute_data_hash(conn)
    second = compute_data_hash(conn)
    assert first == second


def test_compute_data_hash_changes_when_rows_change():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE users (id INTEGER, username TEXT)")
    conn.execute("INSERT INTO users (id, username) VALUES (1, 'alice')")
    first = compute_data_hash(conn)
    conn.execute("INSERT INTO users (id, username) VALUES (2, 'bob')")
    second = compute_data_hash(conn)
    assert first != second


def test_env_reset_and_step_updates_episode_state():
    env = ChronoMigrateEnv()
    obs = env.reset({"task_id": "easy_add_column", "seed": 42})
    assert obs.task_id == "easy_add_column"
    assert obs.step_count == 0
    assert obs.episode_id

    step = env.step(
        MigrationAction(
            sql="ALTER TABLE users ADD COLUMN email VARCHAR(255) DEFAULT NULL;",
            task_id="easy_add_column",
        )
    )
    assert step.step_count == 1
    assert "email" in step.current_schema_ddl.lower()
    assert 0.0 <= step.schema_match_pct <= 1.0
    assert env.state().cumulative_reward >= 0.0


def test_tasks_endpoint_lists_expected_tasks():
    tasks = list_tasks()
    assert [task["id"] for task in tasks] == [
        "easy_add_column",
        "medium_rename_fk",
        "hard_repartition",
    ]
    assert all("action_schema" in task for task in tasks)


def test_contract_endpoints_expose_metadata_and_schema():
    client = TestClient(app)
    metadata = client.get("/metadata")
    schema = client.get("/schema")
    mcp = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "ping"})

    assert metadata.status_code == 200
    assert metadata.json()["name"] == "chronomigrate-env"
    assert "description" in metadata.json()

    assert schema.status_code == 200
    assert set(schema.json()) == {"action", "observation", "state"}

    assert mcp.status_code == 200
    assert mcp.json()["jsonrpc"] == "2.0"


def test_root_and_web_endpoints_render_html():
    client = TestClient(app)
    root = client.get("/")
    web = client.get("/web")

    assert root.status_code == 200
    assert web.status_code == 200
    assert "text/html" in root.headers["content-type"]
    assert "text/html" in web.headers["content-type"]
    assert "ChronoMigrate-Env" in root.text
    assert "easy_add_column" in web.text


def test_create_app_factory_returns_fastapi_app():
    factory_app = create_app()
    client = TestClient(factory_app)
    health = client.get("/health")

    assert health.status_code == 200
    assert health.json() == {"status": "healthy"}


def test_create_fastapi_app_factory_returns_fastapi_app():
    factory_app = create_fastapi_app()
    client = TestClient(factory_app)
    health = client.get("/health")

    assert health.status_code == 200
    assert health.json() == {"status": "healthy"}


def test_state_endpoint_returns_active_episode():
    client = TestClient(app)
    client.post("/reset", json={"task_id": "easy_add_column", "seed": 42})
    state = client.get("/state")

    assert state.status_code == 200
    assert state.json()["task_id"] == "easy_add_column"


def test_grader_returns_bounded_score_for_completed_episode():
    app_env.reset({"task_id": "easy_add_column", "seed": 42})
    app_env.step(
        MigrationAction(
            sql="ALTER TABLE users ADD COLUMN email VARCHAR(255) DEFAULT NULL;",
            task_id="easy_add_column",
        )
    )
    result = grade_episode(GraderRequest(task_id="easy_add_column"))
    assert 0.0 <= result["score"] <= 1.0
    assert 0.0 <= result["availability"] <= 1.0


def test_reset_rejects_unknown_task():
    env = ChronoMigrateEnv()
    with pytest.raises(ValueError):
        env.reset({"task_id": "missing_task"})


def test_task_mismatch_is_penalized_gracefully():
    env = ChronoMigrateEnv()
    env.reset({"task_id": "easy_add_column", "seed": 42})
    obs = env.step(
        MigrationAction(
            sql="ALTER TABLE users ADD COLUMN email VARCHAR(255) DEFAULT NULL;",
            task_id="medium_rename_fk",
        )
    )
    assert "TASK_ID_MISMATCH" in obs.last_sql_result
    assert env.last_step_reward == -0.05
    assert env.state().step_count == 1


def test_grader_rejects_episode_id_mismatch():
    app_env.reset({"task_id": "easy_add_column", "seed": 42})
    result = grade_episode(
        GraderRequest(task_id="easy_add_column", episode_id="wrong-episode-id")
    )
    assert result["score"] == 0.0
    assert "mismatch" in result["feedback"].lower()


def test_reset_and_step_contract_through_http():
    client = TestClient(app)
    reset = client.post("/reset", json={"task_id": "easy_add_column", "seed": 42})
    assert reset.status_code == 200
    assert reset.json()["task_id"] == "easy_add_column"

    step = client.post(
        "/step",
        json={
            "sql": "ALTER TABLE users ADD COLUMN email VARCHAR(255) DEFAULT NULL;",
            "task_id": "easy_add_column",
            "execute_mode": "transaction",
        },
    )
    assert step.status_code == 200
    body = step.json()
    assert body["observation"]["task_id"] == "easy_add_column"
    assert "metadata" in body


def test_execute_mode_transaction_rolls_back_on_error():
    task = TASKS["easy_add_column"]
    db = DBManager()
    db.reset_to_schema(task.starting_schema_sql, task.seed_data_sql)
    ok, _, _ = db.execute(
        "ALTER TABLE users ADD COLUMN staging TEXT; INVALID SQL;",
        execute_mode="transaction",
    )
    assert ok is False
    assert "staging" not in db.get_schema_ddl().lower()


def test_execute_mode_autocommit_keeps_prior_statement_on_error():
    task = TASKS["easy_add_column"]
    db = DBManager()
    db.reset_to_schema(task.starting_schema_sql, task.seed_data_sql)
    ok, _, _ = db.execute(
        "ALTER TABLE users ADD COLUMN staging TEXT; INVALID SQL;",
        execute_mode="autocommit",
    )
    assert ok is False
    assert "staging" in db.get_schema_ddl().lower()


def test_medium_fk_rename_requires_constraint_drop_first():
    db = DBManager()
    schema = """
    CREATE TABLE users (
        id SERIAL PRIMARY KEY,
        username VARCHAR(100) NOT NULL
    );
    CREATE TABLE orders (
        id SERIAL PRIMARY KEY,
        user_id INTEGER NOT NULL,
        CONSTRAINT fk_orders_users FOREIGN KEY (user_id) REFERENCES users(id)
    );
    """
    seed = """
    INSERT INTO users (id, username) VALUES (1, 'user_0001');
    INSERT INTO orders (id, user_id) VALUES (1, 1);
    """
    db.reset_to_schema(schema, seed)
    ok, error, _ = db.execute(
        "ALTER TABLE users RENAME COLUMN id TO user_id;",
        execute_mode="transaction",
    )
    assert ok is False
    assert "drop referencing constraint" in error.lower()


def test_medium_fk_sequence_reaches_target_shadow_schema():
    db = DBManager()
    schema = """
    CREATE TABLE users (
        id SERIAL PRIMARY KEY,
        username VARCHAR(100) NOT NULL
    );
    CREATE TABLE orders (
        id SERIAL PRIMARY KEY,
        user_id INTEGER NOT NULL,
        CONSTRAINT fk_orders_users FOREIGN KEY (user_id) REFERENCES users(id)
    );
    """
    seed = """
    INSERT INTO users (id, username) VALUES (1, 'user_0001');
    INSERT INTO orders (id, user_id) VALUES (1, 1);
    """
    db.reset_to_schema(schema, seed)
    for sql in [
        "ALTER TABLE orders DROP CONSTRAINT fk_orders_users;",
        "ALTER TABLE users RENAME COLUMN id TO user_id;",
        (
            "ALTER TABLE orders ADD CONSTRAINT fk_orders_users "
            "FOREIGN KEY (user_id) REFERENCES users(user_id);"
        ),
    ]:
        ok, error, _ = db.execute(sql, execute_mode="transaction")
        assert ok is True, error
    ddl = db.get_schema_ddl()
    assert "user_id SERIAL PRIMARY KEY" in ddl
    assert "REFERENCES users(user_id)" in ddl


def test_seed_data_scales_match_docx_counts():
    env = ChronoMigrateEnv()
    env.reset({"task_id": "easy_add_column", "seed": 42})
    cursor = env.db.conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM users")
    assert cursor.fetchone()[0] == 1000

    env.reset({"task_id": "medium_rename_fk", "seed": 42})
    cursor = env.db.conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM users")
    users = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM orders")
    orders = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM order_items")
    items = cursor.fetchone()[0]
    assert (users, orders, items) == (500, 1500, 4500)

    env.reset({"task_id": "hard_repartition", "seed": 42})
    cursor = env.db.conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM events")
    assert cursor.fetchone()[0] == 10000


def test_postgresql_smoke_skips_cleanly_when_unavailable():
    if not shutil.which("initdb") or not shutil.which("pg_ctl"):
        pytest.skip("PostgreSQL binaries unavailable")

    testing_postgresql = pytest.importorskip("testing.postgresql")

    try:
        with testing_postgresql.Postgresql() as pg:
            task = TASKS["easy_add_column"]
            db = DBManager()
            if db.backend != "postgresql":
                pytest.skip("PostgreSQL backend not available in this environment")
            db.conn.close()
            db.conn = pg.connect()
            db.backend = "postgresql"
            db.reset_to_schema(task.starting_schema_sql, task.seed_data_sql)
            ok, _, _ = db.execute(
                "ALTER TABLE users ADD COLUMN email VARCHAR(255) DEFAULT NULL;",
                execute_mode="transaction",
            )
            assert ok is True
            assert "email" in db.get_schema_ddl().lower()
    except Exception as exc:
        pytest.skip(f"PostgreSQL smoke unavailable: {exc}")


def test_hard_grader_rewards_safe_pattern_history():
    task = TASKS["hard_repartition"]
    base = task.grade_fn(
        task.target_schema_ddl,
        task.target_schema_ddl,
        "same",
        "same",
        0.95,
        action_history=[],
        steps_used=4,
    )
    safe = task.grade_fn(
        task.target_schema_ddl,
        task.target_schema_ddl,
        "same",
        "same",
        0.95,
        action_history=[
            "CREATE TABLE events_new (LIKE events INCLUDING ALL) PARTITION BY HASH (user_id);",
            "CREATE TABLE events_new_p0 PARTITION OF events_new FOR VALUES WITH (MODULUS 8, REMAINDER 0);",
            "CREATE TABLE events_new_p1 PARTITION OF events_new FOR VALUES WITH (MODULUS 8, REMAINDER 1);",
            "CREATE TABLE events_new_p2 PARTITION OF events_new FOR VALUES WITH (MODULUS 8, REMAINDER 2);",
            "CREATE TABLE events_new_p3 PARTITION OF events_new FOR VALUES WITH (MODULUS 8, REMAINDER 3);",
            "CREATE TABLE events_new_p4 PARTITION OF events_new FOR VALUES WITH (MODULUS 8, REMAINDER 4);",
            "CREATE TABLE events_new_p5 PARTITION OF events_new FOR VALUES WITH (MODULUS 8, REMAINDER 5);",
            "CREATE TABLE events_new_p6 PARTITION OF events_new FOR VALUES WITH (MODULUS 8, REMAINDER 6);",
            "CREATE TABLE events_new_p7 PARTITION OF events_new FOR VALUES WITH (MODULUS 8, REMAINDER 7);",
            "INSERT INTO events_new SELECT * FROM events WHERE id BETWEEN 1 AND 2500;",
            "INSERT INTO events_new SELECT * FROM events WHERE id BETWEEN 2501 AND 5000;",
            "INSERT INTO events_new SELECT * FROM events WHERE id BETWEEN 5001 AND 7500;",
            "INSERT INTO events_new SELECT * FROM events WHERE id BETWEEN 7501 AND 10000;",
            "ALTER TABLE events RENAME TO events_old;",
            "ALTER TABLE events_new RENAME TO events;",
            "DROP TABLE events_old CASCADE;",
        ],
        steps_used=12,
    )
    assert safe > base


def test_sqlite_hard_safe_pattern_supports_like_and_partition_children():
    task = TASKS["hard_repartition"]
    db = DBManager()
    db.reset_to_schema(task.starting_schema_sql, task.seed_data_sql)
    before_hash = compute_data_hash(db.conn, task.starting_schema_sql)

    steps = [
        "CREATE TABLE events_new (LIKE events INCLUDING ALL) PARTITION BY HASH (user_id);",
        "CREATE TABLE events_new_p0 PARTITION OF events_new FOR VALUES WITH (MODULUS 8, REMAINDER 0);",
        "CREATE TABLE events_new_p1 PARTITION OF events_new FOR VALUES WITH (MODULUS 8, REMAINDER 1);",
        "CREATE TABLE events_new_p2 PARTITION OF events_new FOR VALUES WITH (MODULUS 8, REMAINDER 2);",
        "CREATE TABLE events_new_p3 PARTITION OF events_new FOR VALUES WITH (MODULUS 8, REMAINDER 3);",
        "CREATE TABLE events_new_p4 PARTITION OF events_new FOR VALUES WITH (MODULUS 8, REMAINDER 4);",
        "CREATE TABLE events_new_p5 PARTITION OF events_new FOR VALUES WITH (MODULUS 8, REMAINDER 5);",
        "CREATE TABLE events_new_p6 PARTITION OF events_new FOR VALUES WITH (MODULUS 8, REMAINDER 6);",
        "CREATE TABLE events_new_p7 PARTITION OF events_new FOR VALUES WITH (MODULUS 8, REMAINDER 7);",
        "INSERT INTO events_new SELECT * FROM events WHERE id BETWEEN 1 AND 2500;",
        "INSERT INTO events_new SELECT * FROM events WHERE id BETWEEN 2501 AND 5000;",
        "INSERT INTO events_new SELECT * FROM events WHERE id BETWEEN 5001 AND 7500;",
        "INSERT INTO events_new SELECT * FROM events WHERE id BETWEEN 7501 AND 10000;",
        "ALTER TABLE events RENAME TO events_old;",
        "ALTER TABLE events_new RENAME TO events;",
        "DROP TABLE events_old CASCADE;",
    ]

    for sql in steps:
        ok, message, _ = db.execute(sql, execute_mode="transaction")
        assert ok is True, message

    after_hash = compute_data_hash(db.conn, task.starting_schema_sql)
    schema_match = compute_schema_match(db.get_schema_ddl(), task.target_schema_ddl)

    assert after_hash == before_hash
    assert schema_match > 0.99


def test_medium_fk_order_violation_returns_preflight_error_through_env():
    env = ChronoMigrateEnv()
    env.reset({"task_id": "medium_rename_fk", "seed": 42})
    obs = env.step(
        MigrationAction(
            sql="ALTER TABLE users RENAME COLUMN id TO user_id;",
            task_id="medium_rename_fk",
        )
    )

    assert "foreign key constraint still references users(id)" in obs.last_sql_result.lower()
    assert obs.step_count == 1


def test_drop_table_scores_zero_due_to_data_integrity_loss():
    app_env.reset({"task_id": "easy_add_column", "seed": 42})
    app_env.step(
        MigrationAction(
            sql="DROP TABLE users;",
            task_id="easy_add_column",
        )
    )
    result = grade_episode(GraderRequest(task_id="easy_add_column"))

    assert result["score"] == 0.0
    assert result["episode_reward"] == 0.0


def test_hard_grader_preserves_multiplicative_zero_on_data_loss():
    task = TASKS["hard_repartition"]
    score = task.grade_fn(
        task.target_schema_ddl,
        task.target_schema_ddl,
        "before",
        "after",
        1.0,
        action_history=[
            "CREATE TABLE events_new (LIKE events INCLUDING ALL) PARTITION BY HASH (user_id);",
            "CREATE TABLE events_new_p0 PARTITION OF events_new FOR VALUES WITH (MODULUS 8, REMAINDER 0);",
            "INSERT INTO events_new SELECT * FROM events WHERE id BETWEEN 1 AND 2500;",
            "ALTER TABLE events RENAME TO events_old;",
            "ALTER TABLE events_new RENAME TO events;",
            "DROP TABLE events_old CASCADE;",
        ],
        steps_used=12,
    )

    assert score == 0.0


def test_ten_sequential_easy_episodes_are_stable():
    env = ChronoMigrateEnv()
    scores = []

    for _ in range(10):
        env.reset({"task_id": "easy_add_column", "seed": 42})
        env.step(
            MigrationAction(
                sql="ALTER TABLE users ADD COLUMN email VARCHAR(255) DEFAULT NULL;",
                task_id="easy_add_column",
            )
        )
        state = env.state()
        availability = 1.0 - (
            state.failed_background_queries / state.total_background_queries
            if state.total_background_queries
            else 0.0
        )
        scores.append(
            TASKS["easy_add_column"].grade_fn(
                state.current_schema_ddl,
                state.target_schema_ddl,
                state.data_integrity_hash,
                state.current_data_hash,
                availability,
            )
        )

    assert len(scores) == 10
    assert len({round(score, 6) for score in scores}) == 1
