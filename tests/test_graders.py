import sqlite3

import pytest

from models import MigrationAction
from server.app import GraderRequest, grade_episode, list_tasks, env as app_env
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
