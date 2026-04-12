import sqlite3
import shutil
import importlib
from dataclasses import replace
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from inference import (
    _is_obviously_unsafe_sql,
    _normalize_sql,
    _repeats_failed_sql,
    _safe_run_episode,
    _task_guidance,
    run_episode,
)
from models import MigrationAction
from server.app import (
    GraderRequest,
    _parse_subprocess_json,
    _generate_feedback,
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
from server.tasks import (
    SCORE_MAX,
    SCORE_MIN,
    TASKS,
    normalize_task_score,
)


def _manifest_grader_ref(task: dict) -> str:
    grader = task["grader"]
    if isinstance(grader, dict):
        if "callable" in grader:
            return str(grader["callable"])
        return str(grader["entrypoint"])
    if isinstance(grader, str) and ":" not in grader and "grader_spec" in task:
        grader_spec = task["grader_spec"]
        if isinstance(grader_spec, dict):
            if "callable" in grader_spec:
                return str(grader_spec["callable"])
            return str(grader_spec["entrypoint"])
    return str(grader)


def _manifest_grader_refs(task: dict) -> list[str]:
    grader = task["grader"]
    if not isinstance(grader, dict):
        if isinstance(grader, str) and ":" not in grader and "grader_spec" in task:
            grader = task["grader_spec"]
        else:
            return [str(grader)]

    refs = []
    for key in ("path", "callable", "entrypoint"):
        if key in grader:
            refs.append(str(grader[key]))
    return list(dict.fromkeys(refs))


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
        created_at TIMESTAMP DEFAULT '2024-01-01 00:00:00'
    );
    """
    assert compute_schema_match(ddl, ddl) == 1.0


def test_schema_match_penalizes_unexpected_extra_columns():
    current = """
    CREATE TABLE users (
        id SERIAL PRIMARY KEY,
        username VARCHAR(100) NOT NULL,
        created_at TIMESTAMP DEFAULT '2024-01-01 00:00:00',
        email VARCHAR(255) DEFAULT NULL,
        is_active BOOLEAN DEFAULT TRUE,
        debug_notes TEXT
    );
    """
    target = """
    CREATE TABLE users (
        id SERIAL PRIMARY KEY,
        username VARCHAR(100) NOT NULL,
        created_at TIMESTAMP DEFAULT '2024-01-01 00:00:00',
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


def test_compute_data_hash_rolls_back_failed_postgres_reads():
    class FakeCursor:
        def __init__(self, conn):
            self.conn = conn
            self.last_sql = ""

        def execute(self, sql, params=None):
            self.last_sql = sql
            if sql == "SELECT * FROM events ORDER BY 1":
                raise RuntimeError('relation "events" does not exist')

        def fetchall(self):
            return []

    class FakeConn:
        rollback_calls = 0

        def cursor(self):
            return FakeCursor(self)

        def rollback(self):
            self.rollback_calls += 1

    fake_conn = FakeConn()
    fake_conn.__class__.__module__ = "psycopg2.extensions"

    digest = compute_data_hash(fake_conn, "CREATE TABLE events (id BIGSERIAL);")

    assert isinstance(digest, str)
    assert len(digest) == 64
    assert fake_conn.rollback_calls == 1


def test_env_reset_and_step_updates_episode_state():
    env = ChronoMigrateEnv()
    obs = env.reset({"task_id": "easy_add_column", "seed": 42})
    assert obs.task_id == "easy_add_column"
    assert obs.step_count == 0
    assert obs.episode_id
    assert 0.0 < obs.reward < 1.0

    step = env.step(
        MigrationAction(
            sql="ALTER TABLE users ADD COLUMN email VARCHAR(255) DEFAULT NULL;",
            task_id="easy_add_column",
        )
    )
    assert step.step_count == 1
    assert "email" in step.current_schema_ddl.lower()
    assert 0.0 <= step.schema_match_pct <= 1.0
    assert 0.0 < step.reward < 1.0
    assert step.done is False
    assert env.state.cumulative_reward >= 0.0


def test_tasks_endpoint_lists_expected_tasks():
    payload = list_tasks()
    tasks = payload["tasks"]
    assert payload["count"] == 3
    assert [task["id"] for task in tasks] == [
        "easy_add_column",
        "medium_rename_fk",
        "hard_repartition",
    ]
    assert all(
        set(task) == {"id", "description", "difficulty", "max_steps", "grader"}
        for task in tasks
    )


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
    assert state.json()["state"]["task_id"] == "easy_add_column"


def test_grader_returns_bounded_score_for_completed_episode():
    app_env.reset({"task_id": "easy_add_column", "seed": 42})
    app_env.step(
        MigrationAction(
            sql="ALTER TABLE users ADD COLUMN email VARCHAR(255) DEFAULT NULL;",
            task_id="easy_add_column",
        )
    )
    result = grade_episode(GraderRequest(task_id="easy_add_column"))
    assert 0.0 < result["score"] < 1.0
    assert 0.0 < result["availability"] < 1.0
    assert 0.0 < result["schema_match"] < 1.0
    assert 0.0 < result["data_integrity"] < 1.0


def test_grader_cold_start_returns_score_for_all_tasks():
    client = TestClient(create_app())

    for task_id in ["easy_add_column", "medium_rename_fk", "hard_repartition"]:
        response = client.post("/grader", json={"task_id": task_id})
        assert response.status_code == 200
        payload = response.json()
        assert 0.0 < payload["score"] < 1.0
        assert 0.0 < payload["schema_match"] < 1.0
        assert 0.0 < payload["availability"] < 1.0
        assert 0.0 < payload["data_integrity"] < 1.0


def test_grader_uses_task_specific_grade_function(monkeypatch):
    original_task = TASKS["easy_add_column"]

    def fake_grade_fn(**kwargs):
        assert kwargs["steps_used"] == 1
        assert kwargs["action_history"]
        return 0.4321

    monkeypatch.setitem(
        TASKS, "easy_add_column", replace(original_task, grade_fn=fake_grade_fn)
    )

    app_env.reset({"task_id": "easy_add_column", "seed": 42})
    app_env.step(
        MigrationAction(
            sql="ALTER TABLE users ADD COLUMN email VARCHAR(255) DEFAULT NULL;",
            task_id="easy_add_column",
        )
    )

    result = grade_episode(GraderRequest(task_id="easy_add_column"))
    assert result["score"] == 0.432


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
    assert obs.reward == normalize_task_score(0.0)
    assert env.state.step_count == 1


def test_medium_task_completes_after_canonical_three_step_sequence():
    env = ChronoMigrateEnv()
    env.reset({"task_id": "medium_rename_fk", "seed": 42})

    steps = [
        "ALTER TABLE orders DROP CONSTRAINT fk_orders_users;",
        "ALTER TABLE users RENAME COLUMN id TO user_id;",
        (
            "ALTER TABLE orders ADD CONSTRAINT fk_orders_users "
            "FOREIGN KEY (user_id) REFERENCES users(user_id);"
        ),
    ]

    for sql in steps:
        obs = env.step(
            MigrationAction(
                sql=sql,
                task_id="medium_rename_fk",
                execute_mode="transaction",
            )
        )

    assert obs.schema_match_pct == normalize_task_score(1.0)
    assert obs.done is True
    assert env.state.done is True


def test_unexpected_step_error_is_penalized_gracefully(monkeypatch):
    env = ChronoMigrateEnv()
    env.reset({"task_id": "easy_add_column", "seed": 42})

    def explode(*args, **kwargs):
        raise RuntimeError("synthetic grading failure")

    monkeypatch.setattr("server.chrono_migrate_env.compute_schema_match", explode)

    obs = env.step(
        MigrationAction(
            sql="ALTER TABLE users ADD COLUMN email VARCHAR(255) DEFAULT NULL;",
            task_id="easy_add_column",
        )
    )

    assert "EXECUTION_ERROR" in obs.last_sql_result
    assert obs.reward == normalize_task_score(0.0)
    assert env.state.step_count == 1
    assert env.state.done is False


def test_grader_rejects_episode_id_mismatch():
    app_env.reset({"task_id": "easy_add_column", "seed": 42})
    result = grade_episode(
        GraderRequest(task_id="easy_add_column", episode_id="wrong-episode-id")
    )
    assert result["score"] == normalize_task_score(0.0)
    assert "mismatch" in result["feedback"].lower()


def test_feedback_uses_completion_tolerance_for_pass_case():
    feedback = _generate_feedback(0.9999999999, 1.0, 1.0)
    assert feedback == "PASS: Perfect zero-downtime migration achieved."


def test_reset_and_step_contract_through_http():
    client = TestClient(app)
    reset = client.post("/reset", json={"task_id": "easy_add_column", "seed": 42})
    assert reset.status_code == 200
    assert reset.json()["observation"]["task_id"] == "easy_add_column"
    assert 0.0 < reset.json()["observation"]["reward"] < 1.0

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
    assert 0.0 < body["reward"] < 1.0
    assert 0.0 < body["observation"]["reward"] < 1.0
    assert "info" in body
    assert 0.0 < body["info"]["score"] < 1.0


def test_step_contract_allows_missing_task_id_after_reset():
    client = TestClient(app)
    client.post("/reset", json={"task_id": "easy_add_column", "seed": 42})

    response = client.post(
        "/step",
        json={
            "sql": "ALTER TABLE users ADD COLUMN email VARCHAR(255) DEFAULT NULL;",
            "execute_mode": "transaction",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["observation"]["task_id"] == "easy_add_column"
    assert 0.0 < body["info"]["score"] < 1.0


def test_step_contract_allows_string_action_wrapper():
    client = TestClient(app)
    client.post("/reset", json={"task_id": "easy_add_column", "seed": 42})

    response = client.post(
        "/step",
        json={"action": "ALTER TABLE users ADD COLUMN email VARCHAR(255) DEFAULT NULL;"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["observation"]["task_id"] == "easy_add_column"
    assert 0.0 < body["info"]["score"] < 1.0


def test_step_contract_allows_missing_sql_as_safe_noop():
    client = TestClient(app)
    client.post("/reset", json={"task_id": "easy_add_column", "seed": 42})

    response = client.post(
        "/step",
        json={"commands": [{"action_type": "wait"}]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["observation"]["task_id"] == "easy_add_column"
    assert body["observation"]["last_sql_result"] == "SUCCESS"
    assert 0.0 < body["info"]["score"] < 1.0


def test_baseline_endpoint_returns_script_scores(monkeypatch):
    def fake_run(*args, **kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout=(
                '[START] {"task_id":"easy_add_column","seed":42,"model":"gpt-4o-mini"}\n'
                '[STEP] {"task_id":"easy_add_column","step":1,"sql":"ALTER TABLE users ADD COLUMN email VARCHAR(255) DEFAULT NULL;","result":"SUCCESS","schema_match_pct":0.8333,"downtime_pct":0.0,"done":false}\n'
                f'[END] {{"task_id":"easy_add_column","score":{normalize_task_score(1.0)}}}\n'
                f'{{"easy_add_column": {normalize_task_score(1.0)}, "medium_rename_fk": 0.6731, "hard_repartition": 0.244}}'
            ),
        )

    monkeypatch.setattr("server.app.subprocess.run", fake_run)
    client = TestClient(app)
    response = client.post("/baseline")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["baseline_scores"] == {
        "easy_add_column": normalize_task_score(1.0),
        "medium_rename_fk": 0.673,
        "hard_repartition": 0.244,
    }


def test_baseline_endpoint_surfaces_script_error(monkeypatch):
    def fake_run(*args, **kwargs):
        return SimpleNamespace(
            returncode=1,
            stdout='{"error": "API_KEY or OPENAI_API_KEY is required."}',
            stderr="",
        )

    monkeypatch.setattr("server.app.subprocess.run", fake_run)
    client = TestClient(app)
    response = client.post("/baseline")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "error"
    assert payload["baseline_scores"] == {}
    assert payload["error"] == "API_KEY or OPENAI_API_KEY is required."
    assert payload["returncode"] == 1


def test_parse_subprocess_json_ignores_structured_logs():
    stdout = "\n".join(
        [
            '[START] {"task_id":"easy_add_column","seed":42,"model":"gpt-4o-mini"}',
            '[STEP] {"task_id":"easy_add_column","step":1,"sql":"ALTER TABLE users ADD COLUMN email VARCHAR(255) DEFAULT NULL;","result":"SUCCESS","schema_match_pct":0.8333,"downtime_pct":0.0,"done":false}',
            f'[END] {{"task_id":"easy_add_column","score":{normalize_task_score(1.0)}}}',
            f'{{"easy_add_column": {normalize_task_score(1.0)}, "medium_rename_fk": 0.6731, "hard_repartition": 0.244}}',
        ]
    )

    assert _parse_subprocess_json(stdout) == {
        "easy_add_column": normalize_task_score(1.0),
        "medium_rename_fk": 0.6731,
        "hard_repartition": 0.244,
    }


def test_run_episode_passes_seed_to_model_requests(monkeypatch):
    create_calls = []

    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class FakeCompletions:
        def create(self, **kwargs):
            create_calls.append(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="ALTER TABLE noop DO NOTHING;")
                    )
                ]
            )

    class FakeClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    def fake_post(url, json=None, timeout=30):
        if url.endswith("/reset"):
            return FakeResponse(
                {
                    "current_schema_ddl": TASKS["medium_rename_fk"].target_schema_ddl,
                    "target_schema_ddl": TASKS["medium_rename_fk"].target_schema_ddl,
                    "last_sql_result": "RESET",
                    "downtime_pct": 0.0,
                    "step_count": 0,
                    "cumulative_downtime_pct": 0.0,
                    "task_id": "medium_rename_fk",
                    "schema_match_pct": 1.0,
                    "episode_id": "episode-1",
                }
            )
        if url.endswith("/step"):
            return FakeResponse(
                {
                    "observation": {
                        "current_schema_ddl": TASKS["medium_rename_fk"].target_schema_ddl,
                        "target_schema_ddl": TASKS["medium_rename_fk"].target_schema_ddl,
                        "last_sql_result": "SUCCESS",
                        "downtime_pct": 0.0,
                        "step_count": 1,
                        "cumulative_downtime_pct": 0.0,
                        "task_id": "medium_rename_fk",
                        "schema_match_pct": 1.0,
                        "episode_id": "episode-1",
                    },
                    "done": True,
                    "reward": 0.0,
                    "metadata": {},
                }
            )
        if url.endswith("/grader"):
            return FakeResponse({"score": 1.0})
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr("inference.API_KEY", "test-key")
    monkeypatch.setattr("inference._get_client", lambda: FakeClient())
    monkeypatch.setattr("inference._wait_for_env_ready", lambda: None)
    monkeypatch.setattr("inference.requests.post", fake_post)

    score = run_episode("medium_rename_fk", seed=42)

    assert score == normalize_task_score(1.0)
    assert create_calls
    assert create_calls[0]["seed"] == 42
    assert create_calls[0]["temperature"] == 0.0


def test_inference_prefers_api_key_over_hf_token(monkeypatch):
    import inference

    captured = {}

    def fake_openai(*, base_url, api_key):
        captured["base_url"] = base_url
        captured["api_key"] = api_key
        return SimpleNamespace()

    monkeypatch.setenv("API_BASE_URL", "https://litellm.example/v1")
    monkeypatch.setenv("API_KEY", "litellm-key")
    monkeypatch.setenv("HF_TOKEN", "hf-should-not-be-used")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-fallback")

    reloaded = importlib.reload(inference)
    monkeypatch.setattr(reloaded, "OpenAI", fake_openai)

    client = reloaded._get_client()

    assert isinstance(client, SimpleNamespace)
    assert captured == {
        "base_url": "https://litellm.example/v1",
        "api_key": "litellm-key",
    }

    importlib.reload(inference)


def test_safe_run_episode_returns_zero_on_failure(monkeypatch, capsys):
    def boom(task_id, seed=42):
        raise RuntimeError("connection refused")

    monkeypatch.setattr("inference.run_episode", boom)

    score = _safe_run_episode("easy_add_column", seed=42)

    assert score == normalize_task_score(0.0)
    stdout = capsys.readouterr().out
    assert "[END]" in stdout
    assert '"task_id":"easy_add_column"' in stdout
    assert f'"score":{normalize_task_score(0.0)}' in stdout
    assert '"error":"connection refused"' in stdout


def test_normalize_sql_strips_code_fences_and_appends_semicolon():
    sql = "```sql\nALTER TABLE users ADD COLUMN email TEXT\n```"
    assert _normalize_sql(sql) == "ALTER TABLE users ADD COLUMN email TEXT;"


def test_task_guidance_describes_safe_hard_strategy():
    guidance = _task_guidance("hard_repartition")
    assert "create-copy-swap" in guidance
    assert "all eight hash partitions" in guidance
    assert "events_old" in guidance


def test_baseline_marks_destructive_sql_as_unsafe():
    assert _is_obviously_unsafe_sql("DROP TABLE users;") is True
    assert _is_obviously_unsafe_sql("DROP TABLE events_old CASCADE;") is False
    assert _is_obviously_unsafe_sql("DROP TABLE IF EXISTS events_old;") is False
    assert _is_obviously_unsafe_sql("TRUNCATE events;") is True
    assert _is_obviously_unsafe_sql("ALTER TABLE users ADD COLUMN email TEXT;") is False


def test_baseline_detects_repeated_failed_sql():
    history = [
        {
            "sql": "ALTER TABLE users RENAME COLUMN id TO user_id;",
            "result": "foreign key constraint still references users(id)",
            "schema_match": "0.0",
        }
    ]

    assert _repeats_failed_sql("ALTER TABLE users RENAME COLUMN id TO user_id;", history) is True
    assert _repeats_failed_sql("ALTER TABLE orders DROP CONSTRAINT fk_orders_users;", history) is False


def test_run_episode_retries_once_after_unsafe_sql(monkeypatch):
    create_calls = []
    step_calls = []
    responses = iter(
        [
            "DROP TABLE users;",
            "ALTER TABLE users ADD COLUMN email VARCHAR(255) DEFAULT NULL;",
        ]
    )

    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class FakeCompletions:
        def create(self, **kwargs):
            create_calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=next(responses)))]
            )

    class FakeClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    def fake_post(url, json=None, timeout=30):
        if url.endswith("/reset"):
            return FakeResponse(
                {
                    "current_schema_ddl": TASKS["easy_add_column"].starting_schema_sql,
                    "target_schema_ddl": TASKS["easy_add_column"].target_schema_ddl,
                    "last_sql_result": "RESET",
                    "downtime_pct": 0.0,
                    "step_count": 0,
                    "cumulative_downtime_pct": 0.0,
                    "task_id": "easy_add_column",
                    "schema_match_pct": 0.0,
                    "episode_id": "episode-1",
                    "done": False,
                }
            )
        if url.endswith("/step"):
            step_calls.append(json)
            return FakeResponse(
                {
                    "observation": {
                        "current_schema_ddl": TASKS["easy_add_column"].target_schema_ddl,
                        "target_schema_ddl": TASKS["easy_add_column"].target_schema_ddl,
                        "last_sql_result": "SUCCESS",
                        "downtime_pct": 0.0,
                        "step_count": 1,
                        "cumulative_downtime_pct": 0.0,
                        "task_id": "easy_add_column",
                        "schema_match_pct": 1.0,
                        "episode_id": "episode-1",
                        "done": True,
                    },
                    "done": True,
                    "reward": 1.0,
                    "metadata": {},
                }
            )
        if url.endswith("/grader"):
            return FakeResponse({"score": 1.0})
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr("inference.API_KEY", "test-key")
    monkeypatch.setattr("inference._get_client", lambda: FakeClient())
    monkeypatch.setattr("inference._wait_for_env_ready", lambda: None)
    monkeypatch.setattr("inference.requests.post", fake_post)

    score = run_episode("easy_add_column", seed=42)

    assert score == normalize_task_score(1.0)
    assert len(create_calls) == 2
    assert step_calls == [
        {
            "sql": "ALTER TABLE users ADD COLUMN email VARCHAR(255) DEFAULT NULL;",
            "task_id": "easy_add_column",
            "execute_mode": "transaction",
        }
    ]


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


def test_sqlite_fallback_supports_multi_add_column_statement():
    task = TASKS["easy_add_column"]
    db = DBManager()
    db.reset_to_schema(task.starting_schema_sql, task.seed_data_sql)

    ok, error, _ = db.execute(
        (
            "ALTER TABLE users "
            "ADD COLUMN email VARCHAR(255) DEFAULT NULL, "
            "ADD COLUMN is_active BOOLEAN DEFAULT TRUE;"
        ),
        execute_mode="transaction",
    )

    assert ok is True, error
    ddl = db.get_schema_ddl().lower()
    assert "email varchar(255) default null" in ddl
    assert "is_active boolean default true" in ddl


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


def test_medium_drop_constraint_if_exists_is_accepted_in_sqlite_fallback():
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
        "ALTER TABLE orders DROP CONSTRAINT IF EXISTS fk_orders_users;",
        execute_mode="transaction",
    )

    assert ok is True, error
    assert "fk_orders_users" not in db.get_schema_ddl().lower()


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


def test_normalize_task_score_is_bounded():
    assert normalize_task_score(0.0) == SCORE_MIN
    assert normalize_task_score(1.0) == SCORE_MAX
    assert SCORE_MIN <= normalize_task_score(0.4321) <= SCORE_MAX


def test_hard_grader_respects_availability_and_data_integrity():
    task = TASKS["hard_repartition"]
    high_availability = task.grade_fn(
        task.target_schema_ddl,
        task.target_schema_ddl,
        "same",
        "same",
        0.95,
    )
    low_availability = task.grade_fn(
        task.target_schema_ddl,
        task.target_schema_ddl,
        "same",
        "same",
        0.5,
    )
    data_loss = task.grade_fn(
        task.target_schema_ddl,
        task.target_schema_ddl,
        "before",
        "after",
        1.0,
    )
    assert high_availability > low_availability > SCORE_MIN
    assert data_loss == SCORE_MIN


def test_manifest_graders_are_importable_and_bounded_on_empty_input():
    from server.tasks.task_easy import EasyGrader
    from server.tasks.task_hard import HardGrader
    from server.tasks.task_medium import MediumGrader

    expected_floor = SCORE_MIN

    for grader_cls in (EasyGrader, MediumGrader, HardGrader):
        grader = grader_cls()
        assert grader.grade() == expected_floor
        assert grader() == expected_floor


def test_manifest_graders_accept_dict_payloads():
    from server.tasks.task_easy import EasyGrader
    from server.tasks.task_hard import HardGrader
    from server.tasks.task_medium import MediumGrader

    payload = {
        "current_schema_ddl": "CREATE TABLE t (id INT);",
        "target_schema_ddl": "CREATE TABLE t (id INT);",
        "data_hash_before": "same",
        "data_hash_after": "same",
        "availability_pct": 1.0,
    }

    for grader_cls in (EasyGrader, MediumGrader, HardGrader):
        grader = grader_cls()
        assert grader.grade(payload) == SCORE_MAX


def test_manifest_graders_accept_positional_payloads():
    from server.tasks.task_easy import EasyGrader
    from server.tasks.task_hard import HardGrader
    from server.tasks.task_medium import MediumGrader

    args = (
        "CREATE TABLE t (id INT);",
        "CREATE TABLE t (id INT);",
        "same",
        "same",
        1.0,
    )

    for grader_cls in (EasyGrader, MediumGrader, HardGrader):
        grader = grader_cls()
        assert grader.grade(*args) == SCORE_MAX


def test_manifest_declares_three_simple_tasks():
    import yaml
    from pathlib import Path

    manifest = yaml.safe_load(Path("openenv.yaml").read_text())

    assert manifest["name"] == "chronomigrate-env"
    assert [task["id"] for task in manifest["tasks"]] == [
        "easy_add_column",
        "medium_rename_fk",
        "hard_repartition",
    ]
    assert manifest["tasks"][0]["grader"] == "grade/task_easy"
    assert manifest["action_space"]["type"] == "text"
    assert manifest["observation_space"]["format"] == "json"


def test_manifest_grader_symbols_are_directly_callable():
    import yaml
    from importlib import import_module
    from pathlib import Path

    manifest = yaml.safe_load(Path("openenv.yaml").read_text())
    args = (
        "CREATE TABLE t (id INT);",
        "CREATE TABLE t (id INT);",
        "same",
        "same",
        1.0,
    )

    for ref in [
        "server.tasks.task_easy:easy_grader",
        "server.tasks.task_medium:medium_grader",
        "server.tasks.task_hard:hard_grader",
    ]:
        module_name, symbol_name = ref.split(":")
        symbol = getattr(import_module(module_name), symbol_name)
        assert callable(symbol)
        assert symbol(*args) == SCORE_MAX
        if hasattr(symbol, "grade"):
            assert symbol.grade(*args) == SCORE_MAX


def test_manifest_grader_symbols_support_empty_and_payload_calls():
    import yaml
    from importlib import import_module
    from pathlib import Path

    manifest = yaml.safe_load(Path("openenv.yaml").read_text())
    args = (
        "CREATE TABLE t (id INT);",
        "CREATE TABLE t (id INT);",
        "same",
        "same",
        1.0,
    )

    for ref in [
        "server.tasks.task_easy:easy_grader",
        "server.tasks.task_medium:medium_grader",
        "server.tasks.task_hard:hard_grader",
    ]:
        module_name, symbol_name = ref.split(":")
        symbol = getattr(import_module(module_name), symbol_name)
        result = symbol()
        if hasattr(result, "grade"):
            assert isinstance(result, float)
            assert float(result) == SCORE_MIN
            assert result.grade(*args) == SCORE_MAX
        else:
            assert result == SCORE_MIN
        assert symbol(*args) == SCORE_MAX
        if hasattr(symbol, "grade"):
            assert symbol.grade(*args) == SCORE_MAX


def test_tasks_endpoint_exposes_simple_task_list():
    client = TestClient(app)
    response = client.get("/tasks")
    assert response.status_code == 200

    payload = response.json()
    assert payload["count"] >= 3
    for task in payload["tasks"]:
        assert set(task) == {"id", "description", "difficulty", "max_steps", "grader"}
        assert task["id"] in TASKS


def test_grade_task_routes_return_bounded_scores():
    client = TestClient(app)
    route_expectations = {
        "/grade/task_easy": "easy_add_column",
        "/grade/task_medium": "medium_rename_fk",
        "/grade/task_hard": "hard_repartition",
    }

    for route, task_id in route_expectations.items():
        response = client.get(route)
        assert response.status_code == 200
        payload = response.json()
        assert 0.0 < payload["score"] < 1.0
        assert payload["feedback"]

        post_response = client.post(route)
        assert post_response.status_code == 200
        assert post_response.json()["score"] == payload["score"]


def test_task_registry_graders_are_safe_on_empty_input():
    expected_floor = SCORE_MIN

    for task in TASKS.values():
        assert task.grade_fn() == expected_floor


def test_task_registry_graders_accept_direct_inputs():
    args = (
        "CREATE TABLE t (id INT);",
        "CREATE TABLE t (id INT);",
        "same",
        "same",
        1.0,
    )

    for task in TASKS.values():
        assert task.grade_fn(*args) == SCORE_MAX


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


def test_drop_table_returns_floor_score_on_data_loss():
    app_env.reset({"task_id": "easy_add_column", "seed": 42})
    app_env.step(
        MigrationAction(
            sql="DROP TABLE users;",
            task_id="easy_add_column",
        )
    )
    result = grade_episode(GraderRequest(task_id="easy_add_column"))

    assert result["score"] == SCORE_MIN
    assert result["feedback"] == "FAIL: Data integrity compromised. Rows dropped or corrupted."


def test_hard_grader_penalizes_data_loss():
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

    assert score == SCORE_MIN


def test_ten_sequential_easy_episodes_are_deterministic():
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
        state = env.state
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
    assert scores == [SCORE_MAX] * 10
