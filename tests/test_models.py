from models import MigrationAction, MigrationObservation, MigrationState


def test_action_model_defaults():
    action = MigrationAction(sql="SELECT 1;", task_id="easy_add_column")
    assert action.execute_mode == "transaction"
    payload = action.model_dump()
    assert payload["sql"] == "SELECT 1;"
    assert payload["task_id"] == "easy_add_column"
    assert payload["execute_mode"] == "transaction"


def test_action_model_accepts_autocommit():
    action = MigrationAction(
        sql="ALTER TABLE users ADD COLUMN email TEXT;",
        task_id="easy_add_column",
        execute_mode="autocommit",
    )
    assert action.execute_mode == "autocommit"


def test_action_schema_exposes_required_fields():
    schema = MigrationAction.model_json_schema()
    assert schema["required"] == ["sql", "task_id"]
    assert schema["properties"]["execute_mode"]["default"] == "transaction"


def test_observation_model_roundtrip():
    observation = MigrationObservation(
        done=False,
        reward=0.25,
        current_schema_ddl="CREATE TABLE users(id INTEGER);",
        target_schema_ddl="CREATE TABLE users(id INTEGER, email TEXT);",
        last_sql_result="SUCCESS",
        downtime_pct=0.0,
        step_count=1,
        cumulative_downtime_pct=0.0,
        task_id="easy_add_column",
        schema_match_pct=0.5,
        episode_id="episode-1",
    )
    assert observation.task_id == "easy_add_column"
    assert observation.model_dump()["episode_id"] == "episode-1"
    assert observation.done is False
    assert observation.reward == 0.25


def test_observation_schema_contains_runtime_fields():
    schema = MigrationObservation.model_json_schema()
    for field in [
        "done",
        "reward",
        "current_schema_ddl",
        "target_schema_ddl",
        "last_sql_result",
        "downtime_pct",
        "step_count",
        "cumulative_downtime_pct",
        "task_id",
        "schema_match_pct",
        "episode_id",
    ]:
        assert field in schema["properties"]


def test_state_model_backend_literal():
    state = MigrationState(
        done=False,
        reward=0.0,
        episode_id="episode-1",
        task_id="easy_add_column",
        step_count=0,
        max_steps=5,
        current_schema_ddl="",
        target_schema_ddl="",
        total_background_queries=0,
        failed_background_queries=0,
        data_integrity_hash="a",
        current_data_hash="a",
        schema_match_pct=0.0,
        cumulative_reward=0.0,
        db_backend="sqlite",
    )
    assert state.db_backend == "sqlite"
    assert state.model_dump()["done"] is False
    assert state.model_dump()["reward"] == 0.0
