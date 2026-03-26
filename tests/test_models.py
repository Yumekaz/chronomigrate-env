from models import MigrationAction, MigrationObservation, MigrationState


def test_action_model_defaults():
    action = MigrationAction(sql="SELECT 1;", task_id="easy_add_column")
    assert action.execute_mode == "transaction"


def test_observation_model_roundtrip():
    observation = MigrationObservation(
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


def test_state_model_backend_literal():
    state = MigrationState(
        episode_id="episode-1",
        task_id="easy_add_column",
        step_count=0,
        max_steps=8,
        current_schema_ddl="",
        target_schema_ddl="",
        total_background_queries=0,
        failed_background_queries=0,
        data_integrity_hash="a",
        current_data_hash="a",
        schema_match_pct=0.0,
        cumulative_reward=0.0,
        done=False,
        db_backend="sqlite",
    )
    assert state.db_backend == "sqlite"
