from typing import Literal

from pydantic import BaseModel, Field

try:
    from openenv.core.env_server.types import Action, Observation
except Exception:
    class Action(BaseModel):
        pass

    class Observation(BaseModel):
        done: bool = False
        reward: float = 0.0


class MigrationAction(Action):
    sql: str = Field(
        ...,
        description="Raw SQL statement to execute against the database.",
        examples=["ALTER TABLE users ADD COLUMN email VARCHAR(255) DEFAULT NULL;"],
    )
    task_id: str = Field(
        ...,
        description="Which task the agent is solving. Must match a task from /tasks.",
        examples=["easy_add_column"],
    )
    execute_mode: Literal["transaction", "autocommit"] = Field(
        default="transaction",
        description="transaction = rollback on error. autocommit = no rollback.",
    )


class MigrationObservation(Observation):
    current_schema_ddl: str = Field(
        description="Full DDL of the current database schema as a string."
    )
    target_schema_ddl: str = Field(description="The target schema the agent must achieve.")
    last_sql_result: str = Field(
        description="Result of last SQL: SUCCESS, or the database error message."
    )
    downtime_pct: float = Field(
        description="Percentage of simulated background queries that failed this step."
    )
    step_count: int = Field(description="Current step number in the episode.")
    cumulative_downtime_pct: float = Field(
        description="Rolling downtime ratio across all steps in the episode."
    )
    task_id: str = Field(description="Current task being solved.")
    schema_match_pct: float = Field(
        description="Percentage of target schema features matched so far. 0.0-1.0."
    )
    episode_id: str = Field(description="Unique episode identifier for reproducibility.")


class MigrationState(Observation):
    episode_id: str
    task_id: str
    step_count: int
    max_steps: int
    current_schema_ddl: str
    target_schema_ddl: str
    total_background_queries: int
    failed_background_queries: int
    data_integrity_hash: str = Field(
        description="SHA-256 hash of source data rows at episode start."
    )
    current_data_hash: str = Field(
        description="SHA-256 hash of current data rows. Must match data_integrity_hash."
    )
    schema_match_pct: float
    cumulative_reward: float
    db_backend: Literal["postgresql", "sqlite"] = Field(
        description="Which backend is active."
    )
