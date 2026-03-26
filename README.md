# ChronoMigrate-Env

**Zero-Downtime Database Migration RL Environment**

An OpenEnv environment that trains AI agents to execute schema migrations without causing production downtime.

## Environment Description

ChronoMigrate-Env simulates a live database under transactional load. An agent must migrate the schema from a starting state to a target state by issuing SQL commands. The grader scores schema correctness, data integrity, and operational availability.

## Action Space

| Field | Type | Description |
|-------|------|-------------|
| sql | string | SQL statement to execute |
| task_id | string | Task being solved |
| execute_mode | string | transaction or autocommit |

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| current_schema_ddl | string | Current schema as DDL |
| target_schema_ddl | string | Target schema |
| last_sql_result | string | SUCCESS or error message |
| downtime_pct | float | Query failure rate this step |
| step_count | int | Current step number |
| cumulative_downtime_pct | float | Rolling downtime ratio |
| schema_match_pct | float | Progress toward target schema |
| episode_id | string | Reproducible episode identifier |

## Tasks

| Task | Difficulty | Description |
|------|-----------|-------------|
| easy_add_column | Easy | Add two defaulted columns safely |
| medium_rename_fk | Medium | Rename a key column and repair foreign key references |
| hard_repartition | Hard | Repartition a table under simulated load |

## Baseline Scores (GPT-4o-mini)

| Task | Score |
|------|-------|
| easy_add_column | ~0.85 |
| medium_rename_fk | ~0.55 |
| hard_repartition | ~0.25 |

## Setup

```bash
docker build -t chronomigrate-env .
docker run -p 7860:7860 -e OPENAI_API_KEY=your_key chronomigrate-env
```

## Usage

```python
from client import ChronoMigrateClient
from models import MigrationAction

async with ChronoMigrateClient(base_url="http://localhost:7860") as env:
    observation = await env.reset(config={"task_id": "easy_add_column"})
    result = await env.step(
        MigrationAction(
            sql="ALTER TABLE users ADD COLUMN email VARCHAR(255) DEFAULT NULL;",
            task_id="easy_add_column",
        )
    )
```
