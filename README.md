---
title: ChronoMigrate Env
emoji: 📊
colorFrom: gray
colorTo: red
sdk: docker
pinned: false
---

# ChronoMigrate-Env
**Zero-Downtime Database Migration RL Environment**

## Environment Description
ChronoMigrate-Env is an OpenEnv-compatible reinforcement learning environment for zero-downtime database schema migration. Agents are graded on three things at once:

- schema correctness
- operational availability during the migration
- data integrity after the migration

The runtime simulates transactional load with a deterministic discrete-event model, analyzes SQL lock impact with `sqlglot`, and supports PostgreSQL-first execution with SQLite fallback.

## Action Space
| Field | Type | Description |
|---|---|---|
| `sql` | string | Raw SQL statement to execute against the database |
| `task_id` | string | Task being solved |
| `execute_mode` | string | `transaction` or `autocommit` |

## Observation Space
| Field | Type | Description |
|---|---|---|
| `done` | bool | Whether the episode has ended |
| `reward` | float | Reward for the most recent step |
| `current_schema_ddl` | string | Current schema as DDL |
| `target_schema_ddl` | string | Target schema |
| `last_sql_result` | string | `SUCCESS` or backend error |
| `downtime_pct` | float | Failure rate for the latest step |
| `step_count` | int | Episode step counter |
| `cumulative_downtime_pct` | float | Rolling downtime signal |
| `task_id` | string | Active task identifier |
| `schema_match_pct` | float | Progress toward target schema |
| `episode_id` | string | Unique episode id |

## Tasks
| Task | Difficulty | Max Steps | Description |
|---|---|---|---|
| `easy_add_column` | Easy | 5 | Add 2 columns with DEFAULT values |
| `medium_rename_fk` | Medium | 10 | Rename a primary key column and update FK references |
| `hard_repartition` | Hard | 20 | Repartition a large table under simulated load |

## Baseline Scores (gpt-4o-mini)
The required baseline entrypoint is the root-level `inference.py` script. It uses a generic DDL-driven fallback strategy for safer migration planning and expects one of `HF_TOKEN`, `API_KEY`, or `OPENAI_API_KEY` to be configured.

Expected target scores after redeploy:

| Task | Score |
|---|---|
| `easy_add_column` | `~0.85+` |
| `medium_rename_fk` | `~0.55+` |
| `hard_repartition` | `~0.25+` |

## Setup
```bash
python -m venv .venv
pip install -r requirements.txt
docker build -t chronomigrate-env .
docker run --rm -p 7860:7860 chronomigrate-env
```

Set one of `HF_TOKEN`, `API_KEY`, or `OPENAI_API_KEY` before running the baseline script. `MODEL_NAME` defaults to `gpt-4o-mini`.

## Usage
```python
from client import ChronoMigrateClient
from models import MigrationAction

async with ChronoMigrateClient(base_url="http://localhost:7860") as env:
    obs = await env.reset(config={"task_id": "easy_add_column", "seed": 42})
    step = await env.step(
        MigrationAction(
            sql="ALTER TABLE users ADD COLUMN email VARCHAR(255) DEFAULT NULL;",
            task_id="easy_add_column",
            execute_mode="transaction",
        )
    )
```

Run the baseline script:
```bash
python inference.py --all-tasks
```

## How Judging Works
Per-step reward is multiplicative:

```text
R_step = schema_match_delta * (1 - downtime_pct) * data_integrity
```

Final grading uses the current episode state:

```text
score = schema_match * availability * data_integrity
```

This design prevents reward hacking. An agent that drops data or causes full downtime gets multiplied toward zero even if the target schema appears to match.
