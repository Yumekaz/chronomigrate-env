# ChronoMigrate-Env

**Zero-Downtime Database Migration RL Environment**

ChronoMigrate-Env is an OpenEnv-style environment for database schema migrations. An agent receives the current schema, the target schema, and runtime feedback about lock impact and schema progress, then issues SQL to complete the migration with minimal downtime.

## What It Exposes

| Surface | Purpose |
|---|---|
| `POST /reset` | Start a new episode for a task |
| `POST /step` | Execute one SQL action |
| `GET /state` | Inspect the active episode state |
| `GET /tasks` | List available tasks and action schema |
| `POST /grader` | Score the completed episode |
| `POST /baseline` | Run the baseline agent across all tasks |
| `GET /health` | Lightweight health check |
| `GET /metadata` | Environment name, description, runtime, and tags |
| `GET /schema` | Action, observation, and state JSON schemas |
| `POST /mcp` | JSON-RPC compatibility stub |
| `GET /` and `GET /web` | Lightweight HTML landing page for smoke checks |

## Data Contract

| Field | Type | Notes |
|---|---|---|
| `sql` | string | Raw SQL statement to execute |
| `task_id` | string | Must match the active task |
| `execute_mode` | string | `transaction` or `autocommit` |

| Observation Field | Type | Notes |
|---|---|---|
| `current_schema_ddl` | string | Current schema snapshot |
| `target_schema_ddl` | string | Goal schema snapshot |
| `last_sql_result` | string | `SUCCESS` or backend error |
| `downtime_pct` | float | Failure rate for the latest step |
| `step_count` | int | Episode step counter |
| `cumulative_downtime_pct` | float | Rolling availability signal |
| `task_id` | string | Active task identifier |
| `schema_match_pct` | float | Progress toward target schema |
| `episode_id` | string | Unique episode id |

## Tasks

| Task | Difficulty | Focus |
|---|---|---|
| `easy_add_column` | Easy | Add two defaulted columns |
| `medium_rename_fk` | Medium | Rename a primary-key column and repair foreign keys |
| `hard_repartition` | Hard | Repartition a table under load |

## Local Setup

```bash
docker build -t chronomigrate-env .
docker run -p 7860:7860 -e OPENAI_API_KEY=your_key chronomigrate-env
```

## Local Usage

```python
from client import ChronoMigrateClient
from models import MigrationAction

async with ChronoMigrateClient(base_url="http://localhost:7860") as env:
    obs = await env.reset(config={"task_id": "easy_add_column", "seed": 42})
    step = await env.step(
        MigrationAction(
            sql="ALTER TABLE users ADD COLUMN email VARCHAR(255) DEFAULT NULL;",
            task_id="easy_add_column",
        )
    )
```

## Verification

Recommended quick checks before shipping:

```bash
python -m pytest -q
openenv validate --url http://127.0.0.1:7860
python baseline/baseline_agent.py --all-tasks
```

`baseline/baseline_agent.py` reads `OPENAI_API_KEY` and uses the OpenAI API directly. Set `ENV_BASE_URL` if the environment is not running on `http://localhost:7860`.

The local test suite covers:

| Area | Coverage |
|---|---|
| Model contracts | Required fields, defaults, and JSON schema shape |
| Runtime episode flow | Reset, step, task mismatch, and grader behavior |
| Contract endpoints | `/`, `/web`, `/health`, `/metadata`, `/schema`, `/mcp`, `/tasks`, `/reset`, `/step`, `/state` |
| Database behavior | Transaction vs autocommit and deterministic data hashing |
| Local realism checks | Seed sizes, DES determinism, lock profiling, and hard-task safe-pattern fallback |

## Notes

The runtime keeps a SQLite fallback path so the environment can still boot when PostgreSQL is unavailable. The fallback is designed to preserve the same external contract, even if the backend is simplified.

The FastAPI entrypoint is available as both `create_fastapi_app()` and `create_app()` in `server/app.py` so the repo stays compatible with the original build spec and the local test harness.
