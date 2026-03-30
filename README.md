# ChronoMigrate-Env
**Zero-Downtime Database Migration RL Environment**

An OpenEnv-style environment for training agents to execute schema migrations
without breaking availability or data integrity.

## Environment Description
ChronoMigrate-Env simulates a live database migration workflow. An agent
receives the current schema, the target schema, and feedback about lock
impact, schema progress, and data safety, then issues SQL to move the
database toward the target state.

The current repository implements:
- `POST /reset`, `POST /step`, and `GET /state`
- `GET /tasks`, `POST /grader`, and `POST /baseline`
- A deterministic DES-style lock simulator
- PostgreSQL-first execution with SQLite fallback
- Three migration tasks with seeded data and task-specific graders

## Action Space
| Field | Type | Description |
|---|---|---|
| `sql` | string | Raw SQL statement to execute |
| `task_id` | string | Task being solved |
| `execute_mode` | string | `transaction` or `autocommit` |

## Observation Space
| Field | Type | Description |
|---|---|---|
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
| Task | Difficulty | Description |
|---|---|---|
| `easy_add_column` | Easy | Add two columns with DEFAULT values |
| `medium_rename_fk` | Medium | Rename a primary-key column and repair foreign keys |
| `hard_repartition` | Hard | Repartition a table under load |

## Baseline Scores (GPT-4o-mini)
The baseline script is implemented in [baseline/baseline_agent.py](C:/Users/patha/Desktop/chronomigrate-env/baseline/baseline_agent.py) and uses the OpenAI API.

| Task | Score |
|---|---|
| `easy_add_column` | `~0.85` |
| `medium_rename_fk` | `~0.55` |
| `hard_repartition` | `~0.25` |

## Setup
```bash
git clone https://huggingface.co/spaces/YOUR_HF_USERNAME/chronomigrate-env
cd chronomigrate-env
docker build -t chronomigrate-env .
docker run -p 7860:7860 -e OPENAI_API_KEY=your_key chronomigrate-env
```

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
        )
    )
```

## Verification
Recommended local checks:
```bash
python -m pytest -q
openenv validate --url http://127.0.0.1:7860
python baseline/baseline_agent.py --all-tasks
```

If the environment is running locally, verify the main contract endpoints:
`/tasks`, `/reset`, `/step`, `/state`, `/grader`, and `/baseline`.

## Notes
The runtime includes a SQLite fallback path so the environment can still boot
when PostgreSQL is unavailable. The FastAPI app is implemented in
[server/app.py](C:/Users/patha/Desktop/chronomigrate-env/server/app.py), and the
core environment logic lives in
[server/chrono_migrate_env.py](C:/Users/patha/Desktop/chronomigrate-env/server/chrono_migrate_env.py).
