---
title: ChronoMigrate Env
emoji: ??
colorFrom: gray
colorTo: red
sdk: docker
pinned: false
---
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
The baseline script is implemented in `baseline/baseline_agent.py` and uses the
OpenAI API exactly as required by the build bible. The table below reflects the
measured scores from the live Hugging Face Space `/baseline` run on 2026-03-31.

| Task | Score |
|---|---|
| `easy_add_column` | `1.0000` |
| `medium_rename_fk` | `0.8700` |
| `hard_repartition` | `0.1250` |

## Setup
```bash
git clone https://huggingface.co/spaces/Tarun431/chronomigrate-env
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
docker build -t chronomigrate-env .
docker run --rm -p 7860:7860 chronomigrate-env
openenv validate --url http://127.0.0.1:7860
python baseline/baseline_agent.py --all-tasks
```

If the environment is running locally, verify the main contract endpoints:
`/tasks`, `/reset`, `/step`, `/state`, `/grader`, and `/baseline`.

Verified locally so far:
- Docker image builds successfully
- Dockerized runtime boots successfully on PostgreSQL
- `openenv validate` passes against the running container
- local Qwen verification on PostgreSQL reaches `easy=1.0`, `medium=0.6731`, `hard=0.244`
- Hugging Face Space is live at `https://tarun431-chronomigrate-env.hf.space`
- public `openenv validate` passes against the HF Space
- public `/tasks`, `/reset`, `/state`, `/step`, and `/grader` respond correctly on the HF Space
- OpenAI `gpt-4o-mini` baseline is verified on the HF Space via `/baseline` with scores `easy=1.0`, `medium=0.87`, `hard=0.125`

## Notes
The runtime includes a SQLite fallback path so the environment can still boot
when PostgreSQL is unavailable. The FastAPI app is implemented in
`server/app.py`, and the core environment logic lives in
`server/chrono_migrate_env.py`.

