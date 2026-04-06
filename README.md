---
title: ChronoMigrate Env
emoji: 📊
colorFrom: gray
colorTo: red
sdk: docker
tags:
  - openenv
  - rl-environment
pinned: false
---

# ChronoMigrate-Env

ChronoMigrate-Env is an OpenEnv-compatible environment for training and evaluating agents that perform zero-downtime database schema migrations.

The environment is designed for realistic migration behavior rather than SQL-only correctness. Agents are rewarded for reaching the target schema while preserving availability and data integrity throughout the migration process.

This repository includes:

- an OpenEnv/HTTP runtime built with FastAPI
- three deterministic migration tasks with increasing difficulty
- step-level reward shaping and final grading logic
- a root-level baseline runner in `inference.py`
- Docker packaging for local use and Hugging Face Spaces deployment
- tests and validation commands for submission checks

## Quick Start

For a first-time reviewer, evaluator, or contributor, the shortest path is:

```bash
docker build -t chronomigrate-env .
docker run --rm -p 7860:7860 chronomigrate-env
openenv validate --url http://127.0.0.1:7860
curl http://127.0.0.1:7860/tasks
```

Public deployment:

- Hugging Face Space: `https://huggingface.co/spaces/Yumekaz/chronomigrate-env`
- Live app: `https://yumekaz-chronomigrate-env.hf.space`

## Environment Description

Real production migrations are judged by more than whether the final schema looks right. A successful migration must preserve data, minimize downtime, avoid destructive shortcuts, and still reach the target schema.

ChronoMigrate turns that problem into a deterministic evaluation environment with:

- OpenEnv-compatible `reset`, `step`, and `state` interactions
- three graded tasks with clear easy-to-hard progression
- multiplicative scoring from schema match, availability, and data integrity
- PostgreSQL-first execution with SQLite fallback
- Docker-ready deployment for Hugging Face Spaces
- baseline inference through root-level `inference.py`

## Action Space

| Field | Type | Description |
|---|---|---|
| `sql` | string | SQL statement to execute |
| `task_id` | string | Task identifier |
| `execute_mode` | string | `transaction` or `autocommit` |

## Observation Space

| Field | Type | Description |
|---|---|---|
| `done` | bool | Whether the episode has terminated |
| `reward` | float | Reward from the latest step |
| `current_schema_ddl` | string | Current schema |
| `target_schema_ddl` | string | Goal schema |
| `last_sql_result` | string | Database execution result |
| `downtime_pct` | float | Downtime caused by the latest step |
| `step_count` | int | Current step number |
| `cumulative_downtime_pct` | float | Rolling downtime signal |
| `task_id` | string | Active task |
| `schema_match_pct` | float | Progress toward target schema |
| `episode_id` | string | Episode identifier |

## Tasks

| Task | Difficulty | Max Steps | Description |
|---|---|---|---|
| `easy_add_column` | Easy | 5 | Add two defaulted columns safely |
| `medium_rename_fk` | Medium | 10 | Rename a primary key column and repair foreign key references |
| `hard_repartition` | Hard | 20 | Repartition a large table under simulated load |

Each task is deterministic for a fixed seed and is graded with task-specific logic.

## Reward Function

Each step is rewarded with a multiplicative objective:

```text
R_step = schema_match_delta * (1 - downtime_pct) * data_integrity
```

Episode grading uses the final state:

```text
score = schema_match * availability * data_integrity
```

This makes destructive shortcuts unattractive. An agent that drops data or causes severe downtime cannot recover to a high score just by matching the final schema.

## Baseline Scores

The baseline entrypoint is the root-level `inference.py` script. It uses the OpenAI client for all model calls and reads `HF_TOKEN` or `OPENAI_API_KEY`, `API_BASE_URL`, and `MODEL_NAME` from the environment.

Reference baseline configuration:

- model: `gpt-4o-mini`
- API base URL: `https://api.openai.com/v1`
- environment base URL: `http://127.0.0.1:7860`

Reference `gpt-4o-mini` scores:

| Task | Score |
|---|---|
| `easy_add_column` | `1.0000` |
| `medium_rename_fk` | `0.7873` |
| `hard_repartition` | `0.6843` |

These baseline numbers are intended as a reproducible reference point. Evaluation results may differ when the environment is tested with another model.


## Setup

### Local Python Setup

```bash
python -m venv .venv
```

Activate the environment:

- Windows PowerShell: `.venv\Scripts\Activate.ps1`
- Unix-like shells: `source .venv/bin/activate`

Install dependencies:

```bash
pip install -r requirements.txt
```

Optional environment variables for the baseline:

- `OPENAI_API_KEY` or `HF_TOKEN`
- `MODEL_NAME` default: `gpt-4o-mini`
- `API_BASE_URL` default: `https://api.openai.com/v1`
- `ENV_BASE_URL` default: `http://127.0.0.1:7860`

### Run with Docker

```bash
docker build -t chronomigrate-env .
docker run --rm -p 7860:7860 chronomigrate-env
```

### Validate the Environment

```bash
openenv validate .
openenv validate --url http://127.0.0.1:7860
```

The repository also contains a deployed Hugging Face Space that can be validated directly:

```bash
openenv validate --url https://yumekaz-chronomigrate-env.hf.space
```

## Usage

### API Endpoints

Core runtime endpoints:

- `GET /health`
- `GET /metadata`
- `GET /schema`
- `GET /tasks`
- `POST /reset`
- `POST /step`
- `GET /state`
- `POST /grader`
- `POST /baseline`

Examples:

```bash
curl http://127.0.0.1:7860/health
curl http://127.0.0.1:7860/metadata
curl http://127.0.0.1:7860/schema
curl http://127.0.0.1:7860/tasks
curl -X POST http://127.0.0.1:7860/reset -H "Content-Type: application/json" -d "{\"task_id\":\"easy_add_column\",\"seed\":42}"
curl -X POST http://127.0.0.1:7860/step -H "Content-Type: application/json" -d "{\"sql\":\"ALTER TABLE users ADD COLUMN email VARCHAR(255) DEFAULT NULL;\",\"task_id\":\"easy_add_column\",\"execute_mode\":\"transaction\"}"
curl http://127.0.0.1:7860/state
curl -X POST http://127.0.0.1:7860/grader -H "Content-Type: application/json" -d "{\"task_id\":\"easy_add_column\"}"
curl -X POST http://127.0.0.1:7860/baseline
```

Endpoint behavior summary:

- `/reset` starts a new episode for a task and returns the initial observation
- `/step` executes one SQL statement and returns the next observation plus step metadata
- `/grader` scores the current episode for the given task
- `/baseline` runs `inference.py --all-tasks` and returns task scores

### Client Usage

```python
from client import ChronoMigrateClient
from models import MigrationAction

async with ChronoMigrateClient(base_url="http://127.0.0.1:7860") as env:
    observation = await env.reset(config={"task_id": "easy_add_column", "seed": 42})
    result = await env.step(
        MigrationAction(
            sql="ALTER TABLE users ADD COLUMN email VARCHAR(255) DEFAULT NULL;",
            task_id="easy_add_column",
            execute_mode="transaction",
        )
    )
```

### Baseline Inference

Set these environment variables before running the baseline script:

- `HF_TOKEN` or `OPENAI_API_KEY`
- `API_BASE_URL` (defaults to `https://api.openai.com/v1`)
- `MODEL_NAME` (defaults to `gpt-4o-mini`)
- `ENV_BASE_URL` (defaults to `http://127.0.0.1:7860`)

The script is intended to be deterministic when run with the same environment variables and seed values.

Run:

```bash
python inference.py --all-tasks
```

The last line of the script prints a JSON score payload so it can also be invoked by the `/baseline` endpoint.

## Deployment Notes

- Hugging Face Space runtime: Docker
- Base image: `ghcr.io/meta-pytorch/openenv-base:latest`
- PostgreSQL is initialized in user space when available
- SQLite fallback is supported and validated
- On resource-constrained deployments, SQLite fallback can be enabled with `USE_SQLITE=true`

## Project Layout

```text
chronomigrate-env/
|-- openenv.yaml
|-- Dockerfile
|-- pyproject.toml
|-- requirements.txt
|-- README.md
|-- models.py
|-- client.py
|-- inference.py
|-- server/
|-- tests/
`-- scripts/
```

Important files:

- `server/app.py`: HTTP/OpenEnv runtime and baseline endpoint
- `server/chrono_migrate_env.py`: episode loop, rewards, and environment state transitions
- `server/tasks/`: task definitions and task-specific grading functions
- `server/schema_grader.py`: deterministic schema comparison logic
- `server/lock_analyzer.py`: SQL lock and downtime heuristics
- `inference.py`: root-level baseline runner

