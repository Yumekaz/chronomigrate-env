import json
import subprocess
import sys
from typing import Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from models import MigrationAction, MigrationObservation
from server.chrono_migrate_env import ChronoMigrateEnv
from server.tasks import TASKS

try:
    from openenv.core.server import create_fastapi_app
except Exception:
    def create_fastapi_app(env, action_model, observation_model):
        app = FastAPI(title="ChronoMigrate-Env")

        @app.get("/health")
        def health() -> Dict[str, str]:
            return {"status": "ok"}

        @app.post("/reset", response_model=observation_model)
        def reset(config: Optional[dict] = None):
            return env.reset(config or {})

        @app.post("/step")
        def step(action: action_model):
            observation = env.step(action)
            state = env.state()
            return {
                "observation": observation.model_dump(),
                "reward": env.last_step_reward,
                "done": state.done,
                "metadata": env.last_metadata,
            }

        @app.get("/state")
        def state():
            return env.state()

        return app


env = ChronoMigrateEnv()
app = create_fastapi_app(env, MigrationAction, MigrationObservation)


@app.get("/tasks")
def list_tasks() -> List[Dict]:
    return [
        {
            "id": task_id,
            "description": task.description,
            "difficulty": task.difficulty,
            "max_steps": task.max_steps,
            "action_schema": {
                "sql": "string - SQL statement to execute",
                "task_id": f'string - must be "{task_id}"',
                "execute_mode": "transaction or autocommit",
            },
        }
        for task_id, task in TASKS.items()
    ]


class GraderRequest(BaseModel):
    task_id: str
    episode_id: Optional[str] = None


@app.post("/grader")
def grade_episode(req: GraderRequest) -> Dict:
    state = env.state()
    if state.task_id != req.task_id:
        return {"score": 0.0, "feedback": "No active episode for this task"}

    total = state.total_background_queries
    failed = state.failed_background_queries
    availability = 1.0 - (failed / total) if total else 1.0
    task = TASKS[req.task_id]
    score = task.grade_fn(
        state.current_schema_ddl,
        state.target_schema_ddl,
        state.data_integrity_hash,
        state.current_data_hash,
        availability,
    )
    return {
        "score": round(score, 4),
        "schema_match": round(state.schema_match_pct, 4),
        "availability": round(availability, 4),
        "data_integrity": 1.0 if state.current_data_hash == state.data_integrity_hash else 0.0,
        "feedback": _generate_feedback(score, state.schema_match_pct, availability),
    }


def _generate_feedback(score: float, schema_match: float, availability: float) -> str:
    if score == 0.0:
        return "FAIL: migration either destroyed data or made no safe progress."
    if schema_match >= 0.999 and availability >= 0.9:
        return "PASS: zero-downtime migration achieved."
    return (
        f"PARTIAL: schema {schema_match * 100:.1f}% complete, "
        f"availability {availability * 100:.1f}%."
    )


@app.post("/baseline")
def run_baseline() -> Dict:
    try:
        result = subprocess.run(
            [sys.executable, "baseline/baseline_agent.py", "--all-tasks"],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        payload = json.loads(lines[-1]) if result.returncode == 0 and lines else {}
        return {"baseline_scores": payload, "status": "ok" if result.returncode == 0 else "error"}
    except Exception as exc:
        return {"baseline_scores": {}, "status": f"error: {exc}"}
