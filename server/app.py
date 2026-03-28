import json
import subprocess
import sys
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from models import MigrationAction, MigrationObservation, MigrationState
from server.chrono_migrate_env import ChronoMigrateEnv
from server.tasks import TASKS

ENV_NAME = "chronomigrate-env"
ENV_VERSION = "0.1.0"
ENV_DESCRIPTION = (
    "An RL environment that trains AI agents to execute zero-downtime "
    "database schema migrations under simulated transactional load."
)
ENV_TAGS = [
    "openenv",
    "database",
    "reinforcement-learning",
    "zero-downtime",
    "schema-migration",
]


env = ChronoMigrateEnv()
app = FastAPI(
    title="ChronoMigrate-Env",
    version=ENV_VERSION,
    description=ENV_DESCRIPTION,
)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "healthy"}


@app.get("/metadata")
def metadata() -> Dict[str, object]:
    return {
        "name": ENV_NAME,
        "version": ENV_VERSION,
        "description": ENV_DESCRIPTION,
        "runtime": "docker",
        "tags": ENV_TAGS,
        "mcp_enabled": False,
    }


@app.get("/schema")
def schema() -> Dict[str, dict]:
    return {
        "action": MigrationAction.model_json_schema(),
        "observation": MigrationObservation.model_json_schema(),
        "state": MigrationState.model_json_schema(),
    }


class MCPRequest(BaseModel):
    jsonrpc: str = "2.0"
    id: Optional[object] = None
    method: Optional[str] = None
    params: Optional[dict] = None


@app.post("/mcp")
def mcp(request: MCPRequest) -> Dict[str, object]:
    return {
        "jsonrpc": "2.0",
        "id": request.id,
        "error": {"code": -32601, "message": "MCP disabled for this environment"},
    }


@app.post("/reset", response_model=MigrationObservation)
def reset(config: Optional[dict] = None):
    try:
        return env.reset(config or {})
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/step")
def step(action: MigrationAction):
    try:
        observation = env.step(action)
        state = env.state()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "observation": observation.model_dump(),
        "reward": env.last_step_reward,
        "done": state.done,
        "metadata": env.last_metadata,
    }


@app.get("/state")
def state():
    try:
        return env.state()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


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
    try:
        state = env.state()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if state.task_id != req.task_id:
        return {"score": 0.0, "feedback": "No active episode for this task"}
    if req.episode_id and req.episode_id != state.episode_id:
        return {"score": 0.0, "feedback": "Episode ID mismatch for this task"}

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
        action_history=env.action_history,
        steps_used=state.step_count,
    )
    episode_reward = env.compute_terminal_reward(
        final_schema_match=state.schema_match_pct,
        final_availability=availability,
        data_integrity=1.0 if state.current_data_hash == state.data_integrity_hash else 0.0,
        steps_used=state.step_count,
        max_steps=state.max_steps,
    )
    return {
        "score": round(score, 4),
        "episode_reward": round(episode_reward, 4),
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
