import json
import subprocess
import sys
from typing import Dict, List, Optional

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
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
router = APIRouter()


def create_fastapi_app() -> FastAPI:
    fastapi_app = FastAPI(
        title="ChronoMigrate-Env",
        version=ENV_VERSION,
        description=ENV_DESCRIPTION,
    )
    fastapi_app.include_router(router)
    return fastapi_app


def create_app() -> FastAPI:
    return create_fastapi_app()


@router.get("/health")
def health() -> Dict[str, str]:
    return {"status": "healthy"}


@router.get("/metadata")
def metadata() -> Dict[str, object]:
    return {
        "name": ENV_NAME,
        "version": ENV_VERSION,
        "description": ENV_DESCRIPTION,
        "action": "MigrationAction",
        "observation": "MigrationObservation",
        "state": "MigrationState",
        "runtime": "docker",
        "tags": ENV_TAGS,
        "mcp_enabled": False,
    }


@router.get("/schema")
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


@router.post("/mcp")
def mcp(request: MCPRequest) -> Dict[str, object]:
    return {
        "jsonrpc": "2.0",
        "id": request.id,
        "error": {"code": -32601, "message": "MCP disabled for this environment"},
    }


@router.get("/", response_class=HTMLResponse)
@router.get("/web", response_class=HTMLResponse)
def web() -> str:
    tasks = "".join(
        f"<li><strong>{task.task_id}</strong>: {task.description}</li>"
        for task in TASKS.values()
    )
    return f"""
    <html>
      <head>
        <title>ChronoMigrate-Env</title>
        <style>
          body {{
            font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
            background: #f4f1ea;
            color: #1d1d1d;
            margin: 0;
            padding: 2rem;
          }}
          main {{
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border: 1px solid #d8d1c5;
            padding: 2rem;
            border-radius: 16px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.05);
          }}
          code {{ background: #f1ece2; padding: 0.15rem 0.35rem; border-radius: 6px; }}
        </style>
      </head>
      <body>
        <main>
          <h1>ChronoMigrate-Env</h1>
          <p>{ENV_DESCRIPTION}</p>
          <p>Core endpoints: <code>/reset</code>, <code>/step</code>, <code>/state</code>, <code>/tasks</code>, <code>/grader</code>, <code>/baseline</code>.</p>
          <h2>Tasks</h2>
          <ul>{tasks}</ul>
        </main>
      </body>
    </html>
    """


@router.post("/reset", response_model=MigrationObservation)
def reset(config: Optional[dict] = None):
    try:
        return env.reset(config or {})
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/step")
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


@router.get("/state")
def state():
    try:
        return env.state()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/tasks")
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


@router.post("/grader")
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


@router.post("/baseline")
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


app = create_fastapi_app()
