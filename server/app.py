import json
import os
import subprocess
import sys
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

try:
    from openenv.core.server import create_fastapi_app as create_openenv_fastapi_app
except Exception:
    create_openenv_fastapi_app = None

from models import MigrationAction, MigrationObservation, MigrationState
from server.chrono_migrate_env import ChronoMigrateEnv
from server.tasks import TASKS

ENV_NAME = "chronomigrate-env"
ENV_VERSION = "0.1.0"
ENV_DESCRIPTION = (
    "An RL environment that trains AI agents to execute zero-downtime "
    "database schema migrations under simulated transactional load."
)
BASELINE_TIMEOUT_SECONDS = int(os.environ.get("BASELINE_TIMEOUT_SECONDS", "180"))
ENV_TAGS = [
    "openenv",
    "database",
    "reinforcement-learning",
    "zero-downtime",
    "schema-migration",
]

env = ChronoMigrateEnv()


class MCPRequest(BaseModel):
    jsonrpc: str = "2.0"
    id: Optional[object] = None
    method: Optional[str] = None
    params: Optional[dict] = None


class GraderRequest(BaseModel):
    task_id: str
    episode_id: Optional[str] = None


def health() -> Dict[str, str]:
    return {"status": "healthy"}


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


def schema() -> Dict[str, dict]:
    return {
        "action": MigrationAction.model_json_schema(),
        "observation": MigrationObservation.model_json_schema(),
        "state": MigrationState.model_json_schema(),
    }


def mcp(request: MCPRequest) -> Dict[str, object]:
    return {
        "jsonrpc": "2.0",
        "id": request.id,
        "error": {"code": -32601, "message": "MCP disabled for this environment"},
    }


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


def reset(config: Optional[dict] = None):
    try:
        return env.reset(config or {})
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def step(action: MigrationAction):
    try:
        observation = env.step(action)
        current_state = env.state
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "observation": observation.model_dump(),
        "reward": env.last_step_reward,
        "done": current_state.done,
        "metadata": env.last_metadata,
    }


def state():
    try:
        return env.state
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


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


def grade_episode(req: GraderRequest) -> Dict:
    try:
        current_state = env.state
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if current_state.task_id != req.task_id:
        return {"score": 0.0, "feedback": "No active episode for this task"}
    if req.episode_id and req.episode_id != current_state.episode_id:
        return {"score": 0.0, "feedback": "Episode ID mismatch for this task"}

    total = current_state.total_background_queries
    failed = current_state.failed_background_queries
    availability = 1.0 - (failed / total) if total else 1.0
    data_integrity = (
        1.0
        if current_state.current_data_hash == current_state.data_integrity_hash
        else 0.0
    )
    score = current_state.schema_match_pct * availability * data_integrity
    return {
        "score": round(score, 4),
        "schema_match": round(current_state.schema_match_pct, 4),
        "availability": round(availability, 4),
        "data_integrity": data_integrity,
        "feedback": _generate_feedback(
            current_state.schema_match_pct, availability, data_integrity
        ),
    }


def _generate_feedback(
    schema_match: float, availability: float, data_integrity: float
) -> str:
    schema_complete = schema_match >= 0.999
    if data_integrity == 0.0:
        return "FAIL: Data integrity compromised. Rows dropped or corrupted."
    if schema_complete and availability >= 0.9:
        return "PASS: Perfect zero-downtime migration achieved."
    if schema_complete:
        return (
            f"PARTIAL: Schema correct but {(1 - availability) * 100:.1f}% downtime occurred."
        )
    return (
        f"PARTIAL: Schema {schema_match * 100:.1f}% complete, "
        f"availability {availability * 100:.1f}%."
    )


def run_baseline() -> Dict:
    try:
        result = subprocess.run(
            [sys.executable, "inference.py", "--all-tasks"],
            capture_output=True,
            text=True,
            timeout=BASELINE_TIMEOUT_SECONDS,
            env={**os.environ},
            check=False,
        )
        payload = _parse_subprocess_json(result.stdout)
        if result.returncode == 0:
            return {
                "baseline_scores": payload if _looks_like_score_payload(payload) else {},
                "status": "ok",
            }

        error_message = _extract_baseline_error(result.stdout, result.stderr, payload)
        return {
            "baseline_scores": {},
            "status": "error",
            "error": error_message,
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {
            "baseline_scores": {},
            "status": "error",
            "error": f"baseline timed out after {BASELINE_TIMEOUT_SECONDS} seconds",
        }
    except Exception as exc:
        return {"baseline_scores": {}, "status": "error", "error": str(exc)}


def _parse_subprocess_json(stdout: str) -> Dict:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    for line in reversed(lines):
        if line.startswith("{") and line.endswith("}"):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
    return {}


def _looks_like_score_payload(payload: Dict) -> bool:
    if not payload:
        return False
    expected_keys = {"easy_add_column", "medium_rename_fk", "hard_repartition"}
    return set(payload) == expected_keys and all(
        isinstance(score, (int, float)) for score in payload.values()
    )


def _extract_baseline_error(stdout: str, stderr: str, payload: Dict) -> str:
    if isinstance(payload, dict) and payload.get("error"):
        return str(payload["error"])
    if stderr.strip():
        return stderr.strip().splitlines()[-1]
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if lines:
        return lines[-1]
    return "baseline subprocess failed without output"


def _register_fallback_core_routes(fastapi_app: FastAPI) -> None:
    fastapi_app.get("/health")(health)
    fastapi_app.get("/", response_class=HTMLResponse)(web)
    fastapi_app.get("/web", response_class=HTMLResponse)(web)
    fastapi_app.post("/reset", response_model=MigrationObservation)(reset)
    fastapi_app.post("/step")(step)
    fastapi_app.get("/state")(state)


def _register_common_routes(fastapi_app: FastAPI) -> None:
    fastapi_app.get("/metadata")(metadata)
    fastapi_app.get("/schema")(schema)
    fastapi_app.post("/mcp")(mcp)
    fastapi_app.get("/tasks")(list_tasks)
    fastapi_app.post("/grader")(grade_episode)
    fastapi_app.post("/baseline")(run_baseline)


def create_fastapi_app() -> FastAPI:
    if create_openenv_fastapi_app is not None:
        try:
            fastapi_app = create_openenv_fastapi_app(
                env, MigrationAction, MigrationObservation
            )
        except Exception:
            fastapi_app = FastAPI(
                title="ChronoMigrate-Env",
                version=ENV_VERSION,
                description=ENV_DESCRIPTION,
            )
            _register_fallback_core_routes(fastapi_app)
    else:
        fastapi_app = FastAPI(
            title="ChronoMigrate-Env",
            version=ENV_VERSION,
            description=ENV_DESCRIPTION,
        )
        _register_fallback_core_routes(fastapi_app)

    _register_common_routes(fastapi_app)
    return fastapi_app


def create_app() -> FastAPI:
    return create_fastapi_app()


def main() -> None:
    import uvicorn

    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


app = create_fastapi_app()


if __name__ == "__main__":
    main()
