import json
import os
import subprocess
import sys
from threading import RLock
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from models import MigrationAction, MigrationObservation, MigrationState
from server.chrono_migrate_env import ChronoMigrateEnv
from server.tasks import TASKS, normalize_task_score

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
TASK_GRADER_PATHS = {
    "easy_add_column": "server.tasks.task_easy:easy_grader",
    "medium_rename_fk": "server.tasks.task_medium:medium_grader",
    "hard_repartition": "server.tasks.task_hard:hard_grader",
}
TASK_GRADER_ENTRYPOINTS = {
    "easy_add_column": "server.tasks.task_easy:EasyGrader",
    "medium_rename_fk": "server.tasks.task_medium:MediumGrader",
    "hard_repartition": "server.tasks.task_hard:HardGrader",
}
TASK_GRADER_ROUTES = {
    "easy_add_column": "grade/task_easy",
    "medium_rename_fk": "grade/task_medium",
    "hard_repartition": "grade/task_hard",
}


def _grader_spec(task_id: str) -> Dict[str, str]:
    return {
        "type": "python",
        "path": TASK_GRADER_PATHS[task_id],
        "callable": TASK_GRADER_PATHS[task_id],
        "entrypoint": TASK_GRADER_ENTRYPOINTS[task_id],
        "function_path": TASK_GRADER_PATHS[task_id],
        "class_path": TASK_GRADER_ENTRYPOINTS[task_id],
    }


def _grader_route(task_id: str) -> str:
    return TASK_GRADER_ROUTES[task_id]

class MCPRequest(BaseModel):
    jsonrpc: str = "2.0"
    id: Optional[object] = None
    method: Optional[str] = None
    params: Optional[dict] = None


class GraderRequest(BaseModel):
    task_id: str
    episode_id: Optional[str] = None


class TaskScopedEnv:
    """Serialize env access and keep one env per task id."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._envs: Dict[str, ChronoMigrateEnv] = {}
        self._active_task_id: Optional[str] = None
        self._last_step_reward = 0.0
        self._last_metadata: Dict[str, object] = {}

    def _resolve_task_id(self, config: Optional[dict] = None) -> Optional[str]:
        if config:
            task_id = config.get("task_id")
            if task_id:
                return str(task_id)
        return self._active_task_id

    def _get_env(self, task_id: Optional[str], create: bool = True) -> ChronoMigrateEnv:
        if task_id is not None:
            env = self._envs.get(task_id)
            if env is None and create:
                env = ChronoMigrateEnv()
                self._envs[task_id] = env
            if env is not None:
                return env

        if self._active_task_id is not None:
            env = self._envs.get(self._active_task_id)
            if env is not None:
                return env

        if not create:
            raise RuntimeError("No active episode. Call /reset first.")

        default_task_id = "__default__"
        env = self._envs.get(default_task_id)
        if env is None:
            env = ChronoMigrateEnv()
            self._envs[default_task_id] = env
        return env

    def reset(self, config: Optional[dict] = None):
        with self._lock:
            task_id = self._resolve_task_id(config)
            env = self._get_env(task_id, create=True)
            observation = env.reset(config or {})
            state = env.state
            self._active_task_id = state.task_id
            self._last_step_reward = 0.0
            self._last_metadata = {"event": "reset", "task_id": state.task_id}
            return observation

    def step(self, action: MigrationAction):
        with self._lock:
            env = self._get_env(action.task_id, create=True)
            observation = env.step(action)
            self._active_task_id = action.task_id
            self._last_step_reward = env.last_step_reward
            self._last_metadata = dict(env.last_metadata or {})
            return observation

    @property
    def state(self):
        with self._lock:
            env = self._get_env(None, create=False)
            return env.state

    @property
    def last_step_reward(self) -> float:
        with self._lock:
            return self._last_step_reward

    @property
    def last_metadata(self) -> Dict[str, object]:
        with self._lock:
            return dict(self._last_metadata)

    def episode_snapshot(self, task_id: str) -> Dict[str, object]:
        with self._lock:
            scoped_env = self._envs.get(task_id)
            if scoped_env is None:
                raise RuntimeError("No active episode for this task")
            return {
                "state": scoped_env.state,
                "action_history": list(scoped_env.action_history),
            }


env = TaskScopedEnv()


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
        observation = env.reset(config or {})
        return {"observation": observation.model_dump()}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def step(payload: Dict[str, object]):
    action_payload = payload.get("action", payload)
    if isinstance(action_payload, str):
        action_payload = {"sql": action_payload}
    if not isinstance(action_payload, dict):
        raise HTTPException(status_code=422, detail="Action payload must be a JSON object or SQL string")
    if "commands" in action_payload:
        commands = action_payload.get("commands") or []
        command = commands[0] if isinstance(commands, list) and commands else {}
        if not isinstance(command, dict):
            command = {}
        action_type = str(command.get("action_type", "wait")).lower()
        sql = (
            command.get("sql")
            or command.get("statement")
            or command.get("query")
            or "SELECT 1;"
        )
        if action_type == "wait":
            sql = "SELECT 1;"
        action_payload = {
            "sql": str(sql),
            "execute_mode": str(command.get("execute_mode", "transaction")),
        }
    if "sql" not in action_payload:
        action_payload = {"sql": "SELECT 1;"}
    else:
        action_payload = {
            "sql": action_payload.get("sql", "SELECT 1;"),
            "execute_mode": action_payload.get("execute_mode", "transaction"),
        }
    if "task_id" not in action_payload:
        try:
            action_payload = {
                **action_payload,
                "task_id": env.state.task_id,
            }
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        action = MigrationAction.model_validate(action_payload)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    try:
        observation = env.step(action)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    current_state = env.state
    snapshot = env.episode_snapshot(action.task_id)
    score_bundle = _score_bundle(current_state, snapshot["action_history"])
    last_error = None
    if observation.last_sql_result not in {"SUCCESS", "RESET", "EPISODE_DONE"}:
        last_error = observation.last_sql_result
    info = {
        "score": score_bundle["score"],
        "schema_match": score_bundle["schema_match"],
        "availability": score_bundle["availability"],
        "data_integrity": score_bundle["data_integrity"],
        "last_error": last_error,
        "success": score_bundle["score"] >= 0.9,
    }
    return {
        "observation": observation.model_dump(),
        "reward": observation.reward,
        "done": observation.done,
        "info": info,
        "metadata": info,
    }


def state():
    try:
        payload = env.state.model_dump()
        for key in ("reward", "cumulative_reward", "schema_match_pct"):
            payload[key] = normalize_task_score(float(payload[key]))
        return {"state": payload}
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def list_tasks() -> Dict[str, object]:
    task_items = [
        {
            "id": task_id,
            "description": task.description,
            "difficulty": task.difficulty,
            "max_steps": task.max_steps,
        }
        for task_id, task in TASKS.items()
    ]
    return {"tasks": task_items, "count": len(task_items)}


def _score_bundle(current_state: MigrationState, action_history: List[str]) -> Dict[str, float]:
    availability = 1.0 - (
        current_state.failed_background_queries / current_state.total_background_queries
    ) if current_state.total_background_queries else 1.0
    data_integrity = (
        1.0
        if current_state.current_data_hash == current_state.data_integrity_hash
        else 0.0
    )
    task_def = TASKS[current_state.task_id]
    raw_score = task_def.grade_fn(
        current_schema_ddl=current_state.current_schema_ddl,
        target_schema_ddl=current_state.target_schema_ddl,
        data_hash_before=current_state.data_integrity_hash,
        data_hash_after=current_state.current_data_hash,
        availability_pct=availability,
        action_history=action_history,
        steps_used=current_state.step_count,
    )
    return {
        "score": normalize_task_score(raw_score),
        "schema_match": normalize_task_score(current_state.schema_match_pct),
        "availability": normalize_task_score(availability),
        "data_integrity": normalize_task_score(data_integrity),
    }


def _grade_task(task_id: str, episode_id: Optional[str] = None) -> Dict:
    return grade_episode(GraderRequest(task_id=task_id, episode_id=episode_id))


def grade_episode(req: GraderRequest) -> Dict:
    try:
        snapshot = env.episode_snapshot(req.task_id)
    except RuntimeError as exc:
        if req.task_id not in TASKS:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        env.reset({"task_id": req.task_id, "seed": 42})
        snapshot = env.episode_snapshot(req.task_id)

    current_state = snapshot["state"]
    if current_state.task_id != req.task_id:
        return {
            "score": normalize_task_score(0.0),
            "feedback": "No active episode for this task",
        }
    if req.episode_id and req.episode_id != current_state.episode_id:
        return {
            "score": normalize_task_score(0.0),
            "feedback": "Episode ID mismatch for this task",
        }

    score_bundle = _score_bundle(current_state, snapshot["action_history"])
    return {
        "score": score_bundle["score"],
        "schema_match": score_bundle["schema_match"],
        "availability": score_bundle["availability"],
        "data_integrity": score_bundle["data_integrity"],
        "feedback": _generate_feedback(
            current_state.schema_match_pct,
            float(score_bundle["availability"]),
            float(score_bundle["data_integrity"]),
        ),
    }


def grade_task_easy() -> Dict:
    return _grade_task("easy_add_column")


def grade_task_medium() -> Dict:
    return _grade_task("medium_rename_fk")


def grade_task_hard() -> Dict:
    return _grade_task("hard_repartition")


def _generate_feedback(
    schema_match: float, availability: float, data_integrity: float
) -> str:
    schema_complete = schema_match >= 0.999
    if data_integrity == 0.0:
        return "FAIL: Data integrity compromised. Rows dropped or corrupted."
    if schema_complete and availability >= 0.9:
        return "PASS: Perfect zero-downtime migration achieved."
    if schema_complete:
        return "PARTIAL: Schema is correct, but availability dropped during migration."
    return "PARTIAL: Schema migration is incomplete."


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
                "baseline_scores": _sanitize_score_payload(payload),
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
    if set(payload) != expected_keys:
        return False
    try:
        for task_id in expected_keys:
            float(payload[task_id])
    except (TypeError, ValueError):
        return False
    return True


def _sanitize_score_payload(payload: Dict) -> Dict[str, float]:
    if not _looks_like_score_payload(payload):
        return {}

    expected_order = ["easy_add_column", "medium_rename_fk", "hard_repartition"]
    return {
        task_id: normalize_task_score(float(payload[task_id]))
        for task_id in expected_order
    }


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
    fastapi_app.post("/reset")(reset)
    fastapi_app.post("/step")(step)
    fastapi_app.post("/step/")(step)
    fastapi_app.get("/state")(state)


def _register_common_routes(fastapi_app: FastAPI) -> None:
    fastapi_app.get("/metadata")(metadata)
    fastapi_app.get("/schema")(schema)
    fastapi_app.post("/mcp")(mcp)
    fastapi_app.get("/tasks")(list_tasks)
    fastapi_app.post("/grader")(grade_episode)
    fastapi_app.get("/grade/task_easy")(grade_task_easy)
    fastapi_app.post("/grade/task_easy")(grade_task_easy)
    fastapi_app.get("/grade/task_medium")(grade_task_medium)
    fastapi_app.post("/grade/task_medium")(grade_task_medium)
    fastapi_app.get("/grade/task_hard")(grade_task_hard)
    fastapi_app.post("/grade/task_hard")(grade_task_hard)
    fastapi_app.post("/baseline")(run_baseline)


def _replace_get_route(fastapi_app: FastAPI, path: str, endpoint) -> None:
    fastapi_app.router.routes = [
        route
        for route in fastapi_app.router.routes
        if not (
            getattr(route, "path", None) == path
            and "GET" in (getattr(route, "methods", None) or set())
        )
    ]
    fastapi_app.get(path)(endpoint)


def create_fastapi_app() -> FastAPI:
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
