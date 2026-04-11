from typing import Any, Dict, Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from server.app import (
    GraderRequest,
    MCPRequest,
    grade_episode,
    health,
    list_tasks,
    metadata,
    mcp,
    reset as reset_env,
    run_baseline,
    schema,
    state as state_env,
    step as step_env,
    web,
)


app = FastAPI(title="ChronoMigrate-Env")


class ResetRequest(BaseModel):
    task_id: Optional[str] = "easy_add_column"
    seed: Optional[int] = 42


@app.get("/")
def root():
    return {"message": "ChronoMigrate-Env is running!"}


@app.get("/health")
def health_route():
    return health()


@app.get("/web", response_class=HTMLResponse)
def web_route():
    return web()


@app.get("/metadata")
def metadata_route():
    return metadata()


@app.get("/schema")
def schema_route():
    return schema()


@app.post("/mcp")
def mcp_route(request: MCPRequest):
    return mcp(request)


@app.get("/tasks")
def tasks_route():
    return list_tasks()


@app.post("/reset")
def reset(req: Optional[ResetRequest] = None):
    config = req.model_dump(exclude_none=True) if req is not None else {}
    return reset_env(config)


@app.post("/step")
def step(payload: Dict[str, Any]):
    return step_env(payload)


@app.post("/step/")
def step_slash(payload: Dict[str, Any]):
    return step_env(payload)


@app.get("/state")
def state():
    return state_env()


@app.post("/grader")
def grader(payload: GraderRequest):
    return grade_episode(payload)


@app.post("/baseline")
def baseline():
    return run_baseline()
