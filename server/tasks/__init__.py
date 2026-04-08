from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple


GradeFunction = Callable[..., float]
SCORE_MIN = 1e-3
SCORE_MAX = 0.998
SCORE_EPSILON = SCORE_MIN
GRADER_REQUIRED_KEYS = (
    "current_schema_ddl",
    "target_schema_ddl",
    "data_hash_before",
    "data_hash_after",
    "availability_pct",
)


def normalize_task_score(score: float) -> float:
    clamped = max(SCORE_MIN, min(SCORE_MAX, float(score)))
    return round(clamped, 3)


def coerce_grader_inputs(*args: object, **kwargs: object) -> Tuple[Dict[str, object], bool]:
    payload: Dict[str, object] = {}
    if args:
        first = args[0]
        if isinstance(first, dict):
            payload.update(first)
        elif len(args) >= len(GRADER_REQUIRED_KEYS):
            for key, value in zip(GRADER_REQUIRED_KEYS, args):
                payload[key] = value
        else:
            for key in GRADER_REQUIRED_KEYS:
                if hasattr(first, key):
                    payload[key] = getattr(first, key)
    payload.update(kwargs)

    is_complete = all(key in payload for key in GRADER_REQUIRED_KEYS)
    availability_raw = payload.get("availability_pct", 0.0)
    try:
        availability = float(availability_raw)
    except (TypeError, ValueError):
        availability = 0.0

    return (
        {
            "current_schema_ddl": str(payload.get("current_schema_ddl", "") or ""),
            "target_schema_ddl": str(payload.get("target_schema_ddl", "") or ""),
            "data_hash_before": str(payload.get("data_hash_before", "") or ""),
            "data_hash_after": str(payload.get("data_hash_after", "") or ""),
            "availability_pct": max(0.0, min(1.0, availability)),
        },
        is_complete,
    )


@dataclass(frozen=True)
class TaskDefinition:
    task_id: str
    description: str
    difficulty: str
    load_level: int
    max_steps: int
    starting_schema_sql: str
    target_schema_ddl: str
    seed_data_sql: str
    grade_fn: GradeFunction


from server.tasks.task_easy import TASK as EASY_TASK
from server.tasks.task_medium import TASK as MEDIUM_TASK
from server.tasks.task_hard import TASK as HARD_TASK

TASKS: Dict[str, TaskDefinition] = {
    EASY_TASK.task_id: EASY_TASK,
    MEDIUM_TASK.task_id: MEDIUM_TASK,
    HARD_TASK.task_id: HARD_TASK,
}
