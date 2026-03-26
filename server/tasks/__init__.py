from dataclasses import dataclass
from typing import Callable, Dict


GradeFunction = Callable[[str, str, str, str, float], float]


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
