import uuid
from typing import Any, Dict, Optional

try:
    from openenv.core.env_server.interfaces import Environment
except Exception:
    class Environment:
        pass

from models import MigrationAction, MigrationObservation, MigrationState
from server.db_manager import DBManager
from server.des_simulator import DiscreteEventSimulator
from server.lock_analyzer import analyze_lock
from server.schema_grader import compute_data_hash, compute_schema_match
from server.tasks import TASKS, normalize_task_score


SCHEMA_COMPLETE_TOLERANCE = 1e-9


class ChronoMigrateEnv(Environment):
    def __init__(self):
        self.db = DBManager()
        self._state: Optional[MigrationState] = None
        self.des: Optional[DiscreteEventSimulator] = None
        self.current_task = None
        self.action_history = []
        self.last_step_reward = 0.0
        self.last_step_downtime_pct = 0.0
        self.last_metadata: Dict[str, Any] = {}

    def reset(self, config: Optional[dict] = None) -> MigrationObservation:
        config = config or {}
        task_id = config.get("task_id", "easy_add_column")
        seed = int(config.get("seed", 42))
        if task_id not in TASKS:
            raise ValueError(f"Unknown task_id: {task_id}")
        self.current_task = TASKS[task_id]
        self.db.reset_to_schema(
            self.current_task.starting_schema_sql,
            self.current_task.seed_data_sql,
        )
        initial_hash = compute_data_hash(self.db.conn, self.current_task.starting_schema_sql)
        self.des = DiscreteEventSimulator(task_load_level=self.current_task.load_level, seed=seed)
        self._state = MigrationState(
            episode_id=str(uuid.uuid4()),
            task_id=task_id,
            step_count=0,
            max_steps=self.current_task.max_steps,
            current_schema_ddl=self.db.get_schema_ddl(),
            target_schema_ddl=self.current_task.target_schema_ddl,
            total_background_queries=0,
            failed_background_queries=0,
            data_integrity_hash=initial_hash,
            current_data_hash=initial_hash,
            schema_match_pct=compute_schema_match(
                self.db.get_schema_ddl(), self.current_task.target_schema_ddl
            ),
            cumulative_reward=0.0,
            done=False,
            reward=0.0,
            db_backend=self.db.backend,
        )
        self.action_history = []
        self.last_step_reward = 0.0
        self.last_step_downtime_pct = 0.0
        self.last_metadata = {"event": "reset"}
        return self._build_observation("RESET", 0.0)

    def step(self, action: MigrationAction) -> MigrationObservation:
        if self._state is None:
            raise RuntimeError("Episode not initialized. Call reset() first.")

        if self._state.done:
            self.last_step_reward = 0.0
            self.last_step_downtime_pct = 0.0
            self.last_metadata = {"event": "episode_done"}
            self._state.reward = 0.0
            return self._build_observation("EPISODE_DONE", 0.0)

        if action.task_id != self._state.task_id:
            return self._handle_invalid_action(
                f"TASK_ID_MISMATCH: active={self._state.task_id} action={action.task_id}"
            )

        try:
            lock_profile = analyze_lock(action.sql)
            des_result = self.des.simulate_step(
                lock_profile.lock_ticks, lock_profile.failure_rate
            )
            prev_schema_match = self._state.schema_match_pct
            self.last_step_downtime_pct = des_result.downtime_pct
            try:
                success, result, _ = self.db.execute(
                    action.sql, execute_mode=action.execute_mode
                )
            except TypeError:
                success, result, _ = self.db.execute(action.sql)
            self.action_history.append(action.sql)

            current_ddl = self.db.get_schema_ddl()
            current_hash = compute_data_hash(
                self.db.conn, self.current_task.starting_schema_sql
            )
            new_schema_match = compute_schema_match(
                current_ddl, self._state.target_schema_ddl
            )
            data_integrity = (
                1.0 if current_hash == self._state.data_integrity_hash else 0.0
            )
            if lock_profile.destroys_data:
                data_integrity = 0.0

            schema_delta = max(0.0, new_schema_match - prev_schema_match)
            step_reward = schema_delta * (1 - des_result.downtime_pct) * data_integrity
            if not success:
                step_reward -= 0.05

            self._state.step_count += 1
            self._state.current_schema_ddl = current_ddl
            self._state.current_data_hash = current_hash
            self._state.schema_match_pct = new_schema_match
            self._state.total_background_queries += des_result.queries_total
            self._state.failed_background_queries += des_result.queries_failed
            self._state.cumulative_reward += step_reward
            self._state.done = (
                self._state.step_count >= self._state.max_steps
                or new_schema_match >= (1.0 - SCHEMA_COMPLETE_TOLERANCE)
            )
            availability = self._compute_availability(
                self._state.total_background_queries, self._state.failed_background_queries
            )
            self._state.reward = step_reward
            self.last_step_reward = step_reward
            self.last_metadata = {
                "lock_type": lock_profile.lock_type,
                "lock_ticks": lock_profile.lock_ticks,
                "availability": round(availability, 4),
                "data_integrity": data_integrity,
                "execute_mode": action.execute_mode,
                "actions_recorded": len(self.action_history),
            }
            return self._build_observation("SUCCESS" if success else result, step_reward)
        except Exception as exc:
            return self._handle_invalid_action(f"EXECUTION_ERROR: {exc}")

    @property
    def state(self) -> MigrationState:
        if self._state is None:
            raise RuntimeError("Episode not initialized. Call reset() first.")
        return self._state

    def _compute_availability(self, total_queries: int, failed_queries: int) -> float:
        return 1.0 - (failed_queries / total_queries) if total_queries else 1.0

    def _handle_invalid_action(self, message: str) -> MigrationObservation:
        self._state.step_count += 1
        self._state.cumulative_reward -= 0.05
        self._state.reward = -0.05
        self._state.done = self._state.step_count >= self._state.max_steps
        self.last_step_reward = -0.05
        self.last_step_downtime_pct = 0.0
        self.action_history.append(f"INVALID::{message}")
        self.last_metadata = {
            "error": message,
            "actions_recorded": len(self.action_history),
        }
        return self._build_observation(message, -0.05)

    def _build_observation(self, last_result: str, reward: float) -> MigrationObservation:
        state = self._state
        if state is None:
            raise RuntimeError("Episode not initialized. Call reset() first.")
        total = state.total_background_queries
        failed = state.failed_background_queries
        public_reward = normalize_task_score(reward)
        return MigrationObservation(
            current_schema_ddl=state.current_schema_ddl,
            target_schema_ddl=state.target_schema_ddl,
            last_sql_result=last_result,
            downtime_pct=normalize_task_score(self.last_step_downtime_pct),
            step_count=state.step_count,
            cumulative_downtime_pct=normalize_task_score((failed / total) if total else 0.0),
            task_id=state.task_id,
            schema_match_pct=normalize_task_score(state.schema_match_pct),
            episode_id=state.episode_id,
            done=state.done,
            reward=public_reward,
        )
