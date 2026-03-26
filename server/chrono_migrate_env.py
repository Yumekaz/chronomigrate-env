import uuid
from typing import Any, Dict, Optional

try:
    from openenv.core.env_server import Environment
except Exception:
    class Environment:
        pass

from models import MigrationAction, MigrationObservation, MigrationState
from server.db_manager import DBManager
from server.des_simulator import DiscreteEventSimulator
from server.lock_analyzer import analyze_lock
from server.schema_grader import compute_data_hash, compute_schema_match
from server.tasks import TASKS


class ChronoMigrateEnv(Environment):
    def __init__(self):
        self.db = DBManager()
        self._state: Optional[MigrationState] = None
        self.des: Optional[DiscreteEventSimulator] = None
        self.current_task = None
        self.last_step_reward = 0.0
        self.last_metadata: Dict[str, Any] = {}

    def reset(self, config: Optional[dict] = None) -> MigrationObservation:
        config = config or {}
        task_id = config.get("task_id", "easy_add_column")
        seed = int(config.get("seed", 42))
        self.current_task = TASKS[task_id]
        self.db.reset_to_schema(
            self.current_task.starting_schema_sql,
            self.current_task.seed_data_sql,
        )
        initial_hash = compute_data_hash(self.db.conn)
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
            db_backend=self.db.backend,
        )
        self.last_step_reward = 0.0
        self.last_metadata = {"event": "reset"}
        return self._build_observation("RESET")

    def step(self, action: MigrationAction) -> MigrationObservation:
        if self._state is None:
            raise RuntimeError("Episode not initialized. Call reset() first.")

        if self._state.done:
            self.last_step_reward = 0.0
            self.last_metadata = {"event": "episode_done"}
            return self._build_observation("EPISODE_DONE")

        lock_profile = analyze_lock(action.sql)
        des_result = self.des.simulate_step(lock_profile.lock_ticks, lock_profile.failure_rate)
        prev_schema_match = self._state.schema_match_pct
        success, result, _ = self.db.execute(action.sql)

        current_ddl = self.db.get_schema_ddl()
        current_hash = compute_data_hash(self.db.conn)
        new_schema_match = compute_schema_match(current_ddl, self._state.target_schema_ddl)
        data_integrity = 1.0 if current_hash == self._state.data_integrity_hash else 0.0
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
            self._state.step_count >= self._state.max_steps or new_schema_match >= 0.999
        )

        self.last_step_reward = step_reward
        self.last_metadata = {
            "lock_type": lock_profile.lock_type,
            "lock_ticks": lock_profile.lock_ticks,
            "availability": round(1.0 - des_result.downtime_pct, 4),
        }
        return self._build_observation("SUCCESS" if success else result)

    def state(self) -> MigrationState:
        if self._state is None:
            raise RuntimeError("Episode not initialized. Call reset() first.")
        return self._state

    def _build_observation(self, last_result: str) -> MigrationObservation:
        state = self.state()
        total = state.total_background_queries
        failed = state.failed_background_queries
        return MigrationObservation(
            current_schema_ddl=state.current_schema_ddl,
            target_schema_ddl=state.target_schema_ddl,
            last_sql_result=last_result,
            downtime_pct=(failed / total) if total else 0.0,
            step_count=state.step_count,
            cumulative_downtime_pct=(failed / total) if total else 0.0,
            task_id=state.task_id,
            schema_match_pct=state.schema_match_pct,
            episode_id=state.episode_id,
        )
