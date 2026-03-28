"""Type-safe client wrapper for ChronoMigrate-Env."""

try:
    from openenv.core.client import HTTPEnvClient
    from openenv.core.models import StepResult
except Exception:
    class HTTPEnvClient:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("openenv-core is required to use ChronoMigrateClient.")

    class StepResult:
        def __init__(self, observation, reward, done, metadata):
            self.observation = observation
            self.reward = reward
            self.done = done
            self.metadata = metadata

from models import MigrationAction, MigrationObservation, MigrationState


class ChronoMigrateClient(HTTPEnvClient):
    """Thin adapter that converts raw JSON payloads into typed models."""

    def _step_payload(self, action: MigrationAction) -> dict:
        return action.model_dump()

    def _parse_result(self, data: dict):
        observation = data.get("observation", {})
        return StepResult(
            observation=MigrationObservation(**observation),
            reward=data["reward"],
            done=data["done"],
            metadata=data.get("metadata", {}),
        )

    def _parse_state(self, data: dict) -> MigrationState:
        return MigrationState(**data)
