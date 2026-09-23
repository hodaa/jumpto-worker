"""Unit tests for the centralized job timeout ladder."""

from app.core.config import Settings
from app.core.timeouts import (
    pipeline_timeout_seconds,
    task_hard_time_limit_seconds,
    task_soft_time_limit_seconds,
)


class TestTimeoutLadder:
    """Tests for the centralized job timeout/retry watchdog policy."""

    def test_pipeline_timeout_equals_job_timeout(self) -> None:
        settings = Settings(_env_file=None, job_timeout_seconds=600)
        assert pipeline_timeout_seconds(settings) == 600

    def test_soft_limit_trails_pipeline_timeout(self) -> None:
        settings = Settings(_env_file=None, job_timeout_seconds=600)
        assert task_soft_time_limit_seconds(settings) == 605

    def test_hard_limit_adds_grace(self) -> None:
        settings = Settings(
            _env_file=None,
            job_timeout_seconds=600,
            task_timeout_grace_seconds=30,
        )
        assert task_hard_time_limit_seconds(settings) == 630

    def test_ordering_invariant_holds(self) -> None:
        settings = Settings(_env_file=None)
        assert (
            pipeline_timeout_seconds(settings)
            < task_soft_time_limit_seconds(settings)
            < task_hard_time_limit_seconds(settings)
        )

    def test_limits_track_job_time_override(self) -> None:
        settings = Settings(
            _env_file=None,
            job_timeout_seconds=300,
            task_timeout_grace_seconds=45,
        )
        assert pipeline_timeout_seconds(settings) == 300
        assert task_soft_time_limit_seconds(settings) == 305
        assert task_hard_time_limit_seconds(settings) == 345
