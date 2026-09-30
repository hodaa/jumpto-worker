"""Unit tests for the Celery app's broker message-budget settings.

CloudAMQP bills ``published + delivered`` against a monthly quota, so Celery's
default event and cluster traffic is a billable cost on top of the real tasks.
These tests pin the settings that keep the worker inside quota; they are here
because their absence is invisible at runtime and silently exhausts the quota.
"""

from pathlib import Path

from app.tasks.celery_app import celery_app

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROD_COMPOSE = _REPO_ROOT / "docker-compose.prod.yml"


def _prod_worker_command() -> list[str]:
    """Return the worker's argv from docker-compose.prod.yml, comments stripped."""
    block = _PROD_COMPOSE.read_text().split("      command:", 1)[1]
    args: list[str] = []
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not line.startswith("        - "):
            break
        args.append(stripped.removeprefix("- "))
    return args


class TestBrokerMessageBudget:
    """Settings that directly reduce billed publishes and deliveries."""

    def test_task_events_disabled(self) -> None:
        # Each task otherwise publishes received/started/succeeded events.
        assert celery_app.conf.worker_send_task_events is False

    def test_event_queue_expires(self) -> None:
        # Drops empty celeryev.* queues instead of letting them accumulate.
        assert celery_app.conf.event_queue_expires == 60

    def test_amqp_heartbeats_disabled(self) -> None:
        # CloudAMQP keeps low TCP keep-alive intervals, so these are redundant.
        assert celery_app.conf.broker_heartbeat is None

    def test_broker_pool_limited_to_a_single_connection(self) -> None:
        # This worker only consumes; one pooled connection is enough.
        assert celery_app.conf.broker_pool_limit == 1

    def test_result_backend_disabled(self) -> None:
        # An AMQP result backend creates a queue per result.
        assert celery_app.conf.result_backend is None


class TestWorkerDeliveryGuarantees:
    """Settings that stop redeliveries from re-billing the same work."""

    def test_prefetch_is_bounded(self) -> None:
        assert celery_app.conf.worker_prefetch_multiplier == 1

    def test_acks_are_late(self) -> None:
        assert celery_app.conf.task_acks_late is True


class TestProdWorkerCommandDisablesClusterTraffic:
    """The --without-* flags live in compose, not in app config."""

    def test_gossip_mingle_and_heartbeat_disabled(self) -> None:
        command = _prod_worker_command()
        for flag in ("--without-gossip", "--without-mingle", "--without-heartbeat"):
            assert flag in command

    def test_flags_follow_the_worker_subcommand(self) -> None:
        # Celery 5 enforces argv order; a flag before `worker` is a startup error.
        command = _prod_worker_command()
        worker_index = command.index("worker")
        for flag in ("--without-gossip", "--without-mingle", "--without-heartbeat"):
            assert command.index(flag) > worker_index
