"""Telemetry must never be load-bearing (spec §9.6).

The assertion that matters is the second one: the app serves requests with no
collector reachable. A stack that only works when its observability backend is
healthy is worse than no observability.
"""

import httpx
from httpx import ASGITransport
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from main import app


async def test_app_serves_with_no_collector_reachable():
    """OTEL_EXPORTER_OTLP_ENDPOINT points nowhere; the shop still opens."""
    import telemetry

    telemetry._configured = False
    telemetry.setup_telemetry(app=None, engine=None)

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200


def test_spans_reach_an_in_memory_exporter():
    """The harness Phase 8 will use to assert the agent's span tree."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    tracer = provider.get_tracer("test")
    with tracer.start_as_current_span("agent.turn") as span:
        span.set_attribute("visit_id", "abc")
        with tracer.start_as_current_span("graph.node.barista"):
            pass

    names = [span.name for span in exporter.get_finished_spans()]
    assert "graph.node.barista" in names
    assert "agent.turn" in names


def test_setup_does_nothing_when_disabled(monkeypatch):
    """OTEL_ENABLED=false must be a complete bypass, not a partial one."""
    import telemetry

    monkeypatch.setattr(telemetry.settings, "otel_enabled", False)
    telemetry._configured = False

    telemetry.setup_telemetry(app=None, engine=None)

    assert telemetry._configured is False


def test_setup_is_idempotent_when_enabled(monkeypatch):
    import telemetry

    monkeypatch.setattr(telemetry.settings, "otel_enabled", True)
    telemetry._configured = False

    telemetry.setup_telemetry(app=None, engine=None)
    telemetry.setup_telemetry(app=None, engine=None)

    assert telemetry._configured is True


def test_log_records_carry_the_active_trace_id():
    """The Loki→Tempo pivot depends on this and nothing else (spec §9.5).

    Grafana's Loki datasource has a derived field matching the `trace_id` label.
    If a record leaves the process without span context attached, the log panel
    shows a line you cannot click through to the trace — which is worse than no
    log line, because it looks like it should work.
    """
    import logging

    from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
    from opentelemetry.sdk._logs.export import InMemoryLogExporter, SimpleLogRecordProcessor

    exporter = InMemoryLogExporter()
    provider = LoggerProvider()
    provider.add_log_record_processor(SimpleLogRecordProcessor(exporter))

    logger = logging.getLogger("test.trace_correlation")
    logger.setLevel(logging.INFO)
    handler = LoggingHandler(level=logging.NOTSET, logger_provider=provider)
    logger.addHandler(handler)

    tracer_provider = TracerProvider()
    tracer = tracer_provider.get_tracer("test")
    with tracer.start_as_current_span("agent.turn") as span:
        expected = span.get_span_context().trace_id
        logger.info("tool.result", extra={"tool": "place_order", "ok": False})

    logger.removeHandler(handler)

    records = exporter.get_finished_logs()
    assert len(records) == 1
    emitted = records[0].log_record
    assert emitted.trace_id == expected
    assert emitted.attributes["tool"] == "place_order"


def test_correlation_tags_every_span_in_the_scope():
    """A visit is queryable across turns without one long-lived trace (spec §9.5).

    The point of the span processor is reach: a span nobody wrote by hand — the
    SQL spans from auto-instrumentation — still carries `visit_id`, which is
    what makes `{ .visit_id = "..." }` in Tempo return the whole day.
    """
    import telemetry

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(telemetry._CorrelationSpanProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")

    with (
        telemetry.correlation_scope(visit_id="visit-1", user_id="user-1", day=3),
        tracer.start_as_current_span("agent.turn"),
        tracer.start_as_current_span("SELECT"),  # nobody tagged this one by hand
    ):
        pass

    # Outside the scope nothing is tagged — otherwise the fields would leak into
    # unrelated requests served by the same worker.
    with tracer.start_as_current_span("health"):
        pass

    tagged = {s.name: s.attributes for s in exporter.get_finished_spans()}
    assert tagged["SELECT"]["visit_id"] == "visit-1"
    assert tagged["SELECT"]["day"] == 3
    assert tagged["agent.turn"]["user_id"] == "user-1"
    assert "visit_id" not in tagged["health"]


def test_correlation_tags_log_records_and_does_not_leak():
    import logging

    import telemetry

    records: list[logging.LogRecord] = []

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = Capture()
    handler.addFilter(telemetry._CorrelationLogFilter())
    logger = logging.getLogger("test.correlation")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    with telemetry.correlation_scope(visit_id="visit-1", user_id="user-1", day=3):
        logger.info("tool.call")
    logger.info("outside")

    logger.removeHandler(handler)

    assert records[0].visit_id == "visit-1"
    assert records[0].day == 3
    # Empty rather than missing: a formatter referencing %(visit_id)s must not
    # blow up on a line emitted outside a turn.
    assert records[1].visit_id == ""


def test_setup_logging_does_not_stack_duplicate_handlers():
    """Re-running setup must not double every log line."""
    import logging

    import telemetry

    telemetry.setup_logging()
    first = list(logging.getLogger().handlers)
    telemetry.setup_logging()
    second = list(logging.getLogger().handlers)

    assert len(first) == len(second)
