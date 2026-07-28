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
