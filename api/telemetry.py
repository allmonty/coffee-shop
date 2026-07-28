"""OpenTelemetry setup (spec §9).

Telemetry is best-effort. Nothing in here may prevent the app from starting or
serving a request: exporters run on a background batch processor, and if the
collector is unreachable the SDK drops spans and carries on. There is a test
that boots the app with no collector at all.

Auto-instrumentation covers the boring half — HTTP and SQL spans. The agent's
own spans are written by hand in Phase 8, because that is the part worth
understanding rather than importing.
"""

from __future__ import annotations

import logging

from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from settings import settings

logger = logging.getLogger(__name__)

_configured = False


def setup_telemetry(app=None, engine=None) -> None:
    """Idempotent. Safe to call when `otel_enabled` is false — it does nothing."""
    global _configured
    if _configured or not settings.otel_enabled:
        return

    resource = Resource.create(
        {
            "service.name": settings.otel_service_name,
            "service.namespace": "coffee-shop",
        }
    )

    try:
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        tracer_provider = TracerProvider(resource=resource)
        tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        trace.set_tracer_provider(tracer_provider)

        metrics.set_meter_provider(
            MeterProvider(
                resource=resource,
                metric_readers=[PeriodicExportingMetricReader(OTLPMetricExporter())],
            )
        )
    except Exception:  # pragma: no cover - exercised by running with otel down
        # A missing or misconfigured collector must not take the shop down.
        logger.warning("telemetry setup failed; continuing without it", exc_info=True)
        _configured = True
        return

    _instrument(app, engine)
    _configured = True


def _instrument(app, engine) -> None:
    try:
        if app is not None:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

            FastAPIInstrumentor.instrument_app(app)
        if engine is not None:
            from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

            SQLAlchemyInstrumentor().instrument(engine=engine.sync_engine)
    except Exception:  # pragma: no cover
        logger.warning("auto-instrumentation failed; continuing", exc_info=True)


def get_tracer(name: str = "coffee-shop"):
    return trace.get_tracer(name)


def get_meter(name: str = "coffee-shop"):
    return metrics.get_meter(name)
