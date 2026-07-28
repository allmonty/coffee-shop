"""OpenTelemetry setup (spec §9).

Telemetry is best-effort. Nothing in here may prevent the app from starting or
serving a request: exporters run on a background batch processor, and if the
collector is unreachable the SDK drops spans and carries on. There is a test
that boots the app with no collector at all.

Auto-instrumentation covers the boring half — HTTP and SQL spans. The agent's
own spans are written by hand in Phase 8, because that is the part worth
understanding rather than importing.

All three signals are wired here. Logs matter for a reason that is easy to
underrate: the OTel `LoggingHandler` stamps every record with the *active*
`trace_id` and `span_id`, which is what makes Grafana's Loki→Tempo pivot work.
Without it you get logs you cannot correlate, or — as was the case here — no
logs in Grafana at all, because nothing was ever exported.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar

from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from settings import settings

logger = logging.getLogger(__name__)

_configured = False
_log_handler: logging.Handler | None = None

# Correlation across the turns of one visit (spec §9.5).
#
# A visit is one in-game day, so `visit_id` is the key that answers "show me
# everything that happened to this customer today". Deliberately NOT done by
# holding one trace open for the whole visit: a span is only exported when it
# ends, so a day-long trace shows you nothing until the customer goes home and
# loses everything if the process restarts. A queryable attribute on every span
# and every log record gives the same view, while each turn stays a bounded,
# individually readable trace.
CORRELATION_FIELDS = ("visit_id", "user_id", "day")

_correlation: ContextVar[dict[str, object] | None] = ContextVar("correlation", default=None)


@contextmanager
def correlation_scope(**fields: object):
    """Tag every span and log record produced inside this block.

    Values propagate into LangGraph's node tasks because asyncio copies the
    context when a task is created — the same mechanism that makes the node
    spans children of `agent.turn`.
    """
    clean = {k: v for k, v in fields.items() if k in CORRELATION_FIELDS and v is not None}
    token = _correlation.set({**current_correlation(), **clean})
    try:
        yield
    finally:
        _correlation.reset(token)


def current_correlation() -> dict[str, object]:
    return _correlation.get() or {}


class _CorrelationSpanProcessor(SpanProcessor):
    """Stamps the active correlation onto every span as it starts.

    Subclasses the SDK's `SpanProcessor` rather than duck-typing it: the
    provider also calls private hooks (`_on_ending`) that only the base class
    supplies, and a bare object raises `AttributeError` the moment a span ends.

    On *start*, not on end, because attributes have to be set before the span is
    exported — and doing it here means auto-instrumented spans (the SQL ones we
    never wrote) get tagged too, for free.
    """

    def on_start(self, span, parent_context=None) -> None:
        for key, value in current_correlation().items():
            span.set_attribute(key, value)


class _CorrelationLogFilter(logging.Filter):
    """Same fields onto every log record, so Loki can filter a whole visit.

    Always sets every field, empty when out of scope: a formatter referencing
    `%(visit_id)s` must not raise on the one log line emitted outside a turn.
    An explicit `extra=` on the call site wins, so a caller can still override.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        active = current_correlation()
        for key in CORRELATION_FIELDS:
            if not hasattr(record, key):
                setattr(record, key, active.get(key, ""))
        return True


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
        # Before the exporting processor: `on_start` hooks run in registration
        # order, and the attributes have to be on the span by the time the batch
        # processor sees it end.
        tracer_provider.add_span_processor(_CorrelationSpanProcessor())
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

    setup_logging()
    _instrument(app, engine)
    _configured = True


def setup_logging() -> None:
    """Bridge stdlib `logging` to OTLP, and to stdout (spec §9.5).

    Two handlers on the root logger, on purpose:

    - the OTel one, so records reach Loki carrying the active trace context;
    - a plain stream one, so `docker compose logs api` still shows something
      when the collector is down. Losing your logs because your log *shipper*
      is down is a bad trade.

    Idempotent — a second call replaces the handler rather than doubling every
    line, which is what happens when a test re-runs setup.
    """
    global _log_handler

    root = logging.getLogger()
    root.setLevel(settings.log_level.upper())

    correlation = _CorrelationLogFilter()

    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        stream = logging.StreamHandler()
        stream.setFormatter(
            logging.Formatter("%(levelname)s %(name)s [visit=%(visit_id)s]: %(message)s")
        )
        stream.addFilter(correlation)
        root.addHandler(stream)

    if _log_handler is not None:
        root.removeHandler(_log_handler)
        _log_handler = None

    try:
        from opentelemetry._logs import set_logger_provider
        from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
        from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
        from opentelemetry.sdk._logs.export import BatchLogRecordProcessor

        provider = LoggerProvider(
            resource=Resource.create(
                {
                    "service.name": settings.otel_service_name,
                    "service.namespace": "coffee-shop",
                }
            )
        )
        provider.add_log_record_processor(BatchLogRecordProcessor(OTLPLogExporter()))
        set_logger_provider(provider)

        _log_handler = LoggingHandler(level=logging.NOTSET, logger_provider=provider)
        _log_handler.addFilter(correlation)
        root.addHandler(_log_handler)
    except Exception:  # pragma: no cover - the logs SDK is the least stable of the three
        # §9.5 anticipated this: traces carry the real weight, so a broken log
        # bridge degrades to stdout rather than taking the process with it.
        logger.warning("log bridge setup failed; logging to stdout only", exc_info=True)


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
