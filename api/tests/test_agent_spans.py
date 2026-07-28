"""The agent's span tree (spec §9.3).

Cheap to write, and it catches the classic regression where a refactor silently
orphans the spans — the dashboards keep rendering, they just stop meaning
anything.
"""

import uuid

import pytest
from langchain_core.messages import HumanMessage
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from sqlalchemy import select

import agent.instrumentation as instrumentation
from agent.graph import build_graph
from shop.models import MenuItem, VisitMenuItem
from shop.seed import seed_catalog
from shop.service import enter
from tests.fakes import FakeToolCallingModel, says, tool_call


@pytest.fixture
def spans(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    monkeypatch.setattr(instrumentation, "tracer", provider.get_tracer("test"))
    return exporter


@pytest.fixture
async def shop(session):
    await seed_catalog(session)
    entered = await enter(session, "Allan")
    visit_id = uuid.UUID(entered.data["visit_id"])
    user_id = uuid.UUID(entered.data["user_id"])

    await session.execute(
        VisitMenuItem.__table__.delete().where(VisitMenuItem.visit_id == visit_id)
    )
    items = (await session.scalars(select(MenuItem).where(MenuItem.name == "Latte"))).all()
    for item in items:
        session.add(VisitMenuItem(visit_id=visit_id, menu_item_id=item.id))
    await session.commit()

    def run(script):
        graph = build_graph(llm=FakeToolCallingModel(script))
        return graph.ainvoke(
            {"messages": [HumanMessage(content="hi")]},
            config={
                "configurable": {
                    "session": session,
                    "visit_id": str(visit_id),
                    "user_id": str(user_id),
                }
            },
        )

    return run


async def test_a_turn_produces_the_expected_span_tree(shop, spans):
    await shop(
        [
            tool_call("add_to_cart", item_name="Latte", quantity=1, size="large"),
            says("One large latte."),
        ]
    )

    names = [span.name for span in spans.get_finished_spans()]

    assert "graph.node.load_context" in names
    assert "graph.node.barista" in names
    assert "tool.add_to_cart" in names
    assert "graph.node.finish" in names
    # Two model round-trips: one that asked for the tool, one that spoke.
    assert names.count("graph.node.barista") == 2


async def test_a_failed_tool_marks_its_own_span_only(shop, spans):
    """ "insufficient funds" is a normal outcome the agent recovers from.

    Marking it a request error would make every dashboard lie.
    """
    await shop(
        [
            tool_call("add_to_cart", item_name="Latte", quantity=1),  # no size
            says("Which size?"),
        ]
    )

    by_name = {span.name: span for span in spans.get_finished_spans()}

    tool = by_name["tool.add_to_cart"]
    assert tool.status.status_code is StatusCode.ERROR
    assert tool.attributes["tool.error"] == "size_required"
    assert tool.attributes["tool.ok"] is False

    # The nodes around it stay clean.
    assert by_name["graph.node.finish"].status.status_code is not StatusCode.ERROR


async def test_llm_spans_carry_gen_ai_attributes(shop, spans):
    await shop([says("hello")])

    barista = [s for s in spans.get_finished_spans() if s.name == "graph.node.barista"][0]

    assert barista.attributes["gen_ai.system"] == "ollama"
    assert barista.attributes["gen_ai.operation.name"] == "chat"
    assert barista.attributes["gen_ai.response.tool_calls"] == 0


async def test_tool_spans_record_success(shop, spans):
    await shop(
        [
            tool_call("add_to_cart", item_name="Latte", quantity=1, size="small"),
            says("Done."),
        ]
    )

    tool = [s for s in spans.get_finished_spans() if s.name == "tool.add_to_cart"][0]

    assert tool.attributes["tool.ok"] is True
    assert "tool.error" not in tool.attributes


async def test_loop_count_reaches_finish(shop, spans):
    """Three laps: two tool calls plus the closing reply."""
    result = await shop(
        [
            tool_call("add_to_cart", item_name="Latte", quantity=1, size="small"),
            tool_call("get_cart"),
            says("A latte, $4.00."),
        ]
    )

    assert result["loop_count"] == 3
