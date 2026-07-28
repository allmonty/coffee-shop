"""The SSE frame sequence, asserted with a scripted model (spec §7.2)."""

import json
import uuid

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport
from sqlalchemy import select

from agent.graph import build_graph
from deps import get_session
from main import app
from routers import chat as chat_router
from shop.models import MenuItem, VisitMenuItem
from shop.seed import seed_catalog
from tests.fakes import FakeToolCallingModel, says, tool_call


def frames(body: str) -> list[dict]:
    return [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ")]


@pytest_asyncio.fixture
async def client(session_factory):
    async def override_get_session():
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_session] = override_get_session
    # The chat endpoint builds its own session (it must outlive the response),
    # so it needs pointing at the test engine too.
    chat_router.get_session_factory = lambda: session_factory
    async with session_factory() as setup:
        await seed_catalog(setup)

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        yield http

    app.dependency_overrides.clear()
    chat_router._graph = None


@pytest.fixture
def scripted(monkeypatch):
    def install(script):
        graph = build_graph(llm=FakeToolCallingModel(script))
        monkeypatch.setattr(chat_router, "get_graph", lambda: graph)

    return install


async def _enter(client, session_factory, name="Allan"):
    entered = (await client.post("/api/enter", json={"name": name})).json()
    async with session_factory() as session:
        visit_id = uuid.UUID(entered["visit_id"])
        await session.execute(
            VisitMenuItem.__table__.delete().where(VisitMenuItem.visit_id == visit_id)
        )
        items = (
            await session.scalars(
                select(MenuItem).where(MenuItem.name.in_(["Latte", "Chocolate Chip Cookie"]))
            )
        ).all()
        for item in items:
            session.add(VisitMenuItem(visit_id=visit_id, menu_item_id=item.id))
        await session.commit()
    return entered


async def test_a_plain_reply_streams_tokens_then_done(client, session_factory, scripted):
    entered = await _enter(client, session_factory)
    scripted([says("What can I get you?")])

    response = await client.post(
        "/api/chat", json={"visit_id": entered["visit_id"], "message": "hi"}
    )

    assert response.status_code == 200
    kinds = [f["type"] for f in frames(response.text)]
    assert "token" in kinds
    assert kinds[-1] == "done"


async def test_cart_updates_arrive_in_the_same_stream(client, session_factory, scripted):
    """One stream carries prose and state, so their ordering is unambiguous."""
    entered = await _enter(client, session_factory)
    scripted(
        [
            tool_call("add_to_cart", item_name="Latte", quantity=1, size="large"),
            says("One large latte."),
        ]
    )

    response = await client.post(
        "/api/chat", json={"visit_id": entered["visit_id"], "message": "a large latte"}
    )

    events = frames(response.text)
    carts = [f for f in events if f["type"] == "cart_updated" and f.get("lines")]
    assert carts, [f["type"] for f in events]
    assert carts[-1]["total_cents"] == 520


async def test_wallet_updates_after_an_order(client, session_factory, scripted):
    entered = await _enter(client, session_factory)
    scripted(
        [
            tool_call("add_to_cart", item_name="Latte", quantity=1, size="small"),
            tool_call("place_order", confirmed_total_cents=400),
            says("Enjoy."),
        ]
    )

    response = await client.post(
        "/api/chat", json={"visit_id": entered["visit_id"], "message": "small latte please"}
    )

    wallets = [f for f in frames(response.text) if f["type"] == "wallet_updated"]
    assert wallets[-1]["wallet_cents"] == 1600


async def test_going_home_ends_the_visit(client, session_factory, scripted):
    entered = await _enter(client, session_factory)
    scripted([tool_call("end_visit", confirmed=True), says("See you tomorrow.")])

    response = await client.post(
        "/api/chat", json={"visit_id": entered["visit_id"], "event": "go_home"}
    )

    events = frames(response.text)
    assert any(f["type"] == "visit_ended" for f in events)
    assert events[-1]["type"] == "done"
    assert events[-1]["visit_ended"] is True


async def test_on_enter_makes_the_barista_speak_first(client, session_factory, scripted):
    entered = await _enter(client, session_factory)
    scripted([says("Morning! What can I get you? I can read you the menu.")])

    response = await client.post(
        "/api/chat", json={"visit_id": entered["visit_id"], "event": "on_enter"}
    )

    text = "".join(f["text"] for f in frames(response.text) if f["type"] == "token")
    assert "menu" in text.lower()


async def test_unknown_visit_is_reported_in_the_stream(client, session_factory, scripted):
    scripted([says("hi")])

    response = await client.post(
        "/api/chat",
        json={"visit_id": "6c9f1a2e-0000-0000-0000-000000000000", "message": "hi"},
    )

    assert frames(response.text)[0]["error"] == "unknown_visit"


async def test_malformed_visit_id_is_422(client, scripted):
    scripted([says("hi")])
    response = await client.post("/api/chat", json={"visit_id": "nope", "message": "hi"})
    assert response.status_code == 422


async def test_sse_buffering_is_disabled(client, session_factory, scripted):
    """nginx would otherwise deliver every token in one lump at the end."""
    entered = await _enter(client, session_factory)
    scripted([says("hi")])

    response = await client.post(
        "/api/chat", json={"visit_id": entered["visit_id"], "message": "hi"}
    )

    assert response.headers["x-accel-buffering"] == "no"
    assert response.headers["content-type"].startswith("text/event-stream")
