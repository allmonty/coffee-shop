"""The SSE frame sequence, asserted with a scripted model (spec §7.2)."""

import json
import uuid

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport
from langgraph.checkpoint.memory import InMemorySaver
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
    def install(script, cashier=None, barista=None):
        graph = build_graph(
            llm=FakeToolCallingModel(script),
            cashier_llm=FakeToolCallingModel(cashier) if cashier is not None else None,
            barista_llm=FakeToolCallingModel(barista) if barista is not None else None,
        )
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
            tool_call("ring_up", request="charge them", quoted_total_cents=400),
            says("Enjoy."),
        ],
        cashier=[tool_call("charge_the_customer"), says("Paid.")],
    )

    response = await client.post(
        "/api/chat", json={"visit_id": entered["visit_id"], "message": "small latte please"}
    )

    wallets = [f for f in frames(response.text) if f["type"] == "wallet_updated"]
    assert wallets[-1]["wallet_cents"] == 1600


async def test_going_home_ends_the_visit(client, session_factory, scripted):
    entered = await _enter(client, session_factory)
    scripted(
        [tool_call("ring_up", request="they are off", going_home=True), says("See you tomorrow.")],
        cashier=[tool_call("send_them_home"), says("Night.")],
    )

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


async def test_the_endpoint_remembers_the_previous_turn(client, session_factory, monkeypatch):
    """Two POSTs, one visit: the second turn must be able to see the first.

    This is the web path's version of what `scripts/shop_cli.py` has always done
    with `open_checkpointer()`. Without a checkpointer the graph starts from an
    empty state on every request, so "a latte" → "which size?" → "large" loses
    the latte, and the barista is amnesiac in the browser while working fine in
    the terminal.
    """
    entered = await _enter(client, session_factory)
    model = FakeToolCallingModel([says("Which size?"), says("One large latte.")])
    graph = build_graph(llm=model, checkpointer=InMemorySaver())
    monkeypatch.setattr(chat_router, "get_graph", lambda: graph)

    for message in ("a latte", "large"):
        await client.post("/api/chat", json={"visit_id": entered["visit_id"], "message": message})

    # What the model was handed on the second turn, minus the system prompt.
    second_turn = [m.content for m in model.calls[1]]

    assert "a latte" in second_turn  # the customer's first message survived
    assert "Which size?" in second_turn  # and so did the barista's own reply


def test_the_graph_is_compiled_with_the_installed_checkpointer(monkeypatch):
    """Regression: `build_graph(checkpointer=None)` was hardcoded here.

    The behavioural test above passes a graph in directly, so it cannot catch a
    router that compiles its own graph without the store.
    """
    captured = {}

    def fake_build_graph(llm=None, checkpointer=None):
        captured["checkpointer"] = checkpointer
        return object()

    monkeypatch.setattr(chat_router, "build_graph", fake_build_graph)
    saver = InMemorySaver()
    chat_router.set_checkpointer(saver)
    try:
        chat_router.get_graph()
    finally:
        chat_router.set_checkpointer(None)

    assert captured["checkpointer"] is saver


async def test_the_decision_record_reaches_the_browser(client, session_factory, scripted):
    """What the agent did, paired call-to-result, in the same stream as the prose."""
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

    results = [f for f in frames(response.text) if f["type"] == "tool_result"]
    assert len(results) == 1
    assert results[0]["tool"] == "add_to_cart"
    assert results[0]["args"]["size"] == "large"
    assert results[0]["ok"] is True
    assert results[0]["steps"] == []


async def test_a_failed_tool_shows_its_error_and_message(client, session_factory, scripted):
    """The panel has to show refusals — they are the interesting half."""
    entered = await _enter(client, session_factory)
    scripted(
        [
            tool_call("add_to_cart", item_name="Latte", quantity=1, size="large"),
            tool_call("ring_up", request="charge them", quoted_total_cents=620),
            says("Sorry, my mistake — it's $5.20."),
        ],
        cashier=[tool_call("charge_the_customer"), says("Wrong total.")],
    )

    response = await client.post(
        "/api/chat", json={"visit_id": entered["visit_id"], "message": "a large latte, pay now"}
    )

    refused = [f for f in frames(response.text) if f["type"] == "tool_result" and not f["ok"]]
    assert len(refused) == 1
    # The refusal happened inside the delegation, and must still surface as a
    # failed ring_up — otherwise the waiter announces an order nobody paid for.
    assert refused[0]["tool"] == "ring_up"
    assert refused[0]["error"] == "total_mismatch"
    step = refused[0]["steps"][-1]
    assert step["tool"] == "charge_the_customer"
    assert "5.20" in step["message"]


async def test_each_tool_result_is_reported_exactly_once(client, session_factory, scripted):
    """A values snapshot arrives per node, so the naive version repeats itself."""
    entered = await _enter(client, session_factory)
    scripted(
        [
            tool_call("add_to_cart", item_name="Latte", quantity=1, size="large"),
            tool_call("add_to_cart", item_name="Chocolate Chip Cookie", quantity=1),
            says("Done."),
        ]
    )

    response = await client.post(
        "/api/chat", json={"visit_id": entered["visit_id"], "message": "latte and a cookie"}
    )

    results = [f for f in frames(response.text) if f["type"] == "tool_result"]
    assert len(results) == 2


async def test_the_turn_reports_how_many_model_calls_it_cost(client, session_factory, scripted):
    """Two inferences for one sentence is invisible without this."""
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

    stats = [f for f in frames(response.text) if f["type"] == "turn_stats"]
    assert stats[0]["loop_count"] == 2


async def test_a_later_turn_does_not_replay_earlier_tool_calls(
    client, session_factory, monkeypatch
):
    """`messages` is the whole checkpointed thread, not this turn's slice.

    Without seeding from the first snapshot, turn two re-reports turn one's
    calls as if they had just happened — and a one-turn probe cannot show it.
    """
    entered = await _enter(client, session_factory)
    graph = build_graph(
        llm=FakeToolCallingModel(
            [
                tool_call("add_to_cart", item_name="Latte", quantity=1, size="large"),
                says("One large latte."),
                tool_call("add_to_cart", item_name="Chocolate Chip Cookie", quantity=1),
                says("And a cookie."),
            ]
        ),
        checkpointer=InMemorySaver(),
    )
    monkeypatch.setattr(chat_router, "get_graph", lambda: graph)

    first = await client.post(
        "/api/chat", json={"visit_id": entered["visit_id"], "message": "a large latte"}
    )
    second = await client.post(
        "/api/chat", json={"visit_id": entered["visit_id"], "message": "a cookie too"}
    )

    assert [f["tool"] for f in frames(first.text) if f["type"] == "tool_result"] == ["add_to_cart"]
    reported = [f for f in frames(second.text) if f["type"] == "tool_result"]
    assert len(reported) == 1
    assert reported[0]["args"]["item_name"] == "Chocolate Chip Cookie"


async def test_a_delegations_internal_calls_reach_the_browser(client, session_factory, scripted):
    """Otherwise ask_barista is an opaque box in the one panel meant to show
    what happened."""
    entered = await _enter(client, session_factory)
    scripted(
        [tool_call("ask_barista", request="a large latte with oat"), says("Coming up.")],
        barista=[
            tool_call(
                "add_to_cart", item_name="Latte", quantity=1, size="large", modifiers=["oat_milk"]
            ),
            says("In."),
        ],
    )

    response = await client.post(
        "/api/chat", json={"visit_id": entered["visit_id"], "message": "large oat latte"}
    )

    results = [f for f in frames(response.text) if f["type"] == "tool_result"]
    assert [f["tool"] for f in results] == ["ask_barista"]
    assert results[0]["agent"] == "barista"
    assert [step["tool"] for step in results[0]["steps"]] == ["add_to_cart"]
    assert results[0]["steps"][0]["agent"] == "barista"


async def test_the_go_home_button_still_ends_the_visit(client, session_factory, scripted):
    """The path that lost its tool.

    The Go Home button is a first-class control and the waiter no longer has
    end_visit, so this turn now has to route through Val.
    """
    entered = await _enter(client, session_factory)
    scripted(
        [tool_call("ring_up", request="they pressed Go Home", going_home=True), says("Night!")],
        cashier=[tool_call("send_them_home"), says("Closed out.")],
    )

    response = await client.post(
        "/api/chat", json={"visit_id": entered["visit_id"], "event": "go_home"}
    )

    assert any(f["type"] == "visit_ended" for f in frames(response.text))


async def test_a_premature_reply_is_withdrawn_before_the_real_one(
    client, session_factory, scripted
):
    """Models talk in the same message as a tool call, and both reach the
    browser — so the customer reads a guess followed by an answer.

    A prompt rule does not hold this at temperature 0, so the stream carries an
    explicit retraction instead.
    """
    entered = await _enter(client, session_factory)
    scripted(
        [
            tool_call("add_to_cart", item_name="Latte", quantity=1, size="large"),
            says("One large latte, $5.20."),
        ]
    )

    response = await client.post(
        "/api/chat", json={"visit_id": entered["visit_id"], "message": "a large latte"}
    )

    kinds = [f["type"] for f in frames(response.text)]
    assert "reset_reply" in kinds
    # It has to arrive before the reply that replaces it.
    assert kinds.index("reset_reply") < len(kinds) - kinds[::-1].index("token") - 1


async def test_a_turn_with_no_tools_never_withdraws_anything(client, session_factory, scripted):
    """Otherwise a plain answer would flicker for no reason."""
    entered = await _enter(client, session_factory)
    scripted([says("What can I get you?")])

    response = await client.post(
        "/api/chat", json={"visit_id": entered["visit_id"], "message": "hi"}
    )

    assert "reset_reply" not in [f["type"] for f in frames(response.text)]


async def test_the_chat_log_only_ever_speaks_as_sam(client, session_factory, scripted):
    """Three roles, one voice (spec §13.11).

    Mo and Val show up in the decision panel and in Sam's own words; they never
    become a second speaker in the conversation itself.
    """
    entered = await _enter(client, session_factory)
    scripted(
        [tool_call("ask_barista", request="a large latte with oat"), says("Mo's on it — $5.80.")],
        barista=[
            tool_call(
                "add_to_cart", item_name="Latte", quantity=1, size="large", modifiers=["oat_milk"]
            ),
            says("Large oat latte in."),
        ],
    )

    response = await client.post(
        "/api/chat", json={"visit_id": entered["visit_id"], "message": "large oat latte"}
    )

    spoken = "".join(f["text"] for f in frames(response.text) if f["type"] == "token")
    # Mo's own line reached the panel, not the conversation.
    assert "Large oat latte in." not in spoken
    results = [f for f in frames(response.text) if f["type"] == "tool_result"]
    assert results[0]["agent"] == "barista"


async def test_go_home_ends_the_visit_even_if_the_model_just_says_goodbye(
    client, session_factory, scripted
):
    """The button must not depend on the model choosing a tool.

    It exists precisely so leaving is unambiguous rather than something the
    model has to read out of "bye" (§13.1). Putting a delegation between the
    waiter and end_visit was enough to break it: with an unpaid cart the real
    model says goodbye, calls nothing, and the customer is stuck in the shop
    with the day never advancing.
    """
    entered = await _enter(client, session_factory)
    scripted([says("Great visit! Have a nice day.")])

    response = await client.post(
        "/api/chat", json={"visit_id": entered["visit_id"], "event": "go_home"}
    )

    events = frames(response.text)
    assert any(f["type"] == "visit_ended" for f in events)
    assert events[-1] == {"type": "done", "visit_ended": True}


async def test_go_home_leaves_an_unpaid_order_behind(client, session_factory, scripted):
    """Walking out without paying is a thing customers do. The wallet is not
    charged and the day still advances."""
    entered = await _enter(client, session_factory)
    scripted([tool_call("add_to_cart", item_name="Latte", quantity=1, size="large"), says("Sure.")])
    await client.post("/api/chat", json={"visit_id": entered["visit_id"], "message": "a latte"})

    scripted([says("See you tomorrow!")])
    response = await client.post(
        "/api/chat", json={"visit_id": entered["visit_id"], "event": "go_home"}
    )

    ended = [f for f in frames(response.text) if f["type"] == "visit_ended"]
    assert ended, "the visit must close even with an unpaid cart"
    assert ended[0]["wallet_cents"] == 2000


async def test_go_home_does_not_double_end_when_the_cashier_already_did(
    client, session_factory, scripted
):
    """The backstop is a backstop, not a second close-out.

    `visit_ended` frames are emitted per state snapshot and so can repeat — that
    predates this and the UI is idempotent about it. What must not happen is the
    day advancing twice.
    """
    entered = await _enter(client, session_factory)
    scripted(
        [tool_call("ring_up", request="they are off", going_home=True), says("Night!")],
        cashier=[tool_call("send_them_home"), says("Closed out.")],
    )

    response = await client.post(
        "/api/chat", json={"visit_id": entered["visit_id"], "event": "go_home"}
    )

    assert any(f["type"] == "visit_ended" for f in frames(response.text))

    # One press, one day. Two close-outs would land them on day 3.
    again = (await client.post("/api/enter", json={"name": "Allan"})).json()
    assert again["day"] == 2
