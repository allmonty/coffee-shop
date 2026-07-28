"""Cross-visit memory (spec §6.5.1).

The rule under test throughout: never ask an LLM for a fact a GROUP BY can
produce. Structured facts are aggregated; only free-text notes are model-written,
and "nothing stood out" has to be a normal, common answer.
"""

import uuid

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent.summarize import extract_notes, render_transcript, summarize_visit
from shop.notes import MAX_NOTES, append_customer_notes
from shop.profile import customer_profile
from shop.seed import seed_catalog
from shop.service import enter
from tests.fakes import FakeToolCallingModel, says


@pytest.fixture
async def customer(session):
    await seed_catalog(session)
    entered = await enter(session, "Allan")
    return session, uuid.UUID(entered.data["user_id"])


# --- parsing the model's reply -------------------------------------------


def test_empty_list_is_a_valid_answer():
    """Most conversations should produce this. If they never do, the model is
    inventing facts."""
    assert extract_notes("[]") == []


def test_plain_json_array():
    assert extract_notes('["mentioned a new job"]') == ["mentioned a new job"]


def test_fenced_json_is_tolerated():
    reply = '```json\n["found the mocha too sweet"]\n```'
    assert extract_notes(reply) == ["found the mocha too sweet"]


def test_chatty_preamble_is_tolerated():
    reply = 'Sure! Here are the notes:\n["always comes in early"]\nHope that helps.'
    assert extract_notes(reply) == ["always comes in early"]


def test_garbage_yields_nothing_rather_than_raising():
    assert extract_notes("I could not find anything useful.") == []
    assert extract_notes("[not json") == []


def test_at_most_two_notes_are_kept():
    reply = '["a", "b", "c", "d"]'
    assert extract_notes(reply) == ["a", "b"]


def test_transcript_renders_only_speech():
    messages = [
        HumanMessage(content="a large latte"),
        AIMessage(content="", tool_calls=[{"name": "add_to_cart", "args": {}, "id": "1"}]),
        AIMessage(content="Coming up."),
    ]

    transcript = render_transcript(messages)

    assert "Customer: a large latte" in transcript
    assert "Barista: Coming up." in transcript
    assert "add_to_cart" not in transcript


# --- storage --------------------------------------------------------------


async def test_notes_are_capped_oldest_dropped_first(customer):
    session, user_id = customer

    for index in range(MAX_NOTES + 4):
        await append_customer_notes(session, user_id, [f"note {index}"])

    notes = (await customer_profile(session, user_id))["notes"]

    assert len(notes) == MAX_NOTES
    assert notes[0] == "note 4"
    assert notes[-1] == f"note {MAX_NOTES + 3}"


async def test_empty_notes_are_a_no_op(customer):
    session, user_id = customer

    result = await append_customer_notes(session, user_id, [])

    assert result.ok is True
    assert result.data["notes_added"] == 0
    assert (await customer_profile(session, user_id))["notes"] == []


async def test_blank_strings_are_discarded(customer):
    session, user_id = customer

    await append_customer_notes(session, user_id, ["  ", "", "real note"])

    assert (await customer_profile(session, user_id))["notes"] == ["real note"]


# --- the summarization pass -----------------------------------------------


async def test_summarize_stores_what_the_model_extracted(customer):
    session, user_id = customer
    llm = FakeToolCallingModel([says('["mentioned starting a new job"]')])

    notes = await summarize_visit(
        session,
        user_id,
        [HumanMessage(content="just started a new job"), AIMessage(content="Congratulations!")],
        llm=llm,
    )

    assert notes == ["mentioned starting a new job"]
    assert (await customer_profile(session, user_id))["notes"] == ["mentioned starting a new job"]


async def test_summarize_stores_nothing_when_nothing_stood_out(customer):
    session, user_id = customer
    llm = FakeToolCallingModel([says("[]")])

    notes = await summarize_visit(
        session, user_id, [HumanMessage(content="a latte"), AIMessage(content="Sure.")], llm=llm
    )

    assert notes == []
    assert (await customer_profile(session, user_id))["notes"] == []


async def test_a_failing_model_does_not_fail_the_visit(customer):
    """Memory is a nicety; going home is not."""
    session, user_id = customer

    class Broken(FakeToolCallingModel):
        async def ainvoke(self, *args, **kwargs):
            raise RuntimeError("ollama is down")

    notes = await summarize_visit(
        session, user_id, [HumanMessage(content="hi"), AIMessage(content="hello")], llm=Broken([])
    )

    assert notes == []


async def test_an_empty_transcript_skips_the_model_entirely(customer):
    session, user_id = customer

    # No script at all: if the model were called, FakeToolCallingModel raises.
    notes = await summarize_visit(session, user_id, [], llm=FakeToolCallingModel([]))

    assert notes == []


async def test_notes_reach_the_context_block(customer):
    """The whole point: what was remembered shows up in the next prompt."""
    from agent.prompts import render_context

    session, user_id = customer
    await append_customer_notes(session, user_id, ["found the mocha too sweet"])
    profile = await customer_profile(session, user_id)

    block = render_context(
        {"customer_profile": {**profile, "visit_count": 4}, "day": 2, "menu": [], "cart": {}}
    )

    assert "found the mocha too sweet" in block
