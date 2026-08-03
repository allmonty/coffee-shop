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


async def test_the_context_block_lists_extras_in_one_line():
    """The §6.6 compactness guard, made executable.

    Modifier prices belong in one line for the whole menu, not a surcharge
    printed against each of seventeen drinks. If someone ever renders them
    per-item, this fails.
    """
    from agent.prompts import render_context

    menu = [
        {"name": "Latte", "category": "drink", "price_cents": 400},
        {"name": "Mocha", "category": "drink", "price_cents": 500},
        {"name": "Croissant", "category": "food", "price_cents": 350},
    ]
    block = render_context(
        {
            "customer_profile": None,
            "day": 1,
            "menu": menu,
            "cart": {},
            "size_deltas": {"small": 0, "medium": 60, "large": 120},
            "modifier_deltas": {"oat_milk": 60, "almond_milk": 60, "extra_shot": 100},
        }
    )

    extras = [line for line in block.splitlines() if "oat milk" in line]
    assert len(extras) == 1
    assert extras[0].strip() == (
        "Extras (drinks only): almond milk +$0.60 · extra shot +$1.00 · oat milk +$0.60"
    )


async def test_a_modified_cart_line_shows_its_extras():
    from agent.prompts import render_context

    block = render_context(
        {
            "customer_profile": None,
            "day": 1,
            "menu": [],
            "cart": {
                "lines": [
                    {
                        "item": "Latte",
                        "size": "large",
                        "modifiers": ["oat_milk"],
                        "quantity": 1,
                        "line_total_cents": 580,
                    }
                ],
                "total_cents": 580,
            },
        }
    )

    assert "1 x large Latte (oat milk) = $5.80" in block


async def test_the_summarizer_falls_back_to_the_main_model(monkeypatch):
    """Unset is the default, and must behave exactly as before the setting existed."""
    from agent.llm import build_summary_llm
    from settings import settings

    monkeypatch.setattr(settings, "ollama_summary_model", None)

    assert build_summary_llm().model_name == settings.ollama_model


async def test_the_summarizer_uses_the_summary_model_when_one_is_set(monkeypatch):
    from agent.llm import build_summary_llm
    from settings import settings

    monkeypatch.setattr(settings, "ollama_summary_model", "qwen2.5:3b")

    model = build_summary_llm()
    assert model.model_name == "qwen2.5:3b"
    # Same endpoint: this swaps the model, not where it runs.
    assert str(model.openai_api_base) == settings.ollama_base_url


async def test_an_empty_summary_model_is_treated_as_unset(monkeypatch):
    """compose passes `OLLAMA_SUMMARY_MODEL: ${...:-}`, i.e. an empty string."""
    from agent.llm import build_summary_llm
    from settings import settings

    monkeypatch.setattr(settings, "ollama_summary_model", "")

    assert build_summary_llm().model_name == settings.ollama_model


def test_the_summarize_prompt_carries_no_reusable_example_text():
    """Learned from `qwen2.5:3b` copying an example note verbatim.

    A small model treats a concrete example as content to reuse rather than a
    shape to imitate, so it invented "found the mocha too sweet" for a customer
    who never mentioned a mocha. Examples must be placeholders.
    """
    from agent.summarize import SUMMARIZE_PROMPT

    examples = [
        line
        for line in SUMMARIZE_PROMPT.splitlines()
        if line.startswith("[") and line.strip() not in ("[]",)
    ]
    assert examples, "the prompt should still show the output shape"
    for line in examples:
        assert "<" in line and ">" in line, f"concrete example text is copyable: {line}"


def test_a_milestone_visit_is_flagged_in_the_context_block():
    """The payoff for the memory layer, and it costs nothing: visit_count is
    already aggregated and already in the block."""
    from agent.prompts import render_context

    block = render_context(
        {"customer_profile": {"name": "Allan", "visit_count": 10}, "day": 9, "menu": [], "cart": {}}
    )

    assert "MILESTONE" in block
    assert "tenth" in block


def test_an_ordinary_visit_gets_no_milestone():
    """A comment on visit 4 and again on visit 5 reads as counting, not
    recognition."""
    from agent.prompts import render_context

    block = render_context(
        {"customer_profile": {"name": "Allan", "visit_count": 4}, "day": 3, "menu": [], "cart": {}}
    )

    assert "MILESTONE" not in block


def test_a_first_timer_is_never_claimed_to_be_remembered():
    from agent.prompts import render_context

    block = render_context(
        {"customer_profile": {"name": "Allan", "visit_count": 1}, "day": 1, "menu": [], "cart": {}}
    )

    assert "First time here" in block
    assert "MILESTONE" not in block
