"""Turn a finished visit into one or two durable notes (spec §6.5.1).

This is the difference between an agent with a transcript and an agent with
memory. Raw transcripts are kept forever for display but never re-injected into
a prompt; everything the barista recalls across visits arrives through the
computed profile plus these notes.

Three rules, all of which exist because of how models behave rather than how
code behaves:

1. **"Nothing stood out" must be a normal outcome.** A model told to always
   produce a fact will invent one, and invented notes compound across visits
   into a barista confidently misremembering things.
2. **Runs after the response is closed.** The customer is already walking out.
3. **A failure here must never fail the visit.** Memory is a nicety; going home
   is not.
"""

from __future__ import annotations

import json
import logging
import uuid

from langchain_core.messages import HumanMessage, SystemMessage

from agent.llm import build_llm
from shop.notes import append_customer_notes

logger = logging.getLogger(__name__)

SUMMARIZE_PROMPT = """\
You are reviewing a conversation between a barista and a customer.

Extract AT MOST TWO durable facts about this customer that would be worth
remembering the next time they come in — a preference, something they mentioned
about themselves, a reaction to something they tried.

Do NOT record:
- what they ordered (that is already tracked separately)
- prices, totals, or anything about the transaction
- pleasantries, greetings, or small talk

If nothing stands out, return an empty list. That is a perfectly good answer and
most conversations should produce it.

Reply with ONLY a JSON array of strings. Examples:
[]
["mentioned starting a new job"]
["found the mocha too sweet", "always comes in early"]
"""


def extract_notes(reply: str) -> list[str]:
    """Parse the model's JSON array, tolerating the usual mess around it."""
    text = reply.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("[") :] if "[" in text else text

    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        return []

    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []

    if not isinstance(parsed, list):
        return []
    return [str(item).strip() for item in parsed if str(item).strip()][:2]


def render_transcript(messages) -> str:
    lines = []
    for message in messages:
        role = getattr(message, "type", "")
        content = getattr(message, "content", "")
        if role == "human" and content:
            lines.append(f"Customer: {content}")
        elif role == "ai" and content:
            lines.append(f"Barista: {content}")
    return "\n".join(lines)


async def summarize_visit(session, user_id: uuid.UUID, messages, llm=None) -> list[str]:
    """Extract notes and store them. Never raises."""
    transcript = render_transcript(messages)
    if not transcript.strip():
        return []

    try:
        model = llm or build_llm()
        reply = await model.ainvoke(
            [
                SystemMessage(content=SUMMARIZE_PROMPT),
                HumanMessage(content=transcript),
            ]
        )
        notes = extract_notes(str(reply.content))
    except Exception:
        logger.warning("visit summarization failed; continuing", exc_info=True)
        return []

    if not notes:
        return []

    try:
        # Through shop.service, never straight to the table — this is the one
        # place model output reaches a domain table (spec §5.1).
        await append_customer_notes(session, user_id, notes)
    except Exception:
        logger.warning("storing visit notes failed; continuing", exc_info=True)
        return []

    return notes
