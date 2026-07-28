"""Storage for model-written notes (spec §6.5.1).

This is the one place LLM-generated content lands in a domain table, which makes
it the easiest place to breach the §5.1 boundary by accident. The write belongs
to `shop/`; the summarising belongs to `agent/`.
"""

from __future__ import annotations

import uuid

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from shop.models import CustomerPreference
from shop.result import Result

# Without a cap this grows until it crowds a small model's context window.
MAX_NOTES = 10


async def append_customer_notes(
    session: AsyncSession, user_id: uuid.UUID, new_notes: list[str]
) -> Result:
    """Append notes, keeping the most recent `MAX_NOTES`, oldest dropped first."""
    cleaned = [note.strip() for note in new_notes if note and note.strip()]
    if not cleaned:
        return Result.success("Nothing worth remembering.", notes_added=0)

    await session.execute(
        pg_insert(CustomerPreference)
        .values(user_id=user_id, notes=[])
        .on_conflict_do_nothing(index_elements=[CustomerPreference.user_id])
    )
    preference = await session.get(CustomerPreference, user_id)
    assert preference is not None

    combined = [*list(preference.notes or []), *cleaned][-MAX_NOTES:]
    preference.notes = combined
    await session.flush()
    await session.commit()

    return Result.success(notes_added=len(cleaned), notes=combined)
