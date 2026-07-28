"""REST endpoints (spec §7.1).

Thin: parse, call one `shop.service` function, translate the `Result` envelope
to HTTP. Any rule that appeared here would be a rule the agent's tools do not
enforce, and the same order would behave differently clicked versus spoken.
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from deps import get_session
from shop import service
from shop.models import User, Visit
from shop.profile import customer_profile
from shop.result import Result
from shop.schemas import EnterRequest

router = APIRouter(prefix="/api", tags=["shop"])

# The Annotated form rather than a `Depends()` default: same behaviour, and it
# is what FastAPI recommends now.
SessionDep = Annotated[AsyncSession, Depends(get_session)]


def unwrap(result: Result) -> dict:
    """422 with the domain's own error and message, never a bare 500."""
    if not result.ok:
        raise HTTPException(status_code=422, detail=result.to_dict())
    return result.to_dict()


def parse_uuid(value: str, what: str) -> uuid.UUID:
    try:
        return uuid.UUID(value)
    except ValueError:
        raise HTTPException(status_code=422, detail={"error": f"invalid_{what}"}) from None


@router.post("/enter")
async def enter(body: EnterRequest, session: SessionDep) -> dict:
    """Find-or-create the customer, open or resume a visit, draw today's menu.

    The only call the landing page makes.
    """
    return unwrap(await service.enter(session, body.name))


@router.get("/users/{user_id}")
async def get_user(user_id: str, session: SessionDep) -> dict:
    parsed = parse_uuid(user_id, "user_id")
    user = await session.get(User, parsed)
    if user is None:
        raise HTTPException(status_code=404, detail={"error": "unknown_user"})
    return {
        "user_id": str(user.id),
        "name": user.name,
        "day": user.current_day,
        "weekday": service.weekday_for(user.current_day),
        "wallet_cents": user.wallet_cents,
    }


@router.get("/users/{user_id}/orders")
async def get_orders(user_id: str, session: SessionDep) -> dict:
    parsed = parse_uuid(user_id, "user_id")
    return {"orders": await service.order_history(session, parsed)}


@router.get("/users/{user_id}/profile")
async def get_profile(user_id: str, session: SessionDep) -> dict:
    parsed = parse_uuid(user_id, "user_id")
    return await customer_profile(session, parsed)


@router.get("/visits/{visit_id}/menu")
async def get_menu(visit_id: str, session: SessionDep) -> dict:
    """Today's menu, not the catalog.

    Scoped to a visit on purpose: there is no such thing as "the menu" without
    one (spec §3.2).
    """
    parsed = parse_uuid(visit_id, "visit_id")
    visit = await session.get(Visit, parsed)
    if visit is None:
        raise HTTPException(status_code=404, detail={"error": "unknown_visit"})
    return {"menu": await service.todays_menu(session, parsed)}


@router.get("/visits/{visit_id}")
async def get_visit(visit_id: str, session: SessionDep) -> dict:
    """Everything needed to rehydrate the page after a refresh."""
    parsed = parse_uuid(visit_id, "visit_id")
    visit = await session.get(Visit, parsed)
    if visit is None:
        raise HTTPException(status_code=404, detail={"error": "unknown_visit"})

    user = await session.get(User, visit.user_id)
    assert user is not None
    return {
        "visit_id": str(visit.id),
        "user_id": str(user.id),
        "name": user.name,
        "day": visit.day,
        "weekday": service.weekday_for(visit.day),
        "wallet_cents": user.wallet_cents,
        "ended": visit.ended_at is not None,
        "menu": await service.todays_menu(session, parsed),
        "cart": await service.cart_payload(session, parsed),
    }
