"""Contract tests for the REST surface (spec §7.1).

These hit the real app through ASGI with a session that commits, then truncate —
the rollback fixture cannot be used because FastAPI's dependency opens its own
session per request.
"""

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport

from deps import get_session
from main import app
from shop.seed import seed_catalog


@pytest_asyncio.fixture
async def client(session_factory):
    async def override_get_session():
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_session] = override_get_session
    async with session_factory() as setup:
        await seed_catalog(setup)

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        yield http

    app.dependency_overrides.clear()


async def test_health(client):
    response = await client.get("/health")
    assert response.status_code == 200


async def test_enter_creates_a_customer_and_a_menu(client):
    response = await client.post("/api/enter", json={"name": "Allan"})

    assert response.status_code == 200
    body = response.json()
    assert body["is_new"] is True
    assert body["day"] == 1
    assert body["weekday"] == "Monday"
    assert body["wallet_cents"] == 2000
    assert 8 <= len(body["menu"]) <= 12


async def test_enter_is_idempotent_for_the_same_name(client):
    first = (await client.post("/api/enter", json={"name": "Allan"})).json()
    again = (await client.post("/api/enter", json={"name": " ALLAN "})).json()

    assert again["user_id"] == first["user_id"]
    assert again["visit_id"] == first["visit_id"]
    assert again["is_new"] is False


@pytest.mark.parametrize("name", ["", "   ", "123"])
async def test_enter_rejects_unusable_names(client, name):
    response = await client.post("/api/enter", json={"name": name})

    # 422 either from pydantic (empty) or from the domain (not a name).
    assert response.status_code == 422


async def test_get_visit_rehydrates_the_page(client):
    entered = (await client.post("/api/enter", json={"name": "Allan"})).json()

    response = await client.get(f"/api/visits/{entered['visit_id']}")

    body = response.json()
    assert body["name"] == "Allan"
    assert body["ended"] is False
    assert body["cart"] == {"lines": [], "total_cents": 0}
    assert body["menu"] == entered["menu"]


async def test_visit_menu_endpoint_matches_enter(client):
    entered = (await client.post("/api/enter", json={"name": "Allan"})).json()

    response = await client.get(f"/api/visits/{entered['visit_id']}/menu")

    assert response.json()["menu"] == entered["menu"]


async def test_unknown_visit_is_404(client):
    response = await client.get("/api/visits/6c9f1a2e-0000-0000-0000-000000000000")
    assert response.status_code == 404


async def test_malformed_uuid_is_422_not_500(client):
    response = await client.get("/api/visits/not-a-uuid")
    assert response.status_code == 422


async def test_profile_of_a_new_customer_is_empty_but_shaped(client):
    entered = (await client.post("/api/enter", json={"name": "Allan"})).json()

    body = (await client.get(f"/api/users/{entered['user_id']}/profile")).json()

    assert body["name"] == "Allan"
    assert body["visit_count"] == 1
    assert body["favorite_drink"] is None
    assert body["usual_order"] == []
    assert body["notes"] == []


async def test_order_history_is_empty_for_a_new_customer(client):
    entered = (await client.post("/api/enter", json={"name": "Allan"})).json()

    body = (await client.get(f"/api/users/{entered['user_id']}/orders")).json()

    assert body["orders"] == []
