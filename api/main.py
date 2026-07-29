"""FastAPI application entrypoint (spec §5.3).

Wiring only: app creation, lifespan, router mounting. Anything that makes a
decision belongs in shop/ or agent/.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agent.checkpointer import open_checkpointer
from db import SessionLocal, engine
from routers import chat as chat_router
from routers import shop as shop_router
from shop.seed import seed_catalog
from telemetry import setup_telemetry


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_telemetry(app, engine)
    async with SessionLocal() as session:
        await seed_catalog(session)

    # The conversation store is process-wide and has to outlive every request —
    # `thread_id = visit_id`, so a visit's turns find each other (spec §6.5).
    # Opening it per turn would defeat the point of having one.
    async with open_checkpointer() as checkpointer:
        chat_router.set_checkpointer(checkpointer)
        yield

    await engine.dispose()


app = FastAPI(title="Coffee Shop API", lifespan=lifespan)

# The frontend is served from a different origin in dev (Vite on 5173).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(shop_router.router)
app.include_router(chat_router.router)


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe used by docker compose and by the test suite."""
    return {"status": "ok"}
