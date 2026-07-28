"""FastAPI application entrypoint (spec §5.3).

Wiring only: app creation, lifespan, router mounting. Anything that makes a
decision belongs in shop/ or agent/.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from db import SessionLocal, engine
from routers import shop as shop_router
from shop.seed import seed_catalog
from telemetry import setup_telemetry


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_telemetry(app, engine)
    async with SessionLocal() as session:
        await seed_catalog(session)
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


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe used by docker compose and by the test suite."""
    return {"status": "ok"}
