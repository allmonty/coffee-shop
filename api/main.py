"""FastAPI application entrypoint (spec §5.3).

Keep this file to wiring only: app creation, lifespan, router mounting. Anything
that makes a decision belongs in shop/ or agent/.
"""

from fastapi import FastAPI

app = FastAPI(title="Coffee Shop API")


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe used by docker compose and by the test suite."""
    return {"status": "ok"}
