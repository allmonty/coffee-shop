"""Every environment variable the API reads, in one place (spec §9.6, §10)."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Domain constants. The wallet amount is referenced by the daily-menu
    # affordability guarantees (spec §3.2 G3), so it lives here rather than
    # being written twice.
    daily_wallet_cents: int = 2000

    database_url: str = "postgresql+asyncpg://coffee:coffee@localhost:5432/coffee_shop"

    # Set false to skip OTel wiring entirely. Telemetry is best-effort: the app
    # must start and serve with no collector reachable (spec §9.6).
    otel_enabled: bool = True


settings = Settings()
