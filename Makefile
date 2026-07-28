.PHONY: up down logs test lint fmt db-only

# Full stack.
up:
	docker compose up --build -d
	@echo "api → http://localhost:8000/health"

down:
	docker compose down

logs:
	docker compose logs -f api

# Just Postgres — what the test suite needs. Faster than building the api image
# when you are only running tests.
db-only:
	docker compose up -d db

test: db-only
	cd api && uv run pytest -q

lint:
	cd api && uv run ruff check . && uv run ruff format --check .

fmt:
	cd api && uv run ruff format . && uv run ruff check --fix .
