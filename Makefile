.PHONY: up up-with-llm down logs test lint fmt db-only

# Default stack: web, api, db, otel. Expects Ollama on the HOST at :11434,
# which is the fast path on Apple Silicon (Docker gets no GPU there).
up:
	docker compose up --build -d
	@echo
	@echo "  app      → http://localhost:3000"
	@echo "  api      → http://localhost:8000/health"
	@echo "  Grafana  → http://localhost:3001"
	@echo
	@curl -sf -m 3 http://localhost:11434/api/tags >/dev/null 2>&1 \
		&& echo "  model    → host Ollama, reachable" \
		|| echo "  model    → NOT REACHABLE. Run 'ollama serve' on the host, or 'make up-with-llm'."

# Everything in containers, model included. Slower on Apple Silicon (CPU only),
# but it is one command and needs nothing installed on the host.
up-with-llm:
	OLLAMA_BASE_URL=http://llm:11434/v1 docker compose --profile llm up --build -d
	@echo
	@echo "  First run downloads several GB into the llm_models volume."
	@echo "  Watch it:  docker compose logs -f llm"

down:
	docker compose --profile llm down

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
