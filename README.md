# Coffee Shop

A virtual coffee shop where you order from an LLM barista by typing. Built to
learn **agentic architecture**: LangGraph, tool-calling, agent state, a local
model — with a deliberately small domain so the interesting complexity stays in
the agent layer.

**New here? Read [TOUR.md](TOUR.md)** — a guided walkthrough of the code in the
order that makes the agent make sense, with experiments to try.

`coffee-shop.md` is the full specification, including the reasoning behind each
decision. `CLAUDE.md` lists the invariants that must survive changes.

## Running it

Needs Docker and [uv](https://docs.astral.sh/uv/). Ollama is strongly
recommended **on the host** rather than in Docker — containers get no GPU on
Apple Silicon, which makes prompt iteration painful.

```bash
ollama pull qwen2.5:7b-instruct
ollama serve

docker compose up --build     # app :3000 · api :8000 · Grafana :3001
```

Then open <http://localhost:3000>, type a name, and order a coffee.

## Talking to the barista in a terminal

The better debugging surface, and how most of this was built — the whole
transcript stays in the scrollback next to the trace:

```bash
cd api
uv run python scripts/shop_cli.py --name Allan
```

`/cart`, `/wallet` and `/state` inspect things without talking to the model.

## Before you debug a graph

If the barista behaves strangely, check the model can tool-call **at all**
before suspecting your code. This is the single biggest time-saver here:

```bash
cd api
uv run python scripts/check_tool_calling.py
```

## Tests

```bash
make test                                     # 158 tests, no Ollama needed
cd api && uv run pytest -m scenario           # 6 conversations with the real model
```

The default suite runs with a scripted fake model, so it is fast and
deterministic. The scenario suite drives the real thing, is flaky by nature, and
asserts on final domain state rather than wording.

## How it fits together

```
web/        React + Vite. Two screens, no router.
api/
  routers/  HTTP: REST + the SSE chat endpoint
  agent/    LangGraph — graph.py is the file that explains the whole agent
  shop/     The domain. Owns the money. Has no idea an LLM exists.
```

The rule that keeps it honest: **`agent/` may import `shop/`, never the
reverse**, and `agent/` never touches SQLAlchemy models — only `shop.service`.

## Two things worth knowing

**The domain enforces what matters, not the prompt.** `place_order` takes the
total the barista quoted out loud and refuses a mismatch, so charging without
confirming means guessing the exact cart total. A prompt rule would have
silently charged — and did, in an early run.

**The trace of one turn is the agent loop, drawn.** Open Grafana on :3001 after
a conversation. `agent.loop.iterations` is the only place a model going in
circles is visible.
