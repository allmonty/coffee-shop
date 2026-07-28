# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository status

**This repo contains a specification and no implementation.** The only tracked files are
`coffee-shop.md` and `.gitignore`. There is no build, no test suite, no dependency manifest, and no
application code yet.

Do not infer commands from this file that you have not verified exist. Anything under "Planned
toolchain" below is what the spec *calls for*, not what is installed.

`coffee-shop.md` (~1100 lines) is the source of truth. Read the relevant section before implementing
anything — most design decisions in it are load-bearing and were made deliberately against a plausible
alternative, and the reasoning is recorded inline.

## What is being built

A conversational coffee shop: the user types a name, enters the shop, and orders from an LLM-powered
barista through free text. $20 per in-game day, resets when they go home. The barista remembers previous
visits and suggests "the usual".

The purpose of the project is **learning agentic architecture** — tool-calling agents, LangGraph
orchestration, persistent agent state, local LLM. The coffee-shop domain is deliberately small so the
interesting complexity stays in the agent layer. Weigh design choices accordingly: complexity in
`agent/` is often justified, complexity anywhere else usually is not.

## Spec section map

| Section | Contents |
| --- | --- |
| §2, §3 | Domain rules; catalog vs. daily menu; the G1–G4 affordability guarantees; drink sizes |
| §4 | UX, identity-by-name, and the edge-case table the barista must handle |
| §5 | Architecture, the `agent/`↔`shop/` boundary, and the backend file layout (§5.3) |
| §6 | **The agent** — graph shape, tools, memory layers, system prompt. The densest section |
| §7 | REST + SSE contracts |
| §8 | Data model |
| §9 | OpenTelemetry: trace shape, metrics, logs |
| §11 | Milestone plan, M1–M7 |
| §13 | Decisions already made, with reasoning, plus what is still open |

## Planned toolchain (not yet present)

Per the spec: Python 3.12 + FastAPI + LangGraph, `uv` for dependencies, SQLAlchemy 2.0 + Alembic,
pytest, React + Vite, and a five-service `docker compose` (`web`, `api`, `db`, `llm` via Ollama, `otel`
via `grafana/otel-lgtm`). App on :3000, API on :8000, Grafana on :3001.

When creating these, follow §5.3's layout — it exists so the code stays readable while learning.

## Invariants that must survive implementation

These are the rules most easily broken by a well-intentioned change. Each is argued in the spec.

- **`agent/` may import `shop/`, never the reverse.** `agent/` never imports SQLAlchemy models — only
  `shop/service.py` functions. The domain layer must not know an LLM exists. Enforce with import-linter.
- **Every domain function and tool returns the `{ok, error, message}` envelope.** Tools pass it through
  verbatim; `message` is written to be read aloud by the barista.
- **No tool takes a price argument.** `place_order(confirmed_total_cents)` is not the model setting a
  price — the domain rejects a mismatch, so the model can only fail a charge, never lower one.
- **Money-spending tools are gated in the domain, not the prompt.** `place_order` and `end_visit`
  require confirmation; a prompt instruction is the weakest available enforcement.
- **The menu lives in the context block; do not add a mandatory `get_menu` call per turn.** That was
  removed on purpose — it forced an extra `barista → tools → barista` lap, i.e. two local-model
  inferences instead of one, to re-read data that cannot change mid-visit.
- **Today's menu ≠ the catalog.** Everything agent-facing is scoped through `visit_menu_items`.
  `unknown_item` and `not_available_today` are distinct errors because they are distinct truths.
- **Sizes are drinks-only**, enforced by a check constraint. Never ask what size a cookie is.
- **Money is integer cents everywhere.** `order_lines` snapshots unit price including size surcharge.
- **Only model-written `notes` are stored** in `customer_preferences`; `favorite_*`, `usual_order`,
  `visit_count`, `last_visit_day` are aggregated at read time. Never ask an LLM for a fact a `GROUP BY`
  can produce.
- **Raw transcripts are never re-injected into a prompt.** Cross-visit memory arrives only via the
  computed profile and summarized notes.
- **Telemetry is best-effort.** `api` must not `depends_on: otel`; a failed exporter must never fail a
  request. Metrics are for what only the runtime knows — order counts and revenue come from SQL, not
  from counters.

## Known cleanup

`.gitignore` is still the Elixir one from the original plan (`/_build`, `/deps`, `*.beam`,
`.elixir_ls/`). The backend is Python now; it needs Python and Node entries when implementation starts.

## Working style for this repo

- Prefer editing `coffee-shop.md` in place over appending new sections; it has been restructured several
  times and section numbers are cross-referenced throughout (`§N.N`). After renumbering, verify every
  reference still resolves.
- The spec records decisions *and their alternatives*. When changing one, update §13 rather than silently
  reversing it, so a future change is a decision rather than a drift.
