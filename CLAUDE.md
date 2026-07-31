# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository status

Implemented and working end to end. `TOUR.md` is a guided walkthrough of the code for someone learning
it. `coffee-shop.md` (~1100 lines) remains the source of truth for
*why* — most decisions in it are load-bearing and were made against a plausible alternative, with the
reasoning recorded inline. Read the relevant section before changing behaviour.

## Commands

```bash
make test          # 253 tests, needs only Postgres — no Ollama
make lint          # ruff check + format --check
make up            # full stack: app :3000, api :8000, Grafana :3001

cd api
uv run pytest tests/test_graph.py -q          # one file
uv run pytest -m scenario                     # 6 conversations with the real model
uv run python scripts/check_tool_calling.py   # does the model tool-call at all?
uv run python scripts/shop_cli.py --name Allan
uv run alembic revision --autogenerate -m "..." && uv run alembic upgrade head
```

`make test` starts only the `db` container. Tests run against real Postgres in a `coffee_shop_test`
database, migrated with alembic, each test in a rolled-back transaction.

**Debugging order when the barista misbehaves:** run `check_tool_calling.py` first, then look at the
trace in Grafana, then suspect the code. Most failures are the model or the prompt.

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

## Testing approach

The default suite uses `tests/fakes.py::FakeToolCallingModel`, which replays a script of `AIMessage`s
with and without tool calls. That is what keeps graph tests fast, deterministic and CI-safe. Use
`tool_calls(...)` (plural) when you need several calls in ONE assistant turn — that shape is what
exposed the concurrent-tools bug, and no single-call test would have caught it.

Scenario tests (`-m scenario`) drive the real model and are flaky by nature. They assert on final
domain state, never on wording.

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
- **Three roles, one voice.** Sam (waiter) fronts every conversation; Mo (barista) and Val (cashier)
  are LLM sub-agents called as tools via `ask_barista` / `ring_up`. Sam has no money tools and no
  `modifiers` argument — both gates are structural, because measurement showed prose gates do not
  hold: given the argument, the real model never delegated once. A role owns a tool when its knowledge
  fills the arguments, not when the tool sounds thematically like its job.
- **Val decides *whether* to charge, never *what*.** `charge_the_customer` and `send_them_home` take no
  arguments; the quoted total and "they said they're leaving" are injected through config. A second
  model supplying `confirmed_total_cents` would read the cart, always match, and turn the project's
  central guard into a rubber stamp. Val is also handed only the tools Sam authorised for that job.
- **A delegation reports `charged` and `visit_ended` as facts.** One `ok` cannot carry the money path,
  and Sam's "never say it is paid for" rule keys on `charged`, not on `ok`.
- **A sub-agent's tool calls never enter the waiter's context.** They go to `delegation_steps` in graph
  state, not into the envelope — the envelope becomes the ToolMessage content, which Sam re-reads every
  later turn. The SSE layer joins them back for the UI.
- **The menu lives in the context block; do not add a mandatory `get_menu` call per turn.** That was
  removed on purpose — it forced an extra `barista → tools → barista` lap, i.e. two local-model
  inferences instead of one, to re-read data that cannot change mid-visit.
- **Today's menu ≠ the catalog.** Everything agent-facing is scoped through `visit_menu_items`.
  `unknown_item` and `not_available_today` are distinct errors because they are distinct truths.
- **Sizes and modifiers are drinks-only**, enforced by check constraints against the same `sized` flag.
  Never ask what size a cookie is, or what milk it takes.
- **Modifiers are optional; sizes are not.** There is deliberately no `modifier_required` — a drink with
  no modifiers is a complete order. A modifier request can only be over-specified (`unknown_modifier`,
  `modifier_conflict`), never under-specified. That asymmetry is why the two axes have different error
  shapes even though they otherwise look alike.
- **Line identity is `(item, size, modifier set)`**, held as a canonical sorted text key built *only* by
  `shop/modifiers.py::canonical_key`. Never build that string anywhere else: it is what makes the same
  drink ordered twice one line and two different drinks two lines. The barista never *offers* a
  modifier — §3.5 and §4.5 already spend two upsell budgets.
- **Money is integer cents everywhere.** `order_lines` snapshots unit price including size and modifier
  surcharges.
- **Only model-written `notes` are stored** in `customer_preferences`; `favorite_*`, `usual_order`,
  `visit_count`, `last_visit_day` are aggregated at read time. Never ask an LLM for a fact a `GROUP BY`
  can produce.
- **Raw transcripts are never re-injected into a prompt.** Cross-visit memory arrives only via the
  computed profile and summarized notes.
- **Every caller of the graph supplies the checkpointer.** `main.py`'s lifespan opens it and hands it
  to `routers/chat.py` via `set_checkpointer()`; the CLI opens its own. Compiling the graph without one
  does not fail — it silently makes the agent amnesiac, because each turn starts from empty state.
- **Character names live in `agent/`, never in `shop/`.** Domain messages stay in the shop's own voice;
  "Mo says" is applied by the agent layer to a sentence the domain wrote. `temperature=0` stays — do
  not raise it to buy personality. Variety comes from varied inputs (visit milestones, weekday, cart),
  never from model spontaneity.
- **No concrete few-shot examples in a prompt.** A small model reads an example as content to reuse,
  not as a shape to imitate: `qwen2.5:3b` copied a sample note verbatim and invented a mocha nobody
  ordered. Examples use `<placeholder>` text.
- **Telemetry is best-effort.** `api` must not `depends_on: otel`; a failed exporter must never fail a
  request. Metrics are for what only the runtime knows — order counts and revenue come from SQL, not
  from counters.
- **Tool calls run sequentially** (`agent/graph.py::run_tools`). LangGraph's prebuilt `ToolNode` runs
  them concurrently, which breaks both the shared `AsyncSession` and the causal order of
  "add it, then charge me". Do not swap it back.
- **A malformed tool call returns an envelope, never raises.** Small models drop required arguments
  constantly; the barista has to be able to fix and retry. An *invented* tool name is the same case:
  `{ok, error, message}` like everything else — the message lists the real tools — plus a
  `tool.unknown` span and an `agent.tool.malformed` increment. Model-written strings never become span
  names or metric labels.
- **A failed tool marks its own span, not the parent turn.** "insufficient funds" is a normal outcome,
  and marking it a request error makes every dashboard lie.

## Working style for this repo

- Prefer editing `coffee-shop.md` in place over appending new sections; it has been restructured several
  times and section numbers are cross-referenced throughout (`§N.N`). After renumbering, verify every
  reference still resolves.
- The spec records decisions *and their alternatives*. When changing one, update §13 rather than silently
  reversing it, so a future change is a decision rather than a drift.
