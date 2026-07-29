# A guided tour of the code

Read this with the repo open beside you. It goes in the order that makes the
agent make sense, which is **not** the order the files are laid out in.

Roughly 45 minutes of reading, plus the experiments. Every stop names the file
and the function, and most end with something to try — the experiments are the
part that actually teaches, because watching a rule break tells you what it was
holding up.

`coffee-shop.md` is the spec and explains *why* for every decision. This document
explains *where* and *how*.

---

## Before you read anything: run it

Reading an agent is much easier once you've watched one work.

```bash
ollama serve                       # in another terminal
cd api
uv run python scripts/check_tool_calling.py
uv run python scripts/shop_cli.py --name YourName
```

Order a coffee. Try "a latte please" without a size. Try ordering a mocha. Try
spending more than $20. Then come back.

---

## Stop 1 — The shape (5 minutes)

Three layers, and one rule that holds them apart:

```
routers/    HTTP. Parses, calls one function, formats the answer.
agent/      The conversation. LangGraph, prompts, tools.
shop/       The domain. Owns the money. Has no idea an LLM exists.
```

**The rule: `agent/` may import `shop/`, never the reverse.** And `agent/` never
imports SQLAlchemy models — only `shop.service` functions.

That single constraint is what makes the project legible. The domain can be read
and tested as an ordinary Python app; the agent can be read as a thing that
*talks to* an ordinary Python app.

> **Look:** `api/agent/graph.py`, the `_load_context` function. It needs the
> current day. The obvious way is `session.get(Visit, visit_id)` — but that
> would import a model into `agent/`. It calls `service.get_wallet_balance()`
> instead, which already returns the day. The comment there says so.

---

## Stop 2 — The envelope (`api/shop/result.py`, 47 lines)

The smallest file in the project, and everything else is shaped by it.

Every domain function and every tool returns a `Result`:

```python
Result.success(wallet_cents=1480)
Result.failure("insufficient_funds", "That's $6.50 and you've got $3.50 left today.")
```

Two things to notice.

**`Result.failure()` requires a message.** Not optional. That message is not a
log line — it is *the sentence the barista says out loud*. A failure without one
leaves the model to invent an explanation, which is precisely how an agent starts
making up prices.

**`to_dict()` is flat**, not `{"ok": ..., "data": {...}}`. The model reads this
JSON directly, and small models handle `balance_cents` noticeably better than
`result.data.balance_cents`.

> **Try it:** `uv run pytest tests/test_result.py -v`. Five tests, one file, and
> `test_failure_requires_a_message` is the one that matters.

---

## Stop 3 — One domain function (`api/shop/cart.py`, `add_to_cart`)

Skim the rest of `shop/` later. Read this one function properly.

It can fail in four different ways, and **each is a different truth the barista
has to tell the customer**:

| error | what it means |
| --- | --- |
| `unknown_item` | "We don't do bubble tea." |
| `not_available_today` | "No mochas *today*." — real item, not drawn today |
| `size_required` | "Which size?" — the request was incomplete, not wrong |
| `size_not_applicable` | "A croissant only comes the one size." |

Collapsing these into a generic "rejected" would make the barista lie. "We don't
sell mochas" is false when the shop sells mochas on other days.

`size_required` is the interesting one. **It is not really a failure** — it's how
the domain tells the agent the customer's request was incomplete, so the barista
asks instead of guessing. Its message is a question, read aloud verbatim.

> **Try it:** `uv run pytest tests/test_cart.py -v` and read the test names as a
> list. They're written to be a description of the behaviour.

---

## Stop 4 — The loop (`api/agent/graph.py`)

**This is the file.** If you read one thing, read this.

```
START → load_context → barista ⇄ tools → refresh → finish → END
```

Read `build_graph()` at the bottom first — it's just wiring, ~20 lines. Then
`route_after_barista`:

```python
def route_after_barista(state):
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "finish"
```

**That function is the agent.** Everything else in the file is setup and
teardown. A reply carrying tool calls goes to the tool node and comes back for
another pass; a reply that is just words ends the turn. The model decides how
many laps by choosing whether to ask for a tool.

Then read the nodes in this order:

1. **`load_context`** — runs on the *first turn only*. Loads profile, menu,
   wallet, cart. The menu can't change mid-visit, so re-reading it every turn
   would be a round-trip for data that cannot have moved.
2. **`barista`** (inside `make_barista`) — the LLM call. Note the system message
   is rebuilt from live state *every turn*, so what the model sees is never
   stale. Note also `llm` is a parameter: that's what lets tests inject a fake.
3. **`run_tools`** — see Stop 7.
4. **`refresh`** — re-reads cart and wallet after tools ran. Without this the
   model reads a stale total out loud immediately after changing the cart.
5. **`finish`** — records how many laps the turn took.

> **Experiment:** delete the `refresh` node from `build_graph` and wire
> `tools → barista` directly. Run
> `uv run pytest tests/test_graph.py::test_cart_is_refreshed_before_the_model_speaks_again`.
> Then put it back.

### The one bit of LangGraph magic worth understanding

Open `api/agent/state.py`:

```python
messages: Annotated[list[AnyMessage], add_messages]
```

Nodes return **partial** state, and a reducer decides how it merges. For
`messages`, `add_messages` means append-with-dedup-by-id rather than replace —
which is why `barista` can return one new message and the conversation keeps the
rest. Every other field in that TypedDict is plain replace-on-write.

---

## Stop 5 — Tools are an API, not database access (`api/agent/tools.py`)

This file is deliberately boring, and the boringness is the point.

**Every tool is named after the service function it calls.** The `add_to_cart`
tool calls `service.add_to_cart`. One grep takes you from a model's tool call to
the SQL it caused — the single most useful property this codebase has while
you're learning it.

**Wrappers hold no logic.** Unpack config, call the domain, return the envelope
verbatim. Any rule that lived here would be a rule the REST API doesn't enforce,
so the same order would behave differently clicked versus spoken.

Look at `_context()`. Session and `visit_id` arrive through LangGraph's
`config["configurable"]`, injected by the caller — not globals, not closures.
Explicit, greppable, and no long-lived closure pinning a request scope. If
they're missing it raises loudly, because a tool silently operating on nothing is
far worse than a stack trace.

> **Try it:** `uv run pytest tests/test_graph.py::test_tools_require_injected_config`

---

## Stop 6 — The prompt is code (`api/agent/prompts.py`)

Not a string constant — a function of live state, re-rendered every turn.

Read `render_context()`. It assembles: today's menu with prices, the customer's
name, their usual, the wallet, the current cart, notes, and the upsell flags.

**Today's menu lives in the prompt, not behind a tool call.** This is the most
consequential performance decision in the project. An earlier version required
`get_menu` before quoting any price, which forced an extra
`barista → tools → barista` lap — *two local-model inferences instead of one* —
on the most common turn in the app, to re-read data that cannot change mid-visit.
Menu-sized text in every prompt is far cheaper than a round-trip per turn.

Size prices are one surcharge line rather than three prices per drink; printing
them per item would triple the longest block in the prompt to say the same thing
thirty times.

> **Try it:** in `scripts/shop_cli.py` add `print(system_prompt(dict(state)))`
> somewhere, or just read `tests/test_memory.py::test_notes_reach_the_context_block`
> which asserts a remembered note actually shows up in the next prompt.

---

## Stop 7 — What stops it falling over

Three defences, each learned from a real failure. This is the most important
section for building your own agent.

### 1. Money rules live in the domain, not the prompt

`place_order(confirmed_total_cents)` takes the total the barista just said out
loud. `shop/orders.py` compares it to the real cart total and refuses a mismatch.

The model **cannot lower a charge, only fail one**. It's proving it quoted
correctly. Charging without confirming now requires guessing the exact cart total
— a far higher bar than "the system prompt says confirm first".

This fired on the very first real conversation: the model quoted $6.20 for an
$11.90 cart and got refused. A prompt rule would have silently charged.

### 2. Tool calls run sequentially (`run_tools`)

LangGraph's prebuilt `ToolNode` runs a turn's tool calls **concurrently**. Wrong
here twice over: they share one `AsyncSession` (not safe for concurrent use), and
they're causally ordered. A model emitting `add_to_cart` + `place_order` in one
message means "add it, *then* charge me" — run concurrently, `place_order` reads
the cart before `add_to_cart` committed and fails with `empty_cart`.

Every scripted test emitted one call per turn, so the suite was green while the
real thing was broken.

### 3. A malformed tool call returns an envelope, never raises

Small models drop required arguments constantly. `run_tools` catches the
exception and hands back
`{"ok": false, "error": "invalid_arguments", "message": "..."}` so the barista
fixes it and retries — exactly as with any other tool error.

A tool name the model *invented* takes the same path (`_unknown_tool`), and the
`message` matters just as much there: it lists the tools that do exist, so the
model's next move is a correction rather than another guess. An envelope without
a message would leave it to invent an explanation for the customer — the failure
mode Stop 2 is about. It also gets a `tool.unknown` span and an
`agent.tool.malformed` increment, because a model hallucinating tools is exactly
what that metric is for. The span is named `unknown` rather than after the
invented name: that string came from a language model, and model-written text in
a span name or a metric label is unbounded cardinality.

> **Try it:** `uv run pytest tests/test_graph.py -k "in_one_message or unknown_tool" -v`

---

## Stop 8 — Memory is three different things

A common confusion, kept deliberately separate here:

| Layer | Scope | Where |
| --- | --- | --- |
| Conversation | One visit | LangGraph checkpointer, `thread_id = visit_id` |
| Customer profile | All visits | Aggregated at read time from `orders` |
| Notes | All visits | `customer_preferences.notes`, model-written |

**The rule: never ask an LLM for a fact a `GROUP BY` can produce.** Favourites,
the usual, visit count — all SQL (`api/shop/profile.py`). Only free-text notes
come from the model (`api/agent/summarize.py`).

`usual_order` groups by **(item, size)**, which is what lets the barista say
"large, like always?" instead of asking a regular the same question daily.

Read `summarize.py` for the three rules that exist because of how models behave:

- **"Nothing stood out" must be a normal answer.** A model told to always produce
  a fact will invent one, and invented notes compound across visits into a
  barista confidently misremembering things.
- It runs **after** the stream closes — the customer is already walking out.
- A failure never fails the visit. Memory is a nicety; going home is not.

> **Try it:** `uv run pytest tests/test_memory.py -v` — the parsing tests show
> the mess models actually emit (fenced blocks, chatty preambles, invalid JSON).

---

## Stop 9 — Reading a trace (`api/agent/instrumentation.py`)

**The trace of one turn *is* the agent loop, drawn.** Have a conversation, then
open Grafana on <http://localhost:3001> and find it:

```
agent.turn
├── graph.node.load_context
├── graph.node.barista      → gen_ai.chat   (LLM call #1)
├── graph.node.tools        → tool.add_to_cart
├── graph.node.barista      → gen_ai.chat   (LLM call #2)
└── graph.node.finish
```

Reading that once tells you things a log line never will: that a turn cost two
model round-trips, where the context tokens went, that the database was never the
problem.

Two conventions worth stealing:

- **A failed tool marks its own span, not the parent turn.** "insufficient funds"
  is a normal outcome the agent recovers from. Marking it a request error makes
  every dashboard lie.
- **`agent.loop.iterations` counts laps *per turn*, not one per lap.** It's the
  only place a model going in circles is visible.

> **Try it:** `uv run pytest tests/test_agent_spans.py -v` asserts the span tree
> without needing a collector.

---

## Stop 10 — Getting it to the browser

`api/agent/runner.py` (`run_turn`) turns one turn into typed frames — `token`,
`cart_updated`, `wallet_updated`, `visit_ended`, `done`. Both the CLI and the SSE
endpoint consume it, which is why the terminal and the browser show the same
thing.

`api/routers/chat.py` is the only streaming code. Note it builds its **own**
session rather than using the request-scoped dependency: FastAPI closes those
when the handler returns, which is before the first token.

`web/src/api.ts` (`streamTurn`) reads the stream by hand — `EventSource` only
does GET, and this is a POST. The buffering matters: chunks can split mid-frame,
so the tail is held until its terminator arrives. Otherwise tokens get dropped
exactly when the model is fastest, which is the hardest version of that bug to
notice.

**One stream carries both prose and state changes**, which is what makes their
ordering unambiguous and lets the cart update mid-sentence.

---

## Experiments worth doing

Each one breaks something on purpose. Run the named test, watch it fail,
undo it.

| Change | What breaks | The lesson |
| --- | --- | --- |
| Make `place_order` ignore `confirmed_total_cents` | `test_wrong_quoted_total_is_refused` | Domain gates aren't decoration |
| Swap `run_tools` for `ToolNode(ALL_TOOLS)` | `test_two_tool_calls_in_one_message_run_in_order` | Concurrency vs. a shared session |
| Delete the `refresh` node | `test_cart_is_refreshed_before_the_model_speaks_again` | Stale context makes the model lie |
| Drop the menu from `render_context` | Nothing fails — but watch `agent.loop.iterations` climb | Tests can't catch every regression |
| Remove the `size` check constraint | `test_sized_food_line_is_rejected` | Make bad states unrepresentable |
| Make `summarize` always return a note | `test_summarize_stores_nothing_when_nothing_stood_out` | Models invent when told to produce |

The fourth one is the most instructive: a change that no test catches, but which
doubles the cost of every turn. That's what the metrics are for.

---

## The four bugs this project actually hit

Each is a commit you can read, and each teaches something a tutorial wouldn't.

1. **`69ac67f` — concurrent tool calls.** Found by talking to the real model. No
   scripted test emitted two calls in one message, so the suite was green.
   *Lesson: test the shapes real models produce, not the shapes you imagine.*

2. **`00fc43c` — a malformed call crashed the turn.** Pydantic raised straight
   through the graph.
   *Lesson: the tool boundary must absorb bad input, because the caller is a
   language model.*

3. **`b6d26fb` — a test passing for the wrong reason.** The double-`place_order`
   test hit `empty_cart` and never reached the idempotency constraint at all.
   *Lesson: check which branch your test actually exercises.*

4. **`c54f7c0` — a race the spec missed.** Writing a concurrency test for users
   revealed the same flaw in the visit path: two tabs, two wallets, one day.
   *Lesson: when you find a race in one place, look for its twin.*

---

## Where to go next

The spec's §13 lists what's still open. Good next exercises, hardest last:

- **Add a modifier** (oat milk, extra shot). Touches the catalog, pricing, the
  tool schema, the prompt, and the clarifying-question logic — a full lap of the
  whole system.
- **Show the notes in the UI.** Makes the memory layer visible; also a good
  debugging surface.
- **Use a smaller model for summarization.** Summarising is a different job from
  conversation.
- **Add a second agent** — a manager who restocks or changes prices. This is
  where multi-agent orchestration starts, and the boundary in Stop 1 is what
  makes it tractable.
