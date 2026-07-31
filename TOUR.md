# A guided tour of the code

Read this with the repo open beside you. It goes in the order that makes the
agent make sense, which is **not** the order the files are laid out in.

Roughly an hour of reading, plus the experiments. Every stop names the file
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

Note what holds it up today: nothing but this paragraph. The spec (§5.1) calls for
an import-linter contract in CI and there isn't one yet, which makes it the
cheapest useful contribution to the repo — an in-process boundary is far easier to
breach by accident than a network one.

> **Look:** `api/agent/graph.py`, the `_load_context` function. It needs the
> current day. The obvious way is `session.get(Visit, visit_id)` — but that
> would import a model into `agent/`. It calls `service.get_wallet_balance()`
> instead, which already returns the day. The comment there says so.

---

## Stop 2 — The envelope (`api/shop/result.py`, 44 lines)

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

It returns ten different errors, and six of them are the interesting ones —
**each is a different truth the barista has to tell the customer** (the other
four, `visit_closed`, `invalid_quantity`, `unknown_size` and
`modifier_not_applicable`, are mechanical):

| error | what it means |
| --- | --- |
| `unknown_item` | "We don't do bubble tea." |
| `not_available_today` | "No mochas *today*." — real item, not drawn today |
| `size_required` | "Which size?" — the request was incomplete, not wrong |
| `size_not_applicable` | "A croissant only comes the one size." |
| `unknown_modifier` | "We don't do soy — there's oat or almond." |
| `modifier_conflict` | "Oat or almond, not both." |

Collapsing these into a generic "rejected" would make the barista lie. "We don't
sell mochas" is false when the shop sells mochas on other days.

`size_required` is the interesting one. **It is not really a failure** — it's how
the domain tells the agent the customer's request was incomplete, so the barista
asks instead of guessing. Its message is a question, read aloud verbatim.

Then read the **order** those checks run in, which is itself a decision: the whole
size block fires before any modifier check, so "a latte with soy" comes back
"which size?" rather than "we don't do soy". `size_required` is the only branch
that asks a question, so it has to reach the barista first — otherwise the model
fixes one problem, retries, and discovers the second. Two tests in `test_cart.py`
lock that ordering from both directions.

Note also what modifiers *don't* have: there is no `modifier_required`. A drink
with no modifiers is a complete order, so a modifier request can only ever be
over-specified, never under-specified — the exact mirror of size. That asymmetry
is why the two axes need different error shapes despite looking alike.

> **Try it:** `uv run pytest tests/test_cart.py -v` and read the test names as a
> list. They're written to be a description of the behaviour.

---

## Stop 4 — The loop (`api/agent/graph.py`)

**This is the file.** If you read one thing, read this.

```
START → load_context → barista ⇄ (tools → refresh) → finish → END
```

The brackets matter: the cycle is `barista → tools → refresh → barista`, and
`finish` hangs off `barista`, not off `refresh`. Only a reply with no tool calls
leaves the loop.

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
   would be a round-trip for data that cannot have moved. Note *how* it knows
   which turn it is: `if state.get("menu"): return {}`. That's a bet on state
   surviving between turns — read the next-but-one section before you copy it.
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

### Where the conversation lives between turns

The graph is invoked with **one** message — look at `run_turn`:

```python
graph.astream({"messages": [HumanMessage(content=text)]}, config=config, ...)
```

Nothing in that call carries the previous turn. What makes the barista
conversational is the **checkpointer** plus one line in that config:

```python
"thread_id": str(visit_id),      # agent/runner.py
```

LangGraph loads the thread's saved state, the `add_messages` reducer appends the
new message to it, and the whole state is saved again at the end of the turn.
`agent/checkpointer.py` is the store: Postgres, in its own `agent_checkpoints`
schema so that LangGraph's table format — which will change under you — stays out
of your own tables. `thread_id = visit_id` is the design: one visit is one
conversation, and re-entering a visit resumes it mid-sentence.

This is the piece to get right first in any agent you build, and the easiest to
leave half-wired: the checkpointer is process-wide, opened in `main.py`'s lifespan
and handed to the router with `set_checkpointer()`. If it is missing, nothing
crashes — every turn just starts from an empty state and the agent quietly
becomes amnesiac. That exact bug shipped here; see bug 6 at the end.

> **Try it:** run the CLI, order a latte over two messages ("a latte" then
> "large"), then type `/state` — it prints the message count from
> `graph.aget_state`, i.e. what the checkpointer is holding for that visit.

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

**The docstrings are prompt text.** This is the part people miss. `bind_tools`
sends each tool's name, signature and docstring to the model as its schema, so
`add_to_cart`'s docstring —

```
size: REQUIRED for drinks, and it must be what the customer actually said —
ask them if they did not say one. Omit this argument entirely for food;
passing it for a cookie or pastry is an error.

modifiers: drinks only, and only if the customer asked for one. Use these
exact codes: "oat_milk", "almond_milk", "extra_shot". At most one milk.
Plain or regular milk is what the drink already comes with — send no
modifier for it. Omit this argument entirely otherwise.
```

— is not documentation for you. It is an instruction to the model, written in the
place the model is guaranteed to read it, and it is the cheapest lever you have
on tool-calling accuracy. Note what it does *not* do: it names the sizes and the
modifier codes, but never says what either **costs**. Prices stay in the domain,
so the schema can't teach the model to invent one — the same reason no tool takes
a price argument.

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

> **Try it:** the state the prompt is rendered from lives inside the graph, so
> reach it through the checkpointer rather than by adding a print: the CLI has
> `/state`, `/cart` and `/wallet` for exactly this (`scripts/shop_cli.py`,
> `graph.aget_state`). Then read
> `tests/test_memory.py::test_notes_reach_the_context_block`, which asserts a
> remembered note actually shows up in the next prompt.

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

`run_tools` executes a turn's tool calls one at a time, in the order the model
emitted them. That is the current, correct behaviour — read on for why it is worth
a hand-written node.

The obvious alternative is LangGraph's prebuilt `ToolNode`, which runs a turn's
calls **concurrently**. That would be wrong here twice over: they share one
`AsyncSession` (not safe for concurrent use), and they're causally ordered. A
model emitting `add_to_cart` + `place_order` in one message means "add it, *then*
charge me"; run concurrently, `place_order` reads the cart before `add_to_cart`
committed and fails with `empty_cart`.

The concurrent version did ship once (`69ac67f`, bug 1 below) and was caught by
talking to the real model, not by the suite: every scripted test emitted one call
per turn, so CI stayed green while the real thing was broken. The spec records the
decision and its rejected alternative in §13.7.

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

`usual_order` is stitched from **two** aggregates in `customer_profile()`, and
it's worth seeing why: `_ordered_totals()` groups by (item, category) to find the
favourite drink and food, then `_favourite_size_per_item()` groups by (item,
size) to find how they take it. The second query is what lets the barista say
"large, like always?" instead of asking a regular the same question daily. One
combined `GROUP BY (item, size)` would split someone's Latte habit across three
size rows and make their favourite drink look like three lesser ones.

Each line also carries `available_today`, set by `_available_today()`. A profile
that suggests something the shop didn't draw today is worse than one that
suggests nothing.

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
├── graph.node.load_context      (first turn of the visit only)
├── graph.node.barista      → gen_ai.chat   (LLM call #1)
├── graph.node.tools        → tool.add_to_cart
├── graph.node.refresh
├── graph.node.barista      → gen_ai.chat   (LLM call #2)
└── graph.node.finish
```

Grafana needs the stack up (`make up`); the span tree itself is asserted by tests
without a collector.

Reading that once tells you things a log line never will: that a turn cost two
model round-trips, where the context tokens went, that the database was never the
problem.

Two conventions worth stealing:

- **A failed tool marks its own span, not the parent turn.** "insufficient funds"
  is a normal outcome the agent recovers from. Marking it a request error makes
  every dashboard lie.
- **`agent.loop.iterations` counts laps *per turn*, not one per lap.** It's the
  only place a model going in circles is visible.
- **Every node that touches the database gets a span**, or the trace has a hole
  in it. `refresh` was missing one, so its two queries appeared under the parent
  with nothing to attribute them to — the kind of gap you only notice when you
  are already lost. Same class of mistake as `run_tools` missing one before it.
- **Model-written strings never become span names or metric labels.** An invented
  tool name goes in an attribute (Stop 7); the span stays `tool.unknown`.
  Cardinality is a property of your dashboard's health, not of the model's mood.

> **Try it:** `uv run pytest tests/test_agent_spans.py -v` asserts the span tree
> without needing a collector.

---

## Stop 10 — Getting it to the browser

`api/agent/runner.py` (`run_turn`) turns one turn into typed frames — `token`,
`cart_updated`, `wallet_updated`, `visit_ended`, `done`, and `error`. Both the CLI
and the SSE endpoint consume it, which is why the terminal and the browser show
the same thing — and why the CLI is a usable debugging surface for a web bug.

The frames come from two of LangGraph's stream modes at once:
`stream_mode=["messages", "values"]`. `messages` gives token-by-token output —
filtered to `AIMessage` chunks only, because forwarding `ToolMessage` chunks
printed raw envelope JSON into the conversation — and `values` gives a state
snapshot per node, which `_domain_frames` turns into `cart_updated` /
`wallet_updated`. That is why the cart can move mid-sentence.

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

## Stop 11 — Three roles, one voice (`api/agent/delegates.py`)

Sam works the counter. Mo works the machine. Val works the till. Mo and Val are
LLM sub-agents that Sam reaches through `ask_barista` and `ring_up` — **agents as
tools**, not three peers passing control between them.

That shape was chosen on cost. Peer handoff costs 3–5 local inferences per turn
against today's 1–2, and Stop 6 is about a decision that deleted *one*. It also
keeps every invariant in this document intact for free: a delegation returns the
same `{ok, error, message}` envelope, so `run_tools`, the malformed-call path and
the `tool.*` spans all work unchanged.

**Read `_run_subagent`.** It is a small tool loop — the same shape as the main
graph, capped at three laps — that collapses to one envelope.

### The part that actually matters: every gate is structural

The project's rule has always been "gate in code, not prose" (Stop 7). Adding a
second model tested that rule three times, and prose lost every time:

| What was tried | What the real model did |
| --- | --- |
| Telling Sam to send extras to Mo | Handled "a large espresso with oat milk" alone. Mo never ran once. |
| Letting Val name the total it charges | Would read the cart, always match, and turn `confirmed_total_cents` into a rubber stamp. |
| Giving Val every till tool | Asked only to take payment, it closed out the visit too. |

So: Sam's `add_to_cart` has **no `modifiers` argument** — there are two shapes of
that tool over the same service function, and only Mo gets the full one.
`charge_the_customer` and `send_them_home` take **no arguments at all**; the
quoted total arrives through `config`, the same path `session` does. And Val is
handed only the tools Sam authorised for that one job.

> **Experiment:** put `modifiers` back on the waiter's `add_to_cart` and talk to
> the real model. Watch `agent.delegations` stay at zero. That is what a role
> nothing can reach looks like.

### Two bugs a scripted test would never have found

**LangChain silently drops an argument a tool does not declare**, and runs the
call without it — so Sam asking for oat milk with a tool that has no such
argument produced a *plain* espresso, reported as a success. A silent wrong
order is worse than either alternative. `dispatch.execute_tool_call` now rejects
unknown arguments outright.

**One `ok` cannot carry the money path.** "Any failed step fails the delegation"
labelled a charge that had gone through as a failure, and Sam told the customer
their payment had not worked. "Any success succeeds it" would announce an order
nobody paid for. `ring_up` reports `charged` and `visit_ended` as separate facts,
and Sam's rule keys on `charged`.

### And one in the browser

Models talk *and* call a tool in the same message. Both reach the SSE stream, so
the customer read "Latte added." followed by "Sure thing! That's a large latte…".
A prompt rule does not hold that at temperature 0 — the model narrates its own
tool call regardless. The stream carries an explicit `reset_reply` frame instead:
if tools ran, whatever was said beforehand was premature, so the UI drops it.

> **Try it:** `uv run pytest tests/test_graph.py -k "cashier or barista" -v`

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
| `set_checkpointer(None)` in `main.py`'s lifespan | `test_the_endpoint_remembers_the_previous_turn` | No store, no conversation |

The fourth one is the most instructive: a change that no test catches, but which
doubles the cost of every turn. That's what the metrics are for.

---

## The bugs this project actually hit

Each is a commit you can read, and each teaches something a tutorial wouldn't.
The first four landed inside the phase commit that fixed them, so expect a large
diff around a small fix.

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

5. **An invented tool name came back without a `message`.** Every other envelope
   in the project carries the sentence the barista says; this one branch returned
   a bare `{ok, error}`, leaving the model to make one up. It was also invisible
   in the trace, while `agent.tool.malformed` claimed in its own description to
   count invented tools.
   *Lesson: an invariant holds only on the paths you actually wrote it on. Grep
   for the branch that skips it — it will be the error path.*

6. **The browser was amnesiac; the CLI was not.** `routers/chat.py` compiled the
   graph with `checkpointer=None`, so every POST started from an empty state:
   "a latte" → "which size?" → "large" lost the latte, `load_context` re-ran
   every turn, and the upsell backstops reset between messages. `scripts/shop_cli.py`
   opened a real checkpointer, so the terminal — where all the manual testing
   happened — worked perfectly.
   *Lesson: the second caller of your agent is where the wiring rots. Test the
   entry point the user actually uses, and remember that a missing checkpointer
   fails silently rather than loudly.*

7. **A small model copied the prompt's own example.** The summarizer's few-shot
   examples were realistic notes; `qwen2.5:3b` returned one verbatim, inventing
   a mocha the customer never mentioned. `14b` did not, so it had been hiding
   for the whole project.
   *Lesson: a small model reads an example as content to reuse, not as a shape
   to imitate — and the bigger model masks it. Run your prompts on a smaller
   model on purpose.*

8. **The decision panel replayed old turns.** `messages` is the whole
   checkpointed thread, not the current turn's slice, so a per-turn "already
   reported" set that starts empty re-emits every earlier turn's tool calls
   under the current one.
   *Lesson: with a checkpointer, "what happened this turn" is never just "what
   is in state" — and a one-turn probe cannot show you the difference.*

9. **A tool argument that vanished.** LangChain silently drops an argument the
   tool does not declare and runs the call anyway, so a waiter asking for oat
   milk with a tool that has no such argument produced a plain drink and
   reported success.
   *Lesson: the framework's forgiveness is your silent wrong answer. Validate
   the call against the schema you actually bound.*

---

## The same lessons, without the coffee

Everything above is one domain's version of a general rule. If you build a
different agent, this is the transferable part — the shop is only the example.

| Build this | Here it is | Because |
| --- | --- | --- |
| A domain layer that doesn't know an LLM exists | `shop/` (Stop 1) | It stays testable as ordinary code, and the same rules apply to every caller |
| One result shape, message mandatory | `shop/result.py` (Stop 2) | A failure without a sentence is a failure the model narrates for you |
| Errors that are distinct truths, not one "rejected" | `add_to_cart` (Stop 3) | Collapsing them is how an agent ends up lying politely |
| Loop control in one readable function | `route_after_barista` (Stop 4) | If you can't point at the loop, you can't reason about its cost |
| State that survives the turn, keyed on the conversation | checkpointer + `thread_id` (Stop 4) | Without it your agent is a one-shot prompt with extra steps |
| Tools named after the functions they call | `agent/tools.py` (Stop 5) | One grep from tool call to SQL, forever |
| Tool docstrings written *at the model* | `add_to_cart`'s schema (Stop 5) | It is the one instruction the model always reads |
| A prompt rendered from live state each turn | `render_context()` (Stop 6) | A stale context block makes the model confidently wrong |
| Irreversible actions gated in code, not prose | `confirmed_total_cents` (Stop 7) | "The prompt says not to" is the weakest enforcement available |
| Sequential tools when calls are causal | `run_tools` (Stop 7) | The model's order is the customer's intent |
| A tool boundary that absorbs bad input | `invalid_arguments`, `unknown_tool` (Stop 7) | The caller is a language model; it will send nonsense |
| Facts from SQL, opinions from the model | `profile.py` vs `summarize.py` (Stop 8) | Never ask an LLM for something a `GROUP BY` can produce |
| A trace shaped like the loop | `instrumentation.py` (Stop 9) | "Two inferences per turn" is invisible in logs |
| A scripted model in the test suite | `tests/fakes.py` | Deterministic tests of the shapes real models produce |
| Sub-agents as tools, not peers | `agent/delegates.py` (Stop 11) | A handoff you can't afford is a handoff you won't ship |
| Role boundaries enforced by schema | Sam's `add_to_cart` (Stop 11) | Given the argument, the model will never delegate |
| Authority injected, not argued | `charge_the_customer()` (Stop 11) | A second model must not be able to restate the first one's promise |

The ordering is roughly the order to build them in. The first five are structure;
the rest are the things you only learn by watching a small model behave badly.

---

## Where to go next

The spec's §13 lists what's still open. Good next exercises, hardest last:

- ~~**Add a modifier** (oat milk, extra shot).~~ **Done** — see §3.6 and §13.8.
  It was a full lap of the whole system, and the two bugs it surfaced are worth
  reading: `change_size` merged an oat latte into a plain one at the plain price,
  and picked between modifier variants with `session.scalar()`, which returns the
  first row without complaining about the rest. Both were silent, and neither
  would have failed a test written before modifiers existed.
- ~~**Show the notes in the UI.**~~ **Done** — §13.9. Two panels: "What Sam
  remembers" and "What Sam did". The second shows the *tool record*, not a
  model-authored rationale, because `qwen2.5` is not a reasoning model and would
  invent an explanation that could contradict the calls listed beside it.
  The bug worth reading: `messages` is the whole checkpointed thread, so a
  per-turn "already reported" set that starts empty replays every earlier turn's
  calls under the current one. A one-turn probe cannot show it.
- ~~**Use a smaller model for summarization.**~~ **Done** — §13.10, opt-in via
  `OLLAMA_SUMMARY_MODEL`. The interesting part was not the speed. The prompt used
  to illustrate its output with real-looking notes, and `3b` copied one verbatim,
  inventing a mocha nobody ordered. **A small model reads a few-shot example as
  content to reuse, not as a shape to imitate** — and a bigger model hides it, so
  running your prompt on a smaller one on purpose is the cheap way to find it.
- ~~**Add a second agent.**~~ **Done, three of them** — §13.11. See Stop 11.

Still open, and the spec's §13 lists more:

- **Wire the upsell backstops, or delete them.** `upsell_used`, `size_offers` and
  `size_declines` are declared, carried forward and rendered — but nothing ever
  writes them, so the rules are prompt-only and the metrics §13.2 cites as
  verification do not exist. Deciding whether the model *made* an upsell means
  classifying its prose, which is exactly what the rest of the project refuses to
  ask an LLM for. That is the hard part, and it is why nobody has done it.
- **The import-linter contract from Stop 1.** Still nothing but a paragraph.
