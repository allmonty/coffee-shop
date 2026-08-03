/**
 * The whole UI (spec §4).
 *
 * Two screens: the storefront with a name box, and the shop. No routing —
 * there is one decision (are we inside?) and `useState` covers it.
 *
 * Free text only, no suggested-reply chips (spec §13 decision 1). Onboarding
 * happens in the conversation: the barista's opening line asks what they'd like
 * and mentions the menu. The only non-text control is Go Home, which fires an
 * unambiguous event rather than relying on the model reading "bye" correctly.
 */

import { useEffect, useRef, useState } from "react";
import {
  Cart,
  EnterResponse,
  Frame,
  MenuItem,
  Profile,
  ToolResult,
  formatCents,
  enterShop,
  fetchProfile,
  streamTurn,
} from "./api";

type Turn = { who: "you" | "sam"; text: string };

/** One turn's worth of what the agent did, for the decision panel. */
type Decision = { turn: number; calls: ToolResult[]; loops: number };

const EMPTY_CART: Cart = { lines: [], total_cents: 0 };

export default function App() {
  const [visit, setVisit] = useState<EnterResponse | null>(null);
  const [transitioning, setTransitioning] = useState(false);

  if (!visit) return <Storefront onEnter={setVisit} />;

  return (
    <Shop
      key={visit.visit_id}
      visit={visit}
      transitioning={transitioning}
      onDayEnd={async () => {
        setTransitioning(true);
        // A beat, so the day change reads as a change rather than a flicker.
        await new Promise((r) => setTimeout(r, 1400));
        setTransitioning(false);
        setVisit(null);
      }}
    />
  );
}

function Storefront({ onEnter }: { onEnter: (v: EnterResponse) => void }) {
  // Remembered only to pre-fill. The name IS the identity; clearing this just
  // means typing it again (spec §4.1).
  const [name, setName] = useState(localStorage.getItem("coffee-shop-name") ?? "");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    if (!name.trim() || busy) return;
    setBusy(true);
    setError(null);
    try {
      const visit = await enterShop(name);
      localStorage.setItem("coffee-shop-name", visit.name);
      onEnter(visit);
    } catch (problem) {
      setError((problem as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="storefront">
      <div className="sign">☕</div>
      <h1>The Coffee Shop</h1>
      <form onSubmit={submit}>
        <label htmlFor="name">What&rsquo;s your name?</label>
        <input
          id="name"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="Allan"
          maxLength={40}
          autoFocus
        />
        <button type="submit" disabled={!name.trim() || busy}>
          {busy ? "Opening the door…" : "Enter Coffee Shop"}
        </button>
      </form>
      {error && <p className="error">{error}</p>}
    </main>
  );
}

function Shop({
  visit,
  transitioning,
  onDayEnd,
}: {
  visit: EnterResponse;
  transitioning: boolean;
  onDayEnd: () => void;
}) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [cart, setCart] = useState<Cart>(EMPTY_CART);
  const [wallet, setWallet] = useState(visit.wallet_cents);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [decisions, setDecisions] = useState<Decision[]>([]);
  const [profile, setProfile] = useState<Profile | null>(null);
  const greeted = useRef(false);
  const log = useRef<HTMLDivElement>(null);

  useEffect(() => {
    void fetchProfile(visit.user_id).then(setProfile);
  }, [visit.user_id]);

  useEffect(() => {
    // The barista speaks first (spec §4.3). StrictMode double-invokes effects
    // in dev, so this guard stops a duplicate greeting.
    if (greeted.current) return;
    greeted.current = true;
    void send({ event: "on_enter" });
  }, []);

  useEffect(() => {
    log.current?.scrollTo({ top: log.current.scrollHeight, behavior: "smooth" });
  }, [turns]);

  async function send(body: { message?: string; event?: string }) {
    setBusy(true);
    if (body.message) setTurns((t) => [...t, { who: "you", text: body.message! }]);
    setTurns((t) => [...t, { who: "sam", text: "" }]);

    let ended = false;
    const turnNumber = decisions.length + 1;
    try {
      await streamTurn({ visit_id: visit.visit_id, ...body }, (frame: Frame) => {
        switch (frame.type) {
          case "tool_result": {
            const { type: _type, ...call } = frame;
            setDecisions((d) => {
              const next = [...d];
              const last = next[next.length - 1];
              if (last?.turn === turnNumber) {
                next[next.length - 1] = { ...last, calls: [...last.calls, call] };
              } else {
                next.push({ turn: turnNumber, calls: [call], loops: 0 });
              }
              return next;
            });
            break;
          }
          case "turn_stats":
            setDecisions((d) =>
              d.map((entry) =>
                entry.turn === turnNumber ? { ...entry, loops: frame.loop_count } : entry,
              ),
            );
            break;
          case "token":
            // Appended as it arrives, so the reply types itself out.
            setTurns((t) => {
              const next = [...t];
              next[next.length - 1] = {
                who: "sam",
                text: next[next.length - 1].text + frame.text,
              };
              return next;
            });
            break;
          case "cart_updated":
            setCart({ lines: frame.lines ?? [], total_cents: frame.total_cents ?? 0 });
            break;
          case "reset_reply":
            // The model spoke in the same message as a tool call, before it
            // knew the result. Drop that; the real reply is coming.
            setTurns((t) => {
              const next = [...t];
              next[next.length - 1] = { who: "sam", text: "" };
              return next;
            });
            break;
          case "wallet_updated":
            setWallet(frame.wallet_cents);
            break;
          case "visit_ended":
            ended = true;
            // Notes are written as the visit closes, so this is the one moment
            // the panel can gain a line without a reload.
            void fetchProfile(visit.user_id).then(setProfile);
            break;
          case "error":
            setTurns((t) => {
              const next = [...t];
              next[next.length - 1] = { who: "sam", text: "…sorry, I lost my train of thought." };
              return next;
            });
            break;
        }
      });
    } finally {
      setBusy(false);
    }

    if (ended) onDayEnd();
  }

  function submit(event: React.FormEvent) {
    event.preventDefault();
    const text = input.trim();
    if (!text || busy) return;
    setInput("");
    void send({ message: text });
  }

  if (transitioning) {
    return (
      <main className="transition">
        <p>— next morning —</p>
      </main>
    );
  }

  return (
    <main className="shop">
      <header>
        <span className="who">{visit.name}</span>
        <span className="day">
          Day {visit.day} · {visit.weekday}
        </span>
        <span className="wallet">{formatCents(wallet)}</span>
      </header>

      <div className="body">
        <section className="chat">
          <div className="log" ref={log}>
            {turns.map((turn, index) => (
              <p key={index} className={turn.who}>
                <span className="speaker">{turn.who === "you" ? visit.name : "Sam"}</span>
                {turn.text || (busy && index === turns.length - 1 ? "…" : "")}
              </p>
            ))}
          </div>
          <form onSubmit={submit}>
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Say something…"
              disabled={busy}
              autoFocus
            />
            <button type="submit" disabled={busy || !input.trim()}>
              Send
            </button>
          </form>
        </section>

        <aside>
          <CartPanel cart={cart} />
          <MemoryPanel profile={profile} />
          <DecisionPanel decisions={decisions} />
          <MenuPanel menu={visit.menu} />
          <button className="go-home" onClick={() => void send({ event: "go_home" })} disabled={busy}>
            Go Home
          </button>
        </aside>
      </div>
    </main>
  );
}

function CartPanel({ cart }: { cart: Cart }) {
  return (
    <div className="panel">
      <h2>Your order</h2>
      {cart.lines.length === 0 ? (
        <p className="empty">Nothing yet.</p>
      ) : (
        <>
          <ul>
            {cart.lines.map((line, index) => (
              <li key={index}>
                <span>
                  {line.quantity} × {line.size ? `${line.size} ` : ""}
                  {line.item}
                  {line.modifiers?.length ? (
                    <em className="mods"> {line.modifiers.map(pretty).join(", ")}</em>
                  ) : null}
                </span>
                <span>{formatCents(line.line_total_cents)}</span>
              </li>
            ))}
          </ul>
          <p className="total">
            <span>Total</span>
            <span>{formatCents(cart.total_cents)}</span>
          </p>
        </>
      )}
    </div>
  );
}

/** `oat_milk` -> `oat milk`. Codes are the model's vocabulary, not the customer's. */
function pretty(code: string): string {
  return code.replace(/_/g, " ");
}

/**
 * What Sam remembers about you (spec §13, decision 9).
 *
 * The structured lines come from a GROUP BY; only the notes were written by the
 * model. Showing which is which is most of the value — it is the difference
 * between an agent with a transcript and an agent with memory, made visible.
 */
function MemoryPanel({ profile }: { profile: Profile | null }) {
  if (!profile || profile.visit_count === 0) return null;

  return (
    <details className="panel" open>
      <summary>
        <h2>What Sam remembers</h2>
      </summary>
      <ul>
        <li>
          <span>Visits</span>
          <span>{profile.visit_count}</span>
        </li>
        {profile.favorite_drink && (
          <li>
            <span>Usual drink</span>
            <span>{profile.favorite_drink}</span>
          </li>
        )}
        {profile.favorite_food && (
          <li>
            <span>Usual food</span>
            <span>{profile.favorite_food}</span>
          </li>
        )}
      </ul>
      {profile.notes.length > 0 ? (
        <ul className="notes">
          {profile.notes.map((note, index) => (
            <li key={index}>“{note}”</li>
          ))}
        </ul>
      ) : (
        <p className="empty">No notes yet — Sam writes those up after you leave.</p>
      )}
    </details>
  );
}

/**
 * What the agent actually did, not what it says it did.
 *
 * Deliberately the tool record rather than a model-authored rationale: the
 * model would be inventing an explanation after the fact, and could contradict
 * the calls listed right beside it.
 */
function DecisionPanel({ decisions }: { decisions: Decision[] }) {
  if (decisions.length === 0) return null;
  const recent = decisions.slice(-4).reverse();

  return (
    <details className="panel decisions">
      <summary>
        <h2>What Sam did</h2>
      </summary>
      {recent.map((decision) => (
        <div className="turn" key={decision.turn}>
          <h3>
            Turn {decision.turn}
            {decision.loops > 0 && (
              <span className="loops">
                {decision.loops} model call{decision.loops === 1 ? "" : "s"}
              </span>
            )}
          </h3>
          {decision.calls.map((call, index) => (
            <Call call={call} key={index} />
          ))}
        </div>
      ))}
    </details>
  );
}

function Call({ call }: { call: ToolResult }) {
  return (
    <div className={call.ok ? "call" : "call failed"}>
      <code>
        {call.agent ? <span className="by">{call.agent} · </span> : null}
        {call.tool}
      </code>
      <dl>
        {Object.entries(call.args).map(([key, value]) => (
          <div key={key}>
            <dt>{key}</dt>
            <dd>{Array.isArray(value) ? value.map(String).map(pretty).join(", ") : String(value)}</dd>
          </div>
        ))}
      </dl>
      <p>
        <span className="mark">{call.ok ? "✓" : "✗"}</span>
        {call.ok ? call.message : `${call.error} — ${call.message ?? ""}`}
      </p>
      {call.steps?.length > 0 && (
        <div className="steps">
          {call.steps.map((step, index) => (
            <Call call={step} key={index} />
          ))}
        </div>
      )}
    </div>
  );
}

function MenuPanel({ menu }: { menu: MenuItem[] }) {
  const drinks = menu.filter((item) => item.category === "drink");
  const foods = menu.filter((item) => item.category === "food");

  return (
    <div className="panel">
      <h2>Today</h2>
      <h3>Drinks</h3>
      <ul>
        {drinks.map((item) => (
          <li key={item.name}>
            <span>{item.name}</span>
            <span>{formatCents(item.price_cents)}</span>
          </li>
        ))}
      </ul>
      <h3>Food</h3>
      <ul>
        {foods.map((item) => (
          <li key={item.name}>
            <span>{item.name}</span>
            <span>{formatCents(item.price_cents)}</span>
          </li>
        ))}
      </ul>
      <p className="hint">
        Drink prices are for small · medium +$0.60 · large +$1.20
        <br />
        Extras (drinks only): oat or almond milk +$0.60 · extra shot +$1.00
      </p>
    </div>
  );
}
