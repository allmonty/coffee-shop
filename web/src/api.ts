/**
 * Everything that talks to the backend (spec §7).
 *
 * The chat endpoint is SSE over POST, which `EventSource` cannot do — it only
 * does GET. So the stream is read off `fetch` manually. That is the one piece
 * of real work in this file.
 */

export type MenuItem = {
  name: string;
  category: "drink" | "food";
  price_cents: number;
  sized: boolean;
};

export type CartLine = {
  item: string;
  size: string | null;
  modifiers: string[];
  quantity: number;
  unit_price_cents: number;
  line_total_cents: number;
};

/** One thing the agent did, paired call-to-result. `steps` holds a delegation's
 *  own tool calls, which never reach the graph's message list. */
export type ToolResult = {
  tool: string;
  args: Record<string, unknown>;
  ok: boolean;
  error: string | null;
  message: string | null;
  agent: string | null;
  steps: ToolResult[];
};

export type Profile = {
  name: string;
  visit_count: number;
  favorite_drink: string | null;
  favorite_food: string | null;
  last_visit_day: number | null;
  notes: string[];
};

export type Cart = { lines: CartLine[]; total_cents: number };

export type EnterResponse = {
  user_id: string;
  name: string;
  is_new: boolean;
  visit_id: string;
  day: number;
  weekday: string;
  wallet_cents: number;
  menu: MenuItem[];
};

export type Frame =
  | { type: "token"; text: string }
  | { type: "cart_updated"; lines: CartLine[]; total_cents: number }
  | { type: "wallet_updated"; wallet_cents: number }
  | { type: "visit_ended"; day: number; wallet_cents: number }
  | ({ type: "tool_result" } & ToolResult)
  | { type: "reset_reply" }
  | { type: "turn_stats"; loop_count: number }
  | { type: "done"; visit_ended: boolean }
  | { type: "error"; error: string; detail?: string };

export function formatCents(cents: number): string {
  return `$${(cents / 100).toFixed(2)}`;
}

export async function enterShop(name: string): Promise<EnterResponse> {
  const response = await fetch("/api/enter", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body?.detail?.message ?? "Could not open the door.");
  }
  return response.json();
}

/**
 * What Sam remembers. Notes are written when a visit ends, so this is empty on a
 * first visit and fills in from the next day on.
 */
export async function fetchProfile(userId: string): Promise<Profile | null> {
  const response = await fetch(`/api/users/${userId}/profile`);
  if (!response.ok) return null;
  return response.json();
}

/**
 * Stream one turn, calling `onFrame` as each event arrives.
 *
 * Frames are newline-delimited `data: {...}` blocks. Chunks can split mid-frame,
 * so the tail is kept in a buffer until its terminator shows up — otherwise
 * tokens get dropped exactly when the model is fastest.
 */
export async function streamTurn(
  body: { visit_id: string; message?: string; event?: string },
  onFrame: (frame: Frame) => void,
): Promise<void> {
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!response.body) throw new Error("No stream from the shop.");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const blocks = buffer.split("\n\n");
    buffer = blocks.pop() ?? "";

    for (const block of blocks) {
      const line = block.split("\n").find((l) => l.startsWith("data: "));
      if (!line) continue;
      try {
        onFrame(JSON.parse(line.slice(6)) as Frame);
      } catch {
        // A malformed frame should not kill the conversation.
      }
    }
  }
}
