// main.ts — one server, two APIs, ten free-tier providers.
//
// Serves POST /v1/chat/completions (OpenAI) and POST /v1/messages
// (Anthropic), in JSON and in server-sent events, and GET /v1/models
// in the wire its auth header names. At start it reads the brief at
// $MAKEFUN_PROMPT_FILE, probes every provider's catalog route,
// registers its port and catalog at $MAKEFUN_ENV_URL/register, serves
// the twenty requests the env's client sends one at a time, and
// exits 0 when GET /work says the workload is done.
//
// Routing: for each request, one candidate model per provider — that
// provider's best-rated free model meeting the request's
// requirements — sorted by rating, best first. A provider's published
// per-trial minimum budget (calls_min) is its cap: the first pass
// only tries providers still under their cap, the second pass tries
// the rest, so a request is still served when every candidate is at
// cap. A refusal (402, 429 or 5xx) fails over to the next candidate;
// a provider that fails three times in the trial is treated as dead
// and never dialled again. Exactly one successful provider call
// answers each request. Every choice is a function of the brief, the
// requests and the providers' answers: no wall clock, no unseeded
// randomness.
//
// node main.ts --list prints the catalog document the server
// registers, byte for byte; outside a trial, with no brief to read
// and no provider route to probe, it prints {"providers":[]}.
import { createServer } from "node:http";
import type { IncomingMessage, ServerResponse } from "node:http";
import type { AddressInfo } from "node:net";
import { readFileSync } from "node:fs";
import {
  parsePrompt,
  parseOpenAI,
  parseAnthropic,
  toOpenAI,
  toAnthropic,
  toGemini,
  dialProvider,
  probeCatalog,
  usageOf,
} from "./lib/upstream.ts";
import type {
  Brief,
  ProviderDir,
  RatingRow,
  ManifestReq,
  Requires,
  ClientReq,
  CatalogModel,
} from "./lib/upstream.ts";

// The providers' published per-trial minimum call budgets. A trial
// draws each provider's real budget between this floor and its
// ceiling, so staying under the floor never spends a budget refusal;
// the second pass spends the headroom above it when a request would
// otherwise go unserved.
const CALLS_MIN: Record<string, number> = {
  groq: 9,
  cerebras: 5,
  gemini: 6,
  zai: 4,
  openrouter: 10,
  sambanova: 4,
  pollinations: 10,
  cloudflare: 7,
  ovh: 5,
  kilol: 12,
};

const FIXED_CREATED = 1767225600;
const ANTHROPIC_VERSION = "2023-06-01";

interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: string;
  models: CatalogModel[];
}

interface Catalog {
  providers: CatalogProvider[];
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function jsonBody(res: ServerResponse, status: number, value: unknown): void {
  const text = `${JSON.stringify(value)}\n`;
  res.writeHead(status, { "content-type": "application/json", "content-length": String(Buffer.byteLength(text)) });
  res.end(text);
}

function modelMeets(row: RatingRow, requires: Requires): boolean {
  if (!row.free) return false;
  if (requires.tools === true && !row.tools) return false;
  if (requires.vision === true && !row.vision) return false;
  if (requires.json === true && !row.json) return false;
  if (requires.min_context !== undefined && row.context_length < requires.min_context) return false;
  return true;
}

// One catalog document for the whole brief: every provider in
// directory order, with the models its own catalog route lists. A
// route that cannot be read falls back to the brief's quality table,
// which names every free model; a paid model is never listed as free.
async function buildCatalog(brief: Brief): Promise<Catalog> {
  const providers: CatalogProvider[] = [];
  for (const provider of brief.providers) {
    let models = await probeCatalog(provider);
    if (models.length === 0) {
      models = brief.ratings
        .filter((row) => row.provider === provider.id && row.free)
        .map((row) => ({ id: row.model, free: true, context_length: row.context_length }));
    }
    providers.push({
      id: provider.id,
      base_url: provider.path,
      key_env: provider.key_env,
      wire: provider.wire,
      models,
    });
  }
  return { providers };
}

interface Candidate {
  provider: ProviderDir;
  model: string;
  rating: number;
}

// One model per provider: its best-rated free model meeting the
// request's requirements, best rating first. A JSON-mode request
// from an OpenAI client never goes to an Anthropic-wire provider:
// that wire carries no response_format, so the provider would echo
// json_mode false and the request would fail its echo check.
function candidatesFor(brief: Brief, req: ManifestReq, client: ClientReq): Candidate[] {
  const out: Candidate[] = [];
  for (const provider of brief.providers) {
    if (client.jsonMode && client.api === "openai" && provider.wire === "anthropic") continue;
    let best: RatingRow | null = null;
    for (const row of brief.ratings) {
      if (row.provider !== provider.id) continue;
      if (!modelMeets(row, req.requires)) continue;
      if (best === null || row.rating_milli > best.rating_milli) best = row;
    }
    if (best !== null) out.push({ provider, model: best.model, rating: best.rating_milli });
  }
  out.sort((a, b) => b.rating - a.rating);
  return out;
}

// ---- Rebuilding a provider's answer in the client's wire ----

// Stop reasons fall in four families; the provider's family is what
// the client compares, so the answer is re-emitted with the same
// family's reason in the client's own words.
function stopFamily(reason: string): number {
  if (reason === "stop" || reason === "end_turn" || reason === "STOP") return 0;
  if (reason === "length" || reason === "max_tokens" || reason === "MAX_TOKENS") return 1;
  if (reason === "tool_calls" || reason === "function_call" || reason === "tool_use") return 2;
  if (reason === "content_filter" || reason === "refusal" || reason === "SAFETY") return 3;
  return 0;
}

function clientStop(providerStop: string, api: "openai" | "anthropic"): string {
  const family = stopFamily(providerStop);
  if (api === "openai") return ["stop", "length", "tool_calls", "content_filter"][family];
  return ["end_turn", "max_tokens", "tool_use", "refusal"][family];
}

function splitText(text: string, size: number): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += size) pieces.push(text.slice(at, at + size));
  return pieces.length === 0 ? [""] : pieces;
}

function sseFrame(event: string | null, data: unknown): string {
  const head = event === null ? "" : `event: ${event}\n`;
  return `${head}data: ${JSON.stringify(data)}\n\n`;
}

function openAIResponse(
  model: string,
  text: string,
  usage: { input: number; output: number },
  stop: string,
  stream: boolean,
): { status: number; contentType: string; body: string } {
  const finish = clientStop(stop, "openai");
  const tokens = { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output };
  if (!stream) {
    return {
      status: 200,
      contentType: "application/json",
      body: `${JSON.stringify({
        id: "chatcmpl-agg",
        object: "chat.completion",
        created: FIXED_CREATED,
        model,
        choices: [{ index: 0, message: { role: "assistant", content: text }, finish_reason: finish, logprobs: null }],
        usage: tokens,
      })}\n`,
    };
  }
  const chunk = (delta: Record<string, unknown>, finishReason: string | null): Record<string, unknown> => ({
    id: "chatcmpl-agg",
    object: "chat.completion.chunk",
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finishReason }],
  });
  const frames: string[] = [sseFrame(null, chunk({ role: "assistant", content: "" }, null))];
  for (const piece of splitText(text, 13)) frames.push(sseFrame(null, chunk({ content: piece }, null)));
  frames.push(sseFrame(null, chunk({}, finish)));
  // The client's stream carries no usage unless it asked for it, so
  // the usage the provider reported is appended here as its own
  // chunk, after the finish chunk and before [DONE].
  frames.push(sseFrame(null, { id: "chatcmpl-agg", object: "chat.completion.chunk", created: FIXED_CREATED, model, choices: [], usage: tokens }));
  frames.push("data: [DONE]\n\n");
  return { status: 200, contentType: "text/event-stream; charset=utf-8", body: frames.join("") };
}

function anthropicResponse(
  model: string,
  text: string,
  usage: { input: number; output: number },
  stop: string,
  stream: boolean,
): { status: number; contentType: string; body: string } {
  const reason = clientStop(stop, "anthropic");
  if (!stream) {
    return {
      status: 200,
      contentType: "application/json",
      body: `${JSON.stringify({
        id: "msg_agg",
        type: "message",
        role: "assistant",
        model,
        content: [{ type: "text", text }],
        stop_reason: reason,
        stop_sequence: null,
        usage: { input_tokens: usage.input, output_tokens: usage.output },
      })}\n`,
    };
  }
  const frames: string[] = [
    sseFrame("message_start", {
      type: "message_start",
      message: {
        id: "msg_agg",
        type: "message",
        role: "assistant",
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: usage.input, output_tokens: 0 },
      },
    }),
    sseFrame("content_block_start", { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } }),
  ];
  for (const piece of splitText(text, 13)) {
    frames.push(sseFrame("content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: piece } }));
  }
  frames.push(
    sseFrame("content_block_stop", { type: "content_block_stop", index: 0 }),
    sseFrame("message_delta", { type: "message_delta", delta: { stop_reason: reason, stop_sequence: null }, usage: { output_tokens: usage.output } }),
    sseFrame("message_stop", { type: "message_stop" }),
  );
  return { status: 200, contentType: "text/event-stream; charset=utf-8", body: frames.join("") };
}

// ---- The server ----

interface Served {
  status: number;
  contentType: string;
  body: string;
}

class Aggregator {
  private brief: Brief;
  private envUrl: string;
  private catalog: Catalog;
  private used = new Map<string, number>();
  private failures = new Map<string, number>();
  private handled = 0;

  constructor(brief: Brief, envUrl: string, catalog: Catalog) {
    this.brief = brief;
    this.envUrl = envUrl;
    this.catalog = catalog;
  }

  // The request the env's client is sending now: /work names it
  // while it is in flight, and the manifest order is the fallback.
  private async currentReq(): Promise<ManifestReq> {
    const fallback = this.brief.manifest[Math.min(this.handled, this.brief.manifest.length - 1)];
    try {
      const response = await fetch(`${this.envUrl}/work`, { signal: AbortSignal.timeout(2000) });
      if (response.ok) {
        const work: unknown = await response.json();
        if (typeof work === "object" && work !== null && typeof (work as { next?: unknown }).next === "string") {
          const found = this.brief.manifest.find((m) => m.id === (work as { next: string }).next);
          if (found !== undefined) return found;
        }
      }
    } catch {
      // Fall through to the manifest order.
    }
    return fallback;
  }

  private dead(providerId: string): boolean {
    return (this.failures.get(providerId) ?? 0) >= 3;
  }

  private async serve(api: "openai" | "anthropic", raw: Record<string, unknown>): Promise<Served> {
    const req = await this.currentReq();
    const client: ClientReq = api === "openai" ? parseOpenAI(raw) : parseAnthropic(raw);
    const candidates = candidatesFor(this.brief, req, client);
    const tried = new Set<string>();
    // Pass 1: providers still under their published minimum budget.
    // Pass 2: the rest, so a request is still served when every
    // candidate is at cap. A provider that failed for this request
    // is never tried twice for it.
    for (const pass of [0, 1]) {
      for (const candidate of candidates) {
        const id = candidate.provider.id;
        if (tried.has(id) || this.dead(id)) continue;
        const used = this.used.get(id) ?? 0;
        if (pass === 0 && used >= (CALLS_MIN[id] ?? 0)) continue;
        tried.add(id);
        this.used.set(id, used + 1);
        const wireBody =
          candidate.provider.wire === "openai"
            ? toOpenAI(raw, client, candidate.model)
            : candidate.provider.wire === "anthropic"
              ? toAnthropic(client, candidate.model)
              : toGemini(client);
        const answer = await dialProvider(candidate.provider, wireBody, candidate.model, client);
        if (!answer.ok) {
          this.failures.set(id, (this.failures.get(id) ?? 0) + 1);
          continue;
        }
        const usage =
          answer.usage.input === 0 && answer.usage.output === 0
            ? usageOf(client.systemText, client.userText, client.tools.map((t) => t.name), answer.text)
            : answer.usage;
        const rebuilt =
          api === "openai"
            ? openAIResponse(candidate.model, answer.text, usage, answer.stop, client.stream)
            : anthropicResponse(candidate.model, answer.text, usage, answer.stop, client.stream);
        this.handled += 1;
        return rebuilt;
      }
    }
    this.handled += 1;
    if (api === "openai") {
      return {
        status: 503,
        contentType: "application/json",
        body: `${JSON.stringify({ error: { message: "every provider refused this request", type: "api_error", param: null, code: "all_providers_refused" } })}\n`,
      };
    }
    return {
      status: 503,
      contentType: "application/json",
      body: `${JSON.stringify({ type: "error", error: { type: "api_error", message: "every provider refused this request" } })}\n`,
    };
  }

  // GET /v1/models in the wire the auth header names, listing
  // exactly the free model ids the registered catalog holds, each
  // once, in catalog order.
  private modelsList(req: IncomingMessage): { status: number; contentType: string; body: string } {
    const seen = new Set<string>();
    const ids: { id: string; context_length: number | null }[] = [];
    for (const provider of this.catalog.providers) {
      for (const model of provider.models) {
        if (!model.free || seen.has(model.id)) continue;
        seen.add(model.id);
        ids.push({ id: model.id, context_length: model.context_length });
      }
    }
    const anthropic = typeof req.headers["x-api-key"] === "string";
    if (anthropic) {
      return {
        status: 200,
        contentType: "application/json",
        body: `${JSON.stringify({
          data: ids.map((m) => ({
            type: "model",
            id: m.id,
            display_name: m.id,
            created_at: "2026-01-01T00:00:00Z",
            free: true,
            context_length: m.context_length,
          })),
          has_more: false,
        })}\n`,
      };
    }
    return {
      status: 200,
      contentType: "application/json",
      body: `${JSON.stringify({
        object: "list",
        data: ids.map((m) => ({
          id: m.id,
          object: "model",
          created: FIXED_CREATED,
          owned_by: "free-model-aggregator",
          free: true,
          context_length: m.context_length,
        })),
      })}\n`,
    };
  }

  async handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
    try {
      const url = new URL(req.url ?? "/", "http://entry");
      if (req.method === "GET" && url.pathname === "/v1/models") {
        const answer = this.modelsList(req);
        res.writeHead(answer.status, { "content-type": answer.contentType, "content-length": String(Buffer.byteLength(answer.body)) });
        res.end(answer.body);
        return;
      }
      if (req.method === "POST" && (url.pathname === "/v1/chat/completions" || url.pathname === "/v1/messages")) {
        const api = url.pathname === "/v1/messages" ? "anthropic" : "openai";
        const text = await readBody(req);
        let raw: unknown = null;
        try {
          raw = JSON.parse(text);
        } catch {
          raw = null;
        }
        if (raw === null || typeof raw !== "object" || Array.isArray(raw)) {
          jsonBody(res, 400, { error: { message: "the request body is not one JSON object", type: "invalid_request_error", param: null, code: "invalid_request" } });
          return;
        }
        const answer = await this.serve(api, raw as Record<string, unknown>);
        res.writeHead(answer.status, { "content-type": answer.contentType, "content-length": String(Buffer.byteLength(answer.body)) });
        res.end(answer.body);
        return;
      }
      jsonBody(res, 404, { error: "not_found", message: `no route ${req.method ?? "GET"} ${url.pathname}` });
    } catch (err) {
      jsonBody(res, 500, { error: { message: err instanceof Error ? err.message : "internal error", type: "api_error", param: null, code: "internal" } });
    }
  }
}

function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let size = 0;
    req.on("data", (chunk: Buffer) => {
      size += chunk.length;
      if (size <= 2 * 1024 * 1024) chunks.push(chunk);
    });
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    req.on("error", reject);
  });
}

async function main(): Promise<void> {
  const listOnly = process.argv.slice(2).includes("--list");
  const promptFile = process.env.MAKEFUN_PROMPT_FILE;
  const envUrl = (process.env.MAKEFUN_ENV_URL ?? "").replace(/\/$/, "");
  let brief: Brief | null = null;
  if (promptFile !== undefined && promptFile !== "") {
    try {
      brief = parsePrompt(readFileSync(promptFile, "utf8"), envUrl);
    } catch {
      brief = null;
    }
  }
  if (brief === null || envUrl === "") {
    process.stdout.write(`${JSON.stringify({ providers: [] })}\n`);
    return;
  }
  if (listOnly) {
    const catalog = await buildCatalog(brief);
    process.stdout.write(`${JSON.stringify(catalog)}\n`);
    return;
  }
  // The catalog is probed before the server listens, so the register
  // below carries the same document --list would print.
  const catalog = await buildCatalog(brief);
  const aggregator = new Aggregator(brief, envUrl, catalog);
  const server = createServer((req, res) => {
    void aggregator.handle(req, res);
  });
  await new Promise<void>((resolve) => server.listen(0, "0.0.0.0", resolve));
  const address = server.address() as AddressInfo | null;
  if (address === null || typeof address === "string") throw new Error("the server has no TCP address");
  const register = await fetch(`${envUrl}/register`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ protocol_version: 1, port: address.port, catalog }),
    signal: AbortSignal.timeout(5000),
  });
  if (!register.ok) {
    process.stderr.write(`register failed: HTTP ${register.status}\n`);
    process.exit(1);
  }
  for (;;) {
    await sleep(100);
    try {
      const work = await fetch(`${envUrl}/work`, { signal: AbortSignal.timeout(2000) });
      if (work.ok) {
        const state: unknown = await work.json();
        if (typeof state === "object" && state !== null && (state as { done?: unknown }).done === true) break;
      }
    } catch {
      // The env is local; a missed poll is retried on the next tick.
    }
  }
  server.close();
  process.exit(0);
}

void main().catch((err: unknown) => {
  process.stderr.write(`${err instanceof Error ? err.message : String(err)}\n`);
  process.exit(1);
});
