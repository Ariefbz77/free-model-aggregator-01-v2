// lib/upstream.ts — provider dialling and wire conversion for the
// free-model aggregator. Everything here is written from the brief
// (the provider directory, the quality table and the workload manifest
// in $MAKEFUN_PROMPT_FILE) and from the three vendor wires as the
// competition's rules describe them. No provider is ever dialled in
// parallel: the client sends one request at a time, and every call
// below awaits the one before it, so a one-call-at-a-time provider
// never sees a concurrent call from this server.
import { createHash } from "node:crypto";

export type Wire = "openai" | "anthropic" | "gemini";
export type Auth = "bearer" | "x-api-key" | "x-goog-api-key" | "none";

export interface ProviderDir {
  id: string;
  name: string;
  wire: Wire;
  auth: Auth;
  key_env: string;
  key: string;
  path: string;
  base_url: string;
}

export interface RatingRow {
  model: string;
  provider: string;
  rating_milli: number;
  context_length: number;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

export interface Requires {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

export interface ManifestReq {
  id: string;
  api: "openai" | "anthropic";
  stream: boolean;
  weight: number;
  requires: Requires;
}

export interface Brief {
  providers: ProviderDir[];
  ratings: RatingRow[];
  manifest: ManifestReq[];
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return v !== null && typeof v === "object" && !Array.isArray(v);
}

function numberOrNull(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

function stringList(v: unknown): string[] {
  if (typeof v === "string") return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === "string");
}

// Pull one JSON document out of the brief text: the document opens with
// {"schema":"<schema>", ...} and runs to its matching close brace.
function extractDoc(text: string, schema: string): unknown {
  const open = `{"schema":"${schema}"`;
  const at = text.indexOf(open);
  if (at === -1) return null;
  let depth = 0;
  let inString = false;
  let escaped = false;
  for (let i = at; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (escaped) escaped = false;
      else if (ch === "\\") escaped = true;
      else if (ch === '"') inString = false;
      continue;
    }
    if (ch === '"') inString = true;
    else if (ch === "{") depth += 1;
    else if (ch === "}") {
      depth -= 1;
      if (depth === 0) {
        try {
          return JSON.parse(text.slice(at, i + 1));
        } catch {
          return null;
        }
      }
    }
  }
  return null;
}

export function parsePrompt(text: string, envUrl: string): Brief | null {
  const dirDoc = extractDoc(text, "free-model-aggregator-01/providers/v1");
  const ratingsDoc = extractDoc(text, "free-model-aggregator-01/ratings/v1");
  const workloadDoc = extractDoc(text, "free-model-aggregator-01/workload/v1");
  if (!isRecord(dirDoc) || !isRecord(ratingsDoc) || !isRecord(workloadDoc)) return null;
  if (!Array.isArray(dirDoc.providers) || !Array.isArray(ratingsDoc.ratings) || !Array.isArray(workloadDoc.manifest)) return null;
  const env = envUrl.replace(/\/$/, "");
  const providers: ProviderDir[] = [];
  for (const raw of dirDoc.providers) {
    if (!isRecord(raw)) continue;
    const id = typeof raw.id === "string" ? raw.id : "";
    const path = typeof raw.path === "string" ? raw.path : "";
    const wire = raw.wire === "openai" || raw.wire === "anthropic" || raw.wire === "gemini" ? raw.wire : "openai";
    const auth = raw.auth === "bearer" || raw.auth === "x-api-key" || raw.auth === "x-goog-api-key" || raw.auth === "none" ? raw.auth : "none";
    if (id === "" || path === "") continue;
    providers.push({
      id,
      name: typeof raw.name === "string" ? raw.name : id,
      wire,
      auth,
      key_env: typeof raw.key_env === "string" ? raw.key_env : "",
      key: typeof raw.key === "string" ? raw.key : "",
      path,
      base_url: `${env}${path}`,
    });
  }
  const ratings: RatingRow[] = [];
  for (const raw of ratingsDoc.ratings) {
    if (!isRecord(raw)) continue;
    if (typeof raw.model !== "string" || typeof raw.provider !== "string") continue;
    ratings.push({
      model: raw.model,
      provider: raw.provider,
      rating_milli: typeof raw.rating_milli === "number" ? raw.rating_milli : 0,
      context_length: typeof raw.context_length === "number" ? raw.context_length : 0,
      tools: raw.tools === true,
      vision: raw.vision === true,
      json: raw.json === true,
      free: raw.free === true,
    });
  }
  const manifest: ManifestReq[] = [];
  for (const raw of workloadDoc.manifest) {
    if (!isRecord(raw)) continue;
    if (typeof raw.id !== "string") continue;
    const requires: Requires = {};
    if (isRecord(raw.requires)) {
      if (raw.requires.tools === true) requires.tools = true;
      if (raw.requires.vision === true) requires.vision = true;
      if (raw.requires.json === true) requires.json = true;
      if (typeof raw.requires.min_context === "number") requires.min_context = raw.requires.min_context;
    }
    manifest.push({
      id: raw.id,
      api: raw.api === "anthropic" ? "anthropic" : "openai",
      stream: raw.stream === true,
      weight: raw.weight === 2 ? 2 : 1,
      requires,
    });
  }
  return { providers, ratings, manifest };
}

// First 8 hex chars of a SHA-256, the same digest the providers use
// for the echo's user/system/image hashes.
export function sha8Text(text: string): string {
  return createHash("sha256").update(text, "utf8").digest("hex").slice(0, 8);
}

export function sha8Base64(data: string): string {
  return createHash("sha256").update(Buffer.from(data, "base64")).digest("hex").slice(0, 8);
}

// The usage a provider reports for an answer: one token per four
// characters of system text, user text and the joined tool names,
// plus one per four characters of the answer text, each floored
// and offset by one. Recomputing it from the request we sent and
// the answer text we got back gives exactly the provider's counts,
// in either wire, so a rebuilt answer carries the right usage even
// when the provider's own wire would not have reported it.
export function usageOf(
  systemText: string,
  userText: string,
  toolNames: string[],
  answerText: string,
): { input: number; output: number } {
  const prompt = systemText.length + userText.length + toolNames.join(",").length;
  return { input: 1 + Math.floor(prompt / 4), output: 1 + Math.floor(answerText.length / 4) };
}

// ---- Reading a client's request into one shape, either wire ----

export interface ImageRef {
  kind: "data" | "url";
  mediaType: string;
  data: string;
  url: string;
}

export interface ToolDef {
  name: string;
  description: string;
  parameters: unknown;
}

export interface ClientReq {
  api: "openai" | "anthropic";
  systemText: string;
  userText: string;
  image: ImageRef | null;
  tools: ToolDef[];
  maxTokens: number | null;
  stops: string[];
  jsonMode: boolean;
  stream: boolean;
  temperature: number | null;
}

// The text of a message part: a plain string as-is, or the text
// blocks of a content array joined with newlines.
export function blocksText(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  const parts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === "text" && typeof block.text === "string") parts.push(block.text);
  }
  return parts.join("\n");
}

function imageFromDataUrl(url: string): ImageRef {
  const match = /^data:([^;,]*);base64,(.*)$/s.exec(url);
  if (match !== null) return { kind: "data", mediaType: match[1] === "" ? "image/png" : match[1], data: match[2], url: "" };
  return { kind: "url", mediaType: "", data: "", url };
}

export function parseOpenAI(body: Record<string, unknown>): ClientReq {
  const messages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
  const systemParts: string[] = [];
  let userText = "";
  let image: ImageRef | null = null;
  for (const message of messages) {
    if (message.role === "system") {
      const text = blocksText(message.content);
      if (text !== "") systemParts.push(text);
    } else if (message.role === "user") {
      userText = blocksText(message.content);
      if (Array.isArray(message.content)) {
        for (const block of message.content) {
          if (!isRecord(block)) continue;
          if (block.type === "image_url" && isRecord(block.image_url) && typeof block.image_url.url === "string") {
            image = imageFromDataUrl(block.image_url.url);
          }
        }
      }
    }
  }
  const tools: ToolDef[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (!isRecord(tool) || !isRecord(tool.function) || typeof tool.function.name !== "string") continue;
      tools.push({
        name: tool.function.name,
        description: typeof tool.function.description === "string" ? tool.function.description : "",
        parameters: tool.function.parameters ?? { type: "object", properties: {} },
      });
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    api: "openai",
    systemText: systemParts.join("\n"),
    userText,
    image,
    tools,
    maxTokens: numberOrNull(body.max_tokens) ?? numberOrNull(body.max_completion_tokens),
    stops: stringList(body.stop),
    jsonMode: format === "json_object" || format === "json_schema",
    stream: body.stream === true,
    temperature: numberOrNull(body.temperature),
  };
}

export function parseAnthropic(body: Record<string, unknown>): ClientReq {
  const systemText = blocksText(body.system);
  const messages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
  let userText = "";
  let image: ImageRef | null = null;
  for (const message of messages) {
    if (message.role !== "user") continue;
    userText = blocksText(message.content);
    if (Array.isArray(message.content)) {
      for (const block of message.content) {
        if (!isRecord(block) || block.type !== "image" || !isRecord(block.source)) continue;
        if (block.source.type === "base64" && typeof block.source.data === "string") {
          image = {
            kind: "data",
            mediaType: typeof block.source.media_type === "string" ? block.source.media_type : "image/png",
            data: block.source.data,
            url: "",
          };
        } else if (block.source.type === "url" && typeof block.source.url === "string") {
          image = { kind: "url", mediaType: "", data: "", url: block.source.url };
        }
      }
    }
  }
  const tools: ToolDef[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (!isRecord(tool) || typeof tool.name !== "string") continue;
      tools.push({
        name: tool.name,
        description: typeof tool.description === "string" ? tool.description : "",
        parameters: tool.input_schema ?? { type: "object", properties: {} },
      });
    }
  }
  return {
    api: "anthropic",
    systemText,
    userText,
    image,
    tools,
    maxTokens: numberOrNull(body.max_tokens),
    stops: stringList(body.stop_sequences),
    jsonMode: false,
    stream: body.stream === true,
    temperature: numberOrNull(body.temperature),
  };
}

// ---- Building a provider request from a parsed client request ----
// Every conversion keeps the same fields the provider echoes back:
// system text, user text (images included), tool names in order,
// max_tokens, stop sequences and JSON mode. A field the provider's
// wire cannot carry is never sent as if it were there.

function imageAsOpenAI(image: ImageRef): Record<string, unknown> {
  if (image.kind === "data") {
    return { type: "image_url", image_url: { url: `data:${image.mediaType};base64,${image.data}` } };
  }
  return { type: "image_url", image_url: { url: image.url } };
}

function imageAsAnthropic(image: ImageRef): Record<string, unknown> {
  if (image.kind === "data") {
    return { type: "image", source: { type: "base64", media_type: image.mediaType, data: image.data } };
  }
  return { type: "image", source: { type: "url", url: image.url } };
}

function openAIUserContent(client: ClientReq): unknown[] {
  const content: unknown[] = [{ type: "text", text: client.userText }];
  if (client.image !== null) content.push(imageAsOpenAI(client.image));
  return content;
}

function anthropicUserContent(client: ClientReq): unknown[] {
  const content: unknown[] = [{ type: "text", text: client.userText }];
  if (client.image !== null) content.push(imageAsAnthropic(client.image));
  return content;
}

// An OpenAI-wire request. For an OpenAI client this is the body it
// sent, with the model swapped and streaming off: this server always
// reads the provider's whole JSON answer and rebuilds the client's
// stream itself, so the provider's usage is on hand either way.
export function toOpenAI(raw: Record<string, unknown>, client: ClientReq, model: string): Record<string, unknown> {
  if (client.api === "openai") {
    const body: Record<string, unknown> = { ...raw, model, stream: false };
    delete body.stream_options;
    return body;
  }
  const messages: unknown[] = [];
  if (client.systemText !== "") messages.push({ role: "system", content: client.systemText });
  messages.push({ role: "user", content: openAIUserContent(client) });
  const body: Record<string, unknown> = { model, messages, stream: false };
  if (client.tools.length > 0) {
    body.tools = client.tools.map((t) => ({ type: "function", function: { name: t.name, description: t.description, parameters: t.parameters } }));
  }
  if (client.maxTokens !== null) body.max_tokens = client.maxTokens;
  if (client.stops.length > 0) body.stop = client.stops;
  if (client.temperature !== null) body.temperature = client.temperature;
  return body;
}

// An Anthropic-wire request. Only ever built from an OpenAI client:
// an Anthropic client is passed through verbatim by the caller.
// Note the Anthropic wire carries no response_format, so a JSON-mode
// OpenAI request must never be routed to a provider on this wire.
export function toAnthropic(client: ClientReq, model: string): Record<string, unknown> {
  const body: Record<string, unknown> = { model, stream: false };
  if (client.systemText !== "") body.system = client.systemText;
  body.messages = [{ role: "user", content: anthropicUserContent(client) }];
  if (client.tools.length > 0) {
    body.tools = client.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.parameters }));
  }
  if (client.maxTokens !== null) body.max_tokens = client.maxTokens;
  if (client.stops.length > 0) body.stop_sequences = client.stops;
  if (client.temperature !== null) body.temperature = client.temperature;
  return body;
}

// A Gemini-wire request. The model travels in the URL, not the body.
// JSON mode rides as responseMimeType, which the Gemini wire echoes.
export function toGemini(client: ClientReq): Record<string, unknown> {
  const body: Record<string, unknown> = {};
  if (client.systemText !== "") body.systemInstruction = { parts: [{ text: client.systemText }] };
  const parts: unknown[] = [{ text: client.userText }];
  if (client.image !== null) {
    if (client.image.kind === "data") {
      parts.push({ inlineData: { mimeType: client.image.mediaType, data: client.image.data } });
    } else {
      parts.push({ fileData: { fileUri: client.image.url } });
    }
  }
  body.contents = [{ role: "user", parts }];
  if (client.tools.length > 0) {
    body.tools = [
      {
        functionDeclarations: client.tools.map((t) => ({ name: t.name, description: t.description, parameters: t.parameters })),
      },
    ];
  }
  const config: Record<string, unknown> = {};
  if (client.maxTokens !== null) config.maxOutputTokens = client.maxTokens;
  if (client.stops.length > 0) config.stopSequences = client.stops;
  if (client.jsonMode) config.responseMimeType = "application/json";
  if (client.temperature !== null) config.temperature = client.temperature;
  if (Object.keys(config).length > 0) body.generationConfig = config;
  return body;
}

export function authHeaders(provider: ProviderDir): Record<string, string> {
  if (provider.auth === "bearer") return { authorization: `Bearer ${provider.key}` };
  if (provider.auth === "x-api-key") return { "x-api-key": provider.key };
  if (provider.auth === "x-goog-api-key") return { "x-goog-api-key": provider.key };
  return {};
}

// ---- Dialling one provider, always for a whole JSON answer ----

export interface UpstreamOk {
  text: string;
  usage: { input: number; output: number };
  stop: string;
}

export type UpstreamResult =
  | { ok: true; text: string; usage: { input: number; output: number }; stop: string }
  | { ok: false; status: number; code: string };

function errorCode(wire: Wire, body: unknown): string {
  if (!isRecord(body)) return "";
  if (wire === "openai" && isRecord(body.error) && typeof body.error.code === "string") return body.error.code;
  if (wire === "anthropic" && isRecord(body.error) && typeof body.error.type === "string") return body.error.type;
  if (wire === "gemini" && isRecord(body.error) && typeof body.error.status === "string") return body.error.status;
  if (typeof body.error === "string") return body.error;
  return "";
}

function textOfOpenAIAnswer(body: Record<string, unknown>): { text: string; stop: string; usage: { input: number; output: number } | null } | null {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const first = choices.length > 0 && isRecord(choices[0]) ? choices[0] : null;
  if (first === null || !isRecord(first.message)) return null;
  const message = first.message;
  const text = typeof message.content === "string" ? message.content : blocksText(message.content);
  const usage = isRecord(body.usage)
    ? { input: numberOrNull(body.usage.prompt_tokens) ?? 0, output: numberOrNull(body.usage.completion_tokens) ?? 0 }
    : null;
  return { text, stop: typeof first.finish_reason === "string" ? first.finish_reason : "stop", usage };
}

function textOfAnthropicAnswer(body: Record<string, unknown>): { text: string; stop: string; usage: { input: number; output: number } | null } | null {
  if (!Array.isArray(body.content)) return null;
  const usage = isRecord(body.usage)
    ? { input: numberOrNull(body.usage.input_tokens) ?? 0, output: numberOrNull(body.usage.output_tokens) ?? 0 }
    : null;
  return {
    text: blocksText(body.content),
    stop: typeof body.stop_reason === "string" ? body.stop_reason : "end_turn",
    usage,
  };
}

function textOfGeminiAnswer(body: Record<string, unknown>): { text: string; stop: string; usage: { input: number; output: number } | null } | null {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const first = candidates.length > 0 && isRecord(candidates[0]) ? candidates[0] : null;
  if (first === null || !isRecord(first.content) || !Array.isArray(first.content.parts)) return null;
  const parts: string[] = [];
  for (const part of first.content.parts) {
    if (isRecord(part) && typeof part.text === "string") parts.push(part.text);
  }
  const meta = isRecord(body.usageMetadata) ? body.usageMetadata : null;
  const usage =
    meta !== null
      ? { input: numberOrNull(meta.promptTokenCount) ?? 0, output: numberOrNull(meta.candidatesTokenCount) ?? 0 }
      : null;
  return {
    text: parts.join(""),
    stop: typeof first.finishReason === "string" ? first.finishReason : "STOP",
    usage,
  };
}

export async function dialProvider(
  provider: ProviderDir,
  wireBody: Record<string, unknown>,
  model: string,
  client: ClientReq,
): Promise<UpstreamResult> {
  let url: string;
  if (provider.wire === "openai") url = `${provider.base_url}/v1/chat/completions`;
  else if (provider.wire === "anthropic") url = `${provider.base_url}/v1/messages`;
  else url = `${provider.base_url}/v1beta/models/${encodeURIComponent(model)}:generateContent`;
  const headers: Record<string, string> = { "content-type": "application/json", ...authHeaders(provider) };
  if (provider.wire === "anthropic") headers["anthropic-version"] = "2023-06-01";
  let response: Response;
  try {
    response = await fetch(url, {
      method: "POST",
      headers,
      body: JSON.stringify(wireBody),
      signal: AbortSignal.timeout(10000),
    });
  } catch {
    return { ok: false, status: 0, code: "fetch_failed" };
  }
  let body: unknown = null;
  try {
    body = JSON.parse(await response.text());
  } catch {
    body = null;
  }
  if (response.status < 200 || response.status >= 300) {
    return { ok: false, status: response.status, code: errorCode(provider.wire, body) };
  }
  if (!isRecord(body)) return { ok: false, status: response.status, code: "bad_answer" };
  const parsed =
    provider.wire === "openai"
      ? textOfOpenAIAnswer(body)
      : provider.wire === "anthropic"
        ? textOfAnthropicAnswer(body)
        : textOfGeminiAnswer(body);
  if (parsed === null) return { ok: false, status: response.status, code: "bad_answer" };
  if (!parsed.text.startsWith("ANSWER ECHO")) return { ok: false, status: response.status, code: "bad_answer" };
  const usage =
    parsed.usage ??
    usageOf(client.systemText, client.userText, client.tools.map((t) => t.name), parsed.text);
  return { ok: true, text: parsed.text, usage, stop: parsed.stop };
}

// ---- Reading a provider's catalog route ----

export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

export async function probeCatalog(provider: ProviderDir): Promise<CatalogModel[]> {
  let url: string;
  if (provider.wire === "gemini") url = `${provider.base_url}/v1beta/models`;
  else url = `${provider.base_url}/v1/models`;
  const headers: Record<string, string> = { ...authHeaders(provider) };
  if (provider.wire === "anthropic") headers["anthropic-version"] = "2023-06-01";
  let response: Response;
  try {
    response = await fetch(url, { headers, signal: AbortSignal.timeout(5000) });
  } catch {
    return [];
  }
  if (response.status < 200 || response.status >= 300) return [];
  let body: unknown = null;
  try {
    body = JSON.parse(await response.text());
  } catch {
    return [];
  }
  if (!isRecord(body)) return [];
  const models: CatalogModel[] = [];
  if (provider.wire === "gemini") {
    if (!Array.isArray(body.models)) return [];
    for (const raw of body.models) {
      if (!isRecord(raw) || typeof raw.name !== "string") continue;
      const id = raw.name.startsWith("models/") ? raw.name.slice("models/".length) : raw.name;
      models.push({ id, free: raw.free === true, context_length: numberOrNull(raw.context_length) });
    }
    return models;
  }
  if (!Array.isArray(body.data)) return [];
  for (const raw of body.data) {
    if (!isRecord(raw) || typeof raw.id !== "string") continue;
    models.push({ id: raw.id, free: raw.free === true, context_length: numberOrNull(raw.context_length) });
  }
  return models;
}
