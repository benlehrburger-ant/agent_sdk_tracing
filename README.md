# Agent SDK Tracing

A chatbot web app that compares two Claude integration approaches side-by-side (**Agent SDK** vs **Messages API**) while capturing detailed tracing data through three complementary mechanisms.

## Setup

```bash
npm install
npm start          # http://localhost:3000
```

Requires a `.env` file with `ANTHROPIC_API_KEY`.

For OTel tracing, run a Jaeger instance on `localhost:16686` (or set `JAEGER_URL`) with an OTLP collector on `localhost:4318` (or set `BETA_TRACING_ENDPOINT`).

## Tracing Approaches

### 1. HTTP Proxy

The Express server acts as a man-in-the-middle between the Agent SDK and the Anthropic API. Instead of the SDK calling `api.anthropic.com` directly, it calls `localhost:3000/proxy/...`, which forwards the request and captures everything.

**What it tracks:**

- The exact HTTP request the SDK sends — model, system prompt, messages, tools, streaming config
- The exact HTTP response — the full SSE stream reassembled into content blocks (text, thinking, tool_use), token usage, stop reason
- Context window growth — how many messages are in each request, their sizes, estimated token counts

**In short:** A wiretap on the network calls. You see exactly what goes over the wire.

### 2. Agent SDK JSONL Logs

The SDK's `query()` function yields a stream of structured message objects (init, assistant, user, result, etc.). Every one of these is logged to `logs/<conversationId>/<conversationId>.jsonl`. Hooks (`PreToolUse`, `PostToolUse`, `SessionEnd`) also capture the transcript path.

**What it tracks:**

- Session lifecycle — init (session ID, model), result (duration, cost, token totals, turns)
- Assistant messages — full text and tool_use blocks with complete content (not truncated)
- Tool results — what the tools returned back to the model
- High-level stats — cost in USD, total duration, API duration, per-model usage breakdown, permission denials

**In short:** The SDK's own view of the conversation — what the agent said, what tools it called, how much it cost. Application-level events, not raw HTTP.

### 3. OpenTelemetry (OTel) via Jaeger

The SDK emits OTel trace spans (enabled by `ENABLE_BETA_TRACING_DETAILED=1`) to a collector at `BETA_TRACING_ENDPOINT`. The `generateRichTrace()` function queries Jaeger's API to pull these spans back out.

**What it tracks:**

- `claude_code.llm_request` spans — each LLM round-trip with timing, token counts, cache stats, system prompt hash, tool count, truncated input/output
- `claude_code.tool` spans — each tool execution with duration and parent/child relationships
- `claude_code.tool.blocked_on_user` spans — permission checks (approved/denied) and how long the user took to respond
- Span hierarchy — parent-child relationships showing causal flow (e.g., which tool call belongs to which LLM request)

**In short:** A timing/performance trace. Like a flame graph — it shows *when* things happened, *how long* they took, and *how they're nested*. Standard observability format compatible with Datadog, Honeycomb, etc.

## Rich Trace

After each Agent SDK conversation, `generateRichTrace()` merges all three sources into a single `<conversationId>.trace.txt` file:

- **OTel spans** for structure and timing
- **JSONL logs** for full content (OTel truncates long outputs)
- **Proxy logs** for raw API call details

## Production Readiness

As implemented, all three tracing approaches are local development tools. Here's what would need to change for production use:

| Approach | Production-ready today? | What it would need |
|---|---|---|
| HTTP Proxy | No | Single point of failure, in-memory SSE buffering, localhost routing |
| JSONL Logs | No (transport), yes (data) | Sync file I/O blocks the event loop, local disk is ephemeral on containers |
| OTel | Yes (protocol), no (wiring) | OTel is production-grade, but pulling spans back from Jaeger into the app is a dev pattern |

### HTTP Proxy

Replace the in-process proxy with infrastructure-level interception — an API gateway (Envoy, nginx, AWS API Gateway) that logs natively, or use the Anthropic SDK's `fetch` override to inject logging middleware at the client level. Removes the single point of failure and unbounded memory buffering.

### JSONL Logs

Swap the transport, keep the data. Replace `appendFileSync` with async writes to a durable store (Postgres, BigQuery, S3, or a logging pipeline like Kafka). The SDK's message stream and hooks are already the right abstraction — this is mostly a plumbing change and the smallest lift of the three.

### OTel

Point `BETA_TRACING_ENDPOINT` at a production collector (Datadog Agent, Honeycomb, Grafana Alloy, AWS ADOT) instead of a local Jaeger. Delete `generateRichTrace()` — view spans in your observability platform's UI instead of pulling them back into the app. The open question is whether `ENABLE_BETA_TRACING_DETAILED` is stable enough for production since it's still flagged as beta.

### Recommended Production Setup

OTel + a production-grade version of the JSONL logging. OTel covers timing, span hierarchy, and token usage. JSONL log data fills the gap OTel doesn't — full conversation content and billing. The proxy becomes redundant since it's a lower-level view of the same API calls the JSONL logs already capture at the application level.

## Architecture

```
server.js              Express server with two chat handlers
├── /proxy/:id/*       HTTP proxy intercepting Agent SDK → Anthropic API calls
├── handleAgentSDK()   Uses query() from Agent SDK, resumes sessions via session_id
├── handleMessagesAPI() Raw Messages API with agent loop (max 10 turns)
├── generateRichTrace() Merges OTel + JSONL + proxy into .trace.txt
└── appendLog()        Writes JSON to logs/<conversationId>.jsonl

public/index.html      Single-file frontend (HTML + CSS + JS)

logs/                  Per-conversation directories with .jsonl and .trace.txt files
```
