# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A chatbot web app comparing two Claude integration approaches side-by-side, with three complementary tracing mechanisms for full observability into Agent SDK behavior:
- **Agent SDK** (`@anthropic-ai/claude-agent-sdk`) — uses `query()` with built-in session management, tool orchestration, and resume
- **Messages API** (`@anthropic-ai/sdk`) — raw `anthropic.messages.stream()` with a hand-rolled agent loop and custom tools

Both modes share the same UI (toggle in sidebar) but use separate conversation stores and logging.

## Commands

```bash
npm install        # Install dependencies
npm start          # Start server on http://localhost:3000
```

No build step, no tests, no linting. Single entry point: `node server.js`.

## Architecture

```
server.js              Express server with two chat handlers
├── /proxy/:id/*       HTTP proxy intercepting Agent SDK → Anthropic API traffic
├── handleAgentSDK()   Uses query() from Agent SDK, resumes sessions via session_id
├── handleMessagesAPI() Raw Messages API with agent loop (max 10 turns), custom tool execution
├── generateRichTrace() Merges OTel spans + JSONL logs + proxy logs into .trace.txt
├── parseSSEResponse()  Reassembles streamed SSE chunks into structured response objects
└── appendLog()        Writes pretty-printed JSON to logs/<conversationId>.jsonl

public/index.html      Single-file frontend (HTML + CSS + JS, no framework)
├── Mode toggle        Switches between "Agent SDK" and "Messages API"
├── Separate stores    localStorage keys per mode for conversation history
└── File handling      Drag-and-drop/paste with text vs base64 detection

logs/                  Per-conversation directories
├── <id>.jsonl         SDK message objects / Messages API request-response pairs
├── <id>.trace.txt     Rich merged trace (generated after Agent SDK conversations)
└── <id>.errors.txt    Context & API error report (only generated when issues detected)
.env                   ANTHROPIC_API_KEY (required)
```

## Tracing (Agent SDK mode)

Three complementary tracing mechanisms capture different layers of data:

1. **HTTP Proxy** (`/proxy/:conversationId/*`) — intercepts raw API traffic between the SDK and Anthropic. Captures exact requests (model, system prompt, messages, tools) and responses (SSE reassembled into content blocks, usage, stop reason). Set via `ANTHROPIC_BASE_URL` env var passed to `query()`.

2. **JSONL Logs** (`appendLog()` + SDK hooks) — logs every message object yielded by `query()` (init, assistant, user, result). Hooks on `PreToolUse`, `PostToolUse`, and `SessionEnd` capture transcript paths. Tracks session lifecycle, full assistant content, tool results, cost, duration, and token totals.

3. **OpenTelemetry via Jaeger** — the SDK emits OTel spans (enabled by `ENABLE_BETA_TRACING_DETAILED=1`) to an OTLP collector. `generateRichTrace()` queries Jaeger's API for spans like `claude_code.llm_request`, `claude_code.tool`, and `claude_code.tool.blocked_on_user`. Captures timing, span hierarchy, and performance data.

After each Agent SDK conversation, `generateRichTrace()` merges all three into a `.trace.txt` file.

## Error Detection & Reporting

The proxy analyzes every API call for context and error issues. When problems are detected, two things happen:

1. **Banner in `.trace.txt`** — a `!!!! CONTEXT HEALTH: CRITICAL !!!!` banner appears at the top of the trace with a pointer to the error report.
2. **`.errors.txt` generated** — a separate human-readable error report with root cause analysis, trace line references, and fix suggestions.

Each API call in the trace also shows a **context utilization bar** (`[##..................] 7.8%`) and inline warnings at 75%/90% thresholds.

### Error classifications

**Critical (session-breaking):**
| Type | Trigger |
|---|---|
| `request_too_large_attachment` | Binary file (PDF/image) exceeds API's ~10MB HTTP body limit |
| `request_too_large` | Text-only request exceeds HTTP body limit |
| `context_overflow` | Token count exceeds model's context window |
| `context_near_overflow` | Input token usage > 90% of context window |
| `api_overloaded` | Anthropic servers at capacity (transient) |

**Warnings:**
| Type | Trigger |
|---|---|
| `context_high_utilization` | Input token usage > 75% of context window |
| `output_truncated` | Response cut off (`stop_reason: max_tokens`) |
| `rapid_context_growth` | > 50K token growth across API calls in one session |
| `rate_limit` | 429 / rate limit error |

**API Errors:**
| Type | Trigger |
|---|---|
| `invalid_image` / `invalid_document` | Malformed or unsupported file content |
| `invalid_tool` | Tool definition error |
| `content_filtered` | Content policy violation |
| `auth_error` | Invalid API key |
| `server_error` | Anthropic 500 errors |

Binary attachments (PDFs, images) are detected and sized in the trace context breakdown — the report distinguishes between "file too large to transmit" (`request_too_large_attachment`) vs "too many tokens" (`context_overflow`), with different root cause explanations and fix suggestions for each.

## Key Details

- **ESM project** — `"type": "module"` in package.json, use `import` not `require`
- **File handling differs by mode**: Agent SDK gets files as `<file>` tags in the prompt (or native content blocks for images/PDFs); Messages API gets images/PDFs as native content blocks and text files via a `read_file` tool
- **Session persistence**: Agent SDK sessions resume via `options.resume` with stored session IDs; Messages API conversation history is in-memory only (lost on restart)
- **Logging**: Both modes log to `logs/` — Agent SDK logs raw SDK message objects; Messages API logs request params, full API responses, and tool executions
- **Proxy routing**: Agent SDK calls are routed through the local proxy via `ANTHROPIC_BASE_URL` env override in `query()` options
- **OTel config**: Requires Jaeger on `localhost:16686` (or `JAEGER_URL`) and OTLP collector on `localhost:4318` (or `BETA_TRACING_ENDPOINT`)
- **`.gitignore`** excludes `node_modules/`, `logs/`, and `.env`
- **100MB JSON body limit** on Express for file uploads (large enough to proxy oversized files so the error is captured in traces rather than rejected by Express)
