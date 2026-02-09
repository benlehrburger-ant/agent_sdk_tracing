import "dotenv/config";
import express from "express";
import Anthropic from "@anthropic-ai/sdk";
import { query } from "@anthropic-ai/claude-agent-sdk";
import { fileURLToPath } from "url";
import { dirname, join } from "path";
import { appendFileSync, mkdirSync, readFileSync, writeFileSync } from "fs";

const anthropic = new Anthropic();

const __dirname = dirname(fileURLToPath(import.meta.url));
const app = express();
const PORT = 3000;

app.use(express.json({ limit: "20mb" }));
app.use(express.static(join(__dirname, "public")));

const logsDir = join(__dirname, "logs");
mkdirSync(logsDir, { recursive: true });

function appendLog(conversationId, msg) {
  const convDir = join(logsDir, conversationId);
  mkdirSync(convDir, { recursive: true });
  const logFile = join(convDir, `${conversationId}.jsonl`);
  appendFileSync(logFile, JSON.stringify(msg, null, 2) + "\n---\n");
}

// Store session IDs so we can resume conversations
const sessions = new Map();

// Store Messages API conversation histories
const messageHistories = new Map();

// Store dropped files per Messages API conversation turn
const turnFiles = new Map();

// Store transcript paths captured from Agent SDK hooks
const transcriptPaths = new Map();

// Store intercepted API calls per conversation (from HTTP proxy)
const proxyLogs = new Map(); // conversationId -> [{request, response, timestamp}]

// ---------------------------------------------------------------------------
// HTTP Proxy: intercepts Agent SDK calls to Anthropic API for full logging
// ---------------------------------------------------------------------------

app.all('/proxy/:conversationId/*', async (req, res) => {
  const { conversationId } = req.params;
  // Strip /proxy/<conversationId> prefix to get the real API path
  const apiPath = req.originalUrl.replace(`/proxy/${conversationId}`, '');
  const targetUrl = `https://api.anthropic.com${apiPath}`;

  // Initialize log array for this conversation if needed
  if (!proxyLogs.has(conversationId)) {
    proxyLogs.set(conversationId, []);
  }

  const logEntry = {
    timestamp: new Date().toISOString(),
    method: req.method,
    path: apiPath,
    request: null,
    response: null,
  };

  // Log the request body (for POST/PUT/PATCH)
  if (req.body && Object.keys(req.body).length > 0) {
    logEntry.request = {
      model: req.body.model,
      max_tokens: req.body.max_tokens,
      stream: req.body.stream,
      system: req.body.system,
      thinking: req.body.thinking,
      tools: req.body.tools
        ? { count: req.body.tools.length, names: req.body.tools.map(t => t.name) }
        : undefined,
      messages: req.body.messages
        ? req.body.messages.map(m => {
            const blocks = Array.isArray(m.content) ? m.content : [{ type: 'text', text: m.content }];
            return {
              role: m.role,
              block_count: blocks.length,
              types: blocks.map(b => b.type),
              char_length: blocks.reduce((sum, b) => {
                if (b.type === 'text') return sum + (b.text?.length || 0);
                if (b.type === 'tool_use') return sum + JSON.stringify(b.input || {}).length;
                if (b.type === 'tool_result') return sum + (typeof b.content === 'string' ? b.content.length : JSON.stringify(b.content || '').length);
                if (b.type === 'thinking') return sum + (b.thinking?.length || 0);
                return sum;
              }, 0),
            };
          })
        : undefined,
    };
    // Store full system prompt text separately for the trace
    logEntry.request.system_full = req.body.system;
    logEntry.request.messages_full = req.body.messages;
  }

  // Build headers for forwarding
  const forwardHeaders = {};
  for (const [key, value] of Object.entries(req.headers)) {
    const lower = key.toLowerCase();
    if (lower === 'host' || lower === 'content-length') continue;
    forwardHeaders[key] = value;
  }
  forwardHeaders['host'] = 'api.anthropic.com';

  try {
    const fetchOptions = {
      method: req.method,
      headers: forwardHeaders,
    };
    if (req.method !== 'GET' && req.method !== 'HEAD' && req.body) {
      fetchOptions.body = JSON.stringify(req.body);
    }

    const upstream = await fetch(targetUrl, fetchOptions);

    const isStreaming = upstream.headers.get('content-type')?.includes('text/event-stream');

    if (isStreaming) {
      // Forward SSE headers
      res.writeHead(upstream.status, {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
      });

      // Stream through while buffering
      const chunks = [];
      const reader = upstream.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        chunks.push(chunk);
        res.write(chunk);
      }
      res.end();

      // Parse the buffered SSE to extract the full response
      const fullSSE = chunks.join('');
      logEntry.response = parseSSEResponse(fullSSE);
    } else {
      // Non-streaming: read body, log, forward
      const body = await upstream.text();
      res.writeHead(upstream.status, {
        'Content-Type': upstream.headers.get('content-type') || 'application/json',
      });
      res.end(body);

      try {
        logEntry.response = JSON.parse(body);
      } catch {
        logEntry.response = { raw: body.slice(0, 2000) };
      }
    }
  } catch (err) {
    console.error('Proxy error:', err);
    logEntry.response = { error: err.message };
    if (!res.headersSent) {
      res.status(502).json({ error: 'Proxy error', message: err.message });
    }
  }

  proxyLogs.get(conversationId).push(logEntry);
});

/**
 * Parse buffered SSE text into a structured response object.
 * Extracts content blocks, usage, stop_reason from Anthropic's SSE format.
 */
function parseSSEResponse(sseText) {
  const result = {
    content: [],
    stop_reason: null,
    usage: null,
    model: null,
    id: null,
  };

  // Track content blocks being built incrementally
  const blockMap = new Map(); // index -> block

  for (const line of sseText.split('\n')) {
    if (!line.startsWith('data: ')) continue;
    const data = line.slice(6).trim();
    if (data === '[DONE]') continue;

    let event;
    try { event = JSON.parse(data); } catch { continue; }

    switch (event.type) {
      case 'message_start':
        if (event.message) {
          result.model = event.message.model;
          result.id = event.message.id;
          if (event.message.usage) {
            result.usage = { ...event.message.usage };
          }
        }
        break;

      case 'content_block_start':
        if (event.content_block) {
          const block = { ...event.content_block };
          if (block.type === 'thinking') block.thinking = '';
          if (block.type === 'text') block.text = '';
          if (block.type === 'tool_use') block.input_json = '';
          blockMap.set(event.index, block);
        }
        break;

      case 'content_block_delta':
        if (event.delta && blockMap.has(event.index)) {
          const block = blockMap.get(event.index);
          if (event.delta.type === 'thinking_delta') {
            block.thinking += event.delta.thinking || '';
          } else if (event.delta.type === 'text_delta') {
            block.text += event.delta.text || '';
          } else if (event.delta.type === 'input_json_delta') {
            block.input_json += event.delta.partial_json || '';
          }
        }
        break;

      case 'content_block_stop':
        if (blockMap.has(event.index)) {
          const block = blockMap.get(event.index);
          // Parse accumulated tool_use input JSON
          if (block.type === 'tool_use' && block.input_json) {
            try { block.input = JSON.parse(block.input_json); } catch { block.input = block.input_json; }
            delete block.input_json;
          }
          result.content.push(block);
        }
        break;

      case 'message_delta':
        if (event.delta) {
          if (event.delta.stop_reason) result.stop_reason = event.delta.stop_reason;
        }
        if (event.usage) {
          result.usage = { ...result.usage, ...event.usage };
        }
        break;
    }
  }

  return result;
}

app.post("/api/chat", async (req, res) => {
  const { message, conversationId, files, mode } = req.body;
  if (!message && (!files || files.length === 0)) {
    return res.status(400).json({ error: "message or files required" });
  }

  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");

  if (mode === "messages-api") {
    await handleMessagesAPI(message, files, conversationId, res);
  } else {
    // Build prompt for Agent SDK — use content blocks for binary files
    const hasBinaryFiles = files?.some((f) => f.encoding === "base64");

    if (hasBinaryFiles) {
      // Use SDKUserMessage format with native content blocks
      const content = [];
      for (const file of files || []) {
        if (file.encoding === "base64" && IMAGE_TYPES.has(file.mediaType)) {
          content.push({
            type: "image",
            source: { type: "base64", media_type: file.mediaType, data: file.content },
          });
        } else if (file.encoding === "base64" && file.mediaType === "application/pdf") {
          content.push({
            type: "document",
            source: { type: "base64", media_type: "application/pdf", data: file.content },
          });
        } else if (file.encoding === "text") {
          content.push({
            type: "text",
            text: `<file name="${file.name}">\n${file.content}\n</file>`,
          });
        }
      }
      content.push({
        type: "text",
        text: message || "Please review the attached file(s).",
      });
      await handleAgentSDK(content, conversationId, res);
    } else {
      // Plain text prompt (with text files embedded)
      let prompt = "";
      if (files && files.length > 0) {
        for (const file of files) {
          prompt += `<file name="${file.name}">\n${file.content}\n</file>\n\n`;
        }
      }
      prompt += message || "Please review the attached file(s).";
      await handleAgentSDK(prompt, conversationId, res);
    }
  }

  res.write("data: [DONE]\n\n");
  res.end();
});

// ---------------------------------------------------------------------------
// Rich trace generation: merges OTEL spans (Jaeger) + JSONL logs
// ---------------------------------------------------------------------------

const JAEGER_URL = process.env.JAEGER_URL || "http://localhost:16686";

async function generateRichTrace(conversationId, sessionId) {
  if (!sessionId) return;

  // Wait for OTEL spans to flush to Jaeger
  await new Promise((r) => setTimeout(r, 3000));

  // 1. Query Jaeger for all traces with this session ID
  const tagsParam = encodeURIComponent(JSON.stringify({ "session.id": sessionId }));
  const url = `${JAEGER_URL}/api/traces?service=claude-code&tags=${tagsParam}&limit=50`;
  const resp = await fetch(url);
  if (!resp.ok) {
    console.error("Jaeger query failed:", resp.status, await resp.text());
    return;
  }
  const jaegerData = await resp.json();

  // Collect all spans across traces, flatten
  const spans = [];
  for (const trace of jaegerData.data || []) {
    for (const span of trace.spans || []) {
      const tags = {};
      for (const t of span.tags || []) tags[t.key] = t.value;
      spans.push({
        operationName: span.operationName,
        startTime: span.startTime, // microseconds since epoch
        duration: span.duration,   // microseconds
        spanID: span.spanID,
        traceID: span.traceID,
        parentSpanID: span.references?.[0]?.spanID || null,
        tags,
      });
    }
  }

  if (spans.length === 0) {
    console.log("No OTEL spans found for session", sessionId);
    return;
  }

  // Sort by start time
  spans.sort((a, b) => a.startTime - b.startTime);

  // 2. Read JSONL log entries for this conversation
  const jsonlPath = join(logsDir, conversationId, `${conversationId}.jsonl`);
  let jsonlEntries = [];
  try {
    const raw = readFileSync(jsonlPath, "utf-8");
    jsonlEntries = raw
      .split("\n---\n")
      .filter((s) => s.trim())
      .map((s) => {
        try { return JSON.parse(s); } catch { return null; }
      })
      .filter(Boolean);
  } catch { /* no jsonl yet */ }

  // Extract result entry for summary
  const resultEntry = jsonlEntries.find(
    (e) => e.type === "result" && e.session_id === sessionId
  );

  // Extract init entry for metadata
  const initEntry = jsonlEntries.find(
    (e) => e.type === "system" && e.subtype === "init" && e.session_id === sessionId
  );

  // Extract assistant messages for full content (OTEL truncates)
  const assistantMessages = jsonlEntries.filter(
    (e) => e.type === "assistant" && e.session_id === sessionId
  );

  // Extract user messages (tool results, etc.)
  const userMessages = jsonlEntries.filter(
    (e) => e.type === "user" && e.session_id === sessionId
  );

  // 3. Build the rich trace
  const lines = [];
  const sessionShort = sessionId.slice(0, 8);
  const model = initEntry?.model || spans[0]?.tags?.model || "unknown";
  const startDate = new Date(spans[0].startTime / 1000);

  lines.push("========================================");
  lines.push(`AGENT SDK TRACE - Session ${sessionShort}`);
  lines.push(`Started: ${startDate.toLocaleString()}`);
  lines.push(`Model: ${model}`);
  lines.push(`Session ID: ${sessionId}`);
  lines.push("========================================");
  lines.push("");

  let llmRequestNum = 0;

  for (const span of spans) {
    const durationMs = span.duration / 1000;
    const time = new Date(span.startTime / 1000).toLocaleTimeString();
    const t = span.tags;

    if (span.operationName === "claude_code.llm_request") {
      llmRequestNum++;
      lines.push(`--- LLM Request #${llmRequestNum} (${(durationMs / 1000).toFixed(2)}s) @ ${time} ---`);
      lines.push("");

      // Source context
      const source = t.query_source || "sdk";
      if (source !== "sdk") {
        lines.push(`  Source: ${source}`);
      }

      // User input
      if (t.new_context) {
        const userText = t.new_context.replace(/^\[USER\]\n?/, "").trim();
        lines.push(`  USER: ${userText}`);
        lines.push("");
      }

      // Assistant output — prefer full content from JSONL, fall back to OTEL
      const matchingAssistant = assistantMessages.find((a) => {
        // Match by approximate timing or message content
        const otelOutput = t["response.model_output"] || "";
        return a.message?.content?.some(
          (b) => b.type === "text" && b.text && otelOutput.startsWith(b.text.slice(0, 50))
        );
      });

      const fullText = matchingAssistant?.message?.content
        ?.filter((b) => b.type === "text")
        .map((b) => b.text)
        .join("\n");

      const assistantText = fullText || t["response.model_output"] || "(no text output)";
      lines.push(`  ASSISTANT: ${assistantText}`);
      lines.push("");

      // Tool calls from assistant content
      if (matchingAssistant?.message?.content) {
        for (const block of matchingAssistant.message.content) {
          if (block.type === "tool_use") {
            lines.push(`  TOOL CALL: ${block.name}`);
            lines.push(`    Input: ${JSON.stringify(block.input)}`);
            lines.push("");
          }
        }
      } else if (t["response.has_tool_call"] === true || t["response.has_tool_call"] === "True") {
        lines.push("  TOOL CALL: (see tool span below)");
        lines.push("");
      }

      // Token usage
      const input = t.input_tokens || 0;
      const output = t.output_tokens || 0;
      const cached = t.cache_read_tokens || 0;
      const created = t.cache_creation_tokens || 0;
      let tokenLine = `  Tokens: ${Number(input).toLocaleString()} in / ${Number(output).toLocaleString()} out`;
      if (cached > 0) tokenLine += ` (${Number(cached).toLocaleString()} cached)`;
      if (created > 0) tokenLine += ` (${Number(created).toLocaleString()} cache created)`;
      lines.push(tokenLine);

      // System prompt info
      if (t.system_prompt_length) {
        lines.push(`  System prompt: ${t.system_prompt_length} chars (hash: ${t.system_prompt_hash || "N/A"})`);
      }

      // Tools available
      if (t.tools_count) {
        lines.push(`  Tools available: ${t.tools_count}`);
      }

      lines.push("");

    } else if (span.operationName === "claude_code.tool") {
      const toolName = t.tool_name || "unknown";
      lines.push(`--- Tool: ${toolName} (${durationMs.toFixed(2)}ms) @ ${time} ---`);
      lines.push("");

      // Find child blocked_on_user spans
      const childSpans = spans.filter((s) => s.parentSpanID === span.spanID);
      for (const child of childSpans) {
        if (child.operationName === "claude_code.tool.blocked_on_user") {
          const decision = child.tags.decision || "unknown";
          const childMs = child.duration / 1000;
          lines.push(`  Decision: ${decision.toUpperCase()} (${childMs.toFixed(2)}ms)`);
        }
      }

      // Find matching tool result from JSONL
      const toolResult = userMessages.find((u) =>
        u.message?.content?.some?.(
          (c) => c.type === "tool_result" && c.tool_use_id
        )
      );
      if (toolResult) {
        for (const c of toolResult.message.content) {
          if (c.type === "tool_result") {
            const preview = typeof c.content === "string"
              ? c.content.slice(0, 200)
              : JSON.stringify(c.content).slice(0, 200);
            lines.push(`  Result: ${c.is_error ? "ERROR - " : ""}${preview}`);
          }
        }
      }

      lines.push("");

    } else if (span.operationName === "claude_code.tool.blocked_on_user") {
      // Already handled as child of tool span above — only show if orphaned
      if (!span.parentSpanID || !spans.find((s) => s.spanID === span.parentSpanID)) {
        const decision = t.decision || "unknown";
        lines.push(`--- Permission Check (${durationMs.toFixed(2)}ms) @ ${time} ---`);
        lines.push(`  Decision: ${decision.toUpperCase()}`);
        lines.push("");
      }

    } else {
      // Other span types
      lines.push(`--- ${span.operationName} (${durationMs.toFixed(2)}ms) @ ${time} ---`);
      lines.push("");
    }
  }

  // 4. Full transcript from hook-captured transcript_path (with fallback derivation)
  let transcriptPath = transcriptPaths.get(conversationId);
  if (!transcriptPath && sessionId) {
    // Derive transcript path from session ID — Claude Code stores transcripts at:
    // ~/.claude/projects/<cwd-with-slashes-as-dashes>/<session-id>.jsonl
    const homeDir = process.env.HOME || process.env.USERPROFILE;
    const cwdSlug = process.cwd().replace(/[/ ]/g, "-");
    const derived = join(homeDir, ".claude", "projects", cwdSlug, `${sessionId}.jsonl`);
    try {
      readFileSync(derived, { flag: "r" }); // check it exists
      transcriptPath = derived;
    } catch { /* not found, skip */ }
  }
  if (transcriptPath) {
    try {
      const transcriptRaw = readFileSync(transcriptPath, "utf-8");
      const transcriptEntries = transcriptRaw
        .split("\n")
        .filter((line) => line.trim())
        .map((line) => {
          try { return JSON.parse(line); } catch { return null; }
        })
        .filter(Boolean);

      if (transcriptEntries.length > 0) {
        lines.push("========================================");
        lines.push("FULL TRANSCRIPT (from hooks)");
        lines.push(`Source: ${transcriptPath}`);
        lines.push("========================================");
        lines.push("");

        let msgNum = 0;
        for (const entry of transcriptEntries) {
          msgNum++;
          // Transcript format: { type: "user"|"assistant"|..., message: { role, content } }
          const role = entry.type || "unknown";
          const content = entry.message?.content;

          if (role === "user") {
            lines.push(`--- Message #${msgNum}: USER ---`);
            if (typeof content === "string") {
              lines.push(`  ${content}`);
            } else if (Array.isArray(content)) {
              for (const block of content) {
                if (block.type === "text") {
                  lines.push(`  ${block.text}`);
                } else if (block.type === "tool_result") {
                  const preview = typeof block.content === "string"
                    ? block.content.slice(0, 500)
                    : JSON.stringify(block.content).slice(0, 500);
                  lines.push(`  [Tool Result ${block.tool_use_id || ""}]: ${block.is_error ? "ERROR - " : ""}${preview}`);
                } else if (block.type === "image") {
                  lines.push(`  [Image: ${block.source?.media_type || "unknown"}]`);
                } else if (block.type === "document") {
                  lines.push(`  [Document: ${block.source?.media_type || "unknown"}]`);
                }
              }
            } else if (!content) {
              lines.push("  (no content)");
            }
            lines.push("");

          } else if (role === "assistant") {
            lines.push(`--- Message #${msgNum}: ASSISTANT ---`);
            if (typeof content === "string") {
              lines.push(`  ${content}`);
            } else if (Array.isArray(content)) {
              for (const block of content) {
                if (block.type === "text") {
                  lines.push(`  ${block.text}`);
                } else if (block.type === "tool_use") {
                  lines.push(`  [Tool Use: ${block.name}]`);
                  lines.push(`    ID: ${block.id}`);
                  const inputStr = JSON.stringify(block.input, null, 2);
                  for (const inputLine of inputStr.split("\n")) {
                    lines.push(`    ${inputLine}`);
                  }
                } else if (block.type === "thinking") {
                  const preview = (block.thinking || "").slice(0, 300);
                  lines.push(`  [Thinking]: ${preview}${block.thinking?.length > 300 ? "..." : ""}`);
                }
              }
            } else if (!content) {
              lines.push("  (no content)");
            }
            lines.push("");

          } else {
            // queue-operation, system, or other entry types
            lines.push(`--- Message #${msgNum}: ${role.toUpperCase()} ---`);
            lines.push(`  ${JSON.stringify(entry).slice(0, 500)}`);
            lines.push("");
          }
        }
      }
    } catch (err) {
      lines.push(`(Could not read transcript: ${err.message})`);
      lines.push("");
    }
    // Clean up stored path
    transcriptPaths.delete(conversationId);
  } else {
    lines.push("(No transcript captured — no tool calls occurred in this turn)");
    lines.push("");
  }

  // 5. Raw API Calls from proxy logs
  const proxyCalls = proxyLogs.get(conversationId) || [];
  if (proxyCalls.length > 0) {
    lines.push("========================================");
    lines.push("RAW API CALLS (via proxy)");
    lines.push("========================================");
    lines.push("");

    for (let i = 0; i < proxyCalls.length; i++) {
      const call = proxyCalls[i];
      const time = new Date(call.timestamp).toLocaleTimeString();
      lines.push(`--- API Call #${i + 1} @ ${time} ---`);
      lines.push("");
      lines.push(`  ${call.method} ${call.path}`);

      if (call.request) {
        const r = call.request;
        if (r.model) lines.push(`  Model: ${r.model}`);
        if (r.max_tokens) lines.push(`  Max tokens: ${r.max_tokens}`);
        if (r.stream != null) lines.push(`  Stream: ${r.stream}`);

        // System prompt
        if (r.system_full) {
          const sysText = typeof r.system_full === 'string'
            ? r.system_full
            : JSON.stringify(r.system_full);
          lines.push(`  System prompt: ${sysText.length} chars`);
        }

        // Thinking config
        if (r.thinking) {
          lines.push(`  Thinking: ${JSON.stringify(r.thinking)}`);
        }

        // Tools
        if (r.tools) {
          lines.push(`  Tools: ${r.tools.count} [${r.tools.names?.join(', ')}]`);
        }

        // Messages summary
        if (r.messages && r.messages.length > 0) {
          const roleCounts = {};
          let totalChars = 0;
          for (const m of r.messages) {
            roleCounts[m.role] = (roleCounts[m.role] || 0) + 1;
            totalChars += m.char_length || 0;
          }
          const roleStr = Object.entries(roleCounts).map(([role, count]) => `${count} ${role}`).join(', ');
          lines.push(`  Messages: ${r.messages.length} (${roleStr})`);
          lines.push("");

          lines.push("  Context breakdown:");
          if (r.system_full) {
            const sysLen = typeof r.system_full === 'string' ? r.system_full.length : JSON.stringify(r.system_full).length;
            lines.push(`    System prompt: ~${Math.round(sysLen / 4)} tokens`);
          }
          for (let j = 0; j < r.messages.length; j++) {
            const m = r.messages[j];
            const estTokens = Math.round((m.char_length || 0) / 4);
            const types = m.types?.join(', ') || 'text';
            lines.push(`    Messages[${j}] (${m.role}, ${m.block_count} blocks: ${types}): ~${estTokens} tokens`);
          }
          const totalEst = Math.round(totalChars / 4) + (r.system_full ? Math.round((typeof r.system_full === 'string' ? r.system_full.length : JSON.stringify(r.system_full).length) / 4) : 0);
          lines.push(`    Total estimated: ~${totalEst} tokens`);
        }
      }

      // Response
      if (call.response) {
        const resp = call.response;
        lines.push("");
        lines.push("  Response:");
        if (resp.stop_reason) lines.push(`    Stop reason: ${resp.stop_reason}`);
        if (resp.model) lines.push(`    Model: ${resp.model}`);

        // Content blocks
        if (resp.content && Array.isArray(resp.content)) {
          for (const block of resp.content) {
            if (block.type === 'thinking') {
              lines.push(`    Thinking: ${(block.thinking || '').length} chars`);
              const preview = (block.thinking || '').slice(0, 500);
              for (const tLine of preview.split('\n')) {
                lines.push(`      ${tLine}`);
              }
              if ((block.thinking || '').length > 500) lines.push("      ...");
            } else if (block.type === 'text') {
              const preview = (block.text || '').slice(0, 300);
              lines.push(`    Text: "${preview}${(block.text || '').length > 300 ? '...' : ''}"`);
            } else if (block.type === 'tool_use') {
              lines.push(`    Tool use: ${block.name} (id: ${block.id})`);
              lines.push(`      Input: ${JSON.stringify(block.input).slice(0, 300)}`);
            }
          }
        }

        // Usage
        if (resp.usage) {
          lines.push(`    Usage: ${JSON.stringify(resp.usage)}`);
        }
      }

      lines.push("");
    }

    // Clean up proxy logs for this conversation
    proxyLogs.delete(conversationId);
  }

  // 6. Summary from JSONL result entry
  lines.push("========================================");
  lines.push("SUMMARY");
  lines.push("========================================");

  if (resultEntry) {
    if (resultEntry.duration_ms != null) {
      lines.push(`Total Duration: ${(resultEntry.duration_ms / 1000).toFixed(2)}s`);
    }
    if (resultEntry.duration_api_ms != null) {
      lines.push(`API Duration: ${(resultEntry.duration_api_ms / 1000).toFixed(2)}s`);
    }
    if (resultEntry.num_turns != null) {
      lines.push(`Turns: ${resultEntry.num_turns}`);
    }
    if (resultEntry.total_cost_usd != null) {
      lines.push(`Cost: $${resultEntry.total_cost_usd.toFixed(6)}`);
    }

    // Token totals
    const u = resultEntry.usage;
    if (u) {
      let tokenSummary = `Tokens: ${(u.input_tokens || 0).toLocaleString()} in / ${(u.output_tokens || 0).toLocaleString()} out`;
      if (u.cache_read_input_tokens) tokenSummary += ` (${u.cache_read_input_tokens.toLocaleString()} cached)`;
      lines.push(tokenSummary);
    }

    // Per-model breakdown
    if (resultEntry.modelUsage && Object.keys(resultEntry.modelUsage).length > 0) {
      lines.push("");
      lines.push("Model Usage:");
      for (const [m, mu] of Object.entries(resultEntry.modelUsage)) {
        lines.push(`  ${m}: ${mu.inputTokens?.toLocaleString() || 0} in / ${mu.outputTokens?.toLocaleString() || 0} out ($${mu.costUSD?.toFixed(6) || "0"})`);
      }
    }

    // Permission denials
    if (resultEntry.permission_denials?.length > 0) {
      lines.push("");
      lines.push("Permission Denials:");
      for (const d of resultEntry.permission_denials) {
        lines.push(`  ${d.tool_name}: ${JSON.stringify(d.tool_input)}`);
      }
    }

    // Errors
    if (resultEntry.is_error) {
      lines.push("");
      lines.push(`Error: ${resultEntry.result || "unknown error"}`);
    }
  } else {
    lines.push("(No result entry found in JSONL)");
  }

  // OTEL span summary
  lines.push("");
  lines.push(`OTEL Spans: ${spans.length}`);
  lines.push(`LLM Requests: ${llmRequestNum}`);

  const totalSpanDuration = spans.reduce((sum, s) => sum + s.duration, 0);
  lines.push(`Total Span Duration: ${(totalSpanDuration / 1_000_000).toFixed(2)}s`);

  lines.push("");

  // Write the rich trace file
  const traceFile = join(logsDir, conversationId, `${conversationId}.trace.txt`);
  writeFileSync(traceFile, lines.join("\n"), "utf-8");
  console.log(`Rich trace written to ${traceFile}`);
}

async function handleAgentSDK(promptOrContent, conversationId, res) {
  try {
    // query() accepts prompt: string | AsyncIterable<SDKUserMessage>
    // For content blocks (images/PDFs), wrap in an async generator yielding an SDKUserMessage
    let prompt;
    if (Array.isArray(promptOrContent)) {
      prompt = (async function* () {
        yield {
          type: "user",
          message: { role: "user", content: promptOrContent },
          parent_tool_use_id: null,
          session_id: "",
        };
      })();
    } else {
      prompt = promptOrContent;
    }

    const options = {
      systemPrompt:
        "You are a helpful assistant. Respond clearly and concisely.",
      allowedTools: ["WebSearch", "WebFetch"],
      maxTurns: 10,
      env: {
        ...process.env,
        ANTHROPIC_BASE_URL: `http://localhost:${PORT}/proxy/${conversationId}`,
        ENABLE_BETA_TRACING_DETAILED: "1",
        BETA_TRACING_ENDPOINT:
          process.env.BETA_TRACING_ENDPOINT || "http://localhost:4318",
      },
      hooks: {
        PreToolUse: [
          {
            hooks: [
              async (input) => {
                // Capture the transcript path on every tool invocation
                if (input.transcript_path) {
                  transcriptPaths.set(conversationId, input.transcript_path);
                }
                return { continue: true };
              },
            ],
          },
        ],
        PostToolUse: [
          {
            hooks: [
              async (input) => {
                if (input.transcript_path) {
                  transcriptPaths.set(conversationId, input.transcript_path);
                }
                return { continue: true };
              },
            ],
          },
        ],
        SessionEnd: [
          {
            hooks: [
              async (input) => {
                if (input.transcript_path) {
                  transcriptPaths.set(conversationId, input.transcript_path);
                }
                return { continue: true };
              },
            ],
          },
        ],
      },
    };

    // Resume session if we have one for this conversation
    const sessionId = sessions.get(conversationId);
    if (sessionId) {
      options.resume = sessionId;
    }

    const stream = query({ prompt, options });

    for await (const msg of stream) {
      // Log every message to local logs directory
      appendLog(conversationId, msg);

      // Capture session ID from init message
      if (msg.type === "system" && msg.subtype === "init" && msg.session_id) {
        sessions.set(conversationId, msg.session_id);
      }

      // Stream assistant text to client
      if (msg.type === "assistant" && msg.message?.content) {
        for (const block of msg.message.content) {
          if (block.type === "text" && block.text) {
            res.write(`data: ${JSON.stringify({ text: block.text })}\n\n`);
          }
        }
      }

      // Send result info
      if (msg.type === "result" && msg.subtype === "success") {
        res.write(
          `data: ${JSON.stringify({ done: true, result: msg.result })}\n\n`
        );
      }
    }
    // Fire-and-forget: generate rich trace after stream completes
    generateRichTrace(conversationId, sessions.get(conversationId)).catch(
      (err) => console.error("Rich trace generation failed:", err)
    );
  } catch (err) {
    console.error("Agent SDK error:", err);
    res.write(
      `data: ${JSON.stringify({ error: err.message || "Something went wrong" })}\n\n`
    );
  }
}

// ---------------------------------------------------------------------------
// Messages API: raw agent loop with tools
// ---------------------------------------------------------------------------

const IMAGE_TYPES = new Set(["image/png", "image/jpeg", "image/gif", "image/webp"]);

const TOOLS = [
  {
    name: "read_file",
    description:
      "Read the contents of a file that was attached/dropped by the user in this conversation turn. " +
      "Returns the file contents as text. Use this to inspect files the user has shared.",
    input_schema: {
      type: "object",
      properties: {
        filename: {
          type: "string",
          description: "The exact name of the attached file to read.",
        },
      },
      required: ["filename"],
    },
  },
];

function executeToolCall(toolName, toolInput, files) {
  if (toolName === "read_file") {
    const file = files.find((f) => f.name === toolInput.filename);
    if (!file) {
      const available = files.map((f) => f.name).join(", ");
      return {
        type: "tool_result",
        is_error: true,
        content: `File "${toolInput.filename}" not found. Available files: ${available || "(none)"}`,
      };
    }
    if (file.encoding === "base64") {
      // For binary files, return a note about the type; image content is
      // already provided inline in the user message.
      return {
        type: "tool_result",
        content: `[Binary file: ${file.name} (${file.mediaType}, ${Math.round((file.content.length * 3) / 4 / 1024)} KB)]`,
      };
    }
    return { type: "tool_result", content: file.content };
  }
  return { type: "tool_result", is_error: true, content: `Unknown tool: ${toolName}` };
}

function buildUserContent(message, files) {
  const content = [];

  if (files && files.length > 0) {
    // Add images and PDFs as native content blocks
    for (const file of files) {
      if (file.encoding === "base64" && IMAGE_TYPES.has(file.mediaType)) {
        content.push({
          type: "image",
          source: {
            type: "base64",
            media_type: file.mediaType,
            data: file.content,
          },
        });
      } else if (file.encoding === "base64" && file.mediaType === "application/pdf") {
        content.push({
          type: "document",
          source: {
            type: "base64",
            media_type: "application/pdf",
            data: file.content,
          },
        });
      }
      // Text files are accessible via the read_file tool
    }

    // List available files so the model knows what's attached
    const fileNames = files.map((f) => f.name);
    content.push({
      type: "text",
      text: `[Attached files: ${fileNames.join(", ")}]\n\n${message || "Please review the attached file(s)."}`,
    });
  } else {
    content.push({ type: "text", text: message });
  }

  return content;
}

async function handleMessagesAPI(message, files, conversationId, res) {
  const MAX_TURNS = 10;

  try {
    if (!messageHistories.has(conversationId)) {
      messageHistories.set(conversationId, []);
    }
    const history = messageHistories.get(conversationId);

    // Store files for tool access this turn
    turnFiles.set(conversationId, files || []);

    // Build user message with native content blocks for images/PDFs
    const userContent = buildUserContent(message, files);
    history.push({ role: "user", content: userContent });

    // Determine if we need tools (only when text files are attached)
    const hasTextFiles = (files || []).some((f) => f.encoding === "text");
    const toolsForRequest = hasTextFiles ? TOOLS : [];

    // Helper: strip base64 blobs from content blocks for logging
    function stripBase64(content) {
      if (!Array.isArray(content)) return content;
      return content.map((b) => {
        if (b.type === "image") return { type: "image", media_type: b.source?.media_type, data: "[base64 omitted]" };
        if (b.type === "document") return { type: "document", media_type: b.source?.media_type, data: "[base64 omitted]" };
        return b;
      });
    }

    // Agent loop
    for (let turn = 0; turn < MAX_TURNS; turn++) {
      const apiParams = {
        model: "claude-sonnet-4-5-20250929",
        max_tokens: 4096,
        system: "You are a helpful assistant. Respond clearly and concisely.",
        messages: history,
      };
      if (toolsForRequest.length > 0) {
        apiParams.tools = toolsForRequest;
      }

      // Log the full request (messages with base64 stripped)
      appendLog(conversationId, {
        type: "messages-api-request",
        turn,
        timestamp: new Date().toISOString(),
        request: {
          model: apiParams.model,
          max_tokens: apiParams.max_tokens,
          system: apiParams.system,
          tools: apiParams.tools || [],
          messages: apiParams.messages.map((m) => ({
            role: m.role,
            content: stripBase64(m.content),
          })),
        },
      });

      // Stream this turn
      const stream = anthropic.messages.stream(apiParams);

      stream.on("text", (text) => {
        res.write(`data: ${JSON.stringify({ text })}\n\n`);
      });

      const finalMessage = await stream.finalMessage();

      // Log the complete raw API response
      appendLog(conversationId, {
        type: "messages-api-response",
        turn,
        timestamp: new Date().toISOString(),
        response: {
          id: finalMessage.id,
          type: finalMessage.type,
          role: finalMessage.role,
          model: finalMessage.model,
          stop_reason: finalMessage.stop_reason,
          stop_sequence: finalMessage.stop_sequence,
          usage: finalMessage.usage,
          content: finalMessage.content,
        },
      });

      // Add full assistant message to history
      history.push({ role: "assistant", content: finalMessage.content });

      // If no tool use, we're done
      if (finalMessage.stop_reason !== "tool_use") {
        break;
      }

      // Execute tool calls and add results
      const toolResults = [];
      for (const block of finalMessage.content) {
        if (block.type !== "tool_use") continue;

        const result = executeToolCall(
          block.name,
          block.input,
          turnFiles.get(conversationId) || []
        );

        const toolResultEntry = {
          type: "tool_result",
          tool_use_id: block.id,
          content: result.content,
          ...(result.is_error ? { is_error: true } : {}),
        };

        appendLog(conversationId, {
          type: "messages-api-tool-execution",
          turn,
          timestamp: new Date().toISOString(),
          tool_use_id: block.id,
          tool_name: block.name,
          tool_input: block.input,
          tool_result: toolResultEntry,
        });

        toolResults.push(toolResultEntry);
      }

      history.push({ role: "user", content: toolResults });
    }

    // Clean up turn files
    turnFiles.delete(conversationId);

    res.write(
      `data: ${JSON.stringify({ done: true })}\n\n`
    );
  } catch (err) {
    console.error("Messages API error:", err);
    res.write(
      `data: ${JSON.stringify({ error: err.message || "Something went wrong" })}\n\n`
    );
  }
}

app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
});
