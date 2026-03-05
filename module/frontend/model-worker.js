/**
 * Model Worker
 * Handles streaming chat and Agent events from backend
 * No build required - Pure JavaScript Web Worker
 */

const STREAM_API_URL = "/mori/chat/stream";
const DEFAULT_THREAD_ID = "mori";

// ===== Utility Functions =====

function postWorkerError(message) {
  postMessage({ type: "error", payload: { message: String(message || "Unknown error") } });
}

function parseSseEvent(rawEvent) {
  const lines = String(rawEvent || "")
    .replace(/\r/g, "")
    .split("\n");
  const dataLines = [];
  for (const line of lines) {
    if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trim());
    }
  }
  if (dataLines.length === 0) {
    return "";
  }
  return dataLines.join("\n");
}

function bytesToBase64(bytes) {
  let binary = "";
  const chunkSize = 0x8000;
  for (let i = 0; i < bytes.length; i += chunkSize) {
    const chunk = bytes.subarray(i, i + chunkSize);
    binary += String.fromCharCode.apply(null, chunk);
  }
  return btoa(binary);
}

async function normalizeUploadFiles(rawFiles) {
  if (!Array.isArray(rawFiles) || rawFiles.length === 0) {
    return [];
  }
  const out = [];
  for (const item of rawFiles) {
    if (!item || typeof item !== "object") {
      continue;
    }

    if (typeof item.arrayBuffer === "function") {
      const ab = await item.arrayBuffer();
      const bytes = new Uint8Array(ab);
      out.push({
        name: String(item.name || "upload.bin"),
        mime: String(item.type || "application/octet-stream"),
        size: Number(item.size || bytes.length || 0),
        content_base64: bytesToBase64(bytes),
      });
      continue;
    }

    if (typeof item.content_base64 === "string" && item.content_base64) {
      out.push({
        name: String(item.name || "upload.bin"),
        mime: String(item.mime || "application/octet-stream"),
        size: Number(item.size || 0),
        content_base64: item.content_base64,
      });
    }
  }
  return out;
}

// ===== Main Streaming Function =====

async function streamChat(message, files, onSent) {
  const payload = {
    message: String(message || ""),
    thread_id: DEFAULT_THREAD_ID,
  };
  
  const normalizedFiles = await normalizeUploadFiles(files);
  if (normalizedFiles.length > 0) {
    payload.files = normalizedFiles;
  }

  const response = await fetch(STREAM_API_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const bodyText = await response.text();
    throw new Error(`HTTP ${response.status}: ${bodyText || "empty response"}`);
  }
  
  if (!response.body) {
    throw new Error("Missing stream body.");
  }

  postMessage({ type: "messageSent" });
  if (typeof onSent === "function") {
    onSent();
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  let doneNotified = false;

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    buffer = buffer.replace(/\r/g, "");

    while (true) {
      const idx = buffer.indexOf("\n\n");
      if (idx < 0) {
        break;
      }
      const rawEvent = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);

      const data = parseSseEvent(rawEvent);
      if (!data) {
        continue;
      }
      if (data === "[DONE]") {
        if (!doneNotified) {
          doneNotified = true;
          postMessage({ type: "tokensDone" });
        }
        return;
      }

      let payload;
      try {
        payload = JSON.parse(data);
      } catch (_e) {
        continue;
      }
      if (!payload || typeof payload !== "object") {
        continue;
      }

      // Handle different event types
      handleEventPayload(payload, doneNotified, () => { doneNotified = true; });
    }
  }

  if (!doneNotified) {
    postMessage({ type: "tokensDone" });
  }
}

// ===== Event Payload Handler =====

function handleEventPayload(payload, doneNotified, markDone) {
  // Standard token event
  if (payload.type === "token" && payload.token) {
    postMessage({ type: "newToken", payload: { token: String(payload.token) } });
    return;
  }

  // Error event
  if (payload.type === "error") {
    postWorkerError(payload.message || "Backend stream error.");
    return;
  }

  // File uploads event
  if (payload.type === "uploads" && Array.isArray(payload.files)) {
    postMessage({ type: "uploads", payload: { files: payload.files } });
    return;
  }

  // Done event
  if (payload.type === "done") {
    if (!doneNotified) {
      markDone();
      postMessage({ type: "tokensDone" });
    }
    return;
  }

  // ===== Agent-specific events =====

  // Tool call started
  if (payload.type === "tool_call" || payload.type === "toolCall" || payload.type === "tool_use") {
    postMessage({
      type: "toolCall",
      payload: {
        name: payload.name || payload.tool_name || payload.tool || "unknown",
        arguments: payload.arguments || payload.args || payload.parameters || {}
      }
    });
    return;
  }

  // Tool call result
  if (payload.type === "tool_result" || payload.type === "toolResult" || payload.type === "tool_response") {
    postMessage({
      type: "toolResult",
      payload: {
        result: payload.result || payload.output || payload.response,
        error: payload.error || payload.err
      }
    });
    return;
  }

  // Thinking/reasoning event
  if (payload.type === "thinking" || payload.type === "reasoning" || payload.type === "thought") {
    postMessage({
      type: "thinking",
      payload: {
        content: payload.content || payload.text || payload.thought || ""
      }
    });
    return;
  }

  // Status/message event
  if (payload.type === "status" || payload.type === "message" || payload.type === "info") {
    postMessage({
      type: "status",
      payload: {
        message: payload.message || payload.text || payload.info || ""
      }
    });
    return;
  }

  // Code execution event
  if (payload.type === "code_execution" || payload.type === "codeExecution") {
    postMessage({
      type: "toolCall",
      payload: {
        name: "code_execution",
        arguments: { code: payload.code || payload.source || "" }
      }
    });
    
    if (payload.output || payload.result) {
      postMessage({
        type: "toolResult",
        payload: {
          result: payload.output || payload.result,
          error: payload.error
        }
      });
    }
    return;
  }

  // File operation event
  if (payload.type === "file_operation" || payload.type === "fileOperation") {
    postMessage({
      type: "toolCall",
      payload: {
        name: payload.operation || "file_operation",
        arguments: {
          path: payload.path || "",
          content: payload.content || ""
        }
      }
    });
    return;
  }

  // Generic event with content - treat as token if it has content
  if (payload.content || payload.text || payload.delta) {
    const token = payload.content || payload.text || payload.delta;
    if (typeof token === "string") {
      postMessage({ type: "newToken", payload: { token } });
    }
    return;
  }

  // Unknown event type - log for debugging
  console.log("Unknown event payload:", payload);
}

// ===== Worker Message Handler =====

self.onmessage = async function(event) {
  if (event.data.type === "init") {
    return;
  }
  
  if (event.data.type !== "chatMessage") {
    return;
  }

  let sentNotified = false;
  try {
    await streamChat(event.data.message, event.data.files, () => {
      sentNotified = true;
    });
  } catch (e) {
    postWorkerError(e && e.message ? e.message : String(e));
    if (!sentNotified) {
      postMessage({ type: "messageSent" });
    }
    postMessage({ type: "tokensDone" });
  }
};
