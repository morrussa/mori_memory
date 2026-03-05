/**
 * Model Worker (Graph V1)
 * Multipart upload + SSE event protocol
 */

const STREAM_API_URL = "/mori/chat/stream";

function postWorkerError(message) {
  postMessage({ type: "error", payload: { message: String(message || "Unknown error") } });
}

function appendFormDataFiles(form, files) {
  if (!Array.isArray(files)) return;
  for (const file of files) {
    if (!file || typeof file !== "object") continue;
    if (typeof file.arrayBuffer === "function") {
      const name = String(file.name || "upload.bin");
      form.append("files[]", file, name);
    }
  }
}

function dispatchEvent(eventName, payload, doneState) {
  const event = String(eventName || "").trim();
  const data = (payload && typeof payload === "object") ? payload : {};

  if (event === "token") {
    const token = typeof data.token === "string" ? data.token : "";
    if (token) {
      doneState.tokenCount = (doneState.tokenCount || 0) + 1;
      postMessage({ type: "newToken", payload: { token } });
    }
    return;
  }

  if (event === "uploads") {
    postMessage({ type: "uploads", payload: { files: Array.isArray(data.files) ? data.files : [] } });
    return;
  }

  if (event === "run_start") {
    postMessage({ type: "runStart", payload: data });
    return;
  }

  if (event === "node_start") {
    postMessage({ type: "nodeStart", payload: data });
    return;
  }

  if (event === "node_end") {
    postMessage({ type: "nodeEnd", payload: data });
    return;
  }

  if (event === "tool_call") {
    postMessage({
      type: "toolCall",
      payload: {
        name: data.tool || data.name || "unknown",
        arguments: data.args || data.arguments || {},
        callId: data.call_id || data.callId || "",
      }
    });
    return;
  }

  if (event === "tool_result") {
    postMessage({
      type: "toolResult",
      payload: {
        name: data.tool || data.name || "unknown",
        callId: data.call_id || data.callId || "",
        result: data.result,
        error: data.error,
        ok: data.ok,
      }
    });
    return;
  }

  if (event === "status") {
    const statusMsg = String(data.message || data.phase || "");
    postMessage({ type: "status", payload: { message: statusMsg } });
    return;
  }

  if (event === "error") {
    postWorkerError(data.message || "Backend stream error.");
    return;
  }

  if (event === "done") {
    if (!doneState.done) {
      doneState.done = true;
      const finalMessage = typeof data.message === "string" ? data.message : "";
      if ((doneState.tokenCount || 0) === 0 && finalMessage) {
        postMessage({ type: "newToken", payload: { token: finalMessage } });
      }
      postMessage({ type: "runDone", payload: data });
      postMessage({ type: "tokensDone" });
    }
    return;
  }

  // Unknown event -> status fallback
  if (event) {
    postMessage({ type: "status", payload: { message: `[${event}] ${JSON.stringify(data)}` } });
  }
}

async function streamChat(message, files, onSent) {
  const form = new FormData();
  form.append("message", String(message || ""));
  appendFormDataFiles(form, files);

  const response = await fetch(STREAM_API_URL, {
    method: "POST",
    body: form,
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
  const doneState = { done: false, tokenCount: 0 };

  let curEvent = "message";
  let dataLines = [];

  function flushEvent() {
    if (dataLines.length === 0) {
      curEvent = "message";
      dataLines = [];
      return;
    }
    const rawData = dataLines.join("\n");
    if (rawData === "[DONE]") {
      dispatchEvent("done", {}, doneState);
      curEvent = "message";
      dataLines = [];
      return;
    }
    let payload = {};
    try {
      payload = rawData ? JSON.parse(rawData) : {};
    } catch (_e) {
      payload = { message: rawData };
    }
    dispatchEvent(curEvent, payload, doneState);
    curEvent = "message";
    dataLines = [];
  }

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    buffer = buffer.replace(/\r/g, "");

    while (true) {
      const newline = buffer.indexOf("\n");
      if (newline < 0) break;
      const line = buffer.slice(0, newline);
      buffer = buffer.slice(newline + 1);

      if (line === "") {
        flushEvent();
        continue;
      }
      if (line.startsWith(":")) {
        continue;
      }
      if (line.startsWith("event:")) {
        curEvent = line.slice(6).trim() || "message";
        continue;
      }
      if (line.startsWith("data:")) {
        dataLines.push(line.slice(5).trim());
        continue;
      }
    }
  }

  buffer += decoder.decode();
  if (buffer.length > 0) {
    buffer = buffer.replace(/\r/g, "");
    if (!buffer.endsWith("\n")) {
      buffer += "\n";
    }
    while (true) {
      const newline = buffer.indexOf("\n");
      if (newline < 0) break;
      const line = buffer.slice(0, newline);
      buffer = buffer.slice(newline + 1);

      if (line === "") {
        flushEvent();
        continue;
      }
      if (line.startsWith(":")) {
        continue;
      }
      if (line.startsWith("event:")) {
        curEvent = line.slice(6).trim() || "message";
        continue;
      }
      if (line.startsWith("data:")) {
        dataLines.push(line.slice(5).trim());
        continue;
      }
    }
  }

  flushEvent();

  if (!doneState.done) {
    postMessage({ type: "tokensDone" });
  }
}

self.onmessage = async function(event) {
  const data = event.data || {};

  if (data.type === "init") {
    return;
  }

  if (data.type !== "chatMessage") {
    postWorkerError(`Unsupported message type: ${data.type}`);
    return;
  }

  const message = String(data.message || "");
  const files = Array.isArray(data.files) ? data.files : [];

  try {
    await streamChat(message, files, null);
  } catch (err) {
    postWorkerError(err && err.message ? err.message : String(err || "Unknown error"));
  }
};
