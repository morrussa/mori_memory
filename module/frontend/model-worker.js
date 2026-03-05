const STREAM_API_URL = "/mori/chat/stream";
const DEFAULT_THREAD_ID = "mori";

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

            if (payload.type === "token" && payload.token) {
                postMessage({ type: "newToken", payload: { token: String(payload.token) } });
                continue;
            }
            if (payload.type === "error") {
                postWorkerError(payload.message || "Backend stream error.");
                continue;
            }
            if (payload.type === "uploads" && Array.isArray(payload.files)) {
                postMessage({ type: "uploads", payload: { files: payload.files } });
                continue;
            }
            if (payload.type === "done") {
                if (!doneNotified) {
                    doneNotified = true;
                    postMessage({ type: "tokensDone" });
                }
            }
        }
    }

    if (!doneNotified) {
        postMessage({ type: "tokensDone" });
    }
}

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
