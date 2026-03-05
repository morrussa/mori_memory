## Mori Local No-Build Frontend

This directory contains the static no-build chat UI used by `MORI_RUN_MODE=webui`.
The page is served by `main.py` and talks to Mori native endpoints.

### Runtime integration

- Static assets are served from `MORI_FRONTEND_ROOT` (default: `module/frontend`).
- Chat streaming endpoint: `POST /mori/chat/stream` (SSE).
- Non-stream endpoint: `POST /mori/chat`.
- File upload is embedded in chat payload: `files[]` with base64 content.
- Frontend reads `/mori/session/status` at startup to sync server-side upload limits.
- Legacy OpenAI endpoint `/v1/chat/completions` is deprecated in this mode and returns `410`.

### File map

- `index.html`: main page shell
- `styles.css`: global styles
- `controller.js`: wires web components and worker events
- `messageInput.js`: input component
- `messagesArea.js`: output component and markdown rendering
- `model-worker.js`: stream client for `/mori/chat/stream`

### Notes

- This frontend assumes same-origin deployment with `main.py` and does not require API keys.
- Markdown rendering now has a built-in fallback (no CDN dependency required).
- Uploaded files are stored server-side under `./workspace/download/`.
- The UI supports selecting multiple files, removing individual files before send, and client-side size/count pre-checks.
- If you serve files separately, make sure your static host and Mori backend CORS/network rules allow the worker fetch to `/mori/chat/stream`.
