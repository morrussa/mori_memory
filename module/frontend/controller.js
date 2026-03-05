/**
 * Modern Controller
 * Wires up components, manages theme, handles Agent events
 * No build required - Pure JavaScript
 */

(function() {
  'use strict';

  // Initialize worker
  const worker = new Worker('model-worker.js');

  // Initialize components
  const messagesArea = document.querySelector('messages-area');
  const messageInput = document.querySelector('message-input');

  messagesArea.init(worker);
  messageInput.init(worker);
  messageInput.setMessagesArea(messagesArea);

  // Initialize worker
  worker.postMessage({ type: 'init' });

  // Show initial system message
  messagesArea.appendSystemMessage('已连接到 Agent 服务');

  // ===== Theme Management =====
  const ThemeManager = {
    STORAGE_KEY: 'agent-chat-theme',
    DARK_THEME: 'dark',
    LIGHT_THEME: 'light',

    init() {
      // Check for saved preference or system preference
      const saved = localStorage.getItem(this.STORAGE_KEY);
      const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;

      const theme = saved || (prefersDark ? this.DARK_THEME : this.LIGHT_THEME);
      this.setTheme(theme);

      // Listen for system theme changes
      window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
        if (!localStorage.getItem(this.STORAGE_KEY)) {
          this.setTheme(e.matches ? this.DARK_THEME : this.LIGHT_THEME);
        }
      });
    },

    setTheme(theme) {
      document.documentElement.setAttribute('data-theme', theme);
      localStorage.setItem(this.STORAGE_KEY, theme);
      this.updateToggleButton();
    },

    toggle() {
      const current = document.documentElement.getAttribute('data-theme');
      const next = current === this.DARK_THEME ? this.LIGHT_THEME : this.DARK_THEME;
      this.setTheme(next);
    },

    updateToggleButton() {
      const btn = document.querySelector('#themeToggle');
      if (!btn) return;

      const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
      const icon = isDark
        ? `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>`
        : `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>`;

      btn.innerHTML = icon;
      btn.title = isDark ? '切换到浅色模式' : '切换到深色模式';
    }
  };

  ThemeManager.init();

  // Make theme toggle available globally
  window.toggleTheme = () => ThemeManager.toggle();

  // ===== Status Check =====
  const statusAbort = new AbortController();
  setTimeout(() => statusAbort.abort(), 4000);

  fetch('/mori/session/status', { signal: statusAbort.signal })
    .then((resp) => resp.ok ? resp.json() : null)
    .then((status) => {
      if (!status || typeof status !== 'object') return;

      if (typeof messageInput.setUploadLimits === 'function' && status.upload_limits) {
        messageInput.setUploadLimits(status.upload_limits);
      }

      if (typeof messagesArea.appendSystemMessage === 'function') {
        if (status.upload_dir) {
          messagesArea.appendSystemMessage(`上传目录: ${status.upload_dir}`);
        }
        if (status.model_name) {
          messagesArea.appendSystemMessage(`模型: ${status.model_name}`);
        }
      }
    })
    .catch((e) => {
      const msg = (e && e.name === 'AbortError')
        ? '状态检查超时（后端繁忙）'
        : '状态检查失败';
      messagesArea.appendSystemMessage(msg);
    });

  // ===== Worker Message Handler =====
  // Track current tool call for status updates
  let currentToolEl = null;

  worker.onmessage = function(event) {
    const { type, payload } = event.data;

    switch (type) {
      case 'messageSent':
        messageInput.handleMessageSent();
        break;

      case 'newToken':
        messagesArea.handleNewToken(payload.token);
        break;

      case 'tokensDone':
        messagesArea.handleTokensDone();
        currentToolEl = null;
        break;

      case 'error':
        messagesArea.handleError(payload && payload.message ? payload.message : '未知错误');
        currentToolEl = null;
        break;

      case 'uploads':
        if (typeof messagesArea.handleUploads === 'function') {
          messagesArea.handleUploads(payload && payload.files ? payload.files : []);
        }
        break;

      // Agent-specific events
      case 'toolCall':
        // Tool call started
        currentToolEl = messagesArea.appendToolCall(
          payload.name || 'unknown',
          payload.arguments || {},
          'running'
        );
        break;

      case 'toolResult':
        // Tool call completed
        if (currentToolEl) {
          messagesArea.updateToolStatus(
            currentToolEl,
            payload.error ? 'error' : 'success',
            payload.result || payload.error
          );
        }
        currentToolEl = null;
        break;

      case 'thinking':
        // Thinking/reasoning process
        messagesArea.appendThinkingBlock(payload.content || '');
        break;

      case 'status':
        // Status update message
        messagesArea.appendSystemMessage(payload.message || '');
        break;

      default:
        console.log('Unknown event type from worker:', type, payload);
    }
  };

  // ===== Keyboard Shortcuts =====
  document.addEventListener('keydown', (e) => {
    // Ctrl/Cmd + / to toggle theme
    if ((e.ctrlKey || e.metaKey) && e.key === '/') {
      e.preventDefault();
      ThemeManager.toggle();
    }

    // Escape to focus input
    if (e.key === 'Escape') {
      messageInput.focus();
    }
  });

  // ===== Visibility Change Handler =====
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') {
      messageInput.focus();
    }
  });

  // Log initialization
  console.log('Agent Chat initialized');
})();
