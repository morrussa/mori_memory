/**
 * Modern Messages Area Component
 * Supports Agent features: tool calls, thinking steps, code blocks
 * No build required - Pure Web Components
 */

const messagesAreaTemplate = document.createElement('template');
messagesAreaTemplate.innerHTML = `
<style>
  :host {
    display: flex;
    flex-direction: column;
    flex: 1;
    overflow: hidden;
    background: var(--color-bg-primary);
  }

  #messagesContainer {
    flex: 1;
    overflow-y: auto;
    overflow-x: hidden;
    padding: var(--space-lg);
    display: flex;
    flex-direction: column;
    gap: var(--space-md);
    scroll-behavior: smooth;
  }

  /* Welcome Screen */
  .welcome-screen {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: var(--space-2xl);
    text-align: center;
  }

  .welcome-icon {
    width: 72px;
    height: 72px;
    background: linear-gradient(135deg, var(--color-primary), var(--color-secondary));
    border-radius: var(--radius-xl);
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: var(--space-xl);
    box-shadow: var(--shadow-lg);
    animation: fadeIn 0.5s ease;
  }

  .welcome-icon svg {
    width: 40px;
    height: 40px;
    color: white;
  }

  .welcome-title {
    font-size: var(--font-size-xl);
    font-weight: var(--font-weight-semibold);
    color: var(--color-text-primary);
    margin-bottom: var(--space-sm);
  }

  .welcome-subtitle {
    font-size: var(--font-size-sm);
    color: var(--color-text-tertiary);
    max-width: 320px;
    line-height: var(--line-height-relaxed);
  }

  .welcome-features {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-sm);
    margin-top: var(--space-xl);
    justify-content: center;
  }

  .feature-tag {
    display: flex;
    align-items: center;
    gap: var(--space-xs);
    padding: var(--space-xs) var(--space-md);
    background: var(--color-bg-tertiary);
    border-radius: var(--radius-full);
    font-size: var(--font-size-xs);
    color: var(--color-text-secondary);
  }

  .feature-tag svg {
    width: 14px;
    height: 14px;
  }

  /* Message Styles */
  .message-group {
    display: flex;
    flex-direction: column;
    gap: var(--space-sm);
    animation: fadeIn 0.3s ease;
  }

  .message-header {
    display: flex;
    align-items: center;
    gap: var(--space-sm);
  }

  .message-avatar {
    width: 32px;
    height: 32px;
    border-radius: var(--radius-md);
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
  }

  .message-avatar svg {
    width: 18px;
    height: 18px;
    color: white;
  }

  .message-avatar.user {
    background: linear-gradient(135deg, var(--color-primary), var(--color-primary-dark));
  }

  .message-avatar.assistant {
    background: linear-gradient(135deg, var(--color-secondary), #7c3aed);
  }

  .message-avatar.system {
    background: var(--color-text-tertiary);
  }

  .message-avatar.tool {
    background: linear-gradient(135deg, var(--color-warning), var(--color-warning));
  }

  .message-avatar.thinking {
    background: linear-gradient(135deg, var(--color-info), #0891b2);
  }

  .message-sender {
    font-size: var(--font-size-sm);
    font-weight: var(--font-weight-medium);
    color: var(--color-text-primary);
  }

  .message-time {
    font-size: var(--font-size-xs);
    color: var(--color-text-tertiary);
  }

  .message-content {
    padding: var(--space-md) var(--space-lg);
    border-radius: var(--radius-lg);
    font-size: var(--font-size-md);
    line-height: var(--line-height-relaxed);
    max-width: 85%;
    word-wrap: break-word;
    overflow-wrap: break-word;
  }

  .message-content.user {
    background: var(--color-user-bg);
    color: var(--color-text-primary);
    align-self: flex-end;
    border-bottom-right-radius: var(--radius-sm);
  }

  .message-content.assistant {
    background: var(--color-assistant-bg);
    color: var(--color-text-primary);
    align-self: flex-start;
    border-bottom-left-radius: var(--radius-sm);
  }

  .message-content.system {
    background: var(--color-system-bg);
    color: var(--color-text-secondary);
    font-size: var(--font-size-sm);
    align-self: center;
    text-align: center;
    padding: var(--space-sm) var(--space-md);
  }

  .message-content.error {
    background: var(--color-error-light);
    color: var(--color-error-dark);
    border-left: 3px solid var(--color-error);
  }

  /* Tool Call Styles */
  .tool-call {
    background: var(--color-tool-bg);
    border: 1px solid var(--color-warning);
    border-radius: var(--radius-md);
    margin: var(--space-sm) 0;
    overflow: hidden;
  }

  .tool-header {
    display: flex;
    align-items: center;
    gap: var(--space-sm);
    padding: var(--space-sm) var(--space-md);
    background: rgba(245, 158, 11, 0.1);
    cursor: pointer;
    user-select: none;
  }

  .tool-header:hover {
    background: rgba(245, 158, 11, 0.15);
  }

  .tool-icon {
    width: 20px;
    height: 20px;
    color: var(--color-warning);
  }

  .tool-name {
    font-size: var(--font-size-sm);
    font-weight: var(--font-weight-medium);
    color: var(--color-text-primary);
    flex: 1;
  }

  .tool-status {
    font-size: var(--font-size-xs);
    padding: 2px 8px;
    border-radius: var(--radius-full);
    font-weight: var(--font-weight-medium);
  }

  .tool-status.running {
    background: var(--color-info-light);
    color: var(--color-info);
  }

  .tool-status.success {
    background: var(--color-success-light);
    color: var(--color-success-dark);
  }

  .tool-status.error {
    background: var(--color-error-light);
    color: var(--color-error);
  }

  .tool-expand {
    width: 16px;
    height: 16px;
    color: var(--color-text-tertiary);
    transition: transform var(--transition-fast);
  }

  .tool-call.expanded .tool-expand {
    transform: rotate(180deg);
  }

  .tool-body {
    display: none;
    padding: var(--space-md);
    border-top: 1px solid var(--color-border);
    background: var(--color-surface);
  }

  .tool-call.expanded .tool-body {
    display: block;
  }

  .tool-section {
    margin-bottom: var(--space-md);
  }

  .tool-section:last-child {
    margin-bottom: 0;
  }

  .tool-section-label {
    font-size: var(--font-size-xs);
    font-weight: var(--font-weight-medium);
    color: var(--color-text-tertiary);
    text-transform: uppercase;
    margin-bottom: var(--space-xs);
  }

  .tool-section-content {
    background: var(--color-bg-tertiary);
    padding: var(--space-sm);
    border-radius: var(--radius-sm);
    font-family: var(--font-family-mono);
    font-size: var(--font-size-xs);
    overflow-x: auto;
    white-space: pre-wrap;
    word-break: break-all;
  }

  /* Thinking Block Styles */
  .thinking-block {
    background: var(--color-thinking-bg);
    border: 1px solid var(--color-info);
    border-radius: var(--radius-md);
    margin: var(--space-sm) 0;
    overflow: hidden;
  }

  .thinking-header {
    display: flex;
    align-items: center;
    gap: var(--space-sm);
    padding: var(--space-sm) var(--space-md);
    background: rgba(6, 182, 212, 0.1);
    cursor: pointer;
    user-select: none;
  }

  .thinking-header:hover {
    background: rgba(6, 182, 212, 0.15);
  }

  .thinking-icon {
    width: 20px;
    height: 20px;
    color: var(--color-info);
  }

  .thinking-label {
    font-size: var(--font-size-sm);
    font-weight: var(--font-weight-medium);
    color: var(--color-text-primary);
    flex: 1;
  }

  .thinking-expand {
    width: 16px;
    height: 16px;
    color: var(--color-text-tertiary);
    transition: transform var(--transition-fast);
  }

  .thinking-block.expanded .thinking-expand {
    transform: rotate(180deg);
  }

  .thinking-content {
    display: none;
    padding: var(--space-md);
    border-top: 1px solid var(--color-border);
    background: var(--color-surface);
    font-size: var(--font-size-sm);
    color: var(--color-text-secondary);
    line-height: var(--line-height-relaxed);
    white-space: pre-wrap;
  }

  .thinking-block.expanded .thinking-content {
    display: block;
  }

  /* Code Block Styles */
  .code-block {
    background: var(--color-code-bg);
    border-radius: var(--radius-md);
    margin: var(--space-md) 0;
    overflow: hidden;
  }

  .code-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: var(--space-sm) var(--space-md);
    background: rgba(0, 0, 0, 0.2);
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  }

  .code-language {
    font-size: var(--font-size-xs);
    color: #94a3b8;
    text-transform: uppercase;
    font-weight: var(--font-weight-medium);
  }

  .code-copy-btn {
    display: flex;
    align-items: center;
    gap: var(--space-xs);
    padding: var(--space-xs) var(--space-sm);
    background: transparent;
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: var(--radius-sm);
    color: #94a3b8;
    font-size: var(--font-size-xs);
    cursor: pointer;
    transition: all var(--transition-fast);
  }

  .code-copy-btn:hover {
    background: rgba(255, 255, 255, 0.1);
    color: #e2e8f0;
  }

  .code-copy-btn svg {
    width: 14px;
    height: 14px;
  }

  .code-content {
    padding: var(--space-md);
    overflow-x: auto;
    font-family: var(--font-family-mono);
    font-size: var(--font-size-sm);
    line-height: var(--line-height-relaxed);
    color: #e2e8f0;
  }

  .code-content pre {
    margin: 0;
    white-space: pre-wrap;
    word-break: break-word;
  }

  /* Typing Indicator */
  .typing-indicator {
    display: flex;
    align-items: center;
    gap: var(--space-sm);
    padding: var(--space-md) var(--space-lg);
    background: var(--color-assistant-bg);
    border-radius: var(--radius-lg);
    align-self: flex-start;
    animation: fadeIn 0.3s ease;
  }

  .typing-dots {
    display: flex;
    gap: 4px;
  }

  .typing-dot {
    width: 8px;
    height: 8px;
    background: var(--color-secondary);
    border-radius: 50%;
    animation: typing 1.4s infinite ease-in-out;
  }

  .typing-dot:nth-child(1) { animation-delay: 0s; }
  .typing-dot:nth-child(2) { animation-delay: 0.2s; }
  .typing-dot:nth-child(3) { animation-delay: 0.4s; }

  @keyframes typing {
    0%, 60%, 100% { transform: translateY(0); opacity: 0.5; }
    30% { transform: translateY(-6px); opacity: 1; }
  }

  /* Streaming Text */
  .streaming-text {
    display: inline;
  }

  .streaming-cursor {
    display: inline-block;
    width: 2px;
    height: 1.2em;
    background: var(--color-primary);
    margin-left: 2px;
    animation: blink 1s step-end infinite;
    vertical-align: text-bottom;
  }

  @keyframes blink {
    0%, 50% { opacity: 1; }
    51%, 100% { opacity: 0; }
  }

  /* File Upload Display */
  .file-upload-info {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-xs);
    margin-top: var(--space-sm);
  }

  .file-chip {
    display: inline-flex;
    align-items: center;
    gap: var(--space-xs);
    padding: var(--space-xs) var(--space-sm);
    background: var(--color-bg-tertiary);
    border-radius: var(--radius-full);
    font-size: var(--font-size-xs);
    color: var(--color-text-secondary);
  }

  .file-chip svg {
    width: 12px;
    height: 12px;
  }

  /* Inline Code */
  .message-content code:not([class]) {
    background: var(--color-bg-tertiary);
    padding: 2px 6px;
    border-radius: var(--radius-sm);
    font-family: var(--font-family-mono);
    font-size: 0.9em;
    color: var(--color-primary);
  }

  /* Markdown Elements */
  .message-content ul,
  .message-content ol {
    margin: var(--space-sm) 0;
    padding-left: var(--space-xl);
  }

  .message-content li {
    margin-bottom: var(--space-xs);
  }

  .message-content blockquote {
    margin: var(--space-sm) 0;
    padding: var(--space-sm) var(--space-md);
    border-left: 3px solid var(--color-primary);
    background: var(--color-bg-tertiary);
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    color: var(--color-text-secondary);
  }

  .message-content a {
    color: var(--color-primary);
    text-decoration: none;
  }

  .message-content a:hover {
    text-decoration: underline;
  }

  .message-content table {
    width: 100%;
    border-collapse: collapse;
    margin: var(--space-md) 0;
    font-size: var(--font-size-sm);
  }

  .message-content th,
  .message-content td {
    padding: var(--space-sm);
    border: 1px solid var(--color-border);
    text-align: left;
  }

  .message-content th {
    background: var(--color-bg-tertiary);
    font-weight: var(--font-weight-medium);
  }

  /* Animations */
  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
  }

  /* Responsive */
  @media (max-width: 640px) {
    #messagesContainer {
      padding: var(--space-md);
    }

    .message-content {
      max-width: 90%;
      padding: var(--space-sm) var(--space-md);
    }

    .welcome-icon {
      width: 56px;
      height: 56px;
    }

    .welcome-icon svg {
      width: 32px;
      height: 32px;
    }
  }
</style>

<div id="messagesContainer">
  <div class="welcome-screen" id="welcomeScreen">
    <div class="welcome-icon">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M12 2L2 7l10 5 10-5-10-5z"/>
        <path d="M2 17l10 5 10-5"/>
        <path d="M2 12l10 5 10-5"/>
      </svg>
    </div>
    <div class="welcome-title">Agent Chat</div>
    <div class="welcome-subtitle">与智能助手对话，支持文件上传、代码执行和工具调用</div>
    <div class="welcome-features">
      <span class="feature-tag">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/><polyline points="14 2 14 8 20 8"/></svg>
        文件上传
      </span>
      <span class="feature-tag">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>
        代码执行
      </span>
      <span class="feature-tag">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/></svg>
        工具调用
      </span>
    </div>
  </div>
</div>
`;

class MessagesArea extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
    this.shadowRoot.appendChild(messagesAreaTemplate.content.cloneNode(true));
    
    this.container = this.shadowRoot.querySelector('#messagesContainer');
    this.welcomeScreen = this.shadowRoot.querySelector('#welcomeScreen');
    this.accumulatingMessageEl = null;
    this.accumulatingText = '';
    this.currentMessageGroup = null;
    this.hasMessages = false;
    this.isStreaming = false;
    this.messageCount = 0;
  }

  init(worker) {
    this.worker = worker;
  }

  // SVG Icons
  getIcons() {
    return {
      user: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>`,
      assistant: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>`,
      system: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>`,
      tool: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/></svg>`,
      thinking: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>`,
      chevronDown: `<svg class="tool-expand" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"/></svg>`,
      copy: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>`,
      check: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>`,
      file: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/><polyline points="14 2 14 8 20 8"/></svg>`
    };
  }

  hideWelcomeScreen() {
    if (this.welcomeScreen && this.welcomeScreen.parentNode) {
      this.welcomeScreen.remove();
      this.welcomeScreen = null;
    }
  }

  createMessageGroup(type, sender) {
    this.hideWelcomeScreen();
    this.hasMessages = true;
    
    const group = document.createElement('div');
    group.classList.add('message-group');
    
    const icons = this.getIcons();
    const iconMap = {
      user: 'user',
      assistant: 'assistant',
      system: 'system',
      tool: 'tool',
      thinking: 'thinking'
    };
    
    const senderNames = {
      user: '你',
      assistant: 'Assistant',
      system: 'System',
      tool: 'Tool',
      thinking: 'Thinking'
    };
    
    group.innerHTML = `
      <div class="message-header">
        <div class="message-avatar ${type}">${icons[iconMap[type]] || icons.assistant}</div>
        <span class="message-sender">${sender || senderNames[type] || 'Assistant'}</span>
        <span class="message-time">${new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })}</span>
      </div>
    `;
    
    this.container.appendChild(group);
    return group;
  }

  appendUserMessage(messageText, files = null) {
    console.log("appendUserMessage: " + messageText);
    this.hideWelcomeScreen();
    
    const group = this.createMessageGroup('user', '你');
    
    const content = document.createElement('div');
    content.classList.add('message-content', 'user');
    content.textContent = messageText;
    
    if (files && files.length > 0) {
      const fileInfo = document.createElement('div');
      fileInfo.classList.add('file-upload-info');
      
      files.forEach(file => {
        const chip = document.createElement('span');
        chip.classList.add('file-chip');
        chip.innerHTML = `${this.getIcons().file} ${file.name || 'file'}`;
        fileInfo.appendChild(chip);
      });
      
      content.appendChild(fileInfo);
    }
    
    group.appendChild(content);
    this.scrollToBottom();
  }

  appendSystemMessage(messageText) {
    this.hideWelcomeScreen();
    
    const group = this.createMessageGroup('system', 'System');
    
    const content = document.createElement('div');
    content.classList.add('message-content', 'system');
    content.textContent = messageText;
    
    group.appendChild(content);
    this.scrollToBottom();
  }

  appendErrorMessage(messageText) {
    this.hideWelcomeScreen();
    
    const group = this.createMessageGroup('system', 'Error');
    
    const content = document.createElement('div');
    content.classList.add('message-content', 'error');
    content.textContent = messageText;
    
    group.appendChild(content);
    this.scrollToBottom();
  }

  // Tool call display
  appendToolCall(toolName, toolArgs, status = 'running') {
    this.hideWelcomeScreen();
    
    if (!this.currentMessageGroup) {
      this.currentMessageGroup = this.createMessageGroup('tool', toolName);
    }
    
    const toolEl = document.createElement('div');
    toolEl.classList.add('tool-call');
    toolEl.setAttribute('data-tool', toolName);
    
    const statusClass = {
      running: 'running',
      success: 'success',
      error: 'error'
    }[status] || 'running';
    
    const statusText = {
      running: '执行中...',
      success: '完成',
      error: '失败'
    }[status] || '执行中...';
    
    toolEl.innerHTML = `
      <div class="tool-header">
        <span class="tool-icon">${this.getIcons().tool}</span>
        <span class="tool-name">${this.escapeHtml(toolName)}</span>
        <span class="tool-status ${statusClass}">${statusText}</span>
        ${this.getIcons().chevronDown}
      </div>
      <div class="tool-body">
        <div class="tool-section">
          <div class="tool-section-label">参数</div>
          <div class="tool-section-content">${this.escapeHtml(typeof toolArgs === 'string' ? toolArgs : JSON.stringify(toolArgs, null, 2))}</div>
        </div>
      </div>
    `;
    
    // Add click handler for expand/collapse
    const header = toolEl.querySelector('.tool-header');
    header.addEventListener('click', () => {
      toolEl.classList.toggle('expanded');
    });
    
    this.currentMessageGroup.appendChild(toolEl);
    this.scrollToBottom();
    
    return toolEl;
  }

  updateToolStatus(toolEl, status, result = null) {
    if (!toolEl) return;
    
    const statusEl = toolEl.querySelector('.tool-status');
    const bodyEl = toolEl.querySelector('.tool-body');
    
    const statusClass = {
      running: 'running',
      success: 'success',
      error: 'error'
    }[status] || 'running';
    
    const statusText = {
      running: '执行中...',
      success: '完成',
      error: '失败'
    }[status] || '执行中...';
    
    statusEl.className = `tool-status ${statusClass}`;
    statusEl.textContent = statusText;
    
    if (result !== null) {
      const resultSection = document.createElement('div');
      resultSection.classList.add('tool-section');
      resultSection.innerHTML = `
        <div class="tool-section-label">结果</div>
        <div class="tool-section-content">${this.escapeHtml(typeof result === 'string' ? result : JSON.stringify(result, null, 2))}</div>
      `;
      bodyEl.appendChild(resultSection);
    }
  }

  // Thinking block display
  appendThinkingBlock(content) {
    this.hideWelcomeScreen();
    
    if (!this.currentMessageGroup) {
      this.currentMessageGroup = this.createMessageGroup('thinking', 'Thinking');
    }
    
    const thinkingEl = document.createElement('div');
    thinkingEl.classList.add('thinking-block');
    
    thinkingEl.innerHTML = `
      <div class="thinking-header">
        <span class="thinking-icon">${this.getIcons().thinking}</span>
        <span class="thinking-label">思考过程</span>
        ${this.getIcons().chevronDown.replace('tool-expand', 'thinking-expand')}
      </div>
      <div class="thinking-content">${this.escapeHtml(content)}</div>
    `;
    
    // Add click handler for expand/collapse
    const header = thinkingEl.querySelector('.thinking-header');
    header.addEventListener('click', () => {
      thinkingEl.classList.toggle('expanded');
    });
    
    this.currentMessageGroup.appendChild(thinkingEl);
    this.scrollToBottom();
    
    return thinkingEl;
  }

  // Streaming methods
  createNewAccumulatingMessage() {
    if (this.currentMessageGroup && this.accumulatingMessageEl) {
      this.flushAccumulatingMessage();
    }
    
    this.currentMessageGroup = this.createMessageGroup('assistant', 'Assistant');
    
    this.accumulatingMessageEl = document.createElement('div');
    this.accumulatingMessageEl.classList.add('message-content', 'assistant');
    
    this.accumulatingText = '';
    this.isStreaming = true;
    
    this.currentMessageGroup.appendChild(this.accumulatingMessageEl);
  }

  handleNewToken(token) {
    if (!this.accumulatingMessageEl) {
      this.createNewAccumulatingMessage();
    }
    
    this.accumulatingText += token;
    
    // Update display with cursor
    const rendered = this.renderMarkdownSafely(this.accumulatingText);
    this.accumulatingMessageEl.innerHTML = rendered + '<span class="streaming-cursor"></span>';
    
    this.scrollToBottom();
    
    // Check for code blocks to enhance
    if (token.includes('```') || token.includes('\n```\n')) {
      this.enhanceCodeBlocks();
    }
  }

  handleTokensDone() {
    this.flushAccumulatingMessage();
    this.isStreaming = false;
    this.currentMessageGroup = null;
  }

  handleUploads(files) {
    if (!Array.isArray(files) || files.length === 0) return;
    
    const lines = ['文件已保存:'];
    files.forEach(item => {
      const name = item && item.name ? item.name : 'unknown';
      const toolPath = item && item.tool_path ? item.tool_path : '';
      const bytes = item && item.bytes ? item.bytes : 0;
      lines.push(`- ${name} (${toolPath}, ${this.formatBytes(bytes)})`);
    });
    
    this.appendSystemMessage(lines.join('\n'));
  }

  handleError(messageText) {
    this.flushAccumulatingMessage();
    this.appendErrorMessage(messageText || 'Unknown error.');
    this.isStreaming = false;
    this.currentMessageGroup = null;
  }

  flushAccumulatingMessage() {
    if (this.accumulatingMessageEl && this.accumulatingText) {
      // Final markdown render without cursor
      const rendered = this.renderMarkdownSafely(this.accumulatingText);
      this.accumulatingMessageEl.innerHTML = rendered;
      
      // Enhance code blocks
      this.enhanceCodeBlocks();
      
      this.accumulatingMessageEl = null;
      this.accumulatingText = '';
    }
  }

  // Markdown rendering with fallback
  renderMarkdownSafely(fullText) {
    const source = String(fullText || '');
    
    // Try using marked.js if available
    if (typeof globalThis.marked === 'object' && globalThis.marked && typeof globalThis.marked.parse === 'function') {
      try {
        return globalThis.marked.parse(source);
      } catch (_e) {
        // fallback below
      }
    }
    
    // Fallback: basic formatting
    let html = this.escapeHtml(source);
    
    // Code blocks
    html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (match, lang, code) => {
      return `<div class="code-block">
        <div class="code-header">
          <span class="code-language">${lang || 'code'}</span>
          <button class="code-copy-btn" onclick="navigator.clipboard.writeText(this.closest('.code-block').querySelector('pre').textContent)">${this.getIcons().copy} 复制</button>
        </div>
        <div class="code-content"><pre>${code}</pre></div>
      </div>`;
    });
    
    // Inline code
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
    
    // Bold
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    
    // Italic
    html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');
    
    // Line breaks
    html = html.replace(/\n/g, '<br>');
    
    return html;
  }

  enhanceCodeBlocks() {
    if (!this.accumulatingMessageEl) return;
    
    const codeBlocks = this.accumulatingMessageEl.querySelectorAll('pre');
    codeBlocks.forEach(pre => {
      if (pre.closest('.code-block')) return; // Already enhanced
      
      const wrapper = document.createElement('div');
      wrapper.classList.add('code-block');
      
      const header = document.createElement('div');
      header.classList.add('code-header');
      header.innerHTML = `
        <span class="code-language">code</span>
        <button class="code-copy-btn">${this.getIcons().copy} 复制</button>
      `;
      
      const content = document.createElement('div');
      content.classList.add('code-content');
      
      pre.parentNode.insertBefore(wrapper, pre);
      wrapper.appendChild(header);
      wrapper.appendChild(content);
      content.appendChild(pre);
      
      // Add copy functionality
      const copyBtn = header.querySelector('.code-copy-btn');
      copyBtn.addEventListener('click', async () => {
        try {
          await navigator.clipboard.writeText(pre.textContent);
          copyBtn.innerHTML = `${this.getIcons().check} 已复制`;
          setTimeout(() => {
            copyBtn.innerHTML = `${this.getIcons().copy} 复制`;
          }, 2000);
        } catch (e) {
          console.error('Copy failed:', e);
        }
      });
    });
  }

  escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  formatBytes(n) {
    const value = Number(n || 0);
    if (value < 1024) return `${value} B`;
    if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
    return `${(value / (1024 * 1024)).toFixed(2)} MB`;
  }

  scrollToBottom() {
    requestAnimationFrame(() => {
      this.container.scrollTop = this.container.scrollHeight;
    });
  }

  showTypingIndicator() {
    const indicator = document.createElement('div');
    indicator.classList.add('typing-indicator');
    indicator.id = 'typingIndicator';
    indicator.innerHTML = `
      <div class="typing-dots">
        <span class="typing-dot"></span>
        <span class="typing-dot"></span>
        <span class="typing-dot"></span>
      </div>
    `;
    this.container.appendChild(indicator);
    this.scrollToBottom();
  }

  hideTypingIndicator() {
    const indicator = this.container.querySelector('#typingIndicator');
    if (indicator) {
      indicator.remove();
    }
  }
}

customElements.define('messages-area', MessagesArea);
