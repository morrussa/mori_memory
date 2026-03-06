/**
 * Modern Message Input Component
 * Features: file upload, modern styling, accessibility
 * No build required - Pure Web Components
 */

const messageInputTemplate = document.createElement('template');

const CLIENT_UPLOAD_LIMITS = {
  maxFiles: 8,
  maxFileBytes: 10 * 1024 * 1024,
  maxTotalBytes: 30 * 1024 * 1024,
};

function formatBytes(n) {
  const value = Number(n || 0);
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / (1024 * 1024)).toFixed(2)} MB`;
}

messageInputTemplate.innerHTML = `
<style>
  :host {
    display: flex;
    flex-direction: column;
    padding: var(--space-md) var(--space-lg);
    background: var(--color-surface);
    border-top: 1px solid var(--color-border);
    flex-shrink: 0;
  }

  /* File List */
  .file-list {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-sm);
    margin-bottom: var(--space-sm);
    min-height: 0;
  }

  .file-list:empty {
    display: none;
  }

  .file-chip {
    display: inline-flex;
    align-items: center;
    gap: var(--space-sm);
    padding: var(--space-xs) var(--space-md);
    background: var(--color-bg-tertiary);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-full);
    font-size: var(--font-size-xs);
    color: var(--color-text-secondary);
    animation: fadeIn 0.2s ease;
  }

  .file-chip-icon {
    width: 14px;
    height: 14px;
    color: var(--color-primary);
  }

  .file-chip-name {
    max-width: 150px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .file-chip-size {
    color: var(--color-text-tertiary);
  }

  .file-chip-remove {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 18px;
    height: 18px;
    border: none;
    background: transparent;
    color: var(--color-text-tertiary);
    cursor: pointer;
    border-radius: var(--radius-sm);
    transition: all var(--transition-fast);
  }

  .file-chip-remove:hover {
    background: var(--color-error-light);
    color: var(--color-error);
  }

  .file-chip-remove svg {
    width: 14px;
    height: 14px;
  }

  /* Input Container */
  .input-container {
    display: flex;
    align-items: flex-end;
    gap: var(--space-sm);
    background: var(--color-bg-tertiary);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-xl);
    padding: var(--space-sm);
    transition: border-color var(--transition-fast), box-shadow var(--transition-fast);
  }

  .input-container:focus-within {
    border-color: var(--color-primary);
    box-shadow: 0 0 0 3px var(--color-primary-light);
  }

  .input-container.disabled {
    opacity: 0.6;
    pointer-events: none;
  }

  /* Textarea */
  .message-textarea {
    flex: 1;
    min-height: 24px;
    max-height: 200px;
    padding: var(--space-sm) var(--space-md);
    border: none;
    background: transparent;
    font-family: var(--font-family);
    font-size: var(--font-size-md);
    line-height: var(--line-height-normal);
    color: var(--color-text-primary);
    resize: none;
    outline: none;
  }

  .message-textarea::placeholder {
    color: var(--color-text-tertiary);
  }

  .message-textarea:disabled {
    cursor: not-allowed;
  }

  /* Action Buttons */
  .action-buttons {
    display: flex;
    align-items: center;
    gap: var(--space-xs);
  }

  .action-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 36px;
    height: 36px;
    border: none;
    border-radius: var(--radius-md);
    background: transparent;
    color: var(--color-text-tertiary);
    cursor: pointer;
    transition: all var(--transition-fast);
  }

  .action-btn:hover:not(:disabled) {
    background: var(--color-bg-hover);
    color: var(--color-text-primary);
  }

  .action-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .action-btn svg {
    width: 20px;
    height: 20px;
  }

  .action-btn.primary {
    background: var(--color-primary);
    color: white;
  }

  .action-btn.primary:hover:not(:disabled) {
    background: var(--color-primary-hover);
  }

  .action-btn.primary:disabled {
    background: var(--color-text-tertiary);
  }

  /* Send button with loading state */
  .action-btn.sending {
    position: relative;
  }

  .action-btn.sending svg {
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }

  /* Hidden file input */
  .hidden-file-input {
    display: none;
  }

  /* Character counter */
  .char-counter {
    font-size: var(--font-size-xs);
    color: var(--color-text-tertiary);
    padding: var(--space-xs) var(--space-md);
    text-align: right;
  }

  /* File drop zone overlay */
  .drop-zone {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: var(--color-surface-overlay);
    display: flex;
    align-items: center;
    justify-content: center;
    border: 2px dashed var(--color-primary);
    border-radius: var(--radius-lg);
    opacity: 0;
    pointer-events: none;
    transition: opacity var(--transition-fast);
    z-index: var(--z-modal);
  }

  .drop-zone.active {
    opacity: 1;
    pointer-events: auto;
  }

  .drop-zone-content {
    text-align: center;
  }

  .drop-zone-icon {
    width: 48px;
    height: 48px;
    color: var(--color-primary);
    margin-bottom: var(--space-md);
  }

  .drop-zone-text {
    font-size: var(--font-size-md);
    font-weight: var(--font-weight-medium);
    color: var(--color-text-primary);
  }

  /* Animations */
  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(-5px); }
    to { opacity: 1; transform: translateY(0); }
  }

  /* Responsive */
  @media (max-width: 640px) {
    :host {
      padding: var(--space-sm) var(--space-md);
    }

    .file-chip-name {
      max-width: 100px;
    }
  }
</style>

<div class="file-list" id="fileList"></div>
<div class="input-container" id="inputContainer">
  <button class="action-btn" id="attachBtn" title="上传文件">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"/>
    </svg>
  </button>
  <textarea 
    class="message-textarea" 
    id="messageTextarea" 
    placeholder="输入消息，按 Enter 发送..." 
    rows="1"
  ></textarea>
  <div class="action-buttons">
    <button class="action-btn primary" id="sendBtn" title="发送">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <line x1="22" y1="2" x2="11" y2="13"/>
        <polygon points="22 2 15 22 11 13 2 9 22 2"/>
      </svg>
    </button>
  </div>
</div>
<input type="file" class="hidden-file-input" id="hiddenFileInput" multiple>
`;

class MessageInput extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
    this.shadowRoot.appendChild(messageInputTemplate.content.cloneNode(true));

    // Elements
    this._fileList = this.shadowRoot.querySelector('#fileList');
    this._inputContainer = this.shadowRoot.querySelector('#inputContainer');
    this._textarea = this.shadowRoot.querySelector('#messageTextarea');
    this._attachBtn = this.shadowRoot.querySelector('#attachBtn');
    this._sendBtn = this.shadowRoot.querySelector('#sendBtn');
    this._hiddenFileInput = this.shadowRoot.querySelector('#hiddenFileInput');

    // State
    this._pendingFiles = [];
    this._uploadLimits = { ...CLIENT_UPLOAD_LIMITS };
    this._isSending = false;
    this._isAgentRunning = false;

    // Event listeners
    this._textarea.addEventListener('input', this._handleInput.bind(this));
    this._textarea.addEventListener('keydown', this._handleKeyDown.bind(this));
    this._attachBtn.addEventListener('click', this._handleAttachClick.bind(this));
    this._sendBtn.addEventListener('click', this._handleSendClick.bind(this));
    this._hiddenFileInput.addEventListener('change', this._handleFileSelected.bind(this));
    this._fileList.addEventListener('click', this._handleFileListClick.bind(this));

    // Drag and drop
    this.addEventListener('dragover', this._handleDragOver.bind(this));
    this.addEventListener('dragleave', this._handleDragLeave.bind(this));
    this.addEventListener('drop', this._handleDrop.bind(this));
  }

  connectedCallback() {
    this._textarea.focus();
    this._adjustTextareaHeight();
  }

  init(worker) {
    this.worker = worker;
  }

  setMessagesArea(messagesAreaComponent) {
    this.messagesAreaComponent = messagesAreaComponent;
  }

  setUploadLimits(limits) {
    if (!limits || typeof limits !== 'object') return;

    const maxFiles = Number(limits.max_files || limits.maxFiles || this._uploadLimits.maxFiles);
    const maxFileBytes = Number(limits.max_file_bytes || limits.maxFileBytes || this._uploadLimits.maxFileBytes);
    const maxTotalBytes = Number(limits.max_total_bytes || limits.maxTotalBytes || this._uploadLimits.maxTotalBytes);

    this._uploadLimits = {
      maxFiles: Number.isFinite(maxFiles) && maxFiles > 0 ? Math.floor(maxFiles) : this._uploadLimits.maxFiles,
      maxFileBytes: Number.isFinite(maxFileBytes) && maxFileBytes > 0 ? Math.floor(maxFileBytes) : this._uploadLimits.maxFileBytes,
      maxTotalBytes: Number.isFinite(maxTotalBytes) && maxTotalBytes > 0 ? Math.floor(maxTotalBytes) : this._uploadLimits.maxTotalBytes,
    };
  }

  handleMessageSent() {
    this._isSending = false;
    this._textarea.value = '';
    this._pendingFiles = [];
    this._hiddenFileInput.value = '';
    this._renderFileList();
    this._updateUIState(false);
    this._adjustTextareaHeight();
    this._textarea.focus();
  }

  // Input handling
  _handleInput() {
    this._adjustTextareaHeight();
  }

  _adjustTextareaHeight() {
    this._textarea.style.height = 'auto';
    this._textarea.style.height = Math.min(this._textarea.scrollHeight, 200) + 'px';
  }

  _handleKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      this._sendMessage();
    }
  }

  // File handling
  _handleAttachClick() {
    this._hiddenFileInput.click();
  }

  _handleFileSelected(event) {
    const newFiles = Array.from(event.target.files || []);
    this._hiddenFileInput.value = '';
    this._addFiles(newFiles);
  }

  _handleDragOver(event) {
    event.preventDefault();
    event.stopPropagation();
    this._inputContainer.classList.add('dragover');
  }

  _handleDragLeave(event) {
    event.preventDefault();
    event.stopPropagation();
    this._inputContainer.classList.remove('dragover');
  }

  _handleDrop(event) {
    event.preventDefault();
    event.stopPropagation();
    this._inputContainer.classList.remove('dragover');

    const files = Array.from(event.dataTransfer.files || []);
    if (files.length > 0) {
      this._addFiles(files);
    }
  }

  _handleFileListClick(event) {
    const removeBtn = event.target.closest('.file-chip-remove');
    if (!removeBtn) return;

    const index = parseInt(removeBtn.dataset.index, 10);
    if (Number.isFinite(index) && index >= 0 && index < this._pendingFiles.length) {
      this._pendingFiles.splice(index, 1);
      this._renderFileList();
    }
  }

  _addFiles(newFiles) {
    const merged = [...this._pendingFiles, ...newFiles];
    const error = this._validateFiles(merged);

    if (error) {
      if (this.messagesAreaComponent && typeof this.messagesAreaComponent.handleError === 'function') {
        this.messagesAreaComponent.handleError(error);
      } else {
        console.error(error);
      }
      return;
    }

    this._pendingFiles = merged;
    this._renderFileList();
  }

  _validateFiles(files) {
    const limits = this._uploadLimits;

    if (files.length > limits.maxFiles) {
      return `最多只能上传 ${limits.maxFiles} 个文件`;
    }

    let total = 0;
    for (const file of files) {
      const size = file.size || 0;
      total += size;

      if (size > limits.maxFileBytes) {
        return `文件 "${file.name}" 超过单文件限制 ${formatBytes(limits.maxFileBytes)}`;
      }
    }

    if (total > limits.maxTotalBytes) {
      return `总上传大小超过限制 ${formatBytes(limits.maxTotalBytes)}`;
    }

    return '';
  }

  _renderFileList() {
    this._fileList.innerHTML = '';

    this._pendingFiles.forEach((file, index) => {
      const chip = document.createElement('div');
      chip.classList.add('file-chip');

      chip.innerHTML = `
        <svg class="file-chip-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/>
          <polyline points="14 2 14 8 20 8"/>
        </svg>
        <span class="file-chip-name" title="${this._escapeHtml(file.name)}">${this._escapeHtml(file.name)}</span>
        <span class="file-chip-size">${formatBytes(file.size)}</span>
        <button type="button" class="file-chip-remove" data-index="${index}" title="移除">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <line x1="18" y1="6" x2="6" y2="18"/>
            <line x1="6" y1="6" x2="18" y2="18"/>
          </svg>
        </button>
      `;

      this._fileList.appendChild(chip);
    });
  }

  // Send handling
  _handleSendClick() {
    this._sendMessage();
  }

  _sendMessage() {
    const message = this._textarea.value.trim();
    const files = this._pendingFiles;

    if (!message && files.length === 0) return;
    if (this._isSending) return;

    this._isSending = true;
    this._updateUIState(true);

    // Build display message
    let displayMessage = message;
    if (files.length > 0) {
      if (!message) {
        displayMessage = `[上传了 ${files.length} 个文件]`;
      } else {
        displayMessage = `${message}\n[附带 ${files.length} 个文件]`;
      }
    }

    // Show in messages area
    if (this.messagesAreaComponent) {
      this.messagesAreaComponent.appendUserMessage(displayMessage, files);
    }

    // Send to worker
    this.worker.postMessage({
      type: 'chatMessage',
      message: this._textarea.value,
      files: files,
    });
  }

  _updateUIState(isSending) {
    this._isSending = isSending;
    this._updateUIFromState();
  }

  _updateUIFromState() {
    const isDisabled = this._isSending || this._isAgentRunning;
    console.log('[MessageInput] _updateUIFromState:', { 
      isSending: this._isSending, 
      isAgentRunning: this._isAgentRunning,
      isDisabled 
    });
    
    if (isDisabled) {
      this._inputContainer.classList.add('disabled');
      this._sendBtn.classList.add('sending');
      this._sendBtn.disabled = true;
      this._attachBtn.disabled = true;
      this._textarea.disabled = true;
      this._hiddenFileInput.disabled = true;
    } else {
      this._inputContainer.classList.remove('disabled');
      this._sendBtn.classList.remove('sending');
      this._sendBtn.disabled = false;
      this._attachBtn.disabled = false;
      this._textarea.disabled = false;
      this._hiddenFileInput.disabled = false;
    }
  }

  _escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  // Public methods for external control
  focus() {
    this._textarea.focus();
  }

  clear() {
    this._textarea.value = '';
    this._pendingFiles = [];
    this._renderFileList();
    this._adjustTextareaHeight();
  }

  setAgentRunning(running) {
    console.log('[MessageInput] setAgentRunning:', running);
    this._isAgentRunning = running;
    this._updateUIFromState();
  }
}

customElements.define('message-input', MessageInput);
