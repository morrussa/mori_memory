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
            align-items: stretch;
            margin: var(--margin);
            margin-bottom: 20px;
            gap: 8px;
        }
        #inputRow {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 8px;
        }
        #messageInputField {
            flex-grow: 1;
            padding: 10px;
            font-size: 1rem;
            border: 1px solid lightgrey;
            border-radius: 10px;
            outline: none;
        }
        #messageInputField:focus {
            border-color: darkgrey;
        }
        #attachButton {
            width: 44px;
            height: 36px;
            cursor: pointer;
            border-radius: 10px;
            border: 2px solid darkgrey;
            background: #fff;
        }
        #sendButton {
            width: 44px;
            height: 36px;
            cursor: pointer; 
            border-radius: 10px;
            border: 2px solid darkgrey;
        }
        #fileInfo {
            min-height: 16px;
            font-size: 0.85rem;
            color: #555;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        #fileList {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            min-height: 20px;
        }
        .fileTag {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            border: 1px solid #ddd;
            border-radius: 12px;
            padding: 2px 8px;
            font-size: 0.75rem;
            background: #fafafa;
            max-width: 100%;
        }
        .fileName {
            max-width: 220px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .removeFileBtn {
            border: none;
            background: transparent;
            color: #666;
            cursor: pointer;
            font-size: 0.8rem;
            line-height: 1;
            padding: 0;
        }
        #hiddenFileInput {
            display: none;
        }
    </style>
    <div id="inputRow">
        <input type="text" id="messageInputField" placeholder="" focus="true" autocomplete="off">
        <button id="attachButton" title="Attach files">+</button>
        <button id="sendButton"><span class="" height="24" width="24" data-state="closed"><svg width="24" height="24" viewBox="0 0 24 24" fill="none" class="text-white dark:text-black"><path d="M7 11L12 6L17 11M12 18V7" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"></path></svg></span></button>
    </div>
    <div id="fileInfo"></div>
    <div id="fileList"></div>
    <input id="hiddenFileInput" type="file" multiple>
`;

// stop button <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor" class="h-2 w-2 text-gizmo-gray-950 dark:text-gray-200" height="16" width="16"><path d="M0 2a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2H2a2 2 0 0 1-2-2V2z" stroke-width="0"></path></svg>
class MessageInput extends HTMLElement {
    constructor() {
        super();
        const shadowRoot = this.attachShadow({mode: 'open'});
        shadowRoot.appendChild(messageInputTemplate.content.cloneNode(true));

        this._messageInputField = shadowRoot.querySelector('#messageInputField');
        this._sendButton = shadowRoot.querySelector('#sendButton');
        this._attachButton = shadowRoot.querySelector('#attachButton');
        this._fileInfo = shadowRoot.querySelector('#fileInfo');
        this._fileList = shadowRoot.querySelector('#fileList');
        this._hiddenFileInput = shadowRoot.querySelector('#hiddenFileInput');
        this._pendingFiles = [];
        this._uploadLimits = {
            maxFiles: CLIENT_UPLOAD_LIMITS.maxFiles,
            maxFileBytes: CLIENT_UPLOAD_LIMITS.maxFileBytes,
            maxTotalBytes: CLIENT_UPLOAD_LIMITS.maxTotalBytes,
        };

        this._messageInputField.addEventListener('keydown', this._handleKeyDown.bind(this));
        this._sendButton.addEventListener('click', this._handleClick.bind(this));
        this._attachButton.addEventListener('click', this._handleAttachClick.bind(this));
        this._hiddenFileInput.addEventListener('change', this._handleFileSelected.bind(this));
        this._fileList.addEventListener('click', this._handleFileListClick.bind(this));
    }

    connectedCallback() {
        // Set focus to the input field when the element is added to the DOM
        this._messageInputField.focus();
    }

    init(worker) {
        this.worker = worker;
    }

    setMessagesArea(messagesAreaComponent) {
        this.messagesAreaComponent = messagesAreaComponent;
    }

    setUploadLimits(limits) {
        if (!limits || typeof limits !== 'object') {
            return;
        }
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
        console.log("handleMessageSent");
        this._messageInputField.value = '';
        this._sendButton.removeAttribute('disabled');
        this._attachButton.removeAttribute('disabled');
        this._messageInputField.removeAttribute('disabled');
        this._hiddenFileInput.removeAttribute('disabled');
        this._pendingFiles = [];
        this._hiddenFileInput.value = '';
        this._fileInfo.textContent = '';
        this._renderPendingFiles();
    }

    _handleKeyDown(event) {
        if (event.key === 'Enter') {
            this._handleNewChatMessage();
        }
    }

    _handleClick() {
        this._handleNewChatMessage();
    }

    _handleAttachClick() {
        this._hiddenFileInput.click();
    }

    _validatePendingFiles(files) {
        const limits = this._uploadLimits || CLIENT_UPLOAD_LIMITS;
        if (files.length > limits.maxFiles) {
            return `最多只能上传 ${limits.maxFiles} 个文件。`;
        }
        let total = 0;
        for (const f of files) {
            const size = Number(f && f.size ? f.size : 0);
            total += size;
            if (size > limits.maxFileBytes) {
                return `文件 ${f.name || 'unknown'} 超过单文件限制 ${formatBytes(limits.maxFileBytes)}。`;
            }
        }
        if (total > limits.maxTotalBytes) {
            return `总上传大小超过限制 ${formatBytes(limits.maxTotalBytes)}。`;
        }
        return '';
    }

    _renderPendingFiles() {
        this._fileList.innerHTML = '';
        const files = this._pendingFiles || [];
        if (files.length === 0) {
            this._fileInfo.textContent = '';
            return;
        }

        let total = 0;
        for (let i = 0; i < files.length; i += 1) {
            const f = files[i];
            const size = Number(f && f.size ? f.size : 0);
            total += size;
            const tag = document.createElement('span');
            tag.classList.add('fileTag');

            const nameEl = document.createElement('span');
            nameEl.classList.add('fileName');
            nameEl.textContent = `${f.name || 'upload.bin'} (${formatBytes(size)})`;

            const removeBtn = document.createElement('button');
            removeBtn.classList.add('removeFileBtn');
            removeBtn.setAttribute('type', 'button');
            removeBtn.setAttribute('data-index', String(i));
            removeBtn.title = 'Remove';
            removeBtn.textContent = 'x';

            tag.appendChild(nameEl);
            tag.appendChild(removeBtn);
            this._fileList.appendChild(tag);
        }

        this._fileInfo.textContent = `Attached ${files.length} file(s), total ${formatBytes(total)}`;
    }

    _handleFileListClick(event) {
        const target = event.target;
        if (!target || !(target instanceof HTMLElement)) {
            return;
        }
        if (!target.classList.contains('removeFileBtn')) {
            return;
        }
        const idx = Number(target.getAttribute('data-index'));
        if (!Number.isFinite(idx) || idx < 0 || idx >= this._pendingFiles.length) {
            return;
        }
        this._pendingFiles.splice(idx, 1);
        this._renderPendingFiles();
    }

    _handleFileSelected(event) {
        const newFiles = Array.from((event.target && event.target.files) || []);
        this._hiddenFileInput.value = '';
        if (newFiles.length === 0) {
            return;
        }

        const merged = this._pendingFiles.concat(newFiles);
        const err = this._validatePendingFiles(merged);
        if (err) {
            if (this.messagesAreaComponent && typeof this.messagesAreaComponent.handleError === 'function') {
                this.messagesAreaComponent.handleError(err);
            } else {
                console.error(err);
            }
            return;
        }
        this._pendingFiles = merged;
        this._renderPendingFiles();
    }

    _handleNewChatMessage() {
        let messageContent = this._messageInputField.value;
        const trimmed = String(messageContent || '').trim();
        const files = this._pendingFiles || [];
        if (!trimmed && files.length === 0) {
            return;
        }

        // prevent user from interacting while we're waiting
        this._sendButton.setAttribute('disabled', 'disabled');
        this._attachButton.setAttribute('disabled', 'disabled');
        this._messageInputField.setAttribute('disabled', 'disabled');
        this._hiddenFileInput.setAttribute('disabled', 'disabled');

        let userDisplay = messageContent;
        if (!trimmed && files.length > 0) {
            userDisplay = `[Uploaded ${files.length} file(s)]`;
        } else if (trimmed && files.length > 0) {
            userDisplay = `${messageContent}\n[Attached ${files.length} file(s)]`;
        }
        if (this.messagesAreaComponent) {
            this.messagesAreaComponent.appendUserMessage(userDisplay);
        }
        this.worker.postMessage({ type: 'chatMessage', message: messageContent, files: files });
    }
}
customElements.define('message-input', MessageInput);
