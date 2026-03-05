// The controller wires up all the components and workers together, 
// managing the dependencies. A kind of "DI" class. 
const worker = new Worker('model-worker.js');

const messagesArea = document.querySelector('messages-area');
messagesArea.init(worker);

// Initialize the messageInput component and pass the worker to it
const messageInput = document.querySelector('message-input');
messageInput.init(worker);
messageInput.setMessagesArea(messagesArea);

worker.postMessage({ type: 'init' });
if (typeof messagesArea.appendSystemMessage === 'function') {
    messagesArea.appendSystemMessage("Connected to local Mori backend.");
} else {
    messagesArea.appendUserMessage("Connected to local Mori backend.", "System");
}

fetch('/mori/session/status')
    .then((resp) => resp.ok ? resp.json() : null)
    .then((status) => {
        if (!status || typeof status !== 'object') {
            return;
        }
        if (typeof messageInput.setUploadLimits === 'function' && status.upload_limits) {
            messageInput.setUploadLimits(status.upload_limits);
        }
        if (typeof messagesArea.appendSystemMessage === 'function' && status.upload_dir) {
            messagesArea.appendSystemMessage(`Upload dir: ${status.upload_dir}`);
        }
    })
    .catch((_e) => {
        // ignore bootstrap status fetch failures
    });


// Event listeners for worker messages
// TODO I'm sure there's a better way to do this
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
            break;
        case 'error':
            messagesArea.handleError(payload && payload.message ? payload.message : 'Unknown worker error.');
            break;
        case 'uploads':
            if (typeof messagesArea.handleUploads === 'function') {
                messagesArea.handleUploads(payload && payload.files ? payload.files : []);
            }
            break;
        default:
            console.error('Unknown event type from worker:', type);
    }
};
