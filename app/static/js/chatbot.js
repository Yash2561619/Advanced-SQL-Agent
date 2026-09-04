document.addEventListener('DOMContentLoaded', function() {
    // Configure marked for Markdown parsing
    if (typeof marked !== 'undefined') {
        marked.setOptions({
            gfm: true,
            breaks: true
        });
    }

    const chatContainer = document.getElementById('chat-container');
    const chatButton = document.getElementById('chat-button');
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');

    const minimizeBtn = document.getElementById('minimize-btn');
    const maximizeBtn = document.getElementById('maximize-btn');
    const closeBtn = document.getElementById('close-btn');

    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Window controls
    function toggleChat() {
        const isHidden = chatContainer.style.display === 'none';
        chatContainer.style.display = isHidden ? 'flex' : 'none';
        chatButton.style.display = isHidden ? 'none' : 'block';
        if (isHidden) {
            scrollToBottom();
            userInput.focus();
        }
    }

    function minimizeChat() {
        chatContainer.classList.toggle('minimized');
    }

    function maximizeChat() {
        chatContainer.classList.toggle('maximized');
        scrollToBottom();
    }

    function closeChat() {
        chatContainer.style.display = 'none';
        chatButton.style.display = 'block';
    }

    chatButton.addEventListener('click', toggleChat);
    minimizeBtn.addEventListener('click', minimizeChat);
    maximizeBtn.addEventListener('click', maximizeChat);
    closeBtn.addEventListener('click', closeChat);

    // Message rendering
    function addMessage(content, isUser = false) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message', isUser ? 'user-message' : 'bot-message');

        if (isUser) {
            messageElement.textContent = content;
        } else {
            // Converts Markdown (tables, bold, lists) into HTML
            messageElement.innerHTML = typeof marked !== 'undefined' ? marked.parse(content) : content;
        }

        chatMessages.appendChild(messageElement);
        scrollToBottom();
    }

    function addVisualization(imageData, description) {
        const container = document.createElement('div');
        container.classList.add('visualization-container');

        const img = document.createElement('img');
        img.src = `data:image/png;base64,${imageData}`;
        img.alt = description || 'SQL Query Visualization';
        img.classList.add('visualization');

        // Scroll immediately once the image finishes decoding/rendering
        img.onload = () => {
            scrollToBottom();
        };

        container.appendChild(img);

        if (description) {
            const caption = document.createElement('div');
            caption.classList.add('visualization-caption');
            caption.textContent = description;
            container.appendChild(caption);
        }

        chatMessages.appendChild(container);
        scrollToBottom();
    }

    function addLoadingAnimation() {
        const loadingElement = document.createElement('div');
        loadingElement.classList.add('loading');
        loadingElement.innerHTML = `
            <span class="loading-text">Analyzing your data...</span>
            <div class="loading-spinner"></div>
        `;
        chatMessages.appendChild(loadingElement);
        scrollToBottom();
        return loadingElement;
    }

    async function sendMessage() {
        const message = userInput.value.trim();
        if (!message) return;

        // Prevent duplicate submissions while a query is running
        userInput.disabled = true;
        sendButton.disabled = true;

        addMessage(message, true);
        userInput.value = '';

        const loadingElement = addLoadingAnimation();

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: message })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            loadingElement.remove();

            if (data.error) {
                addMessage(`**Error:** ${data.error}`);
            } else {
                addMessage(data.summary || 'No summary available.');

                if (data.visualization && data.visualization.image) {
                    addVisualization(data.visualization.image, data.visualization.description);
                }
            }
        } catch (error) {
            console.error('Error:', error);
            loadingElement.remove();
            addMessage('Sorry, there was an error processing your request. Please try again.');
        } finally {
            // Re-enable inputs
            userInput.disabled = false;
            sendButton.disabled = false;
            userInput.focus();
            scrollToBottom();
        }
    }

    sendButton.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });

    // Initial window state
    chatContainer.style.display = 'flex';
    chatButton.style.display = 'none';
    userInput.focus();
});