/**
 * Our Biggest Deal - MemoRAG Chatbot
 *
 * Simplified chat interface for Q&A about "Our Biggest Deal" book
 * using MemoRAG for intelligent memory-based retrieval.
 */

// API Configuration
const API_BASE_URL = window.location.hostname === 'localhost'
    ? 'http://localhost:8000'
    : 'https://gaiaai.xyz';

const MEMORAG_ENDPOINT = `${API_BASE_URL}/api/memorag/chat`;

// State
let isLoading = false;
let conversationHistory = [];

// DOM Elements
const chatMessages = document.getElementById('chatMessages');
const userInput = document.getElementById('userInput');
const sendButton = document.getElementById('sendButton');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    // Focus input
    userInput.focus();

    // Send on Enter
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Send button click
    sendButton.addEventListener('click', sendMessage);

    // Add welcome message
    addMessage('gaia', `Welcome! I'm here to discuss "Our Biggest Deal" by Aaron William Perry.

Ask me anything about:
- Planetary Prosperity and the Ecocene Era
- Regenerative economics and sustainability
- The book's vision for humanity's future

What would you like to explore?`);
});

/**
 * Send message to MemoRAG API
 */
async function sendMessage() {
    const message = userInput.value.trim();
    if (!message || isLoading) return;

    // Clear input and show user message
    userInput.value = '';
    addMessage('user', message);

    // Show loading state
    isLoading = true;
    sendButton.disabled = true;
    const loadingEl = showLoading();

    try {
        const response = await fetch(MEMORAG_ENDPOINT, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: message,
                max_tokens: 512
            })
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({}));
            throw new Error(error.detail || `Request failed: ${response.status}`);
        }

        const data = await response.json();

        // Remove loading and show response
        loadingEl.remove();
        addMessage('gaia', data.answer, data.query_time_ms, data.shards_queried);

        // Store in history
        conversationHistory.push({
            user: message,
            assistant: data.answer,
            shards_queried: data.shards_queried,
            timestamp: new Date().toISOString()
        });

    } catch (error) {
        console.error('Chat error:', error);
        loadingEl.remove();
        addMessage('error', `Sorry, I encountered an error: ${error.message}. Please try again.`);
    } finally {
        isLoading = false;
        sendButton.disabled = false;
        userInput.focus();
    }
}

/**
 * Add message to chat
 */
function addMessage(type, content, queryTimeMs = null, shardsQueried = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}-message`;

    // Format content with markdown-like processing
    const formattedContent = formatContent(content);

    // Build metadata line (shards + time)
    let metadataHtml = '';
    if (shardsQueried && shardsQueried.length > 0) {
        metadataHtml += `<span class="shards-info">Shards: [${shardsQueried.join(', ')}]</span>`;
    }
    if (queryTimeMs) {
        metadataHtml += `<span class="query-time">${queryTimeMs}ms</span>`;
    }

    let html = '';
    if (type === 'gaia') {
        html = `
            <div class="message-avatar">
                <span class="avatar-icon">🌍</span>
            </div>
            <div class="message-content">
                <div class="message-header">
                    <span class="message-author">Gaia</span>
                    ${metadataHtml}
                </div>
                <div class="message-text">${formattedContent}</div>
                <div class="message-source">Source: Our Biggest Deal (MemoRAG)</div>
            </div>
        `;
    } else if (type === 'user') {
        html = `
            <div class="message-content">
                <div class="message-text">${escapeHtml(content)}</div>
            </div>
            <div class="message-avatar">
                <span class="avatar-icon">👤</span>
            </div>
        `;
    } else if (type === 'error') {
        html = `
            <div class="message-avatar">
                <span class="avatar-icon">⚠️</span>
            </div>
            <div class="message-content">
                <div class="message-text error-text">${escapeHtml(content)}</div>
            </div>
        `;
    }

    messageDiv.innerHTML = html;
    chatMessages.appendChild(messageDiv);

    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

/**
 * Show loading indicator
 */
function showLoading() {
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'message gaia-message loading-message';
    loadingDiv.innerHTML = `
        <div class="message-avatar">
            <span class="avatar-icon">🌍</span>
        </div>
        <div class="message-content">
            <div class="loading-dots">
                <span></span><span></span><span></span>
            </div>
            <div class="loading-text">Searching through memory...</div>
        </div>
    `;
    chatMessages.appendChild(loadingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return loadingDiv;
}

/**
 * Format content with basic markdown
 */
function formatContent(content) {
    // Escape HTML first
    let html = escapeHtml(content);

    // Convert line breaks to paragraphs
    html = html.split('\n\n').map(p => `<p>${p}</p>`).join('');

    // Convert single line breaks within paragraphs
    html = html.replace(/([^>])\n([^<])/g, '$1<br>$2');

    // Bold text
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

    // Italic text
    html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');

    // Lists (simple)
    html = html.replace(/^- (.+)$/gm, '<li>$1</li>');
    html = html.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');

    return html;
}

/**
 * Escape HTML to prevent XSS
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Clear conversation
 */
function clearConversation() {
    chatMessages.innerHTML = '';
    conversationHistory = [];
    addMessage('gaia', `Conversation cleared. What would you like to explore about "Our Biggest Deal"?`);
}

// Export for potential external use
window.OBDChat = {
    sendMessage,
    clearConversation,
    getHistory: () => conversationHistory
};
