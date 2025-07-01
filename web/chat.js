/**
 * YonEarth Gaia Chat Interface JavaScript
 */

class GaiaChat {
    constructor() {
        this.apiUrl = localStorage.getItem('gaiaApiUrl') || 'http://localhost:8000';
        this.sessionId = this.generateSessionId();
        this.isLoading = false;
        
        this.initializeElements();
        this.setupEventListeners();
        this.checkApiConnection();
    }
    
    initializeElements() {
        this.chatMessages = document.getElementById('chatMessages');
        this.messageInput = document.getElementById('messageInput');
        this.sendButton = document.getElementById('sendButton');
        this.chatForm = document.getElementById('chatForm');
        this.loading = document.getElementById('loading');
        this.personalitySelect = document.getElementById('personality');
        this.recommendations = document.getElementById('recommendations');
        this.recommendationsList = document.getElementById('recommendationsList');
        this.status = document.getElementById('status');
    }
    
    setupEventListeners() {
        // Chat form submission
        this.chatForm.addEventListener('submit', (e) => {
            e.preventDefault();
            this.sendMessage();
        });
        
        // Enter key handling (with Shift+Enter for new line)
        this.messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        // Auto-resize input
        this.messageInput.addEventListener('input', () => {
            this.autoResizeInput();
        });
        
        // Personality change
        this.personalitySelect.addEventListener('change', () => {
            this.showPersonalityChange();
        });
        
        // Configuration modal (Ctrl+Shift+C)
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.shiftKey && e.key === 'C') {
                this.showConfigModal();
            }
        });
    }
    
    generateSessionId() {
        return 'session_' + Math.random().toString(36).substr(2, 9) + '_' + Date.now();
    }
    
    autoResizeInput() {
        this.messageInput.style.height = 'auto';
        this.messageInput.style.height = Math.min(this.messageInput.scrollHeight, 120) + 'px';
    }
    
    async checkApiConnection() {
        try {
            const response = await fetch(`${this.apiUrl}/health`);
            if (response.ok) {
                const data = await response.json();
                this.updateStatus('connected', `Connected - ${data.status}`);
            } else {
                this.updateStatus('disconnected', 'API Error');
            }
        } catch (error) {
            this.updateStatus('disconnected', 'Disconnected');
            console.error('API connection failed:', error);
        }
    }
    
    updateStatus(status, text) {
        this.status.textContent = text;
        this.status.className = status;
    }
    
    async sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message || this.isLoading) return;
        
        // Add user message to chat
        this.addMessage(message, 'user');
        
        // Clear input and reset height
        this.messageInput.value = '';
        this.messageInput.style.height = 'auto';
        
        // Show loading
        this.setLoading(true);
        
        try {
            const response = await this.callChatAPI(message);
            
            if (response.error) {
                throw new Error(response.error);
            }
            
            // Add Gaia's response
            this.addMessage(response.response, 'gaia', response.citations);
            
            // Show episode recommendations if available
            if (response.citations && response.citations.length > 0) {
                this.showRecommendations(response.citations);
            }
            
        } catch (error) {
            console.error('Chat error:', error);
            this.addMessage(
                "I apologize, dear one, but I'm having trouble connecting with the Earth's wisdom right now. Please check your connection and try again.",
                'gaia',
                [],
                true
            );
        } finally {
            this.setLoading(false);
        }
    }
    
    async callChatAPI(message) {
        const requestBody = {
            message: message,
            session_id: this.sessionId,
            personality: this.personalitySelect.value,
            max_results: 5
        };
        
        const response = await fetch(`${this.apiUrl}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestBody)
        });
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
        }
        
        return await response.json();
    }
    
    addMessage(text, sender, citations = [], isError = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        
        // Avatar
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.textContent = sender === 'gaia' ? '🌱' : '👤';
        
        // Content container
        const content = document.createElement('div');
        content.className = 'message-content';
        
        // Message text
        const messageText = document.createElement('div');
        messageText.className = 'message-text';
        if (isError) {
            messageText.style.background = 'linear-gradient(135deg, #ffebee, #ffcdd2)';
            messageText.style.borderLeft = '3px solid #f44336';
        }
        messageText.textContent = text;
        
        content.appendChild(messageText);
        
        // Add citations if available
        if (citations && citations.length > 0) {
            const citationsDiv = this.createCitationsDiv(citations);
            content.appendChild(citationsDiv);
        }
        
        messageDiv.appendChild(avatar);
        messageDiv.appendChild(content);
        
        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
    }
    
    createCitationsDiv(citations) {
        const citationsDiv = document.createElement('div');
        citationsDiv.className = 'citations';
        
        const title = document.createElement('div');
        title.className = 'citations-title';
        title.textContent = 'Referenced Episodes:';
        citationsDiv.appendChild(title);
        
        citations.forEach(citation => {
            const citationDiv = document.createElement('div');
            citationDiv.className = 'citation';
            
            const titleDiv = document.createElement('div');
            titleDiv.className = 'citation-title';
            titleDiv.textContent = `Episode ${citation.episode_number}: ${citation.title}`;
            
            const guestDiv = document.createElement('div');
            guestDiv.className = 'citation-guest';
            guestDiv.textContent = `with ${citation.guest_name}`;
            
            citationDiv.appendChild(titleDiv);
            citationDiv.appendChild(guestDiv);
            
            if (citation.url) {
                const linkDiv = document.createElement('div');
                const link = document.createElement('a');
                link.href = citation.url;
                link.target = '_blank';
                link.className = 'citation-link';
                link.textContent = 'Listen to Episode →';
                linkDiv.appendChild(link);
                citationDiv.appendChild(linkDiv);
            }
            
            citationsDiv.appendChild(citationDiv);
        });
        
        return citationsDiv;
    }
    
    showRecommendations(citations) {
        if (citations.length === 0) {
            this.recommendations.style.display = 'none';
            return;
        }
        
        this.recommendationsList.innerHTML = '';
        
        citations.slice(0, 3).forEach(citation => {
            const recDiv = document.createElement('div');
            recDiv.className = 'recommendation';
            
            const titleDiv = document.createElement('div');
            titleDiv.className = 'recommendation-title';
            titleDiv.textContent = `Episode ${citation.episode_number}: ${citation.title}`;
            
            const guestDiv = document.createElement('div');
            guestDiv.className = 'recommendation-guest';
            guestDiv.textContent = `with ${citation.guest_name}`;
            
            recDiv.appendChild(titleDiv);
            recDiv.appendChild(guestDiv);
            
            if (citation.url) {
                const linkDiv = document.createElement('div');
                const link = document.createElement('a');
                link.href = citation.url;
                link.target = '_blank';
                link.className = 'recommendation-link';
                link.textContent = 'Listen Now →';
                linkDiv.appendChild(link);
                recDiv.appendChild(linkDiv);
            }
            
            this.recommendationsList.appendChild(recDiv);
        });
        
        this.recommendations.style.display = 'block';
    }
    
    setLoading(isLoading) {
        this.isLoading = isLoading;
        this.loading.style.display = isLoading ? 'flex' : 'none';
        this.sendButton.disabled = isLoading;
        this.messageInput.disabled = isLoading;
        
        if (isLoading) {
            this.scrollToBottom();
        }
    }
    
    scrollToBottom() {
        setTimeout(() => {
            this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
        }, 100);
    }
    
    showPersonalityChange() {
        const personality = this.personalitySelect.value;
        const personalityNames = {
            'warm_mother': 'Nurturing Mother',
            'wise_guide': 'Ancient Sage',
            'earth_activist': 'Earth Guardian'
        };
        
        this.addMessage(
            `I have shifted into my ${personalityNames[personality]} aspect. How may I guide you on your journey with the Earth?`,
            'gaia'
        );
    }
    
    showConfigModal() {
        // Create modal if it doesn't exist
        let modal = document.getElementById('configModal');
        if (!modal) {
            modal = this.createConfigModal();
            document.body.appendChild(modal);
        }
        
        // Set current API URL
        document.getElementById('apiUrl').value = this.apiUrl;
        modal.style.display = 'flex';
    }
    
    createConfigModal() {
        const modal = document.createElement('div');
        modal.className = 'modal';
        modal.id = 'configModal';
        
        modal.innerHTML = `
            <div class="modal-content">
                <h3>API Configuration</h3>
                <form id="configForm">
                    <div class="form-group">
                        <label for="apiUrl">API URL:</label>
                        <input type="url" id="apiUrl" value="${this.apiUrl}" required>
                    </div>
                    <div class="form-actions">
                        <button type="button" onclick="closeConfig()">Cancel</button>
                        <button type="submit">Save</button>
                    </div>
                </form>
            </div>
        `;
        
        // Setup form handler
        modal.querySelector('#configForm').addEventListener('submit', (e) => {
            e.preventDefault();
            const newUrl = document.getElementById('apiUrl').value.trim();
            if (newUrl) {
                this.apiUrl = newUrl;
                localStorage.setItem('gaiaApiUrl', newUrl);
                this.checkApiConnection();
                modal.style.display = 'none';
            }
        });
        
        // Close on backdrop click
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.style.display = 'none';
            }
        });
        
        return modal;
    }
}

// Global function for modal
function closeConfig() {
    document.getElementById('configModal').style.display = 'none';
}

// Initialize chat when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.gaiaChat = new GaiaChat();
});

// Add some helpful console messages
console.log('🌍 YonEarth Gaia Chat Interface Loaded');
console.log('💡 Press Ctrl+Shift+C to configure API URL');
console.log('🌱 Ask Gaia about regenerative practices, sustainability, and Earth wisdom!');