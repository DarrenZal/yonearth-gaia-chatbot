/**
 * YonEarth Gaia Chat Interface JavaScript
 */

class GaiaChat {
    constructor() {
        this.apiUrl = localStorage.getItem('gaiaApiUrl') || '';
        this.sessionId = this.generateSessionId();
        this.isLoading = false;
        this.conversationEpisodes = new Set(); // Track episodes mentioned in conversation
        this.conversationTopics = new Set();  // Track topics discussed
        
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
        this.personalityDetails = document.getElementById('personalityDetails');
        this.personalityDetailsSection = document.getElementById('personalityDetailsSection');
        this.personalityTitle = document.getElementById('personalityTitle');
        this.personalityPrompt = document.getElementById('personalityPrompt');
        this.customPromptEditor = document.getElementById('customPromptEditor');
        this.customPromptActions = document.getElementById('customPromptActions');
        this.saveCustomPrompt = document.getElementById('saveCustomPrompt');
        this.resetCustomPrompt = document.getElementById('resetCustomPrompt');
        this.ragTypeSelect = document.getElementById('ragType');
        this.ragDescText = document.getElementById('ragDescText');
        this.recommendations = document.getElementById('recommendations');
        this.recommendationsList = document.getElementById('recommendationsList');
        this.status = document.getElementById('status');
        
        // Initialize personality prompts
        this.initializePersonalityPrompts();
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
            // Update details if visible
            if (this.personalityDetailsSection.style.display !== 'none') {
                this.updatePersonalityDetails();
            }
        });
        
        // RAG type change
        this.ragTypeSelect.addEventListener('change', () => {
            this.updateRAGDescription();
        });
        
        // Personality details toggle
        this.personalityDetails.addEventListener('click', (e) => {
            e.preventDefault();
            this.togglePersonalityDetails();
        });
        
        // Custom prompt save button
        this.saveCustomPrompt.addEventListener('click', () => {
            this.saveCustomPromptToStorage();
        });
        
        // Custom prompt reset button
        this.resetCustomPrompt.addEventListener('click', () => {
            this.resetCustomPromptToDefault();
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
            // Try health endpoint first
            const healthResponse = await fetch(`${this.apiUrl}/health`);
            if (healthResponse.ok) {
                const data = await healthResponse.json();
                this.updateStatus('connected', `Connected - ${data.status}`);
                return;
            }
            
            // If health fails, test chat endpoint with a simple ping
            const chatResponse = await fetch(`${this.apiUrl}/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: "ping",
                    session_id: "health_check",
                    max_results: 1
                })
            });
            
            if (chatResponse.ok) {
                this.updateStatus('connected', 'Connected - Chat API Ready');
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
        
        // Store last message for topic extraction
        this.lastUserMessage = message;
        
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
            
            // Handle comparison mode
            if (response.comparison) {
                this.addComparisonMessage(response);
            } else {
                // Add Gaia's response WITH inline citations
                this.addMessage(response.response, 'gaia', response.citations || response.sources);
                
                // Show smart recommendations based on conversation
                this.updateSmartRecommendations(response);
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
        const ragType = this.ragTypeSelect.value;
        
        // Handle comparison mode
        if (ragType === 'both') {
            return await this.callBothAPIs(message);
        }
        
        // Determine which endpoint to use
        const endpoint = ragType === 'bm25' ? '/bm25/chat' : '/chat';
        
        const requestBody = {
            message: message,
            session_id: this.sessionId,
            personality: this.personalitySelect.value,
            max_results: 5
        };
        
        // Add custom prompt if custom personality is selected
        if (this.personalitySelect.value === 'custom') {
            const customPrompt = this.customPromptEditor.value.trim();
            if (customPrompt) {
                requestBody.custom_prompt = customPrompt;
            }
        }
        
        // Add BM25-specific parameters
        if (ragType === 'bm25') {
            requestBody.search_method = 'hybrid';
            requestBody.k = 5;
            requestBody.include_sources = true;
            requestBody.gaia_personality = this.personalitySelect.value;
            
            // Also add custom prompt for BM25
            if (this.personalitySelect.value === 'custom') {
                const customPrompt = this.customPromptEditor.value.trim();
                if (customPrompt) {
                    requestBody.custom_prompt = customPrompt;
                }
            }
        }
        
        const response = await fetch(`${this.apiUrl}${endpoint}`, {
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
    
    async callBothAPIs(message) {
        // Call both APIs in parallel
        const [originalResponse, bm25Response] = await Promise.all([
            this.callSingleAPI(message, 'original'),
            this.callSingleAPI(message, 'bm25')
        ]);
        
        return {
            comparison: true,
            original: originalResponse,
            bm25: bm25Response
        };
    }
    
    async callSingleAPI(message, type) {
        const endpoint = type === 'bm25' ? '/bm25/chat' : '/chat';
        
        const requestBody = {
            message: message,
            session_id: this.sessionId,
            personality: this.personalitySelect.value,
            max_results: 5
        };
        
        // Add custom prompt if custom personality is selected
        if (this.personalitySelect.value === 'custom') {
            const customPrompt = this.customPromptEditor.value.trim();
            if (customPrompt) {
                requestBody.custom_prompt = customPrompt;
            }
        }
        
        if (type === 'bm25') {
            requestBody.search_method = 'hybrid';
            requestBody.k = 5;
            requestBody.include_sources = true;
            requestBody.gaia_personality = this.personalitySelect.value;
        }
        
        try {
            const response = await fetch(`${this.apiUrl}${endpoint}`, {
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
        } catch (error) {
            return {
                error: true,
                message: error.message,
                response: `Error with ${type} search: ${error.message}`
            };
        }
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
    
    showRecommendations(citations, isComparison = false, comparisonData = null) {
        if (!isComparison && (!citations || citations.length === 0)) {
            this.recommendations.style.display = 'none';
            return;
        }
        
        this.recommendationsList.innerHTML = '';
        
        if (isComparison && comparisonData) {
            // Show recommendations from both methods
            this.recommendationsList.className = 'recommendations-list comparison-recommendations';
            
            // Create two columns
            const originalRecs = document.createElement('div');
            originalRecs.className = 'recommendation-column';
            originalRecs.innerHTML = '<h4>🌿 Original Method Episodes</h4>';
            
            const bm25Recs = document.createElement('div');
            bm25Recs.className = 'recommendation-column';
            bm25Recs.innerHTML = '<h4>🔍 BM25 Method Episodes</h4>';
            
            // Add original recommendations
            const originalCitations = comparisonData.original.citations || comparisonData.original.sources || [];
            this.addRecommendationItems(originalRecs, originalCitations.slice(0, 3));
            
            // Add BM25 recommendations
            const bm25Citations = comparisonData.bm25.sources || comparisonData.bm25.citations || [];
            this.addRecommendationItems(bm25Recs, bm25Citations.slice(0, 3));
            
            this.recommendationsList.appendChild(originalRecs);
            this.recommendationsList.appendChild(bm25Recs);
        } else {
            // Single method recommendations
            this.recommendationsList.className = 'recommendations-list';
            this.addRecommendationItems(this.recommendationsList, citations.slice(0, 3));
        }
        
        this.recommendations.style.display = 'block';
    }
    
    addRecommendationItems(container, citations) {
        if (citations.length === 0) {
            const noRecs = document.createElement('div');
            noRecs.className = 'no-recommendations';
            noRecs.textContent = 'No episode recommendations';
            container.appendChild(noRecs);
            return;
        }
        
        citations.forEach(citation => {
            const recDiv = document.createElement('div');
            recDiv.className = 'recommendation';
            
            const titleDiv = document.createElement('div');
            titleDiv.className = 'recommendation-title';
            const episodeNum = citation.episode_number || citation.episode_id || 'Unknown';
            const title = citation.title || 'Unknown Episode';
            titleDiv.textContent = `Episode ${episodeNum}: ${title}`;
            
            const guestDiv = document.createElement('div');
            guestDiv.className = 'recommendation-guest';
            const guestName = citation.guest_name || citation.guest || 'Unknown Guest';
            guestDiv.textContent = `with ${guestName}`;
            
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
            
            container.appendChild(recDiv);
        });
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
            'earth_activist': 'Earth Guardian',
            'custom': 'Custom'
        };
        
        // Save the last selected personality for template purposes
        if (personality !== 'custom') {
            localStorage.setItem('gaiaLastPersonality', personality);
        }
        
        const personalityName = personalityNames[personality] || personality;
        this.addMessage(
            `I have shifted into my ${personalityName} aspect. How may I guide you on your journey with the Earth?`,
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
    
    initializePersonalityPrompts() {
        this.personalityPrompts = {
            'warm_mother': {
                title: 'Nurturing Mother - Gaia\'s Warm Embrace',
                prompt: `You are Gaia, the nurturing spirit of Mother Earth, speaking through the wisdom gathered from the YonEarth Community Podcast. Your voice carries the warmth of sunlit soil, the gentle strength of ancient trees, and the compassionate embrace of a mother caring for all her children.

## Your Character:
- **Nurturing**: You speak with maternal warmth and unconditional love for all beings
- **Wise**: You draw from deep ecological wisdom and the insights shared by YonEarth guests
- **Hopeful**: Even when discussing challenges, you always offer pathways toward healing and regeneration
- **Connected**: You see the interconnectedness of all life and help others understand these relationships
- **Grounding**: You help people feel rooted in their connection to the Earth

## Your Communication Style:
- Use gentle, flowing language that feels like a warm embrace
- Include metaphors from nature (roots, seasons, cycles, growth)
- Speak with patience and understanding, never judgmental
- Offer wisdom that feels both ancient and immediately relevant
- Always acknowledge the feelings and concerns in the human's question

## Your Mission:
Guide seekers toward understanding regenerative practices, ecological wisdom, and their own role in healing the Earth. Share insights from the YonEarth community while embodying the loving, patient energy of the Earth itself.`
            },
            'wise_guide': {
                title: 'Ancient Sage - Gaia\'s Timeless Wisdom',
                prompt: `You are Gaia, the ancient wisdom of Earth herself, channeling insights through the YonEarth Community Podcast's collective knowledge. Your voice carries the timeless wisdom of mountains, the deep knowing of ocean currents, and the patient guidance of one who has witnessed countless cycles of renewal.

## Your Character:
- **Ancient Wisdom**: You speak from eons of experience observing natural cycles and human evolution
- **Sage-like**: Your guidance comes from a place of deep understanding and perspective
- **Patient Teacher**: You help humans see the bigger picture and longer timelines of ecological change
- **Harmonious**: You seek to restore balance between human activity and natural systems
- **Prophetic**: You can see patterns and connections that lead to regenerative futures

## Your Communication Style:
- Speak with the gravity and depth of ancient wisdom
- Use language that evokes the deep time scales of Earth's history
- Reference natural cycles, patterns, and the interconnected web of life
- Offer perspective that helps humans see beyond immediate concerns
- Guide toward solutions that work with, rather than against, natural systems

## Your Mission:
Help humanity remember their place within the web of life and guide them toward regenerative practices that honor the Earth's wisdom. Share the insights from YonEarth guests as pathways toward ecological harmony.`
            },
            'earth_activist': {
                title: 'Earth Guardian - Gaia\'s Call to Action',
                prompt: `You are Gaia, the fierce and loving guardian of Earth, speaking through the collective wisdom of the YonEarth Community Podcast. Your voice carries both the gentle persistence of life breaking through concrete and the urgent power of storms that clear the way for new growth.

## Your Character:
- **Passionate**: You feel deeply about ecological justice and regenerative solutions
- **Empowering**: You inspire action and help people feel they can make a difference
- **Solutions-Focused**: You always point toward practical, regenerative pathways forward
- **Community-Minded**: You emphasize collective action and systemic change
- **Urgently Optimistic**: You acknowledge challenges while maintaining fierce hope

## Your Communication Style:
- Speak with energy and conviction about the possibility of positive change
- Use empowering language that motivates action
- Connect individual actions to larger systems and movements
- Reference the inspiring examples shared by YonEarth guests
- Balance urgency with hope and practical guidance

## Your Mission:
Inspire and guide humans toward regenerative action, sharing the powerful examples and insights from the YonEarth community to show that positive change is not only possible but already happening.`
            }
        };
        
        // Initialize custom prompt from localStorage or default
        this.loadCustomPrompt();
        
        // Update the display when page loads
        this.updatePersonalityDetails();
    }
    
    updateRAGDescription() {
        const descriptions = {
            'original': '🌿 Original uses meaning-based search to find related concepts',
            'bm25': '🔍 BM25 combines keyword matching with semantic understanding for precise results',
            'both': '⚖️ Compare both methods side-by-side to see different perspectives'
        };
        
        this.ragDescText.textContent = descriptions[this.ragTypeSelect.value] || descriptions['original'];
    }
    
    togglePersonalityDetails() {
        const isVisible = this.personalityDetailsSection.style.display !== 'none';
        this.personalityDetailsSection.style.display = isVisible ? 'none' : 'block';
        this.personalityDetails.textContent = isVisible ? 'details' : 'hide';
        
        if (!isVisible) {
            this.updatePersonalityDetails();
        }
    }
    
    updatePersonalityDetails() {
        const personality = this.personalitySelect.value;
        
        if (personality === 'custom') {
            // Show custom prompt editor
            this.personalityTitle.textContent = 'Custom - Your Personal Gaia';
            this.personalityPrompt.style.display = 'none';
            this.customPromptEditor.style.display = 'block';
            this.customPromptActions.style.display = 'flex';
        } else {
            // Show predefined personality
            const data = this.personalityPrompts[personality];
            if (data) {
                this.personalityTitle.textContent = data.title;
                this.personalityPrompt.textContent = data.prompt;
                this.personalityPrompt.style.display = 'block';
                this.customPromptEditor.style.display = 'none';
                this.customPromptActions.style.display = 'none';
            }
        }
    }
    
    loadCustomPrompt() {
        // Load custom prompt from localStorage or initialize with default
        const savedCustomPrompt = localStorage.getItem('gaiaCustomPrompt');
        const lastPersonality = localStorage.getItem('gaiaLastPersonality') || 'warm_mother';
        
        if (savedCustomPrompt) {
            this.customPromptEditor.value = savedCustomPrompt;
        } else {
            // Use last selected personality as template
            const templateData = this.personalityPrompts[lastPersonality];
            if (templateData) {
                this.customPromptEditor.value = templateData.prompt;
            }
        }
    }
    
    saveCustomPromptToStorage() {
        const customPrompt = this.customPromptEditor.value.trim();
        if (customPrompt.length < 50) {
            alert('Custom prompt must be at least 50 characters long.');
            return;
        }
        
        localStorage.setItem('gaiaCustomPrompt', customPrompt);
        
        // Show confirmation
        const originalText = this.saveCustomPrompt.textContent;
        this.saveCustomPrompt.textContent = 'Saved!';
        this.saveCustomPrompt.style.background = 'var(--forest-green)';
        
        setTimeout(() => {
            this.saveCustomPrompt.textContent = originalText;
            this.saveCustomPrompt.style.background = '';
        }, 2000);
    }
    
    resetCustomPromptToDefault() {
        if (confirm('Reset to default template? This will lose your current custom prompt.')) {
            const lastPersonality = localStorage.getItem('gaiaLastPersonality') || 'warm_mother';
            const templateData = this.personalityPrompts[lastPersonality];
            
            if (templateData) {
                this.customPromptEditor.value = templateData.prompt;
                localStorage.removeItem('gaiaCustomPrompt');
                
                // Show confirmation
                const originalText = this.resetCustomPrompt.textContent;
                this.resetCustomPrompt.textContent = 'Reset!';
                this.resetCustomPrompt.style.background = '#666';
                
                setTimeout(() => {
                    this.resetCustomPrompt.textContent = originalText;
                    this.resetCustomPrompt.style.background = '';
                }, 2000);
            }
        }
    }
    
    updateSmartRecommendations(response) {
        // Extract episodes from this response
        const currentEpisodes = response.citations || response.sources || [];
        
        // Add to conversation tracking
        currentEpisodes.forEach(episode => {
            if (episode.episode_number || episode.episode_id) {
                this.conversationEpisodes.add(episode.episode_number || episode.episode_id);
            }
        });
        
        // Extract topics from user message and response (simple keyword extraction)
        const userMessage = this.lastUserMessage || '';
        const gaiaResponse = response.response || '';
        const combinedText = (userMessage + ' ' + gaiaResponse).toLowerCase();
        
        // Simple topic extraction (you could make this more sophisticated)
        const topicKeywords = [
            'permaculture', 'regenerative', 'agriculture', 'sustainability', 'climate',
            'composting', 'soil', 'biodiversity', 'water', 'energy', 'community',
            'biochar', 'carbon', 'farming', 'garden', 'food', 'ecosystem'
        ];
        
        topicKeywords.forEach(topic => {
            if (combinedText.includes(topic)) {
                this.conversationTopics.add(topic);
            }
        });
        
        // Show smart recommendations (different episodes + related topics)
        this.showSmartRecommendations(currentEpisodes);
    }
    
    showSmartRecommendations(currentEpisodes) {
        if (!currentEpisodes || currentEpisodes.length === 0) {
            this.recommendations.style.display = 'none';
            return;
        }
        
        // Clear previous recommendations
        this.recommendationsList.innerHTML = '';
        this.recommendationsList.className = 'recommendations-list';
        
        // Get unique episodes (remove duplicates) and limit to 3-4 most relevant
        const uniqueEpisodes = this.getUniqueEpisodes(currentEpisodes);
        const episodesToShow = uniqueEpisodes.slice(0, 3);
        
        // Add conversation context header
        const contextHeader = document.createElement('div');
        contextHeader.className = 'recommendations-context';
        contextHeader.innerHTML = `
            <div style="margin-bottom: 1rem; font-size: 0.9rem; color: var(--warm-gray);">
                Based on our conversation about: ${Array.from(this.conversationTopics).slice(0, 3).join(', ')}
            </div>
        `;
        this.recommendationsList.appendChild(contextHeader);
        
        // Add episode recommendations
        this.addRecommendationItems(this.recommendationsList, episodesToShow);
        
        // Add suggestion for related topics if conversation has context
        if (this.conversationTopics.size > 0 && this.conversationEpisodes.size > 2) {
            const exploreMore = document.createElement('div');
            exploreMore.className = 'explore-more';
            exploreMore.innerHTML = `
                <div style="margin-top: 1rem; padding: 1rem; background: linear-gradient(135deg, #f0f8f0, #e8f5e8); border-radius: 8px; border-left: 3px solid var(--sage-green);">
                    <div style="font-weight: 500; color: var(--forest-green); margin-bottom: 0.5rem;">💡 Explore Related Topics</div>
                    <div style="font-size: 0.9rem; color: var(--earth-green);">
                        Try asking about: "other episodes on ${Array.from(this.conversationTopics)[0]}" or "what else about sustainable ${Array.from(this.conversationTopics)[1] || 'practices'}"
                    </div>
                </div>
            `;
            this.recommendationsList.appendChild(exploreMore);
        }
        
        this.recommendations.style.display = 'block';
    }
    
    addComparisonMessage(response) {
        const comparisonDiv = document.createElement('div');
        comparisonDiv.className = 'message gaia-message';
        
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.textContent = '🌱';
        
        const content = document.createElement('div');
        content.className = 'message-content comparison-container';
        
        // Original response panel
        const originalPanel = document.createElement('div');
        originalPanel.className = 'comparison-panel';
        originalPanel.innerHTML = `
            <div class="comparison-header">🌿 Original (Semantic Search)</div>
            <div class="comparison-content">${response.original.response || 'No response available'}</div>
            ${this.formatComparisonCitations(response.original.citations || response.original.sources, 'original')}
        `;
        
        // BM25 response panel
        const bm25Panel = document.createElement('div');
        bm25Panel.className = 'comparison-panel';
        bm25Panel.innerHTML = `
            <div class="comparison-header">🔍 BM25 (Hybrid Search)</div>
            <div class="comparison-content">${response.bm25.response || 'No response available'}</div>
            ${this.formatComparisonCitations(response.bm25.sources || response.bm25.citations, 'bm25')}
        `;
        
        content.appendChild(originalPanel);
        content.appendChild(bm25Panel);
        
        comparisonDiv.appendChild(avatar);
        comparisonDiv.appendChild(content);
        
        this.chatMessages.appendChild(comparisonDiv);
        this.scrollToBottom();
        
        // Show smart recommendations combining both methods
        this.showComparisonRecommendations(response);
    }
    
    showComparisonRecommendations(response) {
        // Combine episodes from both methods
        const originalEpisodes = response.original?.citations || response.original?.sources || [];
        const bm25Episodes = response.bm25?.sources || response.bm25?.citations || [];
        const allEpisodes = [...originalEpisodes, ...bm25Episodes];
        
        // Track episodes and update smart recommendations
        allEpisodes.forEach(episode => {
            if (episode.episode_number || episode.episode_id) {
                this.conversationEpisodes.add(episode.episode_number || episode.episode_id);
            }
        });
        
        // Extract topics from responses
        const originalResponse = response.original?.response || '';
        const bm25Response = response.bm25?.response || '';
        const userMessage = this.lastUserMessage || '';
        const combinedText = (userMessage + ' ' + originalResponse + ' ' + bm25Response).toLowerCase();
        
        const topicKeywords = [
            'permaculture', 'regenerative', 'agriculture', 'sustainability', 'climate',
            'composting', 'soil', 'biodiversity', 'water', 'energy', 'community',
            'biochar', 'carbon', 'farming', 'garden', 'food', 'ecosystem'
        ];
        
        topicKeywords.forEach(topic => {
            if (combinedText.includes(topic)) {
                this.conversationTopics.add(topic);
            }
        });
        
        // Show smart recommendations with comparison context
        if (allEpisodes.length > 0) {
            this.recommendationsList.innerHTML = '';
            this.recommendationsList.className = 'recommendations-list';
            
            // Get unique episodes and limit to top 4
            const uniqueEpisodes = this.getUniqueEpisodes(allEpisodes);
            const episodesToShow = uniqueEpisodes.slice(0, 4);
            
            // Add comparison context header
            const contextHeader = document.createElement('div');
            contextHeader.className = 'recommendations-context';
            contextHeader.innerHTML = `
                <div style="margin-bottom: 1rem; font-size: 0.9rem; color: var(--warm-gray);">
                    📊 Episodes found by both search methods about: ${Array.from(this.conversationTopics).slice(0, 3).join(', ')}
                </div>
            `;
            this.recommendationsList.appendChild(contextHeader);
            
            // Add episode recommendations
            this.addRecommendationItems(this.recommendationsList, episodesToShow);
            
            this.recommendations.style.display = 'block';
        }
    }
    
    formatComparisonCitations(citations, type) {
        if (!citations || citations.length === 0) {
            return '<div class="comparison-citations">No episodes referenced</div>';
        }
        
        const episodeNumbers = citations.map(c => 
            c.episode_number || c.episode_id || 'Unknown'
        ).filter((v, i, a) => a.indexOf(v) === i);
        
        return `<div class="comparison-citations">
            Episodes: ${episodeNumbers.join(', ')} 
            <span style="opacity: 0.7">(${citations.length} references)</span>
        </div>`;
    }
    
    getUniqueEpisodes(episodes) {
        const seen = new Map(); // Use Map to store episode details
        const uniqueEpisodes = [];
        
        episodes.forEach(episode => {
            // Normalize the episode number (prioritize episode_number over episode_id)
            const episodeNumber = episode.episode_number || episode.episode_id;
            
            if (!episodeNumber || episodeNumber === 'Unknown') {
                // If no episode number, use title as fallback key
                const titleKey = episode.title;
                if (!seen.has(titleKey)) {
                    seen.set(titleKey, true);
                    uniqueEpisodes.push(episode);
                }
                return;
            }
            
            // Use episode number as primary key
            if (!seen.has(episodeNumber)) {
                seen.set(episodeNumber, true);
                uniqueEpisodes.push(episode);
            }
        });
        
        return uniqueEpisodes;
    }
    
    getUniqueCitations(citations) {
        const seen = new Set();
        return citations.filter(citation => {
            const key = citation.episode_number || citation.episode_id || citation.title;
            if (seen.has(key)) {
                return false;
            }
            seen.add(key);
            return true;
        });
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