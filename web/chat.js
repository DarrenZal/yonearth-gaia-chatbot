/**
 * YonEarth Gaia Chat Interface JavaScript
 */

class GaiaChat {
    constructor() {
        this.apiUrl = localStorage.getItem('gaiaApiUrl') || '/api';
        this.sessionId = this.generateSessionId();
        this.isLoading = false;
        this.conversationEpisodes = new Set(); // Track episodes mentioned in conversation
        this.conversationTopics = new Set();  // Track topics discussed
        this.conversationHistory = []; // Track full conversation for recommendations
        
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
        this.modelTypeSelect = document.getElementById('modelType');
        this.modelDescText = document.getElementById('modelDescText');
        this.maxReferencesSelect = document.getElementById('maxReferences');
        this.referencesDescText = document.getElementById('referencesDescText');
        this.categoryThresholdSelect = document.getElementById('categoryThreshold');
        this.categoryDescText = document.getElementById('categoryDescText');
        this.recommendations = document.getElementById('recommendations');
        this.recommendationsList = document.getElementById('recommendationsList');
        this.status = document.getElementById('status');

        // Collapsible controls
        this.controlsToggle = document.getElementById('controlsToggle');
        this.controlsToggleText = document.getElementById('controlsToggleText');
        this.chatControls = document.getElementById('chatControls');
        if (this.controlsToggle) {
            this.controlsToggle.style.display = 'none';
        }
        
        // Voice controls
        this.voiceToggle = document.getElementById('voiceToggle');
        this.voiceToggleText = document.getElementById('voiceToggleText');
        this.voiceSettings = document.getElementById('voiceSettings');
        this.voiceSelection = document.getElementById('voiceSelection');
        this.autoPlayVoice = document.getElementById('autoPlayVoice');
        this.voicePlayer = document.getElementById('voicePlayer');
        this.voiceIcon = document.querySelector('.voice-icon');

        // Default selections
        if (this.modelTypeSelect) {
            this.modelTypeSelect.value = 'gpt-4o-mini';
        }
        if (this.ragTypeSelect) {
            this.ragTypeSelect.value = 'bm25';
        }
        if (this.maxReferencesSelect) {
            this.maxReferencesSelect.value = '5';
        }

        // Voice state
        this.voiceEnabled = localStorage.getItem('voiceEnabled') === 'true';
        this.autoPlay = localStorage.getItem('autoPlayVoice') !== 'false'; // Default true
        this.selectedVoiceId = localStorage.getItem('selectedVoiceId') || 'piper-gaia';
        this.currentAudio = null;
        this.audioQueue = [];
        
        console.log('Initial voice state:', {
            voiceEnabled: this.voiceEnabled,
            localStorage: localStorage.getItem('voiceEnabled'),
            autoPlay: this.autoPlay
        });
        
        // Initialize personality prompts
        this.initializePersonalityPrompts();
        
        // Initialize descriptions
        this.updateModelDescription();
        this.updateRAGDescription();
        this.updateReferencesDescription();
        this.updateCategoryDescription();
        
        // Initialize voice UI
        this.updateVoiceUI();

        // Initialize controls visibility (always open; toggle hidden)
        this.controlsCollapsed = false;
        this.updateControlsVisibility();
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
        
        // Model type change
        this.modelTypeSelect.addEventListener('change', () => {
            this.updateModelDescription();
        });
        
        // Max references change
        this.maxReferencesSelect.addEventListener('change', () => {
            this.updateReferencesDescription();
        });
        
        // Category threshold change
        this.categoryThresholdSelect.addEventListener('change', () => {
            this.updateCategoryDescription();
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

        // Controls toggle
        if (this.controlsToggle) {
            this.controlsToggle.addEventListener('click', () => {
                this.toggleControlsVisibility();
            });
        }
        
        // Configuration modal (Ctrl+Shift+C)
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.shiftKey && e.key === 'C') {
                this.showConfigModal();
            }
        });
        
        // Voice toggle
        this.voiceToggle.addEventListener('click', () => {
            this.toggleVoice();
        });

        // Voice selection
        this.voiceSelection.addEventListener('change', () => {
            this.selectedVoiceId = this.voiceSelection.value;
            localStorage.setItem('selectedVoiceId', this.selectedVoiceId);
            console.log('Voice changed to:', this.selectedVoiceId);
        });

        // Auto-play checkbox
        this.autoPlayVoice.addEventListener('change', () => {
            this.autoPlay = this.autoPlayVoice.checked;
            localStorage.setItem('autoPlayVoice', this.autoPlay);
        });
    }

    toggleControlsVisibility() {
        this.controlsCollapsed = !this.controlsCollapsed;
        this.updateControlsVisibility();
    }

    updateControlsVisibility() {
        if (!this.chatControls) {
            return;
        }

        if (this.controlsCollapsed) {
            this.chatControls.style.display = 'none';
            if (this.controlsToggle) {
                this.controlsToggle.setAttribute('aria-expanded', 'false');
            }
            if (this.controlsToggleText) {
                this.controlsToggleText.textContent = 'Show settings';
            }
        } else {
            this.chatControls.style.display = '';
            if (this.controlsToggle) {
                this.controlsToggle.setAttribute('aria-expanded', 'true');
            }
            if (this.controlsToggleText) {
                this.controlsToggleText.textContent = 'Hide settings';
            }
        }
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
        
        // Add user message to conversation history
        this.conversationHistory.push({
            role: 'user',
            content: message,
            citations: []
        });
        console.log('Added user message to conversation history. Total messages:', this.conversationHistory.length);
        
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
            
            // Handle model comparison mode
            if (response.modelComparison) {
                console.log('Handling model comparison response:', response);
                this.addModelComparisonMessage(response);
                
                // Add combined response to conversation history
                const allResponses = Object.values(response.models).map(r => r.response).join(' ');
                const allCitations = Object.values(response.models).flatMap(r => r.citations || []);
                console.log('Combined responses for history:', allResponses.length, 'characters');
                console.log('Combined citations:', allCitations.length, 'items');
                
                this.conversationHistory.push({
                    role: 'gaia',
                    content: allResponses,
                    citations: allCitations
                });
                
                // Show smart recommendations based on conversation (skip for hybrid mode for now)
                if (this.ragTypeSelect.value !== 'hybrid') {
                    await this.updateConversationRecommendations();
                }
            } 
            // Handle RAG comparison mode
            else if (response.comparison) {
                this.addComparisonMessage(response);
                
                // Add both responses to conversation history for comparison mode
                this.conversationHistory.push({
                    role: 'gaia',
                    content: response.original.response + ' ' + response.bm25.response,
                    citations: [...(response.original.citations || response.original.sources || []), ...(response.bm25.sources || response.bm25.citations || [])]
                });
                
                // Show smart recommendations based on conversation (skip for hybrid mode for now)
                if (this.ragTypeSelect.value !== 'hybrid') {
                    await this.updateConversationRecommendations();
                }
            } else {
                // Handle hybrid QA response format
                const responseText = response.answer || response.response;
                const responseCitations = response.sources || response.citations || [];

                // Add Gaia's response to conversation history
                this.conversationHistory.push({
                    role: 'gaia',
                    content: responseText,
                    citations: responseCitations
                });
                console.log('Added Gaia response to conversation history. Total messages:', this.conversationHistory.length);
                console.log('Citations in response:', responseCitations);
                console.log('Audio data in response:', !!response.audio_data, 'Length:', response.audio_data ? response.audio_data.length : 0);

                // Add Gaia's response WITH inline citations, audio, and cost breakdown
                this.addMessage(responseText, 'gaia', responseCitations, false, response.audio_data, null);
                
                // Show smart recommendations based on conversation (skip for hybrid mode for now)
                if (this.ragTypeSelect.value !== 'hybrid') {
                    await this.updateConversationRecommendations();
                }
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
        const modelType = this.modelTypeSelect.value;

        // Handle model comparison mode
        if (modelType === 'compare') {
            console.log('🤖 Model comparison mode selected!');
            console.log('📊 Will compare 3 models: gpt-3.5-turbo, gpt-4o-mini, gpt-4');
            return await this.callModelComparisonAPI(message);
        }

        // Handle RAG comparison mode
        if (ragType === 'both') {
            return await this.callBothAPIs(message);
        }

        // Determine which endpoint to use
        let endpoint;
        if (ragType === 'hybrid') {
            endpoint = '/qa/hybrid';
        } else if (ragType === 'bm25') {
            endpoint = '/chat';
        } else {
            endpoint = '/chat';
        }
        
        const maxReferences = parseInt(this.maxReferencesSelect.value);

        let requestBody;

        // Build request body based on endpoint
        if (ragType === 'hybrid') {
            // Hybrid QA endpoint uses different parameter names
            requestBody = {
                query: message,
                k: maxReferences,
                category_threshold: parseFloat(this.categoryThresholdSelect.value)
            };
        } else {
            // Original and BM25 endpoints
            requestBody = {
                message: message,
                session_id: this.sessionId,
                personality: this.personalitySelect.value,
                max_results: 5,
                model: modelType !== 'compare' ? modelType : undefined,
                max_references: maxReferences,
                enable_voice: this.voiceEnabled,
                voice_id: this.voiceEnabled ? this.selectedVoiceId : undefined
            };

            console.log('Voice enabled in request:', requestBody.enable_voice, 'Voice ID:', requestBody.voice_id);

            // Add custom prompt if custom personality is selected
            if (this.personalitySelect.value === 'custom') {
                const customPrompt = this.customPromptEditor.value.trim();
                if (customPrompt) {
                    requestBody.custom_prompt = customPrompt;
                }
            }
        }

        // Add BM25-specific parameters
        if (ragType === 'bm25') {
            requestBody.search_method = 'bm25';
            requestBody.k = 5;
            requestBody.include_sources = true;
            requestBody.gaia_personality = this.personalitySelect.value;
            requestBody.max_citations = maxReferences; // BM25 uses max_citations instead of max_references
            requestBody.category_threshold = parseFloat(this.categoryThresholdSelect.value);
            requestBody.enable_voice = this.voiceEnabled;
            requestBody.voice_id = this.voiceEnabled ? this.selectedVoiceId : undefined;

            // Also add custom prompt for BM25
            if (this.personalitySelect.value === 'custom') {
                const customPrompt = this.customPromptEditor.value.trim();
                if (customPrompt) {
                    requestBody.custom_prompt = customPrompt;
                }
            }
        }
        
        console.log('Sending request to:', `${this.apiUrl}${endpoint}`);
        console.log('Request body:', requestBody);
        
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
    
    async callModelComparisonAPI(message) {
        console.log('callModelComparisonAPI called with message:', message);
        
        // Define the models to compare
        const models = ['gpt-3.5-turbo', 'gpt-4o-mini', 'gpt-4'];
        const results = {};
        
        // Track total processing time
        const startTime = Date.now();
        
        // Make parallel API calls with different models
        const promises = models.map(async (model) => {
            console.log(`Calling API with model: ${model}`);
            
            try {
                const maxReferences = parseInt(this.maxReferencesSelect.value);
                
                const requestBody = {
                    message: message,
                    session_id: this.sessionId,
                    personality: this.personalitySelect.value,
                    max_results: 5,
                    model: model,  // Specify the model explicitly
                    max_references: maxReferences
                };
                
                // Add custom prompt if custom personality is selected
                if (this.personalitySelect.value === 'custom') {
                    const customPrompt = this.customPromptEditor.value.trim();
                    if (customPrompt) {
                        requestBody.custom_prompt = customPrompt;
                    }
                }
                
        // Determine which endpoint to use based on current RAG type (BM25 also uses /chat with search_method)
        const ragType = this.ragTypeSelect.value;
        const endpoint = '/chat';
                
                // Add BM25-specific parameters if using BM25
                if (ragType !== 'original') {
                    requestBody.search_method = 'hybrid';
                    requestBody.k = 5;
                    requestBody.include_sources = true;
                    requestBody.gaia_personality = this.personalitySelect.value;
                }
                
                console.log(`Sending request to ${endpoint} with model ${model}:`, requestBody);
                
                const response = await fetch(`${this.apiUrl}${endpoint}`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(requestBody)
                });
                
                if (!response.ok) {
                    const errorData = await response.json().catch(() => ({}));
                    console.error(`Error with model ${model}:`, errorData);
                    throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
                }
                
                const data = await response.json();
                console.log(`Response from model ${model}:`, data);
                
                // Format the response to match expected structure
                return {
                    model: model,
                    data: {
                        response: data.response,
                        personality: data.personality || this.personalitySelect.value,
                        citations: data.citations || data.sources || [],
                        context_used: data.context_used || data.retrieval_count || 0,
                        session_id: this.sessionId,
                        retrieval_count: data.retrieval_count || 0,
                        processing_time: data.processing_time || 0,
                        model_used: model
                    }
                };
            } catch (error) {
                console.error(`Failed to get response from model ${model}:`, error);
                return {
                    model: model,
                    data: {
                        response: `Error with ${model}: ${error.message}`,
                        personality: this.personalitySelect.value,
                        citations: [],
                        context_used: 0,
                        session_id: this.sessionId,
                        retrieval_count: 0,
                        processing_time: 0,
                        model_used: model
                    }
                };
            }
        });
        
        // Wait for all responses
        const responses = await Promise.all(promises);
        
        // Build the results object
        responses.forEach(({ model, data }) => {
            results[model] = data;
        });
        
        const totalProcessingTime = (Date.now() - startTime) / 1000;
        console.log(`Model comparison completed in ${totalProcessingTime}s`);
        console.log('All model results:', results);
        
        // Return in the expected format
        return {
            modelComparison: true,
            models: results,
            processing_time: totalProcessingTime
        };
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
        const endpoint = '/chat';
        const maxReferences = parseInt(this.maxReferencesSelect.value);
        
        const requestBody = {
            message: message,
            session_id: this.sessionId,
            personality: this.personalitySelect.value,
            max_results: 5,
            max_references: maxReferences
        };
        
        // Add custom prompt if custom personality is selected
        if (this.personalitySelect.value === 'custom') {
            const customPrompt = this.customPromptEditor.value.trim();
            if (customPrompt) {
                requestBody.custom_prompt = customPrompt;
            }
        }
        
        if (type === 'bm25') {
            requestBody.search_method = 'bm25';
            requestBody.k = 5;
            requestBody.include_sources = true;
            requestBody.gaia_personality = this.personalitySelect.value;
            requestBody.max_citations = maxReferences; // BM25 uses max_citations
            requestBody.category_threshold = parseFloat(this.categoryThresholdSelect.value);
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
    
    addMessage(text, sender, citations = [], isError = false, audioData = null, costBreakdown = null) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        
        // Generate unique message ID
        const messageId = `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
        messageDiv.id = messageId;
        
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
        
        // Add hyperlinks to response text if citations are available
        if (sender === 'gaia' && citations && citations.length > 0) {
            messageText.innerHTML = this.addHyperlinksToResponse(text, citations);
        } else {
            messageText.textContent = text;
        }
        
        content.appendChild(messageText);
        
        // Add citations if available
        if (citations && citations.length > 0) {
            const citationsDiv = this.createCitationsDiv(citations);
            content.appendChild(citationsDiv);
        }
        
        // Add voice controls if audio is available
        if (sender === 'gaia' && audioData && this.voiceEnabled) {
            console.log(`Voice audio received for message ${messageId}, length: ${audioData.length}`);
            const audioControls = this.createAudioControls(audioData, messageId);
            content.appendChild(audioControls);
            
            // Auto-play if enabled
            if (this.autoPlay) {
                console.log('Auto-playing audio...');
                setTimeout(() => this.playAudio(audioData, messageId), 500);
            }
        } else if (sender === 'gaia') {
            console.log(`No voice audio for message ${messageId}. audioData: ${!!audioData}, voiceEnabled: ${this.voiceEnabled}`);
        }
        
        // Add feedback section for Gaia's messages (but not errors)
        if (sender === 'gaia' && !isError && this.lastUserMessage) {
            const feedbackDiv = this.createFeedbackDiv(messageId, this.lastUserMessage, text, citations);
            content.appendChild(feedbackDiv);
        }
        
        // Add cost breakdown section for Gaia's messages
        if (sender === 'gaia' && !isError && costBreakdown) {
            const costDiv = this.createCostBreakdownDiv(costBreakdown);
            content.appendChild(costDiv);
        }
        
        messageDiv.appendChild(avatar);
        messageDiv.appendChild(content);
        
        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
    }
    
    addHyperlinksToResponse(text, citations) {
        // Escape HTML to prevent XSS
        let htmlText = text.replace(/&/g, '&amp;')
                           .replace(/</g, '&lt;')
                           .replace(/>/g, '&gt;')
                           .replace(/"/g, '&quot;')
                           .replace(/'/g, '&#039;');
        
        // Create a map of episode numbers and book titles to their URLs
        const citationMap = new Map();
        
        citations.forEach(citation => {
            const isBook = citation.episode_number?.toString().startsWith('Book:');
            
            if (isBook) {
                // Extract book title from episode_number
                const bookTitle = citation.episode_number.substring(5).trim();
                const bookUrl = citation.ebook_url || citation.url;
                if (bookUrl) {
                    // Store multiple variations of the book title for matching
                    citationMap.set(bookTitle, bookUrl);
                    citationMap.set(bookTitle.toLowerCase(), bookUrl);
                    
                    // Also store common variations
                    if (bookTitle.includes('VIRIDITAS') || bookTitle.includes('Viriditas')) {
                        citationMap.set('VIRIDITAS', bookUrl);
                        citationMap.set('Viriditas', bookUrl);
                        citationMap.set('VIRIDITAS: THE GREAT HEALING', bookUrl);
                    } else if (bookTitle.includes('Soil Stewardship')) {
                        citationMap.set('Soil Stewardship Handbook', bookUrl);
                        citationMap.set('Soil Stewardship', bookUrl);
                    } else if (bookTitle.includes('Y on Earth') || bookTitle.includes('Why on Earth')) {
                        citationMap.set('Y on Earth', bookUrl);
                        citationMap.set('Why on Earth', bookUrl);
                        citationMap.set('Y on Earth: Get Smarter, Feel Better, Heal the Planet', bookUrl);
                    }
                }
            } else {
                // For episodes
                const episodeNum = citation.episode_number || citation.episode_id;
                if (episodeNum && citation.url) {
                    citationMap.set(`Episode ${episodeNum}`, citation.url);
                    citationMap.set(`episode ${episodeNum}`, citation.url);
                    citationMap.set(`Ep ${episodeNum}`, citation.url);
                    citationMap.set(`ep ${episodeNum}`, citation.url);
                    citationMap.set(episodeNum.toString(), citation.url);
                }
            }
        });
        
        // Replace episode mentions with links
        // Match patterns like "Episode 115", "Episode 108: Ann Armbrecht", "Ep 120", etc.
        htmlText = htmlText.replace(/\b(Episode|episode|Ep|ep)\s+(\d+)(?:\s*:\s*[^.!?,\n]*)?/g, (match, prefix, number) => {
            const episodeKey = `Episode ${number}`;
            const url = citationMap.get(episodeKey);
            if (url) {
                return `<a href="${url}" target="_blank" class="inline-citation-link">${match}</a>`;
            }
            return match;
        });
        
        // Replace book mentions with links
        // Look for book titles we know about
        const bookPatterns = [
            // VIRIDITAS variations
            /\bVIRIDITAS(?:\s*:\s*THE\s+GREAT\s+HEALING)?\b/gi,
            /\bViriditas(?:\s*:\s*The\s+Great\s+Healing)?\b/g,
            
            // Soil Stewardship variations
            /\bSoil\s+Stewardship\s+Handbook\b/gi,
            /\bSoil\s+Stewardship\b/gi,
            
            // Y on Earth variations
            /\bY\s+on\s+Earth(?:\s*:\s*Get\s+Smarter,\s+Feel\s+Better,\s+Heal\s+the\s+Planet)?\b/gi,
            /\bWhy\s+on\s+Earth\b/gi
        ];
        
        bookPatterns.forEach(pattern => {
            htmlText = htmlText.replace(pattern, (match) => {
                // Try different variations to find the URL
                let url = citationMap.get(match) || 
                         citationMap.get(match.toLowerCase()) ||
                         citationMap.get(match.toUpperCase());
                
                // Try simplified versions
                if (!url) {
                    if (match.toLowerCase().includes('viriditas')) {
                        url = citationMap.get('VIRIDITAS') || citationMap.get('Viriditas');
                    } else if (match.toLowerCase().includes('soil')) {
                        url = citationMap.get('Soil Stewardship Handbook') || citationMap.get('Soil Stewardship');
                    } else if (match.toLowerCase().includes('earth')) {
                        url = citationMap.get('Y on Earth') || citationMap.get('Why on Earth');
                    }
                }
                
                if (url) {
                    return `<a href="${url}" target="_blank" class="inline-citation-link">${match}</a>`;
                }
                return match;
            });
        });
        
        return htmlText;
    }
    
    createFeedbackDiv(messageId, query, response, citations) {
        const feedbackDiv = document.createElement('div');
        feedbackDiv.className = 'feedback-section';
        feedbackDiv.id = `feedback-${messageId}`;
        
        // Divider
        const divider = document.createElement('div');
        divider.className = 'feedback-divider';
        feedbackDiv.appendChild(divider);
        
        // Header
        const header = document.createElement('div');
        header.className = 'feedback-header';
        header.textContent = 'Was this response helpful?';
        feedbackDiv.appendChild(header);
        
        // Thumbs up/down buttons
        const thumbsContainer = document.createElement('div');
        thumbsContainer.className = 'feedback-thumbs';
        
        const thumbsUp = document.createElement('button');
        thumbsUp.className = 'feedback-thumb';
        thumbsUp.innerHTML = '👍';
        thumbsUp.title = 'Helpful';
        thumbsUp.onclick = () => this.submitQuickFeedback(messageId, 'helpful', query, response, citations);
        
        const thumbsDown = document.createElement('button');
        thumbsDown.className = 'feedback-thumb';
        thumbsDown.innerHTML = '👎';
        thumbsDown.title = 'Not helpful';
        thumbsDown.onclick = () => this.submitQuickFeedback(messageId, 'not-helpful', query, response, citations);
        
        thumbsContainer.appendChild(thumbsUp);
        thumbsContainer.appendChild(thumbsDown);
        feedbackDiv.appendChild(thumbsContainer);
        
        // Expandable detailed feedback section
        const detailsContainer = document.createElement('div');
        detailsContainer.className = 'feedback-details';
        detailsContainer.style.display = 'none';
        
        // Relevance rating
        const relevanceDiv = document.createElement('div');
        relevanceDiv.className = 'feedback-relevance';
        relevanceDiv.innerHTML = `
            <label>Relevance of response:</label>
            <div class="star-rating" data-rating="0">
                <span class="star" data-value="1">☆</span>
                <span class="star" data-value="2">☆</span>
                <span class="star" data-value="3">☆</span>
                <span class="star" data-value="4">☆</span>
                <span class="star" data-value="5">☆</span>
            </div>
        `;
        
        // Episodes checkbox
        const episodesDiv = document.createElement('div');
        episodesDiv.className = 'feedback-episodes';
        episodesDiv.innerHTML = `
            <label>
                <input type="checkbox" id="episodes-correct-${messageId}">
                Were the right episodes/books included?
            </label>
        `;
        
        // Detailed feedback text
        const textDiv = document.createElement('div');
        textDiv.className = 'feedback-text';
        textDiv.innerHTML = `
            <label for="feedback-text-${messageId}">Additional feedback (optional):</label>
            <textarea id="feedback-text-${messageId}" rows="3" placeholder="What could be improved? What worked well?"></textarea>
        `;
        
        // Submit button
        const submitBtn = document.createElement('button');
        submitBtn.className = 'feedback-submit';
        submitBtn.textContent = 'Submit Detailed Feedback';
        submitBtn.onclick = () => this.submitDetailedFeedback(messageId, query, response, citations);
        
        detailsContainer.appendChild(relevanceDiv);
        detailsContainer.appendChild(episodesDiv);
        detailsContainer.appendChild(textDiv);
        detailsContainer.appendChild(submitBtn);
        
        feedbackDiv.appendChild(detailsContainer);
        
        // Expand details link
        const expandLink = document.createElement('a');
        expandLink.className = 'feedback-expand';
        expandLink.href = '#';
        expandLink.textContent = 'Provide detailed feedback';
        expandLink.onclick = (e) => {
            e.preventDefault();
            detailsContainer.style.display = detailsContainer.style.display === 'none' ? 'block' : 'none';
            expandLink.textContent = detailsContainer.style.display === 'none' ? 'Provide detailed feedback' : 'Hide detailed feedback';
        };
        feedbackDiv.appendChild(expandLink);
        
        // Setup star rating interaction
        this.setupStarRating(feedbackDiv.querySelector('.star-rating'));
        
        return feedbackDiv;
    }
    
    createCostBreakdownDiv(costBreakdown) {
        const costDiv = document.createElement('div');
        costDiv.className = 'cost-breakdown-section';
        
        // Divider
        const divider = document.createElement('div');
        divider.className = 'cost-divider';
        costDiv.appendChild(divider);
        
        // Header with total cost
        const header = document.createElement('div');
        header.className = 'cost-header';
        header.innerHTML = `
            <span class="cost-title">Cost Breakdown</span>
            <span class="cost-total">${costBreakdown.summary || '$0.0000'}</span>
        `;
        costDiv.appendChild(header);
        
        // Details section
        if (costBreakdown.details && costBreakdown.details.length > 0) {
            const detailsDiv = document.createElement('div');
            detailsDiv.className = 'cost-details';
            
            costBreakdown.details.forEach(detail => {
                const detailRow = document.createElement('div');
                detailRow.className = 'cost-detail-row';
                detailRow.innerHTML = `
                    <div class="cost-service">${detail.service}</div>
                    <div class="cost-model">${detail.model}</div>
                    <div class="cost-usage">${detail.usage}</div>
                    <div class="cost-amount">${detail.cost}</div>
                `;
                detailsDiv.appendChild(detailRow);
            });
            
            costDiv.appendChild(detailsDiv);
        }
        
        // Collapse/expand functionality
        const toggleButton = document.createElement('button');
        toggleButton.className = 'cost-toggle';
        toggleButton.textContent = 'Show details';
        toggleButton.onclick = () => {
            const details = costDiv.querySelector('.cost-details');
            if (details) {
                if (details.style.display === 'none' || !details.style.display) {
                    details.style.display = 'block';
                    toggleButton.textContent = 'Hide details';
                } else {
                    details.style.display = 'none';
                    toggleButton.textContent = 'Show details';
                }
            }
        };
        
        // Initially hide details
        const details = costDiv.querySelector('.cost-details');
        if (details) {
            details.style.display = 'none';
            costDiv.appendChild(toggleButton);
        }
        
        return costDiv;
    }
    
    setupStarRating(ratingElement) {
        const stars = ratingElement.querySelectorAll('.star');
        stars.forEach((star, index) => {
            star.onclick = () => {
                const rating = index + 1;
                ratingElement.setAttribute('data-rating', rating);
                stars.forEach((s, i) => {
                    s.textContent = i < rating ? '★' : '☆';
                });
            };
        });
    }
    
    async submitQuickFeedback(messageId, type, query, response, citations) {
        const feedback = {
            messageId: messageId,
            timestamp: new Date().toISOString(),
            type: type,
            query: query,
            response: response,
            citations: citations,
            sessionId: this.sessionId,
            personality: this.personalitySelect.value,
            ragType: this.ragTypeSelect.value,
            modelType: this.modelTypeSelect.value
        };
        
        await this.saveFeedback(feedback);
        
        // Update UI to show feedback was submitted
        const feedbackSection = document.getElementById(`feedback-${messageId}`);
        if (feedbackSection) {
            const thumbsContainer = feedbackSection.querySelector('.feedback-thumbs');
            thumbsContainer.innerHTML = '<span class="feedback-submitted">✓ Thank you for your feedback!</span>';
        }
    }
    
    async submitDetailedFeedback(messageId, query, response, citations) {
        const feedbackSection = document.getElementById(`feedback-${messageId}`);
        const starRating = feedbackSection.querySelector('.star-rating').getAttribute('data-rating');
        const episodesCorrect = feedbackSection.querySelector(`#episodes-correct-${messageId}`).checked;
        const detailedText = feedbackSection.querySelector(`#feedback-text-${messageId}`).value;
        
        const feedback = {
            messageId: messageId,
            timestamp: new Date().toISOString(),
            type: 'detailed',
            query: query,
            response: response,
            citations: citations,
            sessionId: this.sessionId,
            personality: this.personalitySelect.value,
            ragType: this.ragTypeSelect.value,
            modelType: this.modelTypeSelect.value,
            relevanceRating: parseInt(starRating),
            episodesCorrect: episodesCorrect,
            detailedFeedback: detailedText
        };
        
        await this.saveFeedback(feedback);
        
        // Update UI to show feedback was submitted
        const detailsContainer = feedbackSection.querySelector('.feedback-details');
        detailsContainer.innerHTML = '<span class="feedback-submitted">✓ Thank you for your detailed feedback!</span>';
    }
    
    async saveFeedback(feedback) {
        try {
            // Send feedback to backend
            const response = await fetch(`${this.apiUrl}/feedback`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(feedback)
            });
            
            if (!response.ok) {
                console.error('Failed to save feedback:', response.status);
                // Still save locally even if backend fails
            }
        } catch (error) {
            console.error('Error saving feedback to backend:', error);
        }
        
        // Also save to localStorage for persistence
        const existingFeedback = JSON.parse(localStorage.getItem('gaiaFeedback') || '[]');
        existingFeedback.push(feedback);
        
        // Keep only last 100 feedback entries in localStorage
        if (existingFeedback.length > 100) {
            existingFeedback.splice(0, existingFeedback.length - 100);
        }
        
        localStorage.setItem('gaiaFeedback', JSON.stringify(existingFeedback));
        console.log('Feedback saved:', feedback);
    }
    
    createCitationsDiv(citations) {
        const citationsDiv = document.createElement('div');
        citationsDiv.className = 'citations';

        const title = document.createElement('div');
        title.className = 'citations-title';
        title.textContent = 'References';
        citationsDiv.appendChild(title);

        const citationsList = Array.isArray(citations) ? [...citations] : [];
        
        citationsList.forEach(citation => {
            const citationDiv = document.createElement('div');
            citationDiv.className = 'citation';
            
            const titleDiv = document.createElement('div');
            titleDiv.className = 'citation-title';
            
            // Check if this is a book reference
            if (citation.episode_number && citation.episode_number.toString().startsWith('Book:')) {
                // Extract book title and clean up chapter info
                const bookInfo = citation.episode_number.substring(5).trim(); // Remove "Book:" prefix
                const chapterMatch = citation.title.match(/Chapter\s+(\d+(?:\.\d+)?)/i);
                const chapterText = chapterMatch ? ` - Chapter ${parseInt(chapterMatch[1])}` : '';
                titleDiv.textContent = `Book: ${bookInfo}${chapterText}`;
            } else if (citation.metadata?.book_title || citation.chunk_id?.includes(':')) {
                // Handle hybrid QA book citations
                const bookTitle = citation.metadata?.book_title || 'Book';
                const chapterInfo = citation.metadata?.chapter_title || citation.title || '';
                titleDiv.textContent = `Book: ${bookTitle} - ${chapterInfo}`;
            } else if (citation.metadata?.episode_number) {
                // Handle hybrid QA episode citations with metadata
                titleDiv.textContent = `Episode ${citation.metadata.episode_number}: ${citation.title}`;
            } else if (citation.episode_number) {
                // Standard episode citation
                titleDiv.textContent = `Episode ${citation.episode_number}: ${citation.title}`;
            } else {
                // Fallback for citations without episode number
                titleDiv.textContent = citation.title || 'Reference';
            }
            
            citationDiv.appendChild(titleDiv);
            
            // Only add guest info for episodes, not books
            if (citation.guest_name && !citation.episode_number?.toString().startsWith('Book:')) {
                const guestDiv = document.createElement('div');
                guestDiv.className = 'citation-guest';
                guestDiv.textContent = `with ${citation.guest_name}`;
                citationDiv.appendChild(guestDiv);
            } else if (citation.guest_name && citation.episode_number?.toString().startsWith('Book:')) {
                // For books, show author instead
                const authorDiv = document.createElement('div');
                authorDiv.className = 'citation-guest';
                authorDiv.textContent = `Author: ${citation.guest_name}`;
                citationDiv.appendChild(authorDiv);
            }
            
            if (citation.url) {
                const linkDiv = document.createElement('div');
                linkDiv.className = 'citation-links';
                
                // Check if this is a book
                const isBook = citation.episode_number?.toString().startsWith('Book:');
                
                if (isBook) {
                    // For books, show available format links
                    if (citation.ebook_url || citation.url) {
                        const ebookLink = document.createElement('a');
                        ebookLink.href = citation.ebook_url || citation.url;
                        ebookLink.target = '_blank';
                        ebookLink.className = 'citation-link';
                        ebookLink.textContent = 'Read eBook →';
                        linkDiv.appendChild(ebookLink);
                    }
                    
                    if (citation.audiobook_url) {
                        if (linkDiv.children.length > 0) {
                            linkDiv.appendChild(document.createTextNode(' | '));
                        }
                        const audioLink = document.createElement('a');
                        audioLink.href = citation.audiobook_url;
                        audioLink.target = '_blank';
                        audioLink.className = 'citation-link';
                        audioLink.textContent = 'Listen to Audiobook →';
                        linkDiv.appendChild(audioLink);
                    }
                    
                    if (citation.print_url) {
                        if (linkDiv.children.length > 0) {
                            linkDiv.appendChild(document.createTextNode(' | '));
                        }
                        const printLink = document.createElement('a');
                        printLink.href = citation.print_url;
                        printLink.target = '_blank';
                        printLink.className = 'citation-link';
                        printLink.textContent = 'Order Print →';
                        linkDiv.appendChild(printLink);
                    }
                } else {
                    // For episodes, show single link
                    const link = document.createElement('a');
                    link.href = citation.url;
                    link.target = '_blank';
                    link.className = 'citation-link';
                    link.textContent = 'Listen to Episode →';
                    linkDiv.appendChild(link);
                }
                
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
            const maxRefs = parseInt(this.maxReferencesSelect.value);
            this.addRecommendationItems(originalRecs, originalCitations.slice(0, maxRefs));
            
            // Add BM25 recommendations
            const bm25Citations = comparisonData.bm25.sources || comparisonData.bm25.citations || [];
            this.addRecommendationItems(bm25Recs, bm25Citations.slice(0, maxRefs));
            
            this.recommendationsList.appendChild(originalRecs);
            this.recommendationsList.appendChild(bm25Recs);
        } else {
            // Single method recommendations
            this.recommendationsList.className = 'recommendations-list';
            const maxRefs = parseInt(this.maxReferencesSelect.value);
            this.addRecommendationItems(this.recommendationsList, citations.slice(0, maxRefs));
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
            
            // Check if this is a book reference
            if (episodeNum.toString().startsWith('Book:')) {
                const bookInfo = episodeNum.substring(5).trim();
                const chapterMatch = title.match(/Chapter\s+(\d+(?:\.\d+)?)/i);
                const chapterText = chapterMatch ? ` - Chapter ${parseInt(chapterMatch[1])}` : '';
                titleDiv.textContent = `Book: ${bookInfo}${chapterText}`;
            } else {
                titleDiv.textContent = `Episode ${episodeNum}: ${title}`;
            }
            
            recDiv.appendChild(titleDiv);
            
            const guestName = citation.guest_name || citation.guest || 'Unknown Guest';
            if (!episodeNum.toString().startsWith('Book:')) {
                const guestDiv = document.createElement('div');
                guestDiv.className = 'recommendation-guest';
                guestDiv.textContent = `with ${guestName}`;
                recDiv.appendChild(guestDiv);
            } else {
                // For books, show author
                const authorDiv = document.createElement('div');
                authorDiv.className = 'recommendation-guest';
                authorDiv.textContent = `Author: ${guestName}`;
                recDiv.appendChild(authorDiv);
            }
            
            if (citation.url) {
                const linkDiv = document.createElement('div');
                linkDiv.className = 'recommendation-links';
                
                const isBook = episodeNum.toString().startsWith('Book:');
                
                if (isBook) {
                    // For books, show available format links
                    if (citation.ebook_url || citation.url) {
                        const ebookLink = document.createElement('a');
                        ebookLink.href = citation.ebook_url || citation.url;
                        ebookLink.target = '_blank';
                        ebookLink.className = 'recommendation-link';
                        ebookLink.textContent = 'Read eBook →';
                        linkDiv.appendChild(ebookLink);
                    }
                    
                    if (citation.audiobook_url) {
                        if (linkDiv.children.length > 0) {
                            linkDiv.appendChild(document.createTextNode(' | '));
                        }
                        const audioLink = document.createElement('a');
                        audioLink.href = citation.audiobook_url;
                        audioLink.target = '_blank';
                        audioLink.className = 'recommendation-link';
                        audioLink.textContent = 'Listen to Audiobook →';
                        linkDiv.appendChild(audioLink);
                    }
                    
                    if (citation.print_url) {
                        if (linkDiv.children.length > 0) {
                            linkDiv.appendChild(document.createTextNode(' | '));
                        }
                        const printLink = document.createElement('a');
                        printLink.href = citation.print_url;
                        printLink.target = '_blank';
                        printLink.className = 'recommendation-link';
                        printLink.textContent = 'Order Print →';
                        linkDiv.appendChild(printLink);
                    }
                } else {
                    // For episodes, show single link
                    const link = document.createElement('a');
                    link.href = citation.url;
                    link.target = '_blank';
                    link.className = 'recommendation-link';
                    link.textContent = 'Listen Now →';
                    linkDiv.appendChild(link);
                }
                
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
            'hybrid': '🧬 Hybrid QA fuses knowledge graph data with BM25/vector search for enhanced accuracy',
            'both': '⚖️ Compare both methods side-by-side to see different perspectives'
        };

        this.ragDescText.textContent = descriptions[this.ragTypeSelect.value] || descriptions['original'];
    }
    
    updateModelDescription() {
        const descriptions = {
            'gpt-3.5-turbo': '🤖 GPT-3.5 Turbo: Fast, affordable, and great for most conversations',
            'gpt-4o-mini': '🧠 GPT-4o Mini: Balanced performance with improved reasoning',
            'gpt-4': '⭐ GPT-4: Highest quality responses with advanced reasoning',
            'compare': '🔄 Compare All Models: See responses from all three models side-by-side'
        };
        
        this.modelDescText.textContent = descriptions[this.modelTypeSelect.value] || descriptions['gpt-4o-mini'];
    }
    
    updateReferencesDescription() {
        const count = this.maxReferencesSelect.value;
        const plural = count === '1' ? 'episode' : 'episodes';
        this.referencesDescText.textContent = `📚 Show ${count} most relevant ${plural} per response`;
    }
    
    updateCategoryDescription() {
        const threshold = parseFloat(this.categoryThresholdSelect.value);
        const descriptions = {
            0.6: '📂 Broad matching finds many related topics like soil → biochar',
            0.7: '📂 Normal category matching finds related topics like soil → biochar', 
            0.8: '📂 Strict matching only finds very closely related categories',
            1.1: '📂 Category filtering disabled - searches all content'
        };
        
        this.categoryDescText.textContent = descriptions[threshold] || descriptions[0.7];
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
    
    async updateConversationRecommendations() {
        // Only call recommendations if we have conversation history
        if (this.conversationHistory.length === 0) {
            console.log('No conversation history, hiding recommendations');
            this.recommendations.style.display = 'none';
            return;
        }
        
        console.log('Calling conversation recommendations with history:', this.conversationHistory);
        
        try {
            const response = await fetch(`${this.apiUrl}/conversation-recommendations`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    conversation_history: this.conversationHistory,
                    max_recommendations: 4,
                    session_id: this.sessionId
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            console.log('Received recommendations data:', data);
            
            // Display the conversation-based recommendations
            this.showConversationRecommendations(data);
            
        } catch (error) {
            console.error('Error getting conversation recommendations:', error);
            // Fallback to hiding recommendations
            this.recommendations.style.display = 'none';
        }
    }
    
    showConversationRecommendations(data) {
        if (!data.recommendations || data.recommendations.length === 0) {
            this.recommendations.style.display = 'none';
            return;
        }
        
        // Clear previous recommendations
        this.recommendationsList.innerHTML = '';
        this.recommendationsList.className = 'recommendations-list';
        
        // Add conversation context header
        const contextHeader = document.createElement('div');
        contextHeader.className = 'recommendations-context';
        contextHeader.innerHTML = `
            <div style="margin-bottom: 1rem; font-size: 0.9rem; color: var(--warm-gray);">
                Based on our conversation about: ${data.conversation_topics.join(', ')}
            </div>
        `;
        this.recommendationsList.appendChild(contextHeader);
        
        // Add recommendations
        this.addRecommendationItems(this.recommendationsList, data.recommendations);
        
        // Add exploration suggestion if we have topics
        if (data.conversation_topics.length > 0) {
            const exploreMore = document.createElement('div');
            exploreMore.className = 'explore-more';
            exploreMore.innerHTML = `
                <div style="margin-top: 1rem; padding: 1rem; background: linear-gradient(135deg, #f0f8f0, #e8f5e8); border-radius: 8px; border-left: 3px solid var(--sage-green);">
                    <div style="font-weight: 500; color: var(--forest-green); margin-bottom: 0.5rem;">💡 Explore Related Topics</div>
                    <div style="font-size: 0.9rem; color: var(--earth-green);">
                        Try asking about: "other content on ${data.conversation_topics[0]}" or "what else about sustainable ${data.conversation_topics[1] || 'practices'}"
                    </div>
                </div>
            `;
            this.recommendationsList.appendChild(exploreMore);
        }
        
        this.recommendations.style.display = 'block';
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
        const maxRefs = parseInt(this.maxReferencesSelect.value);
        const episodesToShow = uniqueEpisodes.slice(0, maxRefs);
        
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
    
    showModelComparisonRecommendations(response) {
        // Combine episodes from all models
        const allEpisodes = Object.values(response.models).flatMap(model => 
            model.citations || []
        );
        
        // Add all citations to our cumulative tracking
        if (allEpisodes && allEpisodes.length > 0) {
            this.allCitedReferences = [...this.allCitedReferences, ...allEpisodes];
        }
        
        // Track episodes and update smart recommendations
        allEpisodes.forEach(episode => {
            if (episode.episode_number || episode.episode_id) {
                this.conversationEpisodes.add(episode.episode_number || episode.episode_id);
            }
        });
        
        // Extract topics from all responses
        const allResponses = Object.values(response.models).map(model => model.response || '').join(' ');
        const userMessage = this.lastUserMessage || '';
        const combinedText = (userMessage + ' ' + allResponses).toLowerCase();
        
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
        
        // Show smart recommendations with model comparison context
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
                    🤖 Episodes found by all AI models about: ${Array.from(this.conversationTopics).slice(0, 3).join(', ')}
                </div>
            `;
            this.recommendationsList.appendChild(contextHeader);
            
            // Add episode recommendations
            this.addRecommendationItems(this.recommendationsList, episodesToShow);
            
            this.recommendations.style.display = 'block';
        }
    }
    
    addModelComparisonMessage(response) {
        console.log('addModelComparisonMessage called with:', response);
        
        const comparisonDiv = document.createElement('div');
        comparisonDiv.className = 'message gaia-message';
        
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.textContent = '🌱';
        
        const content = document.createElement('div');
        content.className = 'message-content comparison-container model-comparison';
        
        // Create panels for each model
        const modelNames = {
            'gpt-3.5-turbo': '🤖 GPT-3.5 Turbo',
            'gpt-4o-mini': '🧠 GPT-4o Mini', 
            'gpt-4': '⭐ GPT-4'
        };
        
        console.log('Creating panels for models:', Object.keys(response.models));
        
        Object.entries(response.models).forEach(([modelName, modelResponse]) => {
            console.log(`Creating panel for ${modelName}:`, modelResponse);
            
            const panel = document.createElement('div');
            panel.className = 'comparison-panel model-panel';
            panel.innerHTML = `
                <div class="comparison-header">${modelNames[modelName] || modelName}</div>
                <div class="comparison-content">${modelResponse.response || 'No response available'}</div>
                <div class="model-info">
                    <small>⏱️ Model: ${modelResponse.model_used || modelName}</small>
                </div>
                ${this.formatComparisonCitations(modelResponse.citations || [], modelName)}
            `;
            content.appendChild(panel);
        });
        
        // Add processing time info
        const processingInfo = document.createElement('div');
        processingInfo.className = 'processing-info';
        processingInfo.innerHTML = `
            <small>⏱️ Total processing time: ${response.processing_time?.toFixed(2) || 'N/A'}s</small>
        `;
        content.appendChild(processingInfo);
        
        comparisonDiv.appendChild(avatar);
        comparisonDiv.appendChild(content);
        
        this.chatMessages.appendChild(comparisonDiv);
        this.scrollToBottom();
        
        // Show smart recommendations combining all models
        this.showModelComparisonRecommendations(response);
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
        
        // Add all citations to our cumulative tracking
        if (allEpisodes && allEpisodes.length > 0) {
            this.allCitedReferences = [...this.allCitedReferences, ...allEpisodes];
        }
        
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
    
    // Voice-related methods
    toggleVoice() {
        this.voiceEnabled = !this.voiceEnabled;
        localStorage.setItem('voiceEnabled', this.voiceEnabled);
        console.log('Voice toggled to:', this.voiceEnabled);
        this.updateVoiceUI();
    }
    
    updateVoiceUI() {
        if (this.voiceEnabled) {
            this.voiceToggle.classList.add('active');
            this.voiceToggleText.textContent = 'Voice Enabled';
            this.voiceSettings.style.display = 'block';
            this.voiceIcon.textContent = '🔊';
        } else {
            this.voiceToggle.classList.remove('active');
            this.voiceToggleText.textContent = 'Enable Voice';
            this.voiceSettings.style.display = 'none';
            this.voiceIcon.textContent = '🔇';
        }
        this.autoPlayVoice.checked = this.autoPlay;
        this.voiceSelection.value = this.selectedVoiceId;
    }
    
    async playAudio(audioData, messageId) {
        try {
            console.log(`Playing audio for message ${messageId}`);
            
            // Stop any currently playing audio
            if (!this.voicePlayer.paused) {
                console.log('Stopping current audio playback');
                this.voicePlayer.pause();
            }
            
            // Convert base64 to blob (Piper generates WAV files)
            const audioBlob = this.base64ToBlob(audioData, 'audio/wav');
            const audioUrl = URL.createObjectURL(audioBlob);
            
            // Store current audio URL for cleanup
            if (this.currentAudioUrl) {
                URL.revokeObjectURL(this.currentAudioUrl);
            }
            this.currentAudioUrl = audioUrl;
            
            // Update player
            this.voicePlayer.src = audioUrl;
            
            // Update UI to show playing state
            const playButton = document.querySelector(`#play-${messageId}`);
            if (playButton) {
                playButton.textContent = '⏸️ Pause';
                playButton.onclick = () => this.pauseAudio(messageId);
            }
            
            // Play audio
            await this.voicePlayer.play();
            console.log(`Audio playing for message ${messageId}`);
            
            // Clean up when done
            this.voicePlayer.addEventListener('ended', () => {
                console.log(`Audio ended for message ${messageId}`);
                URL.revokeObjectURL(audioUrl);
                this.currentAudioUrl = null;
                this.resetAudioControls(messageId);
            }, { once: true });
            
        } catch (error) {
            console.error('Error playing audio:', error);
            this.showAudioError(messageId);
        }
    }
    
    pauseAudio(messageId) {
        this.voicePlayer.pause();
        const playButton = document.querySelector(`#play-${messageId}`);
        if (playButton) {
            playButton.textContent = '▶️ Play';
            playButton.onclick = () => this.resumeAudio(messageId);
        }
    }
    
    resumeAudio(messageId) {
        this.voicePlayer.play();
        const playButton = document.querySelector(`#play-${messageId}`);
        if (playButton) {
            playButton.textContent = '⏸️ Pause';
            playButton.onclick = () => this.pauseAudio(messageId);
        }
    }
    
    resetAudioControls(messageId) {
        const playButton = document.querySelector(`#play-${messageId}`);
        if (playButton) {
            playButton.textContent = '▶️ Play Again';
            playButton.onclick = () => {
                const audioData = playButton.getAttribute('data-audio');
                this.playAudio(audioData, messageId);
            };
        }
    }
    
    showAudioError(messageId) {
        const statusElement = document.querySelector(`#audio-status-${messageId}`);
        if (statusElement) {
            statusElement.textContent = 'Error playing audio';
            statusElement.style.color = '#f44336';
        }
    }
    
    base64ToBlob(base64, mimeType) {
        const byteCharacters = atob(base64);
        const byteNumbers = new Array(byteCharacters.length);
        
        for (let i = 0; i < byteCharacters.length; i++) {
            byteNumbers[i] = byteCharacters.charCodeAt(i);
        }
        
        const byteArray = new Uint8Array(byteNumbers);
        return new Blob([byteArray], { type: mimeType });
    }
    
    createAudioControls(audioData, messageId) {
        const audioDiv = document.createElement('div');
        audioDiv.className = 'audio-controls';

        const playButton = document.createElement('button');
        playButton.className = 'audio-button';
        playButton.id = `play-${messageId}`;
        playButton.innerHTML = '<span>▶️</span> Play Gaia\'s Voice';
        playButton.setAttribute('data-audio', audioData);
        playButton.onclick = () => this.playAudio(audioData, messageId);

        // Add speed control dropdown
        const speedSelect = document.createElement('select');
        speedSelect.className = 'audio-speed-select';
        speedSelect.id = `speed-${messageId}`;
        speedSelect.innerHTML = `
            <option value="0.5">0.5x</option>
            <option value="0.75">0.75x</option>
            <option value="1" selected>1x</option>
            <option value="1.25">1.25x</option>
            <option value="1.5">1.5x</option>
            <option value="1.75">1.75x</option>
            <option value="2">2x</option>
        `;
        speedSelect.onchange = () => this.changePlaybackSpeed(parseFloat(speedSelect.value));

        const downloadButton = document.createElement('button');
        downloadButton.className = 'audio-button';
        downloadButton.innerHTML = '<span>💾</span> Download';
        downloadButton.onclick = () => this.downloadAudio(audioData, messageId);

        const statusDiv = document.createElement('div');
        statusDiv.className = 'audio-status';
        statusDiv.id = `audio-status-${messageId}`;

        audioDiv.appendChild(playButton);
        audioDiv.appendChild(speedSelect);
        audioDiv.appendChild(downloadButton);
        audioDiv.appendChild(statusDiv);

        return audioDiv;
    }
    
    changePlaybackSpeed(speed) {
        this.voicePlayer.playbackRate = speed;
        console.log(`Playback speed changed to ${speed}x`);
    }

    downloadAudio(audioData, messageId) {
        try {
            const audioBlob = this.base64ToBlob(audioData, 'audio/wav');
            const url = URL.createObjectURL(audioBlob);

            const a = document.createElement('a');
            a.href = url;
            a.download = `gaia-response-${messageId}.wav`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);

            URL.revokeObjectURL(url);
        } catch (error) {
            console.error('Error downloading audio:', error);
        }
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
