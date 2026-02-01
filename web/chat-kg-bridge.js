/**
 * Chat-KG Bridge Module
 *
 * Provides communication between the chat interface and Knowledge Graph visualization.
 * Extracts entities from chat responses and finds matching nodes in the KG.
 *
 * Session 4A of GaiaAI Feedback Implementation
 */

const ChatKGBridge = {
    // Cache for KG data to avoid repeated API calls
    kgDataCache: null,
    kgDataCacheTime: null,
    CACHE_TTL: 5 * 60 * 1000, // 5 minutes

    /**
     * Extract potential entity names from a chat response
     * Looks for: capitalized terms, quoted terms, episode references, and topic keywords
     *
     * @param {string} responseText - The chat response text
     * @returns {string[]} - Array of potential entity terms to search for
     */
    extractEntities(responseText) {
        const entities = new Set();

        // 1. Extract episode references (e.g., "Episode 115", "Ep 108")
        const episodePattern = /\b(?:Episode|Ep\.?)\s*(\d+)(?:\s*[:\-]\s*([^,.!?\n]+))?/gi;
        let match;
        while ((match = episodePattern.exec(responseText)) !== null) {
            entities.add(`Episode ${match[1]}`);
            // If there's a guest/title after the number, extract that too
            if (match[2]) {
                const guestName = match[2].trim();
                if (guestName.length > 2 && guestName.length < 50) {
                    entities.add(guestName);
                }
            }
        }

        // 2. Extract quoted terms (likely important concepts or names)
        const quotedPattern = /"([^"]+)"|'([^']+)'/g;
        while ((match = quotedPattern.exec(responseText)) !== null) {
            const quoted = (match[1] || match[2]).trim();
            if (quoted.length > 2 && quoted.length < 50) {
                entities.add(quoted);
            }
        }

        // 3. Extract capitalized multi-word terms (likely proper nouns, organizations, concepts)
        // Matches sequences like "Regenerative Agriculture", "Dr. Jane Smith", "YonEarth Community"
        const capitalizedPattern = /\b([A-Z][a-z]+(?:\s+(?:of|the|and|for|in|on|to|with)?\s*[A-Z][a-z]+)+)\b/g;
        while ((match = capitalizedPattern.exec(responseText)) !== null) {
            const term = match[1].trim();
            // Filter out common sentence starters
            if (!term.match(/^(The|This|That|These|Those|When|Where|What|Which|Who|How|In|On|At|By|For)\s/i)) {
                entities.add(term);
            }
        }

        // 4. Extract single capitalized words that might be important (but not at sentence start)
        // Look for capitalized words after lowercase text
        const singleCapPattern = /[a-z,.]\s+([A-Z][a-z]{3,})\b/g;
        while ((match = singleCapPattern.exec(responseText)) !== null) {
            entities.add(match[1]);
        }

        // 5. Known important topic keywords (case-insensitive)
        const knownTopics = [
            'biochar', 'permaculture', 'regenerative', 'soil health', 'composting',
            'agroforestry', 'food sovereignty', 'indigenous', 'sustainability',
            'carbon sequestration', 'climate', 'biodiversity', 'ecosystem',
            'organic farming', 'no-till', 'cover crops', 'water conservation'
        ];

        const lowerText = responseText.toLowerCase();
        knownTopics.forEach(topic => {
            if (lowerText.includes(topic.toLowerCase())) {
                // Add the properly capitalized version
                entities.add(topic.split(' ').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' '));
            }
        });

        return Array.from(entities);
    },

    /**
     * Search the Knowledge Graph API for nodes matching the given terms
     *
     * @param {string[]} terms - Array of search terms
     * @returns {Promise<Object[]>} - Array of matching node objects with id, name, relevance
     */
    async findMatchingNodes(terms) {
        if (!terms || terms.length === 0) {
            return [];
        }

        const allMatches = new Map(); // Use Map to deduplicate by node ID

        // Search for each term
        for (const term of terms) {
            try {
                const response = await fetch(`/api/knowledge-graph/search?q=${encodeURIComponent(term)}&limit=5`);

                if (response.ok) {
                    const data = await response.json();
                    const results = data.results || data || [];

                    results.forEach(result => {
                        const node = result.entity || result;
                        const nodeId = node.id || node.name;

                        if (nodeId && !allMatches.has(nodeId)) {
                            allMatches.set(nodeId, {
                                id: nodeId,
                                name: node.name || nodeId,
                                type: node.type,
                                relevance: result.relevance_score || result.match_score || 1.0,
                                searchTerm: term,
                                domains: node.domains || [],
                                description: node.description || ''
                            });
                        }
                    });
                }
            } catch (error) {
                console.warn(`ChatKGBridge: Error searching for "${term}":`, error);
            }
        }

        // Sort by relevance and return
        return Array.from(allMatches.values())
            .sort((a, b) => b.relevance - a.relevance);
    },

    /**
     * Find matching nodes using cached KG data (faster, no API calls)
     * Falls back to API search if cache is empty
     *
     * @param {string[]} terms - Array of search terms
     * @returns {Promise<Object[]>} - Array of matching node objects
     */
    async findMatchingNodesLocal(terms) {
        // Ensure we have cached KG data
        if (!this.kgDataCache || (Date.now() - this.kgDataCacheTime) > this.CACHE_TTL) {
            try {
                const response = await fetch('/api/knowledge-graph/data');
                if (response.ok) {
                    this.kgDataCache = await response.json();
                    this.kgDataCacheTime = Date.now();
                } else {
                    // Fall back to API search
                    return this.findMatchingNodes(terms);
                }
            } catch (error) {
                console.warn('ChatKGBridge: Error loading KG data:', error);
                return this.findMatchingNodes(terms);
            }
        }

        const matches = new Map();
        const nodes = this.kgDataCache.nodes || [];

        terms.forEach(term => {
            const termLower = term.toLowerCase();

            nodes.forEach(node => {
                // Skip if already matched
                if (matches.has(node.id)) return;

                let score = 0;

                // Check name match
                if (node.name.toLowerCase() === termLower) {
                    score = 10; // Exact match
                } else if (node.name.toLowerCase().includes(termLower)) {
                    score = 7; // Partial name match
                }

                // Check aliases match
                if (node.aliases && Array.isArray(node.aliases)) {
                    const aliasMatch = node.aliases.some(alias =>
                        alias.toLowerCase() === termLower ||
                        alias.toLowerCase().includes(termLower)
                    );
                    if (aliasMatch) {
                        score = Math.max(score, 8); // Alias match
                    }
                }

                // Check description match
                if (score === 0 && node.description && node.description.toLowerCase().includes(termLower)) {
                    score = 3; // Description match
                }

                if (score > 0) {
                    // Boost by importance
                    score *= (1 + (node.importance || 0));

                    matches.set(node.id, {
                        id: node.id,
                        name: node.name,
                        type: node.type,
                        relevance: score,
                        searchTerm: term,
                        domains: node.domains || [],
                        description: node.description || ''
                    });
                }
            });
        });

        return Array.from(matches.values())
            .sort((a, b) => b.relevance - a.relevance);
    },

    /**
     * Trigger highlights in the Knowledge Graph for the given node IDs
     * Works with either inline KG or iframe approach
     *
     * @param {string[]} nodeIds - Array of node IDs to highlight
     */
    highlightInKG(nodeIds) {
        if (!nodeIds || nodeIds.length === 0) {
            console.log('ChatKGBridge: No nodes to highlight');
            return;
        }

        console.log('ChatKGBridge: Highlighting nodes:', nodeIds);

        // Try inline approach first (recommended)
        if (window.knowledgeGraph) {
            // Check if the new highlightEntities method exists (will be added in Session 4B)
            if (typeof window.knowledgeGraph.highlightEntities === 'function') {
                window.knowledgeGraph.highlightEntities(nodeIds);
            } else {
                // Fall back to focusing on the first node
                console.log('ChatKGBridge: highlightEntities not available, using focusOnNode');
                const firstNode = this.kgDataCache?.nodes?.find(n => n.id === nodeIds[0]);
                if (firstNode) {
                    window.knowledgeGraph.focusOnNode(firstNode);
                }
            }
        }

        // Also try BroadcastChannel for iframe approach (if needed)
        try {
            const channel = new BroadcastChannel('chat-kg-sync');
            channel.postMessage({
                type: 'highlight',
                nodeIds: nodeIds,
                timestamp: Date.now()
            });
            channel.close();
        } catch (e) {
            // BroadcastChannel not supported or other error - that's okay
            console.debug('ChatKGBridge: BroadcastChannel not available:', e.message);
        }
    },

    /**
     * Main entry point: Process a chat response and highlight matching KG nodes
     *
     * @param {string} responseText - The chat response text to process
     * @param {boolean} useLocalSearch - Use cached KG data instead of API (faster)
     * @returns {Promise<Object>} - Result object with extracted entities and matched nodes
     */
    async processResponse(responseText, useLocalSearch = true) {
        console.log('ChatKGBridge: Processing chat response...');

        // Extract entities from the response
        const entities = this.extractEntities(responseText);
        console.log('ChatKGBridge: Extracted entities:', entities);

        if (entities.length === 0) {
            return { entities: [], matches: [], highlighted: false };
        }

        // Find matching nodes
        const matches = useLocalSearch
            ? await this.findMatchingNodesLocal(entities)
            : await this.findMatchingNodes(entities);

        console.log('ChatKGBridge: Found matches:', matches.length);

        if (matches.length === 0) {
            return { entities, matches: [], highlighted: false };
        }

        // Highlight the top matches (limit to prevent overwhelming the KG)
        const topMatches = matches.slice(0, 10);
        const nodeIds = topMatches.map(m => m.id);

        this.highlightInKG(nodeIds);

        return {
            entities,
            matches: topMatches,
            highlighted: true
        };
    },

    /**
     * Clear any active highlights in the KG
     */
    clearHighlights() {
        if (window.knowledgeGraph) {
            if (typeof window.knowledgeGraph.clearHighlight === 'function') {
                window.knowledgeGraph.clearHighlight();
            }
        }

        // Also send clear message via BroadcastChannel
        try {
            const channel = new BroadcastChannel('chat-kg-sync');
            channel.postMessage({
                type: 'clear-highlights',
                timestamp: Date.now()
            });
            channel.close();
        } catch (e) {
            // BroadcastChannel not supported - that's okay
        }
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ChatKGBridge;
}

// Make available globally
window.ChatKGBridge = ChatKGBridge;

/**
 * Split View UI Controller
 * Manages the side-by-side Chat + Knowledge Graph layout
 */
const SplitViewController = {
    // State
    splitViewActive: false,
    kgIframe: null,
    originalContainer: null,

    /**
     * Initialize the split view controller
     */
    init() {
        const toggleButton = document.getElementById('splitViewToggle');

        if (!toggleButton) {
            console.log('SplitViewController: Toggle button not found');
            return;
        }

        // Store references
        this.originalContainer = document.querySelector('.container');
        this.kgIframe = document.getElementById('kgIframe');

        // Set up toggle button
        toggleButton.addEventListener('click', () => this.toggle());

        // Listen for messages from KG iframe
        window.addEventListener('message', (event) => this.handleKGMessage(event));

        // Set up BroadcastChannel for cross-frame communication
        try {
            this.channel = new BroadcastChannel('chat-kg-sync');
            this.channel.onmessage = (event) => this.handleChannelMessage(event);
        } catch (e) {
            console.log('SplitViewController: BroadcastChannel not supported');
        }

        console.log('SplitViewController: Initialized');
    },

    /**
     * Toggle split view on/off
     */
    toggle() {
        const splitViewContainer = document.getElementById('splitViewContainer');
        const toggleButton = document.getElementById('splitViewToggle');
        const toggleText = document.getElementById('splitViewText');
        const splitChatPanel = document.getElementById('splitChatPanel');

        if (!splitViewContainer || !this.originalContainer) {
            console.error('SplitViewController: Required elements not found');
            return;
        }

        this.splitViewActive = !this.splitViewActive;

        if (this.splitViewActive) {
            // Activate split view
            splitViewContainer.style.display = 'flex';

            // Move chat content to split view panel
            splitChatPanel.appendChild(this.originalContainer);

            // Update toggle button
            if (toggleButton) {
                toggleButton.classList.add('active');
            }
            if (toggleText) {
                toggleText.textContent = 'Hide Knowledge Graph';
            }

            console.log('SplitViewController: Split view activated');
        } else {
            // Deactivate split view
            document.body.insertBefore(this.originalContainer, splitViewContainer);
            splitViewContainer.style.display = 'none';

            // Update toggle button
            if (toggleButton) {
                toggleButton.classList.remove('active');
            }
            if (toggleText) {
                toggleText.textContent = 'Show Knowledge Graph';
            }

            console.log('SplitViewController: Split view deactivated');
        }
    },

    /**
     * Send highlight command to KG iframe
     * @param {string[]} entityNames - Entity names to highlight
     */
    highlightInIframe(entityNames) {
        if (!entityNames || entityNames.length === 0) return;

        console.log('SplitViewController: Sending highlight to iframe:', entityNames);

        if (this.kgIframe && this.kgIframe.contentWindow) {
            this.kgIframe.contentWindow.postMessage({
                type: 'highlightEntities',
                entities: entityNames
            }, '*');
        }

        if (this.channel) {
            this.channel.postMessage({
                type: 'highlightEntities',
                entities: entityNames
            });
        }
    },

    /**
     * Handle messages from KG iframe
     */
    handleKGMessage(event) {
        if (!event.data || !event.data.type) return;

        switch (event.data.type) {
            case 'kgReady':
                console.log('SplitViewController: KG iframe ready');
                break;

            case 'entitySelected':
                console.log('SplitViewController: Entity selected in KG:', event.data.entity);
                break;

            case 'highlightResult':
                console.log('SplitViewController: Highlight result:', event.data.matched);
                break;
        }
    },

    /**
     * Handle BroadcastChannel messages
     */
    handleChannelMessage(event) {
        if (!event.data || !event.data.type) return;

        if (event.data.type === 'highlightEntities') {
            // Forward to KG iframe
            if (this.kgIframe && this.kgIframe.contentWindow) {
                this.kgIframe.contentWindow.postMessage(event.data, '*');
            }
        }
    },

    /**
     * Show entity tags in a chat message
     * @param {HTMLElement} messageElement - The chat message element
     * @param {Object[]} matchedNodes - Array of matched node objects
     */
    showEntityIndicator(messageElement, matchedNodes) {
        if (!messageElement || !matchedNodes || matchedNodes.length === 0) return;

        // Check if indicator already exists
        if (messageElement.querySelector('.kg-entities-indicator')) return;

        const indicator = document.createElement('div');
        indicator.className = 'kg-entities-indicator';
        indicator.innerHTML = `
            <div class="indicator-label">Related in Knowledge Graph:</div>
            <div class="entity-tags">
                ${matchedNodes.slice(0, 5).map(node =>
                    `<span class="entity-tag" data-node-id="${node.id}">${node.name}</span>`
                ).join('')}
            </div>
        `;

        // Add click handlers to entity tags
        indicator.querySelectorAll('.entity-tag').forEach(tag => {
            tag.addEventListener('click', () => {
                const nodeId = tag.dataset.nodeId;
                const nodeName = tag.textContent;

                // Activate split view if not active
                if (!this.splitViewActive) {
                    this.toggle();
                }

                // Wait for iframe to be ready, then highlight
                setTimeout(() => {
                    this.highlightInIframe([nodeName]);
                }, 500);
            });
        });

        const messageContent = messageElement.querySelector('.message-content') || messageElement;
        messageContent.appendChild(indicator);
    }
};

// Global toggle function for inline onclick handlers
window.toggleSplitView = function() {
    SplitViewController.toggle();
};

// Global highlight function
window.highlightKGEntities = function(entityNames) {
    SplitViewController.highlightInIframe(entityNames);
};

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => SplitViewController.init());
} else {
    SplitViewController.init();
}

// Make controller available globally
window.SplitViewController = SplitViewController;
