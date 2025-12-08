/**
 * GraphRAG Comparison Chat Interface
 * Displays side-by-side responses from BM25 and GraphRAG systems
 */

class GraphRAGChat {
    constructor() {
        this.apiUrl = '/api';
        this.bm25Messages = document.getElementById('bm25Messages');
        this.graphragMessages = document.getElementById('graphragMessages');
        this.messageInput = document.getElementById('messageInput');
        this.chatForm = document.getElementById('chatForm');
        this.comparisonLoading = document.getElementById('comparisonLoading');
        this.comparisonMetrics = document.getElementById('comparisonMetrics');
        this.graphragMetadata = document.getElementById('graphragMetadata');
        this.statusElement = document.getElementById('status');

        // Settings elements
        this.personalitySelect = document.getElementById('personality');
        this.searchModeSelect = document.getElementById('searchMode');
        this.communityLevelSelect = document.getElementById('communityLevel');
        this.kCommunitiesSelect = document.getElementById('kCommunities');
        this.kEntitiesSelect = document.getElementById('kEntities');

        this.isLoading = false;

        this.init();
    }

    init() {
        // Form submission
        this.chatForm.addEventListener('submit', (e) => {
            e.preventDefault();
            this.sendMessage();
        });

        // Search mode description updates
        this.searchModeSelect.addEventListener('change', () => {
            this.updateModeDescription();
        });

        // Check API health
        this.checkHealth();
    }

    updateModeDescription() {
        const mode = this.searchModeSelect.value;
        const descText = document.getElementById('modeDescText');

        const descriptions = {
            'drift': 'DRIFT combines community summaries with entity details for comprehensive answers',
            'global': 'Global search uses community summaries - best for broad thematic questions',
            'local': 'Local search extracts entities and relationships - best for specific queries',
            'auto': 'Auto-detect chooses the best mode based on your question'
        };

        descText.textContent = descriptions[mode] || descriptions['drift'];
    }

    async checkHealth() {
        try {
            const response = await fetch(`${this.apiUrl}/graphrag/health`);
            const data = await response.json();

            if (data.initialized) {
                this.statusElement.textContent = 'GraphRAG Ready';
                this.statusElement.style.color = '#4caf50';
            } else {
                this.statusElement.textContent = 'GraphRAG Initializing...';
                this.statusElement.style.color = '#ff9800';
            }
        } catch (error) {
            this.statusElement.textContent = 'GraphRAG Unavailable';
            this.statusElement.style.color = '#f44336';
        }
    }

    async sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message || this.isLoading) return;

        this.isLoading = true;
        this.messageInput.value = '';

        // Add user message to both panels
        this.addMessage(this.bm25Messages, message, 'user');
        this.addMessage(this.graphragMessages, message, 'user');

        // Show loading
        this.comparisonLoading.style.display = 'flex';

        try {
            // Call both APIs in parallel
            const [bm25Response, graphragResponse] = await Promise.all([
                this.callBM25API(message),
                this.callGraphRAGAPI(message)
            ]);

            // Hide loading
            this.comparisonLoading.style.display = 'none';

            // Display responses
            this.displayBM25Response(bm25Response);
            this.displayGraphRAGResponse(graphragResponse);

            // Update comparison metrics
            this.updateComparisonMetrics(bm25Response, graphragResponse);

        } catch (error) {
            console.error('Error:', error);
            this.comparisonLoading.style.display = 'none';
            this.addMessage(this.bm25Messages, 'Error: ' + error.message, 'error');
            this.addMessage(this.graphragMessages, 'Error: ' + error.message, 'error');
        }

        this.isLoading = false;
    }

    async callBM25API(message) {
        const startTime = performance.now();

        const response = await fetch(`${this.apiUrl}/bm25/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: message,
                search_method: 'hybrid',
                k: 5,
                include_sources: true,
                gaia_personality: this.personalitySelect.value
            })
        });

        const data = await response.json();
        data.client_time = performance.now() - startTime;
        return data;
    }

    async callGraphRAGAPI(message) {
        const startTime = performance.now();

        const response = await fetch(`${this.apiUrl}/graphrag/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: message,
                search_mode: this.searchModeSelect.value,
                community_level: parseInt(this.communityLevelSelect.value),
                k_communities: parseInt(this.kCommunitiesSelect.value),
                k_entities: parseInt(this.kEntitiesSelect.value),
                k_chunks: 3,
                personality: this.personalitySelect.value
            })
        });

        const data = await response.json();
        data.client_time = performance.now() - startTime;
        return data;
    }

    addMessage(container, content, type) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}-message`;

        if (type === 'user') {
            messageDiv.innerHTML = `
                <div class="message-avatar">👤</div>
                <div class="message-content">
                    <div class="message-text">${this.escapeHtml(content)}</div>
                </div>
            `;
        } else if (type === 'error') {
            messageDiv.innerHTML = `
                <div class="message-avatar">⚠️</div>
                <div class="message-content">
                    <div class="message-text error-text">${this.escapeHtml(content)}</div>
                </div>
            `;
        }

        container.appendChild(messageDiv);
        container.scrollTop = container.scrollHeight;
    }

    displayBM25Response(data) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message gaia-message';

        let sourcesHtml = '';
        if (data.sources && data.sources.length > 0) {
            sourcesHtml = '<div class="sources"><strong>Sources:</strong><ul>';
            data.sources.forEach(source => {
                const title = source.title || `Episode ${source.episode_number}`;
                sourcesHtml += `<li>${this.escapeHtml(title)}</li>`;
            });
            sourcesHtml += '</ul></div>';
        }

        const timeHtml = data.processing_time
            ? `<div class="response-time">⏱️ ${data.processing_time.toFixed(2)}s (server) / ${(data.client_time/1000).toFixed(2)}s (total)</div>`
            : '';

        messageDiv.innerHTML = `
            <div class="message-avatar">🌱</div>
            <div class="message-content">
                <div class="message-text">${this.formatResponse(data.response)}</div>
                ${sourcesHtml}
                ${timeHtml}
            </div>
        `;

        this.bm25Messages.appendChild(messageDiv);
        this.bm25Messages.scrollTop = this.bm25Messages.scrollHeight;
    }

    displayGraphRAGResponse(data) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message gaia-message';

        const timeHtml = data.processing_time
            ? `<div class="response-time">⏱️ ${data.processing_time.toFixed(2)}s (server) / ${(data.client_time/1000).toFixed(2)}s (total)</div>`
            : '';

        const modeHtml = `<div class="search-mode-badge">${data.search_mode || 'drift'} mode</div>`;

        messageDiv.innerHTML = `
            <div class="message-avatar">🌱</div>
            <div class="message-content">
                <div class="message-text">${this.formatResponse(data.response)}</div>
                ${modeHtml}
                ${timeHtml}
            </div>
        `;

        this.graphragMessages.appendChild(messageDiv);
        this.graphragMessages.scrollTop = this.graphragMessages.scrollHeight;

        // Update metadata panel
        this.displayGraphRAGMetadata(data);
    }

    displayGraphRAGMetadata(data) {
        this.graphragMetadata.style.display = 'block';

        // Communities
        const communitiesList = document.getElementById('communitiesList');
        communitiesList.innerHTML = '';

        if (data.communities_used && data.communities_used.length > 0) {
            data.communities_used.forEach(comm => {
                const div = document.createElement('div');
                div.className = 'metadata-item community-item';
                div.innerHTML = `
                    <div class="item-header">
                        <span class="item-title">${this.escapeHtml(comm.title || comm.name)}</span>
                        <span class="item-score">${(comm.relevance_score * 100).toFixed(0)}%</span>
                    </div>
                    <div class="item-detail">Level ${comm.level} • ${comm.entity_count} entities</div>
                    <div class="item-summary">${this.escapeHtml(comm.summary)}</div>
                `;
                communitiesList.appendChild(div);
            });
        } else {
            communitiesList.innerHTML = '<div class="no-data">No communities matched</div>';
        }

        // Entities
        const entitiesList = document.getElementById('entitiesList');
        entitiesList.innerHTML = '';

        if (data.entities_matched && data.entities_matched.length > 0) {
            data.entities_matched.forEach(entity => {
                const div = document.createElement('div');
                div.className = 'metadata-item entity-item';
                div.innerHTML = `
                    <div class="item-header">
                        <span class="item-title">${this.escapeHtml(entity.name)}</span>
                        <span class="item-type">${entity.type}</span>
                    </div>
                    <div class="item-detail">${entity.mention_count} mentions</div>
                    <div class="item-summary">${this.escapeHtml(entity.description)}</div>
                `;
                entitiesList.appendChild(div);
            });
        } else {
            entitiesList.innerHTML = '<div class="no-data">No entities extracted</div>';
        }

        // Relationships
        const relationshipsList = document.getElementById('relationshipsList');
        relationshipsList.innerHTML = '';

        if (data.relationships && data.relationships.length > 0) {
            data.relationships.forEach(rel => {
                const div = document.createElement('div');
                div.className = 'metadata-item relationship-item';
                div.innerHTML = `
                    <span class="rel-source">${this.escapeHtml(rel.source)}</span>
                    <span class="rel-predicate">${rel.predicate}</span>
                    <span class="rel-target">${this.escapeHtml(rel.target)}</span>
                `;
                relationshipsList.appendChild(div);
            });
        } else {
            relationshipsList.innerHTML = '<div class="no-data">No relationships found</div>';
        }
    }

    updateComparisonMetrics(bm25Data, graphragData) {
        this.comparisonMetrics.style.display = 'block';

        // Time comparison
        const bm25Time = bm25Data.processing_time || 0;
        const graphragTime = graphragData.processing_time || 0;
        const faster = bm25Time < graphragTime ? 'BM25' : 'GraphRAG';
        document.getElementById('metricTime').innerHTML = `
            BM25: ${bm25Time.toFixed(2)}s<br>
            GraphRAG: ${graphragTime.toFixed(2)}s<br>
            <small>${faster} faster</small>
        `;

        // Episode overlap
        const bm25Episodes = new Set((bm25Data.episode_references || []).map(e => String(e)));
        const graphragEpisodes = new Set((graphragData.source_episodes || []).map(e => String(e)));

        let overlapCount = 0;
        bm25Episodes.forEach(ep => {
            if (graphragEpisodes.has(ep)) overlapCount++;
        });

        const totalUnique = new Set([...bm25Episodes, ...graphragEpisodes]).size;
        const overlapPercent = totalUnique > 0 ? (overlapCount / totalUnique * 100).toFixed(0) : 0;

        document.getElementById('metricOverlap').innerHTML = `
            ${overlapCount} common<br>
            ${overlapPercent}% overlap
        `;

        // GraphRAG communities
        const communities = graphragData.communities_used || [];
        document.getElementById('metricCommunities').innerHTML = `
            ${communities.length} matched<br>
            <small>${communities.slice(0, 2).map(c => c.title || c.name).join(', ')}</small>
        `;

        // GraphRAG entities
        const entities = graphragData.entities_matched || [];
        document.getElementById('metricEntities').innerHTML = `
            ${entities.length} extracted<br>
            <small>${entities.slice(0, 3).map(e => e.name).join(', ')}</small>
        `;
    }

    formatResponse(text) {
        if (!text) return '';

        // Convert markdown-style formatting
        let formatted = this.escapeHtml(text);

        // Bold
        formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

        // Italic
        formatted = formatted.replace(/\*(.*?)\*/g, '<em>$1</em>');

        // Line breaks
        formatted = formatted.replace(/\n/g, '<br>');

        return formatted;
    }

    escapeHtml(text) {
        if (!text) return '';
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    window.graphragChat = new GraphRAGChat();
});
