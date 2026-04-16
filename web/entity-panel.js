/**
 * Entity Panel - Collapsible right panel for showing entity details
 * Opens when entity links in chat responses are clicked
 */

(function() {
    let entityPanelOpen = false;
    let cachedKGData = null;

    // Fetch KG data for entity lookups
    async function loadKGData() {
        if (cachedKGData) return cachedKGData;

        try {
            const response = await fetch('/YonEarth/graph/data/graphrag_hierarchy/graphrag_hierarchy.json');
            if (response.ok) {
                cachedKGData = await response.json();
                console.log('KG data loaded:', Object.keys(cachedKGData.entities || {}).length, 'entities');
                return cachedKGData;
            }
        } catch (err) {
            console.log('Could not load KG data from local file:', err);
        }

        return null;
    }

    // Find entity in KG data - EXACT MATCH ONLY
    function findEntity(entityName, kgData) {
        if (!kgData || !kgData.entities) return null;

        const nameLower = entityName.toLowerCase().trim();

        // Exact match only (case-insensitive)
        for (const [name, info] of Object.entries(kgData.entities)) {
            if (name.toLowerCase() === nameLower) {
                return { name, ...info };
            }
        }

        return null;
    }

    // Find relationships for entity
    function findRelationships(entityName, kgData) {
        if (!kgData || !kgData.relationships) return { incoming: [], outgoing: [] };

        const nameLower = entityName.toLowerCase();
        const incoming = [];
        const outgoing = [];

        for (const rel of kgData.relationships) {
            if (rel.source && rel.source.toLowerCase() === nameLower) {
                outgoing.push(rel);
            }
            if (rel.target && rel.target.toLowerCase() === nameLower) {
                incoming.push(rel);
            }
        }

        return { incoming: incoming.slice(0, 10), outgoing: outgoing.slice(0, 10) };
    }

    // Open entity panel with details
    window.openEntityPanel = async function(entityName) {
        const panel = document.getElementById('entityPanel');
        const content = document.getElementById('entityPanelContent');
        const title = document.getElementById('entityPanelTitle');
        const mainContainer = document.getElementById('mainContainer');

        if (!panel || !content || !title) {
            console.error('Entity panel elements not found');
            return;
        }

        // Show loading
        title.textContent = entityName;
        content.innerHTML = '<p class="entity-loading">Loading entity details...</p>';

        // Open panel
        panel.classList.add('open');
        if (mainContainer) mainContainer.classList.add('entity-panel-open');
        entityPanelOpen = true;

        // Load KG data and find entity
        const kgData = await loadKGData();
        const entity = findEntity(entityName, kgData);

        if (!entity) {
            content.innerHTML = '<p class="entity-not-found">Entity "' + entityName + '" not found in the knowledge graph.</p>' +
                '<p><a href="graph/?search=' + encodeURIComponent(entityName) + '" target="_blank">Search in Knowledge Graph →</a></p>';
            return;
        }

        // Build entity details HTML
        let html = '';
        
        // Type badge
        if (entity.type) {
            html += '<div class="entity-type-badge">' + entity.type + '</div>';
        }

        // Description
        if (entity.description) {
            html += '<div class="entity-description">' + entity.description + '</div>';
        }

        // Relationships
        const rels = findRelationships(entityName, kgData);
        
        if (rels.outgoing.length > 0) {
            html += '<div class="entity-relationships">';
            html += '<h4>Connections</h4>';
            html += '<ul class="relationship-list">';
            for (const rel of rels.outgoing) {
                const targetName = rel.target.replace(/'/g, "\\'");
                html += '<li><a href="#" onclick="openEntityPanel(\'' + targetName + '\'); return false;">' + 
                    rel.target + '</a> <span class="rel-type">(' + (rel.predicate || 'related') + ')</span></li>';
            }
            html += '</ul></div>';
        }

        if (rels.incoming.length > 0) {
            html += '<div class="entity-relationships">';
            html += '<h4>Referenced By</h4>';
            html += '<ul class="relationship-list">';
            for (const rel of rels.incoming) {
                const sourceName = rel.source.replace(/'/g, "\\'");
                html += '<li><a href="#" onclick="openEntityPanel(\'' + sourceName + '\'); return false;">' + 
                    rel.source + '</a> <span class="rel-type">(' + (rel.predicate || 'related') + ')</span></li>';
            }
            html += '</ul></div>';
        }

        // View in KG link
        html += '<div class="entity-actions">';
        html += '<a href="graph/?entity=' + encodeURIComponent(entityName) + '" target="_blank" class="view-in-kg-btn">View in Knowledge Graph</a>';
        html += '</div>';

        content.innerHTML = html;
    };

    // Close entity panel
    window.closeEntityPanel = function() {
        const panel = document.getElementById('entityPanel');
        const mainContainer = document.getElementById('mainContainer');

        if (panel) panel.classList.remove('open');
        if (mainContainer) mainContainer.classList.remove('entity-panel-open');
        entityPanelOpen = false;
    };

    // Preload KG data on page load
    document.addEventListener('DOMContentLoaded', function() {
        loadKGData();
    });
})();
