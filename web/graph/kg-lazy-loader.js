/**
 * Lazy Loading Enhancement for Knowledge Graph
 * 
 * Patches the GraphRAG3DEmbeddingView to:
 * 1. Load subset (413 entities) initially for fast startup
 * 2. Expand neighbors on entity click
 * 3. Search returns matches from full dataset
 */

(function() {
    // Track loaded entities to avoid duplicates
    window.loadedEntityIds = new Set();
    window.isSubsetMode = false;
    
    // Store original click handler
    let originalHandleClick = null;
    
    /**
     * Fetch neighbors for an entity from the API
     */
    async function fetchEntityNeighbors(entityName, depth = 1) {
        try {
            const encodedName = encodeURIComponent(entityName);
            const response = await fetch(`/api/graphrag/entity/${encodedName}?depth=${depth}&max_neighbors=50`);
            if (!response.ok) {
                console.warn(`Failed to fetch neighbors for ${entityName}:`, response.status);
                return null;
            }
            return await response.json();
        } catch (err) {
            console.error('Error fetching neighbors:', err);
            return null;
        }
    }
    
    /**
     * Merge new entities into the graph
     */
    function mergeEntitiesIntoGraph(viewer, neighborData) {
        if (!neighborData || !neighborData.neighbors) return 0;
        
        const data = viewer.data;
        if (!data || !data.entities) return 0;
        
        let addedCount = 0;
        
        // Add new entities
        for (const [name, info] of Object.entries(neighborData.neighbors)) {
            if (!window.loadedEntityIds.has(name)) {
                // Add to entities
                data.entities[name] = {
                    name: name,
                    type: info.type,
                    description: info.description || '',
                    connections: 0, // Will be calculated
                    expanded_from: neighborData.entity.id,
                    lazy_loaded: true
                };
                window.loadedEntityIds.add(name);
                addedCount++;
            }
        }
        
        // Add new relationships
        if (neighborData.edges && data.relationships) {
            for (const edge of neighborData.edges) {
                data.relationships.push({
                    source: edge.source,
                    target: edge.target,
                    predicate: edge.predicate,
                    lazy_loaded: true
                });
            }
        }
        
        return addedCount;
    }
    
    /**
     * Show expand indicator on entity info panel
     */
    function showExpandButton(entityName, viewer) {
        const panel = document.getElementById('entity-info');
        if (!panel) return;
        
        // Check if button already exists
        if (document.getElementById('expand-neighbors-btn')) return;
        
        const btn = document.createElement('button');
        btn.id = 'expand-neighbors-btn';
        btn.innerHTML = '🔍 Load Neighbors';
        btn.style.cssText = `
            margin-top: 12px;
            padding: 8px 16px;
            background: rgba(100, 150, 255, 0.3);
            border: 1px solid rgba(100, 150, 255, 0.5);
            border-radius: 6px;
            color: #c0d8ff;
            cursor: pointer;
            font-size: 13px;
            width: 100%;
        `;
        btn.onclick = async () => {
            btn.innerHTML = '⏳ Loading...';
            btn.disabled = true;
            
            const neighbors = await fetchEntityNeighbors(entityName, 1);
            if (neighbors) {
                const added = mergeEntitiesIntoGraph(viewer, neighbors);
                if (added > 0) {
                    btn.innerHTML = `✅ Added ${added} entities`;
                    // Trigger graph re-render
                    if (viewer.buildEntitiesFromData) {
                        viewer.buildEntitiesFromData();
                    }
                    if (viewer.renderCurrentView) {
                        viewer.renderCurrentView();
                    }
                } else {
                    btn.innerHTML = '✓ All neighbors loaded';
                }
            } else {
                btn.innerHTML = '❌ Failed to load';
            }
            
            setTimeout(() => btn.remove(), 3000);
        };
        
        panel.appendChild(btn);
    }
    
    /**
     * Initialize lazy loading when viewer is ready
     */
    function initLazyLoading() {
        // Wait for viewer to be available
        if (!window.viewer) {
            setTimeout(initLazyLoading, 500);
            return;
        }
        
        const viewer = window.viewer;
        
        // Check if subset is loaded
        if (viewer.data && viewer.data.metadata && viewer.data.metadata.subset) {
            window.isSubsetMode = true;
            console.log('📦 Lazy loading mode: subset of', viewer.data.metadata.subset_entity_count, 
                        'entities (full:', viewer.data.metadata.full_entity_count, ')');
            
            // Track loaded entities
            if (viewer.data.entities) {
                for (const name of Object.keys(viewer.data.entities)) {
                    window.loadedEntityIds.add(name);
                }
            }
            
            // Update status display
            const statusEl = document.querySelector('.hierarchy-block');
            if (statusEl) {
                const subsetNote = document.createElement('div');
                subsetNote.style.cssText = 'margin-top: 8px; font-size: 12px; color: #88ccff;';
                subsetNote.innerHTML = `
                    📦 Showing ${viewer.data.metadata.subset_entity_count} key entities
                    <br><span style="color: #808080;">Click entities to expand • Search finds all ${viewer.data.metadata.full_entity_count}</span>
                `;
                statusEl.appendChild(subsetNote);
            }
        }
        
        // Patch entity click handler to show expand button
        if (viewer.handleEntityClick) {
            originalHandleClick = viewer.handleEntityClick.bind(viewer);
            viewer.handleEntityClick = function(entity) {
                originalHandleClick(entity);
                if (window.isSubsetMode && entity && entity.name) {
                    setTimeout(() => showExpandButton(entity.name, viewer), 100);
                }
            };
        }
        
        console.log('✅ Lazy loading initialized');
    }
    
    // Start when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => setTimeout(initLazyLoading, 1000));
    } else {
        setTimeout(initLazyLoading, 1000);
    }
})();
