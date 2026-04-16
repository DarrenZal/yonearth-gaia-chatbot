// GraphRAG 3D Visualization - Knowledge Graph Explorer
// Uses hierarchical clustering data from GraphRAG pipeline

(function() {
    'use strict';

    // Configuration
    const CONFIG = {
        nodeColors: {
            PERSON: '#4CAF50',
            ORGANIZATION: '#2196F3',
            CONCEPT: '#9C27B0',
            PLACE: '#FF9800',
            PRACTICE: '#00BCD4',
            PRODUCT: '#795548',
            EVENT: '#F44336',
            WORK: '#FFC107',
            // Cluster colors by level
            fine_cluster: '#00ff88',
            medium_cluster: '#00aaff',
            coarse_cluster: '#ff00ff'
        },
        nodeSizes: {
            entity: 2,
            fine_cluster: 4,
            medium_cluster: 8,
            coarse_cluster: 12
        },
        linkColors: {
            hierarchy: '#00ffff',
            relationship: '#ffffff',
            membership: '#ffff00'
        },
        cameraDistance: 800
    };

    // State
    let graphData = null;
    let graph = null;
    let currentLevel = 'all';
    let showRelationships = true;
    let selectedNode = null;

    // Initialize the visualization
    async function init() {
        try {
            // Load GraphRAG data
            console.log('Loading GraphRAG hierarchy data...');
            const response = await fetch('/data/graphrag_hierarchy/graphrag_hierarchy.json');
            if (!response.ok) {
                throw new Error(`Failed to load data: ${response.status}`);
            }

            const data = await response.json();
            console.log('GraphRAG data loaded:', {
                entities: Object.keys(data.entities || {}).length,
                relationships: (data.relationships || []).length,
                clusters: data.clusters ? Object.keys(data.clusters).length : 0
            });

            graphData = processGraphRAGData(data);
            console.log('Processed graph data:', {
                nodes: graphData.nodes.length,
                links: graphData.links.length
            });

            initGraph();
            setupEventListeners();

        } catch (error) {
            console.error('Failed to initialize GraphRAG visualization:', error);
            showError('Failed to load GraphRAG data. Please refresh the page.');
        }
    }

    // Process GraphRAG data into ForceGraph format
    function processGraphRAGData(data) {
        const nodes = [];
        const links = [];
        const nodeMap = new Map();

        // Process all cluster levels
        if (data.clusters) {
            // Level 3 - Coarse clusters
            if (data.clusters.level_3) {
                Object.values(data.clusters.level_3).forEach(cluster => {
                    const node = {
                        id: cluster.id,
                        name: cluster.name || `Category ${cluster.id}`,
                        type: 'coarse_cluster',
                        level: 3,
                        position: cluster.position || [0, 0, 0],
                        children: cluster.children || [],
                        color: CONFIG.nodeColors.coarse_cluster,
                        size: CONFIG.nodeSizes.coarse_cluster
                    };
                    nodes.push(node);
                    nodeMap.set(cluster.id, node);
                });
            }

            // Level 2 - Medium clusters
            if (data.clusters.level_2) {
                Object.values(data.clusters.level_2).forEach(cluster => {
                    const node = {
                        id: cluster.id,
                        name: cluster.name || `Topic ${cluster.id}`,
                        type: 'medium_cluster',
                        level: 2,
                        position: cluster.position || [0, 0, 0],
                        children: cluster.children || [],
                        color: CONFIG.nodeColors.medium_cluster,
                        size: CONFIG.nodeSizes.medium_cluster
                    };
                    nodes.push(node);
                    nodeMap.set(cluster.id, node);

                    // Create hierarchy links to parent clusters
                    cluster.children.forEach(childId => {
                        if (data.clusters.level_3) {
                            Object.values(data.clusters.level_3).forEach(parent => {
                                if (parent.children && parent.children.includes(cluster.id)) {
                                    links.push({
                                        source: parent.id,
                                        target: cluster.id,
                                        type: 'hierarchy',
                                        color: CONFIG.linkColors.hierarchy,
                                        width: 2
                                    });
                                }
                            });
                        }
                    });
                });
            }

            // Level 1 - Fine clusters
            if (data.clusters.level_1) {
                Object.values(data.clusters.level_1).forEach(cluster => {
                    const node = {
                        id: cluster.id,
                        name: cluster.name || `Cluster ${cluster.id}`,
                        type: 'fine_cluster',
                        level: 1,
                        position: cluster.position || [0, 0, 0],
                        entities: cluster.entities || [],
                        children: cluster.children || [],
                        color: CONFIG.nodeColors.fine_cluster,
                        size: CONFIG.nodeSizes.fine_cluster
                    };
                    nodes.push(node);
                    nodeMap.set(cluster.id, node);

                    // Create hierarchy links to parent clusters
                    if (data.clusters.level_2) {
                        Object.values(data.clusters.level_2).forEach(parent => {
                            if (parent.children && parent.children.includes(cluster.id)) {
                                links.push({
                                    source: parent.id,
                                    target: cluster.id,
                                    type: 'hierarchy',
                                    color: CONFIG.linkColors.hierarchy,
                                    width: 1.5
                                });
                            }
                        });
                    }
                });
            }

            // Level 0 - Individual entities
            if (data.clusters.level_0) {
                Object.entries(data.clusters.level_0).forEach(([entityId, cluster]) => {
                    const entity = data.entities[entityId] || {};
                    const node = {
                        id: entityId,
                        name: entity.name || entityId,
                        type: entity.type || 'CONCEPT',
                        level: 0,
                        position: cluster.position || [0, 0, 0],
                        description: entity.description,
                        sources: entity.sources,
                        related_clusters: cluster.related_clusters,
                        color: CONFIG.nodeColors[entity.type] || CONFIG.nodeColors.CONCEPT,
                        size: CONFIG.nodeSizes.entity
                    };
                    nodes.push(node);
                    nodeMap.set(entityId, node);

                    // Create hierarchy links to parent clusters
                    if (data.clusters.level_1) {
                        Object.values(data.clusters.level_1).forEach(parent => {
                            if (parent.entities && parent.entities.includes(entityId)) {
                                links.push({
                                    source: parent.id,
                                    target: entityId,
                                    type: 'hierarchy',
                                    color: CONFIG.linkColors.hierarchy,
                                    width: 1,
                                    opacity: 0.3
                                });
                            }
                        });
                    }

                    // Add multi-membership links
                    if (cluster.related_clusters) {
                        cluster.related_clusters.forEach(clusterId => {
                            if (nodeMap.has(clusterId)) {
                                links.push({
                                    source: clusterId,
                                    target: entityId,
                                    type: 'membership',
                                    color: CONFIG.linkColors.membership,
                                    width: 0.5,
                                    opacity: 0.5
                                });
                            }
                        });
                    }
                });
            }
        }

        // Add entity relationships
        if (showRelationships && data.relationships) {
            data.relationships.forEach(rel => {
                if (nodeMap.has(rel.source) && nodeMap.has(rel.target)) {
                    links.push({
                        source: rel.source,
                        target: rel.target,
                        type: 'relationship',
                        predicate: rel.predicate,
                        color: CONFIG.linkColors.relationship,
                        width: 0.5,
                        opacity: 0.2
                    });
                }
            });
        }

        return { nodes, links };
    }

    // Initialize the 3D graph
    function initGraph() {
        const container = document.getElementById('3d-graph');
        if (!container) {
            console.error('Graph container not found');
            return;
        }

        // Clear existing graph
        container.innerHTML = '';

        graph = ForceGraph3D()(container)
            .graphData(filterGraphByLevel(graphData))
            .nodeId('id')
            .nodeLabel(node => `${node.name}\n${node.type}${node.description ? '\n' + node.description.substring(0, 100) + '...' : ''}`)
            .nodeColor(node => node.color)
            .nodeVal(node => Math.pow(node.size, 2))
            .nodeOpacity(0.9)
            .linkSource('source')
            .linkTarget('target')
            .linkColor(link => link.color)
            .linkWidth(link => link.width || 1)
            .linkOpacity(link => link.opacity || 0.6)
            .linkDirectionalParticles(link => link.type === 'relationship' ? 2 : 0)
            .linkDirectionalParticleWidth(1)
            .linkDirectionalParticleSpeed(0.005)
            .backgroundColor('#0a0e1a')
            .showNavInfo(true);

        // Set initial camera position
        graph.cameraPosition(
            { x: 0, y: 0, z: CONFIG.cameraDistance },
            { x: 0, y: 0, z: 0 },
            0
        );

        // Node click handler
        graph.onNodeClick(handleNodeClick);

        // Node hover handler
        graph.onNodeHover(node => {
            container.style.cursor = node ? 'pointer' : 'default';
        });

        // Custom node rendering for better visibility
        graph.nodeThreeObject(node => {
            if (node.level >= 2) {
                // Render higher level clusters as spheres with glow
                const group = new THREE.Group();

                // Main sphere
                const geometry = new THREE.SphereGeometry(node.size);
                const material = new THREE.MeshPhongMaterial({
                    color: node.color,
                    emissive: node.color,
                    emissiveIntensity: 0.3,
                    shininess: 100
                });
                const sphere = new THREE.Mesh(geometry, material);
                group.add(sphere);

                // Glow effect
                const glowGeometry = new THREE.SphereGeometry(node.size * 1.5);
                const glowMaterial = new THREE.MeshBasicMaterial({
                    color: node.color,
                    opacity: 0.3,
                    transparent: true
                });
                const glow = new THREE.Mesh(glowGeometry, glowMaterial);
                group.add(glow);

                return group;
            }
            return false; // Use default rendering for entities and fine clusters
        });
    }

    // Filter graph data by selected level
    function filterGraphByLevel(data) {
        if (currentLevel === 'all') {
            return data;
        }

        const levelNum = parseInt(currentLevel);
        const filteredNodes = data.nodes.filter(node => {
            if (levelNum === 0) {
                return node.level === 0; // Only entities
            } else {
                return node.level >= levelNum; // Include selected level and above
            }
        });

        const nodeIds = new Set(filteredNodes.map(n => n.id));
        const filteredLinks = data.links.filter(link =>
            nodeIds.has(link.source) && nodeIds.has(link.target)
        );

        return { nodes: filteredNodes, links: filteredLinks };
    }

    // Handle node click
    function handleNodeClick(node) {
        if (!node) return;

        selectedNode = node;

        // Update info panel
        const panel = document.getElementById('node-info-panel');
        const title = document.getElementById('node-title');
        const details = document.getElementById('node-details');
        const relationships = document.getElementById('node-relationships');
        const multiMembership = document.getElementById('multi-membership');
        const relatedClusters = document.getElementById('related-clusters');

        if (panel && title && details) {
            title.textContent = node.name;

            // Update level indicator
            const levelDots = document.querySelectorAll('.level-dot');
            levelDots.forEach(dot => {
                dot.classList.remove('active');
                if (parseInt(dot.dataset.level) === node.level) {
                    dot.classList.add('active');
                }
            });

            // Build details HTML
            let detailsHTML = `
                <div class="entity-type-badge type-${node.type}">${node.type}</div>
                <p><strong>Level:</strong> ${getLevelName(node.level)}</p>
            `;

            if (node.description) {
                detailsHTML += `<p><strong>Description:</strong> ${node.description}</p>`;
            }

            if (node.sources && node.sources.length > 0) {
                detailsHTML += `<p><strong>Sources:</strong> ${node.sources.slice(0, 5).join(', ')}${node.sources.length > 5 ? '...' : ''}</p>`;
            }

            if (node.entities && node.entities.length > 0) {
                detailsHTML += `<p><strong>Contains ${node.entities.length} entities</strong></p>`;
            }

            if (node.children && node.children.length > 0) {
                detailsHTML += `<p><strong>Contains ${node.children.length} sub-clusters</strong></p>`;
            }

            details.innerHTML = detailsHTML;

            // Show multi-membership if applicable
            if (node.related_clusters && node.related_clusters.length > 0 && multiMembership && relatedClusters) {
                relatedClusters.innerHTML = node.related_clusters
                    .map(clusterId => {
                        const cluster = graphData.nodes.find(n => n.id === clusterId);
                        return cluster ? `<span class="cluster-badge">${cluster.name}</span>` : '';
                    })
                    .filter(Boolean)
                    .join(' ');
                multiMembership.style.display = 'block';
            } else if (multiMembership) {
                multiMembership.style.display = 'none';
            }

            // Show relationships
            if (relationships) {
                const nodeRelationships = graphData.links
                    .filter(link =>
                        (link.source === node.id || link.target === node.id) &&
                        link.type === 'relationship'
                    )
                    .slice(0, 10);

                if (nodeRelationships.length > 0) {
                    relationships.innerHTML = `
                        <h4>Relationships:</h4>
                        <ul>
                            ${nodeRelationships.map(rel => {
                                const otherNodeId = rel.source === node.id ? rel.target : rel.source;
                                const otherNode = graphData.nodes.find(n => n.id === otherNodeId);
                                return `<li>${rel.predicate || 'related to'} → ${otherNode ? otherNode.name : otherNodeId}</li>`;
                            }).join('')}
                        </ul>
                    `;
                    relationships.style.display = 'block';
                } else {
                    relationships.style.display = 'none';
                }
            }

            panel.style.display = 'block';
        }

        // Focus camera on node
        const distance = node.level === 0 ? 200 : 300 + (node.level * 100);
        graph.cameraPosition(
            { x: node.position[0], y: node.position[1], z: node.position[2] + distance },
            { x: node.position[0], y: node.position[1], z: node.position[2] },
            1000
        );

        // Highlight connected nodes
        highlightConnections(node);
    }

    // Highlight node connections
    function highlightConnections(node) {
        if (!graph) return;

        const connectedNodeIds = new Set([node.id]);

        // Find all connected nodes
        graphData.links.forEach(link => {
            if (link.source === node.id) {
                connectedNodeIds.add(link.target);
            } else if (link.target === node.id) {
                connectedNodeIds.add(link.source);
            }
        });

        // Update node colors
        graph.nodeColor(n => {
            if (n.id === node.id) {
                return '#ffff00'; // Highlight selected node
            } else if (connectedNodeIds.has(n.id)) {
                return n.color; // Keep connected nodes normal color
            } else {
                return `${n.color}33`; // Dim unconnected nodes
            }
        });

        // Update link visibility
        graph.linkOpacity(link => {
            if (link.source === node.id || link.target === node.id) {
                return 1; // Full opacity for connected links
            } else {
                return 0.1; // Dim other links
            }
        });
    }

    // Get human-readable level name
    function getLevelName(level) {
        switch(level) {
            case 0: return 'Entity';
            case 1: return 'Fine Cluster';
            case 2: return 'Medium Cluster';
            case 3: return 'Coarse Category';
            default: return `Level ${level}`;
        }
    }

    // Setup event listeners
    function setupEventListeners() {
        // Level selector
        const levelSelector = document.getElementById('view-level');
        if (levelSelector) {
            levelSelector.addEventListener('change', (e) => {
                currentLevel = e.target.value;
                if (graph) {
                    graph.graphData(filterGraphByLevel(graphData));
                }
            });
        }

        // Relationship toggle
        const relToggle = document.getElementById('show-relationships');
        if (relToggle) {
            relToggle.addEventListener('change', async (e) => {
                showRelationships = e.target.checked;
                if (graphData) {
                    // Reprocess data with updated relationship setting
                    const response = await fetch('/data/graphrag_hierarchy/graphrag_hierarchy.json');
                    const data = await response.json();
                    graphData = processGraphRAGData(data);
                    if (graph) {
                        graph.graphData(filterGraphByLevel(graphData));
                    }
                }
            });
        }

        // Mobile menu toggle
        const mobileToggle = document.getElementById('mobile-menu-toggle');
        const headerControls = document.getElementById('header-controls');
        if (mobileToggle && headerControls) {
            mobileToggle.addEventListener('click', () => {
                headerControls.classList.toggle('show');
            });
        }
    }

    // Show error message
    function showError(message) {
        const container = document.getElementById('3d-graph');
        if (container) {
            container.innerHTML = `
                <div style="color: #ff6b6b; padding: 20px; text-align: center;">
                    <h3>Error Loading GraphRAG Visualization</h3>
                    <p>${message}</p>
                </div>
            `;
        }
    }

    // Close info panel
    window.graphRAG3D = {
        closeInfoPanel: function() {
            const panel = document.getElementById('node-info-panel');
            if (panel) {
                panel.style.display = 'none';
            }

            // Reset highlighting
            if (graph) {
                graph.nodeColor(node => node.color);
                graph.linkOpacity(link => link.opacity || 0.6);
            }

            selectedNode = null;
        }
    };

    // Initialize on DOM ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();