// GraphRAG 3D Visualization v2 - Progressive Disclosure
// Starts with high-level categories, expands on interaction

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
            entity: 1,
            fine_cluster: 3,
            medium_cluster: 6,
            coarse_cluster: 12
        },
        linkColors: {
            hierarchy: '#00ffff',
            relationship: '#ffffff',
            membership: '#ffff00'
        },
        initialCameraDistance: 400,
        expandedCameraDistance: 200
    };

    // State
    let fullData = null;
    let graph = null;
    let currentView = 'categories'; // 'categories', 'expanded', 'search'
    let expandedNodes = new Set();
    let selectedNode = null;
    let searchResults = [];
    let nodePositionCache = new Map();
    let chunkTimestamps = {}; // Store chunk->timestamp mappings

    // Make graph accessible globally for debugging
    window.graphDebug = {
        getGraph: () => graph,
        getExpandedNodes: () => expandedNodes,
        getData: () => fullData,
        expandNode: (nodeId) => expandNode(nodeId),
        collapseNode: (nodeId) => collapseNode(nodeId)
    };

    // Initialize the visualization
    async function init() {
        try {
            // Show loading indicator
            showLoading(true);
            updateLoadingStatus('Loading full dataset (39,046 entities)...');

            // Determine which dataset to load
            const dataset = localStorage.getItem('graphrag-dataset') || 'full';
            let dataPath = '/data/graphrag_full/graphrag_full.json';

            if (dataset === 'standard') {
                dataPath = '/data/graphrag_hierarchy/graphrag_hierarchy.json';
                updateLoadingStatus('Loading standard dataset (12,287 entities)...');
            }

            // Load GraphRAG data
            console.log(`Loading GraphRAG data from ${dataPath}...`);
            const response = await fetch(dataPath);
            if (!response.ok) {
                throw new Error(`Failed to load data: ${response.status}`);
            }

            const data = await response.json();
            console.log('GraphRAG data loaded:', {
                entities: Object.keys(data.entities || {}).length,
                relationships: (data.relationships || []).length,
                clusters: data.clusters ? Object.keys(data.clusters).length : 0
            });

            // Load chunk timestamp mappings
            console.log('Loading chunk timestamp mappings...');
            try {
                const timestampResponse = await fetch('/data/chunk_timestamps.json');
                if (timestampResponse.ok) {
                    chunkTimestamps = await timestampResponse.json();
                    console.log(`Loaded ${Object.keys(chunkTimestamps).length} chunk timestamp mappings`);
                } else {
                    console.warn('Could not load chunk timestamps, will use default times');
                }
            } catch (error) {
                console.warn('Error loading chunk timestamps:', error);
            }

            // Process and cache full data
            fullData = data;
            preprocessData(data);

            // Initialize with categories view
            initGraph();
            showCategoriesOnly();
            setupEventListeners();

            showLoading(false);

        } catch (error) {
            console.error('Failed to initialize GraphRAG visualization:', error);
            showError('Failed to load GraphRAG data. Please refresh the page.');
            showLoading(false);
        }
    }

    // Preprocess data for efficient access
    function preprocessData(data) {
        // Build parent-child relationships
        if (data.clusters) {
            // Cache positions for smooth transitions
            Object.values(data.clusters.level_3 || {}).forEach(cluster => {
                nodePositionCache.set(cluster.id, cluster.position || [0, 0, 0]);
            });
            Object.values(data.clusters.level_2 || {}).forEach(cluster => {
                nodePositionCache.set(cluster.id, cluster.position || [0, 0, 0]);
            });
            Object.values(data.clusters.level_1 || {}).forEach(cluster => {
                nodePositionCache.set(cluster.id, cluster.position || [0, 0, 0]);
            });
            Object.values(data.clusters.level_0 || {}).forEach(cluster => {
                nodePositionCache.set(cluster.id, cluster.position || [0, 0, 0]);
            });
        }
    }

    // Show only top-level categories
    function showCategoriesOnly() {
        const nodes = [];
        const links = [];

        console.log('Showing categories, fullData.clusters:', fullData.clusters);

        if (!fullData.clusters || !fullData.clusters.level_3) {
            console.error('No level 3 clusters found, available levels:', fullData.clusters ? Object.keys(fullData.clusters) : 'none');
            showError('No category data found in GraphRAG hierarchy');
            return;
        }

        // Add only level 3 (coarse) clusters
        const l3Clusters = Object.values(fullData.clusters.level_3);
        const angleStep = (2 * Math.PI) / l3Clusters.length;
        const radius = 150;

        l3Clusters.forEach((cluster, index) => {
            const angle = index * angleStep;
            const node = {
                id: cluster.id,
                name: getCategoryName(cluster, index),
                type: 'coarse_cluster',
                level: 3,
                position: [
                    Math.cos(angle) * radius,
                    Math.sin(angle) * radius,
                    0
                ],
                children: cluster.children || [],
                childCount: getDescendantCount(cluster),
                color: CONFIG.nodeColors.coarse_cluster,
                size: CONFIG.nodeSizes.coarse_cluster,
                expanded: expandedNodes.has(cluster.id)
            };
            nodes.push(node);
            nodePositionCache.set(cluster.id, node.position);
        });

        // Connect categories in a ring for visual appeal
        for (let i = 0; i < nodes.length; i++) {
            links.push({
                source: nodes[i].id,
                target: nodes[(i + 1) % nodes.length].id,
                type: 'category_ring',
                color: '#00ffff33',
                width: 0.5,
                opacity: 0.3
            });
        }

        console.log('Categories graph data:', { nodes: nodes.length, links: links.length });
        updateGraph({ nodes, links });
    }

    // Get human-readable category name
    function getCategoryName(cluster, index) {
        // Try to get name from cluster data or generate one
        const defaultNames = [
            'Environment & Ecology',
            'Social Systems',
            'Technology & Innovation',
            'Health & Wellness',
            'Economics & Business',
            'Culture & Community'
        ];
        return cluster.name || defaultNames[index] || `Category ${index + 1}`;
    }

    // Count all descendants of a cluster
    function getDescendantCount(cluster) {
        if (!cluster.children || !fullData.clusters) return 0;

        let count = 0;

        if (cluster.type === 'coarse_cluster' && fullData.clusters.level_2) {
            cluster.children.forEach(l2Id => {
                const l2 = fullData.clusters.level_2[l2Id];
                if (l2 && l2.children && fullData.clusters.level_1) {
                    l2.children.forEach(l1Id => {
                        const l1 = fullData.clusters.level_1[l1Id];
                        if (l1 && l1.entities) {
                            count += l1.entities.length;
                        }
                    });
                }
            });
        }

        return count;
    }

    // Expand a node to show its children
    function expandNode(nodeId) {
        if (expandedNodes.has(nodeId)) {
            collapseNode(nodeId);
            return;
        }

        expandedNodes.add(nodeId);
        const expandedData = buildExpandedGraph();
        updateGraph(expandedData);

        // Don't auto-zoom for level 3 (top category) nodes
        // Only zoom in for lower level expansions where detail is needed
        const node = expandedData.nodes.find(n => n.id === nodeId);
        if (node && graph && node.level < 3) {
            // Subtle zoom for lower levels
            const currentPos = graph.cameraPosition();
            const zoomDistance = node.level === 2 ? 300 : 200;
            graph.cameraPosition(
                {
                    x: node.position[0],
                    y: node.position[1],
                    z: node.position[2] + zoomDistance
                },
                { x: node.position[0], y: node.position[1], z: node.position[2] },
                1000
            );
        }
    }

    // Collapse a node
    function collapseNode(nodeId) {
        expandedNodes.delete(nodeId);

        // Also collapse all descendants
        if (fullData.clusters) {
            const collapseDescendants = (id) => {
                expandedNodes.delete(id);

                // Find and collapse children
                ['level_3', 'level_2', 'level_1'].forEach(level => {
                    const cluster = fullData.clusters[level] && fullData.clusters[level][id];
                    if (cluster && cluster.children) {
                        cluster.children.forEach(childId => collapseDescendants(childId));
                    }
                });
            };
            collapseDescendants(nodeId);
        }

        const expandedData = buildExpandedGraph();
        updateGraph(expandedData);
    }

    // Build graph with expanded nodes
    function buildExpandedGraph() {
        const nodes = [];
        const links = [];
        const addedNodes = new Set();

        // Always show level 3 categories
        Object.values(fullData.clusters.level_3 || {}).forEach((cluster, index) => {
            if (!addedNodes.has(cluster.id)) {
                const position = nodePositionCache.get(cluster.id) || [0, 0, 0];
                nodes.push({
                    id: cluster.id,
                    name: getCategoryName(cluster, index),
                    type: 'coarse_cluster',
                    level: 3,
                    position: position,
                    children: cluster.children || [],
                    childCount: getDescendantCount(cluster),
                    color: CONFIG.nodeColors.coarse_cluster,
                    size: CONFIG.nodeSizes.coarse_cluster,
                    expanded: expandedNodes.has(cluster.id),
                    opacity: 1
                });
                addedNodes.add(cluster.id);
            }

            // If this category is expanded, show its medium clusters
            if (expandedNodes.has(cluster.id) && cluster.children) {
                cluster.children.forEach(l2Id => {
                    const l2Cluster = fullData.clusters.level_2 && fullData.clusters.level_2[l2Id];
                    if (l2Cluster && !addedNodes.has(l2Id)) {
                        const l2Position = calculateChildPosition(
                            nodePositionCache.get(cluster.id),
                            cluster.children.indexOf(l2Id),
                            cluster.children.length,
                            100
                        );

                        nodes.push({
                            id: l2Id,
                            name: l2Cluster.name || `Topic ${l2Id}`,
                            type: 'medium_cluster',
                            level: 2,
                            position: l2Position,
                            children: l2Cluster.children || [],
                            color: CONFIG.nodeColors.medium_cluster,
                            size: CONFIG.nodeSizes.medium_cluster,
                            expanded: expandedNodes.has(l2Id),
                            opacity: 0.9
                        });
                        addedNodes.add(l2Id);
                        nodePositionCache.set(l2Id, l2Position);

                        // Link to parent
                        links.push({
                            source: cluster.id,
                            target: l2Id,
                            type: 'hierarchy',
                            color: CONFIG.linkColors.hierarchy,
                            width: 2,
                            opacity: 1.0  // Made fully opaque for visibility
                        });

                        // If this medium cluster is expanded, show its fine clusters
                        if (expandedNodes.has(l2Id) && l2Cluster.children) {
                            l2Cluster.children.forEach(l1Id => {
                                const l1Cluster = fullData.clusters.level_1 && fullData.clusters.level_1[l1Id];
                                if (l1Cluster && !addedNodes.has(l1Id)) {
                                    const l1Position = calculateChildPosition(
                                        l2Position,
                                        l2Cluster.children.indexOf(l1Id),
                                        l2Cluster.children.length,
                                        50
                                    );

                                    nodes.push({
                                        id: l1Id,
                                        name: l1Cluster.name || `Cluster ${l1Id}`,
                                        type: 'fine_cluster',
                                        level: 1,
                                        position: l1Position,
                                        entities: l1Cluster.entities || [],
                                        color: CONFIG.nodeColors.fine_cluster,
                                        size: CONFIG.nodeSizes.fine_cluster,
                                        expanded: expandedNodes.has(l1Id),
                                        opacity: 1.0  // Made fully opaque for visibility
                                    });
                                    addedNodes.add(l1Id);
                                    nodePositionCache.set(l1Id, l1Position);

                                    // Link to parent
                                    links.push({
                                        source: l2Id,
                                        target: l1Id,
                                        type: 'hierarchy',
                                        color: CONFIG.linkColors.hierarchy,
                                        width: 1.5,
                                        opacity: 0.7
                                    });

                                    // If this fine cluster is expanded, show a sample of entities
                                    if (expandedNodes.has(l1Id) && l1Cluster.entities) {
                                        // Limit to first 20 entities for performance
                                        const entitiesToShow = l1Cluster.entities.slice(0, 20);

                                        entitiesToShow.forEach((entityId, idx) => {
                                            const entity = fullData.entities[entityId];
                                            const entityCluster = fullData.clusters.level_0 && fullData.clusters.level_0[entityId];

                                            if (entity && !addedNodes.has(entityId)) {
                                                const entityPosition = calculateChildPosition(
                                                    l1Position,
                                                    idx,
                                                    entitiesToShow.length,
                                                    30
                                                );

                                                nodes.push({
                                                    id: entityId,
                                                    name: entity.name || entityId,
                                                    type: entity.type || 'CONCEPT',
                                                    level: 0,
                                                    position: entityPosition,
                                                    description: entity.description,
                                                    color: CONFIG.nodeColors[entity.type] || CONFIG.nodeColors.CONCEPT,
                                                    size: CONFIG.nodeSizes.entity,
                                                    opacity: 1.0  // Made fully opaque for visibility
                                                });
                                                addedNodes.add(entityId);

                                                // Link to parent cluster
                                                links.push({
                                                    source: l1Id,
                                                    target: entityId,
                                                    type: 'hierarchy',
                                                    color: CONFIG.linkColors.hierarchy,
                                                    width: 0.5,
                                                    opacity: 0.5
                                                });
                                            }
                                        });

                                        // Add indicator if there are more entities
                                        if (l1Cluster.entities.length > 20) {
                                            const morePosition = calculateChildPosition(
                                                l1Position,
                                                20,
                                                21,
                                                30
                                            );

                                            nodes.push({
                                                id: `${l1Id}_more`,
                                                name: `+${l1Cluster.entities.length - 20} more...`,
                                                type: 'indicator',
                                                level: 0,
                                                position: morePosition,
                                                color: '#888888',
                                                size: 1,
                                                opacity: 0.5
                                            });
                                        }
                                    }
                                }
                            });
                        }
                    }
                });
            }
        });

        return { nodes, links };
    }

    // Calculate position for child node
    function calculateChildPosition(parentPos, index, total, radius) {
        const angleStep = (2 * Math.PI) / total;
        const angle = index * angleStep;

        return [
            parentPos[0] + Math.cos(angle) * radius,
            parentPos[1] + Math.sin(angle) * radius,
            parentPos[2] + (Math.random() - 0.5) * 20 // Slight z variation
        ];
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
            .nodeId('id')
            .nodeLabel(node => {
                if (node.level === 0) {
                    return `${node.name}\n${node.type}${node.description ? '\n' + node.description.substring(0, 100) + '...' : ''}`;
                } else {
                    return `${node.name}\n${node.childCount ? node.childCount + ' entities' : ''}`;
                }
            })
            .nodeColor(node => node.color)
            .nodeVal(node => Math.pow(node.size, 2))
            .nodeOpacity(node => node.opacity || 0.9)
            .linkSource('source')
            .linkTarget('target')
            .linkColor(link => link.color)
            .linkWidth(link => link.width || 1)
            .linkOpacity(link => link.opacity || 0.6)
            .backgroundColor('#0a0e1a')
            .showNavInfo(true)
            .enableNodeDrag(false); // Disable dragging for cleaner interaction

        // Set initial camera position
        graph.cameraPosition(
            { x: 0, y: 0, z: CONFIG.initialCameraDistance },
            { x: 0, y: 0, z: 0 },
            0
        );

        // Node click handler - single click toggles expansion
        graph.onNodeClick(node => {
            if (!node) return;

            // Show info panel
            updateInfoPanel(node);

            // For clusters, clicking toggles expansion
            if (node.level > 0 && node.type !== 'indicator') {
                if (expandedNodes.has(node.id)) {
                    collapseNode(node.id);
                } else {
                    expandNode(node.id);
                }
            } else if (node.level === 0) {
                // For entity nodes, show source information
                console.log('Entity clicked:', node.name);
                // Could expand to show relationships or source info here
            }
        });

        // Node hover handler
        graph.onNodeHover(node => {
            container.style.cursor = node ? 'pointer' : 'default';

            // Show tooltip on hover
            if (node) {
                showTooltip(node);
            } else {
                hideTooltip();
            }
        });

        // Custom node rendering
        graph.nodeThreeObject(node => {
            if (node.type === 'indicator') {
                // Special rendering for "more..." indicator
                const sprite = new SpriteText(node.name);
                sprite.color = node.color;
                sprite.textHeight = 4;
                return sprite;
            }

            if (node.level >= 2) {
                // Render higher level clusters with glow
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

                // Add expansion indicator
                if (node.children && node.children.length > 0) {
                    const plusGeometry = new THREE.BoxGeometry(node.size * 0.2, node.size * 1.2, node.size * 0.2);
                    const plusMaterial = new THREE.MeshBasicMaterial({
                        color: '#ffffff',
                        opacity: node.expanded ? 0 : 0.7,
                        transparent: true
                    });
                    const plusVertical = new THREE.Mesh(plusGeometry, plusMaterial);
                    const plusHorizontal = new THREE.Mesh(
                        new THREE.BoxGeometry(node.size * 1.2, node.size * 0.2, node.size * 0.2),
                        plusMaterial
                    );

                    if (!node.expanded) {
                        group.add(plusVertical);
                        group.add(plusHorizontal);
                    }
                }

                return group;
            }

            // Custom rendering for level 1 and 0 nodes to make them more visible
            if (node.level === 1 || node.level === 0) {
                const group = new THREE.Group();

                // Use brighter colors and larger size for visibility
                const size = node.level === 1 ? node.size * 1.5 : node.size * 2;
                const geometry = new THREE.SphereGeometry(size);
                const material = new THREE.MeshPhongMaterial({
                    color: node.color || '#00ffff',
                    emissive: node.color || '#00ffff',
                    emissiveIntensity: 0.5,  // Make it glow
                    shininess: 100,
                    opacity: 1.0,
                    transparent: false
                });
                const sphere = new THREE.Mesh(geometry, material);
                group.add(sphere);

                // Add a label for entity nodes
                if (node.level === 0 && node.name) {
                    const sprite = new SpriteText(node.name);
                    sprite.color = '#ffffff';
                    sprite.textHeight = 3;
                    sprite.position.y = size + 2;
                    group.add(sprite);
                }

                return group;
            }

            return false; // Use default rendering for any other cases
        });
    }

    // Update graph with new data
    function updateGraph(graphData) {
        if (!graph) return;

        // Update graph data with animation
        graph.graphData(graphData);
    }



    // Update info panel
    function updateInfoPanel(node) {
        const panel = document.getElementById('node-info-panel');
        const title = document.getElementById('node-title');
        const details = document.getElementById('node-details');
        const relationships = document.getElementById('node-relationships');

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

            if (node.childCount) {
                detailsHTML += `<p><strong>Contains:</strong> ${node.childCount} entities</p>`;
            }

            if (node.description) {
                detailsHTML += `<p><strong>Description:</strong> ${node.description}</p>`;
            }

            // Add source information for entities with deep links
            if (node.level === 0) {
                // Get entity data
                const entity = fullData.entities[node.id];
                let sourcesHTML = '';
                let deepLinksHTML = '';

                if (entity && entity.sources) {
                    // Group sources by episode
                    const episodeSources = {};
                    entity.sources.forEach(source => {
                        if (source.startsWith('episode_')) {
                            const epNum = source.replace('episode_', '');
                            if (!episodeSources[epNum]) {
                                episodeSources[epNum] = [];
                            }
                            episodeSources[epNum].push(source);
                        }
                    });

                    // Sort episodes numerically
                    const sortedEpisodes = Object.keys(episodeSources)
                        .filter(n => !isNaN(n))
                        .sort((a, b) => parseInt(a) - parseInt(b))
                        .slice(0, 5); // Show first 5 episodes

                    if (sortedEpisodes.length > 0) {
                        // Create deep links for each episode with actual timestamps
                        const episodeLinks = sortedEpisodes.map(epNum => {
                            // Try to find a timestamp from the entity's chunks
                            let timestamp = '0:00'; // Default if no chunk data

                            if (entity.metadata && entity.metadata.chunks && chunkTimestamps) {
                                // Look for a chunk from this episode
                                const chunkForEpisode = entity.metadata.chunks.find(chunkId => {
                                    // Check if chunk is from this episode (e.g., "ep114_chunk5")
                                    return chunkId.startsWith(`ep${epNum}_chunk`);
                                });

                                if (chunkForEpisode && chunkTimestamps[chunkForEpisode]) {
                                    timestamp = chunkTimestamps[chunkForEpisode].start_formatted || '0:00';
                                }
                            }

                            // Format: /PodcastMap3D.html#episode=114&t=34:21
                            return `<a href="/PodcastMap3D.html#episode=${epNum}&t=${timestamp}"
                                     target="_blank"
                                     style="color: #00ffff; text-decoration: underline; margin-right: 10px;"
                                     title="Jump to Episode ${epNum} at ${timestamp}">
                                     Episode ${epNum} (${timestamp})
                                   </a>`;
                        }).join('');

                        sourcesHTML = `
                            <p><strong>Found in Episodes:</strong></p>
                            <div style="margin-left: 10px;">
                                ${episodeLinks}
                                ${Object.keys(episodeSources).length > 5 ?
                                  `<span style="color: #888;">(+${Object.keys(episodeSources).length - 5} more)</span>` : ''}
                            </div>
                        `;

                        // If we have metadata with chunk IDs, show deep links
                        if (entity.metadata && entity.metadata.chunks && chunkTimestamps) {
                            const chunkLinks = entity.metadata.chunks.slice(0, 5).map(chunkId => {
                                const chunkData = chunkTimestamps[chunkId];
                                if (chunkData) {
                                    const epNum = chunkData.episode;
                                    const timestamp = chunkData.start_formatted || '0:00';
                                    return `<a href="/PodcastMap3D.html#episode=${epNum}&t=${timestamp}"
                                             target="_blank"
                                             style="color: #88cc88; text-decoration: underline; margin-right: 8px; font-size: 0.9em;"
                                             title="${chunkId} - Jump to Episode ${epNum} at ${timestamp}">
                                             Ep${epNum} @ ${timestamp}
                                           </a>`;
                                } else {
                                    return `<span style="color: #666; margin-right: 8px; font-size: 0.9em;">${chunkId}</span>`;
                                }
                            }).join('');

                            deepLinksHTML = `
                                <p style="margin-top: 10px;"><strong>Specific Appearances:</strong></p>
                                <div style="margin-left: 10px;">
                                    ${chunkLinks}
                                    ${entity.metadata.chunks.length > 5 ?
                                      `<span style="color: #888; font-size: 0.9em;">(+${entity.metadata.chunks.length - 5} more)</span>` : ''}
                                </div>
                            `;
                        }
                    }
                }

                detailsHTML += `
                    <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid rgba(100, 200, 255, 0.3);">
                        <p><strong>Sources & References:</strong></p>
                        ${sourcesHTML || '<p style="color: #888;">No source information available</p>'}
                        ${deepLinksHTML}

                        <p style="margin-top: 15px;"><strong>Explore in Other Views:</strong></p>
                        <div style="margin-top: 10px;">
                            <a href="/PodcastMap3D.html#search=${encodeURIComponent(node.name)}"
                               target="_blank"
                               style="color: #00ffff; text-decoration: none; padding: 5px 10px; border: 1px solid #00ffff; border-radius: 5px; display: inline-block; margin-right: 10px;">
                                🎧 Podcast Map
                            </a>
                            <a href="/KnowledgeGraph.html#entity=${encodeURIComponent(node.id)}"
                               target="_blank"
                               style="color: #00ffff; text-decoration: none; padding: 5px 10px; border: 1px solid #00ffff; border-radius: 5px; display: inline-block;">
                                🕸️ Knowledge Graph
                            </a>
                        </div>
                    </div>
                `;
            }

            if (node.level > 0 && node.children) {
                detailsHTML += `<p><strong>Sub-items:</strong> ${node.children.length}</p>`;
                detailsHTML += `<p><em>${node.expanded ? '🔽 Click to collapse' : '▶️ Click to expand'}</em></p>`;
            }

            details.innerHTML = detailsHTML;

            panel.style.display = 'block';
        }
    }

    // Search functionality
    function searchGraph(query) {
        if (!query || !fullData) return;

        const lowerQuery = query.toLowerCase();
        searchResults = [];

        // Search through all entities
        Object.entries(fullData.entities).forEach(([id, entity]) => {
            if (entity.name && entity.name.toLowerCase().includes(lowerQuery)) {
                searchResults.push({
                    id: id,
                    name: entity.name,
                    type: entity.type || 'CONCEPT',
                    level: 0,
                    score: entity.name.toLowerCase() === lowerQuery ? 1 : 0.5
                });
            }
        });

        // Sort by relevance
        searchResults.sort((a, b) => b.score - a.score);

        // Show search results
        if (searchResults.length > 0) {
            showSearchResults(searchResults.slice(0, 10));
        } else {
            showMessage('No results found');
        }
    }

    // Show search results
    function showSearchResults(results) {
        const resultsHtml = results.map(result => `
            <div class="search-result" data-id="${result.id}">
                <span class="entity-type-badge type-${result.type}">${result.type}</span>
                <span>${result.name}</span>
            </div>
        `).join('');

        const searchResultsContainer = document.getElementById('search-results');
        if (searchResultsContainer) {
            searchResultsContainer.innerHTML = resultsHtml;
            searchResultsContainer.style.display = 'block';

            // Add click handlers
            searchResultsContainer.querySelectorAll('.search-result').forEach(el => {
                el.addEventListener('click', () => {
                    const entityId = el.dataset.id;
                    navigateToEntity(entityId);
                    searchResultsContainer.style.display = 'none';
                });
            });
        }
    }

    // Navigate to a specific entity
    function navigateToEntity(entityId) {
        // Find which clusters contain this entity
        const path = findEntityPath(entityId);

        if (path) {
            // Expand all clusters in the path
            path.forEach(clusterId => {
                expandedNodes.add(clusterId);
            });

            // Rebuild and update graph
            const expandedData = buildExpandedGraph();
            updateGraph(expandedData);

            // Focus camera on entity
            setTimeout(() => {
                const entityNode = expandedData.nodes.find(n => n.id === entityId);
                if (entityNode && graph) {
                    graph.cameraPosition(
                        {
                            x: entityNode.position[0],
                            y: entityNode.position[1],
                            z: entityNode.position[2] + 100
                        },
                        { x: entityNode.position[0], y: entityNode.position[1], z: entityNode.position[2] },
                        1500
                    );

                    // Select the node
                    handleNodeClick(entityNode);
                }
            }, 500);
        }
    }

    // Find the path of clusters containing an entity
    function findEntityPath(entityId) {
        const path = [];

        // Find level 1 cluster containing the entity
        let l1ClusterId = null;
        Object.entries(fullData.clusters.level_1 || {}).forEach(([id, cluster]) => {
            if (cluster.entities && cluster.entities.includes(entityId)) {
                l1ClusterId = id;
                path.push(id);
            }
        });

        if (!l1ClusterId) return null;

        // Find level 2 cluster containing the level 1 cluster
        let l2ClusterId = null;
        Object.entries(fullData.clusters.level_2 || {}).forEach(([id, cluster]) => {
            if (cluster.children && cluster.children.includes(l1ClusterId)) {
                l2ClusterId = id;
                path.unshift(id);
            }
        });

        if (!l2ClusterId) return path;

        // Find level 3 cluster containing the level 2 cluster
        Object.entries(fullData.clusters.level_3 || {}).forEach(([id, cluster]) => {
            if (cluster.children && cluster.children.includes(l2ClusterId)) {
                path.unshift(id);
            }
        });

        return path;
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

    // Highlight node connections
    function highlightConnections(node) {
        if (!graph) return;

        // Dim all nodes except selected and connected
        const currentData = graph.graphData();
        const connectedNodeIds = new Set([node.id]);

        // Find connected nodes
        currentData.links.forEach(link => {
            if (link.source.id === node.id || link.source === node.id) {
                connectedNodeIds.add(typeof link.target === 'object' ? link.target.id : link.target);
            } else if (link.target.id === node.id || link.target === node.id) {
                connectedNodeIds.add(typeof link.source === 'object' ? link.source.id : link.source);
            }
        });

        // Update node colors
        graph.nodeColor(n => {
            if (n.id === node.id) {
                return '#ffff00'; // Highlight selected
            } else if (connectedNodeIds.has(n.id)) {
                return n.color; // Keep connected nodes normal
            } else {
                return `${n.color}33`; // Dim unconnected
            }
        });
    }

    // Show tooltip
    function showTooltip(node) {
        // Implementation for tooltip display
        // This would be a floating div that follows the mouse
    }

    // Hide tooltip
    function hideTooltip() {
        // Hide the tooltip div
    }

    // Show loading indicator
    function updateLoadingStatus(message) {
        const loadingMsg = document.getElementById("loading-message");
        const loadingStatus = document.getElementById("loading-status");
        if (loadingMsg) loadingMsg.textContent = message;
        if (loadingStatus) loadingStatus.textContent = message;
    }

    function showLoading(show) {
        const container = document.getElementById('3d-graph');
        if (!container) return;

        if (show) {
            container.innerHTML = `
                <div style="color: #00ffff; padding: 20px; text-align: center;">
                    <h3>Loading GraphRAG Knowledge Explorer...</h3>
                    <p>Processing 12,000+ entities and relationships</p>
                    <div class="loading-spinner"></div>
                </div>
            `;
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

    // Show message
    function showMessage(message) {
        // Display temporary message to user
        const messageDiv = document.createElement('div');
        messageDiv.className = 'temp-message';
        messageDiv.textContent = message;
        messageDiv.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: rgba(0, 20, 40, 0.9);
            color: #00ffff;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #00ffff;
            z-index: 1000;
        `;
        document.body.appendChild(messageDiv);

        setTimeout(() => {
            messageDiv.remove();
        }, 3000);
    }

    // Setup event listeners
    function setupEventListeners() {
        // Search bar
        const searchBar = document.getElementById('graph-search');
        if (searchBar) {
            searchBar.addEventListener('input', (e) => {
                const query = e.target.value.trim();
                if (query.length >= 2) {
                    searchGraph(query);
                } else {
                    const searchResults = document.getElementById('search-results');
                    if (searchResults) {
                        searchResults.style.display = 'none';
                    }
                }
            });
        }

        // View controls
        const resetViewBtn = document.getElementById('reset-view');
        if (resetViewBtn) {
            resetViewBtn.addEventListener('click', () => {
                expandedNodes.clear();
                showCategoriesOnly();
                graph.cameraPosition(
                    { x: 0, y: 0, z: CONFIG.initialCameraDistance },
                    { x: 0, y: 0, z: 0 },
                    1000
                );
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

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                // Close info panel
                window.graphRAG3D.closeInfoPanel();
            } else if (e.key === 'r' || e.key === 'R') {
                // Reset view
                expandedNodes.clear();
                showCategoriesOnly();
            } else if (e.key === '/' && !searchBar.matches(':focus')) {
                // Focus search
                e.preventDefault();
                searchBar.focus();
            }
        });
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
                const currentData = graph.graphData();
                graph.nodeColor(node => node.color);
            }

            selectedNode = null;
        }
    };

    // Add required Three.js components
    window.SpriteText = function(text = '') {
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');

        // Measure text
        context.font = '50px Arial';
        const metrics = context.measureText(text);

        canvas.width = metrics.width;
        canvas.height = 60;

        context.font = '50px Arial';
        context.fillStyle = 'white';
        context.fillText(text, 0, 50);

        const texture = new THREE.CanvasTexture(canvas);
        const material = new THREE.SpriteMaterial({ map: texture });
        const sprite = new THREE.Sprite(material);

        sprite.scale.set(canvas.width / 10, canvas.height / 10, 1);

        return sprite;
    };

    // Initialize on DOM ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();