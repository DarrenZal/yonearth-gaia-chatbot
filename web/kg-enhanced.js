(function () {
    'use strict';

    const CONFIG = {
        nodeColors: {
            PERSON: '#4CAF50',
            ORGANIZATION: '#2196F3',
            CONCEPT: '#FFC107',
            PLACE: '#FF9800',
            PRACTICE: '#00BCD4',
            PRODUCT: '#9C27B0',
            EVENT: '#F44336',
            WORK: '#795548',
            fine_cluster: '#00ff88',
            medium_cluster: '#00aaff',
            coarse_cluster: '#ff00ff'
        },
        nodeSizes: {
            entityBase: 1.4,
            fine_cluster: 3,
            medium_cluster: 6,
            coarse_cluster: 12
        },
        linkColors: {
            hierarchy: '#00ffff',
            relationship: '#ffffff',
            categoryRing: '#00ffff33'
        },
        initialCameraDistance: 420
    };
    const MAX_L3_CLUSTERS = 100;  // Show all top-level clusters (was 7)

    let fullData = null;
    let graph = null;
    let expandedNodes = new Set();
    let selectedNode = null;
    let nodePositionCache = new Map();
    let entityDegrees = new Map();
    let entityClusterPath = {};
    let relationshipTypes = [];
    let activeRelationshipTypes = new Set();
    let chunkTimestamps = {};
    let maxEntitiesPerCluster = 200;
    let visibleLevels = new Set([3]);
    let showHierarchyEdges = true;
    let showRelationshipEdges = true;
    let entityTypes = [];
    let activeEntityTypes = new Set();
    let availableEpisodes = [];
    let availableBooks = [];
    let activeEpisodeFilter = '';
    let activeBookFilter = '';
    let pinnedEntityId = null;
    let lastClickTime = 0;
    let lastClickNodeId = null;
    const DOUBLE_CLICK_MS = 300;
    let currentViewMode = 'holographic'; // 'holographic' or 'full-force'
    let clusterMembranes = []; // For Full Force view membranes
    let floorPlaneObjects = []; // Track floor planes for removal
    let nodeSizeMode = 'connectivity'; // 'connectivity' or 'betweenness'
    let connectivityStats = {
        minDegree: 0,
        maxDegree: 0,
        minWeighted: 0,
        maxWeighted: 0,
        minBetweenness: 0,
        maxBetweenness: 0
    };
    function getL2Children(l2, parentId) {
        const arr = [];
        if (!l2) return arr;
        Object.entries(l2).forEach(([id, c]) => {
            if (c.parent === parentId) arr.push(id);
        });
        return arr;
    }

    function getL1Children(l1, parentId) {
        const arr = [];
        if (!l1) return arr;
        Object.entries(l1).forEach(([id, c]) => {
            if (c.parent === parentId) arr.push(id);
        });
        return arr;
    }

    function ensureChildLevelsVisible(level) {
        // Automatically reveal child levels when expanding a cluster
        const levelCheckbox = (lvl) => document.getElementById(`kg-level-${lvl}`);
        if (level === 3 && !visibleLevels.has(2)) {
            visibleLevels.add(2);
            const cb = levelCheckbox(2);
            if (cb) cb.checked = true;
        }
        if (level === 2 && !visibleLevels.has(1)) {
            visibleLevels.add(1);
            const cb = levelCheckbox(1);
            if (cb) cb.checked = true;
        }
        if (level === 1 && !visibleLevels.has(0)) {
            visibleLevels.add(0);
            const cb = levelCheckbox(0);
            if (cb) cb.checked = true;
        }
    }

    function ensureAncestorExpanded(node) {
        if (!node || !fullData || !fullData.clusters) return;
        const clusters = fullData.clusters;
        const l2 = clusters.level_2 || {};
        const l1 = clusters.level_1 || {};

        if (node.level === 2) {
            const parent = l2[node.id]?.parent;
            if (parent) expandedNodes.add(parent);
        } else if (node.level === 1) {
            const parent2 = l1[node.id]?.parent;
            if (parent2) {
                expandedNodes.add(parent2);
                const parent3 = l2[parent2]?.parent;
                if (parent3) expandedNodes.add(parent3);
            }
        } else if (node.level === 0) {
            const path = entityClusterPath[node.id];
            if (path) {
                if (path.level1) expandedNodes.add(path.level1);
                if (path.level2) expandedNodes.add(path.level2);
                if (path.level3) expandedNodes.add(path.level3);
            }
        }
    }

    window.kgDebug = {
        getGraph: () => graph,
        getData: () => fullData,
        getExpandedNodes: () => expandedNodes,
        getEntityDegrees: () => entityDegrees
    };

    async function init() {
        try {
            showLoading(true);
            console.log('[KG] init start');

            await loadData();
            console.log('[KG] data loaded');
            preprocessEntities();
            preprocessHierarchy();
            console.log('[KG] preprocess done');
            initControls();
            console.log('[KG] controls init done');
            initGraph();
            console.log('[KG] graph init done');
            showTopLevelClusters();
            console.log('[KG] showTopLevelClusters called');

            showLoading(false);
            console.log('[KG] init complete');
        } catch (err) {
            console.error('Failed to initialize KG explorer:', err);
            showError('Failed to load knowledge graph data. Please refresh the page.');
            showLoading(false);
        }
    }

    async function loadData() {
        let loaded = false;

        // Preferred: full GraphRAG hierarchy with positions
        try {
            const resp = await fetch('data/graphrag_hierarchy/graphrag_hierarchy.json?v=20251122');
            if (resp.ok) {
                fullData = await resp.json();
                loaded = true;
                console.log('Loaded normalized graphrag_hierarchy dataset', {
                    entities: Object.keys(fullData.entities || {}).length,
                    relationships: (fullData.relationships || []).length,
                    clusters: fullData.clusters ? Object.keys(fullData.clusters) : []
                });
            }
        } catch (err) {
            console.warn('Error loading graphrag_hierarchy, will fall back:', err);
        }

        // Fallback: unified KG + hierarchy subset
        if (!loaded) {
            console.log('Falling back to unified KG + hierarchy subset');
            try {
                const [kgResp, hierarchyResp] = await Promise.all([
                    fetch('data/knowledge_graph_unified/unified.json'),
                    fetch('data/graphrag_hierarchy/graphrag_hierarchy.json')
                ]);

                if (!kgResp.ok) {
                    throw new Error('Failed to load unified knowledge graph JSON');
                }
                if (!hierarchyResp.ok) {
                    throw new Error('Failed to load GraphRAG hierarchy JSON');
                }

                const kgData = await kgResp.json();
                const hierarchy = await hierarchyResp.json();

                fullData = {
                    entities: kgData.entities || {},
                    relationships: kgData.relationships || [],
                    clusters: hierarchy.clusters || {},
                    positions: buildPositionsFromHierarchy(hierarchy),
                    metadata: kgData.metadata || hierarchy.metadata
                };
                loaded = true;
            } catch (err) {
                console.error('Failed to load fallback unified + hierarchy data:', err);
                throw err;
            }
        }

        if (!loaded || !fullData) {
            throw new Error('Knowledge graph data could not be loaded.');
        }

        // Normalize entity names
        Object.entries(fullData.entities || {}).forEach(([id, ent]) => {
            if (!ent.name) {
                ent.name = id;
            }
        });

        // Load chunk timestamp mappings (prefer processed path)
        try {
            let tsResp = await fetch('data/processed/chunk_timestamps.json');
            if (!tsResp.ok) {
                tsResp = await fetch('data/chunk_timestamps.json');
            }
            if (tsResp.ok) {
                chunkTimestamps = await tsResp.json();
                console.log('Loaded chunk timestamp mappings:', Object.keys(chunkTimestamps).length);
            }
        } catch (err) {
            console.warn('Could not load chunk timestamp mappings:', err);
        }
    }

    function buildPositionsFromHierarchy(hierarchy) {
        const positions = {};
        const level0 = hierarchy.clusters && hierarchy.clusters.level_0;
        if (!level0) return positions;

        Object.entries(level0).forEach(([id, node]) => {
            if (node.position && node.position.length === 3) {
                positions[id] = node.position;
            }
        });
        return positions;
    }

    function preprocessEntities() {
        const entities = fullData.entities || {};
        const relationships = fullData.relationships || [];

        const degreeMap = new Map();
        const relTypeSet = new Set();
        const episodeSet = new Set();
        const bookSet = new Set();

        relationships.forEach(rel => {
            const s = rel.source;
            const t = rel.target;
            const type = rel.type || rel.predicate || rel.relationship_type || 'RELATED_TO';

            degreeMap.set(s, (degreeMap.get(s) || 0) + 1);
            degreeMap.set(t, (degreeMap.get(t) || 0) + 1);
            relTypeSet.add(type);
        });

        Object.entries(entities).forEach(([id, ent]) => {
            const degree = degreeMap.get(id) || 0;
            entityDegrees.set(id, degree);

            if (ent.type && !entityTypes.includes(ent.type)) {
                entityTypes.push(ent.type);
            }

            if (Array.isArray(ent.sources)) {
                ent.sources.forEach(src => {
                    if (typeof src !== 'string') return;
                    if (src.startsWith('episode_')) {
                        const n = src.replace('episode_', '');
                        if (!isNaN(parseInt(n, 10))) {
                            episodeSet.add(parseInt(n, 10));
                        }
                    } else {
                        const key = src.toLowerCase();
                        if (['veriditas', 'viriditas', 'y-on-earth', 'soil-stewardship-handbook', 'our-biggest-deal'].includes(key)) {
                            bookSet.add(key);
                        }
                    }
                });
            }
        });

        entityTypes.sort();
        activeEntityTypes = new Set(entityTypes);

        relationshipTypes = Array.from(relTypeSet).sort();
        activeRelationshipTypes = new Set(relationshipTypes);

        availableEpisodes = Array.from(episodeSet).sort((a, b) => a - b);
        availableBooks = Array.from(bookSet);
    }

    function preprocessHierarchy() {
        const clusters = fullData.clusters || {};
        if (!clusters.level_1 || !clusters.level_2 || !clusters.level_3) {
            console.warn('Cluster hierarchy incomplete; levels found:', Object.keys(clusters));
            return;
        }

        // Generate names for clusters that don't have them
        Object.entries(clusters.level_3 || {}).forEach(([id, cluster]) => {
            if (!cluster.name) {
                cluster.name = `Category ${id.replace('l3_', '')}`;
            }
        });
        Object.entries(clusters.level_2 || {}).forEach(([id, cluster]) => {
            if (!cluster.name) {
                cluster.name = `Topic ${id.replace('l2_', '')}`;
            }
        });
        Object.entries(clusters.level_1 || {}).forEach(([id, cluster]) => {
            if (!cluster.name) {
                cluster.name = `Cluster ${id.replace('l1_', '')}`;
            }
        });

        // Compute connectivity for node sizing
        computeConnectivityStats();

        // Build parent relationships if missing
        const level1 = clusters.level_1;
        const level2 = clusters.level_2;
        const level3 = clusters.level_3;

        // L1 -> L2 parent linkage (if parent is null, infer from children arrays)
        Object.values(level2).forEach(l2cluster => {
            const l2children = l2cluster.children || [];
            l2children.forEach(l1id => {
                if (level1[l1id] && !level1[l1id].parent) {
                    level1[l1id].parent = l2cluster.id;
                }
            });
        });

        // L2 -> L3 parent linkage
        Object.values(level3).forEach(l3cluster => {
            const l3children = l3cluster.children || [];
            l3children.forEach(l2id => {
                if (level2[l2id] && !level2[l2id].parent) {
                    level2[l2id].parent = l3cluster.id;
                }
            });
        });

        const positions = fullData.positions || {};

        // HOLOGRAPHIC ELEVATOR: Level 0 (entities) as 3D cloud below L1 plane (y < -200)
        // Keep original x,z,y from embeddings but shift down by 400 units
        Object.entries(positions).forEach(([id, pos]) => {
            if (Array.isArray(pos) && pos.length === 3) {
                nodePositionCache.set(id, [pos[0], pos[1] - 400, pos[2]]);
            }
        });

        // Use already declared level1 variable
        // HOLOGRAPHIC ELEVATOR: Level 1 fixed at y = -200
        Object.values(level1).forEach(cluster => {
            const entityIds = cluster.entities || [];
            let cx = 0, cz = 0, count = 0;
            entityIds.forEach(eid => {
                const pos = nodePositionCache.get(eid);
                if (pos) {
                    cx += pos[0];
                    cz += pos[2];
                    count += 1;
                }
            });
            const pos = count > 0 ? [cx / count, -200, cz / count] : [
                (Math.random() - 0.5) * 200,
                -200,
                (Math.random() - 0.5) * 200
            ];
            cluster.position = pos;
            nodePositionCache.set(cluster.id, pos);
        });

        // Use already declared level2 variable
        // HOLOGRAPHIC ELEVATOR: Level 2 fixed at y = 0
        Object.values(level2).forEach(cluster => {
            const children = cluster.children || [];
            let cx = 0, cz = 0, count = 0;
            children.forEach(cid => {
                const childCluster = level1[cid];
                if (childCluster && childCluster.position) {
                    const pos = childCluster.position;
                    cx += pos[0];
                    cz += pos[2];
                    count += 1;
                }
            });
            const pos = count > 0 ? [cx / count, 0, cz / count] : [
                (Math.random() - 0.5) * 300,
                0,
                (Math.random() - 0.5) * 300
            ];
            cluster.position = pos;
            nodePositionCache.set(cluster.id, pos);
        });

        // Use already declared level3 variable
        // HOLOGRAPHIC ELEVATOR: Level 3 fixed at y = 200
        // SPREAD OUT more to prevent overlapping
        const L3_SPREAD_FACTOR = 2.5;  // Scale positions outward from center
        Object.values(level3).forEach(cluster => {
            const children = cluster.children || [];
            let cx = 0, cz = 0, count = 0;
            children.forEach(cid => {
                const childCluster = level2[cid];
                if (childCluster && childCluster.position) {
                    const pos = childCluster.position;
                    cx += pos[0];
                    cz += pos[2];
                    count += 1;
                }
            });
            if (count > 0) {
                const avgX = cx / count;
                const avgZ = cz / count;
                // Scale outward from center (0,0)
                const pos = [avgX * L3_SPREAD_FACTOR, 200, avgZ * L3_SPREAD_FACTOR];
                cluster.position = pos;
                nodePositionCache.set(cluster.id, pos);
            } else {
                const pos = [
                    (Math.random() - 0.5) * 600,
                    200,
                    (Math.random() - 0.5) * 600
                ];
                cluster.position = pos;
                nodePositionCache.set(cluster.id, pos);
            }
        });

        entityClusterPath = {};
        Object.entries(level1).forEach(([l1Id, cluster]) => {
            const l2Id = cluster.parent;
            const l3Id = l2Id && level2[l2Id] ? level2[l2Id].parent : undefined;

            (cluster.entities || []).forEach(eid => {
                entityClusterPath[eid] = {
                    level1: l1Id,
                    level2: l2Id,
                    level3: l3Id
                };
            });
        });

        normalizeCachedPositions();
    }

    function normalizeCachedPositions() {
        if (!nodePositionCache || nodePositionCache.size === 0) return;

        // HOLOGRAPHIC ELEVATOR: Only normalize X and Z, preserve fixed Y per level
        let sumX = 0, sumZ = 0, count = 0;
        nodePositionCache.forEach(pos => {
            if (!Array.isArray(pos) || pos.length !== 3) return;
            sumX += pos[0];
            sumZ += pos[2];
            count += 1;
        });
        if (!count) return;

        const offsetX = sumX / count;
        const offsetZ = sumZ / count;

        nodePositionCache.forEach((pos, id) => {
            if (!Array.isArray(pos) || pos.length !== 3) return;
            const adjusted = [
                pos[0] - offsetX,
                pos[1],  // Keep Y as-is (fixed per level)
                pos[2] - offsetZ
            ];
            nodePositionCache.set(id, adjusted);
        });

        if (fullData && fullData.positions) {
            Object.entries(fullData.positions).forEach(([id, pos]) => {
                const cached = nodePositionCache.get(id);
                if (cached) {
                    fullData.positions[id] = cached.slice();
                } else if (Array.isArray(pos) && pos.length === 3) {
                    fullData.positions[id] = [
                        pos[0] - offsetX,
                        pos[1],  // Keep Y as-is
                        pos[2] - offsetZ
                    ];
                }
            });
        }

        if (fullData && fullData.clusters) {
            ['level_1', 'level_2', 'level_3'].forEach(levelKey => {
                const layer = fullData.clusters[levelKey];
                if (!layer) return;
                Object.values(layer).forEach(cluster => {
                    if (!cluster || !cluster.id) return;
                    const cached = nodePositionCache.get(cluster.id);
                    if (cached) {
                        cluster.position = cached.slice();
                    } else if (Array.isArray(cluster.position) && cluster.position.length === 3) {
                        cluster.position = [
                            cluster.position[0] - offsetX,
                            cluster.position[1],  // Keep Y as-is
                            cluster.position[2] - offsetZ
                        ];
                    }
                });
            });
        }
    }

    function initControls() {
        const typeContainer = document.getElementById('kg-entity-type-chips');
        if (typeContainer) {
            entityTypes.forEach(t => {
                const chip = document.createElement('button');
                chip.type = 'button';
                chip.className = 'kg-chip active';
                chip.dataset.type = t;
                chip.textContent = t;
                chip.addEventListener('click', () => {
                    if (activeEntityTypes.has(t)) {
                        activeEntityTypes.delete(t);
                        chip.classList.remove('active');
                    } else {
                        activeEntityTypes.add(t);
                        chip.classList.add('active');
                    }
                    refreshGraph();
                });
                typeContainer.appendChild(chip);
            });
        }

        const relContainer = document.getElementById('kg-relationship-type-chips');
        if (relContainer) {
            relationshipTypes.forEach(rt => {
                const chip = document.createElement('button');
                chip.type = 'button';
                chip.className = 'kg-chip active';
                chip.dataset.type = rt;
                chip.textContent = rt;
                chip.addEventListener('click', () => {
                    if (activeRelationshipTypes.has(rt)) {
                        activeRelationshipTypes.delete(rt);
                        chip.classList.remove('active');
                    } else {
                        activeRelationshipTypes.add(rt);
                        chip.classList.add('active');
                    }
                    refreshGraph();
                });
                relContainer.appendChild(chip);
            });
        }

        const epSelect = document.getElementById('kg-episode-filter');
        if (epSelect) {
            availableEpisodes.forEach(ep => {
                const opt = document.createElement('option');
                opt.value = String(ep);
                opt.textContent = `Episode ${ep}`;
                epSelect.appendChild(opt);
            });
            epSelect.addEventListener('change', () => {
                activeEpisodeFilter = epSelect.value;
                refreshGraph();
            });
        }

        const bookSelect = document.getElementById('kg-book-filter');
        if (bookSelect) {
            availableBooks.forEach(b => {
                const opt = document.createElement('option');
                opt.value = b;
                opt.textContent = bookDisplayName(b);
                bookSelect.appendChild(opt);
            });
            bookSelect.addEventListener('change', () => {
                activeBookFilter = bookSelect.value;
                refreshGraph();
            });
        }

        ['kg-level-0', 'kg-level-1', 'kg-level-2', 'kg-level-3'].forEach(id => {
            const checkbox = document.getElementById(id);
            if (!checkbox) return;
            const level = parseInt(id.split('-')[2], 10);

            // Default: only level 3 checked on load
            if (level === 3) {
                checkbox.checked = true;
                visibleLevels.add(3);
            } else {
                checkbox.checked = false;
                visibleLevels.delete(level);
            }

            checkbox.addEventListener('change', () => {
                if (checkbox.checked) {
                    visibleLevels.add(level);
                } else {
                    visibleLevels.delete(level);
                }
                refreshGraph();
            });
        });

        const hierarchyCheckbox = document.getElementById('kg-show-hierarchy');
        if (hierarchyCheckbox) {
            hierarchyCheckbox.addEventListener('change', () => {
                showHierarchyEdges = hierarchyCheckbox.checked;
                refreshGraph();
            });
        }

        const relCheckbox = document.getElementById('kg-show-relationships');
        if (relCheckbox) {
            relCheckbox.addEventListener('change', () => {
                showRelationshipEdges = relCheckbox.checked;
                refreshGraph();
            });
        }

        const maxEntitiesSlider = document.getElementById('kg-max-entities');
        const maxEntitiesLabel = document.getElementById('kg-max-entities-label');
        if (maxEntitiesSlider && maxEntitiesLabel) {
            maxEntitiesSlider.addEventListener('input', () => {
                maxEntitiesPerCluster = parseInt(maxEntitiesSlider.value, 10) || 200;
                maxEntitiesLabel.textContent = String(maxEntitiesPerCluster);
                refreshGraph();
            });
        }

        const searchInput = document.getElementById('kg-search-input');
        if (searchInput) {
            searchInput.addEventListener('input', () => {
                const query = searchInput.value.trim();
                if (query.length >= 2) {
                    searchEntities(query);
                } else {
                    const sr = document.getElementById('kg-search-results');
                    if (sr) sr.style.display = 'none';
                }
            });

            searchInput.addEventListener('keydown', e => {
                if (e.key === 'Enter') {
                    const query = searchInput.value.trim();
                    if (query) {
                        searchEntities(query);
                    }
                }
            });
        }

        const resetBtn = document.getElementById('kg-reset-view');
        if (resetBtn) {
            resetBtn.addEventListener('click', () => {
                expandedNodes.clear();
                selectedNode = null;
                pinnedEntityId = null;
                showTopLevelClusters();
                resetCamera();
                updateInfoPanel(null);
            });
        }

        const exportBtn = document.getElementById('kg-export-subgraph');
        if (exportBtn) {
            exportBtn.addEventListener('click', () => {
                exportSubgraph();
            });
        }

        const focusBtn = document.getElementById('kg-focus-node');
        if (focusBtn) {
            focusBtn.addEventListener('click', () => {
                if (selectedNode) {
                    focusCameraOnNode(selectedNode, 150);
                }
            });
        }

        const closePanelBtn = document.getElementById('kg-close-panel');
        if (closePanelBtn) {
            closePanelBtn.addEventListener('click', () => {
                selectedNode = null;
                updateInfoPanel(null);
                if (graph) {
                    const data = graph.graphData();
                    graph.nodeColor(n => n.color);
                }
            });
        }

        // View mode selector
        const viewModeSelect = document.getElementById('kg-view-mode');
        if (viewModeSelect) {
            viewModeSelect.addEventListener('change', () => {
                const newMode = viewModeSelect.value;
                if (newMode !== currentViewMode) {
                    currentViewMode = newMode;
                    switchViewMode(newMode);
                }
            });
        }

        // Size mode selector for Full Force view
        const sizeModeSelect = document.getElementById('kg-size-mode');
        if (sizeModeSelect) {
            sizeModeSelect.addEventListener('change', () => {
                nodeSizeMode = sizeModeSelect.value;
                if (currentViewMode === 'full-force') {
                    // Refresh the graph to update node sizes
                    showFullForceView();
                }
            });
        }

        // Membrane controls for Full Force view
        ['kg-membrane-l1', 'kg-membrane-l2', 'kg-membrane-l3'].forEach(id => {
            const checkbox = document.getElementById(id);
            if (!checkbox) return;
            checkbox.addEventListener('change', () => {
                if (currentViewMode === 'full-force') {
                    updateMembraneVisibility();
                }
            });
        });

        document.addEventListener('keydown', e => {
            if (e.key === 'Escape') {
                selectedNode = null;
                pinnedEntityId = null;
                updateInfoPanel(null);
                if (graph) {
                    const data = graph.graphData();
                    graph.nodeColor(n => n.color);
                }
            } else if (e.key === 'r' || e.key === 'R') {
                expandedNodes.clear();
                selectedNode = null;
                pinnedEntityId = null;
                showTopLevelClusters();
                resetCamera();
                updateInfoPanel(null);
            } else if (e.key === '/' && searchInput && document.activeElement !== searchInput) {
                e.preventDefault();
                searchInput.focus();
            }
        });
    }

    function initGraph() {
        const container = document.getElementById('kg-3d-graph');
        if (!container) {
            console.error('KG 3D container not found');
            return;
        }

        container.innerHTML = '';

        graph = ForceGraph3D()(container)
            .nodeId('id')
            .nodeLabel(node => {
                if (node.level === 0) {
                    const deg = node.degree || entityDegrees.get(node.id) || 0;

                    // Only show labels for top 20% most connected nodes in Full Force mode
                    if (currentViewMode === 'full-force') {
                        const threshold = connectivityStats.maxDegree * 0.2;
                        if (deg < threshold) return '';  // No label for low-degree nodes
                    }

                    return `${node.name}\n${node.type} · degree ${deg}${node.description ? '\n' + node.description.substring(0, 140) + '...' : ''}`;
                } else {
                    return `${node.name}\n${node.childCount ? node.childCount + ' entities' : ''}`;
                }
            })
            .nodeColor(node => node.color)
            .nodeVal(node => {
                if (node.level === 0) {
                    const deg = entityDegrees.get(node.id) || 0;
                    const base = CONFIG.nodeSizes.entityBase;
                    return Math.pow(base + Math.log2(deg + 1), 2);
                }
                return Math.pow(node.size || 1, 2);
            })
            .nodeOpacity(node => node.opacity || 0.92)
            .linkSource('source')
            .linkTarget('target')
            .linkColor(link => link.color)
            .linkWidth(link => link.width || 0.7)
            .linkOpacity(link => link.opacity || 0.6)
            .backgroundColor('#02040f')
            .showNavInfo(true)
            .enableNodeDrag(false)
            .nodeThreeObject(node => buildNodeObject(node));

        // HOLOGRAPHIC ELEVATOR: Enable force simulation for entity nodes (L0) only
        graph.cooldownTime(3000);  // Allow forces to settle
        graph.d3Force('center', null);  // No centering force

        // Custom charge force that only affects entity nodes
        const chargeForce = graph.d3Force('charge');
        if (chargeForce && chargeForce.strength) {
            chargeForce.strength(node => {
                // Only apply repulsion to entity nodes (Level 0)
                return node.level === 0 ? -30 : 0;
            });
        }

        // Custom link force that only affects entity-entity relationships
        const linkForce = graph.d3Force('link');
        if (linkForce && linkForce.strength) {
            linkForce.strength(link => {
                // Only apply link forces between entities
                const sourceLevel = link.source?.level ?? (typeof link.source === 'object' ? 0 : null);
                const targetLevel = link.target?.level ?? (typeof link.target === 'object' ? 0 : null);
                return (sourceLevel === 0 && targetLevel === 0) ? 0.3 : 0;
            });
        }

        // Custom positioning force to keep cluster nodes fixed on their planes
        // Use d3AlphaDecay to ensure simulation runs and settles
        graph.d3AlphaDecay(0.02);  // Slower decay for smoother settling

        graph.d3VelocityDecay(0.3);  // Damping for smoother motion

        resetCamera();

        graph.onNodeClick(node => {
            if (!node) return;
            handleNodeClick(node);
        });

        graph.onNodeHover(node => {
            container.style.cursor = node ? 'pointer' : 'default';
        });

        // HOLOGRAPHIC ELEVATOR: Add visual floor planes after graph initialization
        setTimeout(() => {
            addFloorPlanes();
        }, 500);
    }

    function addFloorPlanes() {
        if (!graph || !window.THREE) return;

        const scene = graph.scene();
        if (!scene) return;

        // HOLOGRAPHIC ELEVATOR: Create semi-transparent floor planes at each level
        const floors = [
            { y: 200, color: 0xff00ff, label: 'Level 3: Global Themes', type: 'plane' },
            { y: 0, color: 0x00aaff, label: 'Level 2: Communities', type: 'plane' },
            { y: -200, color: 0x00ff88, label: 'Level 1: Topics', type: 'plane' },
            { y: -280, color: 0xffff00, label: 'Level 0: 3D Entity Cloud', type: 'boundary' }  // Boundary marker
        ];

        floors.forEach(floor => {
            if (floor.type === 'plane') {
                // Create circular floor plane (like a glass disc) for L1, L2, L3
                const geometry = new THREE.CircleGeometry(500, 64);
                const material = new THREE.MeshBasicMaterial({
                    color: floor.color,
                    transparent: true,
                    opacity: 0.05,
                    side: THREE.DoubleSide,
                    depthWrite: false
                });
                const plane = new THREE.Mesh(geometry, material);
                plane.rotation.x = -Math.PI / 2;  // Rotate to horizontal
                plane.position.y = floor.y;
                plane.userData = { isFloorPlane: true };
                scene.add(plane);
                floorPlaneObjects.push(plane);

                // Add glowing ring around the edge
                const ringGeometry = new THREE.RingGeometry(490, 500, 64);
                const ringMaterial = new THREE.MeshBasicMaterial({
                    color: floor.color,
                    transparent: true,
                    opacity: 0.3,
                    side: THREE.DoubleSide
                });
                const ring = new THREE.Mesh(ringGeometry, ringMaterial);
                ring.rotation.x = -Math.PI / 2;
                ring.position.y = floor.y;
                ring.userData = { isFloorPlane: true };
                scene.add(ring);
                floorPlaneObjects.push(ring);
            } else if (floor.type === 'boundary') {
                // Create subtle boundary ring for 3D entity zone
                const ringGeometry = new THREE.RingGeometry(480, 500, 64);
                const ringMaterial = new THREE.MeshBasicMaterial({
                    color: floor.color,
                    transparent: true,
                    opacity: 0.15,  // More subtle
                    side: THREE.DoubleSide
                });
                const ring = new THREE.Mesh(ringGeometry, ringMaterial);
                ring.rotation.x = -Math.PI / 2;
                ring.position.y = floor.y;
                ring.userData = { isFloorPlane: true };
                scene.add(ring);
                floorPlaneObjects.push(ring);
            }
        });

        console.log('[KG] HOLOGRAPHIC ELEVATOR: Added floor planes');
    }

    function removeFloorPlanes() {
        if (!graph || !window.THREE) return;

        const scene = graph.scene();
        if (!scene) return;

        floorPlaneObjects.forEach(obj => {
            scene.remove(obj);
            if (obj.geometry) obj.geometry.dispose();
            if (obj.material) obj.material.dispose();
        });

        floorPlaneObjects = [];
    }

    let activeBeams = [];  // Track active beam objects for cleanup

    function addExpansionBeam(parentNode, childrenNodes) {
        if (!graph || !window.THREE) return;

        const scene = graph.scene();
        if (!scene) return;

        const parentPos = getNodePosition(parentNode);
        if (!parentPos) return;

        // HOLOGRAPHIC ELEVATOR: Create beam of light from parent to children
        childrenNodes.forEach(childNode => {
            const childPos = getNodePosition(childNode);
            if (!childPos) return;

            const [px, py, pz] = parentPos;
            const [cx, cy, cz] = childPos;

            // Create conical beam geometry
            const height = Math.abs(py - cy);
            const radiusTop = 2;
            const radiusBottom = 15;

            const geometry = new THREE.ConeGeometry(radiusBottom, height, 16, 1, true);
            const material = new THREE.MeshBasicMaterial({
                color: 0x00ffff,
                transparent: true,
                opacity: 0.15,
                side: THREE.DoubleSide,
                depthWrite: false
            });

            const cone = new THREE.Mesh(geometry, material);

            // Position cone between parent and child
            cone.position.set(px, (py + cy) / 2, pz);

            // Rotate cone to point down (if parent is above child)
            if (py > cy) {
                cone.rotation.x = 0;  // Point down
            } else {
                cone.rotation.x = Math.PI;  // Point up
            }

            scene.add(cone);
            activeBeams.push(cone);

            // Animate beam: fade in, then fade out
            let opacity = 0;
            const fadeIn = setInterval(() => {
                opacity += 0.02;
                material.opacity = Math.min(opacity, 0.25);
                if (opacity >= 0.25) {
                    clearInterval(fadeIn);

                    // After 2 seconds, fade out
                    setTimeout(() => {
                        const fadeOut = setInterval(() => {
                            opacity -= 0.02;
                            material.opacity = Math.max(opacity, 0);
                            if (opacity <= 0) {
                                clearInterval(fadeOut);
                                scene.remove(cone);
                                activeBeams = activeBeams.filter(b => b !== cone);
                                geometry.dispose();
                                material.dispose();
                            }
                        }, 30);
                    }, 2000);
                }
            }, 30);
        });

        console.log('[KG] HOLOGRAPHIC ELEVATOR: Created expansion beam', { parent: parentNode.id, childCount: childrenNodes.length });
    }

    function buildNodeObject(node) {
        if (!window.THREE) return false;

        if (node.level >= 2) {
            const group = new THREE.Group();

            const geometry = new THREE.SphereGeometry(node.size || 6, 24, 24);
            const material = new THREE.MeshPhongMaterial({
                color: node.color,
                emissive: node.color,
                emissiveIntensity: 0.35,
                shininess: 120
            });
            const sphere = new THREE.Mesh(geometry, material);
            group.add(sphere);

            if (node.children && node.children.length > 0) {
                const plusGeom = new THREE.BoxGeometry((node.size || 6) * 0.2, (node.size || 6) * 1.1, (node.size || 6) * 0.2);
                const plusMat = new THREE.MeshBasicMaterial({
                    color: '#ffffff',
                    opacity: node.expanded ? 0.0 : 0.75,
                    transparent: true
                });
                const plusVert = new THREE.Mesh(plusGeom, plusMat);
                const plusHoriz = new THREE.Mesh(
                    new THREE.BoxGeometry((node.size || 6) * 1.1, (node.size || 6) * 0.2, (node.size || 6) * 0.2),
                    plusMat
                );
                if (!node.expanded) {
                    group.add(plusVert);
                    group.add(plusHoriz);
                }
            }

            return group;
        }

        if (node.level === 1 || node.level === 0) {
            const group = new THREE.Group();
            const size = node.level === 1 ? (node.size || 3) * 1.6 : CONFIG.nodeSizes.entityBase * 1.7;

            const geom = new THREE.SphereGeometry(size, 20, 20);
            const mat = new THREE.MeshPhongMaterial({
                color: node.color || '#00ffff',
                emissive: node.color || '#00ffff',
                emissiveIntensity: 0.5,
                shininess: 110
            });
            const sphere = new THREE.Mesh(geom, mat);
            group.add(sphere);

            if (node.level === 0 && node.name) {
                const sprite = makeSpriteText(node.name);
                sprite.color = '#ffffff';
                sprite.textHeight = 3;
                sprite.position.y = size + 2;
                group.add(sprite);
            }

            return group;
        }

        return false;
    }

    function makeSpriteText(text) {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        ctx.font = '50px Arial';
        const metrics = ctx.measureText(text);

        canvas.width = metrics.width + 20;
        canvas.height = 70;

        ctx.font = '50px Arial';
        ctx.fillStyle = 'white';
        ctx.fillText(text, 10, 50);

        const texture = new THREE.CanvasTexture(canvas);
        const material = new THREE.SpriteMaterial({ map: texture });
        const sprite = new THREE.Sprite(material);
        sprite.scale.set(canvas.width / 10, canvas.height / 10, 1);
        return sprite;
    }

    function resetCamera() {
        if (!graph) return;
        // HOLOGRAPHIC ELEVATOR: Side-angle view to see vertical stratification
        // Position camera to see all floors (L3: y=200, L2: y=0, L1: y=-200, L0: y=-400)
        graph.cameraPosition(
            { x: 350, y: 100, z: 400 },  // Side-angle position
            { x: 0, y: 0, z: 0 },         // Look at center
            1000                           // Smooth transition
        );
    }

    function showTopLevelClusters() {
        const clusters = fullData.clusters || {};
        const l3 = clusters.level_3 || {};
        const l2 = clusters.level_2 || {};
        const l1 = clusters.level_1 || {};

        const nodes = [];
        const links = [];

        let l3Values = Object.values(l3);
        if (!l3Values.length) {
            showError('No level 3 clusters found in hierarchy.');
            return;
        }

        // Sort by descendant count and limit to top N, filtering out empty clusters
        l3Values = l3Values
            .map(c => ({ ...c, childCount: countDescendants(c.id, l2, l1) }))
            .filter(c => (c.childCount || 0) > 0)  // Only show clusters with content
            .sort((a, b) => (b.childCount || 0) - (a.childCount || 0))
            .slice(0, MAX_L3_CLUSTERS);

        l3Values.forEach((cluster, idx) => {
            const pos = cluster.position || [
                Math.cos((2 * Math.PI * idx) / l3Values.length) * 180,
                Math.sin((2 * Math.PI * idx) / l3Values.length) * 180,
                0
            ];
            const cachedPos = nodePositionCache.get(cluster.id);
            const finalPos = cachedPos ? cachedPos.slice() : pos;

            const childCount = cluster.childCount;

            const node = {
                id: cluster.id,
                name: cluster.name || `Category ${idx + 1}`,
                type: 'coarse_cluster',
                level: 3,
                position: finalPos,
                children: cluster.children || [],
                childCount,
                color: CONFIG.nodeColors.coarse_cluster,
                size: CONFIG.nodeSizes.coarse_cluster,
                expanded: expandedNodes.has(cluster.id),
                opacity: 1.0
            };
            nodes.push(node);
            nodePositionCache.set(cluster.id, finalPos);
        });

        for (let i = 0; i < nodes.length; i++) {
            links.push({
                source: nodes[i].id,
                target: nodes[(i + 1) % nodes.length].id,
                type: 'category_ring',
                color: CONFIG.linkColors.categoryRing,
                width: 0.5,
                opacity: 0.35
            });
        }

        updateGraph({ nodes, links }, { recenter: true });
        // Reset info panel / breadcrumb to top-level view
        updateInfoPanel(null);
    }

    function countDescendants(l3Id, level2, level1) {
        const cat = level2 && level1 ? level2 : fullData.clusters.level_2;
        const fine = level1 || fullData.clusters.level_1;
        const l3 = fullData.clusters.level_3 || {};
        const cluster = l3[l3Id];
        if (!cluster || !cluster.children) return 0;
        let count = 0;
        cluster.children.forEach(l2Id => {
            const m = cat[l2Id];
            if (!m || !m.children) return;
            m.children.forEach(l1Id => {
                const f = fine[l1Id];
                if (f && Array.isArray(f.entities)) {
                    count += f.entities.length;
                }
            });
        });
        return count;
    }

    function buildExpandedGraph() {
        const clusters = fullData.clusters || {};
        const l3 = clusters.level_3 || {};
        const l2 = clusters.level_2 || {};
        const l1 = clusters.level_1 || {};
        const entities = fullData.entities || {};

        const nodes = [];
        const links = [];
        const addedNodeIds = new Set();

        let l3Values = Object.values(l3);
        // Sort and cap to top N by descendant count, filtering out empty clusters
        l3Values = l3Values
            .map(c => ({ ...c, childCount: countDescendants(c.id, l2, l1) }))
            .filter(c => (c.childCount || 0) > 0)  // Only show clusters with content
            .sort((a, b) => (b.childCount || 0) - (a.childCount || 0))
            .slice(0, MAX_L3_CLUSTERS);

        console.log('[KG] buildExpandedGraph start', {
            l3Count: l3Values.length,
            visibleLevels: Array.from(visibleLevels),
            expandedNodes: Array.from(expandedNodes)
        });

        l3Values.forEach((cluster, idx) => {
            if (!visibleLevels.has(3)) return;
            if (!addedNodeIds.has(cluster.id)) {
                const fallbackPos = [
                    Math.cos((2 * Math.PI * idx) / l3Values.length) * 200,
                    Math.sin((2 * Math.PI * idx) / l3Values.length) * 200,
                    0
                ];
                const cachedPos = nodePositionCache.get(cluster.id);
                const clusterPos = cachedPos
                    ? cachedPos.slice()
                    : (Array.isArray(cluster.position) && cluster.position.length === 3 ? cluster.position.slice() : fallbackPos);

                const childCount = countDescendants(cluster.id, l2, l1);

                nodes.push({
                    id: cluster.id,
                    name: cluster.name || `Category ${idx + 1}`,
                    type: 'coarse_cluster',
                    level: 3,
                    position: clusterPos,
                    children: cluster.children || getL2Children(l2, cluster.id),
                    childCount,
                    color: CONFIG.nodeColors.coarse_cluster,
                    size: CONFIG.nodeSizes.coarse_cluster,
                    expanded: expandedNodes.has(cluster.id),
                    opacity: 1.0
                });
                addedNodeIds.add(cluster.id);
                nodePositionCache.set(cluster.id, clusterPos);
            }

            const l2Children = Array.isArray(cluster.children) && cluster.children.length
                ? cluster.children
                : getL2Children(l2, cluster.id);

            if (expandedNodes.has(cluster.id)) {
                console.log("[KG] L3 expand", {
                    id: cluster.id,
                    l2Count: l2Children.length,
                    l2Sample: l2Children.slice(0, 5),
                    visibleLevels: Array.from(visibleLevels)
                });
            }

            if (expandedNodes.has(cluster.id) && Array.isArray(l2Children) && l2Children.length > 0) {
                l2Children.forEach((l2Id, index2) => {
                    const m = l2[l2Id];
                    if (!m || addedNodeIds.has(l2Id)) return;

                    if (!visibleLevels.has(2)) return;

                    const parentPos = nodePositionCache.get(cluster.id) || [0, 200, 0];  // L3 at y=200
                    const cachedPosL2 = nodePositionCache.get(l2Id);
                    const mPos = cachedPosL2 || calculateChildPosition(
                        [parentPos[0], 0, parentPos[2]],  // L2 children at y=0
                        index2,
                        l2Children.length || 1,
                        140,
                        l2Id,
                        2  // level 2
                    );

                    nodes.push({
                        id: l2Id,
                        name: m.name || `Topic ${l2Id}`,
                        type: 'medium_cluster',
                        level: 2,
                        position: mPos,
                        children: m.children || [],
                        color: CONFIG.nodeColors.medium_cluster,
                        size: CONFIG.nodeSizes.medium_cluster,
                        expanded: expandedNodes.has(l2Id),
                        opacity: 0.95
                    });
                    addedNodeIds.add(l2Id);
                    nodePositionCache.set(l2Id, mPos);

                    if (showHierarchyEdges) {
                        links.push({
                            source: cluster.id,
                            target: l2Id,
                            type: 'hierarchy',
                            color: CONFIG.linkColors.hierarchy,
                            width: 1.8,
                            opacity: 0.9
                        });
                    }

                    const l1Children = Array.isArray(m.children) && m.children.length
                        ? m.children
                        : getL1Children(l1, l2Id);

                    if (expandedNodes.has(l2Id)) {
                        console.log("[KG] L2 expand", {
                            id: l2Id,
                            l1Count: l1Children.length,
                            l1Sample: l1Children.slice(0, 5),
                            visibleLevels: Array.from(visibleLevels)
                        });
                    }

                    if (expandedNodes.has(l2Id) && Array.isArray(l1Children)) {
                        l1Children.forEach((l1Id, index1) => {
                            const f = l1[l1Id];
                            if (!f || addedNodeIds.has(l1Id)) return;

                            if (!visibleLevels.has(1)) return;

                            const cachedPosL1 = nodePositionCache.get(l1Id);
                            const fPos = cachedPosL1 || calculateChildPosition(
                                [mPos[0], -200, mPos[2]],  // L1 children at y=-200
                                index1,
                                l1Children.length || 1,
                                70,
                                l1Id,
                                1  // level 1
                            );

                            nodes.push({
                                id: l1Id,
                                name: f.name || `Cluster ${l1Id}`,
                                type: 'fine_cluster',
                                level: 1,
                                position: fPos,
                                entities: f.entities || [],
                                color: CONFIG.nodeColors.fine_cluster,
                                size: CONFIG.nodeSizes.fine_cluster,
                                expanded: expandedNodes.has(l1Id),
                                opacity: 0.98
                            });
                            addedNodeIds.add(l1Id);
                            nodePositionCache.set(l1Id, fPos);

                            if (showHierarchyEdges) {
                                links.push({
                                    source: l2Id,
                                    target: l1Id,
                                    type: 'hierarchy',
                                    color: CONFIG.linkColors.hierarchy,
                                    width: 1.2,
                                    opacity: 0.7
                                });
                            }

                    if (expandedNodes.has(l1Id)) {
                        const ents = (f.entities || []).filter(eid => entityPassesFilters(entities[eid], eid));
                        const entitiesToShow = ents.slice(0, maxEntitiesPerCluster);

                                entitiesToShow.forEach((eid, idx) => {
                                    if (addedNodeIds.has(eid)) return;
                                    if (!visibleLevels.has(0)) return;

                                    const ent = entities[eid];
                                    if (!ent) return;

                                    const cachedEntityPos = nodePositionCache.get(eid);
                                    let ePos;
                                    if (cachedEntityPos) {
                                        // HOLOGRAPHIC ELEVATOR: Use original 3D position (already shifted down)
                                        ePos = cachedEntityPos;
                                    } else {
                                        // HOLOGRAPHIC ELEVATOR: Entities as 3D force-directed graph below L1
                                        // Create initial position in 3D around parent, below y=-200
                                        const angle = (idx * 2 * Math.PI) / Math.max(entitiesToShow.length, 1);
                                        const radius = 50;
                                        const ySpread = (Math.random() - 0.5) * 60;  // 3D vertical spread
                                        ePos = [
                                            fPos[0] + Math.cos(angle) * radius,
                                            -250 + ySpread,  // Center around y=-250 with spread
                                            fPos[2] + Math.sin(angle) * radius
                                        ];
                                    }

                                    const type = ent.type || 'CONCEPT';
                                    nodes.push({
                                        id: eid,
                                        name: ent.name || eid,
                                        type,
                                        level: 0,
                                        position: ePos,
                                        description: ent.description,
                                        color: CONFIG.nodeColors[type] || CONFIG.nodeColors.CONCEPT,
                                        size: CONFIG.nodeSizes.entityBase,
                                        opacity: 1.0
                                    });
                                    addedNodeIds.add(eid);
                                    nodePositionCache.set(eid, ePos);

                                    if (showHierarchyEdges) {
                                        links.push({
                                            source: l1Id,
                                            target: eid,
                                            type: 'hierarchy',
                                            color: CONFIG.linkColors.hierarchy,
                                            width: 0.6,
                                            opacity: 0.5
                                        });
                                    }
                                });
                            }
                        });
                    }
                });
            }
        });


                    // Fallback: no level-2 children; show top fine clusters directly under this L3
                    if (typeof cluster !== 'undefined' && expandedNodes.has(cluster.id) && l2Children.length === 0 && l1Sorted.length) {
                        // Ensure level 1 is visible for fallback
                        if (!visibleLevels.has(1)) {
                            visibleLevels.add(1);
                            const cb1 = document.getElementById('kg-level-1');
                            if (cb1) cb1.checked = true;
                        }
                        const fallback = l1Sorted.slice(0, FALLBACK_L1_PER_L3);
                        fallback.forEach((fNode, idxF) => {
                            if (addedNodeIds.has(fNode.id)) return;
                            const cachedFallbackPos = nodePositionCache.get(fNode.id);
                            const fPos = cachedFallbackPos || calculateChildPosition(nodePositionCache.get(cluster.id) || [0,0,0], idxF, fallback.length, 120, fNode.id);
                            nodes.push({
                                id: fNode.id,
                                name: fNode.name || fNode.id,
                                type: 'fine_cluster',
                                level: 1,
                                position: fPos,
                                entities: fNode.entities || [],
                                color: CONFIG.nodeColors.fine_cluster,
                                size: CONFIG.nodeSizes.fine_cluster,
                                expanded: expandedNodes.has(fNode.id),
                                opacity: 0.98
                            });
                            addedNodeIds.add(fNode.id);
                            nodePositionCache.set(fNode.id, fPos);
                            if (showHierarchyEdges) {
                                links.push({
                                    source: cluster.id,
                                    target: fNode.id,
                                    type: 'hierarchy',
                                    color: CONFIG.linkColors.hierarchy,
                                    width: 1.2,
                                    opacity: 0.7
                                });
                            }
                        });
                    }

        // Handle fine clusters that are expanded but not reachable via level_3/level_2 parents
        const orphanL1Entries = Object.entries(l1).filter(([l1Id, cluster]) => {
            if (!expandedNodes.has(l1Id)) return false;
            if (addedNodeIds.has(l1Id)) return false;
            const parentL2Id = cluster.parent;
            if (parentL2Id && l2[parentL2Id]) return false;
            return true;
        });

        orphanL1Entries.forEach(([l1Id, cluster], idx) => {
            if (!visibleLevels.has(1)) return;

            const cachedOrphanPos = nodePositionCache.get(l1Id);
            const fPos = cachedOrphanPos || calculateChildPosition([0, 0, 0], idx, orphanL1Entries.length, 220, l1Id);

            nodes.push({
                id: l1Id,
                name: cluster.name || `Cluster ${l1Id}`,
                type: 'fine_cluster',
                level: 1,
                position: fPos,
                entities: cluster.entities || [],
                color: CONFIG.nodeColors.fine_cluster,
                size: CONFIG.nodeSizes.fine_cluster,
                expanded: true,
                opacity: 0.98
            });
            addedNodeIds.add(l1Id);
            nodePositionCache.set(l1Id, fPos);

            if (expandedNodes.has(l1Id)) {
                const ents = (cluster.entities || []).filter(eid => entityPassesFilters(entities[eid], eid));
                const entitiesToShow = ents.slice(0, maxEntitiesPerCluster);

                entitiesToShow.forEach((eid, idx2) => {
                    if (addedNodeIds.has(eid)) return;
                    if (!visibleLevels.has(0)) return;

                    const ent = entities[eid];
                    if (!ent) return;

                    const cachedEntPos = nodePositionCache.get(eid);
                    const ePos = cachedEntPos || calculateChildPosition(
                        fPos,
                        idx2,
                        Math.max(entitiesToShow.length, 1),
                        35,
                        eid
                    );

                    const type = ent.type || 'CONCEPT';
                    nodes.push({
                        id: eid,
                        name: ent.name || eid,
                        type,
                        level: 0,
                        position: ePos,
                        description: ent.description,
                        color: CONFIG.nodeColors[type] || CONFIG.nodeColors.CONCEPT,
                        size: CONFIG.nodeSizes.entityBase,
                        opacity: 1.0
                    });
                    addedNodeIds.add(eid);
                    nodePositionCache.set(eid, ePos);

                    if (showHierarchyEdges) {
                        links.push({
                            source: l1Id,
                            target: eid,
                            type: 'hierarchy',
                            color: CONFIG.linkColors.hierarchy,
                            width: 0.6,
                            opacity: 0.5
                        });
                    }
                });
            }
        });

        // Ensure pinned search entity and its direct neighbors are present
        if (pinnedEntityId) {
            const ent = entities[pinnedEntityId];
            if (ent) {
                let pinnedNode = nodes.find(n => n.id === pinnedEntityId);

                if (!pinnedNode && visibleLevels.has(0)) {
                    const type = ent.type || 'CONCEPT';
                    const pPos = [0, 0, 0];
                    pinnedNode = {
                        id: pinnedEntityId,
                        name: ent.name || pinnedEntityId,
                        type,
                        level: 0,
                        position: pPos,
                        description: ent.description,
                        color: CONFIG.nodeColors[type] || CONFIG.nodeColors.CONCEPT,
                        size: CONFIG.nodeSizes.entityBase,
                        opacity: 1.0
                    };
                    nodes.push(pinnedNode);
                    addedNodeIds.add(pinnedEntityId);
                    nodePositionCache.set(pinnedEntityId, pPos);
                }

                if (pinnedNode && visibleLevels.has(0)) {
                    const rels = fullData.relationships || [];
                    const neighborIds = new Set();
                    rels.forEach(rel => {
                        const s = rel.source;
                        const t = rel.target;
                        if (s === pinnedEntityId) neighborIds.add(t);
                        if (t === pinnedEntityId) neighborIds.add(s);
                    });

                    const neighborArray = Array.from(neighborIds);
                    neighborArray.forEach((nid, idx) => {
                        if (addedNodeIds.has(nid)) return;
                        const nEnt = entities[nid];
                        if (!nEnt) return;

                        const cachedNeighborPos = nodePositionCache.get(nid);
                        const nPos = cachedNeighborPos || calculateChildPosition(
                            pinnedNode.position,
                            idx,
                            Math.max(neighborArray.length, 1),
                            55,
                            nid
                        );

                        const nType = nEnt.type || 'CONCEPT';
                        nodes.push({
                            id: nid,
                            name: nEnt.name || nid,
                            type: nType,
                            level: 0,
                            position: nPos,
                            description: nEnt.description,
                            color: CONFIG.nodeColors[nType] || CONFIG.nodeColors.CONCEPT,
                            size: CONFIG.nodeSizes.entityBase,
                            opacity: 1.0
                        });
                        addedNodeIds.add(nid);
                        nodePositionCache.set(nid, nPos);
                    });
                }
            }
        }

        if (showRelationshipEdges) {
            addRelationshipEdges(nodes, links);
        }

        const levelCounts = nodes.reduce((acc, n) => {
            acc[n.level] = (acc[n.level] || 0) + 1;
            return acc;
        }, {});
        console.log("[KG] buildExpandedGraph end", {
            totalNodes: nodes.length,
            totalLinks: links.length,
            levelCounts,
            expandedNodes: Array.from(expandedNodes),
            visibleLevels: Array.from(visibleLevels)
        });
        return { nodes, links };
    }

    function stableRandom(key) {
        const str = String(key || '');
        let hash = 0;
        for (let i = 0; i < str.length; i++) {
            hash = (hash << 5) - hash + str.charCodeAt(i);
            hash |= 0;
        }
        // Map hash to [0, 1]
        return (Math.sin(hash) + 1) / 2;
    }

    function calculateChildPosition(parentPos, index, total, radius, key, level) {
        const angleStep = (2 * Math.PI) / (total || 1);
        const angle = index * angleStep;
        const jitter = (stableRandom(`${key || ''}-${index}-${total}`) - 0.5) * radius * 0.3;

        // HOLOGRAPHIC ELEVATOR: Spread children in 2D on their designated floor
        // Use angle for x-z positioning, keep y fixed per level
        return [
            parentPos[0] + Math.cos(angle) * radius,
            parentPos[1],  // Keep same y as parent (already on correct floor)
            parentPos[2] + Math.sin(angle) * radius + jitter
        ];
    }

    function addRelationshipEdges(nodes, links) {
        const nodeIdSet = new Set(nodes.filter(n => n.level === 0).map(n => n.id));
        if (!nodeIdSet.size) return;

        const rels = fullData.relationships || [];
        rels.forEach(rel => {
            const s = rel.source;
            const t = rel.target;
            if (!nodeIdSet.has(s) || !nodeIdSet.has(t)) return;

            const type = rel.type || rel.predicate || rel.relationship_type || 'RELATED_TO';
            if (!activeRelationshipTypes.has(type)) return;

            links.push({
                source: s,
                target: t,
                type,
                color: CONFIG.linkColors.relationship,
                width: 0.4,
                opacity: 0.32
            });
        });
    }

    function entityPassesFilters(ent, id) {
        if (!ent) return false;

        if (!activeEntityTypes.has(ent.type)) return false;

        if (activeEpisodeFilter || activeBookFilter) {
            const sources = ent.sources || [];
            let ok = true;

            if (activeEpisodeFilter) {
                const epKey = `episode_${activeEpisodeFilter}`;
                ok = ok && sources.includes(epKey);
            }

            if (activeBookFilter) {
                ok = ok && sources.some(src => src.toLowerCase() === activeBookFilter);
            }

            if (!ok) return false;
        }

        return true;
    }

    function getNodePosition(node) {
        if (!node) return null;
        if (Array.isArray(node.position) && node.position.length === 3) {
            const [x, y, z] = node.position;
            if ([x, y, z].every(v => Number.isFinite(v))) return [x, y, z];
        }
        const coords = [node.x, node.y, node.z];
        if (coords.every(v => Number.isFinite(v))) return coords;
        return null;
    }

    function lockNodePositions(graphData) {
        if (!graphData || !Array.isArray(graphData.nodes)) return graphData;
        graphData.nodes.forEach(node => {
            const pos = getNodePosition(node);
            if (pos) {
                const [x, y, z] = pos;
                node.x = x; node.y = y; node.z = z;

                // HOLOGRAPHIC ELEVATOR: Lock cluster nodes (L1, L2, L3) but allow entities (L0) to move
                if (node.level >= 1) {
                    // Lock cluster nodes to their designated floor
                    node.fx = x;
                    node.fy = y;
                    node.fz = z;
                } else {
                    // Entity nodes (L0): allow free movement for force-directed layout
                    node.fx = null;
                    node.fy = null;
                    node.fz = null;
                }

                node.position = pos;
            }
        });
        return graphData;
    }

    function computeGraphBounds(nodes) {
        if (!Array.isArray(nodes) || !nodes.length) return null;
        let minX = Infinity, minY = Infinity, minZ = Infinity;
        let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
        let count = 0;

        nodes.forEach(n => {
            const pos = getNodePosition(n);
            if (!pos) return;
            const [x, y, z] = pos;
            minX = Math.min(minX, x); maxX = Math.max(maxX, x);
            minY = Math.min(minY, y); maxY = Math.max(maxY, y);
            minZ = Math.min(minZ, z); maxZ = Math.max(maxZ, z);
            count += 1;
        });

        if (!count) return null;
        const center = {
            x: (minX + maxX) / 2,
            y: (minY + maxY) / 2,
            z: (minZ + maxZ) / 2
        };
        const radius = Math.max(maxX - minX, maxY - minY, maxZ - minZ) / 2;
        return { center, radius };
    }

    function centerCameraOnGraph(nodes) {
        if (!graph) return;
        const bounds = computeGraphBounds(nodes);
        if (!bounds) return;
        const distance = Math.max(bounds.radius * 2.2, CONFIG.initialCameraDistance * 0.7);
        graph.cameraPosition(
            { x: bounds.center.x, y: bounds.center.y, z: bounds.center.z + distance },
            bounds.center,
            800
        );
    }

    function updateGraph(graphData, options = {}) {
        if (!graph) return;
        graphData = lockNodePositions(graphData);
        console.log('[KG] updateGraph', { nodes: graphData.nodes.length, links: graphData.links.length, expandedNodes: Array.from(expandedNodes), sampleNodes: graphData.nodes.slice(0,5).map(n=>({id:n.id, level:n.level})) });
        graph.graphData(graphData);
        if (options.recenter) {
            centerCameraOnGraph(graphData.nodes);
        }
    }

    function refreshGraph() {
        if (!graph || !fullData || !fullData.clusters || !fullData.clusters.level_3) return;
        const data = buildExpandedGraph();
        console.log("[KG] refreshGraph", { nodes: data.nodes.length, links: data.links.length, visibleLevels: Array.from(visibleLevels), expandedNodes: Array.from(expandedNodes) });
        const targetId = (selectedNode && selectedNode.id) || pinnedEntityId;
        const shouldRecenter = !targetId;
        updateGraph(data, { recenter: shouldRecenter });

        if (!shouldRecenter && targetId) {
            const updatedNode = data.nodes.find(n => n.id === targetId);
            if (updatedNode) {
                if (selectedNode) selectedNode = updatedNode;
                focusCameraOnNode(updatedNode, updatedNode.level === 0 ? 180 : updatedNode.level === 1 ? 220 : updatedNode.level === 2 ? 320 : 420);
            } else {
                selectedNode = null;
                pinnedEntityId = null;
                centerCameraOnGraph(data.nodes);
            }
        }
    }

    function recursivelyExpandSingleChildren(nodeId) {
        // Recursively expand nodes to reach actual children
        // If a node has no immediate children, drill down to find children at lower levels
        const clusters = fullData.clusters || {};
        const node = findClusterById(nodeId, clusters);

        if (!node) {
            return [nodeId];
        }

        const toExpand = [nodeId];

        // Determine node's level
        let nodeLevel = null;
        if (clusters.level_3 && clusters.level_3[nodeId]) nodeLevel = 3;
        else if (clusters.level_2 && clusters.level_2[nodeId]) nodeLevel = 2;
        else if (clusters.level_1 && clusters.level_1[nodeId]) nodeLevel = 1;

        if (nodeLevel === null) {
            return toExpand;
        }

        // Get children at immediate next level
        let children = node.children || [];

        // If node has children in its children array, process them
        if (children.length > 0) {
            // If exactly one child, recursively expand it too
            if (children.length === 1) {
                const childId = children[0];
                console.log("[KG] single child detected, recursively expanding", { parent: nodeId, child: childId });
                const childExpansions = recursivelyExpandSingleChildren(childId);
                toExpand.push(...childExpansions);
            }
            return toExpand;
        }

        // No children in the children array - drill down to find children at lower levels
        console.log("[KG] no immediate children, drilling down", { nodeId, nodeLevel });

        // Try to find children at the next level down by checking parent relationships
        const l3 = clusters.level_3 || {};
        const l2 = clusters.level_2 || {};
        const l1 = clusters.level_1 || {};

        if (nodeLevel === 3) {
            // Level 3 node - check for Level 2 children
            const l2Children = getL2Children(l2, nodeId);
            if (l2Children.length > 0) {
                console.log("[KG] found L2 children for L3 node", { nodeId, l2Children });
                // Expand all L2 children
                l2Children.forEach(childId => {
                    const childExpansions = recursivelyExpandSingleChildren(childId);
                    toExpand.push(...childExpansions);
                });
            } else {
                // No L2 children, check for L1 children
                const l1Children = getL1Children(l1, nodeId);
                if (l1Children.length > 0) {
                    console.log("[KG] found L1 children for L3 node (skipping L2)", { nodeId, l1Children });
                    // Expand all L1 children
                    l1Children.forEach(childId => {
                        const childExpansions = recursivelyExpandSingleChildren(childId);
                        toExpand.push(...childExpansions);
                    });
                }
            }
        } else if (nodeLevel === 2) {
            // Level 2 node - check for Level 1 children
            const l1Children = getL1Children(l1, nodeId);
            if (l1Children.length > 0) {
                console.log("[KG] found L1 children for L2 node", { nodeId, l1Children });
                // Expand all L1 children
                l1Children.forEach(childId => {
                    const childExpansions = recursivelyExpandSingleChildren(childId);
                    toExpand.push(...childExpansions);
                });
            }
        }

        return toExpand;
    }

    function ensureAllChildLevelsForChain(nodesToExpand) {
        // HOLOGRAPHIC ELEVATOR: Ensure all necessary child levels are visible for recursive expansion
        const clusters = fullData.clusters || {};
        const levelCheckbox = (lvl) => document.getElementById(`kg-level-${lvl}`);
        const levelsToEnable = new Set();

        nodesToExpand.forEach(nodeId => {
            const node = findClusterById(nodeId, clusters);
            if (!node) return;

            // Determine node's level by checking which level it belongs to
            let nodeLevel = null;
            if (clusters.level_3 && clusters.level_3[nodeId]) nodeLevel = 3;
            else if (clusters.level_2 && clusters.level_2[nodeId]) nodeLevel = 2;
            else if (clusters.level_1 && clusters.level_1[nodeId]) nodeLevel = 1;

            if (nodeLevel === null) return;

            // Enable child level for this node
            const childLevel = nodeLevel - 1;
            if (childLevel >= 0) {
                levelsToEnable.add(childLevel);
            }
        });

        // Enable all levels that need to be visible
        levelsToEnable.forEach(level => {
            if (!visibleLevels.has(level)) {
                visibleLevels.add(level);
                const cb = levelCheckbox(level);
                if (cb) cb.checked = true;
                console.log("[KG] auto-enabled level", { level });
            }
        });

        // Always enable Level 0 (entities) when expanding any cluster
        if (!visibleLevels.has(0)) {
            visibleLevels.add(0);
            const cb = levelCheckbox(0);
            if (cb) cb.checked = true;
            console.log("[KG] auto-enabled Level 0 (entities)");
        }
    }

    function findClusterById(id, clusters) {
        for (const level of ['level_3', 'level_2', 'level_1']) {
            if (clusters[level] && clusters[level][id]) {
                return clusters[level][id];
            }
        }
        return null;
    }

    function handleNodeClick(node) {
        selectedNode = node;
        const wasExpanded = expandedNodes.has(node.id);

        console.log("[KG] node click", { id: node.id, level: node.level, type: node.type, expanded: wasExpanded });
        highlightSelection(node);
        updateInfoPanel(node);

        // Update pinned entity anchor: entities pin, clusters clear
        if (node.level === 0) {
            pinnedEntityId = node.id;
        } else if (node.type && node.type.includes('cluster')) {
            pinnedEntityId = null;
        }

        // Single click on cluster: toggle expand/collapse
        if (node.type && node.type.includes('cluster')) {
            ensureAncestorExpanded(node);
            if (wasExpanded) {
                console.log("[KG] collapse cluster", node.id);
                collapseNode(node.id);
                // Don't move camera on collapse, just update the view
            } else {
                // Recursively expand single-child nodes
                const nodesToExpand = recursivelyExpandSingleChildren(node.id);
                console.log("[KG] expand cluster with recursive single-child expansion", {
                    id: node.id,
                    nodesToExpand,
                    visibleLevels: Array.from(visibleLevels)
                });

                // HOLOGRAPHIC ELEVATOR: Ensure all child levels are visible for the entire chain
                ensureAllChildLevelsForChain(nodesToExpand);

                // Expand all nodes in the chain
                nodesToExpand.forEach(id => expandNode(id));

                // HOLOGRAPHIC ELEVATOR: Create beam of light from parent to children
                setTimeout(() => {
                    const clusters = fullData.clusters || {};
                    const expandedNode = findClusterById(node.id, clusters);
                    if (expandedNode && expandedNode.children) {
                        const childNodes = expandedNode.children.map(childId => {
                            const childCluster = findClusterById(childId, clusters);
                            return childCluster;
                        }).filter(c => c);

                        if (childNodes.length > 0) {
                            addExpansionBeam(node, childNodes);
                        }
                    }
                }, 200);

                // HOLOGRAPHIC ELEVATOR: Maintain side-angle view while focusing on expanded cluster
                setTimeout(() => {
                    const pos = getNodePosition(node);
                    if (graph && pos) {
                        const [x, y, z] = pos;
                        // Adjust distance based on level
                        const baseDistance = node.level === 3 ? 400 : node.level === 2 ? 350 : node.level === 1 ? 300 : 250;
                        // Side-angle position: offset x and z, keep y elevated to see floors
                        const camX = x + baseDistance * 0.7;
                        const camY = y + baseDistance * 0.3;  // Slightly above to see vertical separation
                        const camZ = z + baseDistance * 0.8;
                        console.log("[KG] camera focus on expanded cluster (side-angle)", { x, y, z, camX, camY, camZ, level: node.level });
                        graph.cameraPosition(
                            { x: camX, y: camY, z: camZ },
                            { x, y, z },
                            1200  // Slower, smoother transition
                        );
                    }
                }, 100);  // Small delay to let nodes settle
            }
        } else {
            // For non-cluster entity nodes, gentle focus
            const pos = getNodePosition(node);
            if (graph && node && pos) {
                const [x, y, z] = pos;
                const baseDistance = 150;
                console.log("[KG] camera focus on entity", { x, y, z, distance: baseDistance });
                graph.cameraPosition(
                    { x, y, z: z + baseDistance },
                    { x, y, z },
                    800
                );
            }
        }
    }

    function handleNodeDoubleClick(node) {
        if (node.type && node.type.includes('cluster')) {
            if (expandedNodes.has(node.id)) {
                collapseNode(node.id);
            } else {
                expandNode(node.id);
            }
        }
    }

    function expandNode(nodeId) {
        expandedNodes.add(nodeId);
        const data = buildExpandedGraph();
        updateGraph(data);

        // Auto-expand immediate children to force visibility when levels are on
        if (fullData && fullData.clusters && fullData.clusters.level_2 && fullData.clusters.level_1) {
            const clusters = fullData.clusters;
            const l2 = clusters.level_2 || {};
            const l1 = clusters.level_1 || {};
            if (l2[nodeId]) {
                // node is level 2, expand its children if visible level 1
                const kids = l2[nodeId].children && l2[nodeId].children.length ? l2[nodeId].children : getL1Children(l1, nodeId);
                if (visibleLevels.has(1)) {
                    kids.forEach(k => expandedNodes.add(k));
                }
            } else if (clusters.level_3 && clusters.level_3[nodeId]) {
                // node is level 3, expand its l2 children if visible level 2
                const kids = clusters.level_3[nodeId].children && clusters.level_3[nodeId].children.length ? clusters.level_3[nodeId].children : getL2Children(l2, nodeId);
                if (visibleLevels.has(2)) {
                    kids.forEach(k => expandedNodes.add(k));
                }
            }
        }

        const data2 = buildExpandedGraph();
        updateGraph(data2);

        const node = data2.nodes.find(n => n.id === nodeId);
        if (node) {
            selectedNode = node;
            updateInfoPanel(node);
            // Focus camera a bit closer for expanded clusters
            focusCameraOnNode(node, node.level === 3 ? 500 : node.level === 2 ? 400 : 250);
            console.log("[KG] expandNode", { nodeId, level: node.level, visibleLevels: Array.from(visibleLevels), expandedNodes: Array.from(expandedNodes) });
        }
    }

    function collapseNode(nodeId) {
        expandedNodes.delete(nodeId);
        const data = buildExpandedGraph();
        updateGraph(data);
    }

    function focusCameraOnNode(node, distance) {
        if (!graph || !node) return;
        const pos = getNodePosition(node);
        if (!pos) return;
        const [x, y, z] = pos;
        const dz = distance || 200;
        graph.cameraPosition(
            { x, y, z: z + dz },
            { x, y, z },
            1100
        );
    }

    function highlightSelection(node) {
        if (!graph || !node) return;
        const data = graph.graphData();

        const connectedIds = new Set([node.id]);
        data.links.forEach(link => {
            const s = typeof link.source === 'object' ? link.source.id : link.source;
            const t = typeof link.target === 'object' ? link.target.id : link.target;
            if (s === node.id) connectedIds.add(t);
            if (t === node.id) connectedIds.add(s);
        });

        graph.nodeColor(n => {
            if (n.id === node.id) return '#ffff00';
            if (connectedIds.has(n.id)) return n.color;
            return `${n.color}33`;
        });
    }

    function updateInfoPanel(node) {
        const titleEl = document.getElementById('kg-node-title');
        const detailsEl = document.getElementById('kg-node-details');
        const metaEl = document.getElementById('kg-node-meta');
        const sourcesEl = document.getElementById('kg-node-sources');
        const relatedEl = document.getElementById('kg-node-related');
        const relEl = document.getElementById('kg-node-relationships');
        const breadcrumbEl = document.getElementById('kg-breadcrumb');
        const levelDots = document.querySelectorAll('.kg-level-dot');

        if (!titleEl || !detailsEl || !metaEl || !sourcesEl || !relatedEl || !relEl || !breadcrumbEl) return;

        if (!node) {
            titleEl.textContent = 'Select a node';
            detailsEl.textContent = '';
            metaEl.textContent = '';
            sourcesEl.textContent = '';
            relatedEl.textContent = '';
            relEl.textContent = '';
            breadcrumbEl.textContent = 'Coarse categories · click a cluster to explore.';
            levelDots.forEach(d => d.classList.remove('active'));
            return;
        }

        titleEl.textContent = node.name;

        levelDots.forEach(dot => {
            dot.classList.toggle('active', parseInt(dot.dataset.level, 10) === node.level);
        });

        detailsEl.innerHTML = `
            <div class="entity-type-pill">${node.type}</div>
            <div style="margin-top:4px;"><strong>Level:</strong> ${getLevelName(node.level)}</div>
            ${node.description ? `<div style="margin-top:6px;">${node.description}</div>` : ''}
        `;

        metaEl.innerHTML = '';

        const breadcrumb = buildBreadcrumb(node);
        breadcrumbEl.textContent = breadcrumb;

        const entities = fullData.entities || {};
        const ent = node.level === 0 ? entities[node.id] : null;
        if (ent) {
            const degree = entityDegrees.get(node.id) || 0;
            metaEl.innerHTML = `
                <div><strong>Degree:</strong> ${degree}</div>
                ${ent.aliases && ent.aliases.length ? `<div><strong>Aliases:</strong> ${ent.aliases.join(', ')}</div>` : ''}
            `;

            const srcParts = [];
            const episodes = [];
            const books = [];
            (ent.sources || []).forEach(src => {
                if (typeof src !== 'string') return;
                if (src.startsWith('episode_')) {
                    const n = src.replace('episode_', '');
                    if (!isNaN(parseInt(n, 10))) episodes.push(parseInt(n, 10));
                } else {
                    const key = src.toLowerCase();
                    if (['veriditas', 'viriditas', 'y-on-earth', 'soil-stewardship-handbook', 'our-biggest-deal'].includes(key)) {
                        books.push(bookDisplayName(key));
                    } else {
                        srcParts.push(src);
                    }
                }
            });

            const srcHtml = [];
            if (episodes.length) {
                const uniq = Array.from(new Set(episodes)).sort((a, b) => a - b);
                const episodesHtml = uniq.slice(0, 12).map(ep => {
                    const chunkLink = buildEpisodeDeepLink(ent, ep);
                    if (chunkLink) {
                        return `<a href="${chunkLink.href}" target="_blank" class="kg-episode-pill" title="${chunkLink.title}">Ep ${ep} @ ${chunkLink.timestamp}</a>`;
                    }
                    return `<span class="kg-episode-pill">Episode ${ep}</span>`;
                }).join('');
                srcHtml.push(`<div><strong>Episodes:</strong></div><div>${episodesHtml}${uniq.length > 12 ? ` <span style="color:#9ecfff;">(+${uniq.length - 12} more)</span>` : ''}</div>`);
            }

            if (books.length) {
                const uniqBooks = Array.from(new Set(books));
                const booksHtml = uniqBooks.map(b => `<span class="kg-book-pill">${b}</span>`).join('');
                srcHtml.push(`<div style="margin-top:4px;"><strong>Books:</strong></div><div>${booksHtml}</div>`);
            }

            if (srcParts.length) {
                srcHtml.push(`<div style="margin-top:4px;"><strong>Other sources:</strong> ${srcParts.join(', ')}</div>`);
            }

            const chunkLinks = buildChunkDeepLinks(ent);
            sourcesEl.innerHTML = srcHtml.join('') + (chunkLinks ? `<div style="margin-top:6px;"><strong>Specific appearances:</strong></div><div>${chunkLinks}</div>` : '');

            const relInfo = buildRelationshipInfo(node.id);
            relEl.innerHTML = relInfo.relationshipHtml;
            relatedEl.innerHTML = relInfo.relatedHtml;
        } else {
            sourcesEl.innerHTML = '';
            relatedEl.innerHTML = '';
            relEl.innerHTML = '';
        }
    }

    function buildBreadcrumb(node) {
        if (!node) return '';

        if (node.level === 3) {
            return `Level 3 · Coarse category · ${node.name}`;
        }

        const clusters = fullData.clusters || {};
        const l3 = clusters.level_3 || {};
        const l2 = clusters.level_2 || {};
        const l1 = clusters.level_1 || {};

        let l3Node, l2Node, l1Node;

        if (node.level === 2) {
            l2Node = l2[node.id];
            if (l2Node && l2Node.parent) l3Node = l3[l2Node.parent];
        } else if (node.level === 1) {
            l1Node = l1[node.id];
            if (l1Node && l1Node.parent) {
                l2Node = l2[l1Node.parent];
                if (l2Node && l2Node.parent) l3Node = l3[l2Node.parent];
            }
        } else if (node.level === 0) {
            const path = entityClusterPath[node.id];
            if (path) {
                if (path.level1) l1Node = l1[path.level1];
                if (path.level2) l2Node = l2[path.level2];
                if (path.level3) l3Node = l3[path.level3];
            }
        }

        const parts = [];
        if (l3Node) parts.push(`Level 3 · ${l3Node.name || l3Node.id}`);
        if (l2Node) parts.push(`Level 2 · ${l2Node.name || l2Node.id}`);
        if (l1Node) parts.push(`Level 1 · ${l1Node.name || l1Node.id}`);
        if (node.level === 0) parts.push(`Entity · ${node.name}`);

        return parts.join(' › ') || `Level ${node.level}`;
    }

    function buildEpisodeDeepLink(ent, episodeNumber) {
        if (!ent || !ent.metadata || !ent.metadata.chunks || !chunkTimestamps) return null;
        const epPrefix = `ep${episodeNumber}_chunk`;
        const chunkId = ent.metadata.chunks.find(c => c.startsWith(epPrefix));
        if (!chunkId) return null;
        const chunkInfo = chunkTimestamps[chunkId];
        if (!chunkInfo) return null;
        const ts = chunkInfo.start_formatted || '0:00';
        return {
            href: `/PodcastMap3D.html#episode=${chunkInfo.episode}&t=${encodeURIComponent(ts)}`,
            timestamp: ts,
            title: `${chunkId} · Episode ${chunkInfo.episode} at ${ts}`
        };
    }

    function buildChunkDeepLinks(ent) {
        if (!ent || !ent.metadata || !ent.metadata.chunks || !chunkTimestamps) return '';
        const chunks = ent.metadata.chunks;
        const links = chunks.slice(0, 6).map(cid => {
            const info = chunkTimestamps[cid];
            if (!info) {
                return `<span style="color:#7f9bbf; margin-right:4px; font-size:0.78rem;">${cid}</span>`;
            }
            const ts = info.start_formatted || '0:00';
            return `<a href="/PodcastMap3D.html#episode=${info.episode}&t=${encodeURIComponent(ts)}"
                       target="_blank"
                       class="kg-episode-pill"
                       style="background:rgba(80,230,140,0.18);border-color:rgba(80,230,140,0.85);"
                       title="${cid} · Episode ${info.episode} at ${ts}">
                       Ep ${info.episode} @ ${ts}
                    </a>`;
        }).join('');

        if (!links) return '';
        const extra = chunks.length > 6 ? ` <span style="color:#9ecfff;font-size:0.78rem;">(+${chunks.length - 6} more)</span>` : '';
        return links + extra;
    }

    function buildRelationshipInfo(entityId) {
        const rels = fullData.relationships || [];
        const entities = fullData.entities || {};
        const outgoing = [];
        const incoming = [];

        rels.forEach(rel => {
            const s = rel.source;
            const t = rel.target;
            const type = rel.type || rel.predicate || rel.relationship_type || 'RELATED_TO';
            if (s === entityId) {
                const targetEnt = entities[t];
                if (targetEnt) {
                    outgoing.push({
                        type,
                        targetName: targetEnt.name || t
                    });
                }
            } else if (t === entityId) {
                const srcEnt = entities[s];
                if (srcEnt) {
                    incoming.push({
                        type,
                        sourceName: srcEnt.name || s
                    });
                }
            }
        });

        let relationshipHtml = '';
        let relatedHtml = '';

        if (outgoing.length || incoming.length) {
            relationshipHtml = '<div><strong>Relationships:</strong></div>';

            if (outgoing.length) {
                relationshipHtml += '<div style="margin-top:4px;"><em>Outgoing</em></div>';
                relationshipHtml += outgoing.slice(0, 12).map(r => `
                    <div class="kg-relationship-item">
                        <span class="kg-rel-type">${r.type}</span>
                        <span>→ ${r.targetName}</span>
                    </div>
                `).join('');
                if (outgoing.length > 12) {
                    relationshipHtml += `<div style="font-size:0.75rem;color:#9ecfff;">+${outgoing.length - 12} more outgoing</div>`;
                }
            }

            if (incoming.length) {
                relationshipHtml += '<div style="margin-top:4px;"><em>Incoming</em></div>';
                relationshipHtml += incoming.slice(0, 12).map(r => `
                    <div class="kg-relationship-item">
                        <span class="kg-rel-type">${r.type}</span>
                        <span>${r.sourceName} →</span>
                    </div>
                `).join('');
                if (incoming.length > 12) {
                    relationshipHtml += `<div style="font-size:0.75rem;color:#9ecfff;">+${incoming.length - 12} more incoming</div>`;
                }
            }

            const relatedNames = new Set();
            outgoing.forEach(r => relatedNames.add(r.targetName));
            incoming.forEach(r => relatedNames.add(r.sourceName));

            if (relatedNames.size) {
                relatedHtml = `<div><strong>Connected entities:</strong></div><div>${Array.from(relatedNames).slice(0, 20).join(', ')}${relatedNames.size > 20 ? ` … (+${relatedNames.size - 20} more)` : ''}</div>`;
            }
        }

        return { relationshipHtml, relatedHtml };
    }

    function getLevelName(level) {
        switch (level) {
            case 0: return 'Entity';
            case 1: return 'Fine cluster';
            case 2: return 'Medium cluster';
            case 3: return 'Coarse category';
            default: return `Level ${level}`;
        }
    }

    function searchEntities(query) {
        const resultsContainer = document.getElementById('kg-search-results');
        if (!resultsContainer || !fullData || !fullData.entities) return;

        const q = query.toLowerCase();
        const results = [];

        Object.entries(fullData.entities).forEach(([id, ent]) => {
            const name = (ent.name || id).toLowerCase();
            if (name.includes(q)) {
                results.push({
                    id,
                    name: ent.name || id,
                    type: ent.type || 'CONCEPT',
                    score: name === q ? 2 : name.startsWith(q) ? 1.5 : 1
                });
            }
        });

        results.sort((a, b) => b.score - a.score);
        const top = results.slice(0, 20);

        if (!top.length) {
            resultsContainer.style.display = 'none';
            return;
        }

        resultsContainer.innerHTML = top.map(r => `
            <div class="kg-search-result" data-id="${r.id}">
                <span class="entity-type-pill">${r.type}</span>
                <span>${r.name}</span>
            </div>
        `).join('');

        resultsContainer.style.display = 'block';

        resultsContainer.querySelectorAll('.kg-search-result').forEach(el => {
            el.addEventListener('click', () => {
                const id = el.dataset.id;
                navigateToEntity(id);
                resultsContainer.style.display = 'none';
            });
        });
    }

    function navigateToEntity(entityId) {
        if (!fullData || !fullData.clusters || !fullData.clusters.level_1) return;

        pinnedEntityId = entityId;

        const pathIds = [];
        const clusters = fullData.clusters;
        const l1 = clusters.level_1;
        const l2 = clusters.level_2 || {};
        const l3 = clusters.level_3 || {};

        let l1Id;
        Object.entries(l1).some(([id, cluster]) => {
            if (Array.isArray(cluster.entities) && cluster.entities.includes(entityId)) {
                l1Id = id;
                return true;
            }
            return false;
        });
        if (!l1Id) return;

        const l1Cluster = l1[l1Id];
        pathIds.unshift(l1Id);

        const l2Id = l1Cluster.parent;
        if (l2Id && l2[l2Id]) {
            pathIds.unshift(l2Id);
            const l2Cluster = l2[l2Id];
            const l3Id = l2Cluster.parent;
            if (l3Id && l3[l3Id]) {
                pathIds.unshift(l3Id);
            }
        }

        // Ensure all levels visible for navigation
        [2,1,0].forEach(lvl => { visibleLevels.add(lvl); const cb=document.getElementById(`kg-level-${lvl}`); if(cb) cb.checked=true; });
        pathIds.forEach(cid => expandedNodes.add(cid));

        const data = buildExpandedGraph();
        updateGraph(data);

        setTimeout(() => {
            const node = data.nodes.find(n => n.id === entityId);
            if (node) {
                handleNodeClick(node);
                focusCameraOnNode(node, 150);
            }
        }, 450);
    }

    function exportSubgraph() {
        if (!graph) return;
        const data = graph.graphData();
        const payload = {
            nodes: (data.nodes || []).map(n => ({
                id: n.id,
                name: n.name,
                type: n.type,
                level: n.level
            })),
            links: (data.links || []).map(l => ({
                source: typeof l.source === 'object' ? l.source.id : l.source,
                target: typeof l.target === 'object' ? l.target.id : l.target,
                type: l.type
            }))
        };

        const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'yonearth_kg_subgraph.json';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    function showLoading(show) {
        const container = document.getElementById('kg-3d-graph');
        if (!container) return;
        if (show) {
            container.innerHTML = `
                <div style="color:#00ffff;padding:24px;text-align:center;">
                    <h3>Loading YonEarth Knowledge Graph…</h3>
                    <p>Processing 39,046 entities and 43,297 relationships</p>
                    <div style="margin:16px auto;width:40px;height:40px;border-radius:50%;border:3px solid rgba(0,255,255,0.25);border-top-color:#00ffff;animation:kg-spin 1s linear infinite;"></div>
                    <style>
                        @keyframes kg-spin{0%{transform:rotate(0deg);}100%{transform:rotate(360deg);}}
                    </style>
                </div>
            `;
        }
    }

    function showError(message) {
        const container = document.getElementById('kg-3d-graph');
        if (!container) return;
        container.innerHTML = `
            <div style="color:#ff6b6b;padding:24px;text-align:center;">
                <h3>Error loading knowledge graph</h3>
                <p>${message}</p>
            </div>
        `;
    }

    function switchViewMode(mode) {
        // Toggle UI controls
        const hierarchyControls = document.getElementById('kg-hierarchy-controls');
        const membraneControls = document.getElementById('kg-membrane-controls');
        const membraneToggles = document.getElementById('kg-membrane-toggles');

        if (mode === 'holographic') {
            if (hierarchyControls) hierarchyControls.style.display = '';
            if (membraneControls) membraneControls.style.display = 'none';
            if (membraneToggles) membraneToggles.style.display = 'none';
            showTopLevelClusters();
            removeClusterMembranes();
        } else if (mode === 'full-force') {
            if (hierarchyControls) hierarchyControls.style.display = 'none';
            if (membraneControls) membraneControls.style.display = '';
            if (membraneToggles) membraneToggles.style.display = '';
            showFullForceView();
            createClusterMembranes();
        }

        resetCamera();
    }

    function showFullForceView() {
        if (!fullData || !graph) return;

        const entities = fullData.entities || {};
        const relationships = fullData.relationships || [];
        const positions = fullData.positions || {};

        // Build all nodes (entities only, no hierarchy clusters)
        const nodes = [];
        const links = [];

        // Add all entities that pass filters
        Object.entries(entities).forEach(([id, ent]) => {
            // Apply filters
            if (!activeEntityTypes.has(ent.type)) return;
            if (activeEpisodeFilter && !entityMatchesEpisode(ent, activeEpisodeFilter)) return;
            if (activeBookFilter && !entityMatchesBook(ent, activeBookFilter)) return;

            const degree = ent.degree || 0;

            // Filter out isolated nodes by default
            if (degree === 0) return;

            // Use embedding positions as starting point to avoid hairball!
            // SCALE UP positions by 10x to REALLY spread the graph out!
            const pos = positions[id];
            const SCALE_FACTOR = 10.0;  // Much larger scale!
            const initialPos = pos && Array.isArray(pos) && pos.length === 3 ? {
                x: pos[0] * SCALE_FACTOR,
                y: pos[1] * SCALE_FACTOR,
                z: pos[2] * SCALE_FACTOR
            } : null;

            // Scale node size based on connectivity
            const nodeSize = getNodeSize(ent);

            nodes.push({
                id: id,
                name: ent.name || id,
                type: ent.type || 'CONCEPT',
                description: ent.description,
                level: 0,
                color: CONFIG.nodeColors[ent.type] || CONFIG.nodeColors.CONCEPT,
                size: nodeSize,
                opacity: 0.92,
                degree: degree,  // Store for reference
                // Use scaled embedding positions as starting point
                ...(initialPos || {})
            });
        });

        // Build visible entity IDs set
        const visibleIds = new Set(nodes.map(n => n.id));

        // Add relationship links between visible entities
        relationships.forEach(rel => {
            const s = rel.source;
            const t = rel.target;
            const type = rel.type || rel.predicate || rel.relationship_type || 'RELATED_TO';

            if (!visibleIds.has(s) || !visibleIds.has(t)) return;
            if (!showRelationshipEdges) return;
            if (!activeRelationshipTypes.has(type)) return;

            links.push({
                source: s,
                target: t,
                type: type,
                color: CONFIG.linkColors.relationship,
                width: 0.1,  // Very thin edges!
                opacity: 0.05  // Very faint to reduce clutter
            });
        });

        console.log(`[Full Force] Showing ${nodes.length} entities, ${links.length} relationships`);

        graph.graphData({ nodes, links });

        // Configure force simulation for full-force mode
        // Use WEAK forces since we have good initial positions from embeddings
        graph.cooldownTime(3000);  // Moderate cooldown
        graph.d3Force('center', null);  // No centering
        graph.d3Force('charge').strength(-30);  // Weak repulsion (was -80)
        graph.d3Force('link').distance(30).strength(0.1);  // Very weak link force (was 0.5)
        graph.d3AlphaDecay(0.02);  // Faster settling (was 0.01)
        graph.d3VelocityDecay(0.6);  // High damping for stability (was 0.4)

        // Remove floor planes if they exist
        removeFloorPlanes();
    }

    function createClusterMembranes() {
        if (!fullData || !graph || !window.THREE) return;

        removeClusterMembranes();  // Clear existing membranes

        const scene = graph.scene();
        if (!scene) return;

        const clusters = fullData.clusters || {};
        const positions = fullData.positions || {};

        // Handle both array and object formats for clusters
        let clustersObj = clusters;
        if (Array.isArray(clusters)) {
            // Convert array format to object format
            clustersObj = {};
            clusters.forEach(level => {
                if (level && typeof level === 'object') {
                    Object.assign(clustersObj, level);
                }
            });
        }

        // Check which membrane levels are enabled
        const l1Enabled = document.getElementById('kg-membrane-l1')?.checked;
        const l2Enabled = document.getElementById('kg-membrane-l2')?.checked;
        const l3Enabled = document.getElementById('kg-membrane-l3')?.checked;

        // Create membranes for each level (similar to GraphRAG3D_EmbeddingView.js)
        if (l3Enabled) createMembranesForLevel(scene, clustersObj.level_3 || {}, positions, 3, 0xFF4444); // Red
        if (l2Enabled) createMembranesForLevel(scene, clustersObj.level_2 || {}, positions, 2, 0xFFCC00); // Gold
        if (l1Enabled) createMembranesForLevel(scene, clustersObj.level_1 || {}, positions, 1, 0x00CCFF); // Cyan

        console.log(`[Membranes] Created ${clusterMembranes.length} membranes`);
    }

    function createMembranesForLevel(scene, levelClusters, positions, level, colorHex) {
        Object.entries(levelClusters).forEach(([clusterId, cluster]) => {
            const memberPositions = [];

            // Get entity positions for this cluster
            const entityIds = cluster.entities || [];
            entityIds.forEach(entityId => {
                const pos = positions[entityId];
                if (pos && Array.isArray(pos) && pos.length === 3) {
                    memberPositions.push(pos);
                }
            });

            if (memberPositions.length < 3) return;  // Need at least 3 points

            // Fit ellipsoid to positions
            const ellipsoid = fitEllipsoid(memberPositions);

            // Create Fresnel membrane
            const membrane = createFresnelMembrane(ellipsoid, level, colorHex);
            membrane.userData = {
                clusterId: clusterId,
                level: level,
                type: 'membrane'
            };

            scene.add(membrane);
            clusterMembranes.push(membrane);
        });
    }

    function fitEllipsoid(positions) {
        const EPSILON = 5.0;

        // Compute centroid
        const centroid = [0, 0, 0];
        for (const pos of positions) {
            centroid[0] += pos[0];
            centroid[1] += pos[1];
            centroid[2] += pos[2];
        }
        centroid[0] /= positions.length;
        centroid[1] /= positions.length;
        centroid[2] /= positions.length;

        // Compute variances
        let varX = 0, varY = 0, varZ = 0;
        for (const pos of positions) {
            const dx = pos[0] - centroid[0];
            const dy = pos[1] - centroid[1];
            const dz = pos[2] - centroid[2];
            varX += dx * dx;
            varY += dy * dy;
            varZ += dz * dz;
        }

        // Compute radii using standard deviation
        const radii = [
            Math.max(Math.sqrt(varX / positions.length) * 1.5, EPSILON),
            Math.max(Math.sqrt(varY / positions.length) * 1.5, EPSILON),
            Math.max(Math.sqrt(varZ / positions.length) * 1.5, EPSILON)
        ];

        return { center: centroid, radii: radii };
    }

    function createFresnelMembrane(ellipsoid, level, colorHex) {
        const geometry = new THREE.SphereGeometry(1, 32, 32);
        geometry.scale(ellipsoid.radii[0], ellipsoid.radii[1], ellipsoid.radii[2]);

        // Fresnel shader material for cell membrane effect
        const material = new THREE.ShaderMaterial({
            uniforms: {
                baseOpacity: { value: 0.0 },
                edgeOpacity: { value: 0.15 },
                color: { value: new THREE.Color(colorHex) }
            },
            vertexShader: `
                varying vec3 vNormal;
                varying vec3 vViewPosition;
                void main() {
                    vNormal = normalize(normalMatrix * normal);
                    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
                    vViewPosition = -mvPosition.xyz;
                    gl_Position = projectionMatrix * mvPosition;
                }
            `,
            fragmentShader: `
                uniform float baseOpacity;
                uniform float edgeOpacity;
                uniform vec3 color;
                varying vec3 vNormal;
                varying vec3 vViewPosition;
                void main() {
                    vec3 viewDir = normalize(vViewPosition);
                    float fresnel = 1.0 - abs(dot(viewDir, vNormal));
                    float alpha = baseOpacity + edgeOpacity * pow(fresnel, 2.5);
                    gl_FragColor = vec4(color, alpha);
                }
            `,
            transparent: true,
            side: THREE.DoubleSide,
            depthWrite: false,
            blending: THREE.AdditiveBlending
        });

        const mesh = new THREE.Mesh(geometry, material);
        mesh.position.set(ellipsoid.center[0], ellipsoid.center[1], ellipsoid.center[2]);

        return mesh;
    }

    function removeClusterMembranes() {
        if (!graph || !window.THREE) return;

        const scene = graph.scene();
        if (!scene) return;

        clusterMembranes.forEach(membrane => {
            scene.remove(membrane);
            if (membrane.geometry) membrane.geometry.dispose();
            if (membrane.material) membrane.material.dispose();
        });

        clusterMembranes = [];
    }

    function updateMembraneVisibility() {
        // Remove all and recreate based on current settings
        createClusterMembranes();
    }

    function computeConnectivityStats() {
        const entities = fullData.entities || {};
        const relationships = fullData.relationships || [];

        // Initialize stats for all entities
        Object.keys(entities).forEach(id => {
            entities[id].degree = 0;
            entities[id].weightedDegree = 0;
        });

        // Count connections
        relationships.forEach(rel => {
            const s = rel.source;
            const t = rel.target;

            if (entities[s]) {
                entities[s].degree = (entities[s].degree || 0) + 1;
                entities[s].weightedDegree = (entities[s].weightedDegree || 0) + (rel.weight || 1);
            }
            if (entities[t]) {
                entities[t].degree = (entities[t].degree || 0) + 1;
                entities[t].weightedDegree = (entities[t].weightedDegree || 0) + (rel.weight || 1);
            }
        });

        // Compute min/max for normalization
        let minDegree = Infinity, maxDegree = 0;
        let minWeighted = Infinity, maxWeighted = 0;
        let minBetweenness = Infinity, maxBetweenness = 0;

        Object.values(entities).forEach(ent => {
            const deg = ent.degree || 0;
            const wdeg = ent.weightedDegree || 0;
            const btw = ent.betweenness || 0;

            minDegree = Math.min(minDegree, deg);
            maxDegree = Math.max(maxDegree, deg);
            minWeighted = Math.min(minWeighted, wdeg);
            maxWeighted = Math.max(maxWeighted, wdeg);
            minBetweenness = Math.min(minBetweenness, btw);
            maxBetweenness = Math.max(maxBetweenness, btw);
        });

        connectivityStats = {
            minDegree: isFinite(minDegree) ? minDegree : 0,
            maxDegree,
            minWeighted: isFinite(minWeighted) ? minWeighted : 0,
            maxWeighted,
            minBetweenness: isFinite(minBetweenness) ? minBetweenness : 0,
            maxBetweenness
        };

        console.log('[KG] Computed connectivity stats:', connectivityStats);
    }

    function getNodeSize(entity) {
        const MIN_SIZE = 0.3;
        const MAX_SIZE = 8.0;  // Much larger max for high-degree nodes

        if (nodeSizeMode === 'betweenness') {
            const btw = entity.betweenness || 0;
            if (btw === 0) return MIN_SIZE;

            // Normalize betweenness to [0, 1]
            const normalized = (btw - connectivityStats.minBetweenness) /
                             (connectivityStats.maxBetweenness - connectivityStats.minBetweenness + 0.0001);

            // Power scale for dramatic contrast
            return MIN_SIZE + (MAX_SIZE - MIN_SIZE) * Math.pow(normalized, 0.5);
        } else {
            // Connectivity mode
            const degree = entity.degree || 0;
            if (degree === 0) return MIN_SIZE;

            // Normalize degree to [0, 1]
            const normalized = (degree - connectivityStats.minDegree) /
                             (connectivityStats.maxDegree - connectivityStats.minDegree + 0.0001);

            // Power scale for dramatic contrast - high-degree nodes are MUCH larger
            return MIN_SIZE + (MAX_SIZE - MIN_SIZE) * Math.pow(normalized, 0.4);
        }
    }

    function bookDisplayName(key) {
        switch (key) {
            case 'veriditas':
            case 'viriditas':
                return 'VIRIDITAS';
            case 'y-on-earth':
                return 'Y on Earth';
            case 'soil-stewardship-handbook':
                return 'Soil Stewardship Handbook';
            case 'our-biggest-deal':
                return 'Our Biggest Deal';
            default:
                return key;
        }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
