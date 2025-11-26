/**
 * GraphRAG 3D Embedding View
 *
 * Dual-mode visualization:
 * 1. Fixed Embedding View - Shows global semantic structure using UMAP positions
 * 2. Force-Directed View - Local graph exploration with physics simulation
 *
 * Features:
 * - Graph-enriched UMAP 3D positioning
 * - Fresnel shader cluster membranes (cellular aesthetic)
 * - Context lens hover preview (non-destructive)
 * - Betweenness centrality coloring
 * - GPU-accelerated rendering with instancing
 */

class GraphRAG3DEmbeddingView {
    constructor(containerId) {
        this.containerId = containerId;
        this.container = document.getElementById(containerId);

        // Visualization state
        this.mode = 'semantic'; // semantic, contextual, structural
        this.data = null;
        this.graph = null;
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.currentLayout = 'semantic';
        this.transitionDuration = 1500;
        this.activeTransitions = [];

        // Data structures
        this.entities = new Map();
        this.relationships = [];
        this.clusters = {
            level_0: [], // Individual entities
            level_1: [], // Fine clusters (300)
            level_2: [], // Medium clusters (30)
            level_3: []  // Coarse clusters (7)
        };

        // Visual elements
        this.entityMeshes = [];
        this.clusterMeshes = [];
        this.connectionLines = [];
        this.entityMeshMap = new Map();
        this.displayedEntityIds = new Set();

        // Color modes
        this.colorMode = 'type'; // 'type' or 'centrality'
        this.centralityColors = {
            low: 0x2196F3,
            mid: 0xFFC107,
            high: 0xF44336
        };
        this.sizeMode = 'betweenness'; // default to betweenness; other option: 'connectivity'
        this.edgeStyles = {
            defaultColor: 0x6a7fa0,
            attributionColor: 0xFF6B9D,
            minWidth: 0.25,
            maxWidth: 2.0
        };

        // Scene bounds
        this.boundingRadius = 150;

        // Connectivity stats
        this.connectivityStats = {
            minDegree: 0,
            maxDegree: 0,
            minWeighted: 0,
            maxWeighted: 0,
            minBetweenness: 0,
            maxBetweenness: 0
        };

        // Selection state
        this.selectedEntity = null;
        this.hoveredEntity = null;
        this.visibleEntities = new Set();
        this.labelSprites = [];
        this.clusterMeshes = [];
        this.connectionLines = [];
        this.clusterLabelSprites = [];

        // Layout datasets
        this.graphsageLayout = {};
        this.hasContextualLayout = false;

        // Community ID mapping (robust labels from community_summaries.json)
        this.communityIdMapping = {};

        // Force simulation state
        this.forceSimulation = null;
        this.forceLinkForce = null;
        this.forceNodes = [];
        this.forceLinks = [];
        this.d3Force = null;

        // Cluster maintenance
        this.clusterLookup = {
            level_1: new Map(),
            level_2: new Map(),
            level_3: new Map()
        };
        this.pendingMembraneUpdate = false;
        this.membraneRefreshTimeout = null;

        // Performance monitoring
        this.frameCount = 0;
        this.lastFPSUpdate = Date.now();

        // Entity type colors
        this.typeColors = {
            'PERSON': 0x4CAF50,
            'ORGANIZATION': 0x2196F3,
            'CONCEPT': 0x9C27B0,
            'PRACTICE': 0xFF9800,
            'PRODUCT': 0xF44336,
            'PLACE': 0x00BCD4,
            'EVENT': 0xFFEB3B,
            'WORK': 0x795548,
            'CLAIM': 0xE91E63
        };

        // Track which hierarchy key names were detected (Leiden vs. legacy)
        this.hierarchyKeyMap = {
            L0: null,
            L1: null,
            L2: null,
            L3: null
        };

        // Entity type visibility
        this.typeVisibility = {};
        Object.keys(this.typeColors).forEach(type => {
            this.typeVisibility[type] = true;
        });

        // Node size mode options
        this.sizeModes = ['connectivity', 'betweenness'];

        // Mobile safety flags
        this.isMobile = typeof window !== 'undefined' && window.innerWidth < 768;
        this.mobileNodeLimit = 2000;

        // Determine base path for static assets (supports /YonEarth/graph deployment)
        this.assetBasePath = '';
        if (typeof window !== 'undefined' && window.location && window.location.pathname) {
            const pathName = window.location.pathname;
            const marker = '/graph/';
            const idx = pathName.indexOf(marker);
            if (idx !== -1) {
                this.assetBasePath = pathName.substring(0, idx + marker.length - 1);
            }
        }

        // Hierarchy + summaries for 2D organic modes
        this.leidenCommunities = null;
        this.communitySummariesRaw = null;
        this.communitySummaryLookup = {};
        this.hierarchyByLevel = { 0: [], 1: [], 2: [], 3: [] };
        this.hierarchicalData = null;
        this.entityLeafLimit = 20;
        this.circlePackState = {
            svg: null,
            group: null,
            zoom: null,
            width: typeof window !== 'undefined' ? window.innerWidth : 1920,
            height: typeof window !== 'undefined' ? window.innerHeight : 1080
        };
        this.current2DMode = null;
        this.organicTooltip = (typeof d3 !== 'undefined') ? d3.select('#organic-tooltip') : null;
        this.infoPanelMode = 'entity';
        this.webglWarningShown = false;
    }

    resolveAssetPath(subpath = '') {
        const normalized = subpath.startsWith('/') ? subpath : `/${subpath}`;
        return `${this.assetBasePath || ''}${normalized}`;
    }

    async fetchJsonWithFallback(paths, label = 'resource') {
        const urls = Array.isArray(paths) ? paths : [paths];
        let lastError = null;
        // Cache-bust version - increment when deploying new data
        const cacheBuster = 'v=20251126b';

        for (const url of urls) {
            try {
                const bustUrl = url + (url.includes('?') ? '&' : '?') + cacheBuster;
                const response = await fetch(bustUrl);
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}`);
                }
                const text = await response.text();
                try {
                    return JSON.parse(text);
                } catch (parseErr) {
                    const snippet = text.trim().slice(0, 120);
                    const hint = snippet.startsWith('<') ? 'Received HTML instead of JSON' : parseErr.message;
                    throw new Error(hint);
                }
            } catch (err) {
                lastError = err;
                console.warn(`Failed to load ${label} from ${url}: ${err.message}`);
            }
        }

        throw new Error(`${label} load failed (${lastError?.message || 'unknown error'})`);
    }

    /**
     * Initialize the viewer
     */
    async init() {
        this.updateLoadingStatus('Loading knowledge graph data...');

        try {
            // Load data
            await this.loadData();

            // Setup Three.js scene
            this.updateLoadingStatus('Setting up 3D scene...');
            this.setupScene();

            // Create visualizations
            this.updateLoadingStatus('Creating entity nodes...');
            this.createEntityNodes();

        this.updateLoadingStatus('Drawing relationships...');
        this.createRelationshipLines();

        this.updateLoadingStatus('Creating cluster membranes...');
        this.createClusterMembranes();
        this.updateTopLabels();
        this.createClusterLabels();

        // Setup controls and interactions
        this.setupControls();
        this.setupKeyboardShortcuts();
        this.setupSearch();
        this.setupClusterSearch();
        this.hideEntityInfo();
        this.centerOnEntityName('Y on Earth Community');

        // Start rendering
        this.hideLoadingScreen();
        this.hideWebGLError();
        this.animate();

            console.log('✅ GraphRAG 3D Embedding View initialized');
        } catch (error) {
            console.error('❌ Failed to initialize viewer:', error);
            this.updateLoadingStatus(`Error: ${error.message}`);
            this.handleInitializationError(error);
        }
    }

    /**
     * Load graphrag_hierarchy.json data
     */
    async loadData() {
        try {
            const [
                hierarchy,
                sageLayout,
                idMapping,
                leidenData,
                summaryData
            ] = await Promise.all([
                this.fetchHierarchyData(),
                this.fetchGraphSageLayout(),
                this.fetchCommunityIdMapping(),
                this.fetchLeidenCommunities(),
                this.fetchCommunitySummaries()
            ]);

            this.data = hierarchy;
            this.graphsageLayout = sageLayout || {};
            this.communityIdMapping = idMapping || {};
            this.leidenCommunities = leidenData || null;
            this.communitySummariesRaw = summaryData || {};
            this.communitySummaryLookup = this.normalizeCommunitySummaries(summaryData);

            // Process data
            this.processData();

            console.log(`Loaded ${this.entities.size} entities, ${this.relationships.length} relationships`);
            console.log(`Community ID mapping loaded: ${Object.keys(this.communityIdMapping).length} titles`);
            if (this.leidenCommunities?.communities) {
                console.log(`Leiden communities loaded: ${this.leidenCommunities.communities.length} entries`);
            }
            if (Object.keys(this.communitySummaryLookup).length) {
                console.log(`Community summaries available: ${Object.keys(this.communitySummaryLookup).length}`);
            } else {
                console.warn('Community summaries not found; cluster descriptions will fall back to titles.');
            }
            if (Object.keys(this.graphsageLayout).length) {
                console.log(`GraphSAGE layout loaded for ${Object.keys(this.graphsageLayout).length} nodes`);
                this.hasContextualLayout = true;
            } else {
                console.warn('GraphSAGE layout not available; contextual mode disabled until data is deployed.');
                this.hasContextualLayout = false;
            }
            this.updateModeAvailability();
        } catch (error) {
            throw new Error(`Data loading failed: ${error.message}`);
        }
    }

    async fetchHierarchyData() {
        return this.fetchJsonWithFallback([
            this.resolveAssetPath('data/graphrag_hierarchy/graphrag_hierarchy_v2.json'),
            this.resolveAssetPath('data/graphrag_hierarchy/graphrag_hierarchy.json'),
            this.resolveAssetPath('data/graphrag_hierarchy/graphrag_hierarchy_test_sample.json')
        ], 'hierarchy');
    }

    async fetchGraphSageLayout() {
        try {
            return await this.fetchJsonWithFallback([
                this.resolveAssetPath('data/graphrag_hierarchy/graphsage_layout.json')
            ], 'GraphSAGE layout');
        } catch (err) {
            console.warn('GraphSAGE layout unavailable, contextual mode will reuse semantic coordinates.', err);
            return {};
        }
    }

    async fetchCommunityIdMapping() {
        return this.fetchJsonWithFallback([
            this.resolveAssetPath('data/community_id_mapping.json'),
            this.resolveAssetPath('data/graphrag_hierarchy/community_id_mapping.json')
        ], 'community ID mapping').catch(err => {
            console.warn('Community ID mapping unavailable, will use fallback labels.', err);
            return {};
        });
    }

    async fetchLeidenCommunities() {
        return this.fetchJsonWithFallback([
            this.resolveAssetPath('data/leiden_communities.json'),
            this.resolveAssetPath('data/graphrag_hierarchy/checkpoints_microsoft/leiden_communities.json')
        ], 'Leiden communities').catch(err => {
            console.warn('Leiden communities unavailable, organic views disabled until data is deployed.', err);
            return null;
        });
    }

    async fetchCommunitySummaries() {
        return this.fetchJsonWithFallback([
            this.resolveAssetPath('data/summaries_progress.json'),
            this.resolveAssetPath('data/graphrag_hierarchy/checkpoints/summaries_progress.json')
        ], 'community summaries').catch(err => {
            console.warn('Community summaries unavailable; clusters will show titles only.', err);
            return {};
        });
    }

    normalizeCommunitySummaries(raw) {
        const lookup = {};
        if (!raw) return lookup;
        Object.entries(raw).forEach(([levelKey, entries]) => {
            Object.entries(entries || {}).forEach(([key, value]) => {
                lookup[key] = value;
            });
        });
        return lookup;
    }

    /**
     * Get community title from robust ID mapping
     * Extracts numeric ID from cluster ID (e.g., "c66" → 66 → "Brigitte Mars Natural Healing")
     */
    getCommunityTitle(clusterId) {
        if (!this.communityIdMapping || Object.keys(this.communityIdMapping).length === 0) {
            return null;
        }

        // Extract numeric ID from cluster ID
        // Handles: "c66", "66", "l1_66", etc.
        const match = clusterId.match(/(\d+)/);
        if (!match) return null;

        const numericId = match[1];

        // Direct lookup in mapping
        return this.communityIdMapping[numericId] || null;
    }

    /**
     * Process loaded data into usable structures
     */
    processData() {
        // Handle test data format
        if (this.data.test_mode) {
            console.log('Using test data format');
            for (const [entityId, entityData] of Object.entries(this.data.entities)) {
                this.entities.set(entityId, {
                    id: entityId,
                    type: entityData.type,
                    description: entityData.description || '',
                    position: entityData.umap_position || [0, 0, 0],
                    betweenness: entityData.betweenness || 0,
                    relationshipStrengths: entityData.relationship_strengths || {}
                });
            }
            this.relationships = this.data.relationships || [];
            this.computeConnectivityStats();
            document.getElementById('total-count').textContent = this.entities.size;
            this.updateHierarchyLabels();
            return;
        }

        // Full data format
        const clusters = this.data.clusters || {};
        const level0 = clusters.level_0 || clusters.L0 || {};

        this.hierarchyKeyMap = {
            L0: clusters.L0 ? 'L0' : (clusters.level_0 ? 'level_0' : null),
            L1: clusters.L1 ? 'L1' : (clusters.level_1 ? 'level_1' : null),
            L2: clusters.L2 ? 'L2' : (clusters.level_2 ? 'level_2' : null),
            L3: clusters.L3 ? 'L3' : (clusters.level_3 ? 'level_3' : null)
        };

        for (const [clusterId, cluster] of Object.entries(level0)) {
            let entityId = cluster.entity_id || cluster.entity || cluster.id || clusterId;
            let entityData = null;

            if (typeof entityId === 'object') {
                entityData = entityId;
                entityId = cluster.id || clusterId;
            }

            if (!entityData && typeof cluster.entity === 'object') {
                entityData = cluster.entity;
            }

            if (!entityData && this.data.entities) {
                entityData = this.data.entities[entityId];
            }

            if (!entityData) continue;

            // Use UMAP position if available, fallback to PCA
            const position = cluster.umap_position || cluster.position || entityData.umap_position || [0, 0, 0];
            const betweenness = cluster.betweenness ?? entityData.betweenness ?? 0;
            const relationshipStrengths =
                cluster.relationship_strengths ||
                cluster.relationshipStrengths ||
                entityData.relationship_strengths ||
                {};
            const sageCoords = this.graphsageLayout[entityId];
            const basePosition = Array.isArray(position) ? [...position] : [0, 0, 0];
            const contextualPosition = Array.isArray(sageCoords) ? [...sageCoords] : null;

            this.entities.set(entityId, {
                id: entityId,
                type: entityData.type,
                description: entityData.description || '',
                sources: entityData.sources || [],
                position: [...basePosition],
                rawUmapPosition: [...basePosition],
                rawSagePosition: contextualPosition ? [...contextualPosition] : null,
                umapPosition: null,
                sagePosition: contextualPosition ? [...contextualPosition] : null,
                betweenness: betweenness,
                relationshipStrengths: relationshipStrengths,
                clusterId: clusterId
            });
        }

        // Load relationships
        this.relationships = this.data.relationships || [];
        this.computeConnectivityStats();

        // Process cluster hierarchies
        const hierarchyLevels = [
            { key: 'level_1', alt: 'L1' },
            { key: 'level_2', alt: 'L2' },
            { key: 'level_3', alt: 'L3' }
        ];

        for (const { key, alt } of hierarchyLevels) {
            const levelClusters = clusters[key] || clusters[alt];

            if (!levelClusters) continue;

            for (const [clusterId, cluster] of Object.entries(levelClusters)) {
                const center = cluster.center || cluster.position || [0, 0, 0];

                // Use robust ID mapping (community_id_mapping.json) for accurate titles
                const robustTitle = this.getCommunityTitle(clusterId);

                const entry = {
                    id: clusterId,
                    name: robustTitle || cluster.name || cluster.title || clusterId,
                    title: robustTitle || cluster.title || cluster.name || clusterId,
                    center: [...center],
                    rawCenter: [...center],
                    children: cluster.children || [],
                    entities: cluster.entities || [],
                    node_count: (cluster.entities || []).length
                };
                this.clusters[key].push(entry);
                if (this.clusterLookup[key]) {
                    this.clusterLookup[key].set(clusterId, entry);
                }
            }
        }

        // Normalize positions to a consistent scale centered at origin
        this.hasContextualLayout = Array.from(this.entities.values()).some(e => Array.isArray(e.rawSagePosition));
        this.normalizePositions();

        document.getElementById('total-count').textContent = this.entities.size;
        this.updateHierarchyLabels();
    }

    /**
     * Setup Three.js scene, camera, renderer
     */
    setupScene() {
        if (!this.isWebGLAvailable()) {
            throw new Error('WebGL not available in this browser');
        }

        // Scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x0a0e1a);
        this.scene.fog = new THREE.Fog(0x0a0e1a, this.boundingRadius * 0.05, this.boundingRadius * 10);

        // Camera
        this.camera = new THREE.PerspectiveCamera(
            60,
            this.container.clientWidth / this.container.clientHeight,
            0.1,
            1000
        );
        const camDist = this.boundingRadius * 0.5;
        this.camera.position.set(camDist, camDist, camDist);
        this.camera.lookAt(0, 0, 0);

        // Renderer
        this.renderer = new THREE.WebGLRenderer({
            antialias: !this.isMobile,
            alpha: true
        });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, this.isMobile ? 1.0 : 2));
        this.container.appendChild(this.renderer.domElement);

        // Controls
        this.controls = new window.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.minDistance = Math.max(10, this.boundingRadius * 0.05);
        this.controls.maxDistance = this.boundingRadius * 8;
        this.controls.target.set(0, 0, 0);

        // Lights
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        this.scene.add(ambientLight);

        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.4);
        directionalLight.position.set(50, 50, 50);
        this.scene.add(directionalLight);

        // Handle window resize
        window.addEventListener('resize', () => this.onWindowResize());
    }

    isWebGLAvailable() {
        try {
            if (typeof window === 'undefined' || !window.WebGLRenderingContext) {
                return false;
            }
            const canvas = document.createElement('canvas');
            return !!(canvas.getContext('webgl') || canvas.getContext('experimental-webgl'));
        } catch (err) {
            return false;
        }
    }

    showWebGLError(details = '') {
        const warning = document.getElementById('webgl-warning');
        if (!warning) return;
        const info = document.getElementById('webgl-warning-details');
        if (info && details) {
            info.textContent = `Browser message: ${details}`;
        }
        warning.classList.add('visible');
        this.webglWarningShown = true;
    }

    hideWebGLError() {
        const warning = document.getElementById('webgl-warning');
        if (warning) {
            warning.classList.remove('visible');
        }
        this.webglWarningShown = false;
    }

    async handleInitializationError(error) {
        const message = error?.message || '';
        if (message.toLowerCase().includes('webgl')) {
            this.showWebGLError(message);
            this.hideLoadingScreen();
            this.current2DMode = 'circle-pack';
            const selector = document.getElementById('multi-lens-selector');
            if (selector) {
                selector.value = 'circle-pack';
            }

            // Show SVG container, hide 3D container
            const graphContainer = document.getElementById('graph-container');
            const svgContainer = document.getElementById('svg-container-2d');
            if (graphContainer) graphContainer.classList.add('hidden');
            if (svgContainer) svgContainer.classList.add('active');

            try {
                await this.render2DVisualization('circle-pack');
            } catch (renderErr) {
                console.error('Fallback 2D rendering failed:', renderErr);
            }
        }
    }

    /**
     * Create entity node visualizations
     */
    createEntityNodes() {
        const geoSegments = this.isMobile ? 6 : 16;
        const geometry = new THREE.SphereGeometry(1.0, geoSegments, geoSegments);

        let processed = 0;
        const cap = this.isMobile ? this.mobileNodeLimit : Infinity;

        for (const [entityId, entity] of this.entities) {
            if (processed >= cap) break;
            // Determine color based on type
            const color = this.getEntityColor(entity);
            const scale = this.getNodeScale(entity);

            // Create material
            const material = new THREE.MeshStandardMaterial({
                color: color,
                emissive: color,
                emissiveIntensity: 0.4,
                metalness: 0.2,
                roughness: 0.6,
                transparent: true,
                opacity: 0.9
            });

            // Create mesh
            const mesh = new THREE.Mesh(geometry, material);
            mesh.position.set(...entity.position);
            mesh.scale.set(scale, scale, scale);
            mesh.userData = {
                entityId: entityId,
                entity: entity,
                baseScale: scale
            };

            this.scene.add(mesh);
            this.entityMeshes.push(mesh);
            this.visibleEntities.add(entityId);
            this.entityMeshMap.set(entityId, mesh);
            this.displayedEntityIds.add(entityId);
            processed++;
        }

        this.updateVisibleCount();
    }

    /**
     * Create relationship line visualizations
     */
    createRelationshipLines() {
        if (!this.relationships || this.relationships.length === 0) {
            return;
        }

        if (this.isMobile) {
            return; // skip edges on mobile to reduce memory/GPU load
        }

        for (const relationship of this.relationships) {
            const sourceEntity = this.entities.get(relationship.source);
            const targetEntity = this.entities.get(relationship.target);
            const sourceMesh = this.entityMeshMap.get(relationship.source);
            const targetMesh = this.entityMeshMap.get(relationship.target);

            if (!sourceEntity || !targetEntity || !sourceMesh || !targetMesh) continue;

            const geometry = new THREE.BufferGeometry().setFromPoints([
                new THREE.Vector3(...sourceEntity.position),
                new THREE.Vector3(...targetEntity.position)
            ]);

            const strength = this.getRelationshipStrength(relationship);
            const effectiveStrength = strength > 0 ? strength : 0.1; // show edges even when data lacks strengths
            const lineWidth = this.mapStrengthToLineWidth(effectiveStrength);
            const opacity = Math.min(0.7, 0.08 + (effectiveStrength * 0.35));
            const isAttribution = relationship.type === 'MAKES_CLAIM';

            const materialBase = {
                color: isAttribution ? this.edgeStyles.attributionColor : this.edgeStyles.defaultColor,
                transparent: true,
                opacity: opacity,
                depthWrite: false,
                linewidth: lineWidth
            };

            let material;
            if (isAttribution) {
                material = new THREE.LineDashedMaterial({
                    ...materialBase,
                    dashSize: 1.2,
                    gapSize: 0.8,
                    scale: 1
                });
            } else {
                material = new THREE.LineBasicMaterial(materialBase);
            }

            const line = new THREE.Line(geometry, material);

            if (isAttribution && line.computeLineDistances) {
                line.computeLineDistances();
            }

            line.userData = {
                sourceId: relationship.source,
                targetId: relationship.target,
                type: relationship.type,
                strength: strength,
                baseOpacity: opacity,
                sourceMesh,
                targetMesh
            };

            this.scene.add(line);
            this.connectionLines.push(line);
        }

        this.updateRelationshipVisibility();
    }

    /**
     * Update node scales when size mode changes
     */
    updateNodeScales() {
        this.entityMeshes.forEach(mesh => {
            const entity = mesh.userData.entity;
            const scale = this.getNodeScale(entity);
            mesh.userData.baseScale = scale;
            mesh.scale.set(scale, scale, scale);
        });

        // Re-apply selection highlight if any
        this.updateSelectionHighlight();
    }

    /**
     * Create Fresnel shader cluster membranes
     */
    createClusterMembranes() {
        if (this.isMobile) {
            // Mobile: only show Global (L0 Red), very light
            this.createMembranesForLevel(0, 0.0, 0xFF4444);
            return;
        }
        // Desktop: Create membranes for each hierarchy level
        // CORRECTED HIERARCHY (inverted): L0=ROOT (66), L1=MID (762), L2=FINE (583), L3=LEAF (14)
        this.createMembranesForLevel(0, 0.0, 0xFF4444); // L0 Global/Root (Red) - 66 clusters
        this.createMembranesForLevel(1, 0.0, 0xFFCC00); // L1 Community (Gold) - 762 clusters
        this.createMembranesForLevel(2, 0.0, 0x00CCFF); // L2 Fine (Cyan) - 583 clusters
        // L3 (14 leaf clusters) - optional, can be added if needed
        // this.createMembranesForLevel(3, 0.0, 0x00CCFF);
    }

    /**
     * Create membranes for a specific hierarchy level
     */
    createMembranesForLevel(level, baseOpacity, colorHex) {
        const levelKey = `level_${level}`;
        const clusters = this.clusters[levelKey];

        for (const cluster of clusters) {
            // Get member entity positions
            const memberPositions = this.getMemberPositions(cluster);

            if (memberPositions.length < 3) continue;

            // Fit ellipsoid to cluster
            const ellipsoid = this.fitEllipsoid(memberPositions);

            // Create Fresnel shader membrane
            const membrane = this.createFresnelMembrane(ellipsoid, baseOpacity, level, colorHex);
            membrane.userData = {
                clusterId: cluster.id,
                level: level,
                baseOpacity: baseOpacity,
                baseEdgeOpacity: Math.max(baseOpacity * 12.0, 0.05)
            };

            this.scene.add(membrane);
            this.clusterMeshes.push(membrane);
        }
    }

    scheduleMembraneRefresh(delay = 400) {
        if (this.membraneRefreshTimeout) return;
        this.membraneRefreshTimeout = setTimeout(() => {
            this.refreshClusterMembranes();
            this.membraneRefreshTimeout = null;
        }, delay);
    }

    refreshClusterMembranes() {
        if (!this.clusterMeshes.length) return;
        this.clusterMeshes.forEach(mesh => {
            const levelKey = `level_${mesh.userData.level}`;
            const lookup = this.clusterLookup[levelKey];
            const cluster = lookup ? lookup.get(mesh.userData.clusterId) : null;
            if (!cluster) return;
            const positions = this.getMemberPositions(cluster);
            if (positions.length < 3) return;
            const ellipsoid = this.fitEllipsoid(positions);
            cluster.center = ellipsoid.center;
            this.updateMembraneMesh(mesh, ellipsoid);
        });
        this.updateClusterLabelPositions();
    }

    updateClusterLabelPositions() {
        if (!this.clusterLabelSprites || !this.clusterLabelSprites.length) return;
        this.clusterLabelSprites.forEach(sprite => {
            if (!sprite.userData || !sprite.userData.isClusterLabel) return;
            const levelKey = `level_${sprite.userData.level}`;
            const cluster = this.clusterLookup[levelKey]?.get(sprite.userData.clusterId);
            if (cluster && cluster.center) {
                sprite.position.set(...cluster.center);
            }
        });
    }

    /**
     * Get positions of all entities in a cluster
     */
    getMemberPositions(cluster) {
        const positions = [];

        // Get entity IDs from cluster
        const entityIds = cluster.entities || [];

        for (const entityId of entityIds) {
            const entity = this.entities.get(entityId);
            if (entity && entity.position) {
                positions.push(entity.position);
            }
        }

        return positions;
    }

    /**
     * Fit ellipsoid to point cloud using PCA
     */
    fitEllipsoid(positions) {
        const EPSILON = 5.0; // Minimum radius

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

        // Handle degenerate cases
        if (positions.length < 3) {
            return {
                center: centroid,
                radii: [EPSILON, EPSILON, EPSILON],
                rotation: new THREE.Matrix4()
            };
        }

        // Compute covariance matrix (simplified)
        let varX = 0, varY = 0, varZ = 0;
        for (const pos of positions) {
            const dx = pos[0] - centroid[0];
            const dy = pos[1] - centroid[1];
            const dz = pos[2] - centroid[2];
            varX += dx * dx;
            varY += dy * dy;
            varZ += dz * dz;
        }

        // Compute radii using 2 * standard deviation (sigma)
        const radii = [
            Math.max(Math.sqrt(varX / positions.length) * 1.0, EPSILON),
            Math.max(Math.sqrt(varY / positions.length) * 1.0, EPSILON),
            Math.max(Math.sqrt(varZ / positions.length) * 1.0, EPSILON)
        ];

        return {
            center: centroid,
            radii: radii,
            rotation: new THREE.Matrix4() // Simplified - no rotation for now
        };
    }

    /**
     * Create Fresnel shader membrane for ellipsoid
     */
    createFresnelMembrane(ellipsoid, baseOpacity, level, colorHex) {
        // Create ellipsoid geometry
        const segments = this.isMobile ? 8 : 32;
        const geometry = new THREE.SphereGeometry(1, segments, segments);

        // Fresnel shader material
        const material = new THREE.ShaderMaterial({
            uniforms: {
                ellipsoidCenter: { value: new THREE.Vector3(...ellipsoid.center) },
                ellipsoidRadius: { value: Math.max(...ellipsoid.radii) },
                baseOpacity: { value: 0.0 },
                edgeOpacity: { value: Math.max(baseOpacity * 12.0, 0.05) },
                color: { value: new THREE.Color(colorHex || 0x667eea) }
            },
            vertexShader: `
                varying vec3 vNormal;
                varying vec3 vWorldPosition;

                void main() {
                    vNormal = normalize(normalMatrix * normal);
                    vec4 worldPosition = modelMatrix * vec4(position, 1.0);
                    vWorldPosition = worldPosition.xyz;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
            `,
            fragmentShader: `
                uniform vec3 ellipsoidCenter;
                uniform float ellipsoidRadius;
                uniform float baseOpacity;
                uniform float edgeOpacity;
                uniform vec3 color;

                varying vec3 vNormal;
                varying vec3 vWorldPosition;

                void main() {
                    // Fresnel effect (edge glow)
                    vec3 viewDirection = normalize(cameraPosition - vWorldPosition);
                    float fresnel = pow(1.0 - dot(vNormal, viewDirection), 4.0);

                    // Fade when camera is inside ellipsoid
                    float distanceToCamera = length(cameraPosition - ellipsoidCenter);
                    float insideFade = smoothstep(0.0, ellipsoidRadius * 0.5, distanceToCamera);

                    // Combine effects: edge glow only, center near-transparent
                    float edgeOnly = fresnel;
                    float opacity = mix(baseOpacity, edgeOpacity, edgeOnly) * insideFade;

                    gl_FragColor = vec4(color, opacity);
                }
            `,
            transparent: true,
            side: THREE.FrontSide,
            depthWrite: false,
            depthTest: true,
            blending: THREE.NormalBlending
        });

        // Create mesh
        const mesh = new THREE.Mesh(geometry, material);
        this.updateMembraneMesh(mesh, ellipsoid);

        return mesh;
    }

    updateMembraneMesh(mesh, ellipsoid) {
        mesh.position.set(...ellipsoid.center);
        mesh.scale.set(ellipsoid.radii[0], ellipsoid.radii[1], ellipsoid.radii[2]);
        if (mesh.material && mesh.material.uniforms) {
            mesh.material.uniforms.ellipsoidCenter.value.set(...ellipsoid.center);
            mesh.material.uniforms.ellipsoidRadius.value = Math.max(...ellipsoid.radii);
        }
    }

    /**
     * Setup UI controls and interactions
     */
    setupControls() {
        // Multi-Lens Dropdown
        const multiLensSelector = document.getElementById('multi-lens-selector');
        if (multiLensSelector) {
            multiLensSelector.addEventListener('change', (e) => {
                const mode = e.target.value;
                if (mode) {
                    this.setMode(mode).catch(err => console.error('Failed to change mode', err));
                }
            });
        }

        // Color mode buttons
        document.querySelectorAll('.mode-btn[data-color-mode]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const mode = e.target.dataset.colorMode;
                this.setColorMode(mode);
            });
        });

        // Size mode selector
        const sizeModeSelect = document.getElementById('size-mode');
        if (sizeModeSelect) {
            sizeModeSelect.addEventListener('change', (e) => {
                this.setSizeMode(e.target.value);
            });
        }

        // Entity type filters
        const filterContainer = document.getElementById('entity-type-filters');
        for (const [type, color] of Object.entries(this.typeColors)) {
            const count = Array.from(this.entities.values()).filter(e => e.type === type).length;

            const filterItem = document.createElement('div');
            filterItem.className = 'filter-item';
            filterItem.innerHTML = `
                <input type="checkbox" id="filter-${type}" class="filter-checkbox" checked>
                <label for="filter-${type}" class="filter-label">${type}</label>
                <span class="filter-count">${count}</span>
            `;

            filterContainer.appendChild(filterItem);

            const checkbox = filterItem.querySelector('input');
            checkbox.addEventListener('change', (e) => {
                this.typeVisibility[type] = e.target.checked;
                this.updateEntityVisibility();
            });
        }

        // Cluster membrane toggles
        ['l3', 'l2', 'l1'].forEach(level => {
            const checkbox = document.getElementById(`show-clusters-${level}`);
            if (checkbox) {
                checkbox.addEventListener('change', () => this.updateMembraneVisibility());
            }
        });
        this.updateMembraneVisibility();

        // Labels toggle
        const showLabelsCheckbox = document.getElementById('show-labels');
        if (showLabelsCheckbox) {
            showLabelsCheckbox.addEventListener('change', () => {
                this.updateTopLabels();
            });
        }

        // Panel toggle
        document.getElementById('panel-toggle').addEventListener('click', () => {
            const panel = document.getElementById('info-panel');
            panel.classList.toggle('collapsed');
        });

        // Raycaster for hover/click
        this.raycaster = new THREE.Raycaster();
        this.mouse = new THREE.Vector2();

        this.renderer.domElement.addEventListener('mousemove', (e) => this.onMouseMove(e));
        this.renderer.domElement.addEventListener('click', (e) => this.onMouseClick(e));
    }

    /**
     * Setup keyboard shortcuts
     */
    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Don't trigger shortcuts when typing in input fields
            const isInputFocused = e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA';

            switch(e.key.toLowerCase()) {
                case 'e':
                    if (!isInputFocused) this.setMode('semantic').catch(console.error);
                    break;
                case 'c':
                    if (!isInputFocused) this.setMode('contextual').catch(console.error);
                    break;
                case 's':
                    if (!isInputFocused) this.setMode('structural').catch(console.error);
                    break;
                case 'escape':
                    this.deselectEntity();
                    break;
                case 'k':
                    if (e.ctrlKey || e.metaKey) {
                        e.preventDefault();
                        document.getElementById('search-input').focus();
                    }
                    break;
            }
        });
    }

    /**
     * Setup search functionality with autocomplete
     */
    setupSearch() {
        const searchInput = document.getElementById('search-input');
        const dropdown = document.getElementById('autocomplete-dropdown');
        if (!searchInput || !dropdown) return;

        let selectedIndex = -1;
        let currentResults = [];

        const selectEntity = (entityId) => {
            const entity = this.entities.get(entityId);
            if (entity) {
                this.centerOnEntity(entityId);
                this.selectEntity(entityId);
                searchInput.value = entityId;
                hideDropdown();
            }
        };

        const showDropdown = () => {
            dropdown.classList.add('show');
        };

        const hideDropdown = () => {
            dropdown.classList.remove('show');
            selectedIndex = -1;
        };

        const updateDropdown = (results) => {
            currentResults = results;
            dropdown.innerHTML = '';

            if (results.length === 0) {
                dropdown.innerHTML = '<div class="autocomplete-no-results">No entities found</div>';
                showDropdown();
                return;
            }

            results.forEach((result, index) => {
                const item = document.createElement('div');
                item.className = 'autocomplete-item';
                item.innerHTML = `
                    <div class="autocomplete-item-name">${result.id}</div>
                    <div class="autocomplete-item-type">${result.entity.type || 'UNKNOWN'}</div>
                `;

                item.addEventListener('click', () => selectEntity(result.id));
                item.addEventListener('mouseenter', () => {
                    selectedIndex = index;
                    updateSelection();
                });

                dropdown.appendChild(item);
            });

            showDropdown();
        };

        const updateSelection = () => {
            const items = dropdown.querySelectorAll('.autocomplete-item');
            items.forEach((item, index) => {
                item.classList.toggle('selected', index === selectedIndex);
            });
        };

        const searchEntities = (query) => {
            if (!query || query.length < 2) {
                hideDropdown();
                return;
            }

            const q = query.toLowerCase();
            const matches = [];

            // Find matching entities (limit to 10 results)
            for (const [id, entity] of this.entities.entries()) {
                if (id.toLowerCase().includes(q)) {
                    matches.push({ id, entity });
                    if (matches.length >= 10) break;
                }
            }

            // Sort by relevance (exact matches first, then alphabetical)
            matches.sort((a, b) => {
                const aExact = a.id.toLowerCase() === q;
                const bExact = b.id.toLowerCase() === q;
                if (aExact && !bExact) return -1;
                if (!aExact && bExact) return 1;
                return a.id.localeCompare(b.id);
            });

            updateDropdown(matches);
        };

        // Input event for autocomplete
        searchInput.addEventListener('input', (e) => {
            searchEntities(e.target.value.trim());
        });

        // Keyboard navigation
        searchInput.addEventListener('keydown', (e) => {
            const items = dropdown.querySelectorAll('.autocomplete-item');

            if (e.key === 'ArrowDown') {
                e.preventDefault();
                selectedIndex = Math.min(selectedIndex + 1, items.length - 1);
                updateSelection();
                items[selectedIndex]?.scrollIntoView({ block: 'nearest' });
            } else if (e.key === 'ArrowUp') {
                e.preventDefault();
                selectedIndex = Math.max(selectedIndex - 1, -1);
                updateSelection();
                if (selectedIndex >= 0) {
                    items[selectedIndex]?.scrollIntoView({ block: 'nearest' });
                }
            } else if (e.key === 'Enter') {
                e.preventDefault();
                if (selectedIndex >= 0 && currentResults[selectedIndex]) {
                    selectEntity(currentResults[selectedIndex].id);
                } else if (currentResults.length > 0) {
                    selectEntity(currentResults[0].id);
                }
            } else if (e.key === 'Escape') {
                hideDropdown();
                searchInput.blur();
            }
        });

        // Hide dropdown when clicking outside
        document.addEventListener('click', (e) => {
            if (!searchInput.contains(e.target) && !dropdown.contains(e.target)) {
                hideDropdown();
            }
        });

        // Show dropdown when focusing search input
        searchInput.addEventListener('focus', () => {
            if (searchInput.value.trim().length >= 2) {
                searchEntities(searchInput.value.trim());
            }
        });
    }

    /**
     * Setup cluster search and focus mode
     */
    setupClusterSearch() {
        const clusterSearchInput = document.getElementById('cluster-search');
        const clusterDropdown = document.getElementById('cluster-autocomplete-dropdown');
        const clearFocusBtn = document.getElementById('clear-cluster-focus');
        const focusInfo = document.getElementById('cluster-focus-info');
        const focusedClusterName = document.getElementById('focused-cluster-name');

        if (!clusterSearchInput || !clusterDropdown) return;

        this.focusedCluster = null;
        let selectedIndex = -1;
        let currentResults = [];

        const focusEntity = (entity) => {
            if (!entity) return;

            // Update UI
            focusedClusterName.textContent = `Entity: ${entity.name || entity.title || entity.id}`;
            focusInfo.style.display = 'block';
            clearFocusBtn.style.display = 'block';
            clusterSearchInput.value = '';
            hideClusterDropdown();

            // Fly camera to entity
            const mesh = this.entityMeshMap.get(entity.id);
            if (mesh) {
                const targetPos = mesh.position;
                this.tweenCameraToPosition([targetPos.x, targetPos.y, targetPos.z], 150);
            }

            // Apply entity focus mode (100% target, 10% others)
            this.applyEntityFocus(entity.id);
        };

        const focusCluster = (clusterId, level) => {
            const cluster = this.clusters[`level_${level}`]?.find(c => c.id === clusterId);
            if (!cluster) return;

            this.focusedCluster = { id: clusterId, level };
            const clusterName = cluster.name || cluster.title || clusterId;

            // Update UI
            focusedClusterName.textContent = clusterName;
            focusInfo.style.display = 'block';
            clearFocusBtn.style.display = 'block';
            clusterSearchInput.value = '';
            hideClusterDropdown();

            // Fly camera to cluster
            this.tweenCameraToCluster(cluster.center || [0, 0, 0]);

            // Apply focus mode rendering
            this.applyClusterFocus(clusterId, level);
        };

        const clearFocus = () => {
            this.focusedCluster = null;
            focusInfo.style.display = 'none';
            clearFocusBtn.style.display = 'none';
            this.clearClusterFocus();
        };

        const showClusterDropdown = () => {
            clusterDropdown.classList.add('show');
        };

        const hideClusterDropdown = () => {
            clusterDropdown.classList.remove('show');
            selectedIndex = -1;
        };

        const updateClusterDropdown = (results) => {
            currentResults = results;
            clusterDropdown.innerHTML = '';

            if (results.length === 0) {
                clusterDropdown.innerHTML = '<div class="autocomplete-no-results">No results found</div>';
                showClusterDropdown();
                return;
            }

            // Group results by category
            const grouped = {
                'L0 Global Themes': { title: '🔴 L0 Global Themes', items: [], color: '#FF4444' },
                'L1 Communities': { title: '🟡 L1 Communities', items: [], color: '#FFCC00' },
                'Entities': { title: '🔵 Entities', items: [], color: '#00CCFF' }
            };

            results.forEach(r => {
                if (grouped[r.category]) {
                    grouped[r.category].items.push(r);
                }
            });

            // Render grouped results
            ['L0 Global Themes', 'L1 Communities', 'Entities'].forEach(category => {
                const group = grouped[category];
                if (group.items.length === 0) return;

                const groupDiv = document.createElement('div');
                groupDiv.className = 'cluster-group';

                const header = document.createElement('div');
                header.className = 'cluster-group-header';
                header.textContent = group.title;
                groupDiv.appendChild(header);

                group.items.forEach((result, idx) => {
                    const item = document.createElement('div');
                    item.className = 'cluster-autocomplete-item';
                    item.dataset.globalIndex = results.indexOf(result);

                    let statsText = '';
                    if (result.type === 'cluster') {
                        const entityCount = result.cluster.entities?.length || 0;
                        statsText = `${entityCount} entities`;
                    } else if (result.type === 'entity') {
                        statsText = result.entityType || 'Entity';
                    }

                    item.innerHTML = `
                        <div class="cluster-item-name">${result.name}</div>
                        <div class="cluster-item-stats">${statsText}</div>
                    `;

                    item.addEventListener('click', () => {
                        if (result.type === 'cluster') {
                            focusCluster(result.cluster.id, result.level);
                        } else if (result.type === 'entity') {
                            focusEntity(result.entity);
                        }
                    });

                    item.addEventListener('mouseenter', () => {
                        selectedIndex = results.indexOf(result);
                        updateSelection();
                    });

                    groupDiv.appendChild(item);
                });

                clusterDropdown.appendChild(groupDiv);
            });

            showClusterDropdown();
        };

        const updateSelection = () => {
            const items = clusterDropdown.querySelectorAll('.cluster-autocomplete-item');
            items.forEach((item, idx) => {
                const globalIdx = parseInt(item.dataset.globalIndex);
                item.classList.toggle('selected', globalIdx === selectedIndex);
            });
        };

        const searchClusters = (query) => {
            if (!query || query.length < 2) {
                hideClusterDropdown();
                return;
            }

            const q = query.toLowerCase();
            const matches = [];

            // Search L0 (Global Themes - Red)
            const level0Clusters = this.clusters['level_0'] || [];
            level0Clusters.forEach(cluster => {
                const name = cluster.name || cluster.title || cluster.id || '';
                if (name.toLowerCase().includes(q)) {
                    matches.push({
                        name,
                        cluster,
                        level: 0,
                        type: 'cluster',
                        category: 'L0 Global Themes'
                    });
                }
            });

            // Search L1 (Communities - Gold)
            const level1Clusters = this.clusters['level_1'] || [];
            level1Clusters.forEach(cluster => {
                const name = cluster.name || cluster.title || cluster.id || '';
                if (name.toLowerCase().includes(q)) {
                    matches.push({
                        name,
                        cluster,
                        level: 1,
                        type: 'cluster',
                        category: 'L1 Communities'
                    });
                }
            });

            // Search Entities (Cyan)
            Array.from(this.entities.values()).forEach(entity => {
                const name = entity.name || entity.title || '';
                if (name.toLowerCase().includes(q)) {
                    matches.push({
                        name,
                        entity,
                        type: 'entity',
                        entityType: entity.type,
                        category: 'Entities'
                    });
                }
            });

            // Limit to 30 results
            const limited = matches.slice(0, 30);

            updateClusterDropdown(limited);
        };

        // Input event for autocomplete
        clusterSearchInput.addEventListener('input', (e) => {
            searchClusters(e.target.value.trim());
        });

        // Keyboard navigation
        clusterSearchInput.addEventListener('keydown', (e) => {
            const items = clusterDropdown.querySelectorAll('.cluster-autocomplete-item');

            if (e.key === 'ArrowDown') {
                e.preventDefault();
                selectedIndex = Math.min(selectedIndex + 1, currentResults.length - 1);
                updateSelection();
                const selected = Array.from(items).find(item => parseInt(item.dataset.globalIndex) === selectedIndex);
                selected?.scrollIntoView({ block: 'nearest' });
            } else if (e.key === 'ArrowUp') {
                e.preventDefault();
                selectedIndex = Math.max(selectedIndex - 1, -1);
                updateSelection();
                if (selectedIndex >= 0) {
                    const selected = Array.from(items).find(item => parseInt(item.dataset.globalIndex) === selectedIndex);
                    selected?.scrollIntoView({ block: 'nearest' });
                }
            } else if (e.key === 'Enter') {
                e.preventDefault();
                if (selectedIndex >= 0 && currentResults[selectedIndex]) {
                    const result = currentResults[selectedIndex];
                    focusCluster(result.cluster.id, result.level);
                }
            } else if (e.key === 'Escape') {
                hideClusterDropdown();
                clusterSearchInput.blur();
            }
        });

        // Clear focus button
        clearFocusBtn.addEventListener('click', clearFocus);

        // Hide dropdown when clicking outside
        document.addEventListener('click', (e) => {
            if (!clusterSearchInput.contains(e.target) && !clusterDropdown.contains(e.target)) {
                hideClusterDropdown();
            }
        });

        // Show dropdown when focusing search input
        clusterSearchInput.addEventListener('focus', () => {
            if (clusterSearchInput.value.trim().length >= 2) {
                searchClusters(clusterSearchInput.value.trim());
            }
        });
    }

    /**
     * Apply cluster focus mode (brighten selected cluster, dim others)
     */
    applyClusterFocus(clusterId, level) {
        const levelKey = `level_${level}`;
        const focusedCluster = this.clusters[levelKey]?.find(c => c.id === clusterId);
        if (!focusedCluster) return;

        // Get entities in focused cluster
        const focusedEntityIds = new Set(focusedCluster.entities || []);

        // Update cluster membrane opacities
        this.clusterMeshes.forEach(mesh => {
            const isFocused = mesh.userData.level === level && mesh.userData.clusterId === clusterId;
            if (mesh.material && mesh.material.uniforms) {
                mesh.material.uniforms.edgeOpacity.value = isFocused ? 0.8 : 0.05;
            }
        });

        // Update entity opacities and sizes
        this.entityMeshes.forEach(mesh => {
            const entityId = mesh.userData.entityId;
            const isFocused = focusedEntityIds.has(entityId);

            mesh.material.opacity = isFocused ? 0.9 : 0.15;
            mesh.material.emissiveIntensity = isFocused ? 0.4 : 0.1;

            const baseScale = mesh.userData.baseScale || 1;
            const scaleFactor = isFocused ? 1.0 : 0.5;
            mesh.scale.set(baseScale * scaleFactor, baseScale * scaleFactor, baseScale * scaleFactor);
        });

        // Dim relationship lines
        this.connectionLines.forEach(line => {
            const sourceInFocus = focusedEntityIds.has(line.userData.sourceId);
            const targetInFocus = focusedEntityIds.has(line.userData.targetId);
            line.material.opacity = (sourceInFocus && targetInFocus) ? 0.4 : 0.02;
        });
    }

    /**
     * Apply entity focus mode (100% target, 10% others)
     */
    applyEntityFocus(targetEntityId) {
        // Dim cluster membranes
        this.clusterMeshes.forEach(mesh => {
            if (mesh.material && mesh.material.uniforms) {
                mesh.material.uniforms.edgeOpacity.value = 0.05;
            }
        });

        // Update entity opacities: 100% for target, 10% for others
        this.entityMeshes.forEach(mesh => {
            const entityId = mesh.userData.entityId;
            const isFocused = entityId === targetEntityId;

            mesh.material.opacity = isFocused ? 1.0 : 0.1;
            mesh.material.emissiveIntensity = isFocused ? 0.6 : 0.05;

            const baseScale = mesh.userData.baseScale || 1;
            const scaleFactor = isFocused ? 1.5 : 0.5;
            mesh.scale.set(baseScale * scaleFactor, baseScale * scaleFactor, baseScale * scaleFactor);
        });

        // Show only relationship lines connected to focused entity
        this.connectionLines.forEach(line => {
            const connectedToFocus = (line.userData.sourceId === targetEntityId ||
                                      line.userData.targetId === targetEntityId);
            line.material.opacity = connectedToFocus ? 0.6 : 0.01;
        });
    }

    /**
     * Clear cluster focus mode (restore normal rendering)
     */
    clearClusterFocus() {
        // Restore membrane visibility based on checkboxes
        this.updateMembraneVisibility();

        // Restore entity opacities
        this.entityMeshes.forEach(mesh => {
            mesh.material.opacity = 0.9;
            mesh.material.emissiveIntensity = 0.4;

            const baseScale = mesh.userData.baseScale || 1;
            mesh.scale.set(baseScale, baseScale, baseScale);
        });

        // Restore relationship lines
        this.connectionLines.forEach(line => {
            const baseOpacity = line.userData.baseOpacity || 0.4;
            line.material.opacity = this.isStructuralMode() ? baseOpacity : 0.0;
        });

        // Re-apply selection highlight if any entity is selected
        if (this.selectedEntity) {
            this.updateSelectionHighlight();
        }
    }

    /**
     * Handle mouse move for hover effects
     */
    onMouseMove(event) {
        if (this.mode === 'circle-pack' || this.mode === 'voronoi') {
            return;
        }

        // Calculate mouse position in normalized device coordinates
        const rect = this.renderer.domElement.getBoundingClientRect();
        this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

        // Raycast to find hovered entity
        this.raycaster.setFromCamera(this.mouse, this.camera);
        const intersects = this.raycaster.intersectObjects(this.entityMeshes);

        if (intersects.length > 0) {
            const hoveredMesh = intersects[0].object;
            const entityId = hoveredMesh.userData.entityId;

            if (this.hoveredEntity !== entityId) {
                this.hoveredEntity = entityId;
                this.showEntityInfo(entityId);
            }
        } else if (this.hoveredEntity) {
            this.hoveredEntity = null;
            this.hideEntityInfo();
        }
    }

    /**
     * Handle mouse click for selection
     */
    onMouseClick(event) {
        if (this.mode === 'circle-pack' || this.mode === 'voronoi') {
            return;
        }

        // FIX 3: Check for cluster label clicks first (interactive drill-down)
        this.raycaster.setFromCamera(this.mouse, this.camera);
        const labelIntersects = this.raycaster.intersectObjects(this.clusterLabelSprites);

        if (labelIntersects.length > 0) {
            const clickedLabel = labelIntersects[0].object;
            if (clickedLabel.userData.isClusterLabel && clickedLabel.userData.clusterPosition) {
                this.tweenCameraToCluster(clickedLabel.userData.clusterPosition);
                return;
            }
        }

        // Fall back to entity selection
        if (this.hoveredEntity) {
            this.selectEntity(this.hoveredEntity);
        }
    }

    /**
     * Show entity info panel
     */
    showEntityInfo(entityId, forceShow = false) {
        const entity = this.entities.get(entityId);
        if (!entity) return;

        this.setInfoPanelMode('entity');
        document.getElementById('entity-name').textContent = entityId;
        document.getElementById('entity-type').textContent = entity.type;
        document.getElementById('entity-description').textContent =
            entity.description || 'No description available';

        // Count connections
        const connections = this.relationships.filter(
            r => r.source === entityId || r.target === entityId
        ).length;
        document.getElementById('entity-connections').textContent = connections;

        // Show betweenness centrality
        const betweenness = typeof entity.betweenness === 'number' ? entity.betweenness : 0;
        document.getElementById('entity-centrality').textContent =
            betweenness.toFixed(4);

        // Connectivity stats
        const degree = entity.degree || 0;
        const weighted = entity.weightedDegree || 0;
        const statDegree = document.getElementById('entity-degree');
        const statWeighted = document.getElementById('entity-weighted-degree');
        if (statDegree) statDegree.textContent = degree.toString();
        if (statWeighted) statWeighted.textContent = weighted.toFixed(2);

        // Neighbor preview (top 5 by strength)
        const neighborList = document.getElementById('entity-neighbors');
        if (neighborList) {
            neighborList.innerHTML = '';
            const neighbors = this.getTopNeighbors(entityId, 5);
            neighbors.forEach(n => {
                const li = document.createElement('li');
                li.textContent = `${n.id}${n.type ? ` (${n.type})` : ''}`;
                neighborList.appendChild(li);
            });
            if (neighbors.length === 0) {
                const li = document.createElement('li');
                li.textContent = 'No neighbors found';
                neighborList.appendChild(li);
            }
        }

        // Incoming/outgoing edges
        const incomingList = document.getElementById('entity-incoming');
        const outgoingList = document.getElementById('entity-outgoing');
        const { incoming, outgoing } = this.getEdgeDetails(entityId, 8);
        if (incomingList) {
            incomingList.innerHTML = '';
            if (incoming.length === 0) {
                const li = document.createElement('li');
                li.textContent = 'None';
                incomingList.appendChild(li);
            } else {
                incoming.forEach(edge => {
                    const li = document.createElement('li');
                    li.textContent = `${edge.source} —${edge.type}→ ${edge.target}`;
                    incomingList.appendChild(li);
                });
            }
        }
        if (outgoingList) {
            outgoingList.innerHTML = '';
            if (outgoing.length === 0) {
                const li = document.createElement('li');
                li.textContent = 'None';
                outgoingList.appendChild(li);
            } else {
                outgoing.forEach(edge => {
                    const li = document.createElement('li');
                    li.textContent = `${edge.source} —${edge.type}→ ${edge.target}`;
                    outgoingList.appendChild(li);
                });
            }
        }

        if (forceShow) {
            document.getElementById('entity-info').style.display = 'block';
        } else {
            document.getElementById('entity-info').style.display = 'block';
        }
    }

    /**
     * Show cluster info in side panel (used by 2D organic views)
     */
    showClusterInfo(clusterNode) {
        if (!clusterNode) return;

        this.setInfoPanelMode('cluster');

        const name = clusterNode.title || clusterNode.name || clusterNode.id;
        const metaParts = [];
        if (clusterNode.level !== undefined && clusterNode.level !== null) {
            metaParts.push(`Level ${clusterNode.level}`);
        }
        if (typeof clusterNode.community_id !== 'undefined') {
            metaParts.push(`ID ${clusterNode.community_id}`);
        }

        document.getElementById('entity-name').textContent = name;
        document.getElementById('entity-type').textContent = `Cluster · ${metaParts.join(' • ') || 'Community'}`;
        document.getElementById('entity-description').textContent =
            clusterNode.summary || 'No summary available for this community.';

        const formatNumber = (value) => {
            if (typeof value === 'number') {
                return value.toLocaleString();
            }
            return value ?? '0';
        };

        const setText = (id, value) => {
            const el = document.getElementById(id);
            if (el) el.textContent = value;
        };

        setText('entity-connections', formatNumber(clusterNode.entity_count || 0));
        setText('entity-centrality', formatNumber(clusterNode.descendant_count || 0));
        setText('entity-degree', formatNumber(clusterNode.child_cluster_count || 0));
        const sampleCount = clusterNode.entitySamples?.length
            || Math.min(clusterNode.entityIds?.length || 0, this.entityLeafLimit);
        setText('entity-weighted-degree', formatNumber(sampleCount));

        const renderList = (elementId, values, emptyText = 'None') => {
            const list = document.getElementById(elementId);
            if (!list) return;
            list.innerHTML = '';
            if (!values || !values.length) {
                const li = document.createElement('li');
                li.textContent = emptyText;
                list.appendChild(li);
                return;
            }
            values.forEach(value => {
                const li = document.createElement('li');
                li.textContent = value;
                list.appendChild(li);
            });
        };

        const childClusters = (clusterNode.children || []).filter(child => child.level !== 'entity');
        const neighborItems = childClusters.slice(0, 6).map(child => child.title || child.name || child.id);
        if (!neighborItems.length && clusterNode.entitySamples?.length) {
            neighborItems.push(
                ...clusterNode.entitySamples.slice(0, 6).map(sample => sample.name || sample.id)
            );
        }
        renderList('entity-neighbors', neighborItems, 'No child communities');

        const parentItems = [];
        let parentRef = clusterNode.parentRef;
        while (parentRef) {
            parentItems.push(parentRef.title || parentRef.name || parentRef.id);
            parentRef = parentRef.parentRef;
        }
        parentItems.reverse();
        renderList('entity-incoming', parentItems, 'Root community');

        const entitySampleDetails = clusterNode.entitySamples && clusterNode.entitySamples.length
            ? clusterNode.entitySamples
            : (clusterNode.entityIds || [])
                .slice(0, this.entityLeafLimit)
                .map(id => this.buildEntityLeaf(id))
                .filter(Boolean);
        const entitySamples = entitySampleDetails.map(sample => {
            if (!sample) return '';
            return sample.type ? `${sample.name || sample.id} (${sample.type})` : (sample.name || sample.id);
        }).filter(Boolean);
        renderList('entity-outgoing', entitySamples, 'No entity samples available');

        document.getElementById('entity-info').style.display = 'block';
    }

    setInfoPanelMode(mode) {
        if (this.infoPanelMode === mode) return;
        this.infoPanelMode = mode;

        const entityLabels = {
            'stat-label-connections': 'Connections',
            'stat-label-centrality': 'Centrality',
            'stat-label-degree': 'Degree',
            'stat-label-weighted': 'Weighted',
            'neighbors-title': 'Top neighbors',
            'incoming-title': 'Incoming',
            'outgoing-title': 'Outgoing'
        };

        const clusterLabels = {
            'stat-label-connections': 'Entities',
            'stat-label-centrality': 'Total Descendants',
            'stat-label-degree': 'Child Communities',
            'stat-label-weighted': 'Sample Entities',
            'neighbors-title': 'Child Communities',
            'incoming-title': 'Parent Path',
            'outgoing-title': 'Entity Samples'
        };

        const labelMap = mode === 'cluster' ? clusterLabels : entityLabels;
        Object.entries(labelMap).forEach(([id, text]) => {
            const el = document.getElementById(id);
            if (el) el.textContent = text;
        });
    }

    /**
     * Hide entity info panel
     */
    hideEntityInfo() {
        const info = document.getElementById('entity-info');
        if (!info) return;

        this.setInfoPanelMode('entity');
        document.getElementById('entity-name').textContent = 'Select a node';
        document.getElementById('entity-type').textContent = '';
        document.getElementById('entity-description').textContent = 'Click a node to view details here.';
        const fields = ['entity-connections', 'entity-centrality', 'entity-degree', 'entity-weighted-degree'];
        fields.forEach(id => {
            const el = document.getElementById(id);
            if (el) el.textContent = '0';
        });
        ['entity-neighbors', 'entity-incoming', 'entity-outgoing'].forEach(id => {
            const list = document.getElementById(id);
            if (list) {
                list.innerHTML = '';
                const li = document.createElement('li');
                li.textContent = 'None';
                list.appendChild(li);
            }
        });
        info.style.display = 'block';
    }

    /**
     * Center camera/controls on a specific entity by id
     */
    centerOnEntity(entityId) {
        const entity = this.entities.get(entityId);
        if (!entity || !entity.position) return;

        const target = new THREE.Vector3(...entity.position);
        const dir = new THREE.Vector3(1, 1, 1).normalize();
        const distance = this.boundingRadius * 0.4;
        const camPos = target.clone().add(dir.multiplyScalar(distance));

        this.controls.target.copy(target);
        this.camera.position.copy(camPos);
        this.camera.lookAt(target);
    }

    /**
     * FIX 3: Tween camera to cluster position (smooth drill-down navigation)
     */
    tweenCameraToCluster(clusterPosition) {
        const target = new THREE.Vector3(...clusterPosition);
        const dir = new THREE.Vector3(1, 1, 1).normalize();
        const distance = this.boundingRadius * 0.3; // Zoom in closer for clusters
        const camPos = target.clone().add(dir.multiplyScalar(distance));

        // Animate camera position and target
        const duration = 1000; // 1 second animation
        const startPos = this.camera.position.clone();
        const startTarget = this.controls.target.clone();
        const startTime = Date.now();

        const animate = () => {
            const elapsed = Date.now() - startTime;
            const progress = Math.min(elapsed / duration, 1);

            // Ease-in-out function for smooth animation
            const eased = progress < 0.5
                ? 2 * progress * progress
                : 1 - Math.pow(-2 * progress + 2, 2) / 2;

            // Interpolate camera position
            this.camera.position.lerpVectors(startPos, camPos, eased);
            this.controls.target.lerpVectors(startTarget, target, eased);

            if (progress < 1) {
                requestAnimationFrame(animate);
            }
        };

        animate();
    }

    centerOnEntityName(name) {
        const match = this.findEntityByQuery(name);
        if (match) {
            this.centerOnEntity(match.id);
            this.selectEntity(match.id);
        }
    }

    /**
     * Select an entity
     */
    selectEntity(entityId) {
        this.selectedEntity = entityId;
        console.log('Selected entity:', entityId);

        this.showEntityInfo(entityId, true);
        this.updateSelectionHighlight();
    }

    /**
     * Deselect current entity
     */
    deselectEntity() {
        this.selectedEntity = null;
        this.hoveredEntity = null;
        this.hideEntityInfo();

        this.updateSelectionHighlight();
    }

    /**
     * Set visualization mode
     */
    async setMode(mode) {
        if (mode === 'holographic') {
            window.location.href = '/YonEarth/graph/';
            return;
        }

        const validModes = ['semantic', 'contextual', 'structural', 'circle-pack', 'voronoi'];
        if (!validModes.includes(mode)) return;
        const is3DMode = mode === 'semantic' || mode === 'contextual' || mode === 'structural';
        if (is3DMode && this.webglWarningShown) {
            this.showWebGLError('WebGL is disabled, falling back to Circle Pack view.');
            const selector = document.getElementById('multi-lens-selector');
            if (selector) selector.value = 'circle-pack';
            await this.render2DVisualization('circle-pack');
            return;
        }
        if (mode === 'contextual' && !this.hasContextualLayoutAvailable()) {
            console.warn('Contextual mode requires GraphSAGE layout; defaulting to Semantic.');
            alert('Contextual mode is unavailable because GraphSAGE layout data is missing on the server.');
            mode = 'semantic';
        }
        if (mode === this.mode) return;

        this.mode = mode;
        this.currentLayout = mode;

        // Update dropdown selection
        const multiLensSelector = document.getElementById('multi-lens-selector');
        if (multiLensSelector && multiLensSelector.value !== mode) {
            multiLensSelector.value = mode;
        }

        // Handle 2D vs 3D mode switching
        const is2DMode = mode === 'circle-pack' || mode === 'voronoi';
        const graphContainer = document.getElementById('graph-container');
        const svgContainer = document.getElementById('svg-container-2d');

        if (is2DMode) {
            // Switch to 2D Hierarchy View
            this.current2DMode = mode;
            graphContainer.classList.add('hidden');
            svgContainer.classList.add('active');

            // PERFORMANCE: Hide edges in hierarchy views (they focus on volume/territory, not connectivity)
            this.hideEdges();

            this.hideOrganicTooltip();
            await this.render2DVisualization(mode);
        } else {
            // Switch to 3D Network View
            this.current2DMode = null;
            this.hideOrganicTooltip();
            svgContainer.classList.remove('active');
            graphContainer.classList.remove('hidden');

            // Restore edges for network views
            this.showEdges();

            if (mode === 'structural') {
                await this.startStructuralMode();
            } else {
                this.stopStructuralMode();
                const layoutKey = mode === 'contextual' ? 'sagePosition' : 'umapPosition';
                this.transitionToLayout(layoutKey);
            }

            this.updateSelectionHighlight();
            this.updateRelationshipVisibility();
            this.updateTopLabels();
        }

        console.log('Mode:', mode);
    }

    isStructuralMode() {
        return this.mode === 'structural';
    }

    /**
     * Set node coloring mode
     */
    setColorMode(mode) {
        if (!['type', 'centrality'].includes(mode)) return;

        this.colorMode = mode;

        document.querySelectorAll('.mode-btn[data-color-mode]').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.colorMode === mode);
        });

        this.applyColorMode();
    }

    /**
     * Set node size mode
     */
    setSizeMode(mode) {
        if (!this.sizeModes.includes(mode)) return;
        this.sizeMode = mode;

        const select = document.getElementById('size-mode');
        if (select) {
            select.value = mode;
        }

        this.updateNodeScales();
        this.updateTopLabels();
    }

    hasContextualLayoutAvailable() {
        return this.hasContextualLayout;
    }

    updateModeAvailability() {
        const multiLensSelector = document.getElementById('multi-lens-selector');
        if (multiLensSelector) {
            const contextualOption = Array.from(multiLensSelector.options)
                .find(opt => opt.value === 'contextual');
            if (contextualOption) {
                const unavailable = !this.hasContextualLayoutAvailable();
                contextualOption.disabled = unavailable;
                if (unavailable) {
                    contextualOption.text = 'Contextual (GraphSAGE) - Unavailable';
                } else {
                    contextualOption.text = 'Contextual (GraphSAGE)';
                }
            }
        }
    }

    transitionToLayout(layoutKey) {
        const attribute = layoutKey === 'sagePosition' ? 'sagePosition' : 'umapPosition';

        if (attribute === 'sagePosition' && !this.hasContextualLayoutAvailable()) {
            console.warn('Contextual layout unavailable. Falling back to semantic layout.');
            if (layoutKey !== 'umapPosition') {
                this.transitionToLayout('umapPosition');
            }
            return;
        }

        console.log(`Transitioning to ${attribute === 'sagePosition' ? 'Contextual (GraphSAGE)' : 'Semantic (UMAP)'} layout...`);

        const now = performance.now();
        this.activeTransitions = [];

        this.entityMeshes.forEach(mesh => {
            const entity = mesh.userData.entity;

            // Get target position
            let target;
            if (attribute === 'sagePosition') {
                // Use GraphSAGE position from graphsageLayout file
                const sageData = this.graphsageLayout[entity.id];
                target = sageData ? [sageData.x, sageData.y, sageData.z] : entity.umapPosition;

                // Validate target for NaN/undefined values
                if (target && (isNaN(target[0]) || isNaN(target[1]) || isNaN(target[2]) ||
                               target[0] === undefined || target[1] === undefined || target[2] === undefined)) {
                    console.warn(`Invalid GraphSAGE position for entity ${entity.id}, using UMAP fallback`);
                    target = entity.umapPosition;
                }
            } else {
                // Use UMAP position
                target = entity.umapPosition || entity[attribute];
            }

            if (!target) return;

            // Final validation: ensure all coordinates are valid numbers
            if (isNaN(target[0]) || isNaN(target[1]) || isNaN(target[2])) {
                console.warn(`Invalid target position for entity ${entity.id}, skipping transition`);
                return;
            }

            const start = [mesh.position.x, mesh.position.y, mesh.position.z];
            const end = [...target];

            // Add smooth transition with 1.5s duration
            this.activeTransitions.push({
                mesh,
                entity,
                start,
                end,
                startTime: now,
                duration: 1500 // 1.5s tween
            });
        });

        console.log(`Started ${this.activeTransitions.length} position transitions (1.5s duration)`);

        // Mark that membranes need updating after transition completes
        this.pendingMembraneUpdate = true;
        if (!this.activeTransitions.length) {
            this.pendingMembraneUpdate = false;
            this.refreshClusterMembranes();
            this.updateConnectionLinesGeometry();
        }
    }

    updateActiveTransitions() {
        if (!this.activeTransitions.length) return;
        const now = performance.now();
        this.activeTransitions = this.activeTransitions.filter(transition => {
            const elapsed = now - transition.startTime;
            const progress = Math.min(1, elapsed / transition.duration);
            const eased = this.easeInOut(progress);
            const x = transition.start[0] + (transition.end[0] - transition.start[0]) * eased;
            const y = transition.start[1] + (transition.end[1] - transition.start[1]) * eased;
            const z = transition.start[2] + (transition.end[2] - transition.start[2]) * eased;

            transition.mesh.position.set(x, y, z);
            transition.entity.position = [x, y, z];
            transition.entity.x = x;
            transition.entity.y = y;
            transition.entity.z = z;

            return progress < 1;
        });

        if (!this.activeTransitions.length) {
            this.updateConnectionLinesGeometry();
            if (this.pendingMembraneUpdate) {
                this.refreshClusterMembranes();
                this.pendingMembraneUpdate = false;
            }
        }
    }

    easeInOut(t) {
        return t < 0.5
            ? 4 * t * t * t
            : 1 - Math.pow(-2 * t + 2, 3) / 2;
    }

    async ensureForceModule() {
        if (this.d3Force) return;
        const candidates = [
            'https://esm.sh/d3-force-3d@3.0.0?bundle',
            'https://cdn.jsdelivr.net/npm/d3-force-3d@3.0.0/+esm',
            'https://unpkg.com/d3-force-3d@3.0.0/dist/d3-force-3d.min.js'
        ];
        let lastError = null;
        for (const url of candidates) {
            try {
                this.d3Force = await import(url);
                if (this.d3Force) {
                    console.log(`Loaded d3-force-3d from ${url}`);
                    return;
                }
            } catch (err) {
                lastError = err;
                console.warn(`Failed to load d3-force-3d from ${url}`, err);
            }
        }
        throw new Error(`Unable to load d3-force-3d from CDN (${lastError?.message || 'unknown error'})`);
    }

    async prepareForceSimulation() {
        await this.ensureForceModule();

        this.forceNodes = this.entityMeshes.map(mesh => {
            const entity = mesh.userData.entity;
            entity.x = mesh.position.x;
            entity.y = mesh.position.y;
            entity.z = mesh.position.z;
            entity.vx = 0;
            entity.vy = 0;
            entity.vz = 0;
            entity.fx = null;
            entity.fy = null;
            entity.fz = null;
            return entity;
        });

        const visibleSet = new Set(this.forceNodes.map(node => node.id));
        this.forceLinks = (this.relationships || [])
            .filter(rel => visibleSet.has(rel.source) && visibleSet.has(rel.target))
            .map(rel => ({
                source: rel.source,
                target: rel.target,
                value: this.getRelationshipStrength(rel) || 0.1
            }));

        if (!this.forceSimulation) {
            this.forceLinkForce = this.d3Force.forceLink(this.forceLinks)
                .id(node => node.id)
                .distance(d => 80 + (1 - d.value) * 120)
                .strength(d => 0.05 + d.value * 0.4);

            this.forceSimulation = this.d3Force.forceSimulation(this.forceNodes)
                .numDimensions(3)
                .force('link', this.forceLinkForce)
                .force('charge', this.d3Force.forceManyBody().strength(-8))
                .force('center', this.d3Force.forceCenter(0, 0, 0))
                .force('collision', this.d3Force.forceCollide().radius(node => {
                    const base = node.weightedDegree ? Math.min(30, node.weightedDegree * 0.6 + 6) : 6;
                    return base;
                }))
                .alphaDecay(0.05)
                .on('tick', () => this.applyForcePositions());
        } else {
            this.forceSimulation.nodes(this.forceNodes);
            if (this.forceLinkForce) {
                this.forceLinkForce.links(this.forceLinks);
            }
        }
    }

    async startStructuralMode() {
        await this.prepareForceSimulation();
        if (this.forceSimulation) {
            this.forceSimulation.alpha(1).alphaTarget(0.3).restart();
        }
        this.activeTransitions = [];
        this.scheduleMembraneRefresh(600);
    }

    stopStructuralMode() {
        if (this.forceSimulation) {
            this.forceSimulation.stop();
        }
    }

    applyForcePositions() {
        if (!this.isStructuralMode()) return;
        this.entityMeshes.forEach(mesh => {
            const entity = mesh.userData.entity;
            if (typeof entity.x !== 'number') return;
            mesh.position.set(entity.x, entity.y, entity.z);
            entity.position = [entity.x, entity.y, entity.z];
        });
        this.updateConnectionLinesGeometry();
        this.scheduleMembraneRefresh(600);
    }

    /**
     * Keep labels legible by scaling with camera distance
     */
    updateLabelScales() {
        if (!this.labelSprites || this.labelSprites.length === 0) return;

        const cam = this.camera;
        const fovRadians = (cam.fov * Math.PI) / 180;
        const scaleFactor = 2 * Math.tan(fovRadians / 2);

        this.labelSprites.forEach(sprite => {
            const entityId = sprite.userData?.entityId;
            if (entityId) {
                const entity = this.entities.get(entityId);
                if (entity && entity.position) {
                    sprite.position.set(...entity.position);
                }
            }
            const dist = cam.position.distanceTo(sprite.position);
            // Approximate world size needed for consistent screen size
            const desiredWorldSize = dist * scaleFactor * 0.12; // bigger factor for readability
            const minSize = 40;
            const maxSize = 420;
            const size = Math.max(minSize, Math.min(maxSize, desiredWorldSize));
            sprite.scale.set(size, size * 0.5, 1);
        });

        // Cluster label scaling
        if (this.clusterLabelSprites && this.clusterLabelSprites.length) {
            this.clusterLabelSprites.forEach(sprite => {
                if (sprite.userData?.isClusterLabel) {
                    const levelKey = `level_${sprite.userData.level}`;
                    const cluster = this.clusterLookup[levelKey]?.get(sprite.userData.clusterId);
                    if (cluster && cluster.center) {
                        sprite.position.set(...cluster.center);
                    }
                }
                const dist = cam.position.distanceTo(sprite.position);
                const desiredWorldSize = dist * scaleFactor * 0.1;
                const minSize = 30;
                const maxSize = 360;
                const size = Math.max(minSize, Math.min(maxSize, desiredWorldSize));
                sprite.scale.set(size, size * 0.5, 1);
            });
        }
    }

    /**
     * Update entity visibility based on type filters
     */
    updateEntityVisibility() {
        let visibleCount = 0;

        this.entityMeshes.forEach(mesh => {
            const entity = mesh.userData.entity;
            const visible = this.typeVisibility.hasOwnProperty(entity.type)
                ? this.typeVisibility[entity.type]
                : true;
            mesh.visible = visible;

            if (visible) {
                visibleCount++;
                this.visibleEntities.add(entity.id);
            } else {
                this.visibleEntities.delete(entity.id);
            }
        });

        this.updateRelationshipVisibility();
        this.updateVisibleCount();
    }

    /**
     * Update visible entity count in UI
     */
    updateVisibleCount() {
        document.getElementById('visible-count').textContent = this.visibleEntities.size;
    }

    /**
     * Create/update top labels based on current size mode
     */
    updateTopLabels() {
        // Remove old labels
        this.labelSprites.forEach(sprite => this.scene.remove(sprite));
        this.labelSprites = [];

        const labelLimit = this.isMobile ? 4 : 15;
        const showLabels = document.getElementById('show-labels')?.checked;
        if (!showLabels) {
            return;
        }

        const scores = [];
        for (const entity of this.entities.values()) {
            const score = this.sizeMode === 'betweenness'
                ? (typeof entity.betweenness === 'number' ? entity.betweenness : 0)
                : (entity.weightedDegree > 0 ? entity.weightedDegree : (entity.degree || 0));
            if (!isFinite(score)) continue;
            scores.push({ id: entity.id, score, entity });
        }

        scores.sort((a, b) => b.score - a.score);
        const top = scores.slice(0, labelLimit).filter(s => s.score > 0);

        // Ensure Y on Earth Community is always labeled if present
        const yoec = this.findEntityByQuery('Y on Earth Community');
        if (yoec && !top.find(t => t.id === yoec.id)) {
            top.push({ id: yoec.id, score: Number.MAX_VALUE, entity: yoec.entity });
        }

        top.forEach(item => {
            const sprite = this.createTextSprite(item.entity.id);
            sprite.position.set(...item.entity.position);
            sprite.scale.set(60, 30, 1);
            sprite.userData.entityId = item.entity.id;
            this.scene.add(sprite);
            this.labelSprites.push(sprite);
        });

        // Cluster labels
        this.createClusterLabels();
    }

    createClusterLabels() {
        if (this.isMobile) return;
        // Remove existing labels
        this.clusterLabelSprites.forEach(s => this.scene.remove(s));
        this.clusterLabelSprites = [];

        const addLabel = (level, text, position, clusterId) => {
            const sprite = this.createTextSprite(text || '');
            sprite.position.set(...position);
            sprite.scale.set(120, 60, 1);
            sprite.userData.level = level;
            sprite.userData.clusterId = clusterId;
            sprite.userData.clusterPosition = position;
            sprite.userData.isClusterLabel = true;
            this.scene.add(sprite);
            this.clusterLabelSprites.push(sprite);
        };

        // CORRECTED: L0 (Red) labels - Top 30 by node_count (Global/Root categories)
        const l0Clusters = this.clusters['level_0'] || [];
        const l0Sorted = l0Clusters
            .map(c => ({
                cluster: c,
                nodeCount: (c.entities || []).length
            }))
            .sort((a, b) => b.nodeCount - a.nodeCount)
            .slice(0, 30); // Top 30 largest L0 clusters (out of 66)

        l0Sorted.forEach(({ cluster }) => {
            const label = cluster.name || cluster.title || cluster.label || cluster.id || 'L0';
            const pos = cluster.center || [0, 0, 0];
            addLabel(0, label, pos, cluster.id);
        });

        // L1 and L2 labels (on-demand visibility based on camera distance)
        const levelDefs = [
            { level: 1, key: 'level_1', fallback: 'L1', maxLabels: 50 }, // Gold community clusters
            { level: 2, key: 'level_2', fallback: 'L2', maxLabels: 30 }  // Cyan fine clusters
        ];

        for (const def of levelDefs) {
            const clusters = this.clusters[def.key] || [];
            const sorted = clusters
                .map(c => ({
                    cluster: c,
                    nodeCount: (c.entities || []).length
                }))
                .sort((a, b) => b.nodeCount - a.nodeCount)
                .slice(0, def.maxLabels);

            sorted.forEach(({ cluster }) => {
                const label = cluster.name || cluster.title || cluster.label || cluster.id || def.fallback;
                const pos = cluster.center || [0, 0, 0];
                addLabel(def.level, label, pos, cluster.id);
            });
        }
    }

    updateClusterLabelVisibility(l0Weight = 1.0, l1Weight = 0, l2Weight = 0) {
        if (!this.clusterLabelSprites.length) return;
        this.clusterLabelSprites.forEach(sprite => {
            const lvl = sprite.userData.level;
            let weight = 0;
            if (lvl === 0) weight = l0Weight; // L0 (Red) - Global/Root
            if (lvl === 1) weight = l1Weight; // L1 (Gold) - Community
            if (lvl === 2) weight = l2Weight; // L2 (Cyan) - Fine
            sprite.visible = weight > 0.15;
        });
    }

    /**
     * Update LOD (Level of Detail) based on camera distance
     * L0 (Red): Visible when camera > 500
     * L1 (Gold): Visible when camera 200-500
     * L2 (Cyan): Visible when camera < 200
     */
    updateLOD() {
        if (!this.camera || !this.clusterMeshes.length) return;

        const cameraDistance = this.camera.position.length();

        // Update membrane visibility based on camera distance
        this.clusterMeshes.forEach(mesh => {
            const level = mesh.userData.level;
            let visible = false;

            if (level === 0) {
                // L0 (Red): Visible when far (>500)
                visible = cameraDistance > 500;
            } else if (level === 1) {
                // L1 (Gold): Visible when mid-range (200-500)
                visible = cameraDistance >= 200 && cameraDistance <= 500;
            } else if (level === 2) {
                // L2 (Cyan): Visible when close (<200)
                visible = cameraDistance < 200;
            }

            mesh.visible = visible;
        });

        // Update label visibility based on LOD
        const l0Weight = cameraDistance > 500 ? 1.0 : 0.0;
        const l1Weight = (cameraDistance >= 200 && cameraDistance <= 500) ? 1.0 : 0.0;
        const l2Weight = cameraDistance < 200 ? 1.0 : 0.0;

        this.updateClusterLabelVisibility(l0Weight, l1Weight, l2Weight);
    }

    createTextSprite(text) {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        const fontSize = 120;
        canvas.width = 1024;
        canvas.height = 512;
        ctx.font = `bold ${fontSize}px sans-serif`;

        // FIX 2: White text (#FFFFFF), black outline (#000000), larger outline
        ctx.fillStyle = '#FFFFFF';  // White text
        ctx.strokeStyle = '#000000'; // Black outline
        ctx.lineWidth = 8;  // Thicker outline for better contrast
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';

        // Stroke first for outline, then fill for white text on top
        ctx.strokeText(text, canvas.width / 2, canvas.height / 2);
        ctx.fillText(text, canvas.width / 2, canvas.height / 2);

        const texture = new THREE.CanvasTexture(canvas);
        texture.needsUpdate = true;

        // FIX 2: depthTest: false forces text to render ON TOP of membranes
        const material = new THREE.SpriteMaterial({
            map: texture,
            transparent: true,
            depthWrite: false,
            depthTest: false  // Always render on top
        });
        const sprite = new THREE.Sprite(material);
        sprite.userData.baseLabelScale = 1;
        return sprite;
    }

    /**
     * Show Leiden hierarchy labels and counts in the UI
     */
    updateHierarchyLabels() {
        const setCount = (id, value) => {
            const el = document.getElementById(id);
            if (el) {
                el.textContent = typeof value === 'number'
                    ? value.toLocaleString()
                    : (value || '–');
            }
        };

        setCount('hierarchy-count-l0', this.entities.size);
        setCount('hierarchy-count-l1', this.clusters.level_1.length);
        setCount('hierarchy-count-l2', this.clusters.level_2.length);

        const l3Count = this.clusters.level_3.length;
        const l3Pill = document.getElementById('hierarchy-pill-l3');
        if (l3Pill) {
            l3Pill.style.display = l3Count > 0 ? 'block' : 'none';
        }
        setCount('hierarchy-count-l3', l3Count);

        const keyNote = document.getElementById('hierarchy-key-note');
        if (keyNote) {
            const detected = [];
            if (this.hierarchyKeyMap.L2) detected.push(`L2->${this.hierarchyKeyMap.L2}`);
            if (this.hierarchyKeyMap.L1) detected.push(`L1->${this.hierarchyKeyMap.L1}`);
            if (this.hierarchyKeyMap.L0) detected.push(`L0->${this.hierarchyKeyMap.L0}`);
            if (this.hierarchyKeyMap.L3) detected.push(`L3->${this.hierarchyKeyMap.L3}`);

            keyNote.textContent = detected.length
                ? `Hierarchy keys detected: ${detected.join(', ')}`
                : 'No hierarchy keys detected in data.';
        }
    }

    /**
     * Update relationship visibility based on currently visible entities
     */
    updateRelationshipVisibility() {
        if (!this.connectionLines.length) return;
        const structural = this.isStructuralMode();

        this.connectionLines.forEach(line => {
            const sourceVisible = this.visibleEntities.has(line.userData.sourceId);
            const targetVisible = this.visibleEntities.has(line.userData.targetId);
            line.visible = structural && sourceVisible && targetVisible;
        });
    }

    updateConnectionLinesGeometry() {
        if (!this.connectionLines.length) return;
        this.connectionLines.forEach(line => {
            const sourceMesh = line.userData.sourceMesh || this.entityMeshMap.get(line.userData.sourceId);
            const targetMesh = line.userData.targetMesh || this.entityMeshMap.get(line.userData.targetId);
            if (!sourceMesh || !targetMesh) return;

            const positions = line.geometry.attributes.position.array;
            positions[0] = sourceMesh.position.x;
            positions[1] = sourceMesh.position.y;
            positions[2] = sourceMesh.position.z;
            positions[3] = targetMesh.position.x;
            positions[4] = targetMesh.position.y;
            positions[5] = targetMesh.position.z;
            line.geometry.attributes.position.needsUpdate = true;
            if (line.geometry.boundingSphere) {
                line.geometry.boundingSphere.center.copy(sourceMesh.position).lerp(targetMesh.position, 0.5);
            } else {
                line.geometry.computeBoundingSphere();
            }
        });
    }

    /**
     * Hide all edges (for hierarchy views)
     */
    hideEdges() {
        if (!this.connectionLines.length) return;
        this.connectionLines.forEach(line => {
            line.visible = false;
        });
        console.log('Edges hidden for hierarchy view');
    }

    /**
     * Show edges (for network views)
     */
    showEdges() {
        if (!this.connectionLines.length) return;
        // Restore visibility based on current mode
        this.updateRelationshipVisibility();
        console.log('Edges restored for network view');
    }

    /**
     * Update membrane visibility per hierarchy level
     */
    updateMembraneVisibility() {
        if (!this.clusterMeshes.length) return;
        const showL3Checkbox = document.getElementById('show-clusters-l3');
        const showL2Checkbox = document.getElementById('show-clusters-l2');
        const showL1Checkbox = document.getElementById('show-clusters-l1');

        const showL3Setting = showL3Checkbox ? showL3Checkbox.checked : true;
        const showL2Setting = showL2Checkbox ? showL2Checkbox.checked : true;
        const showL1Setting = showL1Checkbox ? showL1Checkbox.checked : false;

        const camDist = this.camera.position.distanceTo(this.controls.target);
        const fadeWidth = 100;

        const l3FadeStart = 450, l3FadeEnd = 550;
        const l2FadeStart = 150, l2FadeEnd = 250;

        // L3 weight (far)
        let l3Weight = camDist >= l3FadeEnd ? 1 : (camDist <= l3FadeStart ? 0 : (camDist - l3FadeStart) / (l3FadeEnd - l3FadeStart));
        // L2 weight (mid)
        let l2Weight = 0;
        if (camDist <= l3FadeStart && camDist >= l2FadeEnd) {
            l2Weight = 1;
        } else if (camDist > l3FadeStart && camDist < l3FadeEnd) {
            l2Weight = 1 - l3Weight; // crossfade with L3
        } else if (camDist > l2FadeStart && camDist < l2FadeEnd) {
            l2Weight = (camDist - l2FadeStart) / (l2FadeEnd - l2FadeStart); // crossfade with L1
        }

        // L1 weight (near)
        let l1Weight = camDist <= l2FadeStart ? 1 : 0;
        if (camDist > l2FadeStart && camDist < l2FadeEnd) {
            l1Weight = 1 - l2Weight; // crossfade with L2
        }

        this.clusterMeshes.forEach(mesh => {
            const level = mesh.userData.level;
            let weight = 0;
            if (level === 3) weight = l3Weight * (showL3Setting ? 1 : 0);
            if (level === 2) weight = l2Weight * (showL2Setting ? 1 : 0);
            if (level === 1) weight = l1Weight * (showL1Setting ? 1 : 0);

            // Ghost parent context: keep parent faint when child active
            if (level === 3 && (l2Weight > 0 || l1Weight > 0) && showL3Setting) {
                weight = Math.max(weight, 0.03);
            }
            if (level === 2 && l1Weight > 0 && showL2Setting) {
                weight = Math.max(weight, 0.03);
            }

            mesh.visible = weight > 0.001;

            if (mesh.material && mesh.material.uniforms) {
                const baseEdge = mesh.userData.baseEdgeOpacity || 0.05;
                mesh.material.uniforms.edgeOpacity.value = baseEdge * weight;
            }
        });

        // Update cluster label visibility based on weights
        this.updateClusterLabelVisibility(l3Weight, l2Weight, l1Weight);
    }

    /**
     * Emphasize selected node and its neighbors
     */
    updateSelectionHighlight() {
        const selected = this.selectedEntity;
        const neighborSet = new Set();

        if (selected) {
            for (const rel of this.relationships || []) {
                if (rel.source === selected) neighborSet.add(rel.target);
                if (rel.target === selected) neighborSet.add(rel.source);
            }
        }

        this.entityMeshes.forEach(mesh => {
            const entityId = mesh.userData.entityId;
            const baseScale = mesh.userData.baseScale || 1;

            let factor = 1.0;
            if (selected) {
                if (entityId === selected) {
                    factor = 2.6;
                    mesh.material.opacity = 1.0;
                    mesh.material.emissiveIntensity = 0.6;
                } else if (neighborSet.has(entityId)) {
                    factor = 1.7;
                    mesh.material.opacity = 0.95;
                    mesh.material.emissiveIntensity = 0.45;
                } else {
                    factor = 0.7;
                    mesh.material.opacity = 0.4;
                    mesh.material.emissiveIntensity = 0.2;
                }
            } else {
                mesh.material.opacity = 0.9;
                mesh.material.emissiveIntensity = 0.4;
            }

            mesh.scale.set(baseScale * factor, baseScale * factor, baseScale * factor);
        });

        const structural = this.isStructuralMode();
        this.connectionLines.forEach(line => {
            const baseOpacity = line.userData.baseOpacity || 0.4;
            if (!structural) {
                line.material.opacity = 0.0;
                return;
            }
            if (!selected) {
                line.material.opacity = baseOpacity;
                return;
            }
            const isConnected = line.userData.sourceId === selected || line.userData.targetId === selected;
            const isNeighbor = neighborSet.has(line.userData.sourceId) || neighborSet.has(line.userData.targetId);

            if (isConnected || isNeighbor) {
                line.material.opacity = Math.min(1.0, baseOpacity * 2.0);
            } else {
                line.material.opacity = Math.max(0.05, baseOpacity * 0.2);
            }
        });
    }

    /**
     * Update node materials based on selected color mode
     */
    applyColorMode() {
        this.entityMeshes.forEach(mesh => {
            const entity = mesh.userData.entity;
            const colorHex = this.getEntityColor(entity);
            mesh.material.color.setHex(colorHex);
            mesh.material.emissive.setHex(colorHex);
        });
    }

    /**
     * Determine entity color based on current color mode
     */
    getEntityColor(entity) {
        if (this.colorMode === 'centrality') {
            return this.getCentralityColor(entity.betweenness || 0);
        }

        return this.typeColors[entity.type] || 0xcccccc;
    }

    /**
     * Map betweenness centrality to a blue→yellow→red gradient
     */
    getCentralityColor(score) {
        const value = Math.max(0, Math.min(1, score || 0));
        const color = new THREE.Color();

        if (value <= 0.33) {
            color.lerpColors(
                new THREE.Color(this.centralityColors.low),
                new THREE.Color(this.centralityColors.mid),
                value / 0.33
            );
        } else if (value <= 0.67) {
            color.lerpColors(
                new THREE.Color(this.centralityColors.mid),
                new THREE.Color(this.centralityColors.high),
                (value - 0.33) / 0.34
            );
        } else {
            color.setHex(this.centralityColors.high);
        }

        return color.getHex();
    }

    /**
     * Compute relationship strength (0-1) from weight or per-entity strengths
     */
    getRelationshipStrength(relationship) {
        if (!relationship) return 0;

        if (typeof relationship.weight === 'number') {
            return this.clampStrength(relationship.weight);
        }

        const sourceStrengths = (this.entities.get(relationship.source) || {}).relationshipStrengths || {};
        const targetStrengths = (this.entities.get(relationship.target) || {}).relationshipStrengths || {};

        const strengths = [];

        if (typeof sourceStrengths[relationship.target] === 'number') {
            strengths.push(sourceStrengths[relationship.target]);
        }

        if (typeof targetStrengths[relationship.source] === 'number') {
            strengths.push(targetStrengths[relationship.source]);
        }

        if (strengths.length === 0) return 0;

        const avg = strengths.reduce((a, b) => a + b, 0) / strengths.length;
        return this.clampStrength(avg);
    }

    /**
     * Precompute degree and weighted degree per entity for sizing
     */
    computeConnectivityStats() {
        const degreeMap = new Map();

        const ensure = (id) => {
            if (!degreeMap.has(id)) {
                degreeMap.set(id, { degree: 0, weighted: 0 });
            }
            return degreeMap.get(id);
        };

        // Seed all entities to track zeros as well
        for (const id of this.entities.keys()) {
            ensure(id);
        }

        for (const rel of this.relationships || []) {
            const s = rel.source;
            const t = rel.target;
            if (!this.entities.has(s) || !this.entities.has(t)) continue;

            const entryS = ensure(s);
            const entryT = ensure(t);

            entryS.degree += 1;
            entryT.degree += 1;

            const strength = this.getRelationshipStrength(rel);
            const effective = strength > 0 ? strength : 0.1;
            entryS.weighted += effective;
            entryT.weighted += effective;
        }

        let minDegree = Infinity, maxDegree = 0;
        let minWeighted = Infinity, maxWeighted = 0;
        let minBetweenness = Infinity, maxBetweenness = 0;

        for (const [id, stats] of degreeMap.entries()) {
            const entity = this.entities.get(id);
            if (!entity) continue;

            entity.degree = stats.degree;
            entity.weightedDegree = stats.weighted;
        }

        for (const entity of this.entities.values()) {
            const deg = entity.degree || 0;
            const wdeg = entity.weightedDegree || 0;
            const btw = typeof entity.betweenness === 'number' ? entity.betweenness : null;

            minDegree = Math.min(minDegree, deg);
            maxDegree = Math.max(maxDegree, deg);
            minWeighted = Math.min(minWeighted, wdeg);
            maxWeighted = Math.max(maxWeighted, wdeg);

            if (btw !== null) {
                minBetweenness = Math.min(minBetweenness, btw);
                maxBetweenness = Math.max(maxBetweenness, btw);
            }
        }

        if (!isFinite(minDegree)) minDegree = 0;
        if (!isFinite(minWeighted)) minWeighted = 0;
        if (!isFinite(minBetweenness)) minBetweenness = 0;

        this.connectivityStats = {
            minDegree,
            maxDegree,
            minWeighted,
            maxWeighted,
            minBetweenness,
            maxBetweenness
        };
    }

    /**
     * Convert relationship strength to line width
     */
    mapStrengthToLineWidth(strength) {
        const clamped = this.clampStrength(strength);
        return this.edgeStyles.minWidth +
            clamped * (this.edgeStyles.maxWidth - this.edgeStyles.minWidth);
    }

    clampStrength(value) {
        const numeric = typeof value === 'string' ? parseFloat(value) : value;
        if (typeof numeric !== 'number' || Number.isNaN(numeric)) return 0;
        return Math.max(0, Math.min(1, numeric));
    }

    /**
     * Get top neighbors by strength
     */
    getTopNeighbors(entityId, limit = 5) {
        const neighbors = [];
        for (const rel of this.relationships || []) {
            if (rel.source === entityId || rel.target === entityId) {
                const otherId = rel.source === entityId ? rel.target : rel.source;
                const other = this.entities.get(otherId);
                if (!other) continue;
                const strength = this.getRelationshipStrength(rel);
                neighbors.push({ id: otherId, type: other.type, strength });
            }
        }
        neighbors.sort((a, b) => (b.strength || 0) - (a.strength || 0));
        return neighbors.slice(0, limit);
    }

    /**
     * Collect incoming/outgoing edges for an entity
     */
    getEdgeDetails(entityId, limit = 10) {
        const incoming = [];
        const outgoing = [];
        for (const rel of this.relationships || []) {
            if (rel.target === entityId) {
                incoming.push({
                    source: rel.source,
                    target: rel.target,
                    type: rel.type || 'related',
                    strength: this.getRelationshipStrength(rel)
                });
            }
            if (rel.source === entityId) {
                outgoing.push({
                    source: rel.source,
                    target: rel.target,
                    type: rel.type || 'related',
                    strength: this.getRelationshipStrength(rel)
                });
            }
        }

        incoming.sort((a, b) => (b.strength || 0) - (a.strength || 0));
        outgoing.sort((a, b) => (b.strength || 0) - (a.strength || 0));

        return {
            incoming: incoming.slice(0, limit),
            outgoing: outgoing.slice(0, limit)
        };
    }

    /**
     * Find an entity by query (case-insensitive substring match on id)
     */
    findEntityByQuery(query) {
        if (!query) return null;
        const q = query.toLowerCase();
        // Exact id match
        for (const [id, entity] of this.entities.entries()) {
            if (id.toLowerCase() === q) {
                return { id, entity };
            }
        }
        // Substring match fallback
        for (const [id, entity] of this.entities.entries()) {
            if (id.toLowerCase().includes(q)) {
                return { id, entity };
            }
        }
        return null;
    }

    /**
     * Map connectivity to node scale
     */
    getNodeScale(entity) {
        const stats = this.connectivityStats || {};
        const baseMinScale = 0.25;
        const baseMaxScale = 10.0;

        const scaleFrom = (value, min, max, customMin = baseMinScale, customMax = baseMaxScale) => {
            if (!isFinite(value)) value = 0;
            if (!isFinite(min)) min = 0;
            if (!isFinite(max) || max <= min) max = min + 1;
            const norm = (value - min) / (max - min);
            const clamped = Math.max(0, Math.min(1, norm));
            return customMin + clamped * (customMax - customMin);
        };

        if (this.sizeMode === 'betweenness' && typeof entity.betweenness === 'number') {
            const span = (stats.maxBetweenness || 0) - (stats.minBetweenness || 0);
            if (span > 1e-6) {
                const customMin = 1.2;
                const customMax = 12.0;
                const min = stats.minBetweenness;
                const max = stats.maxBetweenness;
                const norm = (entity.betweenness - min) / (max - min);
                const eased = Math.pow(Math.max(0, Math.min(1, norm)), 0.2); // very strong spread
                return customMin + eased * (customMax - customMin);
            }
            // fall back if no variation
        }

        const value = (entity && entity.weightedDegree > 0)
            ? entity.weightedDegree
            : (entity && entity.degree > 0 ? entity.degree : 1);
        const min = (stats.minWeighted && stats.minWeighted > 0)
            ? stats.minWeighted
            : (stats.minDegree || 0);
        const max = (stats.maxWeighted && stats.maxWeighted > 0)
            ? stats.maxWeighted
            : (stats.maxDegree || value);
        const logMin = Math.log1p(min);
        const logMax = Math.log1p(max);
        const logVal = Math.log1p(value);
        let norm = (logMax > logMin) ? (logVal - logMin) / (logMax - logMin) : 0;
        norm = Math.pow(Math.max(0, Math.min(1, norm)), 0.25); // very strong spread
        const clamped = Math.max(0, Math.min(1, norm));
        return baseMinScale + clamped * (baseMaxScale - baseMinScale);
    }

    /**
     * Normalize entity and cluster positions to a consistent scale centered at origin
     */
    normalizePositions() {
        if (this.entities.size === 0) return;

        const umapPositions = [];
        const sagePositions = [];

        for (const entity of this.entities.values()) {
            if (Array.isArray(entity.rawUmapPosition)) {
                umapPositions.push(entity.rawUmapPosition);
            }
            if (Array.isArray(entity.rawSagePosition)) {
                sagePositions.push(entity.rawSagePosition);
            }
        }

        const umapTransform = this.createNormalizationTransform(umapPositions);
        const sageTransform = this.createNormalizationTransform(sagePositions);

        let semanticRadius = 0;
        let contextualRadius = 0;

        for (const entity of this.entities.values()) {
            if (entity.rawUmapPosition && umapTransform) {
                entity.umapPosition = umapTransform(entity.rawUmapPosition);
            } else if (entity.rawUmapPosition) {
                entity.umapPosition = [...entity.rawUmapPosition];
            }

            if (entity.rawSagePosition && sageTransform) {
                entity.sagePosition = sageTransform(entity.rawSagePosition);
            } else if (entity.rawSagePosition) {
                entity.sagePosition = [...entity.rawSagePosition];
            } else {
                entity.sagePosition = null;
            }

            const initialPosition = entity.umapPosition
                ? [...entity.umapPosition]
                : (entity.sagePosition ? [...entity.sagePosition] : [0, 0, 0]);

            entity.position = initialPosition;
            entity.x = initialPosition[0];
            entity.y = initialPosition[1];
            entity.z = initialPosition[2];

            if (entity.umapPosition) {
                const r = Math.hypot(...entity.umapPosition);
                semanticRadius = Math.max(semanticRadius, r);
            }
            if (entity.sagePosition) {
                const r = Math.hypot(...entity.sagePosition);
                contextualRadius = Math.max(contextualRadius, r);
            }
        }

        if (umapTransform) {
            ['level_1', 'level_2', 'level_3'].forEach(level => {
                this.clusters[level].forEach(cluster => {
                    const rawCenter = cluster.rawCenter || cluster.center || [0, 0, 0];
                    cluster.center = umapTransform(rawCenter);
                });
            });
        }

        const maxRadius = Math.max(180, semanticRadius, contextualRadius);
        this.boundingRadius = maxRadius * 1.1;
    }

    createNormalizationTransform(positions, targetExtent = 1800) {
        if (!positions || positions.length === 0) {
            return null;
        }

        const mins = [Infinity, Infinity, Infinity];
        const maxs = [-Infinity, -Infinity, -Infinity];

        positions.forEach(pos => {
            if (!Array.isArray(pos)) return;
            mins[0] = Math.min(mins[0], pos[0]);
            mins[1] = Math.min(mins[1], pos[1]);
            mins[2] = Math.min(mins[2], pos[2]);
            maxs[0] = Math.max(maxs[0], pos[0]);
            maxs[1] = Math.max(maxs[1], pos[1]);
            maxs[2] = Math.max(maxs[2], pos[2]);
        });

        const extents = [
            maxs[0] - mins[0],
            maxs[1] - mins[1],
            maxs[2] - mins[2]
        ];

        const maxExtent = Math.max(...extents);
        if (!isFinite(maxExtent) || maxExtent === 0) {
            return (pos) => Array.isArray(pos) ? [...pos] : [0, 0, 0];
        }

        const scale = targetExtent / maxExtent;
        const center = [
            (mins[0] + maxs[0]) / 2,
            (mins[1] + maxs[1]) / 2,
            (mins[2] + maxs[2]) / 2
        ];
        const halfExtent = targetExtent / 2;
        const radialExponent = 0.45;

        return (pos) => {
            if (!Array.isArray(pos)) return [0, 0, 0];
            const x = (pos[0] - center[0]) * scale;
            const y = (pos[1] - center[1]) * scale;
            const z = (pos[2] - center[2]) * scale;
            const r = Math.sqrt(x * x + y * y + z * z);
            if (r === 0) return [0, 0, 0];

            const normalizedR = Math.min(1, r / halfExtent);
            const scaledR = halfExtent * Math.pow(normalizedR, radialExponent);
            const factor = scaledR / r;
            return [x * factor, y * factor, z * factor];
        };
    }

    /**
     * Handle window resize
     */
    onWindowResize() {
        this.camera.aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
    }

    /**
     * Update loading screen status
     */
    updateLoadingStatus(message) {
        const statusEl = document.getElementById('loading-status');
        if (statusEl) {
            statusEl.textContent = message;
        }
    }

    /**
     * Hide loading screen
     */
    hideLoadingScreen() {
        const loadingScreen = document.getElementById('loading-screen');
        if (loadingScreen) {
            loadingScreen.classList.add('hidden');
        }
    }

    hideOrganicTooltip() {
        // Stub method - can be expanded if tooltips are added later
    }

    resetCirclePackZoom() {
        // Reset Circle Pack zoom to root view
        if (!this.circlePackState.zoom || !this.circlePackState.group) {
            return;
        }

        const { svg, group, width, height } = this.circlePackState;

        // Reset zoom transform to identity (no zoom/pan)
        svg.transition()
            .duration(750)
            .call(this.circlePackState.zoom.transform, d3.zoomIdentity);
    }

    /**
     * Animation loop
     */
    animate() {
        requestAnimationFrame(() => this.animate());

        // Update controls
        this.controls.update();
        this.updateActiveTransitions();

        // Keep labels legible relative to camera distance
        this.updateLabelScales();
        this.updateMembraneVisibility();

        // Update LOD (Level of Detail) based on camera distance
        this.updateLOD();

        // Update Fresnel shader uniforms (camera position changes)
        // Render scene
        this.renderer.render(this.scene, this.camera);

        // Update FPS counter
        this.frameCount++;
        const now = Date.now();
        if (now - this.lastFPSUpdate >= 1000) {
            const fps = Math.round((this.frameCount * 1000) / (now - this.lastFPSUpdate));
            document.getElementById('fps-counter').textContent = fps;
            this.frameCount = 0;
            this.lastFPSUpdate = now;
        }
    }

    /**
     * ========================================
     * 2D VISUALIZATION METHODS (Circle Pack & Voronoi)
     * ========================================
     */

    /**
     * Render 2D visualization (Circle Pack or Voronoi)
     */
    async render2DVisualization(mode) {
        console.log('Rendering 2D visualization:', mode);

        // Always rebuild hierarchical data fresh
        this.hierarchicalData = this.buildHierarchicalData();

        if (!this.hierarchicalData || !this.hierarchicalData.children) {
            console.warn('Failed to build hierarchy data; skipping render.');
            return;
        }

        const svg = d3.select('#svg-container-2d');
        svg.selectAll('*').remove();

        const width = window.innerWidth;
        const height = window.innerHeight;
        svg.attr('width', width).attr('height', height);

        const g = svg.append('g').attr('class', 'organic-root');

        this.circlePackState.svg = svg;
        this.circlePackState.group = g;
        this.circlePackState.width = width;
        this.circlePackState.height = height;
        this.circlePackState.zoom = null;

        svg.on('click', () => {
            if (mode === 'circle-pack') {
                this.resetCirclePackZoom();
            }
            this.hideOrganicTooltip();
        });

        if (mode === 'circle-pack') {
            this.renderCirclePacking(svg, g, width, height);
        } else if (mode === 'voronoi') {
            this.renderVoronoiTreemap(svg, g, width, height);
        }
    }

    /**
     * Hide organic tooltip (stub for now)
     */
    hideOrganicTooltip() {
        // Stub method - can be expanded if tooltips are added later
    }

    /**
     * Build hierarchical data structure for D3
     * CORRECTED HIERARCHY: Root -> L0 (66 Red/Global) -> L1 (762 Gold/Community) -> Entities
     */
    buildHierarchicalData() {
        const clusters = this.clusters || {};
        // Use level_3 as top (coarse), level_2 as mid, level_1 as fine
        const level3Clusters = clusters.level_3 || [];
        const level2Clusters = clusters.level_2 || [];
        const level1Clusters = clusters.level_1 || [];

        const root = {
            name: "Knowledge Graph",
            id: "root",
            children: []
        };

        // Process Level 3 (7 coarse - top level as RED)
        for (const l3Cluster of level3Clusters) {
            const topNode = {
                name: l3Cluster.name || l3Cluster.title || l3Cluster.id,
                id: l3Cluster.id,
                level: 3,
                color: '#FF4444', // Red
                value: l3Cluster.node_count || l3Cluster.entities?.length || 10,
                children: []
            };

            // Add L2 children (medium clusters as GOLD)
            const l3Children = l3Cluster.children || [];
            for (const childId of l3Children.slice(0, 20)) { // Limit for performance
                const l2Cluster = level2Clusters.find(c => c.id === childId);
                if (!l2Cluster) continue;

                const midNode = {
                    name: l2Cluster.name || l2Cluster.title || childId,
                    id: childId,
                    level: 2,
                    color: '#FFCC00', // Gold
                    value: l2Cluster.node_count || l2Cluster.entities?.length || 5,
                    children: []
                };

                // Add L1 children (fine clusters as CYAN)
                const l2Children = l2Cluster.children || [];
                for (const l1ChildId of l2Children.slice(0, 10)) {
                    const l1Cluster = level1Clusters.find(c => c.id === l1ChildId);
                    if (!l1Cluster) continue;

                    midNode.children.push({
                        name: l1Cluster.name || l1Cluster.title || l1ChildId,
                        id: l1ChildId,
                        level: 1,
                        color: '#00CCFF', // Cyan
                        value: l1Cluster.node_count || l1Cluster.entities?.length || 1
                    });
                }

                if (midNode.children.length > 0) {
                    topNode.children.push(midNode);
                }
            }

            if (topNode.children.length > 0) {
                root.children.push(topNode);
            }
        }

        console.log(`Built hierarchy: ${root.children.length} top-level clusters`);
        return root;
    }

    /**
     * Render Circle Packing visualization
     * Displays: L0 (Red outer) -> L1 (Gold mid) -> Entities (Cyan inner)
     */
    renderCirclePacking(svg, g, width, height) {
        const hierarchy = d3.hierarchy(this.hierarchicalData)
            .sum(d => d.value || 1)
            .sort((a, b) => b.value - a.value);

        const pack = d3.pack()
            .size([width, height])
            .padding(5);

        const root = pack(hierarchy);

        // Create circles
        const circles = g.selectAll('circle')
            .data(root.descendants())
            .join('circle')
            .attr('cx', d => d.x)
            .attr('cy', d => d.y)
            .attr('r', d => d.r)
            .attr('fill', d => {
                if (d.depth === 0) return 'none'; // Root
                const level = d.data.level;
                if (level === 3) return '#FF4444'; // L3 Red (top)
                if (level === 2) return '#FFCC00'; // L2 Gold (mid)
                if (level === 1) return '#00CCFF'; // L1 Cyan (fine)
                return '#888888';
            })
            .attr('fill-opacity', d => {
                if (d.depth === 0) return 0; // Root
                const level = d.data.level;
                if (level === 3) return 0.3; // L3 faint
                if (level === 2) return 0.5; // L2 medium
                if (level === 1) return 0.7; // L1 bright
                return 0.3;
            })
            .attr('stroke', d => {
                if (d.depth === 0) return 'none';
                const level = d.data.level;
                if (level === 3) return '#FF4444';
                if (level === 2) return '#FFCC00';
                if (level === 1) return '#00CCFF';
                return '#888888';
            })
            .attr('stroke-width', d => d.depth === 0 ? 0 : 2)
            .style('cursor', d => d.depth > 0 && d.data.level >= 1 ? 'pointer' : 'default')
            .on('click', (event, d) => {
                if (d.depth > 0 && d.data.level >= 1) {
                    this.zoomToCircle(d, svg, width, height);
                }
            })
            .on('mouseover', (event, d) => {
                // Show cluster info on hover
                console.log('Cluster:', d.data.name, 'Level:', d.data.level);
            });

        // Add labels (L3 and L2 mainly, L1 if large enough)
        const labels = g.selectAll('text')
            .data(root.descendants().filter(d => d.r > 20 && d.depth > 0 && d.data.level >= 1))
            .join('text')
            .attr('x', d => d.x)
            .attr('y', d => d.y)
            .attr('text-anchor', 'middle')
            .attr('dy', '0.35em')
            .attr('fill', '#ffffff')
            .attr('font-size', d => Math.min(d.r / 4, 16))
            .attr('font-weight', d => d.data.level === 3 ? '700' : (d.data.level === 2 ? '600' : '500'))
            .attr('pointer-events', 'none')
            .text(d => {
                const name = d.data.name;
                const maxLen = Math.floor(d.r / 4);
                return name.length > maxLen ? name.substring(0, maxLen) + '...' : name;
            });

        // Add zoom behavior
        const zoom = d3.zoom()
            .scaleExtent([0.5, 10])
            .on('zoom', (event) => {
                g.attr('transform', event.transform);
            });

        svg.call(zoom);
    }

    /**
     * Zoom to specific circle
     */
    zoomToCircle(d, svg, width, height) {
        const scale = Math.min(width, height) / (d.r * 2 + 100);
        const translate = [width / 2 - d.x * scale, height / 2 - d.y * scale];

        svg.transition()
            .duration(750)
            .call(
                d3.zoom().transform,
                d3.zoomIdentity.translate(translate[0], translate[1]).scale(scale)
            );
    }

    /**
     * Render Voronoi visualization using entity UMAP positions
     * Applies Lloyd's relaxation for organic cell shapes
     */
    renderVoronoiTreemap(svg, g, width, height) {
        // Use entity UMAP positions as seeds (project to 2D)
        const entities = Array.from(this.entities.values());
        if (entities.length === 0) {
            console.warn('No entities found for Voronoi');
            return;
        }

        // Find bounds of UMAP positions
        const umapPositions = entities
            .filter(e => e.umapPosition && e.umapPosition.length >= 2)
            .map(e => ({
                x: e.umapPosition[0],
                y: e.umapPosition[1],
                entity: e
            }));

        if (umapPositions.length === 0) {
            console.warn('No valid UMAP positions');
            return;
        }

        // Normalize to fit viewport
        const xExtent = d3.extent(umapPositions, d => d.x);
        const yExtent = d3.extent(umapPositions, d => d.y);

        // Validate dimensions
        if (!width || !height || width < 200 || height < 200) {
            console.error('Invalid viewport dimensions for Voronoi:', width, height);
            return;
        }

        const padding = Math.min(100, width * 0.1, height * 0.1);
        const xScale = d3.scaleLinear()
            .domain(xExtent)
            .range([padding, width - padding]);
        const yScale = d3.scaleLinear()
            .domain(yExtent)
            .range([padding, height - padding]);

        // Create sites with scaled positions
        let sites = umapPositions.map(pos => ({
            x: xScale(pos.x),
            y: yScale(pos.y),
            entity: pos.entity,
            originalX: pos.x,
            originalY: pos.y
        }));

        // Validate bounds for Voronoi
        const bounds = [padding, padding, width - padding, height - padding];
        if (bounds[0] >= bounds[2] || bounds[1] >= bounds[3]) {
            console.error('Invalid Voronoi bounds:', bounds);
            return;
        }

        // Apply Lloyd's Relaxation (2 iterations)
        console.log('Applying Lloyd\'s relaxation...');
        for (let iteration = 0; iteration < 2; iteration++) {
            const delaunay = d3.Delaunay.from(sites, d => d.x, d => d.y);
            const voronoi = delaunay.voronoi(bounds);

            // Move each site to the centroid of its cell
            sites = sites.map((site, i) => {
                const cell = voronoi.cellPolygon(i);
                if (!cell) return site;

                // Calculate centroid
                const centroidX = d3.mean(cell, p => p[0]);
                const centroidY = d3.mean(cell, p => p[1]);

                return {
                    ...site,
                    x: centroidX,
                    y: centroidY
                };
            });
        }

        // Final Voronoi diagram
        const delaunay = d3.Delaunay.from(sites, d => d.x, d => d.y);
        const voronoi = delaunay.voronoi(bounds);

        // Color scale by entity type
        const entityTypes = [...new Set(sites.map(s => s.entity.type))];
        const colorScale = d3.scaleOrdinal(d3.schemeTableau10)
            .domain(entityTypes);

        // Draw Voronoi cells
        const cells = g.selectAll('.voronoi-cell')
            .data(sites)
            .join('path')
            .attr('class', 'voronoi-cell')
            .attr('d', (d, i) => voronoi.renderCell(i))
            .attr('fill', d => colorScale(d.entity.type))
            .attr('fill-opacity', 0.5)
            .attr('stroke', 'rgba(100, 200, 255, 0.8)')
            .attr('stroke-width', 1.5)
            .style('cursor', 'pointer')
            .on('click', (event, d) => {
                console.log('Clicked entity:', d.entity.name, 'Type:', d.entity.type);
                this.selectEntity(d.entity);
            })
            .on('mouseover', (event, d) => {
                d3.select(event.currentTarget)
                    .attr('fill-opacity', 0.8)
                    .attr('stroke-width', 3);
            })
            .on('mouseout', (event, d) => {
                d3.select(event.currentTarget)
                    .attr('fill-opacity', 0.5)
                    .attr('stroke-width', 1.5);
            });

        // Add entity points (seeds)
        const points = g.selectAll('.voronoi-point')
            .data(sites)
            .join('circle')
            .attr('class', 'voronoi-point')
            .attr('cx', d => d.x)
            .attr('cy', d => d.y)
            .attr('r', 4)
            .attr('fill', d => colorScale(d.entity.type))
            .attr('stroke', '#ffffff')
            .attr('stroke-width', 2)
            .style('pointer-events', 'none');

        // Add labels for larger cells
        const labels = g.selectAll('.voronoi-label')
            .data(sites.filter((d, i) => {
                const cell = voronoi.cellPolygon(i);
                if (!cell) return false;
                // Calculate cell area
                const area = d3.polygonArea(cell);
                return Math.abs(area) > 2000; // Only show labels for larger cells
            }))
            .join('text')
            .attr('class', 'voronoi-label')
            .attr('x', d => d.x)
            .attr('y', d => d.y)
            .attr('text-anchor', 'middle')
            .attr('dy', '0.35em')
            .attr('fill', '#ffffff')
            .attr('font-size', 11)
            .attr('font-weight', '600')
            .attr('pointer-events', 'none')
            .attr('text-shadow', '0 0 4px rgba(0,0,0,0.8)')
            .text(d => {
                const name = d.entity.name || d.entity.title || 'Unknown';
                return name.length > 15 ? name.substring(0, 15) + '...' : name;
            });

        // Add zoom behavior
        const zoom = d3.zoom()
            .scaleExtent([0.5, 10])
            .on('zoom', (event) => {
                g.attr('transform', event.transform);
            });

        svg.call(zoom);

        console.log(`Rendered Voronoi with ${sites.length} cells after Lloyd's relaxation`);
    }

    /**
     * Simple hash function for deterministic positioning
     */
    hashCode(str) {
        let hash = 0;
        for (let i = 0; i < str.length; i++) {
            const char = str.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash;
        }
        return Math.abs(hash);
    }
}

// Export for use
window.GraphRAG3DEmbeddingView = GraphRAG3DEmbeddingView;
