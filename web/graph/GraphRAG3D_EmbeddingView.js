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
        this.mode = 'embedding'; // 'embedding' or 'force'
        this.data = null;
        this.graph = null;
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;

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

        // Color modes
        this.colorMode = 'type'; // 'type' or 'centrality'
        this.centralityColors = {
            low: 0x2196F3,
            mid: 0xFFC107,
            high: 0xF44336
        };
        this.sizeMode = 'connectivity'; // 'connectivity' | 'betweenness'
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

            // Setup controls and interactions
            this.setupControls();
            this.setupKeyboardShortcuts();
            this.setupSearch();
            this.hideEntityInfo();

            // Start rendering
            this.hideLoadingScreen();
            this.animate();

            console.log('✅ GraphRAG 3D Embedding View initialized');
        } catch (error) {
            console.error('❌ Failed to initialize viewer:', error);
            this.updateLoadingStatus(`Error: ${error.message}`);
        }
    }

    /**
     * Load graphrag_hierarchy.json data
     */
    async loadData() {
        try {
            // Try full dataset first, then test sample as fallback
            let dataPath = '/data/graphrag_hierarchy/graphrag_hierarchy.json';
            let response = await fetch(dataPath);

            // If full data fails, try test data
            if (!response.ok) {
                dataPath = '/data/graphrag_hierarchy/graphrag_hierarchy_test_sample.json';
                response = await fetch(dataPath);
            }

            if (!response.ok) {
                throw new Error(`Failed to load data: ${response.statusText}`);
            }

            this.data = await response.json();

            // Process data
            this.processData();

            console.log(`Loaded ${this.entities.size} entities, ${this.relationships.length} relationships`);
        } catch (error) {
            throw new Error(`Data loading failed: ${error.message}`);
        }
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

            this.entities.set(entityId, {
                id: entityId,
                type: entityData.type,
                description: entityData.description || '',
                sources: entityData.sources || [],
                position: position,
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
                this.clusters[key].push({
                    id: clusterId,
                    center: cluster.center || cluster.position || [0, 0, 0],
                    children: cluster.children || [],
                    entities: cluster.entities || []
                });
            }
        }

        // Normalize positions to a consistent scale centered at origin
        this.normalizePositions();

        document.getElementById('total-count').textContent = this.entities.size;
        this.updateHierarchyLabels();
    }

    /**
     * Setup Three.js scene, camera, renderer
     */
    setupScene() {
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
            antialias: true,
            alpha: true
        });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.container.appendChild(this.renderer.domElement);

        // Controls
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.minDistance = Math.max(10, this.boundingRadius * 0.05);
        this.controls.maxDistance = this.boundingRadius * 8;

        // Lights
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        this.scene.add(ambientLight);

        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.4);
        directionalLight.position.set(50, 50, 50);
        this.scene.add(directionalLight);

        // Handle window resize
        window.addEventListener('resize', () => this.onWindowResize());
    }

    /**
     * Create entity node visualizations
     */
    createEntityNodes() {
        const geometry = new THREE.SphereGeometry(1.0, 16, 16);

        for (const [entityId, entity] of this.entities) {
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

        for (const relationship of this.relationships) {
            const sourceEntity = this.entities.get(relationship.source);
            const targetEntity = this.entities.get(relationship.target);

            if (!sourceEntity || !targetEntity) continue;

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
                baseOpacity: opacity
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
        // Create membranes for each hierarchy level
        this.createMembranesForLevel(3, 0.01); // Coarse clusters
        this.createMembranesForLevel(2, 0.02); // Medium clusters
        this.createMembranesForLevel(1, 0.04); // Fine clusters
    }

    /**
     * Create membranes for a specific hierarchy level
     */
    createMembranesForLevel(level, baseOpacity) {
        const levelKey = `level_${level}`;
        const clusters = this.clusters[levelKey];

        for (const cluster of clusters) {
            // Get member entity positions
            const memberPositions = this.getMemberPositions(cluster);

            if (memberPositions.length < 3) continue;

            // Fit ellipsoid to cluster
            const ellipsoid = this.fitEllipsoid(memberPositions);

            // Create Fresnel shader membrane
            const membrane = this.createFresnelMembrane(ellipsoid, baseOpacity, level);
            membrane.userData = {
                clusterId: cluster.id,
                level: level
            };

            this.scene.add(membrane);
            this.clusterMeshes.push(membrane);
        }
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

        // Compute radii (with padding factor)
        const radii = [
            Math.max(Math.sqrt(varX / positions.length) * 2.5, EPSILON),
            Math.max(Math.sqrt(varY / positions.length) * 2.5, EPSILON),
            Math.max(Math.sqrt(varZ / positions.length) * 2.5, EPSILON)
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
    createFresnelMembrane(ellipsoid, baseOpacity, level) {
        // Create ellipsoid geometry
        const geometry = new THREE.SphereGeometry(1, 32, 32);

        // Scale to ellipsoid radii
        geometry.scale(ellipsoid.radii[0], ellipsoid.radii[1], ellipsoid.radii[2]);

        // Fresnel shader material
        const material = new THREE.ShaderMaterial({
            uniforms: {
                ellipsoidCenter: { value: new THREE.Vector3(...ellipsoid.center) },
                ellipsoidRadius: { value: Math.max(...ellipsoid.radii) },
                baseOpacity: { value: baseOpacity },
                edgeOpacity: { value: baseOpacity * 4.0 },
                color: { value: new THREE.Color(0x667eea) }
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
                    float fresnel = pow(1.0 - dot(vNormal, viewDirection), 2.0);

                    // Fade when camera is inside ellipsoid
                    float distanceToCamera = length(cameraPosition - ellipsoidCenter);
                    float insideFade = smoothstep(0.0, ellipsoidRadius * 0.5, distanceToCamera);

                    // Combine effects
                    float opacity = mix(baseOpacity, edgeOpacity, fresnel) * insideFade;

                    gl_FragColor = vec4(color, opacity);
                }
            `,
            transparent: true,
            side: THREE.DoubleSide,
            depthWrite: false
        });

        // Create mesh
        const mesh = new THREE.Mesh(geometry, material);
        mesh.position.set(...ellipsoid.center);

        return mesh;
    }

    /**
     * Setup UI controls and interactions
     */
    setupControls() {
        // Mode buttons
        document.querySelectorAll('.mode-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const mode = e.target.dataset.mode;
                if (mode) {
                    this.setMode(mode);
                }
            });
        });

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

        // Cluster membrane toggle
        document.getElementById('show-clusters').addEventListener('change', (e) => {
            this.clusterMeshes.forEach(mesh => {
                mesh.visible = e.target.checked;
            });
        });
        // Apply default membrane visibility based on checkbox initial state
        const showClustersCheckbox = document.getElementById('show-clusters');
        if (showClustersCheckbox) {
            this.clusterMeshes.forEach(mesh => {
                mesh.visible = showClustersCheckbox.checked;
            });
        }

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
            switch(e.key.toLowerCase()) {
                case 'e':
                    this.setMode('embedding');
                    break;
                case 'f':
                    this.setMode('force');
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
     * Setup search functionality
     */
    setupSearch() {
        const searchInput = document.getElementById('search-input');
        searchInput.addEventListener('input', (e) => {
            const query = e.target.value.toLowerCase();
            // TODO: Implement search with autocomplete
            console.log('Search:', query);
        });
    }

    /**
     * Handle mouse move for hover effects
     */
    onMouseMove(event) {
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
     * Hide entity info panel
     */
    hideEntityInfo() {
        const info = document.getElementById('entity-info');
        if (!info) return;

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

        // Return to embedding view if in force mode
        if (this.mode === 'force') {
            this.setMode('embedding');
        }

        this.updateSelectionHighlight();
    }

    /**
     * Set visualization mode
     */
    setMode(mode) {
        if (mode === 'force') {
            alert('Force graph mode is coming soon. Embedding view stays active for now.');
            mode = 'embedding';
        }
        this.mode = mode;

        // Update UI
        document.querySelectorAll('.mode-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.mode === mode);
        });

        console.log('Mode:', mode);
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

        const labelLimit = 15;
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

        top.forEach(item => {
            const sprite = this.createTextSprite(item.entity.id);
            sprite.position.set(...item.entity.position);
            sprite.scale.set(60, 30, 1);
            this.scene.add(sprite);
            this.labelSprites.push(sprite);
        });
    }

    createTextSprite(text) {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        const fontSize = 96;
        canvas.width = 1024;
        canvas.height = 512;
        ctx.font = `${fontSize}px sans-serif`;
        ctx.fillStyle = '#ffffff';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(text, canvas.width / 2, canvas.height / 2);

        const texture = new THREE.CanvasTexture(canvas);
        texture.needsUpdate = true;

        const material = new THREE.SpriteMaterial({ map: texture, transparent: true, depthWrite: false, depthTest: false });
        const sprite = new THREE.Sprite(material);
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

        this.connectionLines.forEach(line => {
            const sourceVisible = this.visibleEntities.has(line.userData.sourceId);
            const targetVisible = this.visibleEntities.has(line.userData.targetId);
            line.visible = sourceVisible && targetVisible;
        });
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

        this.connectionLines.forEach(line => {
            const baseOpacity = line.userData.baseOpacity || 0.4;
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

        const positions = Array.from(this.entities.values()).map(e => e.position);
        const mins = [Infinity, Infinity, Infinity];
        const maxs = [-Infinity, -Infinity, -Infinity];

        for (const [x, y, z] of positions) {
            mins[0] = Math.min(mins[0], x);
            mins[1] = Math.min(mins[1], y);
            mins[2] = Math.min(mins[2], z);
            maxs[0] = Math.max(maxs[0], x);
            maxs[1] = Math.max(maxs[1], y);
            maxs[2] = Math.max(maxs[2], z);
        }

        const extents = [
            maxs[0] - mins[0],
            maxs[1] - mins[1],
            maxs[2] - mins[2]
        ];

        const maxExtent = Math.max(...extents);
        if (maxExtent === 0 || !isFinite(maxExtent)) return;

        const targetExtent = 1800;
        const scale = targetExtent / maxExtent;
        const center = [
            (mins[0] + maxs[0]) / 2,
            (mins[1] + maxs[1]) / 2,
            (mins[2] + maxs[2]) / 2
        ];

        const transform = (pos) => ([
            (pos[0] - center[0]) * scale,
            (pos[1] - center[1]) * scale,
            (pos[2] - center[2]) * scale
        ]);

        // Apply scaling + mild radial expansion to de-blob PCA layouts
        const halfExtent = targetExtent / 2;
        const radialExponent = 0.45; // <1 pushes midpoints outward more aggressively

        const applyRadialSpread = (pos) => {
            const [x, y, z] = transform(pos);
            const r = Math.sqrt(x * x + y * y + z * z);
            if (r === 0) return [0, 0, 0];

            const normalizedR = Math.min(1, r / halfExtent);
            const scaledR = halfExtent * Math.pow(normalizedR, radialExponent);
            const factor = scaledR / r;
            return [x * factor, y * factor, z * factor];
        };

        // Update entities
        for (const entity of this.entities.values()) {
            entity.position = applyRadialSpread(entity.position);
        }

        // Update cluster centers for higher levels
        for (const level of ['level_1', 'level_2', 'level_3']) {
            this.clusters[level].forEach(cluster => {
                cluster.center = applyRadialSpread(cluster.center || [0, 0, 0]);
            });
        }

        // Record bounding radius for camera/controls
        this.boundingRadius = Math.max(180, (targetExtent / 2));
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

    /**
     * Animation loop
     */
    animate() {
        requestAnimationFrame(() => this.animate());

        // Update controls
        this.controls.update();

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
}

// Export for use
window.GraphRAG3DEmbeddingView = GraphRAG3DEmbeddingView;
