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

        // Selection state
        this.selectedEntity = null;
        this.hoveredEntity = null;
        this.visibleEntities = new Set();

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
            'WORK': 0x795548
        };

        // Entity type visibility
        this.typeVisibility = {};
        Object.keys(this.typeColors).forEach(type => {
            this.typeVisibility[type] = true;
        });
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

            this.updateLoadingStatus('Creating cluster membranes...');
            this.createClusterMembranes();

            // Setup controls and interactions
            this.setupControls();
            this.setupKeyboardShortcuts();
            this.setupSearch();

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
            // First try test data for development
            let dataPath = '/data/graphrag_hierarchy/graphrag_hierarchy_test_sample.json';
            let response = await fetch(dataPath);

            // If test data fails, try full dataset
            if (!response.ok) {
                dataPath = '/data/graphrag_hierarchy/graphrag_hierarchy.json';
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
                    betweenness: entityData.betweenness || 0
                });
            }
            return;
        }

        // Full data format
        const level0 = this.data.clusters.level_0;

        for (const [clusterId, cluster] of Object.entries(level0)) {
            const entityId = cluster.entity;
            const entityData = this.data.entities[entityId];

            if (!entityData) continue;

            // Use UMAP position if available, fallback to PCA
            const position = cluster.umap_position || cluster.position || [0, 0, 0];

            this.entities.set(entityId, {
                id: entityId,
                type: entityData.type,
                description: entityData.description || '',
                sources: entityData.sources || [],
                position: position,
                betweenness: cluster.betweenness || 0,
                clusterId: clusterId
            });
        }

        // Load relationships
        this.relationships = this.data.relationships || [];

        // Process cluster hierarchies
        for (let level = 1; level <= 3; level++) {
            const levelKey = `level_${level}`;
            const levelClusters = this.data.clusters[levelKey];

            if (!levelClusters) continue;

            for (const [clusterId, cluster] of Object.entries(levelClusters)) {
                this.clusters[levelKey].push({
                    id: clusterId,
                    center: cluster.center || cluster.position || [0, 0, 0],
                    children: cluster.children || [],
                    entities: cluster.entities || []
                });
            }
        }

        document.getElementById('total-count').textContent = this.entities.size;
    }

    /**
     * Setup Three.js scene, camera, renderer
     */
    setupScene() {
        // Scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x0a0e1a);
        this.scene.fog = new THREE.Fog(0x0a0e1a, 100, 500);

        // Camera
        this.camera = new THREE.PerspectiveCamera(
            60,
            this.container.clientWidth / this.container.clientHeight,
            0.1,
            1000
        );
        this.camera.position.set(50, 50, 50);
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
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.minDistance = 5;
        this.controls.maxDistance = 300;

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
        const geometry = new THREE.SphereGeometry(0.3, 16, 16);

        for (const [entityId, entity] of this.entities) {
            // Determine color based on type
            const color = this.typeColors[entity.type] || 0xcccccc;

            // Create material
            const material = new THREE.MeshStandardMaterial({
                color: color,
                emissive: color,
                emissiveIntensity: 0.2,
                metalness: 0.3,
                roughness: 0.7
            });

            // Create mesh
            const mesh = new THREE.Mesh(geometry, material);
            mesh.position.set(...entity.position);
            mesh.userData = {
                entityId: entityId,
                entity: entity
            };

            this.scene.add(mesh);
            this.entityMeshes.push(mesh);
            this.visibleEntities.add(entityId);
        }

        this.updateVisibleCount();
    }

    /**
     * Create Fresnel shader cluster membranes
     */
    createClusterMembranes() {
        // Create membranes for each hierarchy level
        this.createMembranesForLevel(3, 0.05); // Coarse clusters
        this.createMembranesForLevel(2, 0.10); // Medium clusters
        this.createMembranesForLevel(1, 0.15); // Fine clusters
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
                cameraPosition: { value: this.camera.position },
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
                uniform vec3 cameraPosition;
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
                this.setMode(mode);
            });
        });

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
    showEntityInfo(entityId) {
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
        document.getElementById('entity-centrality').textContent =
            entity.betweenness.toFixed(4);

        document.getElementById('entity-info').style.display = 'block';
    }

    /**
     * Hide entity info panel
     */
    hideEntityInfo() {
        if (!this.selectedEntity) {
            document.getElementById('entity-info').style.display = 'none';
        }
    }

    /**
     * Select an entity
     */
    selectEntity(entityId) {
        this.selectedEntity = entityId;
        console.log('Selected entity:', entityId);

        // TODO: Implement force-directed neighborhood view
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
    }

    /**
     * Set visualization mode
     */
    setMode(mode) {
        this.mode = mode;

        // Update UI
        document.querySelectorAll('.mode-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.mode === mode);
        });

        console.log('Mode:', mode);
    }

    /**
     * Update entity visibility based on type filters
     */
    updateEntityVisibility() {
        let visibleCount = 0;

        this.entityMeshes.forEach(mesh => {
            const entity = mesh.userData.entity;
            const visible = this.typeVisibility[entity.type];
            mesh.visible = visible;

            if (visible) {
                visibleCount++;
                this.visibleEntities.add(entity.id);
            } else {
                this.visibleEntities.delete(entity.id);
            }
        });

        this.updateVisibleCount();
    }

    /**
     * Update visible entity count in UI
     */
    updateVisibleCount() {
        document.getElementById('visible-count').textContent = this.visibleEntities.size;
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
        this.clusterMeshes.forEach(mesh => {
            if (mesh.material.uniforms) {
                mesh.material.uniforms.cameraPosition.value.copy(this.camera.position);
            }
        });

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
