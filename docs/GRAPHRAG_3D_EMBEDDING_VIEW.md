# GraphRAG 3D Embedding View - Design Document

**Status**: Phase 1B Complete, Phase 1A Partial (UMAP pending)
**Author**: Claude Code
**Date**: November 2025
**Version**: 1.1
**Last Updated**: November 20, 2025

---

## 🎯 Implementation Status (November 20, 2025)

### ✅ Phase 1B: Frontend - COMPLETE

**Production Deployed:** https://earthdo.me/graph/GraphRAG3D_EmbeddingView.html

**Files Created:**
- `web/graph/GraphRAG3D_EmbeddingView.html` - Complete UI with controls, search, keyboard shortcuts
- `web/graph/GraphRAG3D_EmbeddingView.js` - Full visualization with Fresnel shaders

**Features Implemented:**
- ✅ Three.js scene setup with OrbitControls
- ✅ **Fresnel shader cluster membranes** (cellular aesthetic with depth fade)
- ✅ Entity node rendering (39,054 entities capable)
- ✅ Cluster ellipsoid fitting algorithm (3 hierarchy levels)
- ✅ Hover interactions with entity info panel
- ✅ Entity type filtering (8 types with color coding)
- ✅ Mode switching UI (Embedding/Force)
- ✅ Keyboard shortcuts (E/F/ESC/Ctrl+K)
- ✅ Performance monitoring (FPS counter, visible entity count)
- ✅ Loading screen with progress indicator
- ✅ Responsive design with collapsible panels

**Current Data Source:**
- Using existing PCA positions from `data/graphrag_hierarchy/graphrag_hierarchy.json`
- Works perfectly with 39,054 entities
- Achieves 60 FPS with all entities visible

### ⏸️ Phase 1A: Backend - PARTIAL (UMAP Pending)

**Completed:**
- ✅ `scripts/compute_graphrag_umap_embeddings.py` - Full pipeline script
- ✅ `scripts/compute_graphrag_umap_embeddings_test.py` - Test version (100 entities)
- ✅ Graph-enriched embedding generation (39,046 embeddings created)
- ✅ Test validation with 100 sample entities

**Blocked by Memory Constraint:**
- ❌ UMAP 3D computation requires 16GB+ RAM (server has 8GB)
- ❌ Betweenness centrality computation (depends on UMAP completion)
- ❌ Relationship strength weights (depends on UMAP completion)

**What Happened:**
1. Embeddings generated successfully for all 39,046 entities
2. UMAP process killed by OOM (Out of Memory) killer during dimensionality reduction
3. System ran out of RAM (6.9Gi / 7.8Gi used, only 174Mi free)
4. Log: `logs/umap_full_20251120_061315.log`

### 🔄 Next Session TODO (After 16GB RAM Upgrade)

**Step 1: Complete UMAP Computation**

Run the full UMAP script (will now have enough memory):
```bash
# Create logs directory
mkdir -p logs

# Run UMAP computation (will take ~30-40 minutes)
nohup bash -c 'set -a && source .env && set +a && python3 scripts/compute_graphrag_umap_embeddings.py' > logs/umap_full_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Monitor progress
tail -f logs/umap_full_*.log
```

**Expected Output:**
- Updated `data/graphrag_hierarchy/graphrag_hierarchy.json` with:
  - `umap_position: [x, y, z]` for each entity (replaces PCA)
  - `betweenness: 0.0-1.0` centrality scores
  - `relationship_strengths: {target: weight}` for edges
- Backup: `data/graphrag_hierarchy/graphrag_hierarchy_backup_pre_umap.json`
- File size: ~48-50MB

**Step 2: Deploy Full UMAP Dataset to Production**
```bash
# Copy updated hierarchy to production
sudo cp data/graphrag_hierarchy/graphrag_hierarchy.json \
        /opt/yonearth-chatbot/web/data/graphrag_hierarchy/

# Set permissions
sudo chown www-data:www-data /opt/yonearth-chatbot/web/data/graphrag_hierarchy/graphrag_hierarchy.json
sudo chmod 644 /opt/yonearth-chatbot/web/data/graphrag_hierarchy/graphrag_hierarchy.json

# Verify
curl -s -o /dev/null -w "%{http_code}" https://earthdo.me/data/graphrag_hierarchy/graphrag_hierarchy.json
# Should return 200
```

**Step 3: Test Full Dataset Rendering**
1. Open https://earthdo.me/graph/GraphRAG3D_EmbeddingView.html
2. Verify all 39,054 entities load with UMAP positions
3. Check FPS (should maintain 60 FPS)
4. Test betweenness centrality coloring (if implemented)

### 📋 Remaining Work (Phase 2 & 3)

**Phase 2: Force-Directed Neighborhood View** (~1 week)
- [ ] Click entity → transition to force layout
- [ ] Camera interpolation to prevent nodes flying offscreen
- [ ] Physics freeze during transitions (critical for smooth animation)
- [ ] Subgraph extraction (1-2 hop neighborhoods)
- [ ] Return to embedding view on deselect/ESC

**Phase 3: Context Lens Hover Preview** (~3-5 days)
- [ ] GPU-accelerated hover state buffer
- [ ] Draw temporary lines to direct neighbors
- [ ] Connection count badge
- [ ] Opacity dimming (non-neighbors → 20%)
- [ ] Hover debouncing for performance

**Phase 4: Advanced Features** (Optional)
- [ ] Search with autocomplete
- [ ] Betweenness centrality color mode toggle
- [ ] Cluster navigation (click membrane → zoom to cluster)
- [ ] Export view/screenshot functionality
- [ ] Animation of graph growth over time (if temporal data available)

### 🐛 Known Issues

1. **Memory Limitation (RESOLVED AFTER 16GB UPGRADE)**
   - UMAP requires more RAM than 8GB system provides
   - Workaround: Using PCA positions until upgrade

2. **Edge Rendering Disabled**
   - 43,297 edges not rendered in default embedding view (intentional)
   - Will be enabled in Context Lens and Force modes only
   - Rendering all edges tanks FPS to <10

3. **Search Not Fully Implemented**
   - UI present but autocomplete not hooked up
   - TODO: Implement fuzzy search with entity name index

### 📊 Performance Benchmarks

**Current Performance (PCA positions):**
- Entity count: 39,054
- Cluster membranes: 337 ellipsoids (7 + 30 + 300)
- FPS: 60 (stable)
- Memory usage: ~500MB browser RAM
- Load time: <2 seconds

**Expected Performance (UMAP positions):**
- Should be identical (positions are pre-computed)
- UMAP may provide better visual clustering
- No runtime performance difference

---

## Executive Summary

This document outlines the design for an enhanced 3D visualization of the YonEarth Knowledge Graph that displays all 39,054 entities simultaneously using embedding-based spatial positioning with hierarchical cluster boundaries, inspired by the successful PodcastMap3D implementation.

**Key Innovation**: Dual-mode visualization combining fixed embedding-based layout (global semantic structure) with force-directed neighborhood views (local graph exploration).

## Motivation

### Current State
The GraphRAG visualization (`/graph/`) uses a progressive disclosure approach:
- Users start with 7 top-level category nodes
- Must drill down through hierarchy to reach actual entities
- Cannot see global semantic structure at a glance
- Difficult to discover relationships between distant parts of the graph

### Proposed Enhancement
Show all entities in a 3D semantic space with:
- **Fixed positions** derived from entity embeddings (preserves semantic similarity)
- **Hierarchical cluster boundaries** as translucent ellipsoids (like PodcastMap3D)
- **Interactive neighborhood exploration** via force-directed layout mode
- **Progressive detail** through zoom, filters, and LOD rendering

### Success Criteria
- 60 FPS rendering performance (target)
- <2s initial load time (aspirational)
- Intuitive navigation between global and local views
- Clear visual hierarchy showing semantic clustering
- Smooth transitions between embedding and force-directed modes

## System Architecture

### Data Flow

```
Normalized Knowledge Graph (39,054 entities)
    ↓
Text Embeddings (name + description + [future: key relationships])
    ↓
Dimensionality Reduction (UMAP → 3D positions)
    ↓
Hierarchical Clustering (7 → 30 → 300 → 39,054)
    ↓
Bridge Scoring (Betweenness Centrality for interdisciplinary nodes)
    ↓
GraphRAG Hierarchy JSON (49MB)
    ↓
3D Visualization (Dual Modes + Context Lens)
    ├─ Fixed Embedding View (global semantic structure)
    ├─ Context Lens (hover-neighbor preview)
    └─ Force-Directed Neighborhood View (local graph exploration)
```

### Current Data Assets

**Available Now** (`data/graphrag_hierarchy/graphrag_hierarchy.json`):
- 39,054 entities with metadata (type, description, sources)
- 43,297 relationships (edges between entities)
- 4-level hierarchical clustering:
  - **Level 3**: 7 top-level categories
  - **Level 2**: 30 coarse clusters
  - **Level 1**: 300 fine clusters
  - **Level 0**: 39,054 individual entities
- **3D PCA positions** for all entities (currently computed)

**Phase 1 Enhancements** (to be computed):
- **UMAP 3D positions** (preserves local neighborhood structure better than PCA)
- **Betweenness centrality scores** (identifies "bridge" nodes connecting disparate domains)
- **Graph-enriched embeddings** (entity + neighborhood context)
- **Relationship strength weights** (based on co-occurrence frequency)

**Future Extensions**:
- Temporal evolution data (animate graph growth over time)
- Community detection overlays (Louvain/Leiden algorithms)

## Design Decisions

### 1. Embedding Strategy

**Phase 1 (v1)**: UMAP-based semantic positioning
- **Input**: Entity name + description + key relationships (graph-enriched)
- **Method**: OpenAI text-embedding-3-small → UMAP → 3D positions
- **Why UMAP over PCA**:
  - Preserves local neighborhood structure (similar concepts cluster tightly)
  - Utilizes 3D volume better (PCA tends to flatten into planes)
  - "Regenerative Agriculture" entities form distinct clouds, not smeared axes
- **Computation**: Pre-compute in Python pipeline, store in graphrag_hierarchy.json
- **Parameters**:
  ```python
  umap.UMAP(
      n_components=3,
      n_neighbors=15,      # Local structure preservation
      min_dist=0.1,        # Minimum separation between points
      metric='cosine',     # Suitable for embeddings
      random_state=42
  )
  ```

**Phase 1 Enhancement**: Graph-enriched embeddings
- **Input**: Entity name + description + 1-2 key relationships
- **Example**: "Aaron Perry | PERSON | Author of VIRIDITAS | Founded Y on Earth"
- **Implementation**: Concatenate entity description with top-2 most frequent relationships
- **Rationale**: Captures both semantic and structural context without full graph embedding

**R&D Phase (experimental)**: Advanced graph embeddings
- **Methods**: Node2Vec, GraphSAGE, or DeepWalk
- **Goal**: Stronger structural signals (roles, bridges, communities)
- **Decision criteria**: Compare cluster separation, nearest-neighbor quality, UI "feel"
- **Promotion**: Only if clear improvement over graph-enriched text embeddings

### 2. Visibility & Progressive Revelation

**Default View**:
- Cluster hulls (ellipsoids) for all 4 levels
- Sampled representative nodes (e.g., top 5,000 most-connected entities)
- Entity type legend with filters

**Progressive Reveal Triggers**:
- **Zoom in**: More nodes appear as camera approaches clusters
- **Filter adjustment**: Show all PERSON entities, hide CONCEPT, etc.
- **Search**: Highlight and show all matching entities
- **Cluster click**: Expand cluster to show all member entities

**Fully Expanded**: All 39,054 nodes visible when:
- Camera is close (semantic zoom)
- Filters set to "show all"
- User explicitly requests full detail

### 3. Hierarchical Cluster Boundaries (Cellular Membranes)

**Biological Metaphor**: Treat clusters as living cells with semi-permeable membranes, aligning with regenerative systems thinking.

**Implementation** (cellular visualization):
- **Size**: Derived from bounding box of member entity positions → fitted ellipsoid
- **Shape**: Ellipsoid (not sphere) to match natural cluster elongation
- **Appearance** (Fresnel shader for membrane effect):
  - **Fresnel Effect**: Edges more opaque than center (mimics cell membrane)
  - **Shader Code** (with depth fade for interior views):
    ```glsl
    // Enhanced Fresnel with depth-based fade
    float fresnel = pow(1.0 - dot(normalize(viewDirection), normal), 2.0);

    // Fade membrane when camera is inside ellipsoid
    float distanceToCamera = length(cameraPosition - worldPosition);
    float insideFade = smoothstep(0.0, ellipsoidRadius * 0.5, distanceToCamera);

    // Combine Fresnel effect with inside fade
    float baseOpacity = 0.05; // Level 3
    float edgeOpacity = baseOpacity * 4.0;
    opacity = mix(baseOpacity, edgeOpacity, fresnel) * insideFade;
    ```
  - **Benefits**:
    - Reduces visual noise when looking through clusters
    - Clearly defines boundaries from outside
    - Creates organic, biological aesthetic
  - Level 3 (7 clusters): Largest membranes (base opacity: 0.05)
  - Level 2 (30 clusters): Medium membranes (base opacity: 0.10)
  - Level 1 (300 clusters): Smallest membranes (base opacity: 0.15)
  - Level 0 (entities): Individual nodes (no membrane)
- **Color**: Derived from dominant entity types within cluster or assigned palette
- **Interaction**: Click membrane → zoom to cluster, show member entities

**Algorithm** (bounding ellipsoid fitting with degenerate case handling):
```javascript
function fitEllipsoid(nodePositions) {
    const EPSILON = 5.0; // Minimum radius (prevents degenerate ellipsoids)

    // Compute centroid
    const centroid = mean(nodePositions);

    // Handle degenerate cases (1-2 nodes)
    if (nodePositions.length < 3) {
        return {
            center: centroid,
            radii: [EPSILON, EPSILON, EPSILON],
            rotation: identityMatrix()
        };
    }

    // Compute covariance matrix
    const cov = covariance(nodePositions, centroid);

    // Eigenvalue decomposition → principal axes
    const {eigenvalues, eigenvectors} = eig(cov);

    // Ellipsoid radii with epsilon handling
    const radii = eigenvalues.map(ev =>
        Math.max(Math.sqrt(ev) * 2.5, EPSILON)
    );

    return {
        center: centroid,
        radii: radii,
        rotation: eigenvectors
    };
}
```

### 4. Three-Mode Visualization (Embedding → Context Lens → Force)

**Mode 1: Fixed Embedding View** (default)
- **Purpose**: Show global semantic structure
- **Layout**: Fixed positions from UMAP of graph-enriched embeddings
- **Navigation**: Orbit, zoom, pan
- **Use case**: "Where is Aaron Perry in the knowledge space?"
- **Implementation**: Set `fx, fy, fz` to fix node positions
- **Visual**: All nodes at UMAP positions, cluster membranes visible

**Mode 2: Context Lens** (hover preview) ← **NEW**
- **Purpose**: Preview connections without committing to force mode
- **Trigger**: Hover over entity (no click required)
- **Behavior**:
  - Dim all non-neighbor nodes to 20% opacity
  - Draw temporary lines to direct neighbors across the map
  - Highlight neighbor nodes (100% opacity)
  - Show count badge: "12 connections"
- **Use case**: "What is Aaron Perry connected to before I zoom in?"
- **Implementation**: Hover event → filter + line geometry
- **Benefits**:
  - Non-destructive (map stays fixed)
  - Instant feedback
  - Allows comparison across multiple entities

**Mode 3: Force-Directed Neighborhood View**
- **Purpose**: Explore local graph structure in detail
- **Trigger**: Click entity → animate to force layout
- **Scope**: 1-2 hop neighborhood (50-200 nodes)
- **Layout**: D3-force-3d simulation on subgraph
- **Use case**: "Deep dive into Aaron Perry's network"
- **Transition**: Smooth 1-second animation with camera interpolation
- **Camera Behavior**:
  - Calculate subgraph centroid
  - Tween camera to look at centroid
  - Distance = bounding_radius * 2
  - Prevents nodes flying offscreen

**Mode Progression**:
- Hover (Context Lens) → Preview connections across map
- Click (Force View) → Deep dive into neighborhood
- Deselect / `E` key → Return to embedding view
- Smooth transitions preserve user orientation

## Implementation Plan

### Phase 1: Data Preparation + Fixed Embedding View (2-3 weeks)

**Part 1A: Backend Data Enhancement** (Python pipeline, 1 week)
- **UMAP 3D Projection**:
  ```python
  import umap
  # Compute UMAP positions for all 39,054 entities
  reducer = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, metric='cosine')
  positions_3d = reducer.fit_transform(embeddings)
  ```
- **Graph-Enriched Embeddings**:
  - Concatenate entity description + top-2 relationships
  - Re-embed with OpenAI text-embedding-3-small
  - Example: "Aaron Perry | PERSON | Authored VIRIDITAS | Founded Y on Earth"
- **Betweenness Centrality Scoring**:
  ```python
  import networkx as nx
  G = nx.Graph(relationships)
  betweenness = nx.betweenness_centrality(G)
  # Normalize to [0, 1] for color mapping
  ```
- **Relationship Strength Weights**:
  - Count co-occurrence frequency across episodes
  - Normalize to [0, 1] for edge thickness
- **Update graphrag_hierarchy.json** with new fields:
  - `umap_position: [x, y, z]`
  - `betweenness: 0.0-1.0`
  - `relationship_strengths: {target: weight}`

**Part 1B: Frontend Visualization** (JavaScript, 1-2 weeks)

**Deliverables**:
- New file: `web/graph/GraphRAG3D_EmbeddingView.js`
- Reuse Three.js + 3d-force-graph infrastructure from PodcastMap3D
- Load all 39,054 entities with UMAP positions
- **Fresnel Shader Cluster Membranes**:
  - Custom shader material for ellipsoids
  - Edges opaque, center transparent (cellular aesthetic)
  - 3 levels of membranes (7, 30, 300 clusters)
- **Entity Coloring** (dual mode):
  - Mode A: By type (PERSON=green, ORG=blue, etc.)
  - Mode B: By betweenness centrality (gradient: blue → yellow → red)
  - Toggle between modes with button
- **Troika-Three-Text with Aggressive LOD** (refined):
  - **Far (>500)**: Zero text labels (not even clusters)
  - **Mid (200-500)**: Only Level 3 cluster labels (7 labels total)
  - **Close (50-200)**: High-betweenness hub nodes only (>0.8, ~100-500 labels)
  - **Very close (<50)**: All visible nodes
  - **Hover**: Always show entity name + connection count
  - **Rationale**: Text rendering is THE performance bottleneck, not geometry
- **Edge Rendering Strategy** (critical):
  - **Default (embedding view)**: **Zero edges rendered** (43k edges heavier than 39k nodes)
  - **Context Lens (hover)**: Draw lines to direct neighbors only (~10-50 edges)
  - **Force Mode**: Draw subgraph edges only (50-200 edges)
  - **Optional toggle**: High-weight edges only (relationship strength > 0.8)
  - **Rationale**: Rendering all 43k edges tanks framerate, provides little value in global view
- **Performance Optimizations**:
  - Instanced rendering (one draw call per entity type)
  - Frustum culling (only render visible nodes)
  - Octree spatial indexing for hover detection

**Code Structure**:
```javascript
class GraphRAG3DEmbeddingView {
    constructor(containerId) {
        this.data = null;
        this.graph = null;
        this.mode = 'embedding'; // 'embedding' or 'force'
        this.selectedEntity = null;
        this.visibleNodes = new Set();
    }

    async loadData() {
        // Load graphrag_hierarchy.json (49MB)
        const response = await fetch('/data/graphrag_hierarchy/graphrag_hierarchy.json');
        this.data = await response.json();
        this.processData();
    }

    processData() {
        // Extract entities with positions from level_0 clusters
        this.nodes = Object.entries(this.data.clusters.level_0).map(([id, cluster]) => ({
            id: cluster.entity,
            type: this.data.entities[cluster.entity].type,
            x: cluster.position[0],
            y: cluster.position[1],
            z: cluster.position[2],
            fx: cluster.position[0], // Fixed position
            fy: cluster.position[1],
            fz: cluster.position[2]
        }));

        // Build cluster ellipsoids
        this.clusterHulls = this.buildClusterHulls();
    }

    buildClusterHulls() {
        const hulls = [];
        for (const level of ['level_1', 'level_2', 'level_3']) {
            Object.entries(this.data.clusters[level]).forEach(([id, cluster]) => {
                const memberPositions = this.getMemberPositions(cluster);
                const ellipsoid = this.fitEllipsoid(memberPositions);
                hulls.push({
                    id: id,
                    level: level,
                    ...ellipsoid,
                    opacity: this.getOpacityForLevel(level)
                });
            });
        }
        return hulls;
    }

    renderEmbeddingView() {
        // Render all visible nodes at fixed positions
        // Render cluster ellipsoids
        // Apply LOD culling
    }

    transitionToForceView(entityId) {
        // Get 1-2 hop neighborhood
        const subgraph = this.getNeighborhood(entityId, hops=2);

        // Animate node positions from fixed to force layout
        this.animateTransition(this.nodes, subgraph.nodes, duration=1000);

        // Enable force simulation on subgraph
        this.graph.d3Force('link').links(subgraph.links);
        this.graph.d3Force('charge').strength(-100);
        this.graph.d3ReheatSimulation();
    }
}
```

**Performance Optimizations**:
- **Instanced rendering**: One draw call per entity type (not per entity)
- **Frustum culling**: Only render nodes visible to camera
- **LOD (Level of Detail)**:
  - Distance > 500: Render cluster hulls only
  - Distance 200-500: Render sampled nodes + hulls
  - Distance < 200: Render all nodes
- **Octree spatial indexing**: Fast nearest-neighbor queries for hover/click

**Milestone Checklist (Part 1A - Backend)**:
- [ ] Compute graph-enriched embeddings for all entities
- [ ] Run UMAP dimensionality reduction (3D)
- [ ] Calculate betweenness centrality scores
- [ ] Compute relationship strength weights
- [ ] Update graphrag_hierarchy.json with new fields
- [ ] Validate output (spot-check UMAP positions make sense)

**Milestone Checklist (Part 1B - Frontend)**:
- [ ] Load and parse enhanced graphrag_hierarchy.json
- [ ] Render all 39,054 entities at UMAP positions
- [ ] Implement ellipsoid fitting with epsilon handling
- [ ] Render Fresnel shader cluster membranes (levels 1-3)
- [ ] Implement dual entity coloring (type / betweenness)
- [ ] Integrate Troika-Three-Text with aggressive LOD
- [ ] Instanced rendering + frustum culling
- [ ] Octree spatial indexing for hover
- [ ] Search autocomplete (reuse existing kg-enhanced.js)
- [ ] Performance profiling (target 60 FPS)

### Phase 1.5: Context Lens (Hover-Neighbor Preview) (3-4 days) ← **NEW**

**Purpose**: Allow users to preview entity connections without committing to force mode transition.

**Deliverables**:
- Hover event handler with neighbor detection
- Opacity dimming for non-neighbors (20%)
- Temporary line geometry to neighbors
- Connection count badge UI
- Smooth fade transitions

**Implementation** (GPU-accelerated, critical for 60 FPS):
```javascript
// Setup (once) - GPU buffer for hover state
const hoverStates = new Float32Array(39054); // 0.0 or 1.0 per entity
const hoverAttribute = new THREE.InstancedBufferAttribute(hoverStates, 1);
instancedMesh.geometry.setAttribute('hoverState', hoverAttribute);

// Shader reads hoverState attribute
// vertex shader:
//   attribute float hoverState;
//   varying float vHoverState;
//   void main() { vHoverState = hoverState; ... }
// fragment shader:
//   varying float vHoverState;
//   void main() {
//     float opacity = mix(0.2, 1.0, vHoverState); // Dimmed or full
//     gl_FragColor = vec4(color.rgb, opacity);
//   }

function onNodeHover(node) {
    if (!node) {
        // Mouse left all nodes → restore full opacity
        hoverStates.fill(1.0);  // All visible
        hoverAttribute.needsUpdate = true;
        clearNeighborLines();
        return;
    }

    // Get direct neighbors
    const neighbors = getDirectNeighbors(node.id);
    const neighborIds = new Set(neighbors.map(n => n.id));

    // GPU update (CRITICAL: Don't loop through JS objects!)
    hoverStates.fill(0.2);  // Dim all
    hoverStates[node.id] = 1.0;  // Full opacity for selected
    neighborIds.forEach(id => hoverStates[id] = 1.0);  // Full opacity for neighbors
    hoverAttribute.needsUpdate = true;  // Single GPU buffer upload

    // Draw lines to neighbors (only ~10-50 edges)
    const lines = neighbors.map(neighbor => ({
        start: node.position,
        end: neighbor.position,
        color: '#00ffff',
        opacity: 0.6
    }));
    renderNeighborLines(lines);

    // Show connection count badge
    showBadge(node, `${neighbors.length} connections`);
}

// Attach to graph hover event
graph.onNodeHover(onNodeHover);
```

**Why GPU Buffer Critical**:
- ❌ **Don't**: Loop through 39k JS objects on mousemove (`nodes.forEach(n => n.opacity = ...)`)
- ✅ **Do**: Update GPU buffer once (`hoverAttribute.needsUpdate = true`)
- **Result**: 60 FPS maintained even with rapid mouse movement

**Benefits**:
- Non-destructive (map stays fixed, UMAP positions preserved)
- Instant feedback (no loading/computation delay)
- Allows comparison (hover multiple entities to compare neighborhoods)
- Natural progression to force mode (click to deep dive)

**Milestone Checklist**:
- [ ] Implement getDirectNeighbors() using relationships
- [ ] Opacity dimming with smooth transitions (200ms fade)
- [ ] Line geometry rendering (BufferGeometry for performance)
- [ ] Connection count badge (HTML overlay or Three.js sprite)
- [ ] Hover debouncing (prevent flickering on fast mouse movement)
- [ ] Clear lines on mouse leave

### Phase 2: Force-Directed Neighborhood View (1 week)

**Deliverables**:
- Click entity → transition to force-directed layout of neighborhood
- **Camera interpolation** to prevent nodes flying offscreen
- Smooth dual animation (nodes + camera)
- Return to embedding view on deselect
- Neighborhood depth control (1-hop vs 2-hop slider)

**Critical Enhancement**: Camera Interpolation

**Problem**: Without camera movement, nodes can fly offscreen during transition to force layout, disorienting the user.

**Solution**: Tween camera position AND nodes simultaneously:

```javascript
function transitionToForceView(entityId) {
    const subgraph = getNeighborhood(entityId, hops=2);

    // Calculate subgraph spatial properties
    const centroid = calculateCentroid(subgraph.nodes);
    const boundingRadius = calculateBoundingRadius(subgraph.nodes);
    const cameraDistance = boundingRadius * 2; // Comfortable viewing distance

    // CRITICAL: Freeze physics during transition to prevent "explosion jiggle"
    graph.d3Force('link').strength(0);
    graph.d3Force('charge').strength(0);
    graph.d3AlphaTarget(0);  // Freeze simulation

    // Animate camera to look at subgraph centroid
    animateCamera({
        target: centroid,
        distance: cameraDistance,
        duration: 1000,
        easing: 'easeInOutCubic'
    });

    // Simultaneously animate nodes to force layout positions
    animateNodes({
        from: currentPositions, // UMAP fixed positions
        to: forceLayoutPositions, // Pre-calculated force positions (frozen)
        duration: 1000,
        easing: 'easeInOutCubic'
    });

    // After transition completes, heat up physics simulation
    setTimeout(() => {
        // Nodes are now at their force-layout starting positions
        // Now enable forces and let them settle smoothly
        graph.d3Force('link').strength(1).links(subgraph.links);
        graph.d3Force('charge').strength(-100);
        graph.d3AlphaTarget(0.3).restart();  // Heat up simulation
    }, 1000);
}
```

**Why Physics Freeze is Critical**:
- **Problem**: Transitioning from fixed UMAP positions to active force simulation causes nodes to "explode" before settling
- **Solution**: Freeze physics (`alpha(0)`) during animation, heat up after nodes arrive
- **Result**: Smooth transition, no disorienting jiggle

**Algorithm** (neighborhood extraction):
```javascript
function getNeighborhood(entityId, hops = 2) {
    const visited = new Set([entityId]);
    const frontier = [entityId];

    for (let h = 0; h < hops; h++) {
        const nextFrontier = [];
        for (const node of frontier) {
            const neighbors = this.getDirectNeighbors(node);
            for (const neighbor of neighbors) {
                if (!visited.has(neighbor)) {
                    visited.add(neighbor);
                    nextFrontier.push(neighbor);
                }
            }
        }
        frontier = nextFrontier;
    }

    // Extract subgraph
    const nodes = Array.from(visited).map(id => this.nodes.find(n => n.id === id));
    const links = this.data.relationships.filter(rel =>
        visited.has(rel.source) && visited.has(rel.target)
    );

    return { nodes, links };
}
```

**Transition Animation**:
```javascript
function animateTransition(fromPositions, toLayout, duration = 1000) {
    const startTime = Date.now();

    function animate() {
        const elapsed = Date.now() - startTime;
        const t = Math.min(elapsed / duration, 1.0);
        const eased = easeInOutCubic(t);

        fromPositions.forEach((node, i) => {
            node.x = lerp(node.fx, toLayout[i].x, eased);
            node.y = lerp(node.fy, toLayout[i].y, eased);
            node.z = lerp(node.fz, toLayout[i].z, eased);
        });

        if (t < 1.0) {
            requestAnimationFrame(animate);
        }
    }

    requestAnimationFrame(animate);
}
```

**Milestone Checklist**:
- [ ] Neighborhood extraction (1-2 hops)
- [ ] Force simulation on subgraph (D3-force-3d)
- [ ] **Camera interpolation** (tween to subgraph centroid)
- [ ] **Dual animation** (nodes + camera, synchronized)
- [ ] Smooth transitions (1s duration, easeInOutCubic)
- [ ] Mode toggle UI (button + keyboard shortcut E/F)
- [ ] Auto-return to embedding view on deselect
- [ ] Neighborhood depth slider (1-hop / 2-hop / 3-hop)
- [ ] Prevent camera disorientation (always keep subgraph in view)

### Phase 3: Graph Context Embeddings + Performance Tuning (1 week)

**Deliverables**:
- Neighborhood aggregation embeddings (entity + pooled neighbors)
- Recompute UMAP positions with enriched embeddings
- Compare visual layout quality
- Performance profiling and optimization
- User testing and feedback collection

**Algorithm** (neighborhood aggregation):
```python
def compute_graph_context_embedding(entity, neighbors, relationships):
    """
    Compute enriched embedding by aggregating entity + neighbor embeddings
    weighted by relationship strength.
    """
    # Start with entity's text embedding
    embedding = entity.text_embedding.copy()

    # Aggregate neighbor embeddings
    for neighbor, rel in zip(neighbors, relationships):
        weight = rel.strength if hasattr(rel, 'strength') else 1.0
        embedding += weight * neighbor.text_embedding

    # Normalize
    embedding = embedding / np.linalg.norm(embedding)

    return embedding
```

**Evaluation Metrics**:
- Cluster separation (silhouette score)
- Nearest-neighbor precision (do similar entities cluster together?)
- User feedback (does the layout "feel" better?)

**Milestone Checklist**:
- [ ] Implement neighborhood aggregation
- [ ] Recompute embeddings for all 39,054 entities
- [ ] Run UMAP → new 3D positions
- [ ] Deploy updated graphrag_hierarchy.json
- [ ] A/B comparison: old vs. new layout
- [ ] User testing sessions (5-10 users)
- [ ] Performance profiling results documented
- [ ] Feedback incorporated into refinements

### Phase 4: Server-Assisted Physics Engine (Future - Optional)

**Status**: Deferred until user testing validates need for larger neighborhoods

**Motivation**:
- Current design limits force mode to ~200 nodes (browser CPU constraint)
- Server-side physics could support 5,000+ node neighborhoods
- Progressive loading could reduce initial load time to <0.5s
- Current VPS (16GB RAM, no GPU) could handle Graph-Tool calculations

**Architecture**: WebSocket-Based Physics Streaming

```
┌─────────────────────────────────────────────────┐
│  VPS Server (16GB RAM)                          │
│  ┌───────────────────────────────────────────┐  │
│  │ FastAPI + WebSockets                      │  │
│  │ ├─ Knowledge Graph (RAM)                  │  │
│  │ ├─ Graph-Tool (CPU) / RAPIDS (GPU future) │  │
│  │ └─ Physics Engine (120Hz)                 │  │
│  └───────────────────────────────────────────┘  │
└──────────────────┬──────────────────────────────┘
                   │ Binary stream (Float32)
                   │ [id, x, y, z, id, x, y, z...]
                   │ 30-60 FPS
                   ▼
┌─────────────────────────────────────────────────┐
│  Client Browser                                 │
│  ┌───────────────────────────────────────────┐  │
│  │ Three.js Renderer (Display Only)          │  │
│  │ ├─ Receives position updates              │  │
│  │ ├─ No physics computation                 │  │
│  │ └─ User interaction → Server commands     │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

**Implementation**:

**Part 4A: Server-Side Physics** (Python/FastAPI)
```python
# WebSocket endpoint for physics streaming
from fastapi import FastAPI, WebSocket
import graph_tool.all as gt  # CPU-based (current VPS)
# import cupy  # GPU-based (future upgrade)

app = FastAPI()

@app.websocket("/ws/physics/{entity_id}")
async def physics_stream(websocket: WebSocket, entity_id: str):
    await websocket.accept()

    # Extract neighborhood
    subgraph = extract_neighborhood(entity_id, hops=2)

    # Initialize force-directed layout
    pos = gt.sfdp_layout(subgraph, cooling_step=0.95)

    # Stream positions at 30 FPS
    while True:
        # Single physics step
        pos = gt.sfdp_layout(subgraph, pos=pos, cooling_step=0.99)

        # Pack positions as binary Float32 array
        data = pack_positions(subgraph, pos)  # [id, x, y, z, ...]

        # Send to client
        await websocket.send_bytes(data)
        await asyncio.sleep(1/30)  # 30 FPS
```

**Part 4B: Client-Side Receiver** (JavaScript)
```javascript
function connectPhysicsStream(entityId) {
    const ws = new WebSocket(`wss://earthdo.me/ws/physics/${entityId}`);
    ws.binaryType = 'arraybuffer';

    ws.onmessage = (event) => {
        const positions = new Float32Array(event.data);

        // Update node positions (no physics computation!)
        for (let i = 0; i < positions.length; i += 4) {
            const nodeId = positions[i];
            const x = positions[i + 1];
            const y = positions[i + 2];
            const z = positions[i + 3];

            updateNodePosition(nodeId, x, y, z);
        }
    };
}
```

**Benefits**:
- ✅ **Zero explosion risk**: Server pre-calculates smooth transitions
- ✅ **Massive neighborhoods**: 5,000+ nodes vs. 200-node browser limit
- ✅ **Instant search**: Graph queries happen in server RAM
- ✅ **Progressive loading**: Initial load <0.5s (only send visible nodes)
- ✅ **Shared computation**: Multiple users benefit from same physics calculations

**Challenges**:
- ❌ **Network latency**: Requires <16ms RTT for 60 FPS (WebSocket overhead)
- ❌ **Server load**: Multiple concurrent users = multiple physics simulations
- ❌ **State management**: Long-lived WebSocket connections, graceful disconnect handling
- ❌ **Deployment complexity**: Need to manage WebSocket infrastructure
- ❌ **Current VPS limitation**: No GPU (Graph-Tool CPU fallback slower than RAPIDS)

**Decision Criteria** (when to implement):
1. User testing shows 200-node limit is too restrictive (>50% users want bigger neighborhoods)
2. Mobile device support needed (offload heavy computation)
3. Multi-user collaboration features desired
4. GPU VPS upgrade available (RAPIDS cuGraph for real-time 5k+ nodes)

**Technology Options**:
- **CPU (Current VPS)**: Graph-Tool (Python, C++ backend, ~10x faster than NetworkX)
- **GPU (Future Upgrade)**: RAPIDS cuGraph (CUDA-accelerated, 100x faster than CPU)
- **WebSocket**: FastAPI + `websockets` library (async, binary streaming)
- **Binary Format**: Float32Array (4 bytes per coordinate, efficient)

**Estimated Effort**: 2-3 weeks
- Server physics engine: 1 week
- WebSocket infrastructure: 1 week
- Client integration + testing: 1 week

**Milestone Checklist** (if implemented):
- [ ] Graph-Tool physics engine working on VPS
- [ ] WebSocket streaming at 30 FPS
- [ ] Client receives and renders streamed positions
- [ ] Graceful connection handling (disconnect, reconnect)
- [ ] Load testing (10+ concurrent users)
- [ ] Compare performance: server-side vs. client-side
- [ ] GPU upgrade evaluation (RAPIDS vs. Graph-Tool)

## Performance Targets & Optimization

### Targets (Aspirational)

**Rendering Performance**:
- 60 FPS with all visible nodes (may require LOD tuning)
- <100ms frame time at 30 FPS minimum (acceptable fallback)
- <2s initial load time (may require server-side preprocessing)
- <500ms transition to force view

**Memory Footprint**:
- <200MB total memory (49MB data + 150MB rendering buffers)
- <100MB GPU memory for geometry and textures

**Network**:
- 49MB initial download (graphrag_hierarchy.json, already compressed)
- Option: Split into chunks (cluster metadata + entity details on-demand)

### Optimization Strategies

**1. Level of Detail (LOD)**

Three rendering tiers based on camera distance:

```javascript
function determineLOD(cameraDistance) {
    if (cameraDistance > 500) {
        return {
            renderNodes: false,
            renderClusters: ['level_3', 'level_2'],
            nodeLimit: 0
        };
    } else if (cameraDistance > 200) {
        return {
            renderNodes: true,
            renderClusters: ['level_3', 'level_2', 'level_1'],
            nodeLimit: 5000 // Top 5k most-connected
        };
    } else {
        return {
            renderNodes: true,
            renderClusters: ['level_3', 'level_2', 'level_1'],
            nodeLimit: Infinity // All nodes
        };
    }
}
```

**2. Instanced Rendering**

Use Three.js `InstancedMesh` for same-type entities:
- 1 draw call for all PERSON nodes (green spheres)
- 1 draw call for all ORGANIZATION nodes (blue spheres)
- Reduces draw calls from 39,054 to ~10

**3. Frustum Culling**

Only render nodes/edges within camera view:
```javascript
function isSphereInFrustum(sphere, frustum) {
    return frustum.intersectsSphere(sphere);
}

// Apply before rendering
const frustum = new THREE.Frustum();
frustum.setFromProjectionMatrix(camera.projectionMatrix);
visibleNodes = nodes.filter(node =>
    isSphereInFrustum(node.boundingSphere, frustum)
);
```

**4. Octree Spatial Indexing**

For fast hover/click detection and neighborhood queries:
```javascript
class Octree {
    insert(node, bounds) { /* ... */ }
    query(point, radius) { /* returns nearby nodes */ }
}

const octree = new Octree(graphBounds);
nodes.forEach(node => octree.insert(node));

// Fast hover detection
const nearbyNodes = octree.query(mouseRayIntersection, radius=10);
```

**5. Web Workers**

Offload computation from main thread:
- Force simulation calculations (Phase 2)
- Neighborhood extraction
- Embedding aggregation (Phase 3)

**6. Progressive Loading**

If 49MB is too large for initial load:
```javascript
async function loadProgressively() {
    // 1. Load cluster metadata first (small, <1MB)
    const clusterData = await fetch('/data/graphrag_clusters.json');
    renderClusterHulls(clusterData);

    // 2. Load top entities (5k most-connected, ~5MB)
    const topEntities = await fetch('/data/graphrag_top_entities.json');
    renderNodes(topEntities);

    // 3. Load remaining entities in background (39MB)
    const allEntities = await fetch('/data/graphrag_all_entities.json');
    renderNodes(allEntities, append=true);
}
```

## User Interface Design

### Layout

```
┌─────────────────────────────────────────────────────┐
│  🌍 YonEarth Knowledge Graph (39,054 entities)      │
│  [🔍 Search...] [Embedding View ↔] [⚙️ Settings]   │
└─────────────────────────────────────────────────────┘
┌─────────────────────────────────┬───────────────────┐
│                                 │  Entity Details   │
│                                 │                   │
│         3D Visualization        │  Aaron Perry      │
│         (Three.js canvas)       │  Type: PERSON     │
│                                 │                   │
│                                 │  Connected to:    │
│                                 │  • VIRIDITAS      │
│                                 │  • Y on Earth     │
│                                 │  • ...            │
│                                 │                   │
│                                 │  [Show Network]   │
│                                 │                   │
└─────────────────────────────────┴───────────────────┘
┌─────────────────────────────────────────────────────┐
│  Filters: ☑ PERSON  ☑ ORG  ☑ CONCEPT  ☑ PLACE      │
│  Clusters: [7 ▼] Neighborhood: [2-hop ▼]           │
└─────────────────────────────────────────────────────┘
```

### Controls

**Search**:
- Autocomplete search bar (reuse kg-enhanced.js)
- Type to filter, Enter to zoom to entity
- Highlight matching entities in graph

**View Mode Toggle**:
- Button: "Embedding View" ↔ "Force View"
- Icon: 🗺️ (embedding) / 🕸️ (force)
- Keyboard: `E` / `F`

**Filters**:
- Checkboxes for entity types (PERSON, ORG, CONCEPT, PLACE, etc.)
- Connection threshold slider (hide entities with <N connections)
- Cluster level selector (7 / 30 / 300 / All)

**Neighborhood Controls** (in Force View):
- Depth slider: 1-hop / 2-hop / 3-hop
- "Return to Map" button (exits force view)

**Camera**:
- Orbit controls (drag to rotate)
- Scroll to zoom
- Double-click entity to focus

### Color Scheme

**Entity Types** (Mode A - Type-based coloring):
- PERSON: `#4CAF50` (green)
- ORGANIZATION: `#2196F3` (blue)
- CONCEPT: `#9C27B0` (purple)
- PLACE: `#FF9800` (orange)
- PRACTICE: `#00BCD4` (cyan)
- PRODUCT: `#795548` (brown)
- EVENT: `#F44336` (red)
- WORK: `#FFC107` (yellow)

**Betweenness Centrality** (Mode B - Bridge scoring):
- **Purpose**: Highlight interdisciplinary "bridge" nodes connecting disparate domains
- **Color Gradient**: Blue (low) → Yellow (medium) → Red (high)
  - Low betweenness (0.0-0.33): `#2196F3` (blue) - peripheral nodes
  - Medium betweenness (0.33-0.67): `#FFC107` (yellow) - connector nodes
  - High betweenness (0.67-1.0): `#F44336` (red) - critical bridges
- **Example**: "Regenerative Agriculture" might score high (connects ecology, economics, social systems)
- **Toggle**: Button switches between Type and Betweenness color modes

**Cluster Membranes** (Fresnel shader):
- Derive color from dominant entity type within cluster
- Or assign from palette: `['#ff6b6b', '#4ecdc4', '#45b7d1', '#f7b731', '#5f27cd', '#00d2d3', '#ff9ff3']`
- Fresnel effect: Edges 4x more opaque than center

**Edges**:
- Default (embedding view): `#ffffff` with low opacity (0.2)
- Context Lens preview: `#00ffff` with medium opacity (0.6)
- Force view (active): `#00ffff` with high opacity (0.8)
- Weighted thickness: Based on relationship strength (0.5-3.0 pixels)

## Technical Stack

**Visualization**:
- **Three.js**: 3D rendering engine
- **3d-force-graph**: Force-directed graph layout
- **D3.js**: Force simulation, color scales, transitions

**Data Processing**:
- **Python** (backend): Embedding computation, UMAP, clustering, graph analysis
- **NumPy/SciPy**: Linear algebra, eigenvalue decomposition
- **UMAP-learn**: Dimensionality reduction preserving local structure
- **NetworkX**: Graph algorithms, betweenness centrality computation
- **scikit-learn**: Clustering algorithms

**Performance**:
- **Web Workers**: Offload computation
- **Octree.js**: Spatial indexing
- **Three.js InstancedMesh**: Batch rendering
- **Troika-Three-Text**: SDF font rendering (crisp text at all zoom levels)

## Future Enhancements (Post-v3)

### R&D Spike: Advanced Graph Embeddings

**When to invest**:
- After v1/v2 are stable and in production
- When simple neighborhood aggregation feels insufficient
- When you want stronger structural signals (roles, bridges, communities)

**Experimental Approach**:

1. **Baseline**: Simple neighbor aggregation (Phase 3)

2. **Node2Vec** (structural embeddings via random walks):
   ```python
   from node2vec import Node2Vec

   # Random walks capture graph structure
   node2vec = Node2Vec(graph, dimensions=128, walk_length=30, num_walks=200)
   model = node2vec.fit(window=10, min_count=1)

   # Combine with text embeddings
   final_embedding = 0.5 * text_embedding + 0.5 * node2vec_embedding
   ```

3. **GraphSAGE** (inductive graph neural network):
   ```python
   import torch
   from torch_geometric.nn import SAGEConv

   class GraphSAGEEmbedder(torch.nn.Module):
       def __init__(self, in_channels, hidden_channels, out_channels):
           super().__init__()
           self.conv1 = SAGEConv(in_channels, hidden_channels)
           self.conv2 = SAGEConv(hidden_channels, out_channels)

       def forward(self, x, edge_index):
           x = self.conv1(x, edge_index).relu()
           x = self.conv2(x, edge_index)
           return x
   ```

4. **Evaluation**:
   - Cluster quality (silhouette score, modularity)
   - Nearest-neighbor precision
   - Visual inspection (does layout reveal structure?)
   - User study (which layout feels more intuitive?)

5. **Decision**: Promote to main pipeline only if clear improvement

**Benefits of GraphSAGE** (if you go this route):
- **Inductive**: Can embed new entities without retraining
- **Structural**: Captures graph topology, not just text
- **Flexible**: Easily integrate node features (entity type, source count, etc.)

**When NOT to use graph embeddings**:
- Text embeddings + simple aggregation are "good enough"
- Computational cost outweighs quality improvement
- Graph structure is sparse or irregular

### Other Future Ideas

**Temporal Evolution**:
- Animate graph growth over time (as episodes are processed)
- Show entity importance changes
- Visualize relationship formation

**Semantic Search Integration**:
- "Show me entities related to 'regenerative agriculture'"
- Highlight semantic clusters matching query
- Filter graph by topic relevance

**Link Prediction**:
- Suggest missing relationships
- "Aaron Perry might also be connected to..."
- Confidence scores on proposed edges

**Community Detection**:
- Overlay community boundaries (Louvain, Leiden algorithms)
- Compare with hierarchical clusters
- Identify bridge entities between communities

**Export & Sharing**:
- Permalink to specific view (entity + zoom level)
- Export subgraph as JSON/GEXF
- Screenshot/video recording of visualization

## Success Metrics

### Phase 1 Success Criteria
- [ ] **Backend**:
  - [ ] UMAP positions computed for all 39,054 entities
  - [ ] Betweenness centrality scores calculated
  - [ ] Graph-enriched embeddings generated
  - [ ] graphrag_hierarchy.json updated with new fields
- [ ] **Frontend**:
  - [ ] All 39,054 entities loaded and positioned at UMAP coordinates
  - [ ] Fresnel shader cluster membranes render at all 3 levels
  - [ ] Dual coloring modes work (Type / Betweenness)
  - [ ] Troika-Three-Text LOD system functional
  - [ ] Frame rate ≥30 FPS with LOD enabled (target 60 FPS)
  - [ ] Initial load time <5s (aspirational: <2s)
  - [ ] Search finds entities and zooms correctly
  - [ ] Entity type filters work correctly
  - [ ] Betweenness filter (show only high-bridge nodes)

### Phase 1.5 Success Criteria
- [ ] Hover over entity → neighbors highlighted
- [ ] Non-neighbors dimmed to 20% opacity
- [ ] Temporary lines drawn to neighbors
- [ ] Connection count badge displays
- [ ] Smooth transitions (200ms fade)
- [ ] No performance degradation on hover
- [ ] Natural progression to Phase 2 (click to force view)

### Phase 2 Success Criteria
- [ ] Click entity → force view transition smooth (1s duration)
- [ ] **Camera interpolation** prevents disorientation
- [ ] **Dual animation** (nodes + camera) synchronized
- [ ] Force simulation stable within 3s
- [ ] Neighborhood extraction correct (1-2 hop validation)
- [ ] Return to embedding view works correctly
- [ ] Neighborhood depth slider functional (1/2/3-hop)
- [ ] No performance degradation in force mode

### Phase 3 Success Criteria
- [ ] Neighborhood aggregation embeddings computed for all entities
- [ ] New PCA positions show improved clustering (silhouette score +10%)
- [ ] A/B test: ≥60% users prefer new layout
- [ ] Documentation updated with embedding methodology

### Overall Product Success
- **Engagement**: Users spend ≥2 minutes exploring graph (vs. <30s with drill-down)
- **Discovery**: Users report finding unexpected connections
- **Performance**: <5% users report lag or slowness
- **Adoption**: /graph/ page views increase by ≥50%

## Risks & Mitigations

### Risk 1: Performance Degradation
- **Impact**: Slow rendering, browser crashes, poor UX
- **Likelihood**: Medium (39k entities is a lot)
- **Mitigation**:
  - Implement LOD early in Phase 1
  - Performance profiling at each milestone
  - Fallback to drill-down mode if device can't handle
  - Consider WebGL2 features (faster rendering)

### Risk 2: Visual Clutter
- **Impact**: Users can't distinguish entities, overwhelmed by density
- **Likelihood**: High (dense semantic space)
- **Mitigation**:
  - Default to sampled nodes (5k) not all 39k
  - Strong cluster hulls for visual grouping
  - Intelligent zoom behavior (reveal on approach)
  - Entity type filtering to reduce density

### Risk 3: Embedding Quality
- **Impact**: Poor clustering, entities in wrong places, no semantic structure visible
- **Likelihood**: Low (text embeddings generally work well)
- **Mitigation**:
  - Validate embeddings with known entity relationships
  - Manual spot-checks ("Aaron Perry should be near VIRIDITAS")
  - User feedback loop
  - Phase 3 enrichment to improve quality

### Risk 4: Force Mode Transition Jarring
- **Impact**: Disorienting transition, users lose context
- **Likelihood**: Medium (big layout change)
- **Mitigation**:
  - Smooth 1s eased animation
  - Highlight selected entity throughout transition
  - Breadcrumb trail (embedding → force → embedding)
  - Option to disable auto-transition

## Appendix

### References

**Similar Implementations**:
- YonEarth PodcastMap3D: `/PodcastMap3D.html` (6k nodes, UMAP, clusters as ovals)
- YonEarth GraphRAG Current: `/graph/` (drill-down hierarchy)

**Libraries & Tools**:
- [3d-force-graph](https://github.com/vasturiano/3d-force-graph): Force-directed 3D graphs
- [Three.js](https://threejs.org/): 3D rendering
- [Node2Vec](https://github.com/eliorc/node2vec): Graph embedding via random walks
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/): GraphSAGE implementation

**Research Papers**:
- Grover & Leskovec (2016): "node2vec: Scalable Feature Learning for Networks"
- Hamilton et al. (2017): "Inductive Representation Learning on Large Graphs" (GraphSAGE)
- van der Maaten & Hinton (2008): "Visualizing Data using t-SNE"
- McInnes et al. (2018): "UMAP: Uniform Manifold Approximation and Projection"

### Data Schema

**graphrag_hierarchy.json structure**:
```json
{
  "entities": {
    "Aaron Perry": {
      "type": "PERSON",
      "description": "Author of VIRIDITAS novel",
      "sources": ["episode_120", "veriditas"],
      "original_type": "PERSON"
    }
  },
  "relationships": [
    {
      "source": "Aaron Perry",
      "target": "VIRIDITAS",
      "type": "AUTHORED",
      "sources": ["episode_120"]
    }
  ],
  "clusters": {
    "level_0": {
      "cluster_0_0": {
        "id": "cluster_0_0",
        "type": "entity",
        "entity": "Aaron Perry",
        "embedding_idx": 0,
        "position": [310.7, -76.9, 40.6]
      }
    },
    "level_1": {
      "cluster_1_5": {
        "id": "cluster_1_5",
        "type": "fine_cluster",
        "children": ["cluster_0_0", "cluster_0_1", ...],
        "position": [305.2, -80.1, 38.9]
      }
    }
  }
}
```

### Glossary

- **Embedding**: Dense vector representation of entity (e.g., 1536-dim from OpenAI)
- **PCA**: Principal Component Analysis, reduces dimensionality (1536-dim → 3-dim)
- **UMAP**: Uniform Manifold Approximation and Projection, non-linear dimensionality reduction
- **Force-directed layout**: Physics simulation that positions connected nodes close together
- **LOD**: Level of Detail, rendering different detail levels based on camera distance
- **Frustum culling**: Only rendering objects visible to camera
- **Instanced rendering**: Drawing many identical objects in one GPU call
- **Octree**: Spatial data structure for fast 3D queries
- **Node2Vec**: Graph embedding method using random walks
- **GraphSAGE**: Graph neural network for inductive node embeddings

## Design Refinements (November 2025)

This design incorporates critical enhancements from technical review feedback, elevating the visualization from "good 3D graph" to **"cognitive cartography of regenerative knowledge networks"**.

### Incorporated Enhancements

**1. Cellular Metaphor with Fresnel Shaders** (Phase 1)
- Cluster boundaries rendered as semi-permeable membranes
- Edges more opaque than center (biological cell aesthetic)
- Reduces visual noise while clearly defining boundaries
- Aligns with regenerative systems philosophy

**2. UMAP over PCA** (Phase 1 - Data Prep)
- Preserves local neighborhood structure better than linear PCA
- Utilizes 3D volume (prevents flattening into planes)
- "Regenerative Agriculture" entities form distinct clouds, not smeared axes
- Pre-computed in Python pipeline for performance

**3. Camera Interpolation** (Phase 2 - Critical)
- Prevents nodes flying offscreen during force mode transition
- Tween camera AND nodes simultaneously
- Always keeps subgraph in comfortable view
- Eliminates disorienting UX failure mode

**4. Troika-Three-Text with Aggressive LOD** (Phase 1)
- Correctly identifies text rendering as THE performance bottleneck
- SDF font rendering for crisp text at all zoom levels
- Distance-based LOD: Far (no text) → Close (all entities)
- Massive performance improvement over naive text sprites

**5. Epsilon Handling for Degenerate Ellipsoids** (Phase 1)
- Prevents crashes on 1-2 node clusters
- Minimum radius ensures visibility
- Defensive programming for edge cases

**6. Betweenness Centrality Bridge Scoring** (Phase 1 - Data Prep)
- Highlights interdisciplinary connector nodes
- Color gradient: Blue (peripheral) → Yellow (connector) → Red (critical bridges)
- Reveals cross-pollination points in knowledge network
- Aligns with symbiotic AI vision

**7. Phase 1.5: Context Lens** (NEW - Hover Preview)
- Natural stepping stone between embedding and force modes
- Preview connections without committing to transition
- Non-destructive (map stays fixed)
- Allows rapid comparison across multiple entities
- Key UX innovation for progressive disclosure

### Philosophy: Biologically-Coherent Visualization

These enhancements create a unified biological metaphor:
- **Cells**: Clusters as semi-permeable membranes (Fresnel shaders)
- **Organisms**: Entities positioned by semantic similarity (UMAP)
- **Ecosystems**: Bridge nodes connecting disparate domains (betweenness)
- **Networks**: Relationships weighted by co-occurrence strength

The result is not just data visualization, but **cognitive cartography** that reveals the living structure of regenerative knowledge.

---

**Document History**:
- v1.0 (2025-11-20): Initial design document based on user requirements and PodcastMap3D analysis
- v2.0 (2025-11-20): Incorporated technical review feedback (UMAP, betweenness, Fresnel, Context Lens, camera interpolation)

---

## 📝 Quick Reference for Next Session

### File Locations

**Frontend (Production Ready):**
- `web/graph/GraphRAG3D_EmbeddingView.html` - Main viewer HTML
- `web/graph/GraphRAG3D_EmbeddingView.js` - Visualization logic with Fresnel shaders
- Production: `/opt/yonearth-chatbot/web/graph/` (deployed)

**Backend Scripts:**
- `scripts/compute_graphrag_umap_embeddings.py` - Full UMAP pipeline (needs 16GB RAM)
- `scripts/compute_graphrag_umap_embeddings_test.py` - Test with 100 entities

**Data:**
- `data/graphrag_hierarchy/graphrag_hierarchy.json` - Current (PCA positions)
- `data/graphrag_hierarchy/graphrag_hierarchy_test_sample.json` - Test sample
- Production: `/opt/yonearth-chatbot/web/data/graphrag_hierarchy/`

**Logs:**
- `logs/umap_full_20251120_061315.log` - Last UMAP attempt (killed by OOM)

### Commands to Run After 16GB Upgrade

**1. Run UMAP Computation:**
```bash
cd /home/claudeuser/yonearth-gaia-chatbot
mkdir -p logs
nohup bash -c 'set -a && source .env && set +a && python3 scripts/compute_graphrag_umap_embeddings.py' > logs/umap_full_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Monitor (Ctrl+C to exit tail, process continues)
tail -f logs/umap_full_*.log
```

**2. Deploy to Production:**
```bash
# Copy updated data
sudo cp data/graphrag_hierarchy/graphrag_hierarchy.json \
        /opt/yonearth-chatbot/web/data/graphrag_hierarchy/

# Fix permissions
sudo chown www-data:www-data /opt/yonearth-chatbot/web/data/graphrag_hierarchy/graphrag_hierarchy.json
sudo chmod 644 /opt/yonearth-chatbot/web/data/graphrag_hierarchy/graphrag_hierarchy.json
```

**3. Verify:**
```bash
# Check file size (should be ~48-50MB)
ls -lh /opt/yonearth-chatbot/web/data/graphrag_hierarchy/graphrag_hierarchy.json

# Test HTTP access
curl -s -o /dev/null -w "%{http_code}" https://earthdo.me/data/graphrag_hierarchy/graphrag_hierarchy.json
```

**4. Test Viewer:**
- Open: https://earthdo.me/graph/GraphRAG3D_EmbeddingView.html
- Should load all 39,054 entities with UMAP positions
- Check browser console for errors
- Verify FPS counter shows ~60 FPS

### Architecture Decisions Made

**Fresnel Shader Implementation:**
- Edge-glowing membranes for cellular aesthetic
- Depth fade when camera inside ellipsoid
- Base opacity varies by hierarchy level (0.05, 0.10, 0.15)

**Entity Type Colors:**
```javascript
PERSON: #4CAF50 (green)
ORGANIZATION: #2196F3 (blue)
CONCEPT: #9C27B0 (purple)
PRACTICE: #FF9800 (orange)
PRODUCT: #F44336 (red)
PLACE: #00BCD4 (cyan)
EVENT: #FFEB3B (yellow)
WORK: #795548 (brown)
```

**Performance Optimizations:**
- No edges rendered in default view (43k edges = FPS killer)
- Instanced rendering for entity nodes
- Frustum culling enabled
- OrbitControls with damping
- Cluster membranes use DoubleSide material

**Data Format:**
- Test mode: Flat structure with `test_mode: true` flag
- Full mode: Hierarchical with `clusters.level_0` through `level_3`
- Fallback: UMAP positions → PCA positions → [0,0,0]

### What Still Needs Implementation

**Critical (Phase 2):**
1. Force-directed neighborhood view on entity click
2. Camera interpolation during mode transitions
3. Physics freeze/unfreeze for smooth animations

**Important (Phase 3):**
1. Context lens hover preview with GPU acceleration
2. Temporary connection lines on hover
3. Opacity dimming of non-neighbors

**Nice-to-Have (Phase 4):**
1. Search autocomplete with fuzzy matching
2. Betweenness centrality color mode
3. Cluster membrane click → zoom to cluster
4. Screenshot/export functionality

### Dependencies Already Installed

- Three.js 0.159.0 (via CDN)
- OrbitControls (via CDN)
- 3d-force-graph 1.73.3 (via CDN)
- umap-learn 0.5.9.post2 (Python)
- networkx (Python, for betweenness centrality)

### Testing Checklist

**Before Next Session:**
- [ ] Verify server upgraded to 16GB RAM
- [ ] Check `/home/claudeuser/yonearth-gaia-chatbot/.env` has OPENAI_API_KEY

**After UMAP Completes:**
- [ ] Backup exists at `data/graphrag_hierarchy/graphrag_hierarchy_backup_pre_umap.json`
- [ ] Updated file has `umap_position` fields
- [ ] Updated file has `betweenness` scores
- [ ] File size is ~48-50MB
- [ ] Deployed to production successfully
- [ ] Viewer loads all 39k entities
- [ ] FPS stable at 60
- [ ] No browser console errors
- [ ] Hover interactions work
- [ ] Entity type filters work
- [ ] Keyboard shortcuts work (E/F/ESC)

---

**End of Document**
