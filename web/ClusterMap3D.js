/**
 * Cluster Map 3D Visualization
 * Interactive 3D view over semantic topic clusters (books + episodes).
 *
 * Uses the precomputed data/graph_index/cluster_map_3d.json export, which
 * includes:
 *   - levels.coarse / levels.fine
 *   - nodes[cluster_id] with position[x,y,z], size, content_type, summary, etc.
 *
 * Rendering is done via 3d-force-graph with static 3D coordinates (no forces).
 */

class ClusterMap3D {
    constructor(containerId) {
        this.containerId = containerId;
        this.rawData = null;
        this.graph = null;
        this.autoRotate = true;
        this.rotationAngle = 0;
        this.rotationSpeed = 0.05;
        this.levelVisibility = {
            coarse: true,
            fine: true
        };

        // Color palette for levels / content types
        this.levelColors = {
            coarse: '#4CAF50', // green
            fine: '#FF9800'    // orange
        };
        this.contentTypeColors = {
            book: '#3f51b5',
            episode: '#009688',
            mixed: '#9c27b0',
            unknown: '#9e9e9e'
        };

        this.init();
    }

    async init() {
        await this.loadData();
        this.create3DGraph();
        this.setupControls();
        this.startAutoRotation();
        window.clusterMap3D = this; // For panel close button
    }

    async loadData() {
        const filename = '/data/graph_index/cluster_map_3d.json';
        try {
            const response = await fetch(filename, {
                cache: 'no-store',
                headers: {
                    'Cache-Control': 'no-cache',
                    'Pragma': 'no-cache'
                }
            });
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            this.rawData = data;
            console.log('Loaded cluster_map_3d data:', data);
        } catch (error) {
            console.error('Error loading cluster_map_3d.json:', error);
            this.rawData = { levels: { coarse: [], fine: [] }, nodes: {}, meta: {} };
        }
    }

    getFilteredGraphData() {
        const levels = this.rawData.levels || {};
        const nodesById = this.rawData.nodes || {};

        const includeLevels = new Set();
        if (this.levelVisibility.coarse && Array.isArray(levels.coarse)) {
            includeLevels.add('coarse');
        }
        if (this.levelVisibility.fine && Array.isArray(levels.fine)) {
            includeLevels.add('fine');
        }

        const nodes = [];
        const levelIndex = {};

        Object.keys(nodesById).forEach(id => {
            const node = nodesById[id];
            const level = node.level || 'coarse';
            levelIndex[id] = level;
            if (includeLevels.has(level)) {
                const [x, y, z] = node.position || [0, 0, 0];
                nodes.push({
                    id,
                    level,
                    parent: node.parent,
                    children: node.children || [],
                    x,
                    y,
                    z,
                    size: node.size || 1,
                    content_type: node.content_type || 'unknown',
                    top_books: node.top_books || [],
                    summary: node.summary || ''
                });
            }
        });

        // Build hierarchical links between clusters
        const links = [];
        Object.values(nodesById).forEach(node => {
            const fromLevel = node.level || 'coarse';
            const fromId = node.id;
            if (!includeLevels.has(fromLevel)) {
                return;
            }
            const children = node.children || [];
            children.forEach(childId => {
                const childNode = nodesById[childId];
                if (!childNode) return;
                const childLevel = childNode.level || 'fine';
                if (!includeLevels.has(childLevel)) {
                    return;
                }
                links.push({
                    source: fromId,
                    target: childId,
                    level: 'hierarchy'
                });
            });
        });

        return { nodes, links };
    }

    create3DGraph() {
        const elem = document.getElementById(this.containerId.replace('#', ''));
        const graphData = this.getFilteredGraphData();

        this.graph = ForceGraph3D()(elem)
            .graphData(graphData)
            .nodeLabel(node => {
                const books = (node.top_books || []).join(', ') || 'N/A';
                return `<strong>${node.id}</strong><br/>Level: ${node.level}<br/>Content: ${node.content_type}<br/>Top books: ${books}<br/>Chunks: ${node.size}`;
            })
            .nodeColor(node => {
                const base = this.contentTypeColors[node.content_type] || this.levelColors[node.level] || '#9e9e9e';
                return base;
            })
            .nodeVal(node => {
                // Scale node size in a gentle way
                const minSize = 3;
                const maxSize = 18;
                const size = Math.max(1, Math.log10(node.size || 1) + 1);
                return Math.min(maxSize, Math.max(minSize, size * 4));
            })
            .nodeOpacity(0.9)
            .nodeResolution(16)
            .linkColor(() => '#ffffff')
            .linkOpacity(0.15)
            .linkWidth(0.5)
            .backgroundColor('#0a0e1a')
            .showNavInfo(false)
            .onNodeClick(node => this.handleNodeClick(node))
            .onNodeHover(node => this.handleNodeHover(node))
            .enableNodeDrag(false)
            // Use static coordinates, no layout forces
            .d3Force('center', null)
            .d3Force('charge', null)
            .d3Force('link', null);

        // Place camera at a reasonable distance
        this.graph.cameraPosition({ z: 250 });

        // Stop rotation on user click in canvas
        elem.addEventListener('click', () => {
            this.autoRotate = false;
        });
    }

    updateGraph() {
        if (!this.graph || !this.rawData) return;
        const graphData = this.getFilteredGraphData();
        this.graph.graphData(graphData);
    }

    setupControls() {
        const showCoarse = document.getElementById('show-coarse');
        const showFine = document.getElementById('show-fine');

        if (showCoarse) {
            showCoarse.addEventListener('change', (event) => {
                this.levelVisibility.coarse = !!event.target.checked;
                this.updateGraph();
            });
        }
        if (showFine) {
            showFine.addEventListener('change', (event) => {
                this.levelVisibility.fine = !!event.target.checked;
                this.updateGraph();
            });
        }

        const mobileMenuToggle = document.getElementById('mobile-menu-toggle');
        const headerControls = document.getElementById('header-controls');
        if (mobileMenuToggle && headerControls) {
            mobileMenuToggle.addEventListener('click', () => {
                const isVisible = headerControls.style.display === 'flex';
                headerControls.style.display = isVisible ? 'none' : 'flex';
            });
        }
    }

    startAutoRotation() {
        const animate = () => {
            if (this.autoRotate && this.graph) {
                this.rotationAngle += this.rotationSpeed;
                const distance = 260;
                const x = distance * Math.sin(this.rotationAngle * Math.PI / 180);
                const z = distance * Math.cos(this.rotationAngle * Math.PI / 180);

                this.graph.cameraPosition(
                    { x: x, y: 0, z: z },
                    { x: 0, y: 0, z: 0 },
                    0
                );
            }
            requestAnimationFrame(animate);
        };
        animate();
    }

    handleNodeClick(node) {
        this.autoRotate = false;
        const panel = document.getElementById('cluster-info-panel');
        const titleEl = document.getElementById('cluster-title');
        const summaryEl = document.getElementById('cluster-summary');
        const metaEl = document.getElementById('cluster-meta');

        if (!panel || !titleEl || !summaryEl || !metaEl) return;

        const books = (node.top_books || []).join(', ') || 'N/A';
        titleEl.textContent = `${node.id} (${node.level})`;
        summaryEl.textContent = node.summary || 'No summary available.';
        metaEl.textContent = `Content: ${node.content_type} • Chunks: ${node.size} • Top books: ${books}`;

        panel.style.display = 'block';
    }

    handleNodeHover(node) {
        // Future: highlight cluster neighborhood or show lightweight tooltip
        if (!this.graph) return;
        const highlightNodes = new Set();
        const highlightLinks = new Set();

        if (node) {
            highlightNodes.add(node.id);
            // Highlight direct hierarchical neighbors
            const nodesById = this.rawData.nodes || {};
            const current = nodesById[node.id];
            if (current) {
                (current.children || []).forEach(childId => highlightNodes.add(childId));
                if (current.parent) highlightNodes.add(current.parent);
            }
        }

        // Update node styles
        this.graph.nodeColor(n => {
            const base = this.contentTypeColors[n.content_type] || this.levelColors[n.level] || '#9e9e9e';
            if (!highlightNodes.size || highlightNodes.has(n.id)) return base;
            return '#444444';
        });
    }

    closeInfoPanel() {
        const panel = document.getElementById('cluster-info-panel');
        if (panel) {
            panel.style.display = 'none';
        }
    }
}

// Initialize on DOM ready
window.addEventListener('DOMContentLoaded', () => {
    new ClusterMap3D('#3d-graph');
});

