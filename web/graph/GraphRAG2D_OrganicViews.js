/**
 * GraphRAG 2D Organic Views
 *
 * Implements Circle Packing and Voronoi Treemap visualizations
 * with integrated GraphRAG community summaries.
 *
 * Features:
 * - Zoomable Circle Packing (ragged hierarchy support)
 * - Voronoi Treemap (cellular/organic aesthetic)
 * - Side panel with LLM-generated cluster summaries
 * - Semantic zoom interactions
 */

class GraphRAG2DOrganicViews {
    constructor() {
        this.currentMode = 'circle-pack';
        this.colorMode = 'category';

        // Data structures
        this.hierarchyData = null;
        this.summariesData = null;
        this.mergedData = null;

        // D3 elements
        this.svg = null;
        this.g = null;
        this.width = window.innerWidth;
        this.height = window.innerHeight;

        // Color scales
        this.categoryColors = d3.scaleOrdinal(d3.schemeTableau10);
        this.sizeColors = d3.scaleSequential(d3.interpolateViridis);

        // Zoom behavior
        this.zoom = null;
        this.currentZoom = null;

        // Selected cluster
        this.selectedCluster = null;

        // Tooltip
        this.tooltip = d3.select('#tooltip');
    }

    /**
     * Initialize the visualization
     */
    async init() {
        this.updateLoadingStatus('Loading hierarchy data...');

        try {
            // Load data
            await this.loadData();

            // Setup SVG
            this.setupSVG();

            // Setup controls
            this.setupControls();

            // Render initial visualization
            this.updateLoadingStatus('Rendering Circle Packing...');
            this.render();

            // Hide loading screen
            this.hideLoadingScreen();

            console.log('✅ GraphRAG 2D Organic Views initialized');
        } catch (error) {
            console.error('❌ Failed to initialize viewer:', error);
            this.updateLoadingStatus(`Error: ${error.message}`);
        }
    }

    /**
     * Load and merge data from hierarchy and summaries
     */
    async loadData() {
        try {
            // Load hierarchy data
            const hierarchyResponse = await fetch('/data/graphrag_hierarchy/graphrag_hierarchy.json');
            if (!hierarchyResponse.ok) {
                throw new Error('Failed to load hierarchy data');
            }
            this.hierarchyData = await hierarchyResponse.json();

            // Load summaries data
            this.updateLoadingStatus('Loading community summaries...');
            const summariesResponse = await fetch('/data/graphrag_hierarchy/checkpoints/summaries_progress.json');
            if (!summariesResponse.ok) {
                console.warn('Community summaries not available');
                this.summariesData = {};
            } else {
                this.summariesData = await summariesResponse.json();
            }

            // Load Leiden hierarchies for mapping
            this.updateLoadingStatus('Loading Leiden mappings...');
            const leidenResponse = await fetch('/data/graphrag_hierarchy/checkpoints/leiden_hierarchies.json');
            let leidenData = {};
            if (leidenResponse.ok) {
                leidenData = await leidenResponse.json();
            }

            // Merge data
            this.updateLoadingStatus('Merging data structures...');
            this.mergeData(leidenData);

            console.log('Data loaded:', {
                clusters: Object.keys(this.hierarchyData.clusters || {}).length,
                summaries: Object.keys(this.summariesData || {}).length,
                merged: this.mergedData
            });

        } catch (error) {
            console.error('Data loading error:', error);
            throw error;
        }
    }

    /**
     * Merge hierarchy and summary data
     * Maps Leiden IDs (l0_0, l1_0) to cluster IDs (c0, c66)
     */
    mergeData(leidenData) {
        // Create mapping from Leiden clusters to graphrag_hierarchy clusters
        // This is complex because they use different ID schemes

        // For now, we'll create a hierarchical structure for D3
        // Level 3 (Root) -> Level 2 -> Level 1 -> Level 0 (Entities)

        const clusters = this.hierarchyData.clusters || {};
        const level3Clusters = clusters.level_3 || clusters.L3 || {};
        const level2Clusters = clusters.level_2 || clusters.L2 || {};
        const level1Clusters = clusters.level_1 || clusters.L1 || {};
        const level0Entities = clusters.level_0 || clusters.L0 || {};

        // Build D3 hierarchy structure
        const root = {
            name: "Knowledge Graph",
            id: "root",
            level: "root",
            children: []
        };

        // Process Level 3 (Root categories)
        for (const [clusterId, cluster] of Object.entries(level3Clusters)) {
            const level3Node = {
                name: cluster.name || clusterId,
                id: clusterId,
                level: 3,
                entity_count: cluster.entities?.length || 0,
                children: []
            };

            // Add summary if available
            const summaryKey = this.findSummaryKey(cluster, leidenData, 0);
            if (summaryKey && this.summariesData && this.summariesData['0']) {
                const summary = this.summariesData['0'][summaryKey];
                if (summary) {
                    level3Node.summary_title = summary.title;
                    level3Node.summary = summary.summary;
                }
            }

            // Add Level 2 children
            if (cluster.children && cluster.children.length > 0) {
                for (const childId of cluster.children) {
                    const level2Cluster = level2Clusters[childId];
                    if (!level2Cluster) continue;

                    const level2Node = {
                        name: level2Cluster.name || childId,
                        id: childId,
                        level: 2,
                        entity_count: level2Cluster.entities?.length || 0,
                        children: []
                    };

                    // Add summary
                    const l2SummaryKey = this.findSummaryKey(level2Cluster, leidenData, 1);
                    if (l2SummaryKey && this.summariesData && this.summariesData['1']) {
                        const summary = this.summariesData['1'][l2SummaryKey];
                        if (summary) {
                            level2Node.summary_title = summary.title;
                            level2Node.summary = summary.summary;
                        }
                    }

                    // Add Level 1 children
                    if (level2Cluster.children && level2Cluster.children.length > 0) {
                        for (const l1ChildId of level2Cluster.children) {
                            const level1Cluster = level1Clusters[l1ChildId];
                            if (!level1Cluster) continue;

                            const level1Node = {
                                name: level1Cluster.name || l1ChildId,
                                id: l1ChildId,
                                level: 1,
                                entity_count: level1Cluster.entities?.length || 0,
                                children: []
                            };

                            // Add summary
                            const l1SummaryKey = this.findSummaryKey(level1Cluster, leidenData, 2);
                            if (l1SummaryKey && this.summariesData && this.summariesData['2']) {
                                const summary = this.summariesData['2'][l1SummaryKey];
                                if (summary) {
                                    level1Node.summary_title = summary.title;
                                    level1Node.summary = summary.summary;
                                }
                            }

                            // Add entities as leaf nodes
                            if (level1Cluster.entities && level1Cluster.entities.length > 0) {
                                for (const entityId of level1Cluster.entities.slice(0, 10)) { // Limit entities for performance
                                    const entity = level0Entities[entityId];
                                    if (entity && entity.entity) {
                                        level1Node.children.push({
                                            name: entityId,
                                            id: entityId,
                                            level: 0,
                                            type: entity.entity.type || 'UNKNOWN',
                                            description: entity.entity.description || '',
                                            value: 1 // For circle size
                                        });
                                    }
                                }
                            }

                            level2Node.children.push(level1Node);
                        }
                    } else {
                        // If Level 2 has no Level 1 children, add entities directly (ragged hierarchy)
                        if (level2Cluster.entities && level2Cluster.entities.length > 0) {
                            for (const entityId of level2Cluster.entities.slice(0, 10)) {
                                const entity = level0Entities[entityId];
                                if (entity && entity.entity) {
                                    level2Node.children.push({
                                        name: entityId,
                                        id: entityId,
                                        level: 0,
                                        type: entity.entity.type || 'UNKNOWN',
                                        description: entity.entity.description || '',
                                        value: 1
                                    });
                                }
                            }
                        }
                    }

                    level3Node.children.push(level2Node);
                }
            } else {
                // If Level 3 has no children, add entities directly (ragged hierarchy)
                if (cluster.entities && cluster.entities.length > 0) {
                    for (const entityId of cluster.entities.slice(0, 15)) {
                        const entity = level0Entities[entityId];
                        if (entity && entity.entity) {
                            level3Node.children.push({
                                name: entityId,
                                id: entityId,
                                level: 0,
                                type: entity.entity.type || 'UNKNOWN',
                                description: entity.entity.description || '',
                                value: 1
                            });
                        }
                    }
                }
            }

            root.children.push(level3Node);
        }

        this.mergedData = root;
    }

    /**
     * Find summary key for a cluster (simplified matching)
     */
    findSummaryKey(cluster, leidenData, summaryLevel) {
        // This is a simplified approach - in production, you'd need proper mapping
        // For now, we'll try to match by entity overlap or use index-based mapping

        // Try index-based mapping as fallback
        if (cluster.id && typeof cluster.id === 'string') {
            const idNum = parseInt(cluster.id.replace(/[^\d]/g, ''));
            if (!isNaN(idNum)) {
                return `l${summaryLevel}_${idNum}`;
            }
        }

        return null;
    }

    /**
     * Setup SVG canvas
     */
    setupSVG() {
        this.svg = d3.select('#svg-container')
            .attr('width', this.width)
            .attr('height', this.height);

        this.g = this.svg.append('g');

        // Setup zoom
        this.zoom = d3.zoom()
            .scaleExtent([0.1, 10])
            .on('zoom', (event) => {
                this.g.attr('transform', event.transform);
                this.currentZoom = event.transform;
            });

        this.svg.call(this.zoom);

        // Handle window resize
        window.addEventListener('resize', () => {
            this.width = window.innerWidth;
            this.height = window.innerHeight;
            this.svg.attr('width', this.width).attr('height', this.height);
            this.render();
        });
    }

    /**
     * Setup UI controls
     */
    setupControls() {
        // Mode buttons
        d3.selectAll('[data-mode]').on('click', (event) => {
            const mode = event.target.closest('[data-mode]').dataset.mode;

            if (mode === 'holographic') {
                // Redirect to 3D view
                window.location.href = '/YonEarth/graph/GraphRAG3D_EmbeddingView.html';
                return;
            }

            this.currentMode = mode;
            d3.selectAll('[data-mode]').classed('active', false);
            event.target.closest('[data-mode]').classList.add('active');
            this.render();
        });

        // Color mode buttons
        d3.selectAll('[data-color-mode]').on('click', (event) => {
            this.colorMode = event.target.closest('[data-color-mode]').dataset.colorMode;
            d3.selectAll('[data-color-mode]').classed('active', false);
            event.target.closest('[data-color-mode]').classList.add('active');
            this.render();
        });

        // Summary panel close button
        d3.select('#close-summary').on('click', () => {
            this.hideSummaryPanel();
        });
    }

    /**
     * Render current visualization
     */
    render() {
        // Clear existing visualization
        this.g.selectAll('*').remove();

        if (this.currentMode === 'circle-pack') {
            this.renderCirclePacking();
        } else if (this.currentMode === 'voronoi') {
            this.renderVoronoiTreemap();
        }
    }

    /**
     * Render Circle Packing visualization
     */
    renderCirclePacking() {
        // Create D3 hierarchy
        const hierarchy = d3.hierarchy(this.mergedData)
            .sum(d => d.value || d.entity_count || 1)
            .sort((a, b) => b.value - a.value);

        // Create pack layout
        const pack = d3.pack()
            .size([this.width, this.height])
            .padding(3);

        const root = pack(hierarchy);

        // Get top-level categories for color scale
        const topCategories = root.children || [];
        this.categoryColors.domain(topCategories.map(d => d.data.name));

        // Create circles
        const circles = this.g.selectAll('circle')
            .data(root.descendants())
            .join('circle')
            .attr('cx', d => d.x)
            .attr('cy', d => d.y)
            .attr('r', d => d.r)
            .attr('fill', d => this.getCircleColor(d, root))
            .attr('fill-opacity', d => {
                if (d.depth === 0) return 0; // Hide root
                if (d.data.level === 0) return 0.8; // Entities
                return 0.3; // Clusters
            })
            .attr('stroke', d => {
                if (d.depth === 0) return 'none';
                return d.data.level === 0 ? 'none' : 'rgba(100, 200, 255, 0.5)';
            })
            .attr('stroke-width', d => d.data.level === 0 ? 0 : 1.5)
            .style('cursor', d => d.data.level === 0 ? 'default' : 'pointer')
            .on('click', (event, d) => this.handleCircleClick(event, d))
            .on('mouseover', (event, d) => this.showTooltip(event, d))
            .on('mouseout', () => this.hideTooltip());

        // Create labels for larger circles
        const labels = this.g.selectAll('text')
            .data(root.descendants().filter(d => d.r > 30 && d.depth > 0))
            .join('text')
            .attr('x', d => d.x)
            .attr('y', d => d.y)
            .attr('text-anchor', 'middle')
            .attr('dy', '0.35em')
            .attr('fill', '#ffffff')
            .attr('font-size', d => Math.min(d.r / 3, 16))
            .attr('font-weight', d => d.data.level > 1 ? '600' : '400')
            .attr('pointer-events', 'none')
            .text(d => {
                const name = d.data.name;
                const maxLen = Math.floor(d.r / 5);
                return name.length > maxLen ? name.substring(0, maxLen) + '...' : name;
            });

        // Add entity count labels
        const countLabels = this.g.selectAll('.count-label')
            .data(root.descendants().filter(d => d.r > 40 && d.data.level > 0 && d.data.entity_count))
            .join('text')
            .attr('class', 'count-label')
            .attr('x', d => d.x)
            .attr('y', d => d.y + Math.min(d.r / 3, 14))
            .attr('text-anchor', 'middle')
            .attr('dy', '0.35em')
            .attr('fill', 'rgba(255, 255, 255, 0.6)')
            .attr('font-size', d => Math.min(d.r / 4, 12))
            .attr('pointer-events', 'none')
            .text(d => `${d.data.entity_count} entities`);
    }

    /**
     * Get color for circle based on color mode
     */
    getCircleColor(d, root) {
        if (d.depth === 0) return 'none'; // Root

        if (this.colorMode === 'category') {
            // Color by top-level parent
            let current = d;
            while (current.parent && current.parent.depth > 0) {
                current = current.parent;
            }
            return this.categoryColors(current.data.name);
        } else {
            // Color by entity count
            const maxCount = d3.max(root.descendants(), d => d.data.entity_count || 0);
            const count = d.data.entity_count || 0;
            return this.sizeColors(count / maxCount);
        }
    }

    /**
     * Handle circle click for semantic zoom
     */
    handleCircleClick(event, d) {
        event.stopPropagation();

        if (d.data.level === 0) return; // Don't zoom on entities

        // Show summary if available
        if (d.data.summary) {
            this.showSummaryPanel(d);
        }

        // Zoom to this circle
        this.zoomToCircle(d);
    }

    /**
     * Zoom to specific circle
     */
    zoomToCircle(d) {
        const scale = this.width / (d.r * 2 + 100);
        const translate = [this.width / 2 - d.x * scale, this.height / 2 - d.y * scale];

        this.svg.transition()
            .duration(750)
            .call(
                this.zoom.transform,
                d3.zoomIdentity.translate(translate[0], translate[1]).scale(scale)
            );
    }

    /**
     * Render Voronoi Treemap visualization
     */
    renderVoronoiTreemap() {
        // For Voronoi Treemap, we need d3-voronoi-treemap
        // This is a simplified version using weighted Voronoi diagram

        // Get leaf clusters (Level 1)
        const hierarchy = d3.hierarchy(this.mergedData);
        const level1Nodes = hierarchy.descendants().filter(d => d.data.level === 1);

        if (level1Nodes.length === 0) {
            console.warn('No Level 1 nodes found for Voronoi treemap');
            return;
        }

        // Create random but deterministic positions for Voronoi sites
        const padding = 50;
        const sites = level1Nodes.map((node, i) => {
            // Use hash of node ID for consistent positioning
            const hash = this.hashCode(node.data.id);
            const angle = (hash % 360) * Math.PI / 180;
            const radius = ((hash >> 8) % 100) / 100 * Math.min(this.width, this.height) / 3;

            return {
                x: this.width / 2 + Math.cos(angle) * radius,
                y: this.height / 2 + Math.sin(angle) * radius,
                node: node,
                weight: node.data.entity_count || 1
            };
        });

        // Create Voronoi diagram
        const delaunay = d3.Delaunay.from(sites, d => d.x, d => d.y);
        const voronoi = delaunay.voronoi([padding, padding, this.width - padding, this.height - padding]);

        // Get top-level categories for color
        const topLevelParents = new Map();
        level1Nodes.forEach(node => {
            let current = node;
            while (current.parent && current.parent.depth > 0) {
                current = current.parent;
            }
            topLevelParents.set(node.data.id, current.data.name);
        });

        // Draw Voronoi cells
        const cells = this.g.selectAll('.voronoi-cell')
            .data(sites)
            .join('path')
            .attr('class', 'voronoi-cell')
            .attr('d', (d, i) => voronoi.renderCell(i))
            .attr('fill', d => {
                const parentName = topLevelParents.get(d.node.data.id);
                return this.categoryColors(parentName);
            })
            .attr('fill-opacity', 0.6)
            .attr('stroke', 'rgba(100, 200, 255, 0.8)')
            .attr('stroke-width', 2)
            .style('cursor', 'pointer')
            .on('click', (event, d) => {
                event.stopPropagation();
                if (d.node.data.summary) {
                    this.showSummaryPanel(d.node);
                }
            })
            .on('mouseover', (event, d) => this.showTooltip(event, d.node))
            .on('mouseout', () => this.hideTooltip());

        // Add labels
        const labels = this.g.selectAll('.voronoi-label')
            .data(sites)
            .join('text')
            .attr('class', 'voronoi-label')
            .attr('x', d => d.x)
            .attr('y', d => d.y)
            .attr('text-anchor', 'middle')
            .attr('dy', '0.35em')
            .attr('fill', '#ffffff')
            .attr('font-size', 14)
            .attr('font-weight', '600')
            .attr('pointer-events', 'none')
            .text(d => {
                const name = d.node.data.name;
                return name.length > 20 ? name.substring(0, 20) + '...' : name;
            });

        // Add entity count labels
        const countLabels = this.g.selectAll('.voronoi-count')
            .data(sites)
            .join('text')
            .attr('class', 'voronoi-count')
            .attr('x', d => d.x)
            .attr('y', d => d.y + 18)
            .attr('text-anchor', 'middle')
            .attr('dy', '0.35em')
            .attr('fill', 'rgba(255, 255, 255, 0.7)')
            .attr('font-size', 12)
            .attr('pointer-events', 'none')
            .text(d => `${d.node.data.entity_count || 0} entities`);
    }

    /**
     * Simple hash function for deterministic positioning
     */
    hashCode(str) {
        let hash = 0;
        for (let i = 0; i < str.length; i++) {
            const char = str.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash; // Convert to 32bit integer
        }
        return Math.abs(hash);
    }

    /**
     * Show tooltip
     */
    showTooltip(event, d) {
        const data = d.data || d.node.data;

        let title = data.name;
        let content = '';

        if (data.level === 0) {
            content = `Type: ${data.type}<br>${data.description || ''}`;
        } else {
            content = `Level ${data.level} cluster<br>${data.entity_count || 0} entities`;
            if (data.summary_title) {
                content += `<br><br><strong>${data.summary_title}</strong>`;
            }
        }

        this.tooltip.select('#tooltip-title').html(title);
        this.tooltip.select('#tooltip-content').html(content);
        this.tooltip
            .style('left', (event.pageX + 10) + 'px')
            .style('top', (event.pageY + 10) + 'px')
            .classed('visible', true);
    }

    /**
     * Hide tooltip
     */
    hideTooltip() {
        this.tooltip.classed('visible', false);
    }

    /**
     * Show summary panel
     */
    showSummaryPanel(d) {
        const data = d.data || d.node?.data || d;

        d3.select('#summary-title').text(data.summary_title || data.name);
        d3.select('#summary-content').text(data.summary || 'No summary available');

        const metadata = d3.select('#summary-metadata').html('');

        if (data.level !== undefined) {
            metadata.append('div')
                .attr('class', 'metadata-item')
                .html(`
                    <div class="metadata-label">Level</div>
                    <div class="metadata-value">Level ${data.level}</div>
                `);
        }

        if (data.entity_count) {
            metadata.append('div')
                .attr('class', 'metadata-item')
                .html(`
                    <div class="metadata-label">Entity Count</div>
                    <div class="metadata-value">${data.entity_count}</div>
                `);
        }

        if (data.id) {
            metadata.append('div')
                .attr('class', 'metadata-item')
                .html(`
                    <div class="metadata-label">Cluster ID</div>
                    <div class="metadata-value">${data.id}</div>
                `);
        }

        d3.select('#summary-panel').classed('visible', true);
    }

    /**
     * Hide summary panel
     */
    hideSummaryPanel() {
        d3.select('#summary-panel').classed('visible', false);
    }

    /**
     * Update loading status
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
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    const viewer = new GraphRAG2DOrganicViews();
    viewer.init();
    window.viewer = viewer; // For debugging
});
