/**
 * Our Biggest Deal - Knowledge Graph Visualization
 * Book-specific adaptation of D3.js force-directed graph
 */

class BookKnowledgeGraphVisualization {
    constructor(containerId, section = 'front_matter') {
        this.container = d3.select(containerId);
        this.data = null;
        this.simulation = null;
        this.svg = null;
        this.g = null;
        this.currentSection = section;

        // Visual elements
        this.links = null;
        this.nodes = null;
        this.labels = null;

        // State
        this.selectedNode = null;
        this.filters = {
            domains: new Set(),
            entityTypes: new Set(),
            minImportance: 0.5,  // Lower threshold for book (smaller dataset)
            searchQuery: "",
            maxNodes: 500  // Lower limit for better performance
        };

        // Layout parameters
        this.params = {
            gravity: 0.1,
            charge: -300,
            linkDistance: 100,
            collisionRadius: 15
        };

        // Dimensions
        this.width = 800;
        this.height = 600;

        // Zoom behavior
        this.zoom = null;

        // Initialize
        this.init();
    }

    async init() {
        console.log("Initializing Book Knowledge Graph Visualization...");

        // Show loading overlay
        this.showLoading(true);

        // Load data
        await this.loadData();

        // Set up SVG
        this.setupSVG();

        // Set up controls
        this.setupControls();

        // Create visualization
        this.createVisualization();

        // Hide loading overlay
        this.showLoading(false);

        console.log("Book Knowledge Graph Visualization initialized");
    }

    async loadData(section = this.currentSection) {
        // Book-specific data endpoint
        const dataFile = '/data/knowledge_graph_books/our_biggest_deal_visualization.json';

        try {
            const response = await fetch(dataFile);
            if (response.ok) {
                this.data = await response.json();
                console.log('Loaded book data:', this.data);
            } else {
                throw new Error(`Failed to load: ${response.status}`);
            }
        } catch (error) {
            console.error("Error loading book data:", error);
            this.showError('Failed to load book knowledge graph data');
            return;
        }

        // Initialize filters with all domains and types enabled
        this.data.domains.forEach(d => this.filters.domains.add(d.name));
        this.data.entity_types.forEach(t => this.filters.entityTypes.add(t));
    }

    setupSVG() {
        // Get container dimensions
        const containerRect = this.container.node().getBoundingClientRect();
        this.width = containerRect.width;
        this.height = containerRect.height;

        // Create SVG
        this.svg = this.container.append('svg')
            .attr('width', this.width)
            .attr('height', this.height)
            .attr('class', 'knowledge-graph-svg');

        // Create main group for zoom/pan
        this.g = this.svg.append('g');

        // Set up zoom behavior
        this.zoom = d3.zoom()
            .scaleExtent([0.1, 10])
            .on('zoom', (event) => {
                this.g.attr('transform', event.transform);
            });

        this.svg.call(this.zoom);

        // Add legend
        this.createLegend();
    }

    setupControls() {
        // Domain filters
        const domainContainer = d3.select('#domain-filters');
        this.data.domains.forEach(domain => {
            const item = domainContainer.append('div')
                .attr('class', 'filter-item');

            const checkbox = item.append('input')
                .attr('type', 'checkbox')
                .attr('id', `domain-${domain.name}`)
                .attr('checked', true)
                .on('change', () => this.handleDomainFilter(domain.name, checkbox.property('checked')));

            item.append('label')
                .attr('for', `domain-${domain.name}`)
                .html(`<span class="color-indicator" style="background:${domain.color}"></span> ${domain.name}`);
        });

        // Entity type filters
        const typeContainer = d3.select('#entity-type-filters');
        this.data.entity_types.forEach(type => {
            const item = typeContainer.append('div')
                .attr('class', 'filter-item');

            const checkbox = item.append('input')
                .attr('type', 'checkbox')
                .attr('id', `type-${type}`)
                .attr('checked', true)
                .on('change', () => this.handleTypeFilter(type, checkbox.property('checked')));

            item.append('label')
                .attr('for', `type-${type}`)
                .text(type);
        });

        // Display statistics
        this.updateStatistics();
    }

    createVisualization() {
        // Filter data based on current filters
        const filteredData = this.getFilteredData();

        // Create force simulation
        this.simulation = d3.forceSimulation(filteredData.nodes)
            .force('link', d3.forceLink(filteredData.links)
                .id(d => d.id)
                .distance(this.params.linkDistance))
            .force('charge', d3.forceManyBody()
                .strength(this.params.charge))
            .force('center', d3.forceCenter(this.width / 2, this.height / 2)
                .strength(this.params.gravity))
            .force('collision', d3.forceCollide()
                .radius(d => this.getNodeRadius(d) + this.params.collisionRadius));

        // Create links
        this.links = this.g.append('g')
            .attr('class', 'links')
            .selectAll('line')
            .data(filteredData.links)
            .join('line')
            .attr('class', 'link')
            .attr('stroke-width', d => Math.max(1, d.strength * 3));

        // Create nodes
        this.nodes = this.g.append('g')
            .attr('class', 'nodes')
            .selectAll('g')
            .data(filteredData.nodes)
            .join('g')
            .attr('class', 'node')
            .call(this.drag());

        // Add shapes to nodes
        this.nodes.each((d, i, nodes) => {
            const node = d3.select(nodes[i]);
            this.addNodeShape(node, d);
        });

        // Add labels (for important nodes)
        this.labels = this.g.append('g')
            .attr('class', 'labels')
            .selectAll('text')
            .data(filteredData.nodes.filter(d => d.importance > 0.3))
            .join('text')
            .attr('class', 'node-label')
            .text(d => d.name)
            .attr('dx', 12)
            .attr('dy', 4);

        // Add interactions
        this.nodes
            .on('mouseover', (event, d) => this.handleMouseOver(event, d))
            .on('mouseout', () => this.handleMouseOut())
            .on('click', (event, d) => this.handleNodeClick(event, d));

        // Update positions on simulation tick
        this.simulation.on('tick', () => {
            this.links
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);

            this.nodes
                .attr('transform', d => `translate(${d.x},${d.y})`);

            this.labels
                .attr('x', d => d.x)
                .attr('y', d => d.y);
        });
    }

    addNodeShape(node, d) {
        const radius = this.getNodeRadius(d);
        const color = this.getNodeColor(d);

        if (d.domains.length === 1) {
            // Single domain - solid color circle
            node.append('circle')
                .attr('r', radius)
                .attr('fill', color);
        } else {
            // Multiple domains - gradient
            const gradientId = `gradient-${d.id.replace(/\s+/g, '-')}`;
            const defs = this.svg.select('defs').empty() ?
                this.svg.append('defs') : this.svg.select('defs');

            const gradient = defs.append('linearGradient')
                .attr('id', gradientId);

            gradient.append('stop')
                .attr('offset', '0%')
                .attr('stop-color', d.domain_colors[0]);

            gradient.append('stop')
                .attr('offset', '100%')
                .attr('stop-color', d.domain_colors[d.domain_colors.length - 1]);

            node.append('circle')
                .attr('r', radius)
                .attr('fill', `url(#${gradientId})`);
        }

        // Add white border
        node.append('circle')
            .attr('r', radius)
            .attr('fill', 'none')
            .attr('stroke', 'white')
            .attr('stroke-width', 2);
    }

    getNodeRadius(d) {
        // Scale radius based on importance
        return 4 + (d.importance * 12);
    }

    getNodeColor(d) {
        // Return primary domain color
        return d.domain_colors[0] || '#999';
    }

    getFilteredData() {
        // Filter nodes
        let filteredNodes = this.data.nodes.filter(node => {
            // Check importance threshold
            if (node.importance < this.filters.minImportance) {
                return false;
            }

            // Check entity type filter
            if (!this.filters.entityTypes.has(node.type)) {
                return false;
            }

            // Check domain filter
            const hasMatchingDomain = node.domains.some(d => this.filters.domains.has(d));
            if (!hasMatchingDomain) {
                return false;
            }

            // Check search query
            if (this.filters.searchQuery) {
                const query = this.filters.searchQuery.toLowerCase();
                const nameMatch = node.name.toLowerCase().includes(query);
                const descMatch = node.description.toLowerCase().includes(query);
                if (!nameMatch && !descMatch) {
                    return false;
                }
            }

            return true;
        });

        // Apply hard limit on number of nodes
        if (filteredNodes.length > this.filters.maxNodes) {
            filteredNodes = filteredNodes
                .sort((a, b) => b.importance - a.importance)
                .slice(0, this.filters.maxNodes);
        }

        // Create a set of filtered node IDs
        const nodeIds = new Set(filteredNodes.map(n => n.id));

        // Filter links
        const filteredLinks = this.data.links.filter(link => {
            return nodeIds.has(link.source.id || link.source) &&
                   nodeIds.has(link.target.id || link.target);
        });

        return {
            nodes: filteredNodes,
            links: filteredLinks
        };
    }

    drag() {
        function dragstarted(event) {
            if (!event.active) this.simulation.alphaTarget(0.3).restart();
            event.subject.fx = event.subject.x;
            event.subject.fy = event.subject.y;
        }

        function dragged(event) {
            event.subject.fx = event.x;
            event.subject.fy = event.y;
        }

        function dragended(event) {
            if (!event.active) this.simulation.alphaTarget(0);
            event.subject.fx = null;
            event.subject.fy = null;
        }

        return d3.drag()
            .on('start', dragstarted.bind(this))
            .on('drag', dragged.bind(this))
            .on('end', dragended.bind(this));
    }

    handleMouseOver(event, d) {
        // Show tooltip
        const tooltip = d3.select('#tooltip');
        tooltip
            .classed('visible', true)
            .style('left', (event.pageX + 10) + 'px')
            .style('top', (event.pageY - 10) + 'px')
            .html(`
                <strong>${d.name}</strong><br/>
                <span style="font-size:11px; opacity:0.8">${d.type}</span><br/>
                <span style="font-size:11px">${d.description.substring(0, 150)}...</span>
            `);

        // Highlight node
        this.highlightNode(d);
    }

    handleMouseOut() {
        d3.select('#tooltip').classed('visible', false);
        if (!this.selectedNode) {
            this.clearHighlight();
        }
    }

    handleNodeClick(event, d) {
        event.stopPropagation();

        if (this.selectedNode === d) {
            this.selectedNode = null;
            this.clearHighlight();
            this.closeDetails();
            return;
        }

        this.selectedNode = d;
        this.highlightNode(d);
        this.showDetails(d);
    }

    highlightNode(d) {
        const connectedIds = new Set();
        connectedIds.add(d.id);

        this.data.links.forEach(link => {
            const sourceId = link.source.id || link.source;
            const targetId = link.target.id || link.target;

            if (sourceId === d.id) connectedIds.add(targetId);
            if (targetId === d.id) connectedIds.add(sourceId);
        });

        this.nodes
            .classed('highlighted', node => node.id === d.id)
            .classed('dimmed', node => !connectedIds.has(node.id));

        this.links
            .classed('highlighted', link => {
                const sourceId = link.source.id || link.source;
                const targetId = link.target.id || link.target;
                return sourceId === d.id || targetId === d.id;
            })
            .classed('dimmed', link => {
                const sourceId = link.source.id || link.source;
                const targetId = link.target.id || link.target;
                return sourceId !== d.id && targetId !== d.id;
            });
    }

    clearHighlight() {
        this.nodes
            .classed('highlighted', false)
            .classed('dimmed', false);

        this.links
            .classed('highlighted', false)
            .classed('dimmed', false);
    }

    showDetails(d) {
        const detailsContent = d3.select('#details-content');

        // Find relationships
        const outgoingRelationships = [];
        const incomingRelationships = [];

        this.data.links.forEach(link => {
            const sourceId = link.source.id || link.source;
            const targetId = link.target.id || link.target;

            if (sourceId === d.id) {
                const targetNode = this.data.nodes.find(n => n.id === targetId);
                if (targetNode) {
                    outgoingRelationships.push({
                        type: link.type || link.relationship_type || 'RELATED_TO',
                        target: targetNode.name,
                        page: link.page
                    });
                }
            } else if (targetId === d.id) {
                const sourceNode = this.data.nodes.find(n => n.id === sourceId);
                if (sourceNode) {
                    incomingRelationships.push({
                        type: link.type || link.relationship_type || 'RELATED_TO',
                        source: sourceNode.name,
                        page: link.page
                    });
                }
            }
        });

        // Build relationships HTML
        let relationshipsHtml = '';

        if (outgoingRelationships.length > 0 || incomingRelationships.length > 0) {
            relationshipsHtml = '<div class="entity-relationships">';

            if (outgoingRelationships.length > 0) {
                relationshipsHtml += `
                    <div class="relationship-section">
                        <strong>→ Outgoing (${outgoingRelationships.length})</strong>
                        <div class="relationship-list">
                            ${outgoingRelationships.slice(0, 10).map(rel => `
                                <div class="relationship-item">
                                    <span class="relationship-subject">${d.name}</span>
                                    <span class="relationship-type">${rel.type}</span>
                                    <span class="relationship-object">${rel.target}</span>
                                    <span style="font-size:10px; opacity:0.7">(p.${rel.page})</span>
                                </div>
                            `).join('')}
                            ${outgoingRelationships.length > 10 ? `
                                <div class="relationship-more">+ ${outgoingRelationships.length - 10} more</div>
                            ` : ''}
                        </div>
                    </div>
                `;
            }

            if (incomingRelationships.length > 0) {
                relationshipsHtml += `
                    <div class="relationship-section">
                        <strong>← Incoming (${incomingRelationships.length})</strong>
                        <div class="relationship-list">
                            ${incomingRelationships.slice(0, 10).map(rel => `
                                <div class="relationship-item">
                                    <span class="relationship-subject">${rel.source}</span>
                                    <span class="relationship-type">${rel.type}</span>
                                    <span class="relationship-object">${d.name}</span>
                                    <span style="font-size:10px; opacity:0.7">(p.${rel.page})</span>
                                </div>
                            `).join('')}
                            ${incomingRelationships.length > 10 ? `
                                <div class="relationship-more">+ ${incomingRelationships.length - 10} more</div>
                            ` : ''}
                        </div>
                    </div>
                `;
            }

            relationshipsHtml += '</div>';
        }

        detailsContent.html(`
            <div class="entity-name">${d.name}</div>
            <div class="entity-type">${d.type}</div>
            <div class="entity-domains">
                ${d.domains.map((domain, i) =>
                    `<span class="domain-tag" style="background:${d.domain_colors[i]}">${domain}</span>`
                ).join('')}
            </div>
            <div class="entity-description">${d.description}</div>
            <div class="entity-meta">
                <strong>Importance:</strong> ${(d.importance * 100).toFixed(0)}%<br/>
                <strong>Mentions:</strong> ${d.mention_count}<br/>
                <strong>Pages:</strong> ${d.page_count}
                <div class="episodes-list">
                    ${d.pages.slice(0, 15).map(page =>
                        `<span class="episode-badge">p.${page}</span>`
                    ).join('')}
                    ${d.pages.length > 15 ? `<span class="episode-badge">+${d.pages.length - 15} more</span>` : ''}
                </div>
            </div>
            ${relationshipsHtml}
        `);
    }

    closeDetails() {
        const detailsContent = d3.select('#details-content');
        detailsContent.html('<p class="empty-state">Select an entity to view details</p>');
    }

    createLegend() {
        // Domain legend
        const domainLegend = d3.select('#domain-legend');
        this.data.domains.forEach(domain => {
            domainLegend.append('div')
                .attr('class', 'legend-item')
                .html(`<span class="color-indicator" style="background:${domain.color}"></span> ${domain.name}`);
        });

        // Type legend (top types)
        const typeCounts = {};
        this.data.nodes.forEach(n => {
            typeCounts[n.type] = (typeCounts[n.type] || 0) + 1;
        });

        const topTypes = Object.entries(typeCounts)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 10)
            .map(([type, count]) => ({ type, count }));

        const typeLegend = d3.select('#type-legend');
        topTypes.forEach(({ type, count }) => {
            typeLegend.append('div')
                .attr('class', 'legend-item')
                .html(`<span>${type}</span> <span style="opacity:0.6">(${count})</span>`);
        });
    }

    handleDomainFilter(domain, checked) {
        if (checked) {
            this.filters.domains.add(domain);
        } else {
            this.filters.domains.delete(domain);
        }
        this.updateVisualization();
    }

    handleTypeFilter(type, checked) {
        if (checked) {
            this.filters.entityTypes.add(type);
        } else {
            this.filters.entityTypes.delete(type);
        }
        this.updateVisualization();
    }

    updateImportanceFilter(value) {
        this.filters.minImportance = parseFloat(value);
        d3.select('#importance-value').text(value);
        this.updateVisualization();
    }

    updateVisualization() {
        // Clear existing
        this.g.selectAll('*').remove();

        // Recreate visualization
        this.createVisualization();

        // Update statistics
        this.updateStatistics();
    }

    updateStatistics() {
        const filteredData = this.getFilteredData();

        const stats = d3.select('#stats-display');
        stats.html(`
            <div class="stat-item">
                <span class="stat-label">Visible Nodes:</span>
                <span class="stat-value">${filteredData.nodes.length}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Visible Links:</span>
                <span class="stat-value">${filteredData.links.length}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Total Entities:</span>
                <span class="stat-value">${this.data.nodes.length}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Total Relationships:</span>
                <span class="stat-value">${this.data.links.length}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Domains:</span>
                <span class="stat-value">${this.data.statistics.total_communities}</span>
            </div>
        `);
    }

    showLoading(show) {
        const overlay = d3.select('#loading-overlay');
        overlay.classed('hidden', !show);
    }

    showError(message) {
        const overlay = d3.select('#loading-overlay');
        overlay.html(`
            <div style="text-align:center; color:#e74c3c;">
                <h2>Error</h2>
                <p>${message}</p>
            </div>
        `);
    }

    // Public control methods
    updateGravity(value) {
        this.params.gravity = parseFloat(value);
        d3.select('#gravity-value').text(value);
        if (this.simulation) {
            this.simulation.force('center').strength(this.params.gravity);
            this.simulation.alpha(0.3).restart();
        }
    }

    updateCharge(value) {
        this.params.charge = parseFloat(value);
        d3.select('#charge-value').text(value);
        if (this.simulation) {
            this.simulation.force('charge').strength(this.params.charge);
            this.simulation.alpha(0.3).restart();
        }
    }

    updateLinkDistance(value) {
        this.params.linkDistance = parseFloat(value);
        d3.select('#link-distance-value').text(value);
        if (this.simulation) {
            this.simulation.force('link').distance(this.params.linkDistance);
            this.simulation.alpha(0.3).restart();
        }
    }

    resetView() {
        this.svg.transition()
            .duration(750)
            .call(this.zoom.transform, d3.zoomIdentity);

        if (this.simulation) {
            this.simulation.alpha(1).restart();
        }
    }

    searchEntities(query) {
        this.filters.searchQuery = query;
        this.updateVisualization();

        const results = this.data.nodes.filter(n => {
            const q = query.toLowerCase();
            return n.name.toLowerCase().includes(q) ||
                   n.description.toLowerCase().includes(q);
        }).slice(0, 10);

        const resultsContainer = d3.select('#search-results');
        resultsContainer.html('');

        if (results.length > 0) {
            results.forEach(result => {
                resultsContainer.append('div')
                    .attr('class', 'search-result-item')
                    .text(result.name)
                    .on('click', () => {
                        this.focusOnNode(result);
                    });
            });
        }
    }

    clearSearch() {
        this.filters.searchQuery = "";
        d3.select('#search-input').property('value', '');
        d3.select('#search-results').html('');
        this.updateVisualization();
    }

    focusOnNode(node) {
        const nodeData = this.data.nodes.find(n => n.id === node.id);
        if (!nodeData || !nodeData.x || !nodeData.y) return;

        const scale = 2;
        const x = -nodeData.x * scale + this.width / 2;
        const y = -nodeData.y * scale + this.height / 2;

        this.svg.transition()
            .duration(750)
            .call(
                this.zoom.transform,
                d3.zoomIdentity.translate(x, y).scale(scale)
            );

        this.handleNodeClick({ stopPropagation: () => {} }, nodeData);
    }
}

// Global functions for HTML event handlers
let vizInstance = null;

function updateImportanceFilter(value) {
    if (vizInstance) vizInstance.updateImportanceFilter(value);
}

function updateGravity(value) {
    if (vizInstance) vizInstance.updateGravity(value);
}

function updateCharge(value) {
    if (vizInstance) vizInstance.updateCharge(value);
}

function updateLinkDistance(value) {
    if (vizInstance) vizInstance.updateLinkDistance(value);
}

function resetView() {
    if (vizInstance) vizInstance.resetView();
}

function searchEntities() {
    const query = document.getElementById('search-input').value;
    if (vizInstance) vizInstance.searchEntities(query);
}

function clearSearch() {
    if (vizInstance) vizInstance.clearSearch();
}

function closeDetails() {
    if (vizInstance) {
        vizInstance.selectedNode = null;
        vizInstance.clearHighlight();
        vizInstance.closeDetails();
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    vizInstance = new BookKnowledgeGraphVisualization('#graph-svg-container', 'front_matter');
    window.bookKnowledgeGraph = vizInstance; // For debugging
});

// Handle search input Enter key
document.addEventListener('DOMContentLoaded', () => {
    const searchInput = document.getElementById('search-input');
    if (searchInput) {
        searchInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                searchEntities();
            }
        });
    }
});
