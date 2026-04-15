/**
 * Book Knowledge Graph Visualization
 * Simpler visualization for book relationships
 */

class BookKnowledgeGraph {
    constructor(containerId) {
        this.container = d3.select(containerId);
        this.bookData = null;
        this.simulation = null;
        this.svg = null;
        this.g = null;

        // Data arrays
        this.nodesData = null;  // The actual data
        this.linksData = null;  // The actual link data

        // Visual elements (D3 selections)
        this.linkElements = null;
        this.nodeElements = null;
        this.labelElements = null;

        // State
        this.selectedNode = null;
        this.filters = {
            entityTypes: new Set(),
            minConfidence: 0.5,  // Lower threshold for book (p_true)
            searchQuery: "",
            maxNodes: 500
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
        console.log("Initializing Book Knowledge Graph...");

        // Show loading
        this.showLoading(true);

        // Load data
        await this.loadData();

        // Set up SVG
        this.setupSVG();

        // Set up controls
        this.setupControls();

        // Create visualization
        this.createVisualization();

        // Hide loading
        this.showLoading(false);

        console.log("Book Knowledge Graph initialized");
    }

    async loadData() {
        try {
            const response = await fetch('/data/knowledge_graph_books_v3_2_2/our_biggest_deal_v3_2_2.json');
            this.bookData = await response.json();
            console.log("Loaded book data:", this.bookData);

            // Transform relationships into nodes and links
            this.processData();

        } catch (error) {
            console.error("Error loading book data:", error);
            this.showError("Failed to load book knowledge graph data");
        }
    }

    processData() {
        // Build nodes and links from relationships
        const nodeMap = new Map();
        const links = [];

        this.bookData.relationships.forEach(rel => {
            // Add source node
            if (!nodeMap.has(rel.source)) {
                nodeMap.set(rel.source, {
                    id: rel.source,
                    name: rel.source,
                    type: rel.source_type || "Unknown",
                    mentions: 0,
                    totalConfidence: 0,
                    relationships: []
                });
            }

            // Add target node
            if (!nodeMap.has(rel.target)) {
                nodeMap.set(rel.target, {
                    id: rel.target,
                    name: rel.target,
                    type: rel.target_type || "Unknown",
                    mentions: 0,
                    totalConfidence: 0,
                    relationships: []
                });
            }

            // Update node stats
            const sourceNode = nodeMap.get(rel.source);
            const targetNode = nodeMap.get(rel.target);

            sourceNode.mentions++;
            sourceNode.totalConfidence += rel.p_true;
            targetNode.mentions++;
            targetNode.totalConfidence += rel.p_true;

            // Add link
            links.push({
                source: rel.source,
                target: rel.target,
                type: rel.relationship,
                confidence: rel.p_true,
                evidence: rel.evidence_text,
                page: rel.evidence.page_number
            });
        });

        // Calculate importance for each node
        this.nodesData = Array.from(nodeMap.values()).map(node => ({
            ...node,
            importance: node.totalConfidence / Math.max(1, node.mentions),
            avgConfidence: node.totalConfidence / Math.max(1, node.mentions)
        }));

        this.linksData = links;

        // Get unique entity types
        this.entityTypes = [...new Set(this.nodesData.map(n => n.type))].sort();

        // Initialize filters with all types enabled
        this.entityTypes.forEach(t => this.filters.entityTypes.add(t));

        console.log(`Processed ${this.nodesData.length} nodes and ${this.linksData.length} links`);
    }

    setupSVG() {
        const containerRect = this.container.node().getBoundingClientRect();
        this.width = containerRect.width;
        this.height = containerRect.height;

        this.svg = this.container.append('svg')
            .attr('width', this.width)
            .attr('height', this.height)
            .attr('class', 'knowledge-graph-svg');

        this.g = this.svg.append('g');

        // Set up zoom
        this.zoom = d3.zoom()
            .scaleExtent([0.1, 10])
            .on('zoom', (event) => {
                this.g.attr('transform', event.transform);
            });

        this.svg.call(this.zoom);
    }

    setupControls() {
        // Entity type filters
        const typeContainer = d3.select('#entity-type-filters');
        this.entityTypes.forEach(type => {
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

        // Hide domain filters section (not used for book)
        d3.selectAll('.control-section').filter(function() {
            return this.querySelector('#domain-filters') !== null;
        }).style('display', 'none');

        // Update statistics
        this.updateStatistics();
    }

    createVisualization() {
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
        this.linkElements = this.g.append('g')
            .attr('class', 'links')
            .selectAll('line')
            .data(filteredData.links)
            .join('line')
            .attr('class', 'link')
            .attr('stroke-width', d => Math.max(1, d.confidence * 3));

        // Create nodes
        this.nodeElements = this.g.append('g')
            .attr('class', 'nodes')
            .selectAll('g')
            .data(filteredData.nodes)
            .join('g')
            .attr('class', 'node')
            .call(this.drag());

        // Add circles to nodes
        this.nodeElements.append('circle')
            .attr('r', d => this.getNodeRadius(d))
            .attr('fill', d => this.getNodeColor(d.type));

        // Add labels
        this.labelElements = this.g.append('g')
            .attr('class', 'labels')
            .selectAll('text')
            .data(filteredData.nodes.filter(d => d.importance > 0.3))
            .join('text')
            .attr('class', 'node-label')
            .text(d => d.name)
            .attr('dx', 12)
            .attr('dy', 4);

        // Add interactions
        this.nodeElements
            .on('mouseover', (event, d) => this.handleMouseOver(event, d))
            .on('mouseout', () => this.handleMouseOut())
            .on('click', (event, d) => this.handleNodeClick(event, d));

        // Update positions on simulation tick
        this.simulation.on('tick', () => {
            this.linkElements
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);

            this.nodeElements
                .attr('transform', d => `translate(${d.x},${d.y})`);

            this.labelElements
                .attr('x', d => d.x)
                .attr('y', d => d.y);
        });
    }

    getFilteredData() {
        let filteredNodes = this.nodesData.filter(node => {
            if (node.avgConfidence < this.filters.minConfidence) return false;
            if (!this.filters.entityTypes.has(node.type)) return false;
            // Note: We don't filter by search query here anymore
            // Search only populates the results list, doesn't filter the graph
            return true;
        });

        // Apply limit
        if (filteredNodes.length > this.filters.maxNodes) {
            filteredNodes = filteredNodes
                .sort((a, b) => b.importance - a.importance)
                .slice(0, this.filters.maxNodes);
        }

        const nodeIds = new Set(filteredNodes.map(n => n.id));

        const filteredLinks = this.linksData.filter(link => {
            return nodeIds.has(link.source.id || link.source) &&
                   nodeIds.has(link.target.id || link.target);
        });

        return { nodes: filteredNodes, links: filteredLinks };
    }

    getNodeRadius(d) {
        return 4 + (d.importance * 12);
    }

    getNodeColor(type) {
        const colors = {
            'Person': '#3498db',
            'Organization': '#e74c3c',
            'Concept': '#2ecc71',
            'Place': '#f39c12',
            'Practice': '#9b59b6',
            'Product': '#1abc9c'
        };
        return colors[type] || '#95a5a6';
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
        const tooltip = d3.select('#tooltip');
        tooltip
            .classed('visible', true)
            .style('left', (event.pageX + 10) + 'px')
            .style('top', (event.pageY - 10) + 'px')
            .html(`
                <strong>${d.name}</strong><br/>
                <span style="font-size:11px; opacity:0.8">${d.type}</span><br/>
                <span style="font-size:11px">Mentions: ${d.mentions} | Avg Confidence: ${(d.avgConfidence * 100).toFixed(0)}%</span>
            `);

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

        this.linksData.forEach(link => {
            const sourceId = link.source.id || link.source;
            const targetId = link.target.id || link.target;

            if (sourceId === d.id) connectedIds.add(targetId);
            if (targetId === d.id) connectedIds.add(sourceId);
        });

        this.nodeElements
            .classed('highlighted', node => node.id === d.id)
            .classed('dimmed', node => !connectedIds.has(node.id));

        this.linkElements
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
        this.nodeElements
            .classed('highlighted', false)
            .classed('dimmed', false);

        this.linkElements
            .classed('highlighted', false)
            .classed('dimmed', false);
    }

    showDetails(d) {
        const detailsContent = d3.select('#details-content');

        // Find relationships
        const outgoing = [];
        const incoming = [];

        this.linksData.forEach(link => {
            const sourceId = link.source.id || link.source;
            const targetId = link.target.id || link.target;

            if (sourceId === d.id) {
                const targetNode = this.nodesData.find(n => n.id === targetId);
                if (targetNode) {
                    outgoing.push({ ...link, targetName: targetNode.name });
                }
            } else if (targetId === d.id) {
                const sourceNode = this.nodesData.find(n => n.id === sourceId);
                if (sourceNode) {
                    incoming.push({ ...link, sourceName: sourceNode.name });
                }
            }
        });

        let relHtml = '';
        if (outgoing.length > 0 || incoming.length > 0) {
            relHtml = '<div class="entity-relationships">';

            if (outgoing.length > 0) {
                relHtml += `
                    <div class="relationship-section">
                        <strong>→ Outgoing (${outgoing.length})</strong>
                        <div class="relationship-list">
                            ${outgoing.slice(0, 10).map(rel => `
                                <div class="relationship-item">
                                    <span class="relationship-subject">${d.name}</span>
                                    <span class="relationship-type">${rel.type}</span>
                                    <span class="relationship-object">${rel.targetName}</span>
                                    <span style="font-size:10px; opacity:0.7"> (p=${(rel.confidence * 100).toFixed(0)}%, pg${rel.page})</span>
                                </div>
                            `).join('')}
                            ${outgoing.length > 10 ? `<div class="relationship-more">+ ${outgoing.length - 10} more</div>` : ''}
                        </div>
                    </div>
                `;
            }

            if (incoming.length > 0) {
                relHtml += `
                    <div class="relationship-section">
                        <strong>← Incoming (${incoming.length})</strong>
                        <div class="relationship-list">
                            ${incoming.slice(0, 10).map(rel => `
                                <div class="relationship-item">
                                    <span class="relationship-subject">${rel.sourceName}</span>
                                    <span class="relationship-type">${rel.type}</span>
                                    <span class="relationship-object">${d.name}</span>
                                    <span style="font-size:10px; opacity:0.7"> (p=${(rel.confidence * 100).toFixed(0)}%, pg${rel.page})</span>
                                </div>
                            `).join('')}
                            ${incoming.length > 10 ? `<div class="relationship-more">+ ${incoming.length - 10} more</div>` : ''}
                        </div>
                    </div>
                `;
            }

            relHtml += '</div>';
        }

        detailsContent.html(`
            <div class="entity-name">${d.name}</div>
            <div class="entity-type">${d.type}</div>
            <div class="entity-meta">
                <strong>Importance:</strong> ${(d.importance * 100).toFixed(0)}%<br/>
                <strong>Mentions:</strong> ${d.mentions}<br/>
                <strong>Avg Confidence:</strong> ${(d.avgConfidence * 100).toFixed(0)}%
            </div>
            ${relHtml}
        `);
    }

    closeDetails() {
        d3.select('#details-content').html('<p class="empty-state">Select an entity to view details</p>');
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
        this.filters.minConfidence = parseFloat(value);
        d3.select('#importance-value').text(value);
        this.updateVisualization();
    }

    updateVisualization() {
        this.g.selectAll('*').remove();
        this.createVisualization();
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
                <span class="stat-value">${this.nodesData.length}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Total Relationships:</span>
                <span class="stat-value">${this.bookData.relationships.length}</span>
            </div>
        `);
    }

    showLoading(show) {
        d3.select('#loading-overlay').classed('hidden', !show);
    }

    showError(message) {
        d3.select('#loading-overlay').html(`
            <div style="text-align:center; color:#e74c3c;">
                <h2>Error</h2>
                <p>${message}</p>
            </div>
        `);
    }

    // Public methods
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
        // Don't filter by query in main graph - let user see all nodes
        // but highlight search results
        this.filters.searchQuery = "";  // Clear search filter

        const q = query.toLowerCase();
        const results = this.nodesData.filter(n => {
            return n.name.toLowerCase().includes(q);
        }).slice(0, 20);  // Show more results

        const resultsContainer = d3.select('#search-results');
        resultsContainer.html('');

        if (results.length > 0) {
            resultsContainer.append('div')
                .attr('class', 'search-result-header')
                .html(`<strong>Found ${results.length} matches:</strong>`);

            results.forEach(result => {
                resultsContainer.append('div')
                    .attr('class', 'search-result-item')
                    .html(`
                        ${result.name}
                        <span style="font-size:10px; opacity:0.7;">
                            (${result.type}, ${result.mentions} mentions, ${(result.avgConfidence * 100).toFixed(0)}% conf)
                        </span>
                    `)
                    .on('click', () => {
                        this.focusOnNode(result);
                    });
            });
        } else if (query) {
            resultsContainer.html('<div class="search-result-item" style="opacity:0.6;">No results found</div>');
        }

        // Update visualization without search filter
        this.updateVisualization();
    }

    clearSearch() {
        this.filters.searchQuery = "";
        d3.select('#search-input').property('value', '');
        d3.select('#search-results').html('');
        this.updateVisualization();
    }

    focusOnNode(node) {
        // If node is filtered out by importance, temporarily lower threshold
        if (node.avgConfidence < this.filters.minConfidence) {
            this.filters.minConfidence = Math.max(0, node.avgConfidence - 0.05);
            d3.select('#importance-slider').property('value', this.filters.minConfidence);
            d3.select('#importance-value').text(this.filters.minConfidence.toFixed(2));
            this.updateVisualization();
        }

        // Small delay to let visualization update
        setTimeout(() => {
            const nodeData = this.nodesData.find(n => n.id === node.id);
            if (!nodeData) return;

            // Wait for node position to be calculated
            const checkPosition = setInterval(() => {
                if (nodeData.x && nodeData.y) {
                    clearInterval(checkPosition);

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
            }, 50);

            // Safety timeout
            setTimeout(() => clearInterval(checkPosition), 2000);
        }, 100);
    }
}

// Global functions
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

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    vizInstance = new BookKnowledgeGraph('#graph-svg-container');
    window.bookKG = vizInstance;
});

// Handle Enter key in search
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
