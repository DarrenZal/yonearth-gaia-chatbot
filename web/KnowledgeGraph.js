/**
 * Knowledge Graph Visualization
 * Interactive D3.js force-directed graph for YonEarth entities
 */

class KnowledgeGraphVisualization {
    constructor(containerId, version = 'current') {
        this.container = d3.select(containerId);
        this.data = null;
        this.simulation = null;
        this.svg = null;
        this.g = null;
        this.currentVersion = version;  // Track which version is loaded
        this.nodeTypes = []; // Unique node types from data

        // Visual elements
        this.links = null;
        this.nodes = null;
        this.labels = null;

        // State
        this.selectedNode = null;
        this.filters = {
            domains: new Set(),
            entityTypes: new Set(),
            minImportance: 0.7,  // Start with higher threshold to avoid rendering too many nodes
            searchQuery: "",
            maxNodes: 1000  // Hard limit on nodes to prevent browser freeze
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
        console.log("Initializing Knowledge Graph Visualization...");

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

        console.log("Knowledge Graph Visualization initialized");
    }

    async loadData(version = this.currentVersion) {
        // Determine which data source to use based on version
        let apiEndpoint, fileEndpoint;

        if (version === 'v3.2.2') {
            apiEndpoint = '/api/knowledge-graph/data/v3.2.2';
            fileEndpoint = '/data/knowledge_graph_v3_2_2/visualization_data.json';
        } else if (version === 'unified') {
            apiEndpoint = '/api/knowledge-graph/data/unified';
            fileEndpoint = '/data/knowledge_graph_unified/visualization_data.json';
        } else {
            apiEndpoint = '/api/knowledge-graph/data';
            fileEndpoint = '/data/knowledge_graph/visualization_data.json';
        }

        try {
            // Try to load from API first
            const response = await fetch(apiEndpoint);
            if (response.ok) {
                this.data = await response.json();
                console.log(`Loaded ${version} data from API:`, this.data);
            } else {
                throw new Error("API not available");
            }
        } catch (error) {
            console.log(`Loading ${version} from local file...`);
            try {
                const response = await fetch(fileEndpoint);
                this.data = await response.json();
                console.log(`Loaded ${version} data from file:`, this.data);
            } catch (fileError) {
                console.error("Error loading data:", fileError);
                this.showError(`Failed to load knowledge graph data (${version})`);
                return;
            }
        }

        // Initialize filters with all domains and actual node types enabled
        this.filters.domains.clear();
        this.filters.entityTypes.clear();
        this.data.domains.forEach(d => this.filters.domains.add(d.name));
        // Build unique node type list from nodes for accurate filtering/UI
        this.nodeTypes = Array.from(new Set((this.data.nodes || []).map(n => n.type))).sort();
        this.nodeTypes.forEach(t => this.filters.entityTypes.add(t));

        // Adjust default importance threshold to match dataset distribution
        const avgImp = this.data.statistics && typeof this.data.statistics.avg_importance === 'number'
            ? this.data.statistics.avg_importance : null;
        if (version === 'unified' || (avgImp !== null && avgImp <= 0.55)) {
            // Unified dataset centers around ~0.5; start lower so nodes appear
            this.filters.minImportance = Math.max(0, (avgImp ?? 0.5) - 0.1);
        } else {
            this.filters.minImportance = 0.7;
        }
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

        // Entity type filters (use actual node types, not the full catalog)
        const typeContainer = d3.select('#entity-type-filters');
        this.nodeTypes.forEach(type => {
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

        // Sync importance slider to current threshold
        const slider = d3.select('#importance-slider');
        const label = d3.select('#importance-value');
        if (!slider.empty()) slider.property('value', this.filters.minImportance);
        if (!label.empty()) label.text(this.filters.minImportance.toFixed(2));
    }

    createVisualization() {
        // Filter data based on current filters
        const filteredData = this.getFilteredData();
        
        // If filters hide everything, prompt the user and exit
        if (filteredData.nodes.length === 0) {
            this.showError('No nodes match current filters. Try lowering the Importance slider or clearing filters.');
            return;
        } else {
            // Ensure any prior overlay is hidden
            this.showLoading(false);
        }

        // Create force simulation
        this.simulation = d3.forceSimulation(filteredData.nodes)
            .force('link', d3.forceLink(filteredData.links)
                .id(d => d.id)
                .distance(this.params.linkDistance))
            .force('charge', d3.forceManyBody()
                .strength(this.params.charge))
            // Center force (no strength API on forceCenter)
            .force('center', d3.forceCenter(this.width / 2, this.height / 2))
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

        // Add labels (only for important nodes to avoid clutter)
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
            // Multiple domains - pie chart or gradient
            if (d.domains.length === 2) {
                // Gradient for 2 domains
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
                    .attr('stop-color', d.domain_colors[1]);

                node.append('circle')
                    .attr('r', radius)
                    .attr('fill', `url(#${gradientId})`);
            } else {
                // Pie chart for 3+ domains
                const pie = d3.pie().value(1);
                const arc = d3.arc()
                    .innerRadius(0)
                    .outerRadius(radius);

                const pieData = pie(d.domain_colors);

                node.selectAll('path')
                    .data(pieData)
                    .join('path')
                    .attr('d', arc)
                    .attr('fill', (_, i) => d.domain_colors[i]);
            }
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

            // Check domain filter (node must have at least one matching domain)
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

        // Apply hard limit on number of nodes (performance protection)
        if (filteredNodes.length > this.filters.maxNodes) {
            // Sort by importance and take top N
            filteredNodes = filteredNodes
                .sort((a, b) => b.importance - a.importance)
                .slice(0, this.filters.maxNodes);
        }

        // Create a set of filtered node IDs for quick lookup
        const nodeIds = new Set(filteredNodes.map(n => n.id));

        // Filter links (both source and target must be in filtered nodes)
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

        // Highlight node and connections
        this.highlightNode(d);
    }

    handleMouseOut() {
        // Hide tooltip
        d3.select('#tooltip').classed('visible', false);

        // Reset highlighting
        if (!this.selectedNode) {
            this.clearHighlight();
        }
    }

    handleNodeClick(event, d) {
        event.stopPropagation();

        // If clicking the same node, deselect
        if (this.selectedNode === d) {
            this.selectedNode = null;
            this.clearHighlight();
            this.closeDetails();
            return;
        }

        // Select new node
        this.selectedNode = d;
        this.highlightNode(d);
        this.showDetails(d);
    }

    highlightNode(d) {
        // Get connected node IDs
        const connectedIds = new Set();
        connectedIds.add(d.id);

        this.data.links.forEach(link => {
            const sourceId = link.source.id || link.source;
            const targetId = link.target.id || link.target;

            if (sourceId === d.id) connectedIds.add(targetId);
            if (targetId === d.id) connectedIds.add(sourceId);
        });

        // Highlight nodes
        this.nodes
            .classed('highlighted', node => node.id === d.id)
            .classed('dimmed', node => !connectedIds.has(node.id));

        // Highlight links
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

        // Find all relationships (edges) involving this node
        const outgoingRelationships = [];
        const incomingRelationships = [];

        this.data.links.forEach(link => {
            const sourceId = link.source.id || link.source;
            const targetId = link.target.id || link.target;

            if (sourceId === d.id) {
                // Outgoing: this node → relationship → target
                const targetNode = this.data.nodes.find(n => n.id === targetId);
                if (targetNode) {
                    outgoingRelationships.push({
                        type: link.type || link.relationship_type || 'RELATED_TO',
                        target: targetNode.name,
                        strength: link.strength || 1.0
                    });
                }
            } else if (targetId === d.id) {
                // Incoming: source → relationship → this node
                const sourceNode = this.data.nodes.find(n => n.id === sourceId);
                if (sourceNode) {
                    incomingRelationships.push({
                        type: link.type || link.relationship_type || 'RELATED_TO',
                        source: sourceNode.name,
                        strength: link.strength || 1.0
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
                                </div>
                            `).join('')}
                            ${outgoingRelationships.length > 10 ? `
                                <div class="relationship-more">+ ${outgoingRelationships.length - 10} more outgoing</div>
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
                                </div>
                            `).join('')}
                            ${incomingRelationships.length > 10 ? `
                                <div class="relationship-more">+ ${incomingRelationships.length - 10} more incoming</div>
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
                <strong>Episodes:</strong> ${d.episode_count}
                <div class="episodes-list">
                    ${d.episodes.slice(0, 10).map(ep =>
                        `<span class="episode-badge">Ep ${ep}</span>`
                    ).join('')}
                    ${d.episodes.length > 10 ? `<span class="episode-badge">+${d.episodes.length - 10} more</span>` : ''}
                </div>
                ${d.aliases && d.aliases.length > 0 ? `
                    <br/><strong>Aliases:</strong> ${d.aliases.join(', ')}
                ` : ''}
            </div>
            ${relationshipsHtml}
            <a href="#" class="wiki-link" onclick="openWiki('${d.id}'); return false;">
                View in Wiki →
            </a>
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

        // Type legend (show top types only)
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

        // Recreate visualization with filtered data
        this.createVisualization();

        // Update statistics
        this.updateStatistics();
    }

    updateStatistics() {
        const filteredData = this.getFilteredData();

        // Count nodes before max limit
        const preFilterCount = this.data.nodes.filter(node => {
            if (node.importance < this.filters.minImportance) return false;
            if (!this.filters.entityTypes.has(node.type)) return false;
            const hasMatchingDomain = node.domains.some(d => this.filters.domains.has(d));
            if (!hasMatchingDomain) return false;
            if (this.filters.searchQuery) {
                const query = this.filters.searchQuery.toLowerCase();
                const nameMatch = node.name.toLowerCase().includes(query);
                const descMatch = node.description.toLowerCase().includes(query);
                if (!nameMatch && !descMatch) return false;
            }
            return true;
        }).length;

        const hitLimit = preFilterCount > this.filters.maxNodes;

        const stats = d3.select('#stats-display');
        stats.html(`
            <div class="stat-item">
                <span class="stat-label">Visible Nodes:</span>
                <span class="stat-value">${filteredData.nodes.length}${hitLimit ? ' (limited)' : ''}</span>
            </div>
            ${hitLimit ? `
                <div class="stat-item" style="color: #e67e22; font-size: 11px; margin-top: 5px;">
                    ⚠️ Showing top ${this.filters.maxNodes} by importance
                </div>
            ` : ''}
            <div class="stat-item">
                <span class="stat-label">Visible Links:</span>
                <span class="stat-value">${filteredData.links.length}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Total Entities:</span>
                <span class="stat-value">${this.data.nodes.length.toLocaleString()}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Communities:</span>
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
        overlay.classed('hidden', false);
        overlay.html(`
            <div style="text-align:center; color:#e74c3c;">
                <h2>Error</h2>
                <p>${message}</p>
            </div>
        `);
    }

    // Public methods for controls

    updateGravity(value) {
        this.params.gravity = parseFloat(value);
        d3.select('#gravity-value').text(value);
        // Note: d3.forceCenter has no strength; nudge simulation instead
        if (this.simulation) {
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
        // Reset zoom
        this.svg.transition()
            .duration(750)
            .call(this.zoom.transform, d3.zoomIdentity);

        // Restart simulation
        if (this.simulation) {
            this.simulation.alpha(1).restart();
        }
    }

    searchEntities(query) {
        this.filters.searchQuery = query;
        this.updateVisualization();

        // Also show search results list
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
        // Find node position
        const nodeData = this.data.nodes.find(n => n.id === node.id);
        if (!nodeData || !nodeData.x || !nodeData.y) return;

        // Zoom to node
        const scale = 2;
        const x = -nodeData.x * scale + this.width / 2;
        const y = -nodeData.y * scale + this.height / 2;

        this.svg.transition()
            .duration(750)
            .call(
                this.zoom.transform,
                d3.zoomIdentity.translate(x, y).scale(scale)
            );

        // Select and highlight node
        this.handleNodeClick({ stopPropagation: () => {} }, nodeData);
    }

    async reloadWithVersion(version) {
        console.log(`Reloading knowledge graph with version: ${version}`);

        // Show loading overlay
        this.showLoading(true);

        // Update current version
        this.currentVersion = version;

        // Clear existing visualization
        if (this.svg) {
            this.svg.remove();
            this.svg = null;
        }

        // Clear filters and state
        this.filters.domains.clear();
        this.filters.entityTypes.clear();
        this.selectedNode = null;
        this.nodeTypes = [];

        // Clear controls
        d3.select('#domain-filters').html('');
        d3.select('#entity-type-filters').html('');
        d3.select('#domain-legend').html('');
        d3.select('#type-legend').html('');

        // Reload data
        await this.loadData(version);

        // Recreate visualization
        this.setupSVG();
        this.setupControls();
        this.createVisualization();

        // Hide loading overlay
        this.showLoading(false);

        console.log(`Knowledge graph reloaded with version: ${version}`);
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

function openWiki(entityId) {
    // Navigate to wiki page (to be implemented)
    console.log("Opening wiki for:", entityId);
    alert(`Wiki integration coming soon!\nEntity: ${entityId}`);
}

function loadKnowledgeGraphVersion(version) {
    if (vizInstance) {
        vizInstance.reloadWithVersion(version);
    } else {
        console.error("Visualization instance not available");
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    // Check for version in URL parameters
    const urlParams = new URLSearchParams(window.location.search);
    const initialVersion = urlParams.get('version') || 'current';

    // Set dropdown to match initial version
    const versionSelect = document.getElementById('kg-version-select');
    if (versionSelect) {
        versionSelect.value = initialVersion;
    }

    // Initialize visualization with selected version
    vizInstance = new KnowledgeGraphVisualization('#graph-svg-container', initialVersion);
    window.knowledgeGraph = vizInstance; // For debugging
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
