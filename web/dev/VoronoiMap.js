/**
 * Voronoi Map Visualization
 * Recursive hierarchical Voronoi diagram for GraphRAG clusters.
 * Supports arbitrary depth drill-down from natural Leiden hierarchy.
 */

class VoronoiMapVisualization {
    constructor(containerId) {
        this.container = d3.select(containerId);
        this.tooltip = d3.select('#tooltip');
        this.data = null;
        this.entityIndex = null;

        // Navigation state — stack-based for arbitrary depth
        this.navigationStack = []; // [{items, parent, color}] — each level's data
        this.currentItems = null; // currently displayed items (groups/children)
        this.currentParent = null; // parent node of current view
        this.currentColor = null; // inherited color from top-level group

        // Dimensions
        this.margin = { top: 20, right: 20, bottom: 20, left: 20 };
        this.width = 800;
        this.height = 600;

        // SVG elements
        this.svg = null;
        this.g = null;

        // Scales
        this.xScale = null;
        this.yScale = null;

        // Animation
        this.transitionDuration = 600;

        // Entity type colors
        this.typeColors = {
            'CONCEPT': '#4CAF50',
            'PERSON': '#2196F3',
            'ORGANIZATION': '#FF9800',
            'PLACE': '#9C27B0',
            'PRODUCT': '#F44336',
            'PRACTICE': '#00BCD4',
            'EVENT': '#FFEB3B',
            'RESOURCE': '#795548',
            'UNKNOWN': '#999',
        };

        this.init();
    }

    async init() {
        this.setupSVG();
        await this.loadData();
        if (this.data) {
            this.renderTopLevel();
            this.updateStats();
            this.setupSearch();
            this.setupKeyboard();
        }
        this.hideLoading();
    }

    setupSVG() {
        const rect = this.container.node().getBoundingClientRect();
        this.width = rect.width - this.margin.left - this.margin.right;
        this.height = rect.height - this.margin.top - this.margin.bottom;

        this.svg = this.container.append('svg')
            .attr('class', 'voronoi-svg')
            .attr('width', this.width + this.margin.left + this.margin.right)
            .attr('height', this.height + this.margin.top + this.margin.bottom);

        this.svg.append('rect')
            .attr('width', '100%')
            .attr('height', '100%')
            .attr('fill', '#fafafa');

        this.g = this.svg.append('g')
            .attr('transform', `translate(${this.margin.left},${this.margin.top})`);
    }

    async loadData() {
        try {
            let response;
            try {
                response = await fetch('/api/voronoi_data');
                if (!response.ok) throw new Error('API not available');
            } catch {
                response = await fetch('/data/processed/voronoi_hierarchy.json');
            }
            this.data = await response.json();
            console.log('Loaded voronoi data:', this.data.metadata);
        } catch (error) {
            console.error('Error loading voronoi data:', error);
            try {
                const response = await fetch('data/processed/voronoi_hierarchy.json');
                this.data = await response.json();
            } catch (e2) {
                console.error('All data loading attempts failed:', e2);
            }
        }
    }

    async loadEntityIndex() {
        if (this.entityIndex) return;
        try {
            let response;
            try {
                response = await fetch('/data/processed/voronoi_entity_index.json');
                if (!response.ok) throw new Error();
            } catch {
                response = await fetch('data/processed/voronoi_entity_index.json');
            }
            this.entityIndex = await response.json();
            console.log('Loaded entity index:', Object.keys(this.entityIndex).length, 'entities');
        } catch (error) {
            console.error('Error loading entity index:', error);
            this.entityIndex = {};
        }
    }

    // --- Scales ---

    computeScales(items, accessor) {
        const xs = items.map(d => accessor(d).x);
        const ys = items.map(d => accessor(d).y);
        const padding = 40;

        this.xScale = d3.scaleLinear()
            .domain([d3.min(xs), d3.max(xs)])
            .range([padding, this.width - padding]);

        this.yScale = d3.scaleLinear()
            .domain([d3.min(ys), d3.max(ys)])
            .range([padding, this.height - padding]);
    }

    // --- Top Level: Groups ---

    renderTopLevel() {
        this.navigationStack = [];
        this.currentParent = null;
        this.currentColor = null;
        this.currentItems = this.data.groups;

        this.renderVoronoiCells(this.data.groups, {
            getColor: d => d.color,
            getLabel: d => this.truncateLabel(d.name, 25),
            getSubLabel: d => `${d.entityCount.toLocaleString()} entities`,
            onHover: (event, d) => this.showClusterTooltip(event, d),
            onClick: (event, d) => this.drillInto(d, d.color),
            labelSize: d => Math.max(11, Math.min(16, 8 + Math.sqrt(d.entityCount) * 0.3)),
        });

        this.updateBreadcrumb();
        this.updateLegend();
        this.closeDetails();
    }

    // --- Drill into a cluster (recursive) ---

    drillInto(node, color) {
        this.hideTooltip();

        if (node.children && node.children.length > 0) {
            // Push current state onto stack
            this.navigationStack.push({
                items: this.currentItems,
                parent: this.currentParent,
                color: this.currentColor,
            });

            this.currentItems = node.children;
            this.currentParent = node;
            this.currentColor = color || this.currentColor;

            this.renderVoronoiCells(node.children, {
                getColor: () => this.currentColor,
                getLabel: d => this.truncateLabel(d.name, 22),
                getSubLabel: d => {
                    const hasChildren = d.children && d.children.length > 0;
                    const suffix = hasChildren ? ` · ${d.children.length} sub` : '';
                    return `${d.entityCount.toLocaleString()} entities${suffix}`;
                },
                onHover: (event, d) => this.showClusterTooltip(event, d),
                onClick: (event, d) => this.drillInto(d, this.currentColor),
                labelSize: d => Math.max(10, Math.min(14, 8 + Math.sqrt(d.entityCount) * 0.4)),
            });

            this.updateBreadcrumb();
            this.updateLegend();
            this.showNodeDetails(node);
        } else if (node.entities && node.entities.length > 0) {
            // Leaf cluster — show entities
            this.navigationStack.push({
                items: this.currentItems,
                parent: this.currentParent,
                color: this.currentColor,
            });
            this.currentParent = node;
            this.currentItems = null; // entity view

            this.renderEntityView(node, color);
            this.updateBreadcrumb();
            this.updateLegend();
            this.showNodeDetails(node);
        }
    }

    // --- Generic Voronoi cell renderer ---

    renderVoronoiCells(items, opts) {
        if (!items || items.length === 0) return;

        this.computeScales(items, d => ({ x: d.x, y: d.y }));

        const centroids = items.map(d => ([
            this.xScale(d.x),
            this.yScale(d.y),
        ]));

        // Animate out
        this.g.selectAll('*')
            .transition()
            .duration(this.transitionDuration / 2)
            .attr('opacity', 0)
            .remove();

        setTimeout(() => {
            this.g.selectAll('*').remove();

            const delaunay = d3.Delaunay.from(centroids);
            const voronoi = delaunay.voronoi([0, 0, this.width, this.height]);

            // Cells
            const cells = this.g.append('g').attr('class', 'voronoi-cells');

            cells.selectAll('path')
                .data(items)
                .join('path')
                .attr('d', (d, i) => voronoi.renderCell(i))
                .attr('fill', d => opts.getColor(d))
                .attr('opacity', 0)
                .attr('stroke', 'white')
                .attr('stroke-width', 2)
                .attr('class', 'voronoi-cell')
                .on('mouseover', (event, d) => opts.onHover(event, d))
                .on('mousemove', (event) => this.moveTooltip(event))
                .on('mouseout', () => this.hideTooltip())
                .on('click', (event, d) => opts.onClick(event, d))
                .transition()
                .duration(this.transitionDuration)
                .attr('opacity', 0.35);

            // Labels
            const labels = this.g.append('g').attr('class', 'cell-labels');

            labels.selectAll('.cell-label')
                .data(items)
                .join('text')
                .attr('class', 'cell-label')
                .attr('x', (d, i) => centroids[i][0])
                .attr('y', (d, i) => centroids[i][1] - 8)
                .style('font-size', d => `${opts.labelSize(d)}px`)
                .attr('opacity', 0)
                .text(d => opts.getLabel(d))
                .transition()
                .duration(this.transitionDuration)
                .attr('opacity', 1);

            labels.selectAll('.cell-count')
                .data(items)
                .join('text')
                .attr('class', 'cell-count')
                .attr('x', (d, i) => centroids[i][0])
                .attr('y', (d, i) => centroids[i][1] + 12)
                .attr('opacity', 0)
                .text(d => opts.getSubLabel(d))
                .transition()
                .duration(this.transitionDuration)
                .attr('opacity', 0.7);
        }, this.transitionDuration / 2);
    }

    // --- Entity view (leaf cluster) ---

    renderEntityView(cluster, color) {
        const entities = cluster.entities || [];
        if (entities.length === 0) return;

        this.computeScales(entities, d => ({ x: d.x, y: d.y }));

        // Animate out
        this.g.selectAll('*')
            .transition()
            .duration(this.transitionDuration / 2)
            .attr('opacity', 0)
            .remove();

        setTimeout(() => {
            this.g.selectAll('*').remove();

            const centroids = entities.map(e => ([
                this.xScale(e.x),
                this.yScale(e.y),
            ]));

            // Background Voronoi
            if (entities.length >= 3) {
                const delaunay = d3.Delaunay.from(centroids);
                const voronoi = delaunay.voronoi([0, 0, this.width, this.height]);

                this.g.append('g').attr('class', 'voronoi-bg')
                    .selectAll('path')
                    .data(entities)
                    .join('path')
                    .attr('d', (d, i) => voronoi.renderCell(i))
                    .attr('fill', d => this.typeColors[d.type] || this.typeColors.UNKNOWN)
                    .attr('opacity', 0)
                    .attr('stroke', 'white')
                    .attr('stroke-width', 1)
                    .attr('cursor', 'pointer')
                    .on('click', (event, d) => this.showEntityDetails(d))
                    .on('mouseover', (event, d) => this.showEntityTooltip(event, d))
                    .on('mousemove', (event) => this.moveTooltip(event))
                    .on('mouseout', () => this.hideTooltip())
                    .transition()
                    .duration(this.transitionDuration)
                    .attr('opacity', 0.08);
            }

            // Entity circles
            const circleGroup = this.g.append('g').attr('class', 'entity-circles');

            const radiusScale = d3.scaleSqrt()
                .domain([0, d3.max(entities, e => e.mentions || 1)])
                .range([5, 20]);

            circleGroup.selectAll('circle')
                .data(entities)
                .join('circle')
                .attr('class', 'entity-circle')
                .attr('cx', (d, i) => centroids[i][0])
                .attr('cy', (d, i) => centroids[i][1])
                .attr('r', 0)
                .attr('fill', d => this.typeColors[d.type] || this.typeColors.UNKNOWN)
                .attr('stroke', 'white')
                .attr('stroke-width', 1.5)
                .attr('opacity', 0.85)
                .on('mouseover', (event, d) => this.showEntityTooltip(event, d))
                .on('mousemove', (event) => this.moveTooltip(event))
                .on('mouseout', () => this.hideTooltip())
                .on('click', (event, d) => this.showEntityDetails(d))
                .transition()
                .duration(this.transitionDuration)
                .attr('r', d => radiusScale(d.mentions || 1));

            // Labels for top entities
            const labelGroup = this.g.append('g').attr('class', 'entity-labels');
            const topEntities = entities.slice(0, Math.min(15, entities.length));

            labelGroup.selectAll('text')
                .data(topEntities)
                .join('text')
                .attr('class', 'entity-label')
                .attr('x', (d, i) => centroids[i][0])
                .attr('y', (d, i) => centroids[i][1] + radiusScale(d.mentions || 1) + 12)
                .attr('opacity', 0)
                .text(d => this.truncateLabel(d.name, 18))
                .transition()
                .duration(this.transitionDuration)
                .attr('opacity', 0.8);
        }, this.transitionDuration / 2);
    }

    // --- Navigation ---

    navigateBack() {
        if (this.navigationStack.length === 0) return;

        const prev = this.navigationStack.pop();
        this.currentItems = prev.items;
        this.currentParent = prev.parent;
        this.currentColor = prev.color;

        if (this.navigationStack.length === 0) {
            // Back to top level
            this.renderTopLevel();
        } else {
            // Re-render the previous level's Voronoi
            const color = this.currentColor;
            this.renderVoronoiCells(this.currentItems, {
                getColor: d => d.color || color,
                getLabel: d => this.truncateLabel(d.name, 22),
                getSubLabel: d => {
                    const hasChildren = d.children && d.children.length > 0;
                    const suffix = hasChildren ? ` · ${d.children.length} sub` : '';
                    return `${d.entityCount.toLocaleString()} entities${suffix}`;
                },
                onHover: (event, d) => this.showClusterTooltip(event, d),
                onClick: (event, d) => this.drillInto(d, d.color || color),
                labelSize: d => Math.max(10, Math.min(14, 8 + Math.sqrt(d.entityCount) * 0.4)),
            });

            this.updateBreadcrumb();
            this.updateLegend();

            if (this.currentParent) {
                this.showNodeDetails(this.currentParent);
            } else {
                this.closeDetails();
            }
        }
    }

    navigateToLevel(targetDepth) {
        // Pop back to the target depth
        while (this.navigationStack.length > targetDepth) {
            this.navigationStack.pop();
        }

        if (targetDepth === 0) {
            this.renderTopLevel();
        } else {
            const prev = this.navigationStack.pop();
            this.currentItems = prev.items;
            this.currentParent = prev.parent;
            this.currentColor = prev.color;

            // Find the node at this depth and re-drill
            // Actually, just re-render currentItems
            const color = this.currentColor;
            this.renderVoronoiCells(this.currentItems, {
                getColor: d => d.color || color,
                getLabel: d => this.truncateLabel(d.name, 22),
                getSubLabel: d => `${d.entityCount.toLocaleString()} entities`,
                onHover: (event, d) => this.showClusterTooltip(event, d),
                onClick: (event, d) => this.drillInto(d, d.color || color),
                labelSize: d => Math.max(10, Math.min(14, 8 + Math.sqrt(d.entityCount) * 0.4)),
            });

            this.updateBreadcrumb();
            this.updateLegend();
            this.closeDetails();
        }
    }

    updateBreadcrumb() {
        const container = d3.select('#breadcrumb');
        container.selectAll('*').remove();

        // Build path: [All Groups] > [stack parents] > [current parent]
        const allItem = container.append('span')
            .attr('class', `breadcrumb-item ${this.navigationStack.length === 0 && !this.currentParent ? 'current' : ''}`)
            .text('All Groups');
        if (this.navigationStack.length > 0 || this.currentParent) {
            allItem.on('click', () => this.renderTopLevel());
        }

        // Intermediate levels from stack
        for (let i = 0; i < this.navigationStack.length; i++) {
            const stackEntry = this.navigationStack[i];
            if (stackEntry.parent) {
                container.append('span').attr('class', 'breadcrumb-sep').text('\u203A');
                const depth = i;
                const item = container.append('span')
                    .attr('class', 'breadcrumb-item')
                    .text(this.truncateLabel(stackEntry.parent.name, 20))
                    .on('click', () => this.navigateToLevel(depth));
            }
        }

        // Current parent
        if (this.currentParent) {
            container.append('span').attr('class', 'breadcrumb-sep').text('\u203A');
            container.append('span')
                .attr('class', 'breadcrumb-item current')
                .text(this.truncateLabel(this.currentParent.name, 20));
        }
    }

    // --- Tooltips ---

    showClusterTooltip(event, cluster) {
        const hasChildren = cluster.children && cluster.children.length > 0;
        const hasEntities = cluster.entities && cluster.entities.length > 0;

        let childInfo = '';
        if (hasChildren) {
            const childList = cluster.children.slice(0, 5)
                .map(c => this.escapeHtml(c.name)).join(', ');
            const more = cluster.children.length > 5 ? ` +${cluster.children.length - 5} more` : '';
            childInfo = `<div class="tooltip-children">Sub-clusters: ${childList}${more}</div>`;
        } else if (hasEntities) {
            const entList = cluster.entities.slice(0, 5)
                .map(e => this.escapeHtml(e.name)).join(', ');
            childInfo = `<div class="tooltip-children">Top entities: ${entList}</div>`;
        }

        this.tooltip.html(`
            <div class="tooltip-title">${this.escapeHtml(cluster.name)}</div>
            <div class="tooltip-count">${cluster.entityCount.toLocaleString()} entities${hasChildren ? ` · ${cluster.children.length} sub-clusters` : ''}</div>
            ${cluster.summary ? `<div class="tooltip-summary">${this.escapeHtml(cluster.summary.slice(0, 150))}...</div>` : ''}
            ${childInfo}
        `);
        this.tooltip.classed('visible', true);
        this.moveTooltip(event);
    }

    showEntityTooltip(event, entity) {
        this.tooltip.html(`
            <div class="tooltip-title">${this.escapeHtml(entity.name)}</div>
            <div class="tooltip-count">${this.escapeHtml(entity.type)} · ${entity.mentions || 0} mentions</div>
            ${entity.description ? `<div class="tooltip-summary">${this.escapeHtml(entity.description)}</div>` : ''}
        `);
        this.tooltip.classed('visible', true);
        this.moveTooltip(event);
    }

    moveTooltip(event) {
        const x = event.pageX + 15;
        const y = event.pageY - 10;
        const tw = 300;
        const adjustedX = x + tw > window.innerWidth ? event.pageX - tw - 15 : x;
        this.tooltip.style('left', adjustedX + 'px').style('top', y + 'px');
    }

    hideTooltip() {
        this.tooltip.classed('visible', false);
    }

    // --- Details panel ---

    showNodeDetails(node) {
        const panel = d3.select('#details-panel');
        const content = d3.select('#details-content');
        const hasChildren = node.children && node.children.length > 0;
        const hasEntities = node.entities && node.entities.length > 0;

        d3.select('#details-title').text(hasChildren ? 'Cluster Details' : 'Leaf Cluster');

        let childrenHtml = '';
        if (hasChildren) {
            childrenHtml = `
                <h4 style="margin-top: 12px; font-size: 13px; color: #667eea;">Sub-clusters (${node.children.length})</h4>
                <div class="entity-list">
                    ${node.children.map(c => `
                        <div class="entity-list-item" data-node-id="${this.escapeHtml(c.id)}">
                            ${this.escapeHtml(c.name)} <span style="color: #999; font-size: 11px">(${c.entityCount})</span>
                            ${c.children ? '<span style="color: #667eea; font-size: 10px"> ▶</span>' : ''}
                        </div>
                    `).join('')}
                </div>
            `;
        } else if (hasEntities) {
            childrenHtml = `
                <h4 style="margin-top: 12px; font-size: 13px; color: #667eea;">Top Entities (${node.entities.length})</h4>
                <div class="entity-list">
                    ${node.entities.map(e => `
                        <div class="entity-list-item" data-entity-name="${this.escapeHtml(e.name)}">
                            ${this.escapeHtml(e.name)}
                            <span class="entity-type-badge" style="background: ${this.typeColors[e.type] || '#999'}">${this.escapeHtml(e.type)}</span>
                        </div>
                    `).join('')}
                </div>
            `;
        }

        const color = node.color || this.currentColor || '#667eea';
        content.html(`
            <div class="detail-name">${this.escapeHtml(node.name)}</div>
            <div class="detail-type" style="background: ${color}">
                ${hasChildren ? 'Cluster' : 'Leaf Cluster'}
                · Depth ${this.navigationStack.length}
            </div>
            ${node.summary ? `<div class="detail-description">${this.escapeHtml(node.summary)}</div>` : ''}
            <div class="detail-meta">
                <strong>Entities:</strong> ${node.entityCount.toLocaleString()}<br>
                ${hasChildren ? `<strong>Sub-clusters:</strong> ${node.children.length}` : ''}
            </div>
            ${childrenHtml}
        `);

        // Click handlers
        if (hasChildren) {
            const self = this;
            content.selectAll('.entity-list-item').on('click', function() {
                const nid = this.dataset.nodeId;
                const child = node.children.find(c => c.id === nid);
                if (child) self.drillInto(child, self.currentColor);
            });
        } else if (hasEntities) {
            const self = this;
            content.selectAll('.entity-list-item').on('click', function() {
                const name = this.dataset.entityName;
                const ent = node.entities.find(e => e.name === name);
                if (ent) self.showEntityDetails(ent);
            });
        }

        panel.classed('open', true);
    }

    showEntityDetails(entity) {
        const panel = d3.select('#details-panel');
        const content = d3.select('#details-content');
        d3.select('#details-title').text('Entity Details');

        content.html(`
            <div class="detail-name">${this.escapeHtml(entity.name)}</div>
            <div class="detail-type" style="background: ${this.typeColors[entity.type] || '#999'}">${this.escapeHtml(entity.type)}</div>
            ${entity.description ? `<div class="detail-description">${this.escapeHtml(entity.description)}</div>` : ''}
            <div class="detail-meta">
                <strong>Mentions:</strong> ${entity.mentions || 0}<br>
                ${entity.connections ? `<strong>Centrality:</strong> ${entity.connections.toFixed(4)}<br>` : ''}
            </div>
        `);

        panel.classed('open', true);

        // Highlight the entity circle
        this.g.selectAll('.entity-circle')
            .attr('stroke', d => d.name === entity.name ? '#FFD700' : 'white')
            .attr('stroke-width', d => d.name === entity.name ? 3 : 1.5);
    }

    closeDetails() {
        d3.select('#details-panel').classed('open', false);
        this.g.selectAll('.entity-circle')
            .attr('stroke', 'white')
            .attr('stroke-width', 1.5);
    }

    // --- Search ---

    setupSearch() {
        const input = document.getElementById('search-input');
        let debounceTimer;

        input.addEventListener('input', () => {
            clearTimeout(debounceTimer);
            debounceTimer = setTimeout(() => this.doSearch(input.value), 300);
        });

        input.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                input.value = '';
                d3.select('#search-results').html('');
            }
        });
    }

    async doSearch(query) {
        const resultsDiv = d3.select('#search-results');
        if (!query || query.length < 2) {
            resultsDiv.html('');
            return;
        }

        await this.loadEntityIndex();

        const q = query.toLowerCase();
        const matches = [];

        for (const [name, info] of Object.entries(this.entityIndex)) {
            if (name.toLowerCase().includes(q)) {
                matches.push({ name, ...info });
                if (matches.length >= 20) break;
            }
        }

        if (matches.length === 0) {
            resultsDiv.html('<div class="search-result-item" style="color: #999">No results found</div>');
            return;
        }

        resultsDiv.html(matches.map(m => `
            <div class="search-result-item" data-entity="${this.escapeHtml(m.name)}" data-group="${this.escapeHtml(m.group)}" data-cluster="${this.escapeHtml(m.cluster)}">
                <div class="search-result-name">${this.escapeHtml(m.name)}</div>
                <div class="search-result-path">${this.escapeHtml(m.path || `${m.groupName} > ${m.clusterName}`)}</div>
            </div>
        `).join(''));

        const self = this;
        resultsDiv.selectAll('.search-result-item').on('click', function() {
            const entityName = this.dataset.entity;
            const groupId = this.dataset.group;
            const clusterId = this.dataset.cluster;
            self.navigateToEntity(groupId, clusterId, entityName);
        });
    }

    navigateToEntity(groupId, clusterId, entityName) {
        // Find the group
        const group = this.data.groups.find(g => g.id === groupId);
        if (!group) return;

        // Find the leaf cluster by walking the tree
        const findCluster = (node) => {
            if (node.id === clusterId) return node;
            if (node.children) {
                for (const child of node.children) {
                    const found = findCluster(child);
                    if (found) return found;
                }
            }
            return null;
        };

        const cluster = findCluster(group);
        if (!cluster) return;

        // Build the path from group to cluster
        const buildPath = (node, target, path = []) => {
            path.push(node);
            if (node.id === target.id) return path;
            if (node.children) {
                for (const child of node.children) {
                    const result = buildPath(child, target, [...path]);
                    if (result) return result;
                }
            }
            return null;
        };

        const path = buildPath(group, cluster) || [group];

        // Navigate step by step
        this.renderTopLevel();
        let color = group.color;

        // Drill through the path
        const drillPath = (steps, idx) => {
            if (idx >= steps.length) {
                // Highlight entity
                setTimeout(() => {
                    const entity = (cluster.entities || []).find(e => e.name === entityName);
                    if (entity) this.showEntityDetails(entity);
                }, this.transitionDuration + 100);
                return;
            }
            setTimeout(() => {
                this.drillInto(steps[idx], color);
                drillPath(steps, idx + 1);
            }, this.transitionDuration + 100);
        };

        drillPath(path, 0);
    }

    // --- Keyboard ---

    setupKeyboard() {
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                if (d3.select('#details-panel').classed('open')) {
                    this.closeDetails();
                } else {
                    this.navigateBack();
                }
            }
        });
    }

    // --- Legend & Stats ---

    updateLegend() {
        const legend = d3.select('#legend');
        legend.html('');

        const isEntityView = this.currentItems === null && this.currentParent;

        if (!isEntityView) {
            // Show cluster items
            const items = this.currentItems || this.data.groups;
            items.forEach(item => {
                const color = item.color || this.currentColor || '#999';
                const hasChildren = item.children && item.children.length > 0;
                legend.append('div')
                    .attr('class', 'legend-item')
                    .html(`
                        <div class="legend-color" style="background: ${color}"></div>
                        <div class="legend-label">${this.truncateLabel(item.name, 22)}</div>
                        <div class="legend-count">${item.entityCount}${hasChildren ? ' \u25B6' : ''}</div>
                    `)
                    .on('click', () => this.drillInto(item, item.color || this.currentColor));
            });
        } else {
            // Entity type legend
            for (const [type, color] of Object.entries(this.typeColors)) {
                if (type === 'UNKNOWN') continue;
                legend.append('div')
                    .attr('class', 'legend-item')
                    .html(`
                        <div class="legend-color" style="background: ${color}; border-radius: 50%"></div>
                        <div class="legend-label">${type}</div>
                    `);
            }
        }
    }

    updateStats() {
        const stats = d3.select('#stats');
        const m = this.data.metadata;
        stats.html(`
            <div class="stat-item"><span>Total Entities</span><span class="stat-value">${m.totalEntities.toLocaleString()}</span></div>
            <div class="stat-item"><span>Top-Level Clusters</span><span class="stat-value">${m.totalGroups}</span></div>
            <div class="stat-item"><span>Max Depth</span><span class="stat-value">${m.maxDepth || 'N/A'}</span></div>
            <div class="stat-item"><span>Hierarchy</span><span class="stat-value">${m.hierarchyType || 'natural'}</span></div>
        `);
    }

    // --- Helpers ---

    truncateLabel(text, maxLen) {
        if (!text) return '';
        return text.length > maxLen ? text.slice(0, maxLen - 1) + '\u2026' : text;
    }

    escapeHtml(text) {
        if (!text) return '';
        return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    }

    hideLoading() {
        const overlay = document.getElementById('loading-overlay');
        if (overlay) overlay.classList.add('hidden');
    }
}

// Initialize
let voronoiMap;
document.addEventListener('DOMContentLoaded', () => {
    voronoiMap = new VoronoiMapVisualization('#voronoi-svg-container');
});
