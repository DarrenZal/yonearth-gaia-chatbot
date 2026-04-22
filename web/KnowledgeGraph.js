/**
 * Knowledge Graph Visualization
 * Interactive D3.js force-directed graph for YonEarth entities
 */

class KnowledgeGraphVisualization {
    constructor(containerId) {
        this.container = d3.select(containerId);
        this.data = null;
        this.simulation = null;
        this.svg = null;
        this.g = null;

        // Visual elements
        this.links = null;
        this.nodes = null;
        this.labels = null;

        // Detect embedded/simple mode (iframe in /guide/) — tightens defaults for legibility
        this.isSimpleMode =
            new URLSearchParams(window.location.search).get('simple') === '1' ||
            document.documentElement.classList.contains('simple-mode');

        // State
        this.selectedNode = null;
        this.filters = {
            domains: new Set(),
            entityTypes: new Set(),
            themes: new Set(),    // YOE secondary themes (simple-mode only). Empty = all.
            pillars: new Set(),   // YOE primary pillars (simple-mode only). Empty = all.
            // Design 3: EPISODE nodes are hidden by default and auto-revealed when a
            // pillar/theme chip is active. forceShowEpisodes=true overrides (shows all
            // 170 regardless of filter state).
            forceShowEpisodes: false,
            minImportance: this.isSimpleMode ? 0.9 : 0.7,
            searchQuery: "",
            // Bumped from 50 → 200 because a pillar/theme click now adds up to ~120
            // episode nodes alongside the ~30 concepts. Edge-cap + force-layout keep
            // the render legible.
            maxNodes: this.isSimpleMode ? 200 : 1000
        };

        // YOE taxonomy (loaded async in loadTaxonomy()). Null until available.
        this.taxonomy = null;
        this.themeEpisodeIndex = null;
        this.pillarEpisodeIndex = null;

        // Layout parameters.
        // Simple-mode: stronger repulsion, minimal collision padding (let charge handle
        // spacing — uniform collision radius was causing the geometric grid pattern).
        this.params = {
            gravity: this.isSimpleMode ? 0.05 : 0.1,
            charge: this.isSimpleMode ? -900 : -300,
            linkDistance: this.isSimpleMode ? 110 : 100,
            collisionRadius: this.isSimpleMode ? 4 : 15,
            maxEdgesPerNode: this.isSimpleMode ? 5 : Infinity
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

        // Simple/embedded mode gets a compact filter strip since the
        // full left-sidebar controls are hidden by CSS.
        if (this.isSimpleMode) {
            this.setupSimpleFilterStrip();
        }

        // Create visualization
        this.createVisualization();

        // Hide loading overlay
        this.showLoading(false);

        console.log("Knowledge Graph Visualization initialized");
    }

    async loadData() {
        try {
            // Try to load from API first
            const response = await fetch('/api/knowledge-graph/data');
            if (response.ok) {
                this.data = await response.json();
                console.log("Loaded data from API:", this.data);
            } else {
                throw new Error("API not available");
            }
        } catch (error) {
            console.log("Loading from local file...");
            try {
                const response = await fetch('/data/knowledge_graph/visualization_data.json');
                this.data = await response.json();
                console.log("Loaded data from file:", this.data);
            } catch (fileError) {
                console.error("Error loading data:", fileError);
                this.showError("Failed to load knowledge graph data");
                return;
            }
        }

        // Initialize filters with all domains and types enabled
        this.data.domains.forEach(d => this.filters.domains.add(d.name));

        // Simple-mode default is knowledge-centric: only CONCEPT / PRACTICE / EVENT /
        // TECHNOLOGY / SPECIES / ECOSYSTEM visible on load. PLACE / PERSON /
        // ORGANIZATION / PRODUCT / WORK are opt-in via the type chip strip.
        // Aaron's Apr 21 direction: "less focus on podcast episodes and book
        // content now, more emphasis on places, locations, and people" —
        // wait no, the opposite: knowledge/topic/theme-centric, places/people
        // less prominent. This matches his Apr 12 email + Apr 21 voice memo.
        // EPISODE is always in entityTypes when in simple-mode — the episode-specific
        // visibility rule (forceShowEpisodes OR narrow-filter active) happens in
        // getFilteredData() below, not via the type-filter Set.
        const KNOWLEDGE_CENTRIC_TYPES = new Set([
            'CONCEPT', 'PRACTICE', 'EVENT',
            'TECHNOLOGY', 'SPECIES', 'ECOSYSTEM',
            'EPISODE'
        ]);
        this.data.entity_types.forEach(t => {
            if (!this.isSimpleMode || KNOWLEDGE_CENTRIC_TYPES.has(t)) {
                this.filters.entityTypes.add(t);
            }
        });
        // EPISODE is injected after loadData() completes, so explicitly seed it.
        if (this.isSimpleMode) this.filters.entityTypes.add('EPISODE');

        // Load YOE taxonomy (pillars + themes ↔ episode IDs) for /guide/
        // secondary chip filters. Client-side join against entity.episodes.
        if (this.isSimpleMode) {
            await this.loadTaxonomy();
        }
    }

    async loadTaxonomy() {
        try {
            // Fetched from /guide/yoe_taxonomy.json — served by the generic
            // /guide/ nginx alias. Avoids /guide/data/ and /guide/assets/,
            // both of which are aliased to the production /var/www/yonearth/
            // tree for shared content.
            const response = await fetch('./yoe_taxonomy.json?v=1');
            if (!response.ok) throw new Error(`taxonomy fetch: ${response.status}`);
            this.taxonomy = await response.json();
            this.buildTaxonomyIndex();
            // Design 3: promote episodes to first-class nodes.
            this.injectEpisodeNodes();
            console.log(`Loaded YOE taxonomy: ${Object.keys(this.taxonomy.pillars).length} pillars, ${this.taxonomy.themes.length} themes, ${Object.keys(this.taxonomy.episodes).length} episodes`);
        } catch (err) {
            console.warn("YOE taxonomy unavailable; theme/pillar filters disabled:", err);
            this.taxonomy = null;
        }
    }

    /**
     * Inject one EPISODE node per taxonomy episode + MENTIONED_IN edges from
     * every existing entity that lists the episode number in its `episodes: [...]`.
     * Edges are low-strength so the greedy per-node edge cap keeps entity-entity
     * edges preferred when both compete for the same hub.
     */
    injectEpisodeNodes() {
        if (!this.taxonomy || !this.data) return;

        // Map pillar names (uppercase in taxonomy) → KG domain color (title-case in KG).
        const domainColorMap = {};
        (this.data.domains || []).forEach(d => { domainColorMap[d.name.toUpperCase()] = d.color; });

        // Theme display lookup: raw name ('FARMING & FOOD') → display ('Farming & Food').
        const themeDisplayMap = {};
        (this.taxonomy.themes || []).forEach(t => { themeDisplayMap[t.name] = t.display; });

        const existingIds = new Set(this.data.nodes.map(n => n.id));
        const episodeNodes = [];
        Object.values(this.taxonomy.episodes).forEach(ep => {
            const id = `ep_${ep.episode_number}`;
            if (existingIds.has(id)) return;
            const domainsTitleCase = (ep.pillars || []).map(p =>
                p.charAt(0) + p.slice(1).toLowerCase()
            );
            const domainColors = (ep.pillars || [])
                .map(p => domainColorMap[p.toUpperCase()])
                .filter(Boolean);
            const guest = ep.guest || '';
            const display = guest ? `Ep ${ep.episode_number}: ${guest}` : `Ep ${ep.episode_number}`;
            episodeNodes.push({
                id,
                name: `Ep ${ep.episode_number}`,
                display_name: display,
                type: 'EPISODE',
                shape: 'circle',
                description: guest
                    ? `${display}${ep.org ? ' — ' + ep.org : ''}${ep.location ? ' (' + ep.location + ')' : ''}`
                    : display,
                aliases: [],
                domains: domainsTitleCase.length ? domainsTitleCase : ['Community'],
                domain_colors: domainColors.length ? domainColors : ['#a8917a'],
                importance: 1.0,                       // always passes min-importance
                mention_count: 0,
                episode_count: 1,
                episodes: [ep.episode_number],         // self-ref — pillar/theme set-intersect still works
                episode_number: ep.episode_number,
                guest: ep.guest || '',
                org: ep.org || '',
                location: ep.location || '',
                themes: (ep.themes || []).map(t => themeDisplayMap[t] || t),
                pillars: (ep.pillars || []).map(p => p.charAt(0) + p.slice(1).toLowerCase()),
                url: `https://yonearth.org/podcast/episode-${ep.episode_number}/`,
                community: 'episodes'
            });
        });

        this.data.nodes.push(...episodeNodes);
        if (!this.data.entity_types.includes('EPISODE')) {
            this.data.entity_types.push('EPISODE');
        }

        // Synthesize MENTIONED_IN edges from each existing entity to its episodes.
        const synthEdges = [];
        const epNumsInjected = new Set(episodeNodes.map(n => n.episode_number));
        this.data.nodes.forEach(n => {
            if (n.type === 'EPISODE') return;
            const eps = n.episodes;
            if (!Array.isArray(eps)) return;
            eps.forEach(num => {
                if (!epNumsInjected.has(num)) return;
                synthEdges.push({
                    source: n.id,
                    target: `ep_${num}`,
                    type: 'MENTIONED_IN',
                    relationship_type: 'MENTIONED_IN',
                    strength: 0.1
                });
            });
        });
        this.data.links.push(...synthEdges);

        console.log(`Injected ${episodeNodes.length} EPISODE nodes + ${synthEdges.length} MENTIONED_IN edges`);
    }

    buildTaxonomyIndex() {
        // Index: theme name (upper) → Set<episode_id>; pillar name → Set<episode_id>
        this.themeEpisodeIndex = new Map();
        this.pillarEpisodeIndex = new Map();

        this.taxonomy.themes.forEach(t => {
            this.themeEpisodeIndex.set(t.name, new Set(t.episode_ids));
        });
        Object.entries(this.taxonomy.pillars).forEach(([name, p]) => {
            this.pillarEpisodeIndex.set(name, new Set(p.episode_ids));
        });
    }

    /**
     * Does this entity appear in any of the given episode IDs?
     * Entity.episodes is typically an int or array of ints (from the KG dataset).
     */
    entityInEpisodeSet(entity, episodeSet) {
        if (!entity || !episodeSet || episodeSet.size === 0) return false;
        const eps = entity.episodes;
        if (Array.isArray(eps)) {
            for (const id of eps) if (episodeSet.has(id)) return true;
            return false;
        }
        if (typeof eps === 'number') return episodeSet.has(eps);
        return false;
    }

    /** Union all episode-id sets for the selected keys (pillar or theme names). */
    unionEpisodeSet(selectedKeys, index) {
        const union = new Set();
        selectedKeys.forEach(key => {
            const set = index.get(key);
            if (set) set.forEach(id => union.add(id));
        });
        return union;
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

        // Seed node positions near viewport center so the fresh force sim doesn't
        // fling them off-screen on filter changes. Random offset inside a small disc.
        // Without this, d3 spawns nodes at (undefined, undefined) which it treats as
        // 0,0 before the first tick, and the force layout can push them to ±4000px
        // with 100+ nodes + -900 charge.
        const cx = this.width / 2;
        const cy = this.height / 2;
        const seedRadius = Math.min(this.width, this.height) * 0.35;
        filteredData.nodes.forEach(n => {
            if (typeof n.x !== 'number' || typeof n.y !== 'number' ||
                !isFinite(n.x) || !isFinite(n.y) ||
                Math.hypot(n.x - cx, n.y - cy) > seedRadius * 3) {
                const theta = Math.random() * 2 * Math.PI;
                const r = seedRadius * Math.sqrt(Math.random());
                n.x = cx + r * Math.cos(theta);
                n.y = cy + r * Math.sin(theta);
            }
        });

        // Scale charge down for big node sets. 30 nodes → full charge, 200 → ~15%.
        // Use forceX/forceY anchoring (stronger than forceCenter for big sets) —
        // this lets charge spread things *within* the viewport without piling
        // nodes against a boundary-clamp wall.
        const nodeCount = filteredData.nodes.length || 1;
        const chargeScale = Math.min(1, 30 / nodeCount);
        const effectiveCharge = this.params.charge * chargeScale;
        // Strength scales from 0.04 (30 nodes) to ~0.25 (200 nodes) for the x/y pull.
        const positionStrength = Math.min(0.25, 0.03 + nodeCount / 800);
        // Link distance shrinks as crowd grows so episodes huddle near their concepts
        // rather than spreading across the canvas.
        const effectiveLinkDistance = nodeCount > 50
            ? Math.max(30, this.params.linkDistance * (50 / nodeCount))
            : this.params.linkDistance;

        // Create force simulation
        this.simulation = d3.forceSimulation(filteredData.nodes)
            .force('link', d3.forceLink(filteredData.links)
                .id(d => d.id)
                .distance(effectiveLinkDistance)
                .strength(l => l.type === 'MENTIONED_IN' ? 0.15 : 0.6))
            .force('charge', d3.forceManyBody()
                .strength(effectiveCharge)
                .distanceMax(300))   // charge falls off past 300px so distant nodes don't repel
            .force('x', d3.forceX(cx).strength(positionStrength))
            .force('y', d3.forceY(cy).strength(positionStrength))
            .force('collision', d3.forceCollide()
                .radius(d => this.getNodeRadius(d) + this.params.collisionRadius))
            .velocityDecay(0.45);

        // Create links. MENTIONED_IN edges (concept→episode) get a thinner dashed
        // stroke so they read as "loose association" vs the solid entity-entity edges.
        this.links = this.g.append('g')
            .attr('class', 'links')
            .selectAll('line')
            .data(filteredData.links)
            .join('line')
            .attr('class', d => 'link' + (d.type === 'MENTIONED_IN' ? ' link-mentioned' : ''))
            .attr('stroke-width', d => d.type === 'MENTIONED_IN' ? 0.8 : Math.max(1, d.strength * 3))
            .attr('stroke-dasharray', d => d.type === 'MENTIONED_IN' ? '2,3' : null)
            .attr('stroke-opacity', d => d.type === 'MENTIONED_IN' ? 0.35 : null);

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

        // Label only the most-discussed nodes so the graph stays legible.
        // Simple-mode: top 15 by episode_count (the topics Aaron covers most).
        const labelCandidates = filteredData.nodes.filter(d => d.importance > 0.3);
        const labeledNodes = this.isSimpleMode
            ? [...labelCandidates]
                .sort((a, b) => (b.episode_count || 0) - (a.episode_count || 0))
                .slice(0, 15)
            : labelCandidates;
        this.labels = this.g.append('g')
            .attr('class', 'labels')
            .selectAll('text')
            .data(labeledNodes)
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

        // Soft boundary clamp: only hard-stop beyond a generous outer bound
        // (1.15 × viewport). Inside the viewport, forceX/forceY handle the pull.
        // This avoids the "nodes piled along the wall" look a hard clamp creates.
        const padX = 30;
        const padY = 30;
        const outerW = this.width * 1.15;
        const outerH = this.height * 1.15;
        this.simulation.on('tick', () => {
            this.nodes.each(d => {
                if (d.x < -outerW * 0.15 + padX) d.x = -outerW * 0.15 + padX;
                else if (d.x > outerW - padX) d.x = outerW - padX;
                if (d.y < -outerH * 0.15 + padY) d.y = -outerH * 0.15 + padY;
                else if (d.y > outerH - padY) d.y = outerH - padY;
            });

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
        // EPISODE nodes get a uniform smaller radius so they read as a distinct
        // "layer" of the graph (50+ of them, sitting alongside concepts).
        if (d.type === 'EPISODE') return 5;
        // Scale by episode_count so size reflects how often Aaron discusses this topic.
        // sqrt gives a perceptually-linear area scale (2× episodes → ~1.4× radius).
        // Range: ep=1 → 7px, ep=10 → 13.5px, ep=30 → 20px, ep=50+ → 25px.
        const count = d.episode_count || d.mention_count || 1;
        return 4 + Math.sqrt(count) * 3;
    }

    getNodeColor(d) {
        // Return primary domain color
        return d.domain_colors[0] || '#999';
    }

    getFilteredData() {
        // Union of episode_ids from currently-active pillar chips (OR across pillars).
        const activePillarEpisodes = (this.filters.pillars.size && this.pillarEpisodeIndex)
            ? this.unionEpisodeSet(this.filters.pillars, this.pillarEpisodeIndex)
            : null;
        // Union of episode_ids from currently-active theme chips (OR across themes).
        // (The plan says themes are exclusive-select so this set will have ≤1 entry,
        // but the union form makes the code robust to future multi-select.)
        const activeThemeEpisodes = (this.filters.themes.size && this.themeEpisodeIndex)
            ? this.unionEpisodeSet(this.filters.themes, this.themeEpisodeIndex)
            : null;

        const narrowActive = !!activePillarEpisodes || !!activeThemeEpisodes;

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

            // Episode-specific rule (Design 3): episodes hidden by default, shown when
            //   (a) a pillar or theme chip is active, OR
            //   (b) the user has explicitly force-toggled Episodes on.
            // Without this rule the default view would pull in all 170 episode nodes.
            if (node.type === 'EPISODE') {
                if (!this.filters.forceShowEpisodes && !narrowActive) return false;
            }

            // Check domain filter (node must have at least one matching domain)
            const hasMatchingDomain = node.domains.some(d => this.filters.domains.has(d));
            if (!hasMatchingDomain) {
                return false;
            }

            // Check pillar filter (YOE primary pillar — simple-mode only).
            // Node passes iff at least one of its episodes is in an active pillar.
            if (activePillarEpisodes && !this.entityInEpisodeSet(node, activePillarEpisodes)) {
                return false;
            }

            // Check theme filter (YOE secondary theme — simple-mode only).
            // ANDed against pillars: both must pass.
            if (activeThemeEpisodes && !this.entityInEpisodeSet(node, activeThemeEpisodes)) {
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
        let filteredLinks = this.data.links.filter(link => {
            return nodeIds.has(link.source.id || link.source) &&
                   nodeIds.has(link.target.id || link.target);
        });

        // Simple-mode: cap edges per node so hub nodes don't create a visual hairball.
        // Greedy: sort by strength descending, add edge only if both endpoints still have
        // room. This preserves the strongest connections for every node.
        if (this.isSimpleMode && this.params.maxEdgesPerNode < Infinity) {
            const cap = this.params.maxEdgesPerNode;
            const edgeCount = new Map();
            filteredLinks.sort((a, b) => (b.strength || 0) - (a.strength || 0));
            const sparse = [];
            for (const link of filteredLinks) {
                const s = link.source.id || link.source;
                const t = link.target.id || link.target;
                if ((edgeCount.get(s) || 0) < cap && (edgeCount.get(t) || 0) < cap) {
                    sparse.push(link);
                    edgeCount.set(s, (edgeCount.get(s) || 0) + 1);
                    edgeCount.set(t, (edgeCount.get(t) || 0) + 1);
                }
            }
            filteredLinks = sparse;
        }

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

    /**
     * Highlight multiple entities by name/alias
     * Used by chat-kg-bridge.js to highlight entities mentioned in chat responses
     * @param {string[]} entityNames - Array of entity names to highlight
     * @returns {string[]} - Array of matched entity names (for feedback to user)
     */
    highlightEntities(entityNames) {
        if (!entityNames || entityNames.length === 0) {
            this.clearHighlight();
            return [];
        }

        // Normalize search terms
        const searchTerms = entityNames.map(name => name.toLowerCase().trim());

        // Track which nodes we match
        const matchedNodes = new Set();
        const matchedNames = [];

        // Find matching nodes by name or alias
        this.data.nodes.forEach(node => {
            const nodeName = node.name.toLowerCase();
            const nodeId = node.id.toLowerCase();
            const aliases = (node.aliases || []).map(a => a.toLowerCase());

            for (const term of searchTerms) {
                // Check exact match on name
                if (nodeName === term || nodeId === term) {
                    matchedNodes.add(node.id);
                    matchedNames.push(node.name);
                    break;
                }

                // Check alias match
                if (aliases.some(alias => alias === term)) {
                    matchedNodes.add(node.id);
                    matchedNames.push(node.name);
                    break;
                }

                // Check partial match (for multi-word entities)
                if (nodeName.includes(term) || term.includes(nodeName)) {
                    matchedNodes.add(node.id);
                    matchedNames.push(node.name);
                    break;
                }
            }
        });

        if (matchedNodes.size === 0) {
            console.log('highlightEntities: No matches found for', entityNames);
            return [];
        }

        console.log('highlightEntities: Matched', matchedNames);

        // Get all connected node IDs for the matched nodes
        const connectedIds = new Set(matchedNodes);
        this.data.links.forEach(link => {
            const sourceId = link.source.id || link.source;
            const targetId = link.target.id || link.target;

            if (matchedNodes.has(sourceId)) connectedIds.add(targetId);
            if (matchedNodes.has(targetId)) connectedIds.add(sourceId);
        });

        // Apply multi-highlight visual style
        this.nodes
            .classed('highlighted', node => matchedNodes.has(node.id))
            .classed('connected', node => connectedIds.has(node.id) && !matchedNodes.has(node.id))
            .classed('dimmed', node => !connectedIds.has(node.id));

        // Add glow effect to highlighted nodes
        this.nodes.selectAll('circle')
            .style('filter', node => matchedNodes.has(node.id) ? 'drop-shadow(0 0 8px #667eea) drop-shadow(0 0 12px #764ba2)' : null)
            .style('stroke', node => matchedNodes.has(node.id) ? '#667eea' : null)
            .style('stroke-width', node => matchedNodes.has(node.id) ? '3px' : null);

        // Highlight links between matched nodes
        this.links
            .classed('highlighted', link => {
                const sourceId = link.source.id || link.source;
                const targetId = link.target.id || link.target;
                return matchedNodes.has(sourceId) && matchedNodes.has(targetId);
            })
            .classed('connected', link => {
                const sourceId = link.source.id || link.source;
                const targetId = link.target.id || link.target;
                return (matchedNodes.has(sourceId) || matchedNodes.has(targetId)) &&
                       !(matchedNodes.has(sourceId) && matchedNodes.has(targetId));
            })
            .classed('dimmed', link => {
                const sourceId = link.source.id || link.source;
                const targetId = link.target.id || link.target;
                return !connectedIds.has(sourceId) || !connectedIds.has(targetId);
            });

        // Pan/zoom to show highlighted nodes (optional, only if more than one match)
        if (matchedNodes.size > 0 && matchedNodes.size <= 5) {
            this.focusOnNodes(Array.from(matchedNodes));
        }

        return matchedNames;
    }

    /**
     * Focus/zoom the view to show a set of nodes
     * @param {string[]} nodeIds - Array of node IDs to focus on
     */
    focusOnNodes(nodeIds) {
        // Find the matching node data
        const matchingNodes = this.data.nodes.filter(n => nodeIds.includes(n.id));

        if (matchingNodes.length === 0) return;

        // Calculate bounding box of all matched nodes
        let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
        matchingNodes.forEach(node => {
            if (node.x !== undefined && node.y !== undefined) {
                minX = Math.min(minX, node.x);
                maxX = Math.max(maxX, node.x);
                minY = Math.min(minY, node.y);
                maxY = Math.max(maxY, node.y);
            }
        });

        // If we couldn't find positions, skip zooming
        if (!isFinite(minX)) return;

        // Calculate center and scale
        const centerX = (minX + maxX) / 2;
        const centerY = (minY + maxY) / 2;
        const padding = 100;
        const boxWidth = Math.max(maxX - minX + padding * 2, 200);
        const boxHeight = Math.max(maxY - minY + padding * 2, 200);

        const scale = Math.min(
            this.width / boxWidth,
            this.height / boxHeight,
            2 // Max zoom level
        );

        const x = this.width / 2 - centerX * scale;
        const y = this.height / 2 - centerY * scale;

        // Animate to the new view
        this.svg.transition()
            .duration(750)
            .call(
                this.zoom.transform,
                d3.zoomIdentity.translate(x, y).scale(scale)
            );
    }

    clearHighlight() {
        this.nodes
            .classed('highlighted', false)
            .classed('connected', false)
            .classed('dimmed', false);

        // Clear glow effects
        this.nodes.selectAll('circle')
            .style('filter', null)
            .style('stroke', null)
            .style('stroke-width', null);

        this.links
            .classed('highlighted', false)
            .classed('connected', false)
            .classed('dimmed', false);
    }

    _buildResourcePayload(d) {
        const outgoing = [];
        const incoming = [];
        this.data.links.forEach(link => {
            const sourceId = link.source.id || link.source;
            const targetId = link.target.id || link.target;
            if (sourceId === d.id) {
                const targetNode = this.data.nodes.find(n => n.id === targetId);
                if (targetNode) outgoing.push({ type: link.type || link.relationship_type || 'RELATED_TO', target: targetNode.name });
            } else if (targetId === d.id) {
                const sourceNode = this.data.nodes.find(n => n.id === sourceId);
                if (sourceNode) incoming.push({ type: link.type || link.relationship_type || 'RELATED_TO', source: sourceNode.name });
            }
        });
        const payload = {
            id: d.id,
            name: d.name,
            type: d.type,
            domains: d.domains || [],
            description: d.description || '',
            importance: d.importance,
            mentions: d.mention_count,
            // `episodes` is the count (preserves existing UI); `episodeList` is
            // the raw int array of episode numbers used by the resource card's
            // episode-chip row.
            episodes: d.episode_count,
            episodeList: Array.isArray(d.episodes) ? d.episodes.slice() : [],
            aliases: d.aliases || [],
            relationships: { outgoing, incoming }
        };

        // Design 3: pass through EPISODE-specific fields so the resource card can
        // render the listen link + guest + theme chips.
        if (d.type === 'EPISODE') {
            payload.episode_number = d.episode_number;
            payload.display_name = d.display_name || d.name;
            payload.guest = d.guest || '';
            payload.org = d.org || '';
            payload.location = d.location || '';
            payload.themes = d.themes || [];
            payload.pillars = d.pillars || [];
            payload.url = d.url || '';
        }

        return payload;
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
        `);

        // Also notify parent window (chat /guide/ layout) with structured resource data
        if (window.parent !== window) {
            window.parent.postMessage({
                type: 'resourceSelected',
                resource: this._buildResourcePayload(d),
                source: 'node-click'
            }, window.location.origin);
        }
    }

    closeDetails() {
        const detailsContent = d3.select('#details-content');
        detailsContent.html('<p class="empty-state">Click a node to view details</p>');
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

    setupSimpleFilterStrip() {
        this.setupPrimaryPillarStrip();
        this.setupThemeStrip();
        this.setupTypeStrip();
    }

    setupPrimaryPillarStrip() {
        const strip = d3.select('#simple-filter-strip');
        if (strip.empty() || !this.data || !this.data.domains) return;

        strip.selectAll('*').remove();
        strip.append('span').attr('class', 'strip-label').text('Show');

        const setActive = (domainName) => {
            strip.selectAll('.filter-chip').classed('active', function () {
                return this.dataset.domain === domainName;
            });
        };

        strip.append('button')
            .attr('class', 'filter-chip active')
            .attr('data-domain', 'all')
            .text('All domains')
            .on('click', () => {
                this.data.domains.forEach(d => this.filters.domains.add(d.name));
                // Clear the pillar filter (pillar === domain via YOE taxonomy)
                this.filters.pillars.clear();
                setActive('all');
                this.updateVisualization();
            });

        this.data.domains.forEach(domain => {
            strip.append('button')
                .attr('class', 'filter-chip')
                .attr('data-domain', domain.name)
                .html(`<span class="chip-dot" style="background:${domain.color}"></span>${domain.name}`)
                .on('click', () => {
                    this.filters.domains.clear();
                    this.filters.domains.add(domain.name);
                    // If the taxonomy provides this domain as a pillar, also
                    // narrow by pillar.episode_ids (intersection with entity.episodes).
                    // pillarEpisodeIndex keys are UPPERCASE ('CULTURE'); KG domain.name
                    // is Title-case ('Culture'), so normalize before the lookup.
                    this.filters.pillars.clear();
                    const pillarKey = (domain.name || '').toUpperCase();
                    if (this.pillarEpisodeIndex && this.pillarEpisodeIndex.has(pillarKey)) {
                        this.filters.pillars.add(pillarKey);
                    }
                    setActive(domain.name);
                    this.updateVisualization();
                });
        });
    }

    setupThemeStrip() {
        const strip = d3.select('#theme-filter-strip');
        if (strip.empty() || !this.taxonomy || !this.taxonomy.themes) {
            // No taxonomy — hide the row entirely so it doesn't take vertical space.
            strip.style('display', 'none');
            return;
        }

        strip.selectAll('*').remove();
        strip.append('span').attr('class', 'strip-label').text('Themes');

        const setActive = (themeName) => {
            strip.selectAll('.filter-chip').classed('active', function () {
                return this.dataset.theme === themeName;
            });
        };

        strip.append('button')
            .attr('class', 'filter-chip active')
            .attr('data-theme', 'all')
            .text('All themes')
            .on('click', () => {
                this.filters.themes.clear();
                setActive('all');
                this.updateVisualization();
            });

        // Exclusive-select: only one theme at a time (plan §architecture).
        this.taxonomy.themes.forEach(theme => {
            strip.append('button')
                .attr('class', 'filter-chip')
                .attr('data-theme', theme.name)
                .attr('title', `${theme.count} episodes`)
                .text(theme.display)
                .on('click', () => {
                    this.filters.themes.clear();
                    this.filters.themes.add(theme.name);
                    setActive(theme.name);
                    this.updateVisualization();
                });
        });
    }

    setupTypeStrip() {
        const strip = d3.select('#type-filter-strip');
        if (strip.empty() || !this.data || !this.data.entity_types) return;

        strip.selectAll('*').remove();
        strip.append('span').attr('class', 'strip-label').text('Types');

        // Knowledge bundle = CONCEPT + PRACTICE + EVENT + TECHNOLOGY + SPECIES + ECOSYSTEM
        const KNOWLEDGE_BUNDLE = new Set([
            'CONCEPT', 'PRACTICE', 'EVENT',
            'TECHNOLOGY', 'SPECIES', 'ECOSYSTEM'
        ]);
        // Individual toggle types — the ones Aaron wants hidden by default.
        const INDIVIDUAL_TYPES = ['PLACE', 'PERSON', 'ORGANIZATION', 'PRODUCT', 'WORK'];

        const knowledgeTypesInData = this.data.entity_types.filter(t => KNOWLEDGE_BUNDLE.has(t));

        const isKnowledgeActive = () =>
            knowledgeTypesInData.every(t => this.filters.entityTypes.has(t));

        const knowledgeChip = strip.append('button')
            .attr('class', 'filter-chip' + (isKnowledgeActive() ? ' active' : ''))
            .attr('data-type', '__knowledge__')
            .html('Knowledge')
            .on('click', () => {
                const on = !isKnowledgeActive();
                knowledgeTypesInData.forEach(t => {
                    if (on) this.filters.entityTypes.add(t);
                    else this.filters.entityTypes.delete(t);
                });
                knowledgeChip.classed('active', on);
                this.updateVisualization();
            });

        // Episodes chip: force-show toggle for all 170 podcast episode nodes.
        // OFF (default): episodes appear only when a pillar/theme chip narrows.
        // ON: episodes always visible regardless of filter state.
        if (this.data.entity_types.includes('EPISODE')) {
            const epChip = strip.append('button')
                .attr('class', 'filter-chip' + (this.filters.forceShowEpisodes ? ' active' : ''))
                .attr('data-type', '__episodes__')
                .attr('title', 'Always show podcast episode nodes (off = show only when a pillar or theme is selected)')
                .html('Episodes')
                .on('click', () => {
                    this.filters.forceShowEpisodes = !this.filters.forceShowEpisodes;
                    epChip.classed('active', this.filters.forceShowEpisodes);
                    this.updateVisualization();
                });
        }

        INDIVIDUAL_TYPES.forEach(type => {
            // Skip types absent from the data.
            if (!this.data.entity_types.includes(type)) return;
            // Aaron-friendly plural labels — avoid autogenerated "Persons" etc.
            const LABEL_OVERRIDE = {
                PERSON: 'People',
                PLACE: 'Places',
                ORGANIZATION: 'Organizations',
                PRODUCT: 'Products',
                WORK: 'Works'
            };
            const label = LABEL_OVERRIDE[type] || (type.charAt(0) + type.slice(1).toLowerCase() + 's');
            const chip = strip.append('button')
                .attr('class', 'filter-chip' + (this.filters.entityTypes.has(type) ? ' active' : ''))
                .attr('data-type', type)
                .text(label)
                .on('click', () => {
                    const on = !this.filters.entityTypes.has(type);
                    if (on) this.filters.entityTypes.add(type);
                    else this.filters.entityTypes.delete(type);
                    chip.classed('active', on);
                    this.updateVisualization();
                });
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
        // Reset any pan/zoom so freshly-laid nodes are centered in the viewport.
        // Without this, a user who has zoomed in will click a filter chip and
        // end up staring at empty space while the new nodes spawn near the
        // world origin (the pre-zoom center).
        if (this.svg && this.zoom) {
            this.svg.call(this.zoom.transform, d3.zoomIdentity);
        }

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
                <span class="stat-label">Visible:</span>
                <span class="stat-value">${filteredData.nodes.length}${hitLimit ? ' (limited)' : ''}</span>
            </div>
            ${hitLimit ? `
                <div class="stat-item" style="color: #e67e22; font-size: 11px; margin-top: 5px;">
                    ⚠️ Showing top ${this.filters.maxNodes} by importance
                </div>
            ` : ''}
            <div class="stat-item">
                <span class="stat-label">Connections:</span>
                <span class="stat-value">${filteredData.links.length}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Total Topics & People:</span>
                <span class="stat-value">${this.data.nodes.length.toLocaleString()}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Topic Groups:</span>
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

    // Public methods for controls

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
    vizInstance = new KnowledgeGraphVisualization('#graph-svg-container');
    window.knowledgeGraph = vizInstance; // For debugging

    // Handle URL parameters for deep-linking (e.g., ?entity=biochar or ?search=permaculture)
    handleUrlParams();
});

/**
 * Handle URL parameters for entity deep-linking
 * Supports:
 *   ?entity=<name> - Highlight and focus on a specific entity
 *   ?search=<query> - Populate search and filter results
 */
function handleUrlParams() {
    const params = new URLSearchParams(window.location.search);
    const entityName = params.get('entity');
    const searchQuery = params.get('search');

    if (entityName) {
        // Wait for graph to fully initialize and load data
        const checkAndHighlight = () => {
            if (vizInstance && vizInstance.data && vizInstance.data.nodes) {
                // Find matching node by id, name, or alias
                const searchTerm = entityName.toLowerCase().trim();
                const node = vizInstance.data.nodes.find(n => {
                    const nodeId = n.id.toLowerCase();
                    const nodeName = n.name.toLowerCase();
                    const aliases = (n.aliases || []).map(a => a.toLowerCase());

                    return nodeId === searchTerm ||
                           nodeName === searchTerm ||
                           nodeId.includes(searchTerm) ||
                           nodeName.includes(searchTerm) ||
                           aliases.some(a => a === searchTerm || a.includes(searchTerm));
                });

                if (node) {
                    console.log('Deep-link: Found entity', node.name);
                    // Highlight the entity
                    vizInstance.highlightEntities([node.name]);
                    // Show details panel
                    vizInstance.showDetails(node);
                    vizInstance.selectedNode = node;
                    // Focus on the node after a short delay for simulation to stabilize
                    setTimeout(() => {
                        vizInstance.focusOnNode(node);
                    }, 1500);
                } else {
                    console.log('Deep-link: Entity not found:', entityName);
                    // Try a search instead
                    document.getElementById('search-input').value = entityName;
                    vizInstance.searchEntities(entityName);
                }
            } else {
                // Retry until graph is ready
                setTimeout(checkAndHighlight, 200);
            }
        };
        // Start checking after initial load
        setTimeout(checkAndHighlight, 500);

    } else if (searchQuery) {
        // Wait for graph to load, then perform search
        const checkAndSearch = () => {
            if (vizInstance && vizInstance.data) {
                document.getElementById('search-input').value = searchQuery;
                vizInstance.searchEntities(searchQuery);
            } else {
                setTimeout(checkAndSearch, 200);
            }
        };
        setTimeout(checkAndSearch, 500);
    }
}

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

    // Listen for messages from parent window (when embedded in iframe)
    window.addEventListener('message', (event) => {
        if (!event.data || !event.data.type) return;
        if (event.origin !== window.location.origin) {
            console.warn('KG: postMessage from unexpected origin dropped:', event.origin);
            return;
        }

        switch (event.data.type) {
            case 'highlightEntities':
                if (vizInstance && event.data.entities) {
                    const matched = vizInstance.highlightEntities(event.data.entities);
                    if (event.source) {
                        event.source.postMessage({
                            type: 'highlightResult',
                            matched: matched,
                            requested: event.data.entities
                        }, window.location.origin);
                    }
                }
                break;

            case 'clearHighlights':
            case 'clearSelection':
                if (vizInstance) vizInstance.clearHighlight();
                if (vizInstance) vizInstance.selectedNode = null;
                break;

            case 'requestResource': {
                if (!vizInstance || !vizInstance.data) {
                    if (event.source) {
                        event.source.postMessage({
                            type: 'resourceError',
                            requestId: event.data.requestId,
                            error: 'not_found',
                            query: event.data.query
                        }, window.location.origin);
                    }
                    break;
                }
                const queryName = event.data.query && event.data.query.name ? event.data.query.name.toLowerCase() : '';
                const matches = vizInstance.data.nodes.filter(n => n.name && n.name.toLowerCase() === queryName);
                const node = matches.length > 0
                    ? matches.reduce((best, n) => (n.importance || 0) > (best.importance || 0) ? n : best, matches[0])
                    : null;
                if (node) {
                    // Highlight and focus the node in the graph
                    vizInstance.selectedNode = node;
                    vizInstance.highlightEntities([node.name]);
                    event.source.postMessage({
                        type: 'resourceSelected',
                        requestId: event.data.requestId,
                        resource: vizInstance._buildResourcePayload(node),
                        source: 'link-click'
                    }, window.location.origin);
                } else {
                    event.source.postMessage({
                        type: 'resourceError',
                        requestId: event.data.requestId,
                        error: 'not_found',
                        query: event.data.query
                    }, window.location.origin);
                }
                break;
            }
        }
    });

    // Notify parent window that KG is ready
    if (window.parent !== window) {
        window.parent.postMessage({ type: 'kgReady' }, window.location.origin);
    }

    // Also listen on BroadcastChannel for cross-frame communication
    try {
        const channel = new BroadcastChannel('chat-kg-sync');
        channel.onmessage = (event) => {
            if (!event.data || !event.data.type) return;

            if (event.data.type === 'highlight' || event.data.type === 'highlightEntities') {
                const entities = event.data.entities || event.data.nodeIds;
                if (vizInstance && entities) {
                    vizInstance.highlightEntities(entities);
                }
            } else if (event.data.type === 'clear-highlights' || event.data.type === 'clearHighlights') {
                if (vizInstance) {
                    vizInstance.clearHighlight();
                }
            }
        };
    } catch (e) {
        console.log('BroadcastChannel not supported');
    }
});
