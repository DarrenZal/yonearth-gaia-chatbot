/**
 * Podcast Map Visualization
 * Interactive D3.js visualization for podcast episode exploration
 */

class PodcastMapVisualization {
    constructor(containerId) {
        this.container = d3.select(containerId);
        this.data = null;
        this.selectedEpisode = null;
        this.currentActiveChunk = null;
        this.audioPlayer = document.getElementById('audio-element');
        this.tooltip = d3.select('#tooltip');

        // Dimensions
        this.margin = { top: 20, right: 20, bottom: 20, left: 20 };
        this.width = 800;
        this.height = 500;

        // Scales
        this.xScale = null;
        this.yScale = null;

        // Voronoi for regions
        this.voronoi = null;

        // SVG elements
        this.svg = null;
        this.g = null;

        // Animation settings
        this.transitionDuration = 300;

        // Audio synchronization
        this.audioSyncInterval = null;

        // Hierarchical state with semantic zoom
        this.currentLevel = 'c9';  // Start with 9 clusters (medium detail)
        this.hierarchy = null;
        this.clusterLevels = [];  // Will store [1, 2, 3, ..., 18]

        // Zoom state
        this.zoom = null;
        this.currentZoomScale = 1.0;
        this.semanticZoomEnabled = true; // Start with semantic zoom enabled

        // Initialize
        this.init();
    }

    async init() {
        // Set up the SVG
        this.setupSVG();

        // Set up hierarchy controls
        this.setupHierarchyControls();

        // Load data from backend
        await this.loadData();

        // Set up scales
        this.setupScales();

        // Set up zoom behavior (before creating visualization)
        this.setupZoom();

        // Create visualization
        this.createVisualization();

        // Update cluster labels for initial level
        this.updateClusterLabels();

        // Populate legend
        this.populateLegend();

        // Set up event listeners
        this.setupEventListeners();

        // Set up mode toggle and slider
        this.setupModeToggle();
        this.setupClusterSlider();

        // Load episode list
        this.loadEpisodeList();
    }

    setupSVG() {
        // Get container dimensions - fill entire space
        const containerRect = this.container.node().getBoundingClientRect();
        this.width = containerRect.width;
        this.height = containerRect.height;

        // Create SVG that fills the container
        this.svg = this.container.append('svg')
            .attr('width', '100%')
            .attr('height', '100%')
            .attr('viewBox', `0 0 ${this.width} ${this.height}`)
            .attr('preserveAspectRatio', 'xMidYMid meet')
            .attr('class', 'podcast-map-svg');

        // Add background
        this.svg.append('rect')
            .attr('width', this.width)
            .attr('height', this.height)
            .attr('fill', 'transparent');

        // Create main group for zoom transform
        this.g = this.svg.append('g')
            .attr('class', 'zoom-group')
            .attr('transform', `translate(0, 0)`);
    }

    async loadData() {
        try {
            const response = await fetch('/api/map_data_hierarchical');
            const data = await response.json();
            this.data = data;
            this.hierarchy = data.hierarchy;
            this.clusterLevels = data.cluster_levels || [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18];
            this.currentLevel = data.default_level || 'c9';
            console.log('Loaded hierarchical map data:', data);
            console.log('Cluster levels available:', this.clusterLevels);
        } catch (error) {
            console.error('Error loading map data:', error);
            // Use dummy data for testing
            this.data = this.generateDummyData();
        }
    }

    generateDummyData() {
        // Generate dummy data for testing without backend
        const points = [];
        const numPoints = 266;
        const clusters = 7;

        for (let i = 0; i < numPoints; i++) {
            const cluster = Math.floor(Math.random() * clusters);
            const angle = (cluster / clusters) * 2 * Math.PI + (Math.random() - 0.5);
            const radius = 100 + Math.random() * 100;

            points.push({
                id: `point-${i}`,
                text: `This is chunk ${i} from the podcast discussing various topics...`,
                x: Math.cos(angle) * radius + this.width / 2,
                y: Math.sin(angle) * radius + this.height / 2,
                episode_id: `ep-${Math.floor(i / 30)}`,
                episode_title: `Episode ${Math.floor(i / 30) + 1}`,
                timestamp: (i % 30) * 120, // 2 minutes per chunk
                cluster: cluster,
                cluster_name: this.getClusterName(cluster)
            });
        }

        return {
            points: points,
            clusters: this.getClusterInfo(),
            episodes: this.extractEpisodesFromPoints(points),
            total_points: points.length
        };
    }

    getClusterName(clusterId) {
        // For hierarchical data, use current level's cluster name
        if (this.hierarchy && this.currentLevel) {
            const clusters = this.getCurrentClusters();
            const cluster = clusters.find(c => c.id === clusterId);
            if (cluster && cluster.name) {
                return cluster.name;
            }
        }
        // Use semantic labels from data instead of hardcoded names
        if (this.data && this.data.clusters) {
            const cluster = this.data.clusters.find(c => c.id === clusterId);
            if (cluster && cluster.name) {
                return cluster.name;
            }
        }
        // Fallback for old data format
        const names = [
            "Community",
            "Culture",
            "Economy",
            "Ecology",
            "Health"
        ];
        return names[clusterId % names.length];
    }

    getClusterInfo() {
        // For hierarchical data, return current level clusters
        if (this.hierarchy && this.currentLevel) {
            return this.getCurrentClusters();
        }
        // Use clusters from data if available (includes semantic topic names)
        if (this.data && this.data.clusters) {
            return this.data.clusters;
        }
        // Fallback for old data format
        const colors = [
            "#81C784",  // Light Green - Community
            "#BA68C8",  // Light Purple - Culture
            "#FFB74D",  // Light Orange - Economy
            "#64B5F6",  // Light Blue - Ecology
            "#E57373"   // Light Red - Health
        ];
        return colors.map((color, i) => ({
            id: i,
            name: this.getClusterName(i),
            color: color,
            count: 0
        }));
    }

    extractEpisodesFromPoints(points) {
        const episodeMap = new Map();
        points.forEach(p => {
            if (!episodeMap.has(p.episode_id)) {
                episodeMap.set(p.episode_id, {
                    id: p.episode_id,
                    title: p.episode_title,
                    chunk_count: 0
                });
            }
            episodeMap.get(p.episode_id).chunk_count++;
        });
        return Array.from(episodeMap.values());
    }

    setupScales() {
        // Set up scales based on data
        const xExtent = d3.extent(this.data.points, d => d.x);
        const yExtent = d3.extent(this.data.points, d => d.y);

        this.xScale = d3.scaleLinear()
            .domain(xExtent)
            .range([0, this.width]);

        this.yScale = d3.scaleLinear()
            .domain(yExtent)
            .range([0, this.height]);
    }

    setupZoom() {
        // Create D3 zoom behavior with semantic zoom integration
        this.zoom = d3.zoom()
            .scaleExtent([0.5, 8])  // Allow zooming from 0.5x to 8x
            .on('zoom', (event) => {
                // Apply transform to main group
                this.g.attr('transform', event.transform);

                // Get current zoom scale
                const zoomScale = event.transform.k;
                this.currentZoomScale = zoomScale;

                // Only apply semantic zoom if enabled
                if (this.semanticZoomEnabled) {
                    // Map zoom scale to cluster level (semantic zoom)
                    const newLevel = this.zoomScaleToClusterLevel(zoomScale);

                    // Switch cluster level if changed
                    if (newLevel !== this.currentLevel) {
                        console.log(`Semantic zoom: ${this.currentLevel} → ${newLevel} (zoom: ${zoomScale.toFixed(2)}x)`);
                        this.switchLevel(newLevel);
                    }
                }

                // Update breadcrumb regardless of mode
                this.updateBreadcrumb();
            });

        // Apply zoom to SVG
        this.svg.call(this.zoom);
    }

    zoomScaleToClusterLevel(zoomScale) {
        /**
         * Map zoom scale to cluster count using smooth interpolation
         * Zoom 0.5x → 1 cluster (most zoomed out, broadest view)
         * Zoom 1.0x → 9 clusters (default medium detail)
         * Zoom 8.0x → 18 clusters (most zoomed in, finest detail)
         */

        // Define zoom breakpoints
        const minZoom = 0.5;
        const maxZoom = 8.0;
        const minClusters = 1;
        const maxClusters = 18;

        // Clamp zoom scale to valid range
        const clampedZoom = Math.max(minZoom, Math.min(maxZoom, zoomScale));

        // Exponential mapping for better distribution
        // More clusters at higher zoom levels
        const normalizedZoom = (clampedZoom - minZoom) / (maxZoom - minZoom);
        const clusterCount = Math.round(minClusters + normalizedZoom * (maxClusters - minClusters));

        // Ensure we have this cluster level
        const validClusterCount = this.clusterLevels.includes(clusterCount)
            ? clusterCount
            : this.clusterLevels.reduce((prev, curr) =>
                Math.abs(curr - clusterCount) < Math.abs(prev - clusterCount) ? curr : prev
            );

        return `c${validClusterCount}`;
    }

    createVisualization() {
        // Create Voronoi regions for clusters
        this.createVoronoiRegions();

        // Create points
        this.createPoints();

        // Create trajectory lines (initially hidden)
        this.trajectoryGroup = this.g.append('g')
            .attr('class', 'trajectory-lines')
            .style('display', 'none');
    }

    createVoronoiRegions() {
        // Group points by cluster
        const clusterCentroids = d3.rollup(
            this.data.points,
            points => ({
                x: d3.mean(points, p => this.xScale(p.x)),
                y: d3.mean(points, p => this.yScale(p.y)),
                cluster: points[0][`cluster_${this.currentLevel}`] || points[0].cluster,
                color: this.getPointColorForLevel(points[0])
            }),
            d => d.cluster
        );

        const centroids = Array.from(clusterCentroids.values());

        // Create Voronoi diagram
        const delaunay = d3.Delaunay.from(centroids, d => d.x, d => d.y);
        this.voronoi = delaunay.voronoi([0, 0, this.width, this.height]);

        // Draw Voronoi cells - use selectOrAppend pattern to avoid duplicates
        let voronoiGroup = this.g.select('.voronoi-cells');
        if (voronoiGroup.empty()) {
            voronoiGroup = this.g.append('g').attr('class', 'voronoi-cells');
        }

        voronoiGroup.selectAll('path')
            .data(centroids)
            .join('path')
            .attr('d', (d, i) => this.voronoi.renderCell(i))
            .attr('fill', 'none')  // White background - no colored regions
            .attr('opacity', 0)
            .attr('stroke', 'none');

        // Add text labels for each cluster region - use selectOrAppend pattern
        let labelsGroup = this.g.select('.cluster-labels');
        if (labelsGroup.empty()) {
            labelsGroup = this.g.append('g').attr('class', 'cluster-labels');
        }

        labelsGroup.selectAll('text')
            .data(centroids)
            .join('text')
            .attr('x', d => d.x)
            .attr('y', d => d.y)
            .attr('text-anchor', 'middle')
            .attr('dominant-baseline', 'middle')
            .attr('class', 'cluster-label')
            .style('font-size', '18px')
            .style('font-weight', '600')
            .style('pointer-events', 'none')
            .text(d => this.getClusterName(d.cluster));
    }

    createPoints() {
        // Create circles for each data point
        const pointsGroup = this.g.append('g')
            .attr('class', 'points');

        this.circles = pointsGroup.selectAll('circle')
            .data(this.data.points)
            .join('circle')
            .attr('cx', d => this.xScale(d.x))
            .attr('cy', d => this.yScale(d.y))
            .attr('r', 3)
            .attr('fill', d => this.getPointColorForLevel(d))
            .attr('opacity', 0.7)
            .attr('class', 'point')
            .attr('data-episode', d => d.episode_id)
            .attr('data-cluster', d => d.cluster)
            .on('mouseover', (event, d) => this.handleMouseOver(event, d))
            .on('mouseout', () => this.handleMouseOut())
            .on('click', (event, d) => this.handleClick(event, d));
    }

    handleMouseOver(event, d) {
        // Show tooltip
        const [mouseX, mouseY] = d3.pointer(event, document.body);

        this.tooltip
            .style('display', 'block')
            .style('left', (mouseX + 10) + 'px')
            .style('top', (mouseY - 10) + 'px');

        this.tooltip.select('.tooltip-episode').text(d.episode_title || 'Unknown Episode');
        this.tooltip.select('.tooltip-text').text(d.text.substring(0, 150) + '...');
        this.tooltip.select('.tooltip-cluster').text(d.cluster_name);

        // Highlight point
        d3.select(event.target)
            .transition()
            .duration(100)
            .attr('r', 6)
            .attr('opacity', 1);
    }

    handleMouseOut() {
        // Hide tooltip
        this.tooltip.style('display', 'none');

        // Reset point size
        this.circles
            .transition()
            .duration(100)
            .attr('r', d => {
                if (this.selectedEpisode && d.episode_id === this.selectedEpisode) {
                    return 4;
                }
                return 3;
            })
            .attr('opacity', d => {
                if (this.selectedEpisode && d.episode_id === this.selectedEpisode) {
                    return 0.8;
                }
                return 0.7;
            });
    }

    handleClick(event, d) {
        console.log('Clicked point:', d);

        // Play audio at timestamp
        if (d.timestamp !== null && d.timestamp !== undefined) {
            this.playAudioAtTimestamp(d);
        }

        // Show trajectory for this episode
        this.showEpisodeTrajectory(d.episode_id);
    }

    playAudioAtTimestamp(point) {
        // Show audio player
        const playerContainer = document.getElementById('audio-player');
        playerContainer.style.display = 'block';

        // Update episode info
        document.getElementById('current-episode-title').textContent = point.episode_title;
        document.getElementById('current-chunk-text').textContent = point.text.substring(0, 200) + '...';

        // Set audio source (you'll need to configure actual audio URLs)
        const audioUrl = `/audio/episodes/${point.episode_id}.mp3`;
        this.audioPlayer.src = audioUrl;

        // Seek to timestamp
        this.audioPlayer.currentTime = point.timestamp;

        // Play
        this.audioPlayer.play();
    }

    showEpisodeTrajectory(episodeId) {
        this.selectedEpisode = episodeId;
        this.currentActiveChunk = null; // Reset active chunk

        // Get all points for this episode, sorted by timestamp
        const episodePoints = this.data.points
            .filter(p => p.episode_id === episodeId)
            .sort((a, b) => (a.timestamp || 0) - (b.timestamp || 0));

        if (episodePoints.length < 2) return;

        // Clear existing lines
        this.trajectoryGroup.selectAll('*').remove();
        this.trajectoryGroup.style('display', 'block');

        // Create line generator
        const line = d3.line()
            .x(d => this.xScale(d.x))
            .y(d => this.yScale(d.y))
            .curve(d3.curveCardinal.tension(0.5));

        // Draw lines connecting sequential chunks
        for (let i = 0; i < episodePoints.length - 1; i++) {
            const isAdjacent = Math.abs((episodePoints[i].timestamp || 0) - (episodePoints[i + 1].timestamp || 0)) < 180;

            this.trajectoryGroup.append('path')
                .datum([episodePoints[i], episodePoints[i + 1]])
                .attr('d', line)
                .attr('stroke', this.getPointColorForLevel(episodePoints[i]))
                .attr('stroke-width', isAdjacent ? 3 : 1)
                .attr('fill', 'none')
                .attr('opacity', isAdjacent ? 0.6 : 0.3);
        }

        // Highlight episode points - improved contrast
        this.circles
            .transition()
            .duration(this.transitionDuration)
            .attr('r', d => d.episode_id === episodeId ? 5 : 3)
            .attr('opacity', d => {
                if (d.episode_id === episodeId) return 1.0; // Full opacity for selected
                return 0.25; // Visible but clearly secondary
            })
            .attr('stroke', d => d.episode_id === episodeId ? '#ffffff' : 'none')
            .attr('stroke-width', d => d.episode_id === episodeId ? 1 : 0);
    }

    setupEventListeners() {
        // Episode selector
        document.getElementById('episode-select').addEventListener('change', (e) => {
            const episodeId = e.target.value;
            if (episodeId) {
                const firstPoint = this.data.points.find(p => p.episode_id === episodeId);
                if (firstPoint) {
                    this.showEpisodeTrajectory(episodeId);
                    this.loadAudioForEpisode(episodeId);
                }
            } else {
                this.clearSelection();
            }
        });

        // Audio timeupdate for synchronization
        if (this.audioPlayer) {
            this.audioPlayer.addEventListener('timeupdate', () => {
                if (this.selectedEpisode) {
                    this.syncVisualizationWithAudio();
                }
            });

            // Also listen for seeked event (when user manually changes position)
            this.audioPlayer.addEventListener('seeked', () => {
                if (this.selectedEpisode) {
                    console.log('Audio seeked to:', this.audioPlayer.currentTime);
                    this.syncVisualizationWithAudio();
                }
            });

            this.audioPlayer.addEventListener('play', () => {
                console.log('Audio started playing');
            });

            this.audioPlayer.addEventListener('pause', () => {
                console.log('Audio paused');
            });
        }

        // Window resize
        window.addEventListener('resize', () => this.handleResize());
    }

    loadEpisodeList() {
        const select = document.getElementById('episode-select');

        // Clear existing options
        select.innerHTML = '<option value="">Select an episode...</option>';

        // Sort episodes numerically by episode number
        const sortedEpisodes = [...this.data.episodes].sort((a, b) => {
            // Extract episode numbers - IDs can be just numbers or "Episode X" format
            const numA = parseInt(a.id) || parseInt(a.id.match(/\d+/)?.[0] || '999999');
            const numB = parseInt(b.id) || parseInt(b.id.match(/\d+/)?.[0] || '999999');
            return numA - numB;
        });

        // Add episodes
        sortedEpisodes.forEach(episode => {
            const option = document.createElement('option');
            option.value = episode.id;
            option.textContent = episode.title;
            select.appendChild(option);
        });
    }

    clearSelection() {
        this.selectedEpisode = null;

        // Hide trajectory
        this.trajectoryGroup.style('display', 'none');

        // Reset all points
        this.circles
            .transition()
            .duration(this.transitionDuration)
            .attr('r', 3)
            .attr('opacity', 0.7);
    }

    handleResize() {
        // Recalculate dimensions
        const containerRect = this.container.node().getBoundingClientRect();
        this.width = containerRect.width;
        this.height = containerRect.height;

        // Update SVG viewBox for responsive scaling
        this.svg
            .attr('viewBox', `0 0 ${this.width} ${this.height}`);

        // Update scales
        this.xScale.range([0, this.width]);
        this.yScale.range([0, this.height]);

        // Redraw visualization smoothly
        this.updateVisualization();
    }

    populateLegend() {
        const legendContainer = document.getElementById('clusters-legend');
        if (!legendContainer || !this.data || !this.data.clusters) return;

        // Clear existing legend
        legendContainer.innerHTML = '';

        // Create legend items from data
        this.data.clusters.forEach(cluster => {
            const legendItem = document.createElement('div');
            legendItem.className = 'legend-item';
            legendItem.setAttribute('data-cluster', cluster.id);

            // Add color box
            const colorBox = document.createElement('span');
            colorBox.className = 'color-box';
            colorBox.style.backgroundColor = cluster.color;

            // Add label
            const label = document.createElement('span');
            label.textContent = cluster.name;

            legendItem.appendChild(colorBox);
            legendItem.appendChild(label);

            // Add hover tooltip with themes if available
            if (cluster.themes && cluster.themes.length > 0) {
                legendItem.title = `Themes: ${cluster.themes.join(', ')}`;
            }

            legendContainer.appendChild(legendItem);
        });
    }

    updateVisualization() {
        // Update Voronoi regions and labels (createVoronoiRegions now handles reuse)
        this.createVoronoiRegions();

        // Update point positions
        this.circles
            .transition()
            .duration(this.transitionDuration)
            .attr('cx', d => this.xScale(d.x))
            .attr('cy', d => this.yScale(d.y));

        // Update trajectory lines if an episode is selected
        if (this.selectedEpisode) {
            this.updateTrajectoryLines();
        }
    }

    updateTrajectoryLines() {
        // Get episode points sorted by timestamp
        const episodePoints = this.data.points
            .filter(p => p.episode_id === this.selectedEpisode)
            .sort((a, b) => (a.timestamp || 0) - (b.timestamp || 0));

        // Create segments
        const segments = [];
        for (let i = 0; i < episodePoints.length - 1; i++) {
            segments.push([episodePoints[i], episodePoints[i + 1]]);
        }

        // Update trajectory lines with new scale
        const line = d3.line()
            .x(d => this.xScale(d.x))
            .y(d => this.yScale(d.y))
            .curve(d3.curveCatmullRom.alpha(0.5));

        this.trajectoryGroup.selectAll('path')
            .data(segments)
            .transition()
            .duration(this.transitionDuration)
            .attr('d', line);
    }

    async loadAudioForEpisode(episodeId) {
        // Show audio player
        const playerContainer = document.getElementById('audio-player');
        if (playerContainer) {
            playerContainer.style.display = 'block';
        }

        // Get episode info
        const episode = this.data.episodes.find(ep => ep.id === episodeId);
        if (!episode) return;

        // Update episode info in player
        const titleElement = document.getElementById('current-episode-title');
        if (titleElement) {
            titleElement.textContent = episode.title;
        }

        const chunkText = document.getElementById('current-chunk-text');
        if (chunkText) {
            chunkText.textContent = 'Loading audio... Click play when ready.';
        }

        // Fetch the actual audio URL from the transcript file
        try {
            // Episode ID might be just a number or "Episode X" format
            let episodeNumber = episodeId;
            if (episodeId.includes('Episode')) {
                episodeNumber = episodeId.split(' ')[0].replace('Episode', '').trim();
            }
            const response = await fetch(`/data/transcripts/episode_${episodeNumber}.json`);
            const transcriptData = await response.json();

            const audioUrl = transcriptData.audio_url;

            if (this.audioPlayer && audioUrl) {
                this.audioPlayer.src = audioUrl;
                this.audioPlayer.load();
                console.log('Loaded audio from:', audioUrl);

                if (chunkText) {
                    chunkText.textContent = 'Audio loaded! Click play to begin.';
                }
            }
        } catch (error) {
            console.error('Error loading audio URL:', error);
            if (this.audioPlayer) {
                const chunkText = document.getElementById('current-chunk-text');
                if (chunkText) {
                    chunkText.textContent = 'Audio not available for this episode.';
                }
            }
        }
    }

    syncVisualizationWithAudio() {
        if (!this.audioPlayer || !this.selectedEpisode) return;

        const currentTime = this.audioPlayer.currentTime;

        // Find the chunk that corresponds to current audio time
        const episodePoints = this.data.points.filter(p => p.episode_id === this.selectedEpisode);

        // Find the chunk closest to current time
        let activeChunk = null;
        let minDiff = Infinity;

        for (const point of episodePoints) {
            if (point.timestamp === null || point.timestamp === undefined) continue;

            const diff = Math.abs(point.timestamp - currentTime);
            if (diff < minDiff) {
                minDiff = diff;
                activeChunk = point;
            }
        }

        // Only update if we found a different chunk
        if (activeChunk && activeChunk.id !== this.currentActiveChunk?.id) {
            this.currentActiveChunk = activeChunk;
            this.highlightActiveChunk(activeChunk);

            // Update chunk text in player
            const chunkText = document.getElementById('current-chunk-text');
            if (chunkText && activeChunk.text) {
                chunkText.textContent = activeChunk.text.substring(0, 200) + '...';
            }
        }
    }

    highlightActiveChunk(activeChunk) {
        if (!activeChunk) return;

        // Get episode points for calculating neighbors
        const episodePoints = this.data.points
            .filter(p => p.episode_id === this.selectedEpisode)
            .sort((a, b) => (a.timestamp || 0) - (b.timestamp || 0));

        const activeIndex = episodePoints.findIndex(p => p.id === activeChunk.id);
        const prevChunk = activeIndex > 0 ? episodePoints[activeIndex - 1] : null;
        const nextChunk = activeIndex < episodePoints.length - 1 ? episodePoints[activeIndex + 1] : null;

        // Highlight the current, previous, and next chunks - better visibility
        this.circles
            .transition()
            .duration(200)
            .attr('r', d => {
                if (d.id === activeChunk.id) return 10; // Active chunk - largest
                if (d.episode_id === this.selectedEpisode) {
                    if (prevChunk && d.id === prevChunk.id) return 7; // Previous chunk
                    if (nextChunk && d.id === nextChunk.id) return 7; // Next chunk
                    return 5; // Other episode chunks
                }
                return 3; // All other points
            })
            .attr('opacity', d => {
                if (d.id === activeChunk.id) return 1.0; // Active chunk - fully visible
                if (d.episode_id === this.selectedEpisode) {
                    if (prevChunk && d.id === prevChunk.id) return 0.9;
                    if (nextChunk && d.id === nextChunk.id) return 0.9;
                    return 0.7;
                }
                return 0.2; // Visible but clearly secondary
            })
            .attr('stroke', d => {
                if (d.id === activeChunk.id) return '#ffffff';
                return 'none';
            })
            .attr('stroke-width', d => {
                if (d.id === activeChunk.id) return 2;
                return 0;
            });

        // Emphasize edges to/from active chunk
        this.trajectoryGroup.selectAll('path')
            .transition()
            .duration(200)
            .attr('stroke-width', (d, i) => {
                // Check if this path connects to the active chunk
                const pathPoints = d; // d is array of 2 points
                if (pathPoints.some(p => p.id === activeChunk.id)) {
                    return 5; // Thick line for active connections
                }
                if (prevChunk && pathPoints.some(p => p.id === prevChunk.id)) {
                    return 3;
                }
                if (nextChunk && pathPoints.some(p => p.id === nextChunk.id)) {
                    return 3;
                }
                return 1; // Thin line for others
            })
            .attr('opacity', (d) => {
                const pathPoints = d;
                if (pathPoints.some(p => p.id === activeChunk.id)) {
                    return 0.9; // Highly visible for active connections
                }
                if (prevChunk && pathPoints.some(p => p.id === prevChunk.id)) {
                    return 0.7;
                }
                if (nextChunk && pathPoints.some(p => p.id === nextChunk.id)) {
                    return 0.7;
                }
                return 0.3; // Dim other lines
            });
    }

    setupHierarchyControls() {
        // Set up level switching buttons with zoom triggers
        const buttons = document.querySelectorAll('.level-btn');
        buttons.forEach(btn => {
            btn.addEventListener('click', () => {
                const level = btn.getAttribute('data-level');
                const targetZoom = parseFloat(btn.getAttribute('data-zoom')) || 1.0;

                // Apply zoom transition (which will automatically trigger level switch)
                this.svg.transition()
                    .duration(750)
                    .call(this.zoom.scaleTo, targetZoom);

                // Update button states
                buttons.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
            });
        });
    }

    switchLevel(newLevel) {
        console.log(`Switching from ${this.currentLevel} to ${newLevel}`);
        this.currentLevel = newLevel;

        // Update breadcrumb
        this.updateBreadcrumb();

        // Re-render with new level
        this.updateVisualizationLevel();
    }

    updateBreadcrumb() {
        const breadcrumb = document.getElementById('breadcrumb');

        // Extract cluster count from level (e.g., 'c9' → 9)
        const clusterCount = parseInt(this.currentLevel.substring(1));

        // Generate dynamic description based on cluster count
        let description = '';
        if (clusterCount <= 3) {
            description = `Very Broad View (${clusterCount} ${clusterCount === 1 ? 'Cluster' : 'Clusters'})`;
        } else if (clusterCount <= 9) {
            description = `Medium Detail (${clusterCount} Clusters)`;
        } else {
            description = `Fine Detail (${clusterCount} Clusters)`;
        }

        breadcrumb.innerHTML = `<span class="breadcrumb-item">${description} - Zoom: ${this.currentZoomScale.toFixed(2)}x</span>`;
    }

    updateVisualizationLevel() {
        if (!this.data || !this.hierarchy) return;

        // Update point colors and sizes based on current level
        const clusterCount = parseInt(this.currentLevel.substring(1));
        const pointRadius = clusterCount <= 3 ? 6 : clusterCount <= 9 ? 4 : 3;

        const pointsGroup = this.g.select('.points');
        if (!pointsGroup.empty()) {
            pointsGroup.selectAll('circle')
                .transition()
                .duration(500)
                .attr('fill', d => this.getPointColorForLevel(d))
                .attr('r', pointRadius);
        }

        // Update Voronoi colors
        const voronoiGroup = this.g.select('.voronoi');
        if (!voronoiGroup.empty()) {
            voronoiGroup.selectAll('path')
                .transition()
                .duration(500)
                .attr('fill', (d, i) => {
                    const point = this.data.points[i];
                    return this.getPointColorForLevel(point);
                })
                .attr('opacity', 0.1);
        }

        // Update cluster labels to show names for current level
        const labelsGroup = this.g.select('.cluster-labels');
        if (!labelsGroup.empty()) {
            // Need to recalculate centroids for current level
            this.updateClusterLabels();
        }

        console.log(`Visualization updated to level: ${this.currentLevel}`);
    }

    updateClusterLabels() {
        // Group points by current level cluster
        const clusterKey = `cluster_${this.currentLevel}`;
        const clusterGroups = d3.group(this.data.points, d => d[clusterKey]);

        // Calculate centroids for each cluster at current level
        const centroids = [];
        clusterGroups.forEach((points, clusterId) => {
            centroids.push({
                cluster: clusterId,
                x: d3.mean(points, p => this.xScale(p.x)),
                y: d3.mean(points, p => this.yScale(p.y)),
                name: points[0][`cluster_${this.currentLevel}_name`]
            });
        });

        // Update labels
        const labelsGroup = this.g.select('.cluster-labels');
        labelsGroup.selectAll('text')
            .data(centroids, d => d.cluster)
            .join(
                enter => enter.append('text')
                    .attr('x', d => d.x)
                    .attr('y', d => d.y)
                    .attr('text-anchor', 'middle')
                    .attr('dominant-baseline', 'middle')
                    .attr('class', 'cluster-label')
                    .style('font-size', '18px')
                    .style('font-weight', '600')
                    .style('pointer-events', 'none')
                    .style('opacity', 0)
                    .text(d => d.name)
                    .transition()
                    .duration(500)
                    .style('opacity', 1),
                update => update
                    .transition()
                    .duration(500)
                    .attr('x', d => d.x)
                    .attr('y', d => d.y)
                    .text(d => d.name),
                exit => exit
                    .transition()
                    .duration(500)
                    .style('opacity', 0)
                    .remove()
            );
    }

    getPointColorForLevel(point) {
        if (!this.hierarchy) return '#999';

        const levelKey = this.currentLevel;
        const clusterId = point[`cluster_${levelKey}`];

        if (clusterId === undefined) return '#999';

        // Get color from hierarchy
        const cluster = this.hierarchy[levelKey]?.[clusterId];
        return cluster?.color || '#999';
    }

    getCurrentClusterName(point) {
        return point[`cluster_${this.currentLevel}_name`] || 'Unknown';
    }

    getCurrentClusters() {
        // Get clusters for current level from hierarchy
        if (!this.hierarchy || !this.currentLevel) return [];
        return this.hierarchy[this.currentLevel] || [];
    }

    getClusterColor(clusterId) {
        // Get color for a cluster at current level
        const clusters = this.getCurrentClusters();
        const cluster = clusters.find(c => c.id === clusterId);
        return cluster?.color || '#999';
    }

    setupModeToggle() {
        const checkbox = document.getElementById('zoom-mode-checkbox');
        const modeLabel = document.getElementById('mode-label');
        const sliderContainer = document.getElementById('cluster-slider-container');

        if (!checkbox || !modeLabel || !sliderContainer) return;

        checkbox.addEventListener('change', (e) => {
            this.semanticZoomEnabled = e.target.checked;

            if (this.semanticZoomEnabled) {
                // Semantic zoom mode
                modeLabel.textContent = '🔍 Semantic Zoom';
                sliderContainer.style.display = 'none';
            } else {
                // Manual slider mode
                modeLabel.textContent = '🎚️ Manual Slider';
                sliderContainer.style.display = 'flex';
            }

            console.log(`Switched to ${this.semanticZoomEnabled ? 'Semantic Zoom' : 'Manual Slider'} mode`);
        });
    }

    setupClusterSlider() {
        const slider = document.getElementById('cluster-slider');
        const countDisplay = document.getElementById('cluster-count');

        if (!slider || !countDisplay) return;

        slider.addEventListener('input', (e) => {
            const clusterCount = parseInt(e.target.value);
            countDisplay.textContent = clusterCount;

            // Only switch if in manual mode
            if (!this.semanticZoomEnabled) {
                const newLevel = `c${clusterCount}`;
                if (newLevel !== this.currentLevel) {
                    console.log(`Manual slider: ${this.currentLevel} → ${newLevel}`);
                    this.switchLevel(newLevel);
                }
            }
        });
    }
}

// Utility functions
function closeAudioPlayer() {
    document.getElementById('audio-player').style.display = 'none';
    const audio = document.getElementById('audio-element');
    audio.pause();
}

function toggleChat() {
    const modal = document.getElementById('chat-modal');
    modal.style.display = modal.style.display === 'none' ? 'block' : 'none';
}

function sendChatMessage() {
    const input = document.getElementById('chat-input');
    const message = input.value.trim();
    if (message) {
        // Send to chat API
        console.log('Sending message:', message);
        input.value = '';
    }
}

// Initialize visualization when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    const viz = new PodcastMapVisualization('#map-svg-container');
    window.podcastViz = viz; // Make it globally accessible for debugging
});