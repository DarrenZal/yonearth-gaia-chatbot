/**
 * Podcast Map Visualization - Nomic Atlas Version
 * Interactive D3.js visualization using Nomic's UMAP projections and topic clustering
 */

class PodcastMapNomic {
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

        // Nomic topic colors
        this.topicColors = {};
        this.allTopics = [];

        // Initialize
        this.init();
    }

    async init() {
        // Set up the SVG
        this.setupSVG();

        // Load data from backend
        await this.loadData();

        // Extract topics and assign colors
        this.setupTopicColors();

        // Set up scales
        this.setupScales();

        // Create visualization
        this.createVisualization();

        // Update cluster labels
        this.updateClusterLabels();

        // Set up event listeners
        this.setupEventListeners();

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

        // Create main group
        this.g = this.svg.append('g')
            .attr('transform', `translate(0, 0)`);
    }

    async loadData() {
        try {
            const response = await fetch('/api/map_data_nomic');
            const data = await response.json();
            this.data = data;
            console.log('Loaded Nomic map data:', data);
            console.log(`Total points: ${data.points.length}`);
        } catch (error) {
            console.error('Error loading Nomic map data:', error);
            alert('Failed to load Nomic map data. Please ensure the data file exists.');
        }
    }

    setupTopicColors() {
        // Extract all unique topics from the data
        const topicsSet = new Set();
        this.data.points.forEach(p => {
            if (p.topic_depth_1) topicsSet.add(p.topic_depth_1);
        });
        this.allTopics = Array.from(topicsSet).sort();

        // Generate a color palette for topics
        const colorScale = d3.scaleOrdinal(d3.schemeTableau10);

        this.allTopics.forEach((topic, i) => {
            this.topicColors[topic] = colorScale(i);
        });

        console.log('Topics found:', this.allTopics);
    }

    setupScales() {
        // Set up scales based on data
        const xExtent = d3.extent(this.data.points, d => d.x);
        const yExtent = d3.extent(this.data.points, d => d.y);

        this.xScale = d3.scaleLinear()
            .domain(xExtent)
            .range([this.margin.left, this.width - this.margin.right]);

        this.yScale = d3.scaleLinear()
            .domain(yExtent)
            .range([this.margin.top, this.height - this.margin.bottom]);
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
        // Group points by topic_depth_1
        const topicCentroids = d3.rollup(
            this.data.points,
            points => ({
                x: d3.mean(points, p => this.xScale(p.x)),
                y: d3.mean(points, p => this.yScale(p.y)),
                topic: points[0].topic_depth_1,
                color: this.getPointColor(points[0])
            }),
            d => d.topic_depth_1
        );

        const centroids = Array.from(topicCentroids.values());

        // Create Voronoi diagram
        const delaunay = d3.Delaunay.from(centroids, d => d.x, d => d.y);
        this.voronoi = delaunay.voronoi([0, 0, this.width, this.height]);

        // Draw Voronoi cells
        let voronoiGroup = this.g.select('.voronoi-cells');
        if (voronoiGroup.empty()) {
            voronoiGroup = this.g.append('g').attr('class', 'voronoi-cells');
        }

        voronoiGroup.selectAll('path')
            .data(centroids)
            .join('path')
            .attr('d', (d, i) => this.voronoi.renderCell(i))
            .attr('fill', 'none')
            .attr('opacity', 0)
            .attr('stroke', 'none');

        // Add text labels for each topic region
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
            .style('font-size', '16px')
            .style('font-weight', '600')
            .style('pointer-events', 'none')
            .text(d => d.topic || 'Unknown');
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
            .attr('fill', d => this.getPointColor(d))
            .attr('opacity', 0.7)
            .attr('class', 'point')
            .attr('data-episode', d => d.episode_id)
            .attr('data-topic', d => d.topic_depth_1)
            .on('mouseover', (event, d) => this.handleMouseOver(event, d))
            .on('mouseout', () => this.handleMouseOut())
            .on('click', (event, d) => this.handleClick(event, d));
    }

    getPointColor(point) {
        const topic = point.topic_depth_1;
        return this.topicColors[topic] || '#999999';
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
        this.tooltip.select('.tooltip-cluster').text(`Topic: ${d.topic_depth_1 || 'Unknown'}`);

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

        // Show trajectory for this episode
        this.showEpisodeTrajectory(d.episode_id);

        // Play audio at timestamp
        if (d.timestamp !== null && d.timestamp !== undefined) {
            this.playAudioAtTimestamp(d);
        }
    }

    async playAudioAtTimestamp(point) {
        // Show audio player
        const playerContainer = document.getElementById('audio-player');
        playerContainer.style.display = 'block';

        // Update episode info
        document.getElementById('current-episode-title').textContent = point.episode_title;
        document.getElementById('current-chunk-text').textContent = point.text.substring(0, 200) + '...';

        // Fetch the actual audio URL from the transcript file
        try {
            let episodeNumber = point.episode_id;
            if (typeof episodeNumber === 'string' && episodeNumber.includes('Episode')) {
                episodeNumber = episodeNumber.split(' ')[0].replace('Episode', '').trim();
            }

            const response = await fetch(`/data/transcripts/episode_${episodeNumber}.json`);
            const transcriptData = await response.json();
            const audioUrl = transcriptData.audio_url;

            if (this.audioPlayer && audioUrl) {
                // Only change source if it's a different episode
                if (this.audioPlayer.src !== audioUrl) {
                    this.audioPlayer.src = audioUrl;
                    await this.audioPlayer.load();
                }

                // Seek to timestamp and play
                this.audioPlayer.currentTime = point.timestamp || 0;
                await this.audioPlayer.play();
                console.log('Playing audio at timestamp:', point.timestamp);
            }
        } catch (error) {
            console.error('Error loading audio:', error);
            const chunkText = document.getElementById('current-chunk-text');
            if (chunkText) {
                chunkText.textContent = 'Audio not available for this episode.';
            }
        }
    }

    showEpisodeTrajectory(episodeId) {
        this.selectedEpisode = episodeId;
        this.currentActiveChunk = null;

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
                .attr('stroke', this.getPointColor(episodePoints[i]))
                .attr('stroke-width', isAdjacent ? 3 : 1)
                .attr('fill', 'none')
                .attr('opacity', isAdjacent ? 0.6 : 0.3);
        }

        // Highlight episode points
        this.circles
            .transition()
            .duration(this.transitionDuration)
            .attr('r', d => d.episode_id === episodeId ? 5 : 3)
            .attr('opacity', d => {
                if (d.episode_id === episodeId) return 1.0;
                return 0.25;
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

            this.audioPlayer.addEventListener('seeked', () => {
                if (this.selectedEpisode) {
                    this.syncVisualizationWithAudio();
                }
            });
        }

        // Window resize
        window.addEventListener('resize', () => this.handleResize());
    }

    loadEpisodeList() {
        const select = document.getElementById('episode-select');

        // Clear existing options
        select.innerHTML = '<option value="">Select an episode...</option>';

        // Extract unique episodes from points
        const episodeMap = new Map();
        this.data.points.forEach(p => {
            if (!episodeMap.has(p.episode_id)) {
                episodeMap.set(p.episode_id, {
                    id: p.episode_id,
                    title: p.episode_title
                });
            }
        });

        const episodes = Array.from(episodeMap.values());

        // Sort episodes numerically
        const sortedEpisodes = episodes.sort((a, b) => {
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
            .attr('opacity', 0.7)
            .attr('stroke', 'none');
    }

    handleResize() {
        // Recalculate dimensions
        const containerRect = this.container.node().getBoundingClientRect();
        this.width = containerRect.width;
        this.height = containerRect.height;

        // Update SVG viewBox
        this.svg.attr('viewBox', `0 0 ${this.width} ${this.height}`);

        // Update scales
        this.xScale.range([this.margin.left, this.width - this.margin.right]);
        this.yScale.range([this.margin.top, this.height - this.margin.bottom]);

        // Redraw visualization
        this.updateVisualization();
    }

    updateVisualization() {
        // Update Voronoi regions and labels
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

        // Update episode title
        const point = this.data.points.find(p => p.episode_id === episodeId);
        const titleElement = document.getElementById('current-episode-title');
        if (titleElement && point) {
            titleElement.textContent = point.episode_title;
        }

        const chunkText = document.getElementById('current-chunk-text');
        if (chunkText) {
            chunkText.textContent = 'Loading audio... Click play when ready.';
        }

        // Fetch the actual audio URL from the transcript file
        try {
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
            if (chunkText) {
                chunkText.textContent = 'Audio not available for this episode.';
            }
        }
    }

    syncVisualizationWithAudio() {
        if (!this.audioPlayer || !this.selectedEpisode) return;

        const currentTime = this.audioPlayer.currentTime;

        // Find the chunk that corresponds to current audio time
        const episodePoints = this.data.points.filter(p => p.episode_id === this.selectedEpisode);

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

        // Highlight the current, previous, and next chunks
        this.circles
            .transition()
            .duration(200)
            .attr('r', d => {
                if (d.id === activeChunk.id) return 10;
                if (d.episode_id === this.selectedEpisode) {
                    if (prevChunk && d.id === prevChunk.id) return 7;
                    if (nextChunk && d.id === nextChunk.id) return 7;
                    return 5;
                }
                return 3;
            })
            .attr('opacity', d => {
                if (d.id === activeChunk.id) return 1.0;
                if (d.episode_id === this.selectedEpisode) {
                    if (prevChunk && d.id === prevChunk.id) return 0.9;
                    if (nextChunk && d.id === nextChunk.id) return 0.9;
                    return 0.7;
                }
                return 0.2;
            })
            .attr('stroke', d => {
                if (d.id === activeChunk.id) return '#ffffff';
                return 'none';
            })
            .attr('stroke-width', d => {
                if (d.id === activeChunk.id) return 2;
                return 0;
            });
    }

    updateClusterLabels() {
        // Group points by topic
        const topicGroups = d3.group(this.data.points, d => d.topic_depth_1);

        // Calculate centroids for each topic
        const centroids = [];
        topicGroups.forEach((points, topic) => {
            centroids.push({
                topic: topic,
                x: d3.mean(points, p => this.xScale(p.x)),
                y: d3.mean(points, p => this.yScale(p.y))
            });
        });

        // Update labels
        const labelsGroup = this.g.select('.cluster-labels');
        labelsGroup.selectAll('text')
            .data(centroids, d => d.topic)
            .join(
                enter => enter.append('text')
                    .attr('x', d => d.x)
                    .attr('y', d => d.y)
                    .attr('text-anchor', 'middle')
                    .attr('dominant-baseline', 'middle')
                    .attr('class', 'cluster-label')
                    .style('font-size', '16px')
                    .style('font-weight', '600')
                    .style('pointer-events', 'none')
                    .style('opacity', 0)
                    .text(d => d.topic)
                    .transition()
                    .duration(500)
                    .style('opacity', 1),
                update => update
                    .transition()
                    .duration(500)
                    .attr('x', d => d.x)
                    .attr('y', d => d.y)
                    .text(d => d.topic),
                exit => exit
                    .transition()
                    .duration(500)
                    .style('opacity', 0)
                    .remove()
            );
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
        console.log('Sending message:', message);
        input.value = '';
    }
}

// Initialize visualization when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    const viz = new PodcastMapNomic('#map-svg-container');
    window.podcastViz = viz;
});
