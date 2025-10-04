/**
 * Podcast Map 3D Visualization
 * Interactive 3D force-directed graph with auto-rotation
 */

class PodcastMap3D {
    constructor(containerId) {
        this.containerId = containerId;
        this.data = null;
        this.selectedEpisode = null;
        this.currentActiveChunk = null;
        this.audioPlayer = document.getElementById('audio-element');
        this.graph = null;
        this.autoRotate = true;
        this.rotationAngle = 0;
        this.rotationSpeed = 0.2;

        // Cluster colors matching 2D version
        this.clusterColors = {
            0: '#4CAF50',  // Community - Green
            1: '#9C27B0',  // Culture - Purple
            2: '#FF9800',  // Economy - Orange
            3: '#2196F3',  // Ecology - Blue
            4: '#F44336'   // Health - Red
        };

        // Initialize
        this.init();
    }

    async init() {
        // Load data from backend
        await this.loadData();

        // Create 3D visualization
        this.create3DGraph();

        // Set up event listeners
        this.setupEventListeners();

        // Load episode list
        this.loadEpisodeList();

        // Start auto-rotation
        this.startAutoRotation();
    }

    async loadData() {
        try {
            const response = await fetch('/api/map_data');
            const data = await response.json();
            this.data = this.transformDataFor3D(data);
            console.log('Loaded 3D map data:', this.data);
        } catch (error) {
            console.error('Error loading map data:', error);
            // Use dummy data for testing
            this.data = this.generateDummy3DData();
        }
    }

    transformDataFor3D(data) {
        // Transform data to work with 3d-force-graph format
        const nodes = data.points.map(point => ({
            id: point.id,
            text: point.text,
            episode_id: point.episode_id,
            episode_title: point.episode_title,
            timestamp: point.timestamp,
            cluster: point.cluster,
            cluster_name: point.cluster_name,
            color: this.clusterColors[point.cluster] || '#999999',
            val: 3 // Node size
        }));

        // Create links between sequential chunks in same episode
        const links = [];
        const episodeMap = new Map();

        // Group nodes by episode
        nodes.forEach(node => {
            if (!episodeMap.has(node.episode_id)) {
                episodeMap.set(node.episode_id, []);
            }
            episodeMap.get(node.episode_id).push(node);
        });

        // Create links between sequential nodes
        episodeMap.forEach((episodeNodes, episodeId) => {
            // Sort by timestamp
            episodeNodes.sort((a, b) => (a.timestamp || 0) - (b.timestamp || 0));

            // Create links
            for (let i = 0; i < episodeNodes.length - 1; i++) {
                links.push({
                    source: episodeNodes[i].id,
                    target: episodeNodes[i + 1].id,
                    episode_id: episodeId,
                    color: episodeNodes[i].color,
                    opacity: 0.2
                });
            }
        });

        return {
            nodes: nodes,
            links: links,
            episodes: data.episodes,
            clusters: data.clusters
        };
    }

    generateDummy3DData() {
        // Generate dummy data for testing
        const nodes = [];
        const links = [];
        const numNodes = 500;
        const numEpisodes = 20;
        const clusters = 5;

        for (let i = 0; i < numNodes; i++) {
            const cluster = Math.floor(Math.random() * clusters);
            const episodeId = `Episode ${Math.floor(i / (numNodes / numEpisodes))}`;

            nodes.push({
                id: `node-${i}`,
                text: `This is chunk ${i} discussing various topics...`,
                episode_id: episodeId,
                episode_title: episodeId,
                timestamp: (i % 30) * 120,
                cluster: cluster,
                cluster_name: this.getClusterName(cluster),
                color: this.clusterColors[cluster],
                val: 3
            });
        }

        // Create links
        for (let i = 0; i < numNodes - 1; i++) {
            if (nodes[i].episode_id === nodes[i + 1].episode_id) {
                links.push({
                    source: nodes[i].id,
                    target: nodes[i + 1].id,
                    episode_id: nodes[i].episode_id,
                    color: nodes[i].color,
                    opacity: 0.2
                });
            }
        }

        return {
            nodes: nodes,
            links: links,
            episodes: this.extractEpisodesFromNodes(nodes),
            clusters: Object.keys(this.clusterColors).map(id => ({
                id: parseInt(id),
                name: this.getClusterName(parseInt(id)),
                color: this.clusterColors[id]
            }))
        };
    }

    getClusterName(clusterId) {
        const names = ["Community", "Culture", "Economy", "Ecology", "Health"];
        return names[clusterId % names.length];
    }

    extractEpisodesFromNodes(nodes) {
        const episodeMap = new Map();
        nodes.forEach(node => {
            if (!episodeMap.has(node.episode_id)) {
                episodeMap.set(node.episode_id, {
                    id: node.episode_id,
                    title: node.episode_title,
                    chunk_count: 0
                });
            }
            episodeMap.get(node.episode_id).chunk_count++;
        });
        return Array.from(episodeMap.values());
    }

    create3DGraph() {
        const elem = document.getElementById(this.containerId.replace('#', ''));

        this.graph = ForceGraph3D()(elem)
            .graphData(this.data)
            .nodeLabel(node => `${node.episode_title}<br>${node.text.substring(0, 100)}...`)
            .nodeColor(node => node.color)
            .nodeOpacity(0.8)
            .nodeResolution(16)
            .linkColor(link => link.color)
            .linkOpacity(link => link.opacity)
            .linkWidth(0.5)
            .backgroundColor('#0a0e1a')
            .showNavInfo(false)
            .onNodeClick(node => this.handleNodeClick(node))
            .onNodeHover(node => this.handleNodeHover(node))
            .d3Force('charge', d3.forceManyBody().strength(-30))
            .d3Force('link', d3.forceLink().distance(20));

        // Set initial camera position
        this.graph.cameraPosition({ z: 1500 });
    }

    startAutoRotation() {
        const animate = () => {
            if (this.autoRotate && this.graph) {
                this.rotationAngle += this.rotationSpeed;
                const distance = 1500;
                const x = distance * Math.sin(this.rotationAngle * Math.PI / 180);
                const z = distance * Math.cos(this.rotationAngle * Math.PI / 180);

                this.graph.cameraPosition(
                    { x: x, y: 0, z: z },
                    { x: 0, y: 0, z: 0 },
                    0 // No animation for smooth continuous rotation
                );
            }
            requestAnimationFrame(animate);
        };
        animate();
    }

    handleNodeClick(node) {
        console.log('Clicked node:', node);

        // Stop auto-rotation when user interacts
        this.autoRotate = false;

        // Play audio at timestamp
        if (node.timestamp !== null && node.timestamp !== undefined) {
            this.playAudioAtTimestamp(node);
        }

        // Highlight episode
        this.highlightEpisode(node.episode_id);
    }

    handleNodeHover(node) {
        // Change cursor
        document.body.style.cursor = node ? 'pointer' : 'default';
    }

    highlightEpisode(episodeId) {
        this.selectedEpisode = episodeId;

        // Update node appearance
        this.graph.nodeColor(node => {
            if (node.episode_id === episodeId) {
                return node.color;
            }
            return this.hexToRgba(node.color, 0.2);
        });

        this.graph.nodeVal(node => {
            return node.episode_id === episodeId ? 5 : 2;
        });

        // Update link appearance
        this.graph.linkOpacity(link => {
            return link.episode_id === episodeId ? 0.6 : 0.05;
        });

        this.graph.linkWidth(link => {
            return link.episode_id === episodeId ? 1.5 : 0.3;
        });
    }

    hexToRgba(hex, alpha) {
        const r = parseInt(hex.slice(1, 3), 16);
        const g = parseInt(hex.slice(3, 5), 16);
        const b = parseInt(hex.slice(5, 7), 16);
        return `rgba(${r}, ${g}, ${b}, ${alpha})`;
    }

    playAudioAtTimestamp(node) {
        // Show audio player
        const playerContainer = document.getElementById('audio-player');
        playerContainer.style.display = 'block';

        // Update episode info
        document.getElementById('current-episode-title').textContent = node.episode_title;
        document.getElementById('current-chunk-text').textContent = node.text.substring(0, 200) + '...';

        // Load audio for this episode
        this.loadAudioForEpisode(node.episode_id, node.timestamp);
    }

    async loadAudioForEpisode(episodeId, startTime = 0) {
        const titleElement = document.getElementById('current-episode-title');
        const chunkText = document.getElementById('current-chunk-text');

        // Fetch the actual audio URL from the transcript file
        try {
            // Episode ID might be just a number or "Episode X" format
            let episodeNumber = episodeId;
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
                this.audioPlayer.currentTime = startTime || 0;
                await this.audioPlayer.play();
                console.log('Playing audio at timestamp:', startTime);

                if (chunkText) {
                    chunkText.textContent = 'Playing...';
                }

                // Set up time sync for highlighting
                this.setupAudioSync();
            }
        } catch (error) {
            console.error('Error loading audio URL:', error);
            if (chunkText) {
                chunkText.textContent = 'Audio not available for this episode.';
            }
        }
    }

    setupAudioSync() {
        if (!this.audioPlayer) return;

        // Remove existing listener to avoid duplicates
        this.audioPlayer.removeEventListener('timeupdate', this.audioSyncHandler);

        // Create handler
        this.audioSyncHandler = () => {
            if (this.selectedEpisode) {
                this.syncVisualizationWithAudio();
            }
        };

        this.audioPlayer.addEventListener('timeupdate', this.audioSyncHandler);
    }

    syncVisualizationWithAudio() {
        if (!this.audioPlayer || !this.selectedEpisode) return;

        const currentTime = this.audioPlayer.currentTime;

        // Find the chunk closest to current time
        const episodeNodes = this.data.nodes.filter(n => n.episode_id === this.selectedEpisode);

        let activeNode = null;
        let minDiff = Infinity;

        for (const node of episodeNodes) {
            if (node.timestamp === null || node.timestamp === undefined) continue;

            const diff = Math.abs(node.timestamp - currentTime);
            if (diff < minDiff) {
                minDiff = diff;
                activeNode = node;
            }
        }

        if (activeNode && activeNode.id !== this.currentActiveChunk?.id) {
            this.currentActiveChunk = activeNode;
            this.highlightActiveNode(activeNode);

            // Update chunk text in player
            const chunkText = document.getElementById('current-chunk-text');
            if (chunkText && activeNode.text) {
                chunkText.textContent = activeNode.text.substring(0, 200) + '...';
            }
        }
    }

    highlightActiveNode(activeNode) {
        if (!activeNode) return;

        // Make active node extra prominent
        this.graph.nodeVal(node => {
            if (node.id === activeNode.id) return 10;
            if (node.episode_id === this.selectedEpisode) return 4;
            return 2;
        });

        this.graph.nodeColor(node => {
            if (node.id === activeNode.id) {
                return '#ffffff'; // Active node is white
            }
            if (node.episode_id === this.selectedEpisode) {
                return node.color;
            }
            return this.hexToRgba(node.color, 0.2);
        });
    }

    setupEventListeners() {
        // Episode selector
        document.getElementById('episode-select').addEventListener('change', (e) => {
            const episodeId = e.target.value;
            if (episodeId) {
                this.autoRotate = false; // Stop rotation when episode selected
                this.highlightEpisode(episodeId);
                this.loadAudioForEpisode(episodeId);

                // Show audio player
                document.getElementById('audio-player').style.display = 'block';

                // Focus camera on selected episode nodes
                this.focusOnEpisode(episodeId);
            } else {
                this.clearSelection();
            }
        });

        // Resume auto-rotation when clicking on background
        window.addEventListener('click', (e) => {
            if (e.target.id === '3d-graph' || e.target.tagName === 'CANVAS') {
                // Only if not clicking on a node
                if (!e.target.closest('.force-graph-container')) {
                    this.autoRotate = true;
                }
            }
        });
    }

    focusOnEpisode(episodeId) {
        const episodeNodes = this.data.nodes.filter(n => n.episode_id === episodeId);

        if (episodeNodes.length > 0) {
            // Calculate center of episode nodes
            const centerX = episodeNodes.reduce((sum, n) => sum + (n.x || 0), 0) / episodeNodes.length;
            const centerY = episodeNodes.reduce((sum, n) => sum + (n.y || 0), 0) / episodeNodes.length;
            const centerZ = episodeNodes.reduce((sum, n) => sum + (n.z || 0), 0) / episodeNodes.length;

            // Move camera to focus on episode
            const distance = 500;
            this.graph.cameraPosition(
                { x: centerX, y: centerY, z: centerZ + distance },
                { x: centerX, y: centerY, z: centerZ },
                1000 // 1 second transition
            );
        }
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
        this.autoRotate = true; // Resume rotation

        // Reset all nodes
        this.graph.nodeColor(node => node.color);
        this.graph.nodeVal(3);
        this.graph.linkOpacity(0.2);
        this.graph.linkWidth(0.5);

        // Hide audio player
        document.getElementById('audio-player').style.display = 'none';
        if (this.audioPlayer) {
            this.audioPlayer.pause();
        }
    }
}

// Utility function
function closeAudioPlayer() {
    document.getElementById('audio-player').style.display = 'none';
    const audio = document.getElementById('audio-element');
    if (audio) {
        audio.pause();
    }

    // Resume auto-rotation
    if (window.podcastViz3D) {
        window.podcastViz3D.autoRotate = true;
    }
}

// Initialize 3D visualization when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    const viz = new PodcastMap3D('#3d-graph');
    window.podcastViz3D = viz; // Make it globally accessible
});
