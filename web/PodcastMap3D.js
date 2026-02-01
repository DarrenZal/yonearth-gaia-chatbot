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
        this.rotationSpeed = 0.05; // Slower rotation
        this.currentClusterCount = 9; // Default to 9 clusters
        this.legendDragInitialized = false; // Track if drag handlers are set up
        this.lastUserClickTime = null; // Track when user last clicked a node

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

        // Populate topic dropdown
        this.populateTopicDropdown();

        // Create 3D visualization
        this.create3DGraph();

        // Set up event listeners
        this.setupEventListeners();

        // Load episode list
        this.loadEpisodeList();

        // Start auto-rotation
        this.startAutoRotation();
    }

    async loadData(clusterCount = 9) {
        try {
            const filename = `/data/processed/podcast_map_3d_umap_multi_cluster.json`;
            const response = await fetch(filename, {
                cache: 'no-store',  // Bypass cache
                headers: {
                    'Cache-Control': 'no-cache',
                    'Pragma': 'no-cache'
                }
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            this.rawData = data; // Store raw data
            this.data = this.transformDataFor3D(data, clusterCount);
            console.log(`Loaded 3D UMAP map data (${clusterCount} clusters):`, this.data);
        } catch (error) {
            console.error('Error loading map data:', error);
            // Use dummy data for testing
            this.data = this.generateDummy3DData();
        }
    }

    transformDataFor3D(data, clusterLevel) {
        // Build dynamic cluster colors from the discovered topics for this level
        const clusterKey = `cluster_${clusterLevel}`;
        const clusterNameKey = `cluster_${clusterLevel}_name`;

        this.clusterColors = {};
        if (data.clusters_by_level && data.clusters_by_level[clusterLevel]) {
            data.clusters_by_level[clusterLevel].forEach(cluster => {
                this.clusterColors[cluster.id] = cluster.color;
            });
        }

        // Transform data to work with 3d-force-graph format with actual UMAP coordinates
        const nodes = data.points.map(point => ({
            id: point.id,
            text: point.text,
            episode_id: point.episode_id,
            episode_title: point.episode_title,
            timestamp: point.timestamp,
            cluster: point[clusterKey],
            cluster_name: point[clusterNameKey],
            color: this.clusterColors[point[clusterKey]] || '#999999',
            val: 3, // Node size
            // Use UMAP-generated 3D coordinates (these NEVER change)
            x: point.x,
            y: point.y,
            z: point.z
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
            clusters: data.clusters_by_level ? data.clusters_by_level[clusterLevel] : []
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
            .nodeLabel(node => `<strong>${node.episode_title}</strong><br>Topic: ${node.cluster_name}<br>${node.text.substring(0, 100)}...`)
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
            .enableNodeDrag(false) // Disable node dragging
            // Disable force simulation - use UMAP coordinates directly
            .d3Force('center', null)
            .d3Force('charge', null)
            .d3Force('link', d3.forceLink().distance(20).strength(0.1)); // Weak links for visual connection only

        // Set initial camera position - adjust based on UMAP coordinate ranges
        this.graph.cameraPosition({ z: 2500 });

        // Add cluster boundary meshes after a short delay to ensure scene is initialized
        setTimeout(() => this.addClusterBoundaries(), 500);

        // Stop rotation on any click in the 3D area
        elem.addEventListener('click', () => {
            this.autoRotate = false;
        });
    }

    addClusterBoundaries() {
        const scene = this.graph.scene();

        // Find a sample mesh node to get THREE constructors from
        const sampleMesh = scene.children.find(child => child.type === 'Mesh' && child.geometry);
        if (!sampleMesh) {
            console.warn('No mesh nodes found in scene, skipping cluster boundaries');
            return;
        }

        // Group nodes by cluster
        const nodesByCluster = {};
        this.data.nodes.forEach(node => {
            if (!nodesByCluster[node.cluster]) {
                nodesByCluster[node.cluster] = [];
            }
            nodesByCluster[node.cluster].push(node);
        });

        // Get constructor references from existing scene objects
        const SphereGeometryConstructor = sampleMesh.geometry.constructor;
        const MeshConstructor = sampleMesh.constructor;
        const MaterialConstructor = sampleMesh.material.constructor;

        Object.keys(nodesByCluster).forEach(clusterId => {
            const nodes = nodesByCluster[clusterId];

            // Skip small clusters
            if (nodes.length < 4) return;

            // Calculate cluster center and extent
            const bounds = this.calculateClusterBounds(nodes);

            // Get cluster color and make it muted/soft
            const color = this.clusterColors[clusterId] || '#888888';
            const mutedColor = this.getMutedColor(color);

            // Create semi-transparent ellipsoid
            const geometry = new SphereGeometryConstructor(1, 16, 12);
            geometry.scale(bounds.scaleX, bounds.scaleY, bounds.scaleZ);

            // Create semi-transparent material
            const material = new MaterialConstructor({
                color: mutedColor,
                transparent: true,
                opacity: 0.12,
                side: 2, // THREE.DoubleSide
                depthWrite: false
            });

            // Create mesh and position at cluster center
            const mesh = new MeshConstructor(geometry, material);
            mesh.position.set(bounds.centerX, bounds.centerY, bounds.centerZ);
            scene.add(mesh);
        });

        console.log(`Added ${Object.keys(nodesByCluster).length} cluster boundary visualizations`);
    }

    calculateClusterBounds(nodes) {
        // Calculate bounding box
        let minX = Infinity, maxX = -Infinity;
        let minY = Infinity, maxY = -Infinity;
        let minZ = Infinity, maxZ = -Infinity;
        let sumX = 0, sumY = 0, sumZ = 0;

        nodes.forEach(node => {
            minX = Math.min(minX, node.x);
            maxX = Math.max(maxX, node.x);
            minY = Math.min(minY, node.y);
            maxY = Math.max(maxY, node.y);
            minZ = Math.min(minZ, node.z);
            maxZ = Math.max(maxZ, node.z);
            sumX += node.x;
            sumY += node.y;
            sumZ += node.z;
        });

        // Calculate center
        const centerX = sumX / nodes.length;
        const centerY = sumY / nodes.length;
        const centerZ = sumZ / nodes.length;

        // Calculate scale with padding (1.3x to create a soft boundary beyond points)
        const scaleX = ((maxX - minX) / 2) * 1.3 || 10;
        const scaleY = ((maxY - minY) / 2) * 1.3 || 10;
        const scaleZ = ((maxZ - minZ) / 2) * 1.3 || 10;

        return { centerX, centerY, centerZ, scaleX, scaleY, scaleZ };
    }

    getMutedColor(hexColor) {
        // Convert hex to RGB
        const r = parseInt(hexColor.slice(1, 3), 16);
        const g = parseInt(hexColor.slice(3, 5), 16);
        const b = parseInt(hexColor.slice(5, 7), 16);

        // Mute by blending with dark background (20% original, 80% background)
        const bgR = 10, bgG = 14, bgB = 26; // Background color #0a0e1a
        const mutedR = Math.round(r * 0.2 + bgR * 0.8);
        const mutedG = Math.round(g * 0.2 + bgG * 0.8);
        const mutedB = Math.round(b * 0.2 + bgB * 0.8);

        // Convert back to hex
        return `#${mutedR.toString(16).padStart(2, '0')}${mutedG.toString(16).padStart(2, '0')}${mutedB.toString(16).padStart(2, '0')}`;
    }

    startAutoRotation() {
        const animate = () => {
            if (this.autoRotate && this.graph) {
                this.rotationAngle += this.rotationSpeed;
                const distance = 2200;
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

        // Store the clicked node as the current active chunk
        // This ensures we highlight the actual clicked node, not the closest one by timestamp
        this.currentActiveChunk = node;

        // Mark that user just clicked a node - prevent sync from overriding for 2 seconds
        this.lastUserClickTime = Date.now();
        console.log('User clicked node - sync blocked for 2 seconds');

        // Play audio at timestamp
        if (node.timestamp !== null && node.timestamp !== undefined) {
            this.playAudioAtTimestamp(node);
        }

        // Highlight episode and the specific clicked node
        this.highlightEpisode(node.episode_id);
        this.highlightActiveNode(node);
    }

    handleNodeHover(node) {
        // Change cursor
        document.body.style.cursor = node ? 'pointer' : 'default';
    }

    highlightEpisode(episodeId) {
        this.selectedEpisode = episodeId;

        // Update node appearance - make selected episode much larger
        this.graph.nodeColor(node => {
            if (node.episode_id === episodeId) {
                return node.color;
            }
            return this.hexToRgba(node.color, 0.2);
        });

        this.graph.nodeVal(node => {
            return node.episode_id === episodeId ? 8 : 2; // Increased from 5 to 8
        });

        // Update link appearance - make selected episode links thicker and brighter
        this.graph.linkOpacity(link => {
            return link.episode_id === episodeId ? 0.8 : 0.05; // Increased from 0.6 to 0.8
        });

        this.graph.linkWidth(link => {
            return link.episode_id === episodeId ? 3 : 0.3; // Increased from 1.5 to 3
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
                // Pause first
                this.audioPlayer.pause();

                // Only change source if it's a different episode
                const needsNewSource = !this.audioPlayer.src.includes(audioUrl);
                if (needsNewSource) {
                    this.audioPlayer.src = audioUrl;
                    await this.audioPlayer.load();
                }

                // Ensure timestamp is valid (some might be 0 or null)
                let timestamp = (startTime !== null && startTime !== undefined && startTime >= 0) ? startTime : 0;

                // Wait for audio to be ready
                if (this.audioPlayer.readyState < 2) {
                    await new Promise(resolve => {
                        this.audioPlayer.addEventListener('loadeddata', resolve, { once: true });
                    });
                }

                // Cap timestamp at episode duration to prevent seeking beyond the end
                // (some episodes have incorrect timestamps due to wrong chunk duration calculations)
                if (timestamp > this.audioPlayer.duration) {
                    console.warn(`Timestamp ${timestamp}s exceeds episode duration ${this.audioPlayer.duration}s - capping at duration`);
                    timestamp = this.audioPlayer.duration;
                }

                // Seek to timestamp and play
                this.audioPlayer.currentTime = timestamp;

                console.log(`Playing Episode ${episodeNumber} at timestamp: ${timestamp}s (${Math.floor(timestamp/60)}:${String(Math.floor(timestamp%60)).padStart(2,'0')})`);

                await this.audioPlayer.play();

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

        // Remove existing listeners to avoid duplicates
        if (this.audioSyncHandler) {
            this.audioPlayer.removeEventListener('timeupdate', this.audioSyncHandler);
            this.audioPlayer.removeEventListener('seeked', this.audioSyncHandler);
            this.audioPlayer.removeEventListener('seeking', this.audioSyncHandler);
        }

        // Create handler for continuous sync
        this.audioSyncHandler = () => {
            if (this.selectedEpisode) {
                this.syncVisualizationWithAudio();
            }
        };

        // Listen to all relevant events:
        // - timeupdate: fires continuously during playback (~every 250ms)
        // - seeking: fires while user is dragging the timeline slider
        // - seeked: fires when user releases the timeline slider
        this.audioPlayer.addEventListener('timeupdate', this.audioSyncHandler);
        this.audioPlayer.addEventListener('seeking', this.audioSyncHandler);
        this.audioPlayer.addEventListener('seeked', this.audioSyncHandler);

        console.log('Audio synchronization enabled - node will update as audio plays and when scrubbing');
    }

    syncVisualizationWithAudio() {
        if (!this.audioPlayer || !this.selectedEpisode) return;

        // Don't override user clicks for 2 seconds
        if (this.lastUserClickTime && (Date.now() - this.lastUserClickTime) < 2000) {
            console.log('Skipping sync - user recently clicked a node');
            return;
        }

        const currentTime = this.audioPlayer.currentTime;
        console.log(`Syncing visualization at ${currentTime.toFixed(2)}s`);

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

        if (activeNode) {
            // Always update, even if same node (user might have seeked)
            const nodeChanged = activeNode.id !== this.currentActiveChunk?.id;
            this.currentActiveChunk = activeNode;
            console.log(`Found active node: ${activeNode.id}, timestamp: ${activeNode.timestamp}, changed: ${nodeChanged}`);
            this.highlightActiveNode(activeNode);

            // Update chunk text in player (only if node changed)
            if (nodeChanged) {
                const chunkText = document.getElementById('current-chunk-text');
                if (chunkText && activeNode.text) {
                    chunkText.textContent = activeNode.text.substring(0, 200) + '...';
                }
            }
        } else {
            console.log(`No active node found for time ${currentTime.toFixed(2)}s`);
        }
    }

    highlightActiveNode(activeNode) {
        if (!activeNode) return;

        console.log(`Highlighting node ${activeNode.id} at position (${activeNode.x}, ${activeNode.y}, ${activeNode.z})`);

        // Make currently playing node VERY prominent with bright green
        this.graph.nodeVal(node => {
            if (node.id === activeNode.id) return 20; // MUCH LARGER - currently playing chunk
            if (node.episode_id === this.selectedEpisode) return 8; // Selected episode nodes
            return 2; // All other nodes
        });

        this.graph.nodeColor(node => {
            if (node.id === activeNode.id) {
                return '#00ff00'; // Bright green for currently playing chunk
            }
            if (node.episode_id === this.selectedEpisode) {
                return node.color; // Keep cluster color for selected episode
            }
            return this.hexToRgba(node.color, 0.2); // Dimmed for other episodes
        });

        // Move camera to follow the active node (keep it in view)
        if (activeNode.x !== undefined && activeNode.y !== undefined && activeNode.z !== undefined) {
            // Get current camera position
            const camera = this.graph.camera();
            const currentPos = camera.position;

            // Calculate distance from camera to active node
            const dx = currentPos.x - activeNode.x;
            const dy = currentPos.y - activeNode.y;
            const dz = currentPos.z - activeNode.z;
            const distance = Math.sqrt(dx*dx + dy*dy + dz*dz);

            // If node is too far away (likely out of view), smoothly move camera
            if (distance > 800) {
                // Calculate new camera position maintaining current distance but centered on node
                const targetDistance = 500;
                const direction = {
                    x: dx / distance,
                    y: dy / distance,
                    z: dz / distance
                };

                const newCameraPos = {
                    x: activeNode.x + direction.x * targetDistance,
                    y: activeNode.y + direction.y * targetDistance,
                    z: activeNode.z + direction.z * targetDistance
                };

                // Smooth camera transition (500ms)
                this.graph.cameraPosition(
                    newCameraPos,
                    activeNode, // Look at the active node
                    500 // Transition duration
                );
            }
        }
    }

    switchToClusterLevel(clusterLevel) {
        // Update cluster assignments and colors WITHOUT changing node positions
        if (!this.rawData) {
            console.error('No raw data available for cluster switching');
            return;
        }

        const clusterKey = `cluster_${clusterLevel}`;
        const clusterNameKey = `cluster_${clusterLevel}_name`;

        // Update cluster colors
        this.clusterColors = {};
        if (this.rawData.clusters_by_level && this.rawData.clusters_by_level[clusterLevel]) {
            this.rawData.clusters_by_level[clusterLevel].forEach(cluster => {
                this.clusterColors[cluster.id] = cluster.color;
            });
        }

        // Update node cluster assignments and colors
        this.data.nodes.forEach(node => {
            const rawPoint = this.rawData.points.find(p => p.id === node.id);
            if (rawPoint) {
                node.cluster = rawPoint[clusterKey];
                node.cluster_name = rawPoint[clusterNameKey];
                node.color = this.clusterColors[node.cluster] || '#999999';
            }
        });

        // Update clusters metadata
        this.data.clusters = this.rawData.clusters_by_level[clusterLevel];

        // Update graph node colors (smoothly, without rebuilding)
        this.graph.nodeColor(node => node.color);

        console.log(`Updated to ${clusterLevel} clusters with smooth color transition`);
    }

    setupEventListeners() {
        // Mobile menu toggle
        const menuToggle = document.getElementById('mobile-menu-toggle');
        const headerControls = document.getElementById('header-controls');

        if (menuToggle) {
            menuToggle.addEventListener('click', () => {
                headerControls.classList.toggle('show');
            });
        }

        // Cluster count slider
        const clusterSlider = document.getElementById('cluster-slider');
        const clusterCountDisplay = document.getElementById('cluster-count');

        clusterSlider.addEventListener('input', (e) => {
            const clusterCount = parseInt(e.target.value);
            clusterCountDisplay.textContent = clusterCount;
            this.currentClusterCount = clusterCount;

            console.log(`Switching to ${clusterCount} clusters...`);

            // Update cluster assignments and colors WITHOUT changing positions
            this.switchToClusterLevel(clusterCount);

            // Update topic dropdown with new clusters
            this.populateTopicDropdown();

            console.log(`Switched to ${clusterCount} clusters successfully`);
        });

        // Topic selector
        document.getElementById('topic-select').addEventListener('change', (e) => {
            const clusterId = e.target.value;
            if (clusterId !== '') {
                this.autoRotate = false;
                this.highlightCluster(parseInt(clusterId));
            } else {
                this.clearSelection();
            }
        });

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

        // Playback speed control
        const playbackSpeedSelect = document.getElementById('playback-speed');
        if (playbackSpeedSelect) {
            playbackSpeedSelect.addEventListener('change', (e) => {
                const speed = parseFloat(e.target.value);
                if (this.audioPlayer) {
                    this.audioPlayer.playbackRate = speed;
                    console.log(`Playback speed set to ${speed}x`);
                }
            });
        }

        // Play from start button
        const playFromStartBtn = document.getElementById('play-from-start');
        if (playFromStartBtn) {
            playFromStartBtn.addEventListener('click', () => {
                if (this.audioPlayer && this.selectedEpisode) {
                    this.audioPlayer.currentTime = 0;
                    this.audioPlayer.play();
                    console.log('Playing from start (0:00)');
                }
            });
        }
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

    populateTopicDropdown() {
        const select = document.getElementById('topic-select');

        // Clear existing options
        select.innerHTML = '<option value="">Select a topic...</option>';

        // Sort clusters by count (descending)
        const sortedClusters = [...this.data.clusters].sort((a, b) => b.count - a.count);

        // Add topic options
        sortedClusters.forEach(cluster => {
            const option = document.createElement('option');
            option.value = cluster.id;
            option.textContent = `${cluster.label || cluster.name || 'Unknown'} (${cluster.count} chunks)`;
            select.appendChild(option);
        });
    }

    highlightCluster(clusterId) {
        this.autoRotate = false;

        // Update node appearance
        this.graph.nodeColor(node => {
            if (node.cluster === clusterId) {
                return node.color;
            }
            return this.hexToRgba(node.color, 0.2);
        });

        this.graph.nodeVal(node => {
            return node.cluster === clusterId ? 5 : 2;
        });

        // Update link appearance
        this.graph.linkOpacity(link => {
            const sourceCluster = this.data.nodes.find(n => n.id === link.source.id || n.id === link.source)?.cluster;
            const targetCluster = this.data.nodes.find(n => n.id === link.target.id || n.id === link.target)?.cluster;
            return (sourceCluster === clusterId && targetCluster === clusterId) ? 0.6 : 0.05;
        });
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
        // Don't resume rotation - user stopped it

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
    // Don't resume auto-rotation - user stopped it
}

// Initialize 3D visualization when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    const viz = new PodcastMap3D('#3d-graph');
    window.podcastViz3D = viz; // Make it globally accessible
});
