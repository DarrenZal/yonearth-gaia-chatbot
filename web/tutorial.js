/**
 * Tutorial/Guided Tour System
 * Lightweight onboarding for new users across YonEarth pages
 */

const Tutorial = {
    steps: [],
    currentStep: 0,
    pageName: '',
    elements: {
        overlay: null,
        highlight: null,
        tooltip: null,
        helpBtn: null
    },

    // Tutorial content for each page
    pageSteps: {
        // Main chat page
        chat: [
            {
                title: "Welcome to Gaia!",
                text: "I'm here to help you explore wisdom from the YonEarth Community Podcast. Let me show you around!",
                icon: "🌍",
                target: null, // Welcome message, no target
                position: 'center'
            },
            {
                title: "Ask Your Question",
                text: "Type any question about regenerative practices, sustainability, community building, or earth-healing topics. I'll search through 180+ podcast episodes to find relevant insights.",
                icon: "💬",
                target: '#messageInput',
                position: 'top'
            },
            {
                title: "Choose a Personality",
                text: "I can respond in different styles: as a Nurturing Mother, an Ancient Sage, an Earth Guardian, or as a Factual Guide if you prefer straightforward information.",
                icon: "🎭",
                target: '#personality',
                position: 'bottom'
            },
            {
                title: "Enable Voice Responses",
                text: "Want to hear my responses? Enable voice mode and I'll speak to you directly!",
                icon: "🔊",
                target: '#voiceToggle',
                position: 'bottom'
            },
            {
                title: "Explore Episode Links",
                text: "My responses include links to specific podcast episodes. Click them to learn more from the original conversations!",
                icon: "🎧",
                target: '.recommendations',
                position: 'top',
                fallbackText: "After I respond, you'll see links to relevant podcast episodes. Click them to dive deeper!"
            }
        ],

        // Knowledge Graph page
        kg: [
            {
                title: "Welcome to the Knowledge Graph",
                text: "Explore the connections between topics, people, and organizations from the YonEarth podcast network.",
                icon: "🕸️",
                target: null,
                position: 'center'
            },
            {
                title: "Search for Topics",
                text: "Looking for something specific? Search for any topic, person, or organization mentioned in the podcasts.",
                icon: "🔍",
                target: '#search-input',
                position: 'bottom'
            },
            {
                title: "Filter by Category",
                text: "Use these filters to focus on specific types of content: people, organizations, concepts, and more.",
                icon: "🏷️",
                target: '#entity-type-filters',
                position: 'right'
            },
            {
                title: "Click Any Node",
                text: "Click on any circle to see details about that topic or person, including which episodes they appear in.",
                icon: "👆",
                target: '#graph-svg-container',
                position: 'left'
            },
            {
                title: "Navigate the Graph",
                text: "Drag to pan around, scroll to zoom in/out, and drag nodes to rearrange them. Explore the connections!",
                icon: "🖱️",
                target: null,
                position: 'center'
            }
        ],

        // 3D Podcast Map page
        podcastmap: [
            {
                title: "Welcome to the 3D Podcast Map",
                text: "Explore YonEarth podcast episodes in an immersive 3D space where similar topics cluster together.",
                icon: "🌐",
                target: null,
                position: 'center'
            },
            {
                title: "Navigate in 3D",
                text: "Drag to rotate the view, scroll to zoom, and explore the landscape of podcast topics.",
                icon: "🖱️",
                target: '#3d-graph',
                position: 'left'
            },
            {
                title: "Topic Clusters",
                text: "Episodes covering similar themes are grouped together. Notice how related topics form clusters in the space!",
                icon: "🎯",
                target: '#cluster-slider',
                position: 'bottom'
            },
            {
                title: "Select an Episode",
                text: "Use the dropdown to jump directly to a specific episode in the visualization.",
                icon: "🎬",
                target: '#episode-select',
                position: 'bottom'
            },
            {
                title: "Play Episode Clips",
                text: "Click any point in the visualization to hear that moment from the podcast. Explore conversations that interest you!",
                icon: "▶️",
                target: null,
                position: 'center'
            }
        ]
    },

    /**
     * Initialize the tutorial for a specific page
     * @param {string} pageName - 'chat', 'kg', or 'podcastmap'
     */
    init(pageName) {
        this.pageName = pageName;

        // Check if tutorial already completed for this page
        if (this.isCompleted(pageName)) {
            this.createHelpButton();
            return;
        }

        // Load steps for this page
        this.loadSteps(pageName);

        // Create DOM elements
        this.createElements();

        // Start the tutorial
        this.show();
    },

    /**
     * Check if tutorial was already completed
     */
    isCompleted(pageName) {
        return localStorage.getItem(`tutorial_${pageName}_complete`) === 'true';
    },

    /**
     * Load tutorial steps for the specified page
     */
    loadSteps(pageName) {
        this.steps = this.pageSteps[pageName] || [];
        this.currentStep = 0;
    },

    /**
     * Create necessary DOM elements
     */
    createElements() {
        // Remove any existing tutorial elements
        this.cleanup();

        // Create overlay
        this.elements.overlay = document.createElement('div');
        this.elements.overlay.className = 'tutorial-overlay';
        document.body.appendChild(this.elements.overlay);

        // Create highlight element
        this.elements.highlight = document.createElement('div');
        this.elements.highlight.className = 'tutorial-highlight';
        this.elements.highlight.style.display = 'none';
        document.body.appendChild(this.elements.highlight);

        // Create tooltip
        this.elements.tooltip = document.createElement('div');
        this.elements.tooltip.className = 'tutorial-tooltip';
        document.body.appendChild(this.elements.tooltip);

        // Create help button
        this.createHelpButton();
    },

    /**
     * Create the help button to restart tutorial
     */
    createHelpButton() {
        // Remove existing help button if any
        const existing = document.querySelector('.tutorial-help-btn');
        if (existing) existing.remove();

        this.elements.helpBtn = document.createElement('button');
        this.elements.helpBtn.className = 'tutorial-help-btn';
        this.elements.helpBtn.innerHTML = '?<span class="tooltip">Show Tutorial</span>';
        this.elements.helpBtn.onclick = () => this.restart();
        document.body.appendChild(this.elements.helpBtn);
    },

    /**
     * Show the current step
     */
    show() {
        if (this.currentStep >= this.steps.length) {
            this.complete(this.pageName);
            return;
        }

        const step = this.steps[this.currentStep];

        // Show overlay
        this.elements.overlay.classList.add('visible');

        // Position highlight and tooltip
        if (step.target) {
            const targetEl = document.querySelector(step.target);
            if (targetEl) {
                this.highlightElement(targetEl);
                this.positionTooltip(targetEl, step);
            } else if (step.fallbackText) {
                // Target not found, show centered with fallback text
                this.showCenteredTooltip({...step, text: step.fallbackText});
            } else {
                this.showCenteredTooltip(step);
            }
        } else {
            // No target, show centered
            this.showCenteredTooltip(step);
        }
    },

    /**
     * Highlight a target element
     */
    highlightElement(el) {
        const rect = el.getBoundingClientRect();
        const padding = 8;

        this.elements.highlight.style.display = 'block';
        this.elements.highlight.style.top = (rect.top - padding + window.scrollY) + 'px';
        this.elements.highlight.style.left = (rect.left - padding) + 'px';
        this.elements.highlight.style.width = (rect.width + padding * 2) + 'px';
        this.elements.highlight.style.height = (rect.height + padding * 2) + 'px';
    },

    /**
     * Position the tooltip relative to target
     */
    positionTooltip(targetEl, step) {
        const rect = targetEl.getBoundingClientRect();
        const tooltip = this.elements.tooltip;

        // Build tooltip content
        tooltip.innerHTML = this.buildTooltipContent(step);
        tooltip.className = 'tutorial-tooltip';

        // Calculate position based on step.position
        const tooltipRect = tooltip.getBoundingClientRect();
        const margin = 20;

        let top, left;
        let arrowClass = '';

        switch (step.position) {
            case 'top':
                top = rect.top - tooltipRect.height - margin + window.scrollY;
                left = rect.left + (rect.width / 2) - (tooltipRect.width / 2);
                arrowClass = 'arrow-bottom';
                break;
            case 'bottom':
                top = rect.bottom + margin + window.scrollY;
                left = rect.left + (rect.width / 2) - (tooltipRect.width / 2);
                arrowClass = 'arrow-top';
                break;
            case 'left':
                top = rect.top + (rect.height / 2) - (tooltipRect.height / 2) + window.scrollY;
                left = rect.left - tooltipRect.width - margin;
                arrowClass = 'arrow-right';
                break;
            case 'right':
                top = rect.top + (rect.height / 2) - (tooltipRect.height / 2) + window.scrollY;
                left = rect.right + margin;
                arrowClass = 'arrow-left';
                break;
            default:
                top = rect.bottom + margin + window.scrollY;
                left = rect.left + (rect.width / 2) - (tooltipRect.width / 2);
                arrowClass = 'arrow-top';
        }

        // Keep tooltip within viewport
        left = Math.max(10, Math.min(left, window.innerWidth - tooltipRect.width - 10));
        top = Math.max(10, top);

        tooltip.style.top = top + 'px';
        tooltip.style.left = left + 'px';
        tooltip.classList.add(arrowClass);

        // Show tooltip with animation
        requestAnimationFrame(() => {
            tooltip.classList.add('visible');
        });
    },

    /**
     * Show tooltip centered (for welcome messages)
     */
    showCenteredTooltip(step) {
        this.elements.highlight.style.display = 'none';

        const tooltip = this.elements.tooltip;
        tooltip.innerHTML = this.buildTooltipContent(step);
        tooltip.className = 'tutorial-tooltip tutorial-welcome';
        tooltip.style.top = '50%';
        tooltip.style.left = '50%';
        tooltip.style.transform = 'translate(-50%, -50%)';

        requestAnimationFrame(() => {
            tooltip.classList.add('visible');
        });
    },

    /**
     * Build tooltip HTML content
     */
    buildTooltipContent(step) {
        const isLastStep = this.currentStep === this.steps.length - 1;

        // Build progress dots
        let progressDots = '<div class="tutorial-progress">';
        for (let i = 0; i < this.steps.length; i++) {
            let dotClass = 'tutorial-dot';
            if (i < this.currentStep) dotClass += ' completed';
            if (i === this.currentStep) dotClass += ' active';
            progressDots += `<span class="${dotClass}"></span>`;
        }
        progressDots += '</div>';

        return `
            <h3 class="tutorial-title">
                <span class="step-icon">${step.icon}</span>
                ${step.title}
            </h3>
            <p class="tutorial-text">${step.text}</p>
            ${progressDots}
            <div class="tutorial-buttons">
                <button class="tutorial-btn tutorial-btn-skip" onclick="Tutorial.skip()">
                    Skip Tour
                </button>
                <button class="tutorial-btn ${isLastStep ? 'tutorial-btn-finish' : 'tutorial-btn-next'}" onclick="Tutorial.next()">
                    ${isLastStep ? 'Get Started!' : 'Next'}
                </button>
            </div>
        `;
    },

    /**
     * Move to the next step
     */
    next() {
        this.elements.tooltip.classList.remove('visible');

        setTimeout(() => {
            this.currentStep++;
            this.show();
        }, 200);
    },

    /**
     * Skip the entire tutorial
     */
    skip() {
        this.complete(this.pageName);
    },

    /**
     * Complete and hide the tutorial
     */
    complete(pageName) {
        localStorage.setItem(`tutorial_${pageName}_complete`, 'true');
        this.hide();
    },

    /**
     * Hide tutorial elements
     */
    hide() {
        if (this.elements.overlay) {
            this.elements.overlay.classList.remove('visible');
        }
        if (this.elements.tooltip) {
            this.elements.tooltip.classList.remove('visible');
        }
        if (this.elements.highlight) {
            this.elements.highlight.style.display = 'none';
        }

        // Remove elements after animation
        setTimeout(() => {
            if (this.elements.overlay) this.elements.overlay.remove();
            if (this.elements.tooltip) this.elements.tooltip.remove();
            if (this.elements.highlight) this.elements.highlight.remove();
            this.elements = { overlay: null, highlight: null, tooltip: null, helpBtn: this.elements.helpBtn };
        }, 300);
    },

    /**
     * Restart the tutorial for the current page
     */
    restart() {
        localStorage.removeItem(`tutorial_${this.pageName}_complete`);
        this.loadSteps(this.pageName);
        this.createElements();
        this.show();
    },

    /**
     * Clean up any existing tutorial elements
     */
    cleanup() {
        document.querySelectorAll('.tutorial-overlay, .tutorial-tooltip, .tutorial-highlight').forEach(el => el.remove());
    },

    /**
     * Reset all tutorials (useful for testing)
     */
    resetAll() {
        localStorage.removeItem('tutorial_chat_complete');
        localStorage.removeItem('tutorial_kg_complete');
        localStorage.removeItem('tutorial_podcastmap_complete');
        console.log('All tutorials reset');
    }
};

// Make Tutorial available globally
window.Tutorial = Tutorial;
