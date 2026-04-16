// Gaia Chat Integration for Knowledge Graph
// Enhanced with entity linking (Session 5)

let gaiaChatExpanded = true;
let gaiaPersonality = "factual_guide";
let episodeBookData = null;
let kgEntityNames = [];

// Load episode/book data for linking
async function loadEpisodeBookData() {
    try {
        const response = await fetch("/api/knowledge-graph/episodes-books");
        if (response.ok) {
            episodeBookData = await response.json();
            console.log("KG Chat: Loaded episode/book data");
        }
    } catch (err) {
        console.warn("KG Chat: Could not load episode/book data", err);
    }
}

// Load entity names from KG data
async function loadKGEntityNames() {
    try {
        const response = await fetch("/api/knowledge-graph/data");
        if (response.ok) {
            const data = await response.json();
            kgEntityNames = (data.nodes || [])
                .map(n => n.name)
                .filter(n => n && n.length >= 4)
                .sort((a, b) => b.length - a.length);
            console.log("KG Chat: Loaded", kgEntityNames.length, "entity names");
        }
    } catch (err) {
        console.warn("KG Chat: Could not load entity names", err);
    }
}

// Initialize data on load
loadEpisodeBookData();
loadKGEntityNames();

function escapeHtml(text) {
    return text.replace(/&/g, "&amp;")
               .replace(/</g, "&lt;")
               .replace(/>/g, "&gt;")
               .replace(/"/g, "&quot;")
               .replace(/'/g, "&#039;");
}

function escapeRegex(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, "\\\$&");
}

// Navigate to entity in KG visualization
function navigateToEntity(entityName) {
    console.log("KG Chat: Navigating to entity:", entityName);
    if (window.highlightEntitiesByName) {
        window.highlightEntitiesByName([entityName]);
    } else if (window.searchAndFocus) {
        window.searchAndFocus(entityName);
    }
}

function navigateToEpisode(episodeNum) {
    const entityName = "Episode " + episodeNum;
    console.log("KG Chat: Navigating to episode:", entityName);
    navigateToEntity(entityName);
}

function navigateToBook(bookId) {
    const bookName = episodeBookData?.books?.[bookId]?.name || bookId;
    console.log("KG Chat: Navigating to book:", bookName);
    navigateToEntity(bookName);
}

// Add entity links to response text

// Convert markdown to HTML
function markdownToHtml(text) {
    // Escape HTML first to prevent XSS
    let html = escapeHtml(text);
    
    // Convert line breaks to <br>
    html = html.replace(/\n/g, '<br>');
    
    // Convert bold **text** to <strong>text</strong>
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    
    // Convert numbered lists: "1. " after line break  
    html = html.replace(/(<br>)(\d+)\.\s+/g, '$1<span class="list-number">$2.</span> ');
    
    return html;
}

function addEntityLinks(text) {
    let html = markdownToHtml(text);
    
    // Add episode links
    html = html.replace(/\b(Episode|Ep\.?)\s+(\d+)(?:\s*[:\-]\s*[^.!?,\n<]*)?/gi, function(match, prefix, num) {
        const episodeId = "episode_" + num;
        if (episodeBookData?.episodes?.[episodeId]) {
            return '<span class="kg-entity-link kg-episode-link" onclick="navigateToEpisode(' + num + ')">' + match + '</span>';
        }
        return match;
    });
    
    // Add book links
    var bookPatterns = [
        { pattern: /\bY\s+on\s+Earth(?:[:\s]+Get\s+Smarter[^<.!?]*)?/gi, id: "book_y-on-earth" },
        { pattern: /\bVIRIDITAS(?:[:\s]+THE\s+GREAT\s+HEALING)?/gi, id: "book_veriditas" },
        { pattern: /\bViriditas(?:[:\s]+The\s+Great\s+Healing)?/g, id: "book_veriditas" },
        { pattern: /\bSoil\s+Stewardship\s+Handbook/gi, id: "book_soil-stewardship-handbook" }
    ];
    
    bookPatterns.forEach(function(bp) {
        if (episodeBookData && episodeBookData.books && episodeBookData.books[bp.id]) {
            html = html.replace(bp.pattern, function(match) {
                return '<span class="kg-entity-link kg-book-link" onclick="navigateToBook(\'' + bp.id + '\')">' + match + '</span>';
            });
        }
    });
    
    // Add entity links for KG entities (top 100 only for performance)
    kgEntityNames.slice(0, 100).forEach(function(entityName) {
        if (entityName.length < 5) return;
        try {
            var regex = new RegExp("\\b(" + escapeRegex(entityName) + ")\\b", "gi");
            html = html.replace(regex, function(match) {
                return '<span class="kg-entity-link" onclick="navigateToEntity(\'' + escapeRegex(entityName).replace(/'/g, "\\'") + '\')">' + match + '</span>';
            });
        } catch (e) { console.warn(e); }
    });
    
    return html;
}

// Create citations section
function createCitationsHtml(citations) {
    if (!citations || citations.length === 0) return "";
    
    var html = '<div class="kg-citations">';
    html += '<div class="kg-citations-header">References:</div>';
    
    citations.forEach(function(c) {
        var isBook = c.episode_number && c.episode_number.toString().startsWith("Book:");
        if (isBook) {
            var bookTitle = c.episode_number.substring(5).trim();
            var bookId = "book_unknown";
            if (bookTitle.toLowerCase().indexOf("viriditas") >= 0) bookId = "book_veriditas";
            else if (bookTitle.toLowerCase().indexOf("soil") >= 0) bookId = "book_soil-stewardship-handbook";
            else if (bookTitle.toLowerCase().indexOf("earth") >= 0) bookId = "book_y-on-earth";
            
            html += '<div class="kg-citation-item" onclick="navigateToBook(\'' + bookId + '\')">';
            html += '📖 ' + bookTitle + '</div>';
        } else {
            var episodeNum = c.episode_number || c.episode_id;
            var title = c.title || ("Episode " + episodeNum);
            html += '<div class="kg-citation-item" onclick="navigateToEpisode(' + episodeNum + ')">';
            html += '🎙️ ' + title + '</div>';
        }
    });
    
    html += '</div>';
    return html;
}

function toggleGaiaChat() {
    gaiaChatExpanded = !gaiaChatExpanded;
    document.getElementById("gaia-chat-container").style.display = gaiaChatExpanded ? "block" : "none";
    document.getElementById("gaia-chat-toggle").textContent = gaiaChatExpanded ? "▼" : "▶";
}

function toggleGaiaPersonality() {
    var btn = document.getElementById("gaia-personality-toggle");
    if (gaiaPersonality === "factual_guide") {
        gaiaPersonality = "warm_mother";
        btn.textContent = "🌱 Warm";
        btn.title = "Currently: Warm/spiritual tone. Click for factual/neutral tone.";
    } else {
        gaiaPersonality = "factual_guide";
        btn.textContent = "📊 Factual";
        btn.title = "Currently: Factual/neutral tone. Click for warm/spiritual tone.";
    }
}

async function sendGaiaMessage() {
    var input = document.getElementById("gaia-chat-input");
    var messagesDiv = document.getElementById("gaia-chat-messages");
    var message = input.value.trim();
    if (!message) return;

    // Add user message
    messagesDiv.innerHTML += '<div class="kg-chat-user"><strong>You:</strong> ' + escapeHtml(message) + '</div>';
    input.value = "";
    messagesDiv.scrollTop = messagesDiv.scrollHeight;

    // Show loading
    var loadingId = "loading-" + Date.now();
    var loadingEmoji = gaiaPersonality === "factual_guide" ? "📊" : "🌱";
    messagesDiv.innerHTML += '<div id="' + loadingId + '" class="kg-chat-loading">' + loadingEmoji + ' Thinking...</div>';
    messagesDiv.scrollTop = messagesDiv.scrollHeight;

    try {
        var response = await fetch("/api/bm25/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                message: message,
                gaia_personality: gaiaPersonality
            })
        });
        var data = await response.json();

        // Remove loading
        var loadingEl = document.getElementById(loadingId);
        if (loadingEl) loadingEl.remove();

        // Add Gaia response with entity links
        var label = gaiaPersonality === "factual_guide" ? "Guide" : "Gaia";
        var responseClass = gaiaPersonality === "factual_guide" ? "kg-chat-guide" : "kg-chat-gaia";
        
        var responseHtml = '<div class="' + responseClass + '">';
        responseHtml += '<strong>' + label + ':</strong> ';
        responseHtml += addEntityLinks(data.response);
        responseHtml += createCitationsHtml(data.citations);
        responseHtml += '</div>';
        
        messagesDiv.innerHTML += responseHtml;
        messagesDiv.scrollTop = messagesDiv.scrollHeight;

    } catch (err) {
        var loadingEl = document.getElementById(loadingId);
        if (loadingEl) loadingEl.remove();
        messagesDiv.innerHTML += '<div class="kg-chat-error">Error: Could not reach Gaia. Please try again.</div>';
    }
}
