# Remaining TODOs for YonEarth Gaia Chatbot

Based on Aaron's feedback and our discoveries, here are the critical remaining tasks:

## 🚨 Critical Priority (Do First)

### 1. ✅ Add More Books to Dataset [COMPLETED 2025-07-17]
**Impact**: Aaron said responses should "almost always" reference both podcasts AND books

- [x] ~~Add "Why on Earth" book by Aaron William Perry~~ (Added "Y on Earth" - 2,124 chunks)
- [x] Add "Soil Stewardship Handbook" (Added - 136 chunks)
- [x] Ensure book integration works seamlessly with episodes
- [x] Update prompts to prioritize book+episode citations together
- [x] Add multi-format book links (eBook, audiobook, print)
- [x] Fix "Referenced Episodes" to show "References" when books included

**Completed Implementation**:
- Added 2 new books to reach 3 total books
- 4,289+ book chunks now searchable alongside 14,475+ episode chunks
- Books properly display with chapter references and clickable links

### 2. Fix Recommended Content Section
**Impact**: Currently shows mismatched episodes (e.g., Episode 165 referenced but not in recommendations)

- [ ] Ensure ALL referenced episodes appear in recommendations
- [ ] Sync citations with recommended content
- [ ] Fix the episode diversity issue (all 4 biochar episodes should show)

## 🔧 High Priority

### 3. Add Hyperlinks in Responses
**Quote**: "We want to provide navigation guidance into the YonEarth ecosystem"

- [ ] Make episode references clickable
- [ ] Link to yonearth.org/podcast/episode-XXX pages
- [ ] Keep all links within YonEarth ecosystem
- [ ] Format: [Episode 120](https://yonearth.org/podcast/episode-120...)

**Implementation Areas**:
- `src/character/gaia.py` - Update citation formatting
- `web/chat.js` - Render markdown links properly

### 4. ✅ Add Feedback Component [COMPLETED 2025-07-17]
**Impact**: Critical for iterative improvement during Aaron's testing

- [x] Add thumbs up/down buttons per response
- [x] Capture feedback with response ID
- [x] Store in JSON files organized by date
- [x] Show feedback stats with view_feedback.py script

**Completed Implementation**:
- Quick feedback: Thumbs up/down buttons
- Detailed feedback: 5-star rating, episode correctness, text comments
- Frontend integration in chat.js with localStorage backup
- Backend endpoint in both main.py and simple_server.py
- Analysis script at scripts/view_feedback.py

### 5. ✅ Implement Semantic Category Matching [COMPLETED 2025-07-17]
**Impact**: Solves "soil" → BIOCHAR matching issue

- [x] Complete implementation from SEMANTIC_CATEGORY_IMPLEMENTATION_PLAN.md
- [x] Use OpenAI embeddings for category descriptions  
- [x] Add configurable thresholds in UI (Broad/Normal/Strict/Disabled)
- [x] Implement diverse episode search to ensure all relevant episodes appear
- [x] Cache embeddings to avoid repeated API calls

**Completed Implementation**:
- True semantic matching: "teach me about soil" → BIOCHAR (32.1% similarity)
- Episode diversity: All biochar episodes (120, 122, 124, 165) now discoverable
- UI controls: Category threshold selector with 4 levels
- Performance: Cached embeddings in `/data/processed/category_embeddings.json`

### 6. Add Configurable Search Weights UI
**Current**: Fixed at 60% category, 25% semantic, 15% keyword

- [ ] Add sliders for category/semantic/keyword weights
- [ ] Total must equal 100%
- [ ] Show current weights in UI
- [ ] Save preferences in localStorage

## 📊 Medium Priority

### 7. Add Voice Capability
**Quote**: Aaron was "particularly excited about this for accessibility"

- [ ] Integrate 11 Labs API
- [ ] Options: Aaron's cloned voice, "Mama Gaia" voice, generic
- [ ] Add play button next to responses
- [ ] Consider text-to-speech for accessibility

**Implementation**:
```javascript
// Add to web/chat.js
const playVoice = async (text) => {
    const response = await fetch('/api/voice', {
        method: 'POST',
        body: JSON.stringify({ text, voice: 'mama_gaia' })
    });
    // Play audio response
};
```

### 8. Add Cost Calculator
**Impact**: Important for planning public rollout

- [ ] Track OpenAI API costs per query
- [ ] Show cost breakdown (embeddings, completions)
- [ ] Add daily/monthly cost reports
- [ ] Estimate costs for different usage levels

## ✅ Completed

- [x] Max References Configuration (1-10 references per response)
- [x] Episode Categorization Investigation
- [x] Documentation of search weights
- [x] Add More Books to Dataset (Y on Earth & Soil Stewardship Handbook) - 2025-07-17
- [x] Multi-format book links (eBook, audiobook, print)
- [x] Fix "Referenced Episodes" label for book references
- [x] Add Feedback Component for quality improvement - 2025-07-17
- [x] Implement Semantic Category Matching with configurable thresholds - 2025-07-17

## 📝 Quick Wins Order

1. ~~**Add the missing books**~~ ✅ COMPLETED
2. **Fix recommended content** to match citations
3. **Add hyperlinks** to episode references  
4. ~~**Add feedback buttons** for testing phase~~ ✅ COMPLETED

## 🚀 Implementation Timeline

### Week 1 (Quick Wins)
- Add books to dataset
- Fix recommended content matching
- Add hyperlinks to responses

### Week 2 (Core Features)
- Implement semantic category matching
- Add configurable weights UI
- Add feedback component

### Week 3 (Enhancements)
- Voice capability integration
- Cost calculator
- Performance optimizations

## 📌 Notes

- Books are the MOST critical missing piece
- Always reference both podcasts AND books when relevant
- Keep all links within the YonEarth ecosystem
- Voice feature excited Aaron the most for accessibility
- Feedback during testing phase is crucial