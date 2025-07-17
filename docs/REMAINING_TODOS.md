# Remaining TODOs for YonEarth Gaia Chatbot

Based on Aaron's feedback and our discoveries, here are the critical remaining tasks:

## 🚨 Critical Priority (Do First)

### 1. Add More Books to Dataset
**Impact**: Aaron said responses should "almost always" reference both podcasts AND books

- [ ] Add "Why on Earth" book by Aaron William Perry
- [ ] Add "Soil Stewardship Handbook" 
- [ ] Ensure book integration works seamlessly with episodes
- [ ] Update prompts to prioritize book+episode citations together

**Implementation**:
```bash
# Place PDFs in /data/books/book_name/
# Add metadata.json for each book
# Run: python3 -m src.ingestion.process_books
```

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

### 4. Add Feedback Component
**Impact**: Critical for iterative improvement during Aaron's testing

- [ ] Add thumbs up/down buttons per response
- [ ] Capture feedback with response ID
- [ ] Store in database for analysis
- [ ] Show feedback stats in admin view

### 5. Implement Semantic Category Matching
**Impact**: Solves "soil" → BIOCHAR matching issue

- [ ] Complete implementation from SEMANTIC_CATEGORY_IMPLEMENTATION_PLAN.md
- [ ] Use OpenAI embeddings for category descriptions
- [ ] Add configurable thresholds in UI
- [ ] Show matched categories in responses

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

## 📝 Quick Wins Order

1. **Add the missing books** (most critical per Aaron)
2. **Fix recommended content** to match citations
3. **Add hyperlinks** to episode references
4. **Add feedback buttons** for testing phase

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