# Remaining TODOs

Current outstanding tasks and future enhancements for the YonEarth Gaia Chatbot.

**Last Updated**: October 4, 2025

---

## 🚨 Critical Priority

### 1. Add Hyperlinks in Responses
**Impact**: Essential for navigation into YonEarth ecosystem

**Current**: Plain text citations
```
Episode 120: High Plains Biochar with Kelpie Wilson & Troy Cowan
```

**Needed**: Clickable links
```
[Episode 120: High Plains Biochar](https://yonearth.org/podcast/episode-120)
```

**Implementation**:
- Update `src/character/gaia.py` - citation formatting
- Update `web/chat.js` - render markdown links
- Keep all links within YonEarth ecosystem
- Format: `[Episode XXX](https://yonearth.org/podcast/episode-XXX)`

**Status**: Not started

---

### 2. Fix Recommended Content Sync
**Impact**: Improves user experience and trust

**Issue**: Sometimes referenced episodes don't appear in recommendations section

**Example Problem**:
- Response cites Episode 165
- Recommended Content shows Episodes 120, 122, 124 (missing 165)

**Fix Needed**:
- Ensure ALL cited episodes appear in recommendations
- Sync citations with recommended content section
- Verify episode diversity algorithm includes all matches

**Status**: Partial - diversity algorithm works but sync needs verification

---

## 🔧 High Priority

### 3. Configurable Search Weights UI
**Impact**: Power users can optimize search for their needs

**Current**: Fixed weights (60% category, 25% semantic, 15% keyword)

**Needed**:
- Add sliders for category/semantic/keyword weights
- Constraint: Total must equal 100%
- Show current weights in UI
- Save preferences in localStorage
- Reset to defaults button

**Implementation**:
- Add UI controls in `web/index.html`
- Update weight passing in `web/chat.js`
- Modify API to accept custom weights

**Status**: Not started

---

### 4. Cost Calculator Dashboard
**Impact**: Important for planning public rollout and sustainability

**Current**: Shows cost per response in footer

**Needed**:
- Daily/weekly/monthly cost reports
- Cost breakdown by operation type
- Usage statistics (queries per day, avg cost per query)
- Cost projections for different usage levels
- Export cost data to CSV

**Implementation**:
- Create `/api/costs/summary` endpoint
- Add cost dashboard page `web/costs.html`
- Store cost history in database or JSON files
- Analytics and visualizations

**Status**: Basic tracking exists, dashboard not built

---

## 📊 Medium Priority

### 5. Enhanced Feedback Analytics
**Impact**: Better understanding of response quality

**Current**: Basic thumbs up/down and text comments

**Enhancements**:
- Feedback dashboard for reviewing trends
- Filter by date range, rating, search method
- Export feedback to CSV
- Identify common issues from comments
- Track improvement over time

**Implementation**:
- Enhance `scripts/view_feedback.py` into web dashboard
- Add `/api/feedback/analytics` endpoint
- Create `web/feedback_dashboard.html`

**Status**: Basic collection works, analytics needed

---

### 6. Search History & Saved Conversations
**Impact**: Users can return to previous conversations

**Features**:
- Save conversation threads
- Search through past conversations
- Export conversations as markdown/PDF
- Share conversation links
- Delete conversation history

**Implementation**:
- Add localStorage conversation saving
- Create `/api/conversations` CRUD endpoints
- Add conversation list UI in sidebar
- Privacy controls

**Status**: Not started

---

### 7. Episode Tagging & Filtering
**Impact**: Better episode discovery

**Features**:
- Browse episodes by category
- Filter by guest, topic, date
- "Show me all permaculture episodes"
- Episode timeline visualization
- Most referenced episodes

**Implementation**:
- Add `/api/episodes/browse` endpoint
- Create episode browser page
- Use existing category data
- Add guest/date metadata

**Status**: Data exists, UI not built

---

## 🌟 Future Enhancements

### 8. Knowledge Graph Visualization
**Impact**: Visual exploration of concepts and relationships

**Features**:
- Interactive graph of entities and relationships
- Click node to see related episodes/books
- Filter by entity type (person, concept, practice)
- 2D/3D visualization options

**Status**: Data extracted (6,000+ entities), visualization not built
**Reference**: See `docs/archive/KNOWLEDGE_GRAPH_FINAL_REPORT.md`

---

### 9. Email Summaries
**Impact**: Regular engagement without visiting site

**Features**:
- Weekly episode recommendations
- Topics trending in community
- New episodes added
- Personalized based on interests

**Status**: Not started

---

### 10. Mobile App
**Impact**: Better mobile user experience

**Options**:
- Progressive Web App (PWA)
- React Native app
- Flutter app

**Features**:
- Optimized touch interface
- Offline mode
- Push notifications
- Voice-first interaction

**Status**: Not started

---

## ✅ Recently Completed

### Books Integration ✅ (July 2025)
- Added 3 books (VIRIDITAS, Soil Stewardship Handbook, Y on Earth)
- 4,289+ book chunks searchable
- Multi-format links (eBook, audiobook, print)
- Chapter-level citations

### Voice Integration ✅ (August 2025)
- ElevenLabs TTS integration
- Custom voice support
- Toggle controls in UI
- Auto-play and manual replay

### Semantic Category Matching ✅ (July 2025)
- OpenAI embeddings for true semantic understanding
- Configurable thresholds (Broad/Normal/Strict/Disabled)
- Cached embeddings for performance
- Episode diversity algorithm

### User Feedback System ✅ (July 2025)
- Quick feedback (thumbs up/down)
- Detailed feedback (5-star, comments)
- Backend storage and API
- View feedback script

### Cost Tracking ✅ (July 2025)
- Per-response cost calculation
- Breakdown by operation type
- Display in UI footer

### Max References Configuration ✅ (July 2025)
- User-configurable 1-10 references
- Backend properly handles variable counts

---

## 🎯 Quick Wins (Do These First)

1. **Add hyperlinks to episode references** (2-3 hours)
   - High impact, low effort
   - Enables navigation to YonEarth ecosystem

2. **Fix recommended content sync** (1-2 hours)
   - Improves trust and UX
   - Debugging existing code

3. **Create feedback analytics dashboard** (3-4 hours)
   - Leverage existing feedback data
   - Helps prioritize improvements

4. **Add search weights UI** (2-3 hours)
   - Empowers power users
   - Simple UI addition

---

## 📅 Suggested Implementation Timeline

### Week 1 (Quick Wins)
- [ ] Add hyperlinks to responses
- [ ] Fix recommended content sync
- [ ] Test and verify fixes

### Week 2 (User Controls)
- [ ] Add search weights UI
- [ ] Enhance feedback analytics
- [ ] Create cost calculator dashboard

### Week 3 (Discovery Features)
- [ ] Episode browser/tagging
- [ ] Search history
- [ ] Save conversations

### Week 4+ (Advanced Features)
- [ ] Knowledge graph visualization
- [ ] Email summaries
- [ ] Mobile PWA

---

## 📝 Notes

- **Books** were the most critical missing piece - now complete ✅
- **Voice** feature provides accessibility benefits
- **Hyperlinks** are next most important for ecosystem integration
- Focus on improving existing features before adding new ones
- User feedback will guide priority adjustments

---

## 🤝 Contributing

Want to tackle one of these TODOs?

1. Comment on the GitHub issue for that feature
2. Review [CLAUDE.md](../CLAUDE.md) for development setup
3. Create a feature branch
4. Test thoroughly
5. Submit pull request

See [docs/README.md](README.md) for more contribution guidelines.

---

**Questions about priorities?** Open a discussion on GitHub or contact the maintainers.
