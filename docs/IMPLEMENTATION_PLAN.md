# YonEarth Implementation Plan

**Last Updated**: November 22, 2025 12:30 AM

---

## ✅ COMPLETED - Chatbot & RAG System (July 2025)

**All major chatbot features have been successfully deployed:**
- ✅ Full VPS Deployment with Docker, nginx, Redis, FastAPI
- ✅ Web Interface with beautiful Earth-themed chat UI
- ✅ Dual RAG Systems (Original + BM25 Hybrid)
- ✅ Semantic Category Matching with OpenAI embeddings
- ✅ Episode Diversity Algorithm
- ✅ Gaia Character with multiple personalities
- ✅ Voice Integration with ElevenLabs TTS
- ✅ User Feedback System
- ✅ Book Integration (3 books: VIRIDITAS, Soil Stewardship Handbook, Y on Earth)
- ✅ Production APIs with rate limiting, CORS, health checks
- ✅ Episode Processing (172 episodes with word-level timestamps)
- ✅ Pinecone Vector Database (18,764+ vectors)
- ✅ BM25 + Semantic + Cross-encoder reranking
- ✅ WordPress integration capability

**Current Status**: Fully functional production chatbot deployed

---

## ✅ COMPLETED - Knowledge Graph Extraction (November 21, 2025)

### 🎯 Project: Unified Knowledge Graph with Discourse Elements

**Goal**: Build unified knowledge graph from ACE-extracted episodes + books with A+ quality and multi-source consensus tracking

**Status**: ✅ **COMPLETE** - Ready for GraphRAG hierarchy generation

### Completed Tasks (November 21, 2025)

#### 1. ✅ ACE Book Extraction (All 4 Books)

**Pipeline**: ACE V14.3.8 with 18/18 postprocessing modules

| Book | Relationships | Status |
|------|--------------|--------|
| VIRIDITAS: THE GREAT HEALING | 2,302 | ✅ Complete |
| Soil Stewardship Handbook | 263 | ✅ Complete |
| Y on Earth | 2,669 | ✅ Complete |
| Our Biggest Deal | 2,187 | ✅ Complete |
| **TOTAL** | **7,421** | ✅ **All Complete** |

**Location**: `/data/knowledge_graph/books/*_ace_v14_3_8_cleaned.json`

**Features**:
- ✅ Checkpointing implemented (saves every 10 chunks)
- ✅ 18/18 ACE postprocessing modules working
- ✅ Praise quote cleanup (16 endorsements removed)
- ✅ Type-safe entity extraction
- ✅ Context enrichment and pronoun resolution

#### 2. ✅ Classification Flags Added

**Episodes**:
- Added `classification_flags` to 43,297 episode relationships
- Classification: 90.3% factual, 3.9% philosophical, 3.5% opinion, 2.5% recommendation

**Books**:
- Added `classification_flags` to 7,421 book relationships
- Classification: 97.5% factual, 2.1% philosophical, 0.5% opinion, 0.0% recommendation

**Purpose**: Enable discourse graph transformation by identifying claim-worthy relationships

#### 3. ✅ Unified Graph Integration

**Process**:
1. Loaded existing unified graph (172 episodes, 43,297 relationships)
2. Processed 4 cleaned book files
3. Added classification_flags to book relationships
4. Converted to unified graph format
5. Merged with episode graph

**Result**:
- **File**: `/data/knowledge_graph_unified/unified_normalized.json` (30MB)
- **Entities**: 39,046
- **Relationships**: 50,718 (43,297 episodes + 7,421 books)
- **Coverage**: 172 episodes + 4 books with full classification

#### 4. ✅ Discourse Graph Transformation

**Implementation**: Hybrid Model (Option B) - Keep factual relationships, transform opinions/recommendations to claims

**Process**:
1. Identified 5,772 claim-worthy relationships (opinion, recommendation, philosophical)
2. Created 5,506 unique claims (266 duplicates merged via fuzzy matching)
3. Generated 5,772 attribution edges (Person --MAKES_CLAIM--> Claim)
4. Added 5,772 ABOUT edges (Claim --ABOUT--> Concept)
5. Calculated consensus scores for all claims

**Result**:
- **File**: `/data/knowledge_graph_unified/discourse_graph_hybrid.json` (45MB)
- **Entities**: 44,552 (39,046 original + 5,506 claim nodes)
- **Relationships**: 62,262 (50,718 original + 11,544 discourse edges)
- **Multi-Source Claims**: 169 (same claim made by multiple sources)
- **Consensus Tracking**: Enabled for multi-source statements

**Benefits**:
- Know exactly who made each claim (attribution tracking)
- Identify statements multiple sources agree on (consensus detection)
- Similar statements merged into single claims (claim aggregation)
- Track which claims come from episodes vs. books (source diversity)

---

## ⏳ NEXT - GraphRAG Hierarchy & 3D Visualization

### 1. Regenerate GraphRAG Hierarchy (Pending - ~2-3 hours)

**Script**: `scripts/generate_graphrag_hierarchy.py`

**Input**: `/data/knowledge_graph_unified/discourse_graph_hybrid.json`
- 44,552 entities (including 5,506 claim nodes)
- 62,262 relationships (including attribution edges)

**Process**:
1. Generate OpenAI embeddings for all entities
2. Apply UMAP for 3D positioning
3. Build hierarchical clusters (K-means)
4. Calculate cluster metadata and statistics
5. Export to `/data/graphrag_hierarchy/graphrag_hierarchy.json`

**Expected Output**:
- 44,552 entities with 3D coordinates
- Hierarchical cluster structure (L1, L2, L3)
- Cluster metadata (top entities, relationship counts)
- Entity search index

**Estimated Time**: 2-3 hours

### 2. Deploy to 3D Visualization (Pending - ~30 minutes)

**URL**: https://gaiaai.xyz/YonEarth/graph/

**Tasks**:
1. Copy new `graphrag_hierarchy.json` to production server
2. Restart 3D visualization service
3. Verify Moscow ≠ Soil fix (entities properly separated by type)
4. Test search and navigation
5. Verify discourse graph features (claim nodes visible, attribution edges)
6. Test multi-source consensus queries

**Verification Checklist**:
- [ ] Moscow and Soil are separate entities (no merge)
- [ ] Claim nodes visible in graph
- [ ] Attribution edges displayed correctly
- [ ] Search returns accurate results
- [ ] Cluster hierarchies navigate smoothly
- [ ] Multi-source claims highlighted

---

## 📋 COMPLETED FEATURES

### Knowledge Graph Quality

✅ **ACE Postprocessing Pipeline** (18/18 modules working):
1. FieldNormalizer - Standardize field names
2. PraiseQuoteDetector - Identify endorsement noise
3. MetadataFilter - Remove bibliographic noise
4. FrontMatterDetector - Filter front matter relationships
5. DedicationNormalizer - Standardize dedication relationships
6. SubtitleJoiner - Rehydrate split subtitles
7. BibliographicCitationParser - Parse citations correctly
8. ContextEnricher - Add relationship context
9. ListSplitter - Expand list relationships
10. PronounResolver - Resolve pronouns to entities
11. PredicateNormalizer - Normalize relationship types
12. PredicateValidator - Validate predicates
13. TypeCompatibilityValidator - Ensure type-compatible relationships
14. VagueEntityBlocker - Block vague entities
15. TitleCompletenessValidator - Validate title completeness
16. FigurativeLanguageFilter - Filter metaphors
17. ClaimClassifier - Classify factual vs. opinion relationships
18. Deduplicator - Remove duplicate relationships

✅ **Discourse Graph Elements**:
- Claim nodes for opinions/recommendations/philosophical statements
- Attribution edges (Person --MAKES_CLAIM--> Claim)
- ABOUT edges (Claim --ABOUT--> Concept)
- Multi-source consensus scoring
- Source diversity tracking (episodes vs. books)

✅ **Type-Safe Entity Merging**:
- Entity Merge Validator prevents catastrophic merges
- Moscow ≠ Soil (PLACE ≠ CONCEPT)
- Similarity threshold validation
- Semantic blocklist for known bad merges

### Scripts Created/Updated

**New Scripts (November 21, 2025)**:
1. `scripts/add_classification_flags_to_episodes.py` - Classify episode relationships
2. `scripts/add_classification_flags_to_unified_graph.py` - Classify unified graph
3. `scripts/integrate_books_into_unified_graph.py` - Integrate books with classification
4. `scripts/transform_to_discourse_graph.py` - Transform to discourse graph (updated)
5. `scripts/cleanup_book_endorsement_noise.py` - Remove praise quotes (updated for 4 books)

**Existing Scripts**:
- `scripts/extract_books_ace_full.py` - ACE book extraction (with checkpointing)
- `scripts/build_unified_graph_hybrid.py` - Build hybrid unified graph
- `scripts/generate_graphrag_hierarchy.py` - Generate 3D hierarchy (ready to run)

---

## 🎯 RECOMMENDED APPROACH VALIDATED

**Hybrid ACE + Discourse Graph Approach**: ✅ **SUCCESS**

Our final approach combining:
1. ACE-postprocessed episodes (172 episodes, highest quality)
2. ACE-extracted books (4 books, 7,421 relationships)
3. Classification flags (opinion/philosophical/recommendation on all 50,718 relationships)
4. Discourse graph transformation (5,506 claims with attribution and consensus)

**Result**: State-of-the-art knowledge graph with multi-source consensus tracking!

**Quality Metrics**:
- ✅ 18/18 ACE postprocessing modules working
- ✅ 100% classification coverage (all 50,718 relationships)
- ✅ 169 multi-source consensus claims identified
- ✅ Type-safe entity separation (Moscow ≠ Soil)
- ✅ Clean book relationships (16 endorsements removed)
- ✅ 5,506 unique claims with attribution tracking
- ✅ Source diversity metrics (episodes vs. books)

---

## 📊 Final Statistics

### Content Coverage
- **Episodes**: 172 (with word-level timestamps)
- **Books**: 4 (VIRIDITAS, Soil Stewardship Handbook, Y on Earth, Our Biggest Deal)

### Unified Graph (`unified_normalized.json`)
- **Entities**: 39,046
- **Relationships**: 50,718
  - 43,297 from episodes
  - 7,421 from books
- **Classification**: 91.4% factual, 3.7% philosophical, 3.1% opinion, 2.1% recommendation

### Discourse Graph (`discourse_graph_hybrid.json`)
- **Entities**: 44,552 (includes 5,506 claim nodes)
- **Relationships**: 62,262 (includes 11,544 discourse edges)
- **Claims**: 5,506 unique claims
- **Multi-Source Claims**: 169 (consensus tracking enabled)
- **Attribution Edges**: 5,772

---

## 🔮 FUTURE IMPROVEMENTS

### Discourse Graph Enhancements

#### ✅ COMPLETE: Implement Discourse Graph Elements

**Status**: ✅ Implemented (Nov 21, 2025)

**What Was Done**:
- Hybrid Model (Option B) implemented as one-off script
- 5,506 unique claims created from 5,772 claim-worthy relationships
- 169 multi-source claims identified for consensus tracking
- Attribution edges connect people to their claims
- Source diversity metrics track episodes vs. books

**Future Enhancement** (for next extraction cycle):
- Move discourse graph transformation into extraction pipeline
- Real-time claim creation during extraction
- Live consensus scoring as new content is added
- See `/docs/IMPLEMENTATION_PLAN.md` (lines 422-674) for full pipeline integration plan

### GraphRAG Visualization Enhancements

**After current deployment**:
- Add claim node highlighting in 3D visualization
- Show attribution edges with different colors
- Display consensus scores on hover
- Filter by multi-source claims
- Show source diversity (episode vs. book breakdown)

### Content Processing

**Next extraction cycle**:
- Process additional podcast episodes (if new episodes published)
- Extract additional books (if added to collection)
- Re-run with full pipeline integration of discourse graph elements

### GraphRAG LLM Cost Optimization

#### Current Implementation (November 22, 2025)

**Script**: `scripts/build_proper_graphrag.py` ✅ **RUNNING**

**Current Approach**:
- **Model**: gpt-4o-mini ($0.15 input / $0.60 output per 1M tokens)
- **Dataset**: 17,296 entities, 20,508 relationships
- **Leiden Communities**: 6,398 total (3,514 L0 + 755 L1 + 2,129 L2)
- **Estimated Cost**: $1.34 total for all LLM summaries
- **Processing Time**: ~5-6 hours total
- **Retry Logic**: Exponential backoff for 429 and 500/503 errors
- **Features Added**: Betweenness centrality, relationship strengths, UMAP 3D positions

**Checkpointing (Nov 22, 2025 ✅ ADDED)**:
- **Embeddings**: Saved to `checkpoints/embeddings.npy` (102MB)
- **Leiden hierarchies**: Saved to `checkpoints/leiden_hierarchies.json` (1.9MB)
- **Summary progress**: Saved to `checkpoints/summaries_progress.json` every 50 communities
- **Resume capability**: If script crashes, resumes from last checkpoint instead of restarting
- **Rate delay**: Reduced from 0.2s → 0.05s (4x faster, still safe for rate limits)

**Cost Breakdown**:
- Level 0 (3,514 communities): ~$0.26
- Level 1 (755 communities): ~$0.13
- Level 2 (2,129 communities): ~$1.09 (most expensive, finest-grained)
- **Total**: ~$1.34 (very reasonable for 6,398 summaries!)

#### 🔬 Future Optimization Experiments

**1. GPT-5 Nano Alternative** ⭐ **HIGH PRIORITY**

**Pricing Comparison**:
- **gpt-4o-mini**: $0.15 input / $0.60 output per 1M tokens
- **GPT-5 nano**: $0.05 input / $0.40 output per 1M tokens (when available)
- **Potential Savings**: ~67% cheaper on input, ~33% cheaper on output

**Expected Impact**:
- Current run: $1.34 total → **Estimated with GPT-5 nano: ~$0.50-$0.70**
- For larger graphs (100K+ entities): Savings could reach $50-$100+

**Action Items**:
- Monitor GPT-5 nano release and availability
- Test quality on small subset first (100-200 communities)
- Compare summary coherence against gpt-4o-mini baseline
- If quality acceptable, switch default model in script

**2. Batch API for 50% Discount** 💰 **HIGH VALUE**

**Current**: Real-time API calls with retry logic
**Alternative**: Batch API for offline processing

**Benefits**:
- **50% cost reduction** on all API calls
- No rate limiting concerns (process async)
- Better for large-scale runs (10K+ communities)

**Implementation Strategy**:
```python
# Prepare batch job
batch_requests = []
for community_id, members in communities.items():
    batch_requests.append({
        "custom_id": f"summary-{community_id}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {"model": "gpt-4o-mini", "messages": [...]}
    })

# Submit batch job (24-hour turnaround)
batch_job = client.batches.create(
    input_file_id=uploaded_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h"
)

# Poll for completion, then process results
```

**Trade-offs**:
- **Pro**: 50% cheaper, no rate limits
- **Con**: 24-hour turnaround (not real-time)
- **Best For**: Final production runs, not rapid iteration

**3. Output Length Clamping** 📏 **MEDIUM PRIORITY**

**Current**: No explicit output limits (GPT decides length)
**Proposed**: Set `max_tokens` parameter to control verbosity

**Strategy**:
```python
# Tiered max_tokens by level
max_tokens_by_level = {
    0: 100,   # Fine-grained: short, specific
    1: 200,   # Mid-level: moderate detail
    2: 400    # Top-level: comprehensive overview
}
```

**Expected Savings**:
- Level 0 (3,514 communities): 100 tokens max → ~$0.15 (vs $0.26)
- Level 1 (755 communities): 200 tokens max → ~$0.09 (vs $0.13)
- Level 2 (2,129 communities): 400 tokens max → ~$0.85 (vs $1.09)
- **Total**: ~$1.09 (19% savings)

**Risks**:
- Summaries may be too terse if limits too strict
- Need to test quality on subset first

**4. Subset Tuning Before Full Runs** 🧪 **BEST PRACTICE**

**Current**: Run full 6,398 communities without testing
**Proposed**: Always test on small subset first (100-200 communities)

**Workflow**:
```bash
# Step 1: Test on subset (5 minutes, $0.02)
python scripts/build_proper_graphrag.py --max-communities 100 --test-mode

# Step 2: Review output quality
cat /data/graphrag_hierarchy/test_summaries.json | jq '.clusters.level_0[0].summary'

# Step 3: Adjust prompt if needed, re-test
# Step 4: Run full production job with validated settings
python scripts/build_proper_graphrag.py --production
```

**Benefits**:
- Catch prompt issues early (before spending $1.34)
- Test different models (gpt-4o-mini vs GPT-5 nano)
- Experiment with max_tokens settings
- Verify retry logic works correctly

**5. Caching by Community ID** 🗂️ **MEDIUM PRIORITY**

**Problem**: Re-running script regenerates ALL summaries (wasteful if graph changes slightly)

**Solution**: Cache summaries by community ID, only regenerate changed communities

**Implementation**:
```python
# Check cache before generating summary
cache_file = f"/data/graphrag_hierarchy/cache/summaries_level_{level}.json"
cached_summaries = load_cache(cache_file)

if community_id in cached_summaries and not force_regenerate:
    return cached_summaries[community_id]
else:
    summary = generate_llm_summary(community)
    cached_summaries[community_id] = summary
    save_cache(cache_file, cached_summaries)
    return summary
```

**Benefits**:
- Incremental updates (only regenerate changed communities)
- Faster iterations during development
- Cost savings on re-runs (only pay for new/changed communities)

**Trade-offs**:
- Cache invalidation complexity (when to regenerate?)
- Disk space for cache storage (~10-20MB)

**6. Tiered Model Approach** 🎯 **ADVANCED OPTIMIZATION**

**Strategy**: Use cheap models for fine-grained summaries, expensive models for top-level

**Rationale**:
- **Level 0 (3,514 communities)**: Small, specific → GPT-5 nano ($0.05/$0.40)
- **Level 1 (755 communities)**: Mid-level → GPT-5 nano ($0.05/$0.40)
- **Level 2 (2,129 communities)**: High-level, critical → gpt-4o-mini or gpt-4o ($0.15/$0.60+)

**Expected Cost**:
- Level 0 with GPT-5 nano: ~$0.09 (vs $0.26 with gpt-4o-mini)
- Level 1 with GPT-5 nano: ~$0.04 (vs $0.13 with gpt-4o-mini)
- Level 2 with gpt-4o-mini: ~$1.09 (same as current)
- **Total**: ~$1.22 (9% savings, better quality on top-level)

**Implementation**:
```python
model_by_level = {
    0: "gpt-5-nano",      # Cheap, high volume
    1: "gpt-5-nano",      # Cheap, medium volume
    2: "gpt-4o-mini"      # Best quality for top-level
}

summary = generate_llm_summary(
    community,
    model=model_by_level[level]
)
```

**7. Cost Control Strategies for Iterative Development** 🛡️

**Problem**: During development, want to iterate quickly without burning through API budget

**Solutions**:

A. **Development Mode with Smaller Graph**:
```bash
# Use subset of entities (1,000 instead of 17,296)
python scripts/build_proper_graphrag.py --max-entities 1000
# Expected: ~$0.10-$0.20 per run (vs $1.34)
```

B. **Skip LLM Summaries Flag**:
```python
# Add command-line flag to skip expensive LLM step
if args.skip_llm_summaries:
    print("Skipping LLM summarization (development mode)")
    # Still generates embeddings, UMAP, betweenness, clusters
    # Just no text summaries
```

C. **Mock LLM for Testing**:
```python
# Use placeholder summaries for testing visualization
if args.mock_llm:
    summary = f"Mock summary for community {community_id}"
    # Free, instant, good for frontend testing
```

#### 📊 Cost Optimization Matrix

| Strategy | Savings | Effort | Risk | Priority |
|----------|---------|--------|------|----------|
| GPT-5 Nano | 50-67% | Low | Medium (quality unknown) | ⭐ HIGH |
| Batch API | 50% | Medium | Low | ⭐ HIGH |
| Output Clamping | 10-20% | Low | Medium (may truncate) | MEDIUM |
| Subset Tuning | Prevents waste | Low | None | ⭐ BEST PRACTICE |
| Caching | Varies (incremental) | Medium | Low | MEDIUM |
| Tiered Models | 5-15% | Medium | Low | MEDIUM |
| Dev Mode | 90%+ (testing) | Low | None (dev only) | ⭐ HIGH |

#### 🎯 Recommended Next Steps

**Immediate (Next Run)**:
1. ✅ Add `--max-communities` flag for subset testing
2. ✅ Add `--skip-llm` flag for development iterations
3. ✅ Add `--model` parameter to easily switch models
4. ✅ Test GPT-5 nano on 100 communities when available

**Short-Term (1-2 weeks)**:
1. Implement Batch API support for production runs
2. Add caching layer for summaries
3. Experiment with output length limits (100/200/400 tokens)
4. Compare quality: gpt-4o-mini vs GPT-5 nano vs gpt-3.5-turbo

**Long-Term (Future Extraction Cycles)**:
1. Implement tiered model approach (cheap for L0/L1, expensive for L2)
2. Add cache invalidation logic for incremental updates
3. Monitor GPT-5 nano quality and adjust default model
4. Optimize prompts for shorter, more focused summaries

**Key Insight**: Current $1.34 cost is extremely reasonable, but for larger graphs (100K+ entities) or frequent re-runs, these optimizations could save $100-$500+ per iteration while maintaining quality.

---

## 📝 Documentation

### Primary Documentation
- `IMPLEMENTATION_STATUS.md` - Current status and completed work ⭐
- `IMPLEMENTATION_PLAN.md` - This file - Overall project timeline
- `GRAPHRAG_3D_EMBEDDING_VIEW.md` - 3D visualization architecture

### Technical Documentation
- `ACE_FRAMEWORK_DESIGN.md` - ACE extraction pipeline design
- `CONTENT_PROCESSING_PIPELINE.md` - Episode and book processing
- `KNOWLEDGE_GRAPH_REGENERATION_PLAN.md` - Graph regeneration strategy

### Deprecated/Merged
- ~~`CURRENT_STATE_ANALYSIS.md`~~ → Merged into `IMPLEMENTATION_STATUS.md`

---

## 🚀 Timeline to Production

**Current State**: Knowledge graph extraction and transformation COMPLETE ✅

**Next Phase**: GraphRAG hierarchy generation + 3D visualization deployment

**Estimated Timeline**:
1. GraphRAG hierarchy generation: 2-3 hours
2. Deployment + testing: 30 minutes
3. **Total ETA**: ~3-4 hours to live 3D visualization with discourse graph

**Ready to Deploy**: All data prepared, scripts ready, just need to run GraphRAG generation!
