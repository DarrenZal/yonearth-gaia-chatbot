# Phase 2 Implementation Prompt for Claude Code

## Context
You're continuing work on the YonEarth Gaia Chatbot knowledge graph system. Phase 1 is complete with a production-ready unified knowledge graph. Now you need to implement Phase 2 to add entity normalization, discourse extraction, and cross-content linking.

## Current System State

### ✅ Phase 1 Complete (Production-Ready)
- **Unified KG**: 44,836 entities, 47,769 relationships
- **Quality**: 100% canonical types, normalized predicates
- **Orphan Rates**: 1.53% entities, 0.21% edges (excellent)
- **Performance**: <10ms lookups, ~100MB memory
- **Tools**: CI validation, orphan triage, Makefile operations

### 📁 Key Files & Locations
```bash
# Knowledge Graph Data
/data/knowledge_graph_unified/
  ├── unified.json         # Main graph (28MB)
  ├── adjacency.json       # Graph edges (3.1MB)
  ├── stats.json          # Metrics & validation
  └── discourse.json      # Discourse elements (240 assertions, 6 questions)

# Scripts Created in Phase 1
/scripts/
  ├── kg_merge_unified.py           # Merges books + episodes (with normalization)
  ├── validate_kg_thresholds.py     # CI validation
  ├── triage_orphan_edges.py        # Orphan analysis
  └── kg_build_discourse.py         # Discourse extraction

# Configuration
/data/knowledge_graph/ontology.yaml  # Layered ontology (domain + discourse)
/src/config/settings.py              # Configurable evidence caps
/Makefile.kg                         # Operations commands
```

### 🔧 Infrastructure Ready
- GraphRetriever integrated in `/src/rag/chain.py`
- Graph inspection API at `/api/graph/*`
- Evidence capping (configurable via settings)
- Warmup query on startup
- CI/CD validation pipeline

## Phase 2 Tasks

### Task 1: Entity Normalization (Priority: HIGH)
**Goal**: Reduce entity count by 10-15% through intelligent merging

1. Create `/scripts/apply_entity_normalization.py`:
```python
# Three-pass normalization
Pass 1: Exact key matching from existing alias_map
Pass 2: Apply 3.1MB normalization map from triage report
Pass 3: Fuzzy matching (Jaccard >0.85 or Levenshtein ≤3)

# Key requirements:
- Max 100 merges per root entity (prevent super-nodes)
- Generate merges.json audit trail
- Preserve all relationships during merge
- Update both unified.json and adjacency.json
```

2. Use findings from orphan triage:
```bash
# From /data/knowledge_graph_unified/orphan_triage_report.json
- Case mismatches: "TED Talk" vs "Ted Talk"
- Punctuation differences: 19 entities
- Suggested aliases ready to apply
```

3. Validation:
```bash
make -f Makefile.kg normalize
make -f Makefile.kg validate  # Should pass all thresholds
```

### Task 2: Episode Discourse Extraction (Priority: HIGH)
**Goal**: Extract questions, claims, and evidence from all episodes

1. Create `/scripts/extract_episode_discourse.py`:
```python
# Use ACE framework patterns
- Questions: "asks", "wonders", "inquires", sentences with "?"
- Claims: "states", "argues", "believes", "advocates"
- Evidence: Text snippets supporting claims

# Pilot on episodes 100-120 first
# Expected: ~50 assertions, ~30 questions per episode
# Minimum confidence: p_true ≥ 0.6
```

2. Merge with existing book discourse:
```python
# Update /data/knowledge_graph_unified/discourse.json
# Currently has 240 assertions from books
# Add ~8,000 assertions from episodes
# Create discourse_index.json for chunk mapping
```

### Task 3: Cross-Content Linking (Priority: MEDIUM)
**Goal**: Connect books and episodes through shared entities

1. Create `/scripts/build_cross_content_links.py`:
```python
# Build three types of edges:
1. mentioned_in: entity → [episode_ids, book_ids]
2. supports: assertion → assertion (confidence ≥0.7)
3. contradicts: assertion → assertion (opposite polarity)

# Use entity_chunk_map for efficient lookup
# Target: 5,000+ cross-content edges
```

### Task 4: Performance Testing (Priority: MEDIUM)
**Goal**: Validate system meets SLAs

1. Create `/tests/locustfile.py`:
```python
from locust import HttpUser, task, between

class ChatUser(HttpUser):
    wait_time = between(1, 3)

    @task(70)
    def simple_query(self):
        self.client.post("/api/chat", json={
            "message": "What is regenerative agriculture?",
            "k": 5
        })

    @task(20)
    def complex_query(self):
        # Multi-hop reasoning query

    @task(10)
    def graph_heavy_query(self):
        # Entity-rich query requiring graph traversal
```

2. Run tests:
```bash
make -f Makefile.kg perf
# Success criteria: p95 < 2.5s, error rate < 1%
```

### Task 5: Fix Remaining Issues
1. **Fix disallowed predicate**: "associated with" → "linked_to"
   - Add to predicate mapping in `kg_merge_unified.py`
   - Re-run merge

2. **Add aliases for auto-created entities** (267 total):
   - Use suggestions from triage report
   - Update alias_rules.yaml

## Recommended Implementation Order

### Day 1 (First Session)
1. Start with entity normalization script
2. Test on small subset first
3. Run full normalization
4. Validate with CI script

### Day 2
1. Implement episode discourse extraction
2. Pilot on 20 episodes
3. Check quality and adjust patterns
4. Run on all 172 episodes

### Day 3
1. Build cross-content links
2. Create mentioned_in edges
3. Add supports/contradicts relationships
4. Update adjacency.json

### Day 4
1. Performance testing with Locust
2. Optimize any bottlenecks
3. Final validation
4. Deploy to production

## Commands to Run First

```bash
# Check current state
cd /home/claudeuser/yonearth-gaia-chatbot
make -f Makefile.kg stats
make -f Makefile.kg validate

# Review orphan triage for normalization targets
cat /data/knowledge_graph_unified/orphan_triage_report.json | python3 -m json.tool | head -50

# Check discourse status
wc -l /data/knowledge_graph_unified/discourse.json

# Test graph API
curl http://localhost:8000/api/graph/stats
```

## Success Criteria

### Phase 2 Complete When:
- [ ] Entity count reduced by 10-15% (target: ~38,000 entities)
- [ ] All 172 episodes have discourse elements (>80% coverage)
- [ ] 5,000+ cross-content links created
- [ ] Performance tests pass (p95 < 2.5s)
- [ ] CI validation passes all thresholds
- [ ] Zero regression in orphan rates

## Important Notes

1. **Always run validation after changes**:
   ```bash
   make -f Makefile.kg validate
   ```

2. **Keep audit trails**:
   - Save merges.json for entity normalization
   - Log discourse extraction stats
   - Document cross-content link counts

3. **Use feature flags**:
   - `settings.enable_graph_retrieval` is already True
   - Can toggle for A/B testing if needed

4. **Monitor memory usage**:
   - Current: ~100MB
   - Target: Stay under 200MB
   - Use lazy loading if needed

5. **Backup before major changes**:
   ```bash
   make -f Makefile.kg backup
   ```

## Questions to Consider

1. Should we increase the evidence cap from 10 to 15 for richer context?
2. Should we add a "relevance score" to cross-content links?
3. Should discourse extraction use GPT-4o-mini or GPT-3.5-turbo for cost?
4. Should we implement incremental updates or full rebuilds?

## Additional Recommendations

1. **Add monitoring dashboard**: Track KG metrics over time
2. **Implement caching layer**: Redis for frequent entity lookups
3. **Add user feedback loop**: Learn from which evidence is helpful
4. **Create visualization API**: D3.js-ready graph data endpoints
5. **Build admin interface**: Manual entity merge/split tools

Good luck with Phase 2! The foundation is solid, and all the tools are in place for success.