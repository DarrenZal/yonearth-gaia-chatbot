# Phase 2 Implementation Roadmap

## Current State (Phase 1 Complete)
- ✅ 44,836 entities, 47,769 relationships
- ✅ 100% canonical types (8 types)
- ✅ Orphan rates: 1.53% entities, 0.21% edges
- ✅ Graph retrieval integrated with evidence capping
- ✅ CI validation and monitoring in place
- ✅ Performance: <10ms graph lookups, ~100MB memory

## Phase 2 Objectives

### 1. Entity Normalization (2-3 days)

#### Implementation Steps:
```python
# Three-pass normalization pipeline
Pass 1: Exact key matching from alias_map.json
Pass 2: Apply 3.1MB normalization map
Pass 3: Fuzzy matching (Jaccard >0.85, Levenshtein <3)
```

#### Deliverables:
- [ ] `scripts/apply_entity_normalization.py`
- [ ] `data/knowledge_graph_unified/merges.json` - audit trail
- [ ] Updated `unified.json` with merged entities
- [ ] CI validation: max 100 merges per root entity

#### Success Metrics:
- Entity count reduction: 10-15% expected
- No increase in orphan rates (±0.1%)
- All merges logged and reversible

### 2. Episode Discourse Extraction (3-4 days)

#### Pilot Phase (20 episodes):
```python
# ACE extraction for test batch
Episodes: 100-120 (diverse topics)
Expected output: ~50 assertions, ~30 questions per episode
Cost estimate: ~$5-10 for pilot
```

#### Full Rollout:
- [ ] `scripts/extract_episode_discourse.py`
- [ ] `data/knowledge_graph_unified/discourse.json` - merged with books
- [ ] `data/knowledge_graph_unified/discourse_index.json` - chunk mapping

#### Success Metrics:
- Coverage: >80% episodes with ≥1 assertion
- Quality: >90% assertions with p_true ≥0.6
- Discourse elements: 10,000+ assertions, 5,000+ questions

### 3. Cross-Content Linking (2-3 days)

#### Implementation:
```python
# Link builders
mentioned_in: entity → [episode_ids, book_ids]
supports: assertion → assertion (confidence ≥0.7)
contradicts: assertion → assertion (polarity opposite)
```

#### Deliverables:
- [ ] `scripts/build_cross_content_links.py`
- [ ] Enhanced `adjacency.json` with cross-content edges
- [ ] `data/knowledge_graph_unified/content_links.json`

#### Success Metrics:
- 5,000+ mentioned_in edges
- 500+ supports/contradicts relationships
- All books linked to relevant episodes

### 4. Performance Testing (1-2 days)

#### Test Setup:
```python
# Locustfile.py
- 100 concurrent users
- Query mix: 70% simple, 20% complex, 10% graph-heavy
- Target: p95 < 2.5s end-to-end
```

#### Deliverables:
- [ ] `tests/locustfile.py`
- [ ] Performance report with p50/p95/p99
- [ ] Memory profile under load
- [ ] Optimization recommendations

#### Success Criteria:
- p95 latency ≤ 2.5s
- Error rate ≤ 1%
- Memory usage ≤ 200MB
- Graph retrieval p95 ≤ 50ms

## Implementation Schedule

### Week 1 (Days 1-5)
- **Day 1-2**: Entity normalization implementation
- **Day 3**: Normalization validation and testing
- **Day 4-5**: Episode discourse pilot (20 episodes)

### Week 2 (Days 6-10)
- **Day 6-7**: Full episode discourse extraction
- **Day 8-9**: Cross-content linking
- **Day 10**: Performance testing and optimization

## CI/CD Integration

### Validation Gates:
```yaml
# .github/workflows/kg_validation.yml
- Orphan rates below thresholds
- No non-canonical types
- Entity count > 40,000
- Relationship count > 45,000
- Max auto-created entities: 500
```

### Nightly Build Pipeline:
```bash
# cron: 0 2 * * *
1. Pull latest transcripts/books
2. Run kg_merge_unified.py
3. Apply entity normalization
4. Extract discourse elements
5. Build cross-content links
6. Run validation script
7. Archive with build_id
8. Deploy if all checks pass
```

## Monitoring & Rollback

### Key Metrics to Track:
- `graph_evidence_count` - per request
- `graph_retrieval_ms` - p50/p95
- `orphan_rate_delta` - per build
- `merge_count` - per normalization run
- `discourse_coverage` - % with assertions

### Rollback Strategy:
```bash
# Keep last 3 builds
/data/knowledge_graph_unified/
  ├── current/ -> build_20251112_055335/
  ├── build_20251112_055335/
  ├── build_20251111_120000/
  └── build_20251110_080000/

# Quick rollback
ln -sfn build_20251111_120000 current
systemctl restart yonearth-api
```

## Risk Mitigation

### High-Risk Areas:
1. **Entity over-merging**: Cap at 100 merges per root
2. **Discourse noise**: Minimum p_true = 0.6
3. **Cross-content false positives**: Manual review top 100
4. **Performance regression**: Canary 10% traffic first

### Safeguards:
- Feature flags for each component
- Automated rollback on threshold breach
- Daily validation reports
- A/B testing with metrics

## Phase 2 Complete Criteria

- [ ] Entity deduplication complete (10-15% reduction)
- [ ] All episodes have discourse elements
- [ ] Cross-content graph fully connected
- [ ] Performance SLAs met (p95 < 2.5s)
- [ ] Zero regression in quality metrics
- [ ] Production deployment successful

## Next: Phase 3 Preview

Once Phase 2 is complete:
- **Semantic search improvements**: Query expansion with graph
- **Conversational memory**: Session-aware graph traversal
- **Confidence scoring**: ML-based relevance ranking
- **Interactive exploration**: Graph visualization APIs

---

*Timeline: 10-12 days total*
*Risk Level: Low (incremental changes)*
*Rollback Time: <5 minutes*