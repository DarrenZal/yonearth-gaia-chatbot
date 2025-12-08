# GraphRAG vs BM25 Comparison Analysis

## Executive Summary

Testing 25 diverse prompts across 6 categories revealed that the GraphRAG implementation is working well, with some areas for improvement.

### Key Metrics

| Metric | BM25 | GraphRAG |
|--------|------|----------|
| Success Rate | 25/25 (100%) | 25/25 (100%) |
| Average Response Time | 8.91s | 5.15s |
| Response Length | ~1,458 chars | ~1,829 chars |
| Speed Comparison | - | **1.7x faster** |

### Performance Highlights

1. **GraphRAG is consistently faster** (1.7x on average)
   - BM25 range: 3.15s - 49.52s (high variance, some timeouts)
   - GraphRAG range: 2.54s - 7.63s (more consistent)

2. **GraphRAG provides richer responses** (~25% longer on average)
   - Includes community context and entity relationships
   - More structured thematic organization

3. **Both systems had 100% success rate**
   - One BM25 query had a timeout but still returned (49.52s for "soil health")
   - GraphRAG handled all queries under 8 seconds

---

## Strengths of GraphRAG

### 1. Community-Level Context
GraphRAG successfully matches queries to relevant community clusters:
- "Regenerative Soil Health" (6 matches) - correctly triggered for soil/biochar questions
- "Bioregional Regeneration Finance" (3 matches) - for finance/BioFi questions
- "Eco-Spiritual Wisdom" (3 matches) - for spirituality questions
- "Indigenous Sovereignty & Ecology" - for indigenous perspectives

### 2. Thematic Organization
The DRIFT mode provides both:
- High-level thematic context from communities
- Specific entity details and relationships

### 3. Consistent Performance
- No timeouts or extreme latencies
- More predictable response times

---

## Areas for Improvement

### 1. Entity Extraction Noise
The entity extraction is picking up some false positives:
- `O` (PERSON) - 20 matches - likely a bug in alias matching
- `CU`, `pi`, `OM` - single/double letter matches that aren't meaningful
- These short aliases should be filtered or weighted lower

**Recommendation:** Add minimum alias length filter (e.g., 3+ characters) or implement word boundary matching.

### 2. Community Naming Issues
Some community names are too generic or confusing:
- "Sugar-Free Truffle Ingredients" for Savory Institute query (wrong context)
- "Israelite High Priest Aaron Cluster" for Aaron Perry query (same name collision)

**Recommendation:** Review community summaries and ensure cluster titles are topic-appropriate.

### 3. Relationship Type Diversity
All 330 relationships show as `RELATED_TO` - the specific relationship types (FOUNDED, WORKS_FOR, etc.) aren't being surfaced.

**Recommendation:** Include relationship type in the output, not just generic "RELATED_TO".

---

## Category-Specific Observations

### Identity Questions (Who/What is X?)
- **GraphRAG Local mode** worked well for specific entities like Aaron Perry, Brigitte Mars
- Correctly identified key entities like "Aaron William Perry", "Vandana Shiva", "Brigitte Mars"
- Community matching sometimes confused by name collisions

### Concept Questions
- **GraphRAG DRIFT mode** excelled here
- Good community matching: "Bioregional Regeneration Finance" for BioFi, "Regenerative Soil Health" for soil questions
- Provided broader thematic context than BM25

### Practical Questions (How to...)
- Both systems provided useful practical guidance
- GraphRAG added relevant communities: "Urban Community Composting", "Regenerative Soil Health"
- BM25 provided more specific episode citations

### Broad Thematic Questions
- **GraphRAG Global mode** was tested but performed similarly to DRIFT
- Good community matches: "Global Catastrophe Scenarios", "Eco-Spiritual Wisdom"
- Both systems handled abstract questions well

### Episode Recommendations
- **BM25 performed better** for specific episode recommendations
- GraphRAG identified relevant themes but was less precise on episode numbers
- Both mentioned Episode 113 for permaculture (Stephen Brooks)

### Entity-Specific Questions
- GraphRAG correctly extracted key entities: Brigitte Mars, Savory Institute, Vandana Shiva
- Provided entity descriptions and relationships
- BM25 provided more specific episode citations

---

## Recommendations for Next Steps

### High Priority
1. **Fix short alias matching** - Filter out 1-2 character entity aliases
2. **Review problematic community names** - Ensure cluster titles reflect actual topics
3. **Surface relationship types** - Show FOUNDED, ADVOCATES_FOR, etc. instead of generic RELATED_TO

### Medium Priority
4. **Improve episode citation in GraphRAG** - Better link entities to source episodes
5. **Add relevance scoring display** - Show why communities/entities were matched
6. **Implement hybrid mode** - Combine BM25 episode citations with GraphRAG thematic context

### Future Enhancements
7. **Community summary refinement** - Regenerate with better prompts
8. **Entity embedding search** - Add semantic matching for entities (currently lexicon-only)
9. **A/B testing infrastructure** - Track which system users prefer

---

## Conclusion

The GraphRAG implementation is **production-ready** for experimental comparison. Key strengths include:
- Faster, more consistent response times
- Rich thematic context from community summaries
- Entity relationship surfacing

Main areas to improve:
- Entity extraction noise (short aliases)
- Community name quality
- Episode-level citation precision

The side-by-side comparison at https://gaiaai.xyz/YonEarth/graphrag/ will help gather user feedback to guide further improvements.
