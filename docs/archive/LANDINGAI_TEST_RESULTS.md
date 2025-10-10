# LandingAI Knowledge Graph Extraction Test Results

**Date:** October 10, 2025
**Test Episode:** Episode 10 - Lauren Tucker - Kiss the Ground
**Comparison:** LandingAI ADE vs OpenAI gpt-4o-mini Two-Pass Batched

---

## Executive Summary

We tested **LandingAI's Agentic Document Extraction (ADE)** API as an alternative to OpenAI for structured knowledge graph extraction from podcast transcripts. The test extracted entities and relationships from Episode 10 and compared results against the existing OpenAI two-pass batched approach.

### Key Findings

| Metric | LandingAI | OpenAI (Two-Pass) | Winner |
|--------|-----------|-------------------|--------|
| **Relationships Extracted** | 70 | 230 | ✅ OpenAI (3.3x more) |
| **Unique Entities** | 86 | N/A | ✅ LandingAI (structured) |
| **Coverage vs OpenAI** | 3.5% | 100% | ✅ OpenAI |
| **Average Confidence** | 0.99 | 0.90 | ✅ LandingAI (higher) |
| **Entity Type Classification** | ✅ Yes (8 types) | ✅ Yes | 🟰 Tie |
| **Conflict Detection** | ❌ No | ✅ Yes (5 detected) | ✅ OpenAI |
| **Dual-Signal Validation** | ❌ No | ✅ Yes | ✅ OpenAI |

**Conclusion:** OpenAI's two-pass approach extracts **3.3x more relationships** and provides better coverage and quality validation. However, LandingAI offers highly structured entity metadata and may be useful for specific use cases.

---

## Detailed Results

### Extraction Statistics

```
LandingAI:
  Model: landingai-ade-dpt-2
  Approach: landingai-schema-extraction
  Chunks Processed: 9
  Unique Entities: 86
  Total Relationships: 71
  Average Confidence: 0.99

OpenAI:
  Model: gpt-4o-mini
  Approach: two-pass-batched
  Pass 1 Relationships: 233
  Pass 2 Evaluated: 233
  Conflicts Detected: 5
  Type Violations: 1
  Average Confidence: 0.90
```

### Coverage Analysis

- **Shared relationships (both):** 8 pairs (3.5% overlap)
- **LandingAI only:** 62 unique pairs
- **OpenAI only:** 222 unique pairs

This low overlap suggests the approaches extract **different granularities** of relationships. OpenAI captures more implicit and inferred relationships, while LandingAI focuses on more explicit, high-confidence facts.

### Example Shared Relationships (Found by Both)

1. ✅ Lauren Tucker --[graduated from]--> American University
2. ✅ Lauren Tucker --[grew up in]--> West Virginia
3. ✅ Lauren Tucker --[is the executive director of]--> Kiss the Ground
4. ✅ Kiss the Ground --[partnered with]--> Lifelab
5. ✅ Ryland's father --[is one of the founders of]--> Kepaic Gratitude

### Example LandingAI-Only Relationships

1. 🔵 agriculture --[can be a significant emitter of]--> greenhouse gases
2. 🔵 community garden project --[is located in]--> Venice, California
3. 🔵 companies --[purchase]--> CSA shares
4. 🔵 dry-formed prickly pear --[is made from]--> Santa Barbara

### Example OpenAI-Only Relationships

1. 🟢 agriculture --[has put]--> carbon in the atmosphere
2. 🟢 animals and trees --[should be integrated into]--> farming practices
3. 🟢 beaver --[has a role in]--> ecosystem
4. 🟢 humans --[are stewards of]--> places

---

## Entity Type Distribution (LandingAI)

LandingAI provides structured entity type classification:

| Entity Type | Count | Examples |
|-------------|-------|----------|
| **CONCEPT** | 21 | carbon, soil, regeneration of ecosystems |
| **ORG** | 15 | Kiss the Ground, American University, Lifelab |
| **PRACTICE** | 13 | regenerative agriculture, not tilling, soil stewardship |
| **PERSON** | 11 | Lauren Tucker, Ryland Englehart, Wendell Berry |
| **PLACE** | 11 | West Virginia, Africa, Venice, Santa Barbara |
| **PRODUCT** | 8 | purchasing guide, CSA shares, The Soil Story |
| **TECHNOLOGY** | 4 | kisstheground.com, Instagram |
| **EVENT** | 3 | Katrina, carbon farming movement |

This structured classification is a **unique advantage** of LandingAI's schema-driven approach.

---

## Strengths & Weaknesses

### LandingAI Strengths ✅

1. **Structured Entity Metadata**: Rich entity descriptions with type classifications
2. **Schema-Driven**: Consistent JSON structure guaranteed by schema validation
3. **High Confidence**: Average confidence of 0.99 (vs 0.90 for OpenAI)
4. **Simplicity**: Single API call per chunk (no two-pass needed)
5. **Entity-Focused**: Better at extracting and categorizing entities

### LandingAI Weaknesses ❌

1. **Low Coverage**: Only 3.5% overlap with OpenAI (missed 222 relationships)
2. **No Conflict Detection**: Lacks dual-signal validation
3. **Less Comprehensive**: Extracted 3.3x fewer relationships
4. **No Knowledge Grounding**: Cannot validate facts against world knowledge
5. **Rate Limits**: Hit rate limits quickly during testing

### OpenAI Strengths ✅

1. **Comprehensive Coverage**: 3.3x more relationships extracted
2. **Dual-Signal Validation**: Separates text confidence from knowledge plausibility
3. **Conflict Detection**: Identifies contradictions and implausible claims
4. **Type Validation**: Detects entity type constraint violations
5. **Better for Implicit Facts**: Captures inferred and implicit relationships

### OpenAI Weaknesses ❌

1. **Two API Calls**: More expensive (2x API calls per chunk)
2. **Slower**: Pass 1 + Pass 2 takes longer
3. **Less Structured Entities**: Entity metadata less consistent
4. **Lower Confidence**: Average confidence of 0.90 (still good)

---

## Cost & Performance Comparison

### LandingAI

- **API Calls:** 9 chunks × 1 call = **9 calls**
- **Rate Limits:** Hit rate limit after 5-6 calls (strict limits)
- **Required Delay:** 2 seconds between calls
- **Total Time:** ~20 seconds (with delays)
- **Cost:** Unknown (requires LandingAI pricing info)

### OpenAI (Two-Pass Batched)

- **API Calls:** Pass 1 (9 chunks) + Pass 2 (5 batches of 50) = **14 calls**
- **Rate Limits:** More generous (1,200 requests/min for gpt-4o-mini)
- **Required Delay:** 0.05 seconds between calls
- **Total Time:** ~60 seconds
- **Cost:** ~$0.02 for episode 10 (gpt-4o-mini pricing)

---

## Use Case Recommendations

### When to Use LandingAI 🔵

1. **Entity-Focused Applications**: When you need rich entity metadata with type classifications
2. **Structured Data Requirements**: When downstream systems require strict JSON schemas
3. **High-Confidence Facts Only**: When you prefer fewer, more confident relationships
4. **Document-Heavy Workloads**: When extracting from PDFs, forms, or mixed documents (LandingAI's primary use case)

### When to Use OpenAI (Two-Pass) 🟢

1. **Comprehensive Knowledge Graphs**: When you need maximum coverage
2. **Quality Validation**: When you need conflict detection and dual-signal evaluation
3. **Transcript Analysis**: When working with conversational/podcast text (our use case)
4. **Implicit Relationship Extraction**: When you want inferred facts, not just explicit ones

### Hybrid Approach 🟰

**Recommendation:** Use **both** approaches in parallel:

1. **OpenAI for relationships**: Comprehensive coverage with conflict detection
2. **LandingAI for entity enrichment**: Structured entity metadata and descriptions
3. **Merge results**: Combine OpenAI's relationships with LandingAI's entity data

---

## Technical Implementation

### Test Scripts Created

1. **`scripts/test_landingai_extraction.py`**
   - Extracts knowledge graph using LandingAI ADE Extract API
   - Uses JSON schema to define entity/relationship structure
   - Outputs to `data/knowledge_graph_landingai_test/`

2. **`scripts/compare_landingai_vs_openai.py`**
   - Compares LandingAI vs OpenAI extractions
   - Calculates overlap, unique pairs, and coverage metrics
   - Generates detailed comparison report

### LandingAI API Usage

```python
# API endpoint
LANDINGAI_API_URL = "https://api.va.landing.ai/v1/ade/extract"

# Authentication
headers = {"Authorization": f"Bearer {LANDINGAI_API_KEY}"}

# Request format (multipart/form-data)
files = {'markdown': ('chunk.txt', io.StringIO(text), 'text/plain')}
data = {'schema': json.dumps(schema)}  # Schema as JSON string

# Response
response = requests.post(LANDINGAI_API_URL, headers=headers, files=files, data=data)
result = response.json()
extraction = result['extraction']  # Extracted entities and relationships
```

### Key Learnings

1. **Schema must be JSON string**: Pass schema as form data, not as file
2. **Rate limits are strict**: 2-second delay required between calls
3. **Markdown format expected**: Plain text works, but markdown is preferred
4. **Schema validation enforced**: Results guaranteed to match schema structure

---

## Sample Extraction Quality

### High-Quality Extraction (Both Found)

```json
{
  "source": "Lauren Tucker",
  "relationship": "graduated from",
  "target": "American University",
  "context": "She graduated from American University with degree in psychology and international studies.",
  "confidence": 1.0
}
```

✅ **LandingAI:** Found with confidence 1.0
✅ **OpenAI:** Found with overall_confidence 1.0
🎯 **Quality:** Excellent - explicit, factual, high-confidence

### LandingAI Unique Extraction

```json
{
  "source": "community garden project",
  "relationship": "is located in",
  "target": "Venice, California",
  "context": "The community garden project is based on city-owned property in Venice, California.",
  "confidence": 1.0
}
```

🔵 **LandingAI:** Found with confidence 1.0
❌ **OpenAI:** Missed (extracted more granular relationships instead)
🎯 **Quality:** Good - explicit location relationship

### OpenAI Unique Extraction

```json
{
  "source": "humans",
  "relationship": "are stewards of",
  "target": "places",
  "context": "...our role is to be stewards of these places...",
  "text_confidence": 0.85,
  "knowledge_plausibility": 0.95,
  "overall_confidence": 0.90
}
```

❌ **LandingAI:** Missed (implicit, conceptual relationship)
✅ **OpenAI:** Found with dual-signal validation
🎯 **Quality:** Excellent - captures implicit meaning and philosophical concepts

---

## Recommendations

### For Current Production Use

**Continue using OpenAI two-pass batched extraction** for the following reasons:

1. ✅ **3.3x better coverage** (230 vs 70 relationships)
2. ✅ **Dual-signal validation** catches errors and conflicts
3. ✅ **Works well with conversational text** (podcasts)
4. ✅ **More comprehensive** for building knowledge graphs
5. ✅ **Proven results** with existing 172 episodes

### For Future Exploration

**Consider LandingAI for specific use cases:**

1. 📄 **Document extraction** (PDFs, forms, tables) - LandingAI's primary strength
2. 🏷️ **Entity enrichment** - Add structured entity metadata to existing graphs
3. 🔄 **Hybrid pipeline** - Use both APIs and merge results
4. 📊 **Structured data applications** - When strict schemas are required

### Next Steps

1. ✅ **Decision:** Stick with OpenAI two-pass for full 172-episode extraction
2. 📊 **Monitor:** Track LandingAI API improvements and pricing
3. 🔬 **Test:** Try LandingAI on book chapters (more document-like structure)
4. 🤝 **Combine:** Explore merging LandingAI entity data with OpenAI relationships

---

## Files Generated

- **Extraction Results:**
  - `data/knowledge_graph_landingai_test/episode_10_landingai.json` (86 entities, 71 relationships)
  - `data/knowledge_graph_two_pass_batched_test/episode_10_two_pass_batched.json` (233 relationships)

- **Test Scripts:**
  - `scripts/test_landingai_extraction.py` (LandingAI extraction)
  - `scripts/compare_landingai_vs_openai.py` (Comparison analysis)

- **Documentation:**
  - `docs/LANDINGAI_TEST_RESULTS.md` (this document)

---

## Conclusion

**LandingAI's ADE is a powerful tool for structured document extraction, but OpenAI's two-pass approach is better suited for comprehensive knowledge graph extraction from podcast transcripts.**

The 3.3x difference in relationship coverage (230 vs 70) and low overlap (3.5%) demonstrate that OpenAI's approach is more appropriate for our use case. However, LandingAI's structured entity metadata and schema-driven extraction could complement OpenAI in a hybrid pipeline.

**Recommendation:** Continue with **OpenAI two-pass batched extraction** for production knowledge graph generation, and revisit LandingAI for document-heavy tasks or entity enrichment in the future.

---

**Test Completed:** October 10, 2025
**Tested By:** Claude Code + YonEarth Chatbot Team
**Next Action:** Proceed with OpenAI-based full extraction of 172 episodes
