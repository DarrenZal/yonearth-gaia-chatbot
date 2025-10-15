# 🧠 Knowledge Graph System Documentation

## 🚨 **CURRENT SYSTEM: ACE Framework (October 2025)**

**The YonEarth Knowledge Graph now uses the ACE (Agentic Context Engineering) Framework for autonomous quality improvement.**

### Active Systems

#### 📚 **Book Extraction: ACE V7 (Meta-ACE Enhanced)** ⭐ **CURRENT**
- **Status**: V7 extraction running (October 12, 2025)
- **Quality**: Targeting <5% issues (A- grade)
- **Features**: Enhanced praise quote detection, multi-pass pronoun resolution, vague entity blocking
- **See**: [ACE_CYCLE_1_COMPLETE.md](ACE_CYCLE_1_COMPLETE.md) for full system details

#### 🎙️ **Episode Extraction: v3.2.2 (Production)** ✅ **COMPLETE**
- **Status**: All 172 episodes extracted (October 12, 2025)
- **Quality**: 93.1% high confidence relationships
- **Data**: 45,478 relationships across all episodes

---

## 📖 Quick Links

### ACE Framework (Book Extraction)
- **[ACE_CYCLE_1_COMPLETE.md](ACE_CYCLE_1_COMPLETE.md)** - ACE Cycle 1 summary and results
- **[ACE_META_TUNING_RECOMMENDATIONS.md](ACE_META_TUNING_RECOMMENDATIONS.md)** - Meta-ACE manual review findings
- **[ACE_KG_EXTRACTION_VISION.md](ACE_KG_EXTRACTION_VISION.md)** - ACE framework vision

### Episode Extraction (v3.2.2)
- Scroll down for full v3.2.2 documentation
- See archived docs in `../archive/knowledge_graph/pre_ace/` for historical implementations

---

## Overview (Episode Extraction v3.2.2)

The YonEarth Knowledge Graph extraction system transforms **172 podcast episodes** into a structured, queryable knowledge base using a production-ready **three-stage extraction pipeline**.

### What It Does

- **Extracts** relationships between people, organizations, places, concepts, and practices
- **Validates** using dual-signal analysis (text clarity + world knowledge)
- **Links** every fact to exact audio timestamps (word-level precision)
- **Calibrates** confidence scores for reliability (when it says 80%, it's actually right 80% of the time)
- **Prevents** duplicates through canonicalization and stable claim UIDs

### Episode Extraction Status

✅ **v3.2.2 COMPLETE - Full Production Extraction** (October 12, 2025)

**Extraction Complete**:
- **172/172 episodes** extracted successfully (100% coverage)
- **45,478 total relationships** across all episodes
- **93.1% high confidence** (42,356 relationships with p_true ≥ 0.75)
- **43.6 hours** total extraction time (two runs combined)
- **Zero failures** after applying critical bug fixes

**Performance**:
- **Coverage**: 3.6× improvement over baseline (347.6% on test data)
- **Quality**: 93.1% high confidence relationships (p_true ≥ 0.75)
- **Cost**: ~$6 for full 172-episode extraction
- **Speed**: ~3 hours per run with batching and caching

---

## 📖 Documentation Guide

### 🚀 Getting Started

**New to the system? Start here:**

1. **[KG_MASTER_GUIDE_V3.md](KG_MASTER_GUIDE_V3.md)** ⭐ **START HERE**
   - Complete v3.2.2 architecture and implementation
   - Three-stage extraction pipeline (Extract → Type Validate → Score)
   - Production schema with evidence tracking
   - All critical fixes and robustness features
   - Code examples and usage patterns
   - **Status**: Production-ready (pending Go/No-Go checklist)

2. **[KG_V3_2_2_IMPLEMENTATION_GUIDE.md](KG_V3_2_2_IMPLEMENTATION_GUIDE.md)** 🎯 **QUICK START**
   - What's new in v3.2.2 vs previous versions
   - Usage instructions and examples
   - Performance comparison tables
   - Migration path from previous implementations
   - Expected results and output format

3. **[KG_V3_2_2_MIGRATION_SUMMARY.md](KG_V3_2_2_MIGRATION_SUMMARY.md)** 📊 **CONTEXT**
   - Why we updated from batched two-pass test
   - Test results that led to v3.2.2 (347.6% coverage improvement)
   - Decision matrix: when to migrate
   - What comes next (refinement, database, audio timestamps)

### 📋 Implementation & Deployment

**Ready to deploy? Follow these:**

4. **[KG_IMPLEMENTATION_CHECKLIST.md](KG_IMPLEMENTATION_CHECKLIST.md)** ✅ **DEPLOYMENT CHECKLIST**
   - Step-by-step implementation tasks
   - v3.2.2 release blocker fixes (all completed)
   - Go/No-Go checklist for production deployment
   - Acceptance tests (AT-01 through AT-05)
   - Timeline and success metrics

5. **[TYPE_CHECKING_STRATEGY.md](TYPE_CHECKING_STRATEGY.md)** 🔍 **TYPE VALIDATION**
   - Multi-source type resolution (GeoNames + Wikidata + Local + LLM)
   - Soft validation approach (prevents 30-40% data loss)
   - SHACL-lite domain/range rules
   - Coverage analysis and implementation guide

### 🎨 System Design & Philosophy

**Want to understand WHY it's designed this way?**

6. **[LEARNING_SYSTEM_ARCHITECTURE.md](LEARNING_SYSTEM_ARCHITECTURE.md)** 💡 **DESIGN PHILOSOPHY**
   - What can/cannot be learned from corrections
   - 4 types of errors and their solutions
   - Why type checking vs fact checking
   - Dual-signal extraction innovation
   - Pattern prior learning approach

7. **[KNOWLEDGE_GRAPH_ARCHITECTURE.md](KNOWLEDGE_GRAPH_ARCHITECTURE.md)** 🏗️ **ORIGINAL DESIGN**
   - Historical context and evolution
   - Original system architecture
   - Background on design decisions
   - Pre-v3 implementation history

8. **[EMERGENT_ONTOLOGY.md](EMERGENT_ONTOLOGY.md)** 🌿 **DOMAIN KNOWLEDGE**
   - YonEarth podcast domain ontology
   - Entity types and relationship types
   - Common patterns in sustainability/regenerative content
   - Vocabulary and taxonomies

### 🔬 Future Enhancements

**After v3.2.2 is deployed and stable:**

9. **[KG_POST_EXTRACTION_REFINEMENT.md](KG_POST_EXTRACTION_REFINEMENT.md)** 🚀 **REFINEMENT PHASE**
   - Neural-symbolic validation (10-20% accuracy improvement)
   - Entity resolution with Splink (5-10 seconds for 11K+ entities)
   - SHACL validation with pySHACL (catches errors instantly)
   - Embedding validation with PyKEEN (15 min initial, 2 min incremental)
   - Active learning (65% reduction in human annotation)
   - **Timeline**: 3-5 days implementation
   - **Status**: Planned for AFTER v3.2.2 deployment

---

## 🎯 Quick Lookup by Task

### "I want to understand the system"
→ Read: **[KG_MASTER_GUIDE_V3.md](KG_MASTER_GUIDE_V3.md)**

### "I want to run extraction on test episodes"
→ Read: **[KG_V3_2_2_IMPLEMENTATION_GUIDE.md](KG_V3_2_2_IMPLEMENTATION_GUIDE.md)**
→ Run: `python3 scripts/extract_kg_v3_2_2.py`

### "I want to deploy to full 172 episodes"
→ Read: **[KG_IMPLEMENTATION_CHECKLIST.md](KG_IMPLEMENTATION_CHECKLIST.md)**
→ Verify: Run Go/No-Go checklist
→ Deploy: Scale extraction script to all episodes

### "I want to understand why it's designed this way"
→ Read: **[LEARNING_SYSTEM_ARCHITECTURE.md](LEARNING_SYSTEM_ARCHITECTURE.md)**

### "I want to implement type validation"
→ Read: **[TYPE_CHECKING_STRATEGY.md](TYPE_CHECKING_STRATEGY.md)**
→ Integrate: GeoNames API for geographic entities
→ Integrate: Wikidata for organizations/people

### "I want to improve accuracy after extraction"
→ Read: **[KG_POST_EXTRACTION_REFINEMENT.md](KG_POST_EXTRACTION_REFINEMENT.md)**
→ Install: `pip install splink pyshacl pykeen`
→ Timeline: 3-5 days for full refinement system

### "I want to understand the domain ontology"
→ Read: **[EMERGENT_ONTOLOGY.md](EMERGENT_ONTOLOGY.md)**

### "I migrated from previous test, what changed?"
→ Read: **[KG_V3_2_2_MIGRATION_SUMMARY.md](KG_V3_2_2_MIGRATION_SUMMARY.md)**

---

## 🏗️ Three-Stage Architecture (v3.2.2)

```
┌─────────────────────────────────────────────────────────────┐
│                    PASS 1: EXTRACTION                       │
│                                                             │
│  Input:   Podcast transcript chunks (800 tokens)            │
│  Model:   gpt-4o-mini                                       │
│  Output:  ~250 candidate relationships per episode          │
│  Goal:    High recall - extract EVERYTHING                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│            TYPE VALIDATION QUICK PASS (NEW!)                │
│                                                             │
│  Uses:    Cached GeoNames/Wikidata/Local lookups           │
│  Logic:   Soft validation (only filter KNOWN violations)   │
│  Output:  ~220 valid candidates (filters 10-20% nonsense)  │
│  Saves:   API costs by filtering before expensive Pass 2   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│          PASS 2: BATCHED DUAL-SIGNAL EVALUATION             │
│                                                             │
│  Batch:   50 relationships per API call (NDJSON format)    │
│  Model:   gpt-4o-mini                                       │
│  Signals: • Text confidence (reading comprehension)         │
│           • Knowledge plausibility (world knowledge)        │
│  Output:  Calibrated p_true scores + conflict detection    │
│  Result:  ~220 validated relationships per episode          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   POST-PROCESSING                           │
│                                                             │
│  • Canonicalize entities (Y on Earth = YonEarth)           │
│  • Generate stable claim UIDs                               │
│  • Extract evidence spans with SHA256                       │
│  • Compute calibrated p_true                                │
│  • Preserve surface forms for review                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Performance Metrics

### Coverage (Tested on 10 Episodes)

| Approach | Relationships/Episode | vs Baseline | Winner |
|----------|---------------------|-------------|--------|
| Baseline (v2) | 65 | 100% | - |
| Single-pass | 71 | 109% | ❌ |
| Dual-signal | 94 | 145% | ❌ |
| Two-pass | 198 | 305% | ✅ |
| **Batched Two-Pass (v3.2.2)** | **~220** | **~340%** | **🥇** |

### Quality Distribution

- **High confidence** (p_true ≥ 0.75): 85%+
- **Medium confidence** (0.5 ≤ p_true < 0.75): 10-12%
- **Low confidence** (p_true < 0.5): 3-5%
- **Conflicts detected**: ~5% (flagged for review)

### Cost & Speed

- **Total cost**: ~$6 for 172 episodes
- **Processing time**: ~3 hours with batching and caching
- **API calls**: ~60 per episode (reduced from ~70 by type validation)
- **Cache hit rate**: 30%+ on re-runs

---

## 🔑 Key Innovations

### 1. **Dual-Signal Extraction**
Separates text comprehension from world knowledge to detect conflicts early:
- Text signal: "How clearly does the text state this?"
- Knowledge signal: "Is this plausible based on what I know?"
- Conflict detection: Flags when signals disagree

### 2. **Calibrated Confidence**
Not just a score - actually reliable:
- `p_true = 0.8` means the system is right 80% of the time
- Logistic regression with fixed coefficients
- Trained on ~150 labeled edges
- ECE (Expected Calibration Error) ≤ 0.07

### 3. **Stable Claim UIDs**
Facts don't duplicate when you improve prompts:
- Based on canonicalized entities + evidence hash + doc SHA256
- **Doesn't** include prompt_version or model info
- Re-runs with updated prompts update existing facts instead of creating duplicates

### 4. **Evidence Tracking**
Every fact linked to exact source:
- SHA256 of transcript (detects changes)
- Character offsets for text spans
- Surface forms preserved (original mentions)
- Ready for audio timestamp mapping

### 5. **Soft Type Validation**
Only filters KNOWN violations, not unknowns:
- Previous systems lost 30-40% of valid relationships
- Now only filters when BOTH types are known AND violate rules
- Unknown entities pass through for later enrichment

### 6. **NDJSON Robustness**
Partial batch failures don't kill extraction:
- One JSON object per line
- Parse errors on one line don't affect others
- Uses candidate_uid for result joining (not list order)
- Recovers gracefully from API hiccups

---

## 🚦 Implementation Status

### ✅ Complete (v3.2.2)

- [x] Three-stage extraction pipeline
- [x] Type validation quick pass
- [x] Batched dual-signal evaluation
- [x] Calibrated confidence scoring
- [x] Evidence tracking with SHA256
- [x] Stable claim UIDs
- [x] Canonicalization
- [x] Surface form preservation
- [x] NDJSON robustness
- [x] Scorer-aware caching
- [x] All critical bug fixes
- [x] **Full 172-episode extraction (October 2025)**
- [x] **Critical bug fixes for candidate_uid mismatch and token limits**
- [x] **Unified knowledge graph dataset (45,478 relationships)**

### 📋 Planned (Post-Extraction)

- [ ] PostgreSQL database integration
- [ ] Audio timestamp mapping
- [ ] Web visualization with version dropdown
- [ ] Post-extraction refinement phase
- [ ] Entity resolution with Splink
- [ ] SHACL validation with pySHACL
- [ ] Embedding validation with PyKEEN

---

## 🛠️ Tools & Technologies

### Current Stack

- **Extraction**: OpenAI gpt-4o-mini (cost-effective, fast)
- **Schema Validation**: Pydantic (structured outputs, 100% valid JSON)
- **Batching**: NDJSON format (robustness)
- **Type Resolution**: GeoNames + Wikidata + Local cache (planned integration)
- **Evidence Tracking**: SHA256 hashing

### Planned Stack (Refinement Phase)

- **Entity Resolution**: Splink with DuckDB backend
- **Logical Validation**: pySHACL for constraint checking
- **Embedding Validation**: PyKEEN with RotatE model
- **Confidence Calibration**: Temperature scaling
- **Active Learning**: Uncertainty sampling

---

## 📁 Output Structure

### Per-Episode Results

```json
{
  "episode": 10,
  "version": "v3.2.2",
  "doc_sha256": "abc123...",

  "pass1_candidates": 250,
  "type_valid": 220,
  "pass2_evaluated": 220,

  "high_confidence_count": 185,
  "medium_confidence_count": 25,
  "low_confidence_count": 10,

  "conflicts_detected": 5,
  "cache_hit_rate": 0.35,

  "relationships": [
    {
      "source": "Y on Earth",
      "relationship": "founded_by",
      "target": "Aaron William Perry",
      "source_type": "Org",
      "target_type": "Person",

      "text_confidence": 0.95,
      "knowledge_plausibility": 0.90,
      "pattern_prior": 0.5,
      "p_true": 0.89,

      "signals_conflict": false,
      "claim_uid": "def456...",

      "evidence": {
        "doc_sha256": "abc123...",
        "source_surface": "YonEarth",
        "target_surface": "Aaron Perry",
        "window_text": "...Aaron William Perry founded YonEarth..."
      },

      "flags": {}
    }
  ]
}
```

---

## 🎓 Learning & Improvement

### What the System Learns

✅ **Pattern Priors**: Frequency of relationship patterns from existing graph
✅ **Type Mappings**: Entity types from GeoNames/Wikidata/corrections
✅ **Alias Resolution**: "Y on Earth" = "YonEarth" = "yon earth"
✅ **Calibration**: Adjusting confidence scores based on labeled data

### What It Doesn't Learn (By Design)

❌ **Facts**: Doesn't learn "Boulder is in Colorado" - that's in GeoNames
❌ **Arbitrary Rules**: Uses structured SHACL constraints instead
❌ **Domain Ontology**: Uses predefined YonEarth taxonomy

**Why?** See [LEARNING_SYSTEM_ARCHITECTURE.md](LEARNING_SYSTEM_ARCHITECTURE.md) for detailed explanation.

---

## 🔗 Related Documentation

### General Project Docs

- **[../README.md](../README.md)** - Project overview and quick start
- **[../CONTENT_PROCESSING_PIPELINE.md](../CONTENT_PROCESSING_PIPELINE.md)** - How to process episodes and books
- **[../TRANSCRIPTION_SETUP.md](../TRANSCRIPTION_SETUP.md)** - Word-level timestamp setup
- **[../FEATURES_AND_USAGE.md](../FEATURES_AND_USAGE.md)** - User-facing features

### Research & Background

Located in `docs/archive/`:
- **KG_Research_1.md** - Technical implementation blueprint (28K words)
- **KG_Research_2.md** - Academic framework with 79 citations (74K words)
- **KG_Research_3.md** - Performance metrics and tools (49K words)
- **KG_REFINEMENT_SYNTHESIS.md** - Research synthesis

---

## 📞 Support & Questions

### Common Questions

**Q: Why three stages instead of two?**
A: Type validation filters 10-20% of nonsense BEFORE expensive Pass 2, saving API costs and improving quality.

**Q: What's the difference between `overall_confidence` and `p_true`?**
A: `overall_confidence` was uncalibrated. `p_true` is actually reliable - when it says 0.8, it's right 80% of the time.

**Q: How do I prevent duplicates on re-runs?**
A: Stable claim UIDs ensure facts don't duplicate when you update prompts or re-run extraction.

**Q: Can I update transcripts after extraction?**
A: Yes! SHA256 tracking detects changes and marks evidence as "stale" for review.

**Q: When should I do the refinement phase?**
A: AFTER v3.2.2 is deployed and stable. Refinement adds 10-20% accuracy improvement through entity resolution, SHACL validation, and embeddings.

---

## 📊 Extraction History

### Full Production Extraction (October 2025)

**Run 1: Initial Extraction** (October 10-11, 2025)
- Episodes: 156 successful, 16 failed
- Relationships: 40,686 (93.0% high confidence)
- Time: 39.7 hours
- Issues: candidate_uid mismatch, token limit exceeded

**Run 2: Retry Extraction** (October 12, 2025)
- Episodes: 16 previously failed, all successful
- Relationships: 4,792 (93.9% high confidence)
- Time: 3.8 hours
- Fixes applied: candidate_uid handling, max_completion_tokens limit

**Final Dataset**:
- Total episodes: 172/172 (100% coverage)
- Total relationships: 45,478
- High confidence: 42,356 (93.1%)
- Total time: 43.6 hours
- Location: `/data/knowledge_graph_v3_2_2/`

---

## 📝 Version History

- **v3.2.2** (October 2025) - Production-ready release + Full extraction
  - All critical bugs fixed
  - Three-stage architecture
  - Evidence tracking with SHA256
  - Stable claim UIDs
  - Calibrated confidence
  - **COMPLETE: 172 episodes, 45,478 relationships**

- **v3.2.1** (October 2025) - Bug fixes and robustness
  - Mutable default fix
  - Soft type validation
  - Scorer-aware caching

- **v3.2** (October 2025) - Data loss prevention
  - Canonicalization before UID
  - Missing vs invalid data handling

- **v3.1** (October 2025) - Production hardening
  - Type validation between passes
  - Admin-aware geo validation

- **v3.0** (October 2025) - Two-pass architecture
  - Dual-signal extraction
  - Batched evaluation

---

**Last Updated**: October 12, 2025
**Current Version**: v3.2.2 (Complete - 172 episodes extracted)
**Maintainer**: YonEarth Team

For questions or contributions, see the main project README.
