# Repository Cleanup & Organization Plan

**Date**: October 12, 2025
**Context**: ACE Cycle 1 Complete, V7 extraction running

---

## 🎯 Cleanup Objectives

1. Archive old extraction scripts (V3, V4, V5)
2. Organize ACE documentation
3. Archive pre-ACE architectural docs
4. Clean up output directories
5. Update main documentation

---

## 📂 Scripts Directory

### Archive to `scripts/archive/kg_extraction/`:
- ✅ `extract_kg_v3_2_2.py` (old episode extraction)
- ✅ `extract_kg_v3_2_2_book_v4_comprehensive.py` (V4 book extraction)
- ✅ `extract_kg_v5_book.py` (V5 - superseded by V6)

### Keep in `scripts/`:
- ✅ `extract_kg_v6_book.py` (V6 - production baseline)
- ✅ `extract_kg_v7_book.py` (V7 - current production)
- ✅ `run_reflector_on_v5.py` (historical record)
- ✅ `run_reflector_on_v6.py` (used for V6 analysis)
- ✅ `run_ace_cycle.py` (if exists - ACE orchestration)

### New Script to Create:
- 🆕 `run_reflector_on_v7.py` (for V7 analysis)

---

## 📚 Documentation Cleanup

### ACE Documentation (Keep in `docs/knowledge_graph/`):
**Current Status & Results:**
- ✅ `ACE_CURRENT_STATUS.md` - Latest ACE status
- ✅ `ACE_CYCLE_1_COMPLETE.md` - Cycle 1 completion summary
- ✅ `ACE_CYCLE_1_V6_RESULTS.md` - V6 detailed results
- ✅ `ACE_META_TUNING_RECOMMENDATIONS.md` - Meta-ACE manual review findings
- ✅ `V6_ANALYSIS_RESULTS.md` - V6 Reflector analysis

**Design & Vision:**
- ✅ `ACE_KG_EXTRACTION_VISION.md` - ACE framework vision

**Version Implementation Plans:**
- ✅ `V5_IMPLEMENTATION_PLAN.md` - V5 design (historical)

**Version Quality Reports:**
- ✅ `V4_EXTRACTION_QUALITY_ISSUES_REPORT.md` - V4 baseline
- ✅ `V4_COMPLETE_COMPARISON.md` - V4 comparison
- ✅ `V4_ADDITIONAL_QUALITY_ISSUES.md` - V4 extra findings

### Archive to `docs/archive/knowledge_graph/pre_ace/`:
**Pre-ACE Architecture Docs (no longer current):**
- 📦 `KNOWLEDGE_GRAPH_ARCHITECTURE.md`
- 📦 `EXTRACTION_PHILOSOPHY.md`
- 📦 `EMERGENT_ONTOLOGY.md`
- 📦 `TYPE_CHECKING_STRATEGY.md`
- 📦 `LEARNING_SYSTEM_ARCHITECTURE.md`
- 📦 `ENTITY_RESOLUTION_GUIDE.md`
- 📦 `ENTITY_RESOLUTION_COMPREHENSIVE_GUIDE.md`
- 📦 `COMPLEX_CLAIMS_AND_QUANTITATIVE_RELATIONSHIPS.md`

**Old Implementation Guides (superseded):**
- 📦 `KG_MASTER_GUIDE_V3.md`
- 📦 `KG_V3_2_2_IMPLEMENTATION_GUIDE.md`
- 📦 `KG_V3_2_2_MIGRATION_SUMMARY.md`
- 📦 `KG_IMPLEMENTATION_CHECKLIST.md`
- 📦 `KG_POST_EXTRACTION_REFINEMENT.md`

### Update:
- 🔄 `README.md` - Add ACE system overview, link to ACE docs

---

## 🗂️ Playbook Outputs

### Archive to `kg_extraction_playbook/output/archive/`:
- 📦 `v5/` → `archive/v5/` (superseded by V6)

### Keep:
- ✅ `v6/` - Production baseline
- ✅ `v7/` - Current production (once complete)

### Analysis Reports:
- ✅ Keep all in `kg_extraction_playbook/analysis_reports/` (historical record)

---

## 📋 New Documentation to Create

### 1. **ACE_CYCLE_1_V7_RESULTS.md**
Complete V7 analysis with:
- V7 vs V6 comparison
- Meta-ACE fixes validation
- <5% target achievement
- Production readiness assessment

### 2. **ACE_FRAMEWORK_GUIDE.md** (Comprehensive)
Master guide explaining:
- ACE system architecture
- How to apply ACE to new content
- Reflector agent usage
- Curator improvements
- Meta-validation protocol

### 3. **KG_PRODUCTION_STATUS.md**
Current production status:
- Which version is production (V6 or V7)
- Quality metrics
- Known limitations
- Next steps

### 4. Update **docs/knowledge_graph/README.md**
- Link to ACE framework
- Link to current production status
- Archived docs reference

---

## 🧹 Other Cleanup

### Root Directory:
- 📦 Move `kg_extraction_playbook/` contents if not actively used
- ✅ Keep `src/ace_kg/` (ACE framework code)
- ✅ Keep `src/ace/` (if exists)

### Data Directories:
- ✅ Keep `data/knowledge_graph_v3_2_2/` (historical)
- ✅ Keep `data/playbook/` (if exists)

---

## ✅ Execution Order

1. Create archive directories
2. Move scripts to archive
3. Move docs to archive
4. Move v5 output to archive
5. Create new V7 analysis script
6. Create new documentation
7. Update README files
8. Commit with message: "♻️ Cleanup: Archive pre-ACE artifacts, organize ACE Cycle 1 docs"

---

## 🎯 End State

**Active Files:**
- Scripts: V6, V7 extractors + V5/V6/V7 reflector runners
- Docs: ACE framework + Current results + Framework guide
- Outputs: V6 baseline + V7 current

**Archived:**
- Scripts: V3, V4, V5 extractors
- Docs: Pre-ACE architecture, old implementation guides
- Outputs: V5 extraction results

**New:**
- ACE Framework Guide (comprehensive)
- V7 Results Doc
- Production Status Doc
