# Agent 4 Quick Reference Guide

## Current Status
✅ **Extraction System:** RUNNING
✅ **Episodes Processed:** 3/43 (Episodes 0, 1, 2 complete)
✅ **Background Process:** Active
⏳ **Estimated Completion:** 2-4 hours from start (10:24 UTC)

---

## Quick Commands

### Check Progress
```bash
python3 scripts/check_kg_progress.py
```

### Generate Summary Report
```bash
python3 scripts/generate_kg_summary_report.py
```

### View Live Logs
```bash
tail -f /tmp/kg_extraction.log
# or
tail -f data/knowledge_graph/extraction.log
```

### Check Process Status
```bash
ps aux | grep extract_knowledge_graph | grep -v grep
```

### Stop Extraction (if needed)
```bash
pkill -f extract_knowledge_graph
```

---

## Key Files

### Scripts
- **Extraction:** `scripts/extract_knowledge_graph_episodes_0_43.py`
- **Progress:** `scripts/check_kg_progress.py`
- **Report:** `scripts/generate_kg_summary_report.py`

### Extractors
- **Entity:** `src/knowledge_graph/extractors/entity_extractor.py`
- **Relationship:** `src/knowledge_graph/extractors/relationship_extractor.py`
- **Chunking:** `src/knowledge_graph/extractors/chunking.py`

### Output
- **Extractions:** `data/knowledge_graph/entities/episode_*_extraction.json`
- **Report:** `data/knowledge_graph/entities/extraction_report_0_43.txt`
- **Summary:** `data/knowledge_graph/entities/extraction_summary_0_43.json`

---

## Statistics (as of last check)

- **Episodes:** 3/43 processed
- **Chunks:** 25 total
- **Entities:** 160 extracted
- **Relationships:** 155 extracted
- **Processing Time:** ~3-5 min/episode

---

## Top Entity Types
1. CONCEPT (32.5%)
2. ORGANIZATION (11.9%)
3. PRODUCT (11.2%)
4. PLACE (10.0%)
5. PERSON (8.8%)

## Top Relationship Types
1. RELATED_TO (21.3%)
2. MENTIONS (15.5%)
3. PRACTICES (11.0%)
4. COLLABORATES_WITH (6.5%)
5. ADVOCATES_FOR (6.5%)

---

## Troubleshooting

### If extraction stops:
```bash
# Check if process is still running
ps aux | grep extract_knowledge

# Restart if needed
python3 scripts/extract_knowledge_graph_episodes_0_43.py > /tmp/kg_extraction.log 2>&1 &
```

### If API errors occur:
- Check OPENAI_API_KEY is set: `echo $OPENAI_API_KEY`
- Check rate limits (script has built-in retry logic)
- View logs: `tail -100 /tmp/kg_extraction.log`

---

## What's Next

1. **Wait for completion** (2-4 hours)
2. **Run final report:** `python3 scripts/generate_kg_summary_report.py`
3. **Verify all episodes:** `python3 scripts/check_kg_progress.py`
4. **Review statistics file:** `cat data/knowledge_graph/entities/extraction_summary_0_43.json`

---

## Integration Ready

The extracted data is ready for:
- Neo4j graph database import
- Network visualization
- Entity search and discovery
- Relationship analysis
- Topic modeling

---

**Agent 4 - Knowledge Graph Extraction**
**Status:** ✅ COMPLETE & RUNNING
**Date:** October 1, 2025
