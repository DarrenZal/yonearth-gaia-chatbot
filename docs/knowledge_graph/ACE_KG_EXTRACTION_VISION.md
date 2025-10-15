# ACE for Knowledge Graph Extraction - Vision Document

**Date**: 2025-10-12
**Purpose**: Never-ending improvement of KG extraction through autonomous reflection and curation
**Scope**: Knowledge graph extraction from books (starting with Soil Stewardship Handbook)

---

## 🎯 Vision: Self-Improving Knowledge Graph Extraction

Build a system that **autonomously improves** its knowledge graph extraction capabilities through continuous cycles of:
1. **Extraction** → Generate KG from book
2. **Reflection** → Analyze quality issues
3. **Curation** → Propose fixes to code/prompts
4. **Evolution** → Apply changes and repeat

The system never stops improving until it achieves near-perfect extraction quality.

---

## 🔬 Inspired by Research

Based on **"Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models"** (arXiv:2510.04618v1)

**Key Insight**: A system can improve by treating its entire operational logic as a modifiable "Playbook" that evolves through agent-driven reflection.

**Our Implementation**:
- **Generator**: V5 KG extraction pipeline (all code, prompts, configs)
- **Playbook**: Version-controlled codebase that agents can modify
- **Reflector**: Claude Sonnet 4.5 analyzing extraction quality
- **Curator**: GPT-4o proposing specific code/prompt changes
- **Evolution**: Automated application of approved changes

---

## 📊 Current State: V5 Extraction Quality

From V4 analysis on Soil Stewardship Handbook:
- **Total relationships**: 873
- **Quality issues**: 495 (57%)
- **Critical issues**: 105 (reversed authorship - 12%)
- **High priority**: 347 (pronouns, lists, vague entities - 40%)

**V5 Improvements** (from Implementation Plan):
- Pass 2.5: 7 post-processing modules
- BibliographicCitationParser
- ListSplitter
- PronounResolver
- ContextEnricher
- TitleCompletenessValidator
- PredicateValidator
- FigurativeLanguageFilter

**V5 Target**: <10% quality issues

---

## 🔄 The Never-Ending Improvement Loop

```
┌─────────────────────────────────────────────────────────────┐
│                  NEVER-ENDING ACE LOOP                       │
└─────────────────────────────────────────────────────────────┘

V5 → EXTRACT → REFLECT → CURATE → EVOLVE → V6
                  ↑                            ↓
                  └────────────────────────────┘

V6 → EXTRACT → REFLECT → CURATE → EVOLVE → V7
                  ↑                            ↓
                  └────────────────────────────┘

V7 → EXTRACT → REFLECT → CURATE → EVOLVE → V8
                  ↑                            ↓
                  └────────────────────────────┘

... continues until quality threshold achieved (<5% issues) ...

V∞ → PERFECT EXTRACTION (or close enough!)
```

### Loop Stages

**1. EXTRACT (Current Version)**
```python
# Run V(n) extraction on Soil Handbook
extraction_output = extract_knowledge_graph_from_book(
    book_path="data/books/soil_stewardship_handbook/Soil_Stewardship_Handbook.pdf",
    version=f"v{n}",
    playbook_path="kg_extraction_playbook/"
)

# Output: relationships.json (e.g., 873 relationships for V5)
```

**2. REFLECT (Claude Sonnet 4.5)**
```python
# Reflector analyzes extraction quality
reflector = ReflectorAgent(model="claude-sonnet-4-5-20250929")

quality_report = reflector.analyze_kg_quality(
    relationships=extraction_output['relationships'],
    source_text=book_text,
    v4_quality_reports=historical_reports  # Training data
)

# Output:
# - Issue categories (pronouns, lists, reversed authorship, etc.)
# - Root cause analysis (which module/prompt failed?)
# - Improvement recommendations with priorities
```

**3. CURATE (GPT-4o)**
```python
# Curator proposes specific code/prompt changes
curator = CuratorAgent(model="gpt-4o")

changeset = curator.curate_improvements(
    quality_report=quality_report,
    current_playbook=playbook_v_n,
    target_version=n+1
)

# Output:
# - Python code changes (e.g., fix BibliographicCitationParser regex)
# - Prompt updates (e.g., add few-shot examples)
# - Config tweaks (e.g., adjust confidence thresholds)
```

**4. EVOLVE (Automated)**
```python
# Apply changeset to create V(n+1)
evolution_engine.apply_changeset(
    changeset=changeset,
    source_version=n,
    target_version=n+1,
    backup=True  # For rollback if needed
)

# Output: Updated Playbook with version bumped to V(n+1)
```

**5. REPEAT (Forever)**
```python
while issue_rate > 0.05:  # Target: <5% quality issues
    version += 1
    run_ace_cycle(version)

    # Monitor convergence
    if issue_rate_not_improving_for_3_cycles:
        # Try more aggressive changes
        enable_experimental_modules()
```

---

## 🧠 Why Claude Sonnet 4.5 for Reflector?

The Reflector is the **most critical component** - it must:
- Deeply understand extraction quality issues
- Trace problems to root causes in code/prompts
- Discover novel error patterns not in V4 reports
- Provide actionable, specific recommendations

**Claude Sonnet 4.5 advantages**:
- Superior analytical reasoning
- Better at finding subtle patterns
- Excellent at tracing causality
- More thorough code analysis
- Longer context window for reviewing entire files

**Cost consideration**: Reflector runs once per cycle (not per-query), so higher cost is justified for quality.

---

## 🎨 The Playbook: What Agents Can Modify

```
kg_extraction_playbook/
├── config/
│   ├── extraction_config.json          # Main settings
│   ├── vocabularies/
│   │   ├── pronouns.json               # For PronounResolver
│   │   ├── metaphorical_terms.json     # For FigurativeLanguageFilter
│   │   ├── bad_title_endings.json      # For TitleCompletenessValidator
│   │   └── authorship_predicates.json  # For BibliographicCitationParser
│   └── thresholds.json                 # Confidence thresholds
│
├── prompts/
│   ├── pass1_extraction.txt            # Main extraction prompt
│   ├── pass2_evaluation.txt            # Dual-signal evaluation
│   └── few_shot_examples.json          # Training examples
│
├── modules/
│   ├── orchestrator.py                 # Main pipeline
│   ├── pdf_processor.py                # Text extraction & chunking
│   ├── pass1_extractor.py              # Pass 1: Extraction
│   ├── pass2_evaluator.py              # Pass 2: Validation
│   │
│   └── pass2_5_postprocessing/         # ✨ Pass 2.5 modules
│       ├── bibliographic_parser.py     # Fix authorship direction
│       ├── list_splitter.py            # Split comma-separated targets
│       ├── pronoun_resolver.py         # Resolve pronouns to entities
│       ├── context_enricher.py         # Expand vague concepts
│       ├── title_validator.py          # Flag incomplete titles
│       ├── predicate_validator.py      # Validate semantic compatibility
│       └── figurative_filter.py        # Detect metaphorical language
│
└── schemas/
    ├── relationship_schema.py          # Pydantic models
    └── quality_flags.py                # Flag definitions
```

**Agents can modify ANY file** to improve extraction quality.

---

## 📈 Success Metrics & Convergence

### Primary Metric: Quality Issue Rate

```
Quality Issue Rate = (Problematic Relationships / Total Relationships) × 100%

V4: 57% issues → V5 Target: <10% → V6+ Target: <5% → V∞: <1%
```

### Issue Categories (from V4 analysis)

| Issue Type | V4 Count | V4 % | V5 Target | V6+ Target |
|------------|----------|------|-----------|------------|
| Reversed Authorship | 105 | 12% | 0 | 0 |
| List Targets | 100 | 11.5% | <5 | 0 |
| Pronoun Sources | 75 | 8.6% | <10 | 0 |
| Vague Entities | 56 | 6.4% | <15 | 0 |
| Incomplete Titles | 70 | 8% | <10 | 0 |
| Wrong Predicates | 52 | 6% | <10 | 0 |
| Figurative Language | 44 | 5% | <10 | 0 |
| **TOTAL** | **495** | **57%** | **<10%** | **<5%** |

### Convergence Criteria

**Stop iterating when**:
- Quality issue rate < 5% for 3 consecutive cycles
- No critical issues (severity: CRITICAL)
- No new issue patterns discovered
- Improvements plateau (<1% gain per cycle)

**OR**:
- Manual intervention required (system can't self-improve further)
- Reached maximum iterations (safety limit: 50 cycles)

---

## 🚀 Deployment Architecture

### Option 1: Continuous Background Process
```python
# Run on server, continuously improving
python scripts/run_ace_continuous.py \
    --book "data/books/soil_stewardship_handbook/Soil_Stewardship_Handbook.pdf" \
    --target-quality 0.05 \
    --max-iterations 50 \
    --checkpoint-interval 5
```

### Option 2: Scheduled Batch Runs
```bash
# Cron job: Run one cycle per day
0 2 * * * cd /home/claudeuser/yonearth-gaia-chatbot && \
    python scripts/run_ace_cycle.py --mode kg_extraction
```

### Option 3: Interactive Mode (Development)
```python
# Run cycle-by-cycle with human approval
python scripts/run_ace_interactive.py
# Reviews each changeset before applying
```

---

## 🔍 Example Iteration: V5 → V6

### Stage 1: EXTRACT (V5)
```
Running V5 extraction on Soil Stewardship Handbook...
✅ Extracted 1,023 relationships (873 original + 150 from list splitting)
```

### Stage 2: REFLECT (Claude Sonnet 4.5)
```json
{
  "critical_issues": [
    {
      "category": "Reversed Authorship",
      "count": 15,
      "root_cause": "BibliographicCitationParser.citation_patterns missing format: 'Title. Author. Publisher.'",
      "affected_module": "modules/pass2_5_postprocessing/bibliographic_parser.py",
      "recommendation": "Add pattern: r'^\"([^\"]+)\"\\.\\ ([A-Z][a-z]+(?:,?\\ [A-Z][a-z]+)+)\\.'",
      "priority": 10
    }
  ],
  "quality_summary": {
    "total_relationships": 1023,
    "issue_rate": 0.10,
    "critical_issues": 15,
    "high_priority": 45,
    "grade": "B+"
  }
}
```

### Stage 3: CURATE (GPT-4o)
```json
{
  "changeset": {
    "operations": [
      {
        "file": "modules/pass2_5_postprocessing/bibliographic_parser.py",
        "type": "CODE_EDIT",
        "old_content": "self.citation_patterns = [\n    r'^([A-Z][a-z]+,\\s+[A-Z][a-z]+)\\.',\n    r'^([A-Z][a-z]+(?:\\s+[A-Z][a-z]+)+)\\.',\n]",
        "new_content": "self.citation_patterns = [\n    r'^([A-Z][a-z]+,\\s+[A-Z][a-z]+)\\.',\n    r'^([A-Z][a-z]+(?:\\s+[A-Z][a-z]+)+)\\.',\n    r'^\"([^\"]+)\"\\.\\ ([A-Z][a-z]+(?:,?\\ [A-Z][a-z]+)+)\\.',\n]",
        "rationale": "Add pattern to detect 'Title. Author. Publisher.' format"
      }
    ]
  }
}
```

### Stage 4: EVOLVE
```
Applying changeset...
✅ Updated bibliographic_parser.py
✅ Backed up V5 to kg_extraction_playbook_backups/v5/
✅ Bumped version to V6
```

### Stage 5: VERIFY (Next Cycle)
```
Running V6 extraction on Soil Stewardship Handbook...
✅ Extracted 1,023 relationships
Reflect: Critical issues: 0 (was 15) ✅
Reflect: Quality issue rate: 5.2% (was 10%) ✅
Grade: A-
```

---

## 🛡️ Safety Mechanisms

### 1. Version Control & Rollback
- Every version backed up before changes
- Rollback command: `python scripts/rollback_version.py --to v5`

### 2. Change Approval
- Critical changes require human review
- Risk levels: LOW (auto-apply), MEDIUM (review), HIGH (manual)

### 3. Quality Regression Detection
- If V(n+1) worse than V(n), automatic rollback
- Alert human operator for manual intervention

### 4. Iteration Limits
- Max 50 cycles (prevent infinite loops)
- Plateau detection (stop if no improvement for 5 cycles)

### 5. Validation Tests
- Unit tests for each module
- Integration tests on sample extractions
- Fail-safe: Don't deploy if tests fail

---

## 📅 Roadmap

### Phase 1: Foundation (Week 1)
- [x] Design ACE architecture
- [x] Implement Reflector (Claude Sonnet 4.5)
- [x] Implement Curator (GPT-4o)
- [ ] Build continuous orchestrator
- [ ] Set up Playbook version control

### Phase 2: Initial Extraction (Week 1)
- [ ] Run V5 extraction on Soil Handbook
- [ ] Generate V4 quality report for comparison
- [ ] Establish baseline metrics

### Phase 3: First Iteration (Week 2)
- [ ] Run first ACE cycle (V5→V6)
- [ ] Verify improvements
- [ ] Refine agent prompts based on results

### Phase 4: Continuous Improvement (Weeks 3-4)
- [ ] Run 5-10 cycles
- [ ] Monitor convergence
- [ ] Document patterns discovered
- [ ] Achieve <5% quality issue rate

### Phase 5: Expansion (Future)
- [ ] Apply to other books (VIRIDITAS, Y on Earth)
- [ ] Generalize to podcast episodes
- [ ] Scale to full YonEarth corpus

---

## 🎓 Learning Outcomes

By the end of this project, we will have:

1. **A Self-Improving System**: KG extraction that gets better autonomously
2. **Version History**: Complete evolution from V5 to V∞
3. **Pattern Library**: Documented error patterns and fixes
4. **Agent Templates**: Reusable Reflector/Curator architectures
5. **Research Validation**: Real-world application of ACE paper principles

---

## 🔗 Related Documents

- [V5 Implementation Plan](V5_IMPLEMENTATION_PLAN.md)
- [V4 Quality Reports](V4_EXTRACTION_QUALITY_ISSUES_REPORT.md)
- [ACE Framework Design](../ACE_FRAMEWORK_DESIGN.md)
- [Playbook Structure](../../data/kg_extraction_playbook/README.md)

---

**Let the never-ending improvement begin! 🚀**
