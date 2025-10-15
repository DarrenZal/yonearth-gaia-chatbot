# ACE System with Prompt Optimization

**Date**: October 2025
**Status**: Enhanced - Now handles both code AND prompt optimization
**Version**: 2.0

---

## 🎯 What Changed

The ACE (Agentic Context Engineering) system has been extended to optimize **prompts** in addition to code, configs, and modules. This means the Reflector can now analyze prompt quality issues and the Curator can propose prompt improvements.

### Before (ACE 1.0)
- ✅ Code fixes (regex, algorithms, modules)
- ✅ Config updates (thresholds, vocabularies)
- ❌ Prompts were **hardcoded** in Python scripts

### After (ACE 2.0)
- ✅ Code fixes (regex, algorithms, modules)
- ✅ Config updates (thresholds, vocabularies)
- ✅ **Prompt optimization** (version-controlled, ACE-managed)

---

## 📂 New Architecture

### Prompt Storage

Prompts are now stored in version-controlled files:

```
kg_extraction_playbook/
├── prompts/
│   ├── pass1_extraction_v7.txt       # Current Pass 1 prompt
│   ├── pass1_extraction_v8.txt       # ACE-evolved version
│   ├── pass2_evaluation_v7.txt       # Current Pass 2 prompt
│   └── pass2_evaluation_v8.txt       # ACE-evolved version
├── prompt_loader.py                   # Utility to load prompts
├── analysis_reports/                  # Reflector outputs
└── changesets/                        # Curator outputs
```

### Prompt Loader

The `PromptLoader` class manages version-controlled prompts:

```python
from kg_extraction_playbook.prompt_loader import PromptLoader

loader = PromptLoader()

# Load current version
prompt = loader.load_prompt("pass1_extraction", "v7")

# Format with placeholders
formatted = prompt.format(text=chunk_text)

# Save new version (done by Curator)
loader.save_prompt("pass1_extraction", "v8", updated_content)
```

---

## 🔄 ACE Cycle with Prompts

### Enhanced Workflow

```
V7 → EXTRACT → REFLECT → CURATE → EVOLVE → V8
              (analyzes    (proposes
               prompts)     prompt changes)
```

### Step-by-Step

#### 1. **Extract** (V7)
- Uses `prompts/pass1_extraction_v7.txt` and `pass2_evaluation_v7.txt`
- Produces knowledge graph with relationships

#### 2. **Reflect** (Reflector Agent)
- Analyzes extraction quality
- **NEW**: Analyzes prompt quality
  - "Does Pass 1 prompt allow pronouns through?"
  - "Is Pass 2 evaluation instruction clear?"
  - "Would few-shot examples help?"
- Outputs `prompt_analysis` section in report

#### 3. **Curate** (Curator Agent)
- Proposes fixes for issues
- **NEW**: Proposes `PROMPT_ENHANCEMENT` operations
  - Decides prompt vs code vs both
  - Generates `old_content` → `new_content` diffs
  - Specifies `prompt_version: "v8"`

#### 4. **Evolve** (Apply Changes)
- Applies code fixes
- **NEW**: Creates `prompts/pass1_extraction_v8.txt`
- Version bump: V7 → V8

---

## 🧠 Reflector Prompt Analysis

### What the Reflector Now Does

The Reflector's system prompt now includes:

```
PROMPT ANALYSIS GUIDELINES:
- **Pass 1 Issues**: If entities are extracted incorrectly from the start
  (e.g., pronouns, vague terms), consider if the extraction prompt encourages these errors

- **Pass 2 Issues**: If evaluation scores are miscalibrated, consider if the
  evaluation prompt is unclear

- **When to Recommend Prompt Changes**:
  - If errors appear BEFORE Pass 2.5 modules can fix them
  - If multiple Pass 2.5 modules are working around an upstream prompt issue
  - If the pattern affects >10% of relationships
  - If code fixes seem hacky/brittle compared to clearer prompts
```

### Example Reflector Output

```json
{
  "prompt_analysis": {
    "pass1_extraction_issues": [
      {
        "issue": "Prompt says 'Extract ALL' which encourages over-extraction of vague entities",
        "current_wording": "Extract ALL relationships you can find in this text.",
        "suggested_fix": "Change to 'Extract clearly stated relationships' and add examples",
        "examples_needed": true
      }
    ],
    "pass2_evaluation_issues": [
      {
        "issue": "Instruction 'ignore world knowledge' confuses LLM",
        "current_wording": "TEXT SIGNAL (ignore world knowledge)",
        "suggested_fix": "Rephrase as 'TEXT SIGNAL (focus only on text clarity)'"
      }
    ]
  },
  "improvement_recommendations": [
    {
      "priority": "HIGH",
      "type": "PROMPT_ENHANCEMENT",
      "target_file": "prompts/pass1_extraction_v7.txt",
      "recommendation": "Add 'CRITICAL RULES' section prohibiting pronouns",
      "expected_impact": "Reduces pronoun issues from 8.6% to <2%",
      "rationale": "Preventing pronouns at extraction is more reliable than fixing them in Pass 2.5"
    }
  ]
}
```

---

## 🛠️ Curator Prompt Enhancement

### Decision Framework

The Curator now decides **when to use prompts vs code**:

#### Use PROMPT_ENHANCEMENT when:
- ✅ Errors occur in Pass 1 extraction (before any code processing)
- ✅ LLM behavior can be guided with clearer instructions
- ✅ Adding constraints/examples would prevent the issue
- ✅ Multiple Pass 2.5 modules are compensating for prompt weakness

**Example**: "Stop extracting pronouns" → Add to Pass 1 prompt

#### Use CODE_FIX when:
- ✅ Pattern is systematic and can be detected with rules
- ✅ LLM output needs deterministic transformation
- ✅ Domain-specific logic is required (e.g., bibliographic citations)
- ✅ Prompt changes alone won't be reliable

**Example**: "Reverse author/title" → Code is more reliable than prompt

#### Use BOTH when:
- ✅ Prompt prevents error from occurring (upstream)
- ✅ Code catches any that slip through (downstream safety net)

**Example**: Prompt says "no pronouns" + PronounResolver as backup

### Example Curator Changeset

```json
{
  "file_operations": [
    {
      "operation_id": "change_001",
      "operation_type": "PROMPT_ENHANCEMENT",
      "file_path": "prompts/pass1_extraction_v7.txt",
      "priority": "HIGH",
      "rationale": "Add explicit pronoun prohibition to prevent extraction errors upstream",
      "risk_level": "low",
      "affected_issue_category": "Pronoun Sources",
      "expected_improvement": "Reduces pronoun issues from 8.6% to <2%",

      "edit_details": {
        "target_section": "OUTPUT FORMAT",
        "old_content": "## 📝 OUTPUT FORMAT ##",
        "new_content": "## ⚠️ CRITICAL RULES ##\n\n**NEVER use pronouns as entities:**\n   ❌ BAD: (He, resides in, Colorado)\n   ✅ GOOD: (Aaron William Perry, resides in, Colorado)\n\n## 📝 OUTPUT FORMAT ##",
        "validation": "Test prompt on sample text with pronouns",
        "prompt_version": "v8"
      }
    }
  ]
}
```

---

## 🚀 Usage Examples

### Run ACE Cycle with Prompt Analysis

```bash
# Run full ACE cycle (Reflect → Curate → Evolve)
python scripts/run_ace_cycle.py
```

The cycle will now:
1. Load V7 extraction results
2. Reflector analyzes code AND prompts
3. Curator proposes code fixes AND prompt enhancements
4. System creates V8 with both types of changes

### Manual Prompt Optimization

```python
from src.ace_kg.kg_reflector import KGReflectorAgent
from src.ace_kg.kg_curator import KGCuratorAgent

# Step 1: Reflect
reflector = KGReflectorAgent()
analysis = reflector.analyze_kg_extraction(
    relationships=v7_relationships,
    source_text=book_text,
    extraction_metadata={"version": "v7"}
)

# Step 2: Curate
curator = KGCuratorAgent()
changeset = curator.curate_improvements(
    reflector_report=analysis,
    current_version=7
)

# Step 3: Review prompt changes
for op in changeset["file_operations"]:
    if op["operation_type"] == "PROMPT_ENHANCEMENT":
        print(f"Prompt change: {op['file_path']}")
        print(f"Rationale: {op['rationale']}")
        print(f"Expected impact: {op['expected_improvement']}")

# Step 4: Apply (with approval for prompts)
results = curator.apply_changeset(
    changeset=changeset,
    auto_apply_low_risk=True  # Prompts are usually low-risk
)
```

### Load Prompts in Extraction Scripts

Update your extraction scripts to use `PromptLoader`:

```python
# OLD WAY (hardcoded)
BOOK_EXTRACTION_PROMPT = """Extract ALL relationships..."""

# NEW WAY (ACE-managed)
from kg_extraction_playbook.prompt_loader import PromptLoader

loader = PromptLoader()
BOOK_EXTRACTION_PROMPT = loader.load_prompt("pass1_extraction", "v8")
```

---

## 📊 Expected Benefits

### 1. **Faster Iteration**
- **Before**: Manually edit prompts in Python files, test, repeat
- **After**: ACE automatically proposes and applies prompt changes

### 2. **Better Quality**
- **Before**: Prompts optimized by trial and error
- **After**: Claude Sonnet 4.5 analyzes prompts systematically

### 3. **Version Control**
- **Before**: Prompt history lost in git diffs
- **After**: Each prompt version saved as separate file (v7, v8, v9...)

### 4. **Interpretability**
- **Before**: "Why did this prompt work better?" → unclear
- **After**: Curator explains rationale for each prompt change

### 5. **Complementary to Code**
- **Before**: Only code fixes, prompts ignored
- **After**: Upstream prompt fixes + downstream code safety nets

---

## 🎓 Comparison to DSPy

### ACE 2.0 (This System)
- ✅ **Interpretable**: Claude Sonnet explains WHY prompts need changing
- ✅ **Strategic**: Chooses prompt vs code based on reasoning
- ✅ **Domain-aware**: Leverages KG extraction expertise
- ⚠️ **Manual approval**: Human reviews prompt changes
- ⚠️ **3-5 variations** per cycle

### DSPy
- ✅ **Automated**: No human reasoning needed
- ✅ **Exhaustive**: Tests 50-200 prompt variations
- ✅ **Few-shot optimization**: Algorithmically selects best examples
- ⚠️ **Black box**: "This works" without explanation
- ⚠️ **Expensive**: High API costs for exhaustive search

### When to Add DSPy

If ACE 2.0 plateaus (e.g., stuck at 4.5% error rate), consider:
1. Use ACE 2.0 for interpretable, strategic improvements
2. Add DSPy for few-shot example selection
3. Use DSPy for exhaustive search when stuck

**Hybrid approach**: ACE handles prompt reasoning, DSPy handles example optimization.

---

## 🔮 Future Enhancements

### Phase 1 (Current)
- ✅ Reflector analyzes prompts
- ✅ Curator proposes prompt changes
- ✅ Version-controlled prompt files
- ✅ Prompt loader utility

### Phase 2 (Next)
- [ ] Few-shot example management
- [ ] Prompt A/B testing framework
- [ ] Automated prompt regression tests
- [ ] Prompt performance metrics

### Phase 3 (Future)
- [ ] DSPy integration for example selection
- [ ] Prompt optimization search
- [ ] Multi-model prompt adaptation
- [ ] Prompt effectiveness analytics

---

## 📞 Usage Notes

### For V8 Extraction

Once V8 completes, you can run an ACE cycle:

```bash
# 1. Let V8 finish running
# 2. Run ACE cycle on V8 results
python scripts/run_ace_cycle.py

# 3. Review prompt changes in changeset
# 4. Apply approved changes
# 5. V9 will use updated prompts
```

### Backward Compatibility

Old scripts still work! If they don't use `PromptLoader`, they'll use hardcoded prompts. New scripts should use `PromptLoader` for ACE management.

### Prompt Versioning

- `v7`: Current stable prompts (extracted from V7 script)
- `v8`: First ACE-evolved prompts (will be created by ACE)
- `v9`, `v10`, etc.: Future ACE iterations

---

## ✅ Summary

**ACE 2.0 now optimizes the ENTIRE knowledge graph extraction pipeline:**

1. **Code** (modules, algorithms, regex) ← ACE 1.0
2. **Configs** (thresholds, vocabularies) ← ACE 1.0
3. **Prompts** (extraction, evaluation) ← **NEW in ACE 2.0!**

This gives you:
- ✅ Interpretable prompt optimization (unlike DSPy)
- ✅ Strategic prompt vs code decisions
- ✅ Version-controlled prompt evolution
- ✅ Claude Sonnet 4.5's analytical reasoning
- ✅ Most of DSPy's value without the black box

**Next step**: Let V8 finish, then run your first ACE cycle with prompt optimization! 🚀

---

**Last Updated**: October 13, 2025
**Version**: ACE 2.0 (Prompt Optimization)
**Maintainer**: YonEarth KG Team
