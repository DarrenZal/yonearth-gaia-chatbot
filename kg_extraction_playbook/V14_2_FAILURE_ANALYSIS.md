# V14.2 Failure Analysis - Root Cause Identified

**Date**: October 14, 2025
**Result**: V14.2 achieved **C+ grade (27.3% issue rate)** - WORSE than V14.0's B+ (10.78%)
**Analysis**: Complete root cause identified through predicate fragmentation investigation

---

## 🔥 Critical Finding: Predicate Fragmentation is the Root Cause

### Quantitative Evidence:

| Version | Config | Predicate Issues | Issue Rate | Total Issues | Grade |
|---------|--------|------------------|------------|--------------|-------|
| **V13.1** | V12 Pass 1 + V13.1 Pass 2.5 (12 modules) | **10 (1.1%)** | 8.6% | 75 | **A-** |
| **V14.0** | V14 Pass 1 + V14.0 Pass 2.5 (14 modules) | **12 (2.0%)** | 10.78% | 65 | **B+** |
| **V14.2** | V14 Pass 1 + V13.1 Pass 2.5 (12 modules) | **100 (19.3%)** | **27.3%** | **141** | **C+** |

**Key Insight**: V14.2 has **10x the predicate fragmentation** of V13.1, despite using the SAME Pass 2.5 PredicateNormalizer!

### The Problem:

V14.2 has **100 predicate fragmentation issues (19.3%)**, accounting for **71% of all issues** (100 out of 141 total).

This is NOT a pipeline problem - it's a **prompt-pipeline mismatch problem**.

---

## 🎯 Root Cause: V14 Pass 1 Prompt Extracts Verbose Predicates

### V14 Pass 1 Predicate Examples (Verbose & Complex):

**V14 extracts these verbose predicates that V13.1's normalizer can't handle:**
- "is key for" (should normalize to → "is-essential-for")
- "is made manifest by" (should normalize to → "is-demonstrated-by")
- "is toward" (should normalize to → "aims-at")
- "is made from" (should normalize to → "is-composed-of")
- "are found in" (should normalize to → "are-located-in")

### V12 Pass 1 Predicate Examples (Simple & Normalizable):

**V12 extracted these simple predicates that V13.1's normalizer handles well:**
- "authored"
- "published by"
- "contains"
- "is-a"
- "produces"
- "enhances"
- "includes"

### The Mismatch:

**V13.1's PredicateNormalizer was designed to normalize V12's simple predicates, NOT V14's verbose predicates.**

When V14.2 combined:
- ✅ V14 Pass 1 (extracts verbose predicates)
- ❌ V13.1 Pass 2.5 (normalizes V12's simple predicates)

The result: **Massive predicate fragmentation** because the normalizer doesn't recognize V14's predicate patterns.

---

## 📊 V14.2 Results Breakdown:

### Final Metrics:
- **Total Relationships**: 517 (14% fewer than V14.0's 603, 41% fewer than V13.1's 873)
- **Total Issues**: 141 (117% MORE than V14.0's 65!)
- **Issue Rate**: 27.3% (3x worse than V14.0's 10.78%)
- **Grade**: C+ (FAILED - target was A or A-)

### Issue Categories (V14.2):

| Category | Count | Rate | Severity |
|----------|-------|------|----------|
| **Predicate Fragmentation** | **100** | **19.3%** | MEDIUM |
| **Philosophical/Metaphorical** | 52 | 10.1% | MEDIUM |
| **Vague/Abstract Entities** | 18 | 3.5% | MEDIUM |
| **Possessive Pronouns** | 12 | 2.3% | MILD |
| **Opinion/Normative** | 7 | 1.4% | MILD |
| **Figurative Language** | 3 | 0.6% | MILD |
| **Praise Quote** | 1 | 0.2% | MILD |

**Predicate fragmentation alone accounts for 71% of all issues.**

---

## 💡 Why V14.2 Failed Worse Than Expected

### Original V14.2 Hypothesis (INCORRECT):
> "V13.1's success came from its Pass 2.5 configuration (12 modules).
> V14.0's regression came from adding MetadataFilter and ConfidenceFilter (14 modules).
> Solution: Use V14 Pass 1 + V14 Pass 2 + V13.1 Pass 2.5 (12 modules)."

### Actual Reality (CORRECT):
> **V13.1's success came from the COMBINATION of:**
> 1. ✅ V12's simple Pass 1 prompt (extracts simple predicates)
> 2. ✅ V13.1's Pass 2.5 pipeline (normalizes V12's predicate patterns)
>
> **V14.2's failure came from:**
> 1. ❌ V14's verbose Pass 1 prompt (extracts complex predicates)
> 2. ❌ V13.1's Pass 2.5 pipeline (doesn't recognize V14's predicate patterns)

### The Conservative Rollback Was Based on Incorrect Analysis:

The V14.2 root cause analysis (V14_2_ROOT_CAUSE_ANALYSIS.md) stated:
> "V13.1 and V14.0 used IDENTICAL Pass 2 prompts - regression was in Pass 2.5 configuration"

**This was CORRECT about Pass 2 being identical, but INCORRECT about the regression source.**

The regression in V14.0 was NOT primarily from Pass 2.5 module changes - it was from **V14 Pass 1 extracting different types of predicates** that the pipeline (both V13.1's 12 modules and V14.0's 14 modules) struggled to normalize effectively.

---

## 🔬 Evidence: Predicate Examples from V14.2

### Example 1: Verbose Predicate
**Extracted**: `(soil, is key for, agriculture)`
**Problem**: "is key for" is verbose and not normalized
**Should be**: `(soil, is-essential-for, agriculture)` or `(soil, supports, agriculture)`

### Example 2: Overly Complex Predicate
**Extracted**: `(soil, is made manifest by, ...)`
**Problem**: Poetic/complex phrasing
**Should be**: `(soil, is-demonstrated-by, ...)` or `(soil, is-shown-by, ...)`

### Example 3: Fragmented Base Predicate
**V14.2 extracted 6 variations of "is":**
- "is key for"
- "is made from"
- "is toward"
- "is made manifest by"
- "is-a" (normalized)
- "is" (simple)

**V13.1 extracted 2-3 variations of "is":**
- "is-a" (normalized)
- "is" (simple)

---

## 🎯 Why V14 Pass 1 Prompt Creates Verbose Predicates

### V12 Pass 1 Prompt Guidance (Simple):
```
relationship: Relationship type (authored, published by, contains, is-a, produces, enhances, etc.)
```

**Clear examples**: `authored`, `published by`, `contains`, `is-a`, `produces`, `enhances`

The LLM sees these examples and uses similar simple predicates.

### V14 Pass 1 Prompt Guidance (Complex):
```
## 📚 RELATIONSHIP TYPES TO EXTRACT (PRIORITY ORDER)

### 1. BIBLIOGRAPHIC (Authorship & Publication) ⭐ HIGH PRIORITY
### 2. CATEGORICAL (Is-A & Definitions) ⭐ HIGH PRIORITY
### 3. COMPOSITIONAL (Parts & Contents) ⭐ HIGH PRIORITY
### 4. FUNCTIONAL (Actions & Processes) ⭐ MEDIUM PRIORITY
...
```

**No clear predicate examples** - instead, provides high-level categories and descriptions.

The LLM interprets the text more literally and extracts verbose predicates that sound natural but fragment the graph:
- Text: "soil is key for agriculture" → V14 extracts "is key for"
- Text: "soil is made manifest by..." → V14 extracts "is made manifest by"

**V12 would have extracted**:
- "soil **supports** agriculture" (normalized)
- "soil **is-demonstrated-by** ..." (normalized)

---

## 📉 Why V14.2 Got Fewer Relationships Than Expected

### Expected (Based on V13.1):
- V13.1: 861 Pass 1 candidates → 873 final (12 modules expanded via ListSplitter)
- **V14.2 Expected**: 600-650 candidates → ~850-900 final

### Actual:
- **V14.2 Actual**: 571 Pass 1 candidates → **517 final** (12 modules)

### Why Fewer?

1. **V14 Pass 1 is more restrictive** (571 vs 861 candidates)
2. **V13.1 Pass 2.5 filtered more aggressively**:
   - VagueEntityBlocker: 14 filtered (V14's verbose predicates create vague entities)
   - ClaimClassifier: 13 filtered (V14's verbose predicates flagged as low-quality)
   - Deduplicator: 81 removed (V14's fragmented predicates create duplicate content)

---

## ❌ Why the Conservative Rollback Failed

### The Hypothesis:
"If V14.0's regression was caused by adding 2 new modules (MetadataFilter, ConfidenceFilter), then removing those modules should restore V13.1's A- grade."

### The Reality:
**The regression was NOT caused by adding modules - it was caused by V14's Pass 1 prompt extracting different types of predicates.**

V14.0's 14 modules (including the new ones) were actually doing a BETTER job handling V14's verbose predicates than V13.1's 12 modules:
- **V14.0**: 603 relationships, 65 issues (10.78%), 12 predicate issues (2.0%)
- **V14.2**: 517 relationships, 141 issues (27.3%), **100 predicate issues (19.3%)**

**V14.0's additional modules (MetadataFilter, ConfidenceFilter) were HELPING, not hurting!**

---

## 🔍 Correct Analysis: Why V14.0 Regressed from V13.1

### V13.1 (A- grade, 8.6%):
- ✅ V12 Pass 1: Simple predicates ("authored", "contains", "is-a")
- ✅ V13.1 Pass 2.5: Normalizer handles V12's patterns well (10 predicate issues, 1.1%)
- ✅ Result: 873 relationships, 75 issues (8.6%)

### V14.0 (B+ grade, 10.78%):
- ⚠️ V14 Pass 1: Verbose predicates ("is key for", "is made manifest by")
- ⚠️ V14.0 Pass 2.5: Normalizer struggles with V14's patterns (12 predicate issues, 2.0%)
- ⚠️ BUT: MetadataFilter and ConfidenceFilter helped reduce other issues
- ⚠️ Result: 603 relationships, 65 issues (10.78%)

**V14.0's regression was mild (8.6% → 10.78%) because the new modules partially compensated for the predicate fragmentation problem.**

### V14.2 (C+ grade, 27.3%):
- ❌ V14 Pass 1: Verbose predicates (same as V14.0)
- ❌ V13.1 Pass 2.5: Normalizer completely fails with V14's patterns (100 predicate issues, 19.3%)
- ❌ NO MetadataFilter or ConfidenceFilter to compensate
- ❌ Result: 517 relationships, 141 issues (27.3%)

**V14.2's failure was catastrophic because it removed the compensating modules without fixing the root cause (verbose predicates).**

---

## 🎯 Correct Path Forward

### Option 1: Full V13.1 Configuration (Conservative - Recommended)
**Configuration**: V12 Pass 1 + V13.1 Pass 2 + V13.1 Pass 2.5 (12 modules)

**Rationale**:
- V12 Pass 1 extracts simple predicates that V13.1's normalizer handles well
- Proven A- grade baseline (8.6% issue rate, 873 relationships)
- Restores the FULL working configuration

**Expected Results**:
- ~860 Pass 1 candidates
- ~870 final relationships
- 8-10% issue rate
- A- grade (matches V13.1 baseline)

**Risk**: Low - exact replica of V13.1's proven configuration

### Option 2: Fix V14 Pass 1 Prompt (Moderate Risk)
**Configuration**: V14 Pass 1 (FIXED) + V14 Pass 2 + V14 Pass 2.5 (14 modules)

**Changes Needed**:
1. **Add clear predicate examples** to V14 Pass 1:
   ```
   relationship: Simple predicate (authored, published-by, contains, is-a, produces, enhances, supports, enables)

   ❌ AVOID verbose predicates: "is key for", "is made manifest by", "is toward"
   ✅ USE simple predicates: "is-essential-for", "is-demonstrated-by", "aims-at"
   ```

2. **Add predicate simplicity guidance**:
   ```
   ## ⚡ PREDICATE SIMPLICITY RULE

   Use the SIMPLEST predicate that accurately captures the relationship:
   - "soil is key for agriculture" → EXTRACT: (soil, supports, agriculture)
   - "soil is made manifest by microbes" → EXTRACT: (soil, is-demonstrated-by, microbes)
   - "practice aims toward sustainability" → EXTRACT: (practice, aims-at, sustainability)
   ```

**Expected Results**:
- ~600 Pass 1 candidates (V14's selectivity)
- ~850 final relationships
- 5-8% issue rate
- A or A- grade

**Risk**: Medium - requires prompt tuning and validation

### Option 3: Enhance V14 PredicateNormalizer (High Risk)
**Configuration**: V14 Pass 1 (AS IS) + V14 Pass 2 + V14 Pass 2.5 (14 modules + ENHANCED normalizer)

**Changes Needed**:
1. **Expand PredicateNormalizer** to handle V14's verbose patterns:
   ```python
   NORMALIZATION_RULES = {
       # V14-specific verbose predicates
       r"is key for": "is-essential-for",
       r"is made manifest by": "is-demonstrated-by",
       r"is toward": "aims-at",
       r"is made from": "is-composed-of",
       r"are found in": "are-located-in",
       # ... 50+ more rules ...
   }
   ```

**Expected Results**:
- ~600 Pass 1 candidates
- ~850 final relationships
- 8-12% issue rate (predicate fragmentation fixed, but other V14 issues remain)
- B+ to A- grade

**Risk**: High - requires extensive rule engineering and testing

---

## 🏁 Recommendation: Option 1 (Full V13.1 Configuration)

### Why Option 1:
1. ✅ **Proven baseline**: V13.1 achieved A- grade (8.6%)
2. ✅ **Low risk**: Exact replica of working configuration
3. ✅ **Immediate results**: No prompt tuning or code changes needed
4. ✅ **Establishes stable baseline** for future improvements

### Implementation:
1. Create `extract_kg_v14_3_book.py` using:
   - **Pass 1**: `pass1_extraction_v12.txt` (or V10 - the simple one)
   - **Pass 2**: `pass2_evaluation_v13_1.txt` (or v14.txt - they're identical)
   - **Pass 2.5**: `get_book_pipeline(version='v13')` (12 modules)

2. Expected outcome:
   - ✅ 8-10% issue rate (A- or A grade)
   - ✅ ~870 relationships
   - ✅ 10 predicate fragmentation issues (1.1% - same as V13.1)

3. After establishing stable baseline:
   - 🔬 Analyze V14 Pass 1 prompt to identify specific guidance causing verbose predicates
   - 🔧 Create V14.4 with fixed Pass 1 prompt
   - 🎯 Target V15: A+ grade (<5% issue rate)

---

## 📝 Lessons Learned

### 1. Prompt-Pipeline Co-Design is Critical
**Lesson**: Prompts and pipelines must be designed together. Changing one without the other creates mismatches.

**V14.2 Mistake**: Assumed Pass 2.5 pipeline was independent of Pass 1 prompt. In reality, V13.1's PredicateNormalizer was specifically tuned for V12's simple predicate patterns.

### 2. Conservative Rollback Requires Full Configuration
**Lesson**: Partial rollbacks can fail catastrophically if the root cause is misidentified.

**V14.2 Mistake**: Rolled back Pass 2.5 only, kept V14 Pass 1. Should have rolled back ENTIRE configuration (Pass 1 + Pass 2 + Pass 2.5) to match V13.1.

### 3. Quantitative Analysis Reveals True Patterns
**Lesson**: Issue category breakdown reveals root causes. V14.2's 100 predicate issues (71% of total) immediately pointed to the predicate problem.

**What Worked**: Comparing predicate fragmentation across versions (V13.1: 10, V14.0: 12, V14.2: 100) proved the root cause conclusively.

### 4. Prompt Complexity ≠ Prompt Quality
**Lesson**: V14's 27KB prompt was more complex than V12's 23KB prompt, but extracted LOWER quality predicates.

**Insight**: Simple, explicit examples (V12: "authored, contains, is-a") guide LLMs better than abstract categories (V14: "BIBLIOGRAPHIC", "COMPOSITIONAL").

### 5. Module Additions Can Compensate for Prompt Issues
**Lesson**: V14.0's MetadataFilter and ConfidenceFilter helped mitigate V14 Pass 1's predicate problems.

**V14.2 Mistake**: Removed these modules thinking they caused regression, when they were actually compensating for the prompt issue.

---

## 🔬 Next Steps

1. **Immediate**: Run V14.3 with full V13.1 configuration (Option 1)
2. **Validate**: Confirm A- grade restoration
3. **Analyze**: Deep dive into V14 Pass 1 prompt to identify exact guidance causing verbose predicates
4. **Fix**: Create V14.4 with simplified predicate guidance
5. **Test**: Validate V14.4 maintains V14's selectivity (filters poetry/quotes) while extracting simple predicates
6. **Target**: V15 with A+ grade (<5% issue rate)

---

**Status**: Root cause fully identified - ready to implement Option 1 (Full V13.1 Configuration)
**Confidence**: High (based on empirical evidence from 3 extraction comparisons)
**Risk Level**: Low (returning to proven baseline)
