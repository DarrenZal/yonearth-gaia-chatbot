# ACE Cycle Full Automation Plan

**Goal**: Automate the complete ACE (Analyze-Cure-Evaluate) cycle with statistically significant validation testing at each stage.

## Current State (Manual Process)

1. ✅ Run extraction (Vn)
2. ✅ Run Reflector → identifies issues
3. ✅ Run Curator → generates changeset
4. ✅ Apply changeset (via Applicator)
5. ⚠️  **MANUAL**: Decide if changes are good
6. ⚠️  **MANUAL**: Run full extraction (Vn+1)
7. ✅ Run Reflector on Vn+1
8. ⚠️  **MANUAL**: Compare metrics, decide next steps

**Problems**:
- No pre-validation before running expensive full extraction
- Testing only on 16 examples (not statistically significant)
- No automated pipeline testing (Pass 1 → Pass 2 → Pass 2.5)

## Target State (Fully Automated)

```
┌─────────────────────────────────────────────────────────────┐
│                   ACE CYCLE AUTOMATION                       │
└─────────────────────────────────────────────────────────────┘

1. Run Extraction Vn
   ↓
2. Run Reflector
   - Identifies ALL issues (not just examples)
   - Outputs: issues_detailed.json (all 70+ issues with evidence)
   ↓
3. Run Curator
   - Generates changeset Vn+1
   ↓
4. Apply Changeset
   - Intelligent Applicator applies all changes
   ↓
5. 🆕 VALIDATION GATE: Test on Statistically Significant Sample
   - Extract 30-50 problematic chunks (95% confidence, ±10% margin)
   - Run FULL pipeline: Pass 1 → Pass 2 → Pass 2.5
   - Compare: Vn issues vs Vn+1 final relationships
   - Metrics: Issue reduction %, false positives, new issues
   ↓
6. 🤖 AUTOMATED DECISION
   - If improvement ≥ 30% → Proceed to full extraction
   - If improvement < 30% → Flag for review, suggest Curator iteration
   ↓
7. Run Full Extraction Vn+1
   - Only if validation gate passed
   ↓
8. Run Reflector on Vn+1
   ↓
9. 🤖 AUTOMATED COMPARISON
   - Compare Vn vs Vn+1 metrics
   - Generate report with visualizations
   - Decide: Success (stop) or Iterate (loop back to step 3)
```

## Required Components

### 1. Enhanced Reflector Output

**Current**: `reflection_vN.json` with `issue_categories[].examples[]` (16 examples)

**Needed**: `reflection_vN_detailed.json` with ALL issues:

```json
{
  "issues_detailed": [
    {
      "issue_id": "v11_2_2_001",
      "category": "Possessive Pronoun Sources",
      "severity": "HIGH",
      "source": "my people",
      "relationship": "love",
      "target": "the land",
      "evidence_text": "My people love the land...",
      "page": 6,
      "p_true": 0.65,
      "what_is_wrong": "...",
      "should_be": {...}
    },
    // ... all 70 issues
  ]
}
```

**Implementation**: Modify Reflector to:
- Track ALL issues (not just examples)
- Include full evidence text for each
- Add issue_id for tracking
- Support sampling if >200 issues

### 2. Validation Testing Script

**Script**: `scripts/validate_ace_improvements.py`

**Inputs**:
- Previous version reflection (e.g., `reflection_v11_2_2_detailed.json`)
- New version extraction script (e.g., `extract_kg_v12_book.py`)

**Process**:
1. Load ALL issues from previous Reflector output
2. Calculate sample size (95% confidence, ±10% margin):
   ```python
   n = (1.96^2 * p * (1-p)) / (0.1^2)
   # For p=0.08 (8% error rate): n ≈ 28 issues
   # Round up to 30-50 for safety
   ```
3. Stratified random sampling (proportional to severity):
   - HIGH severity: 30%
   - MEDIUM severity: 50%
   - MILD severity: 20%
4. For each sampled issue:
   - Extract evidence text + context
   - Run full pipeline (Pass 1 → Pass 2 → Pass 2.5)
   - Compare with original issue
5. Calculate metrics:
   - **Fixed**: Issue no longer present in final output
   - **Still present**: Issue appears in final output
   - **New issue**: Different problem introduced
   - **Improvement %**: (Fixed - New) / Total * 100

**Output**: `validation_report_vN.json`

```json
{
  "sample_size": 30,
  "issues_fixed": 22,
  "issues_still_present": 5,
  "new_issues_introduced": 3,
  "improvement_percent": 63.3,
  "confidence_level": 0.95,
  "margin_of_error": 0.10,
  "recommendation": "PROCEED",  // or "REVIEW" or "ITERATE"
  "details": [...]
}
```

### 3. Automated ACE Orchestrator

**Script**: `scripts/run_ace_cycle_automated.py`

**Configuration**:
```python
config = {
    "validation_threshold": 0.30,  # 30% improvement required
    "max_curator_iterations": 3,
    "sample_size": 30,
    "auto_proceed": True,  # If False, ask for confirmation
}
```

**Workflow**:
```python
def run_automated_ace_cycle(base_version):
    # 1. Reflector (already done for base_version)

    # 2. Run Curator
    changeset = run_curator(base_version)

    # 3. Apply changeset
    apply_changeset(changeset)

    # 4. Validation gate
    validation = validate_improvements(
        base_version=base_version,
        new_version=base_version + 1,
        sample_size=30
    )

    # 5. Automated decision
    if validation.improvement_percent < config["validation_threshold"]:
        if curator_iterations < config["max_curator_iterations"]:
            print(f"❌ Improvement {validation.improvement_percent:.1f}% < threshold")
            print("🔄 Iterating Curator...")
            restore_baseline()
            return run_automated_ace_cycle(base_version)  # Retry
        else:
            print("❌ Max Curator iterations reached. Manual review required.")
            return False

    # 6. Proceed with full extraction
    print(f"✅ Validation passed: {validation.improvement_percent:.1f}% improvement")
    run_full_extraction(base_version + 1)

    # 7. Run Reflector on new version
    reflection_new = run_reflector(base_version + 1)

    # 8. Compare metrics
    comparison = compare_versions(base_version, base_version + 1)

    if comparison.error_rate < 0.045:  # <4.5% target
        print("🎉 SUCCESS! Quality target achieved.")
        return True
    else:
        print("🔄 Quality target not met. Starting next ACE cycle...")
        return run_automated_ace_cycle(base_version + 1)
```

### 4. Enhanced Reflector (All Issues Output)

**Modify**: `src/ace_kg/kg_reflector.py`

**Add method**:
```python
def _generate_detailed_issues_list(self) -> List[Dict[str, Any]]:
    """Generate complete list of ALL issues (not just examples)"""
    all_issues = []
    issue_id = 1

    for category in self.issue_categories:
        for issue in category.all_issues:  # Not just .examples
            all_issues.append({
                'issue_id': f"{self.version}_{issue_id:03d}",
                'category': category.name,
                'severity': category.severity,
                'source': issue.source,
                'relationship': issue.relationship,
                'target': issue.target,
                'evidence_text': issue.evidence_text,
                'page': issue.page,
                'p_true': issue.p_true,
                'what_is_wrong': issue.what_is_wrong,
                'should_be': issue.should_be
            })
            issue_id += 1

    return all_issues
```

## Statistical Sampling Methodology

**Goal**: Test enough issues to be 95% confident that observed improvement rate is within ±10% of true rate.

**Formula** (for binomial proportion):
```
n = (Z^2 * p * (1-p)) / E^2

Where:
- Z = 1.96 (95% confidence)
- p = estimated error rate (e.g., 0.08 for 8%)
- E = margin of error (0.10 for ±10%)
```

**For V11.2.2** (70 issues, 7.86% rate):
```
n = (1.96^2 * 0.0786 * (1-0.0786)) / (0.1^2)
n = (3.84 * 0.0724) / 0.01
n ≈ 28 issues
```

**Recommendation**: Sample 30-50 issues (adds safety margin)

**Stratification**: Proportional to severity distribution
- HIGH: 4 issues (13%)
- MEDIUM: 47 issues (67%)
- MILD: 19 issues (27%)

Sample:
- HIGH: 4 issues (all of them, since only 4)
- MEDIUM: 20 issues
- MILD: 10 issues
- **Total: 34 issues** (well-powered for 95% confidence)

## Implementation Priority

### Phase 1 (Immediate)
1. ✅ Fix test script Pydantic models
2. ✅ Enhance Reflector to output ALL issues
3. ✅ Create `validate_ace_improvements.py`

### Phase 2 (Next)
4. Create `run_ace_cycle_automated.py`
5. Add configuration management
6. Add visualization dashboard

### Phase 3 (Future)
7. Continuous learning: Track which improvements work across cycles
8. Meta-meta-ACE: Learn optimal Curator strategies
9. Automated A/B testing of different improvement approaches

## Benefits

1. **No more guessing**: Statistically rigorous validation before full extraction
2. **Fast iteration**: Test improvements in 5-10 minutes vs 30-40 minutes full extraction
3. **Objective decisions**: Automated thresholds replace manual judgment
4. **Full automation**: Entire ACE cycle runs without human intervention
5. **Continuous improvement**: System learns from each cycle

## Success Metrics

- **Time saved**: 30 min full extraction → 5 min validation test
- **Accuracy**: 95% confidence in improvement estimates
- **False starts prevented**: Reject bad improvements before full extraction
- **Cycle speed**: Complete ACE cycle in <1 hour (vs 2-3 hours manual)

## Next Steps

1. Wait for V12 extraction to complete
2. Run Reflector on V12 to establish baseline
3. Implement Phase 1 components
4. Test on V12 → V13 transition
5. Fully automate for future cycles
