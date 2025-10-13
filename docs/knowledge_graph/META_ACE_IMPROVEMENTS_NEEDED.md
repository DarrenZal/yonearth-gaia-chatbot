# Meta-ACE Improvements: Fixing the Curator & Applicator

**Date**: 2025-10-13
**Status**: Identified Issues, Ready to Implement
**Context**: V10 Curator hit token limits due to design flaw

---

## 🔍 Issues Identified

### 1. Curator JSON Truncation (CRITICAL)
**Problem**: Curator generates responses >16K tokens, JSON gets cut off
**Root Cause**: Curator includes FULL CODE BLOCKS in `edit_details` fields
**Example**: Trying to include 500-line code rewrites in JSON

### 2. Wrong Separation of Concerns
**Current Design**:
- Curator tries to be both strategic AND tactical
- Generates complete code/prompt rewrites
- Massive JSON output

**Should Be**:
- **Curator (Strategic)**: WHAT to change + WHY
- **Applicator (Tactical)**: HOW to change it (reads files, makes edits)

### 3. No Generic Applicator Script
We have version-specific scripts (`run_curator_on_v9.py`) instead of generic autonomous ones.

---

## ✅ What We've Fixed So Far

1. **✅ Improved Reflector**:
   - Added duplicate relationship detection (automatic)
   - Added predicate fragmentation analysis (automatic)
   - Pre-computes statistics before sending to Claude
   - Committed: `355a5dd`

2. **✅ Created Generic Scripts**:
   - `run_curator_generic.py` - works on ANY version
   - Finds latest Reflector analysis automatically
   - No more version-specific scripts needed!

---

## 🎯 Fixes Needed

### Fix 1: Redesign Curator Output (HIGH PRIORITY)

**Current JSON Structure** (TOO VERBOSE):
```json
{
  "file_operations": [{
    "edit_details": {
      "old_content": "500 lines of code here...",  // ❌ TOO BIG
      "new_content": "500 lines of code here..."   // ❌ TOO BIG
    }
  }]
}
```

**New JSON Structure** (CONCISE):
```json
{
  "file_operations": [{
    "operation_id": "change_001",
    "operation_type": "CODE_FIX",
    "file_path": "modules/dedication_parser.py",
    "priority": "CRITICAL",
    "change_description": "Rewrite process_batch() to use single parsing strategy instead of concatenating multiple strategies",
    "affected_function": "DedicationParser.process_batch",
    "change_type": "function_rewrite",  // or "regex_fix", "add_section", etc.
    "guidance": {
      "current_issue": "Creating 6+ relationships per dedication by running multiple parsers",
      "fix_approach": "Use only comma-splitting, remove full-target append, deduplicate",
      "test_with": "Example: 'dedicated to my two children, Osha and Hunter' should create 2 relationships"
    }
  }]
}
```

**Benefits**:
- JSON ~10x smaller (1-2K tokens instead of 15K+)
- No truncation issues
- Applicator reads actual files and makes surgical edits
- More maintainable

### Fix 2: Improve Applicator Intelligence (HIGH PRIORITY)

**Current Applicator** (`_apply_file_operation`):
- Simple string replacement: `content.replace(old_content, new_content)`
- Requires exact matches
- Brittle and error-prone

**Improved Applicator**:
```python
def _apply_file_operation_intelligent(self, operation):
    """Apply changes using Claude Code to read and understand context."""
    file_path = operation['file_path']
    change_desc = operation['change_description']
    function_name = operation.get('affected_function')

    # Read current file
    current_content = self._read_file(file_path)

    # Ask Claude to make the specific change
    prompt = f"""
    File: {file_path}
    Function: {function_name}

    Current Code:
    {current_content}

    Change Needed:
    {change_desc}

    Guidance:
    {operation['guidance']}

    Please provide ONLY the modified version of the file.
    """

    # Use Claude to generate the fixed version
    modified_content = self._ask_claude_to_fix(prompt)

    # Write back
    self._write_file(file_path, modified_content)
```

**Benefits**:
- Applicator can understand context
- Makes intelligent edits
- Handles variations in code
- More robust than string matching

### Fix 3: Add Iterative Improvement Loop (MEDIUM PRIORITY)

**Current**: One-shot fixes
**Better**: Validate each fix, iterate if needed

```python
def apply_and_validate(self, operation):
    """Apply change, then validate it worked."""
    # Apply change
    self._apply_file_operation_intelligent(operation)

    # Run quick validation
    if operation.get('validation'):
        test_result = self._run_validation(operation['validation'])
        if not test_result.passed:
            # Rollback and try different approach
            self._rollback()
            return self._retry_with_different_approach(operation)

    return {"success": True}
```

---

## 📋 Implementation Plan

### Phase 1: Curator Redesign (2-3 hours) ✅ COMPLETE
1. ✅ Modified `_get_curator_system_prompt()` to use concise JSON schema
2. ✅ Removed `old_content`/`new_content` from edit_details
3. ✅ Added `change_description` and `guidance` fields
4. ✅ Tested on V10 Reflector report - output is ~7.5K tokens (no truncation!)

**Results**:
- ✅ Valid JSON with 12 operations
- ✅ No truncation errors
- ✅ Expected impact: 13.6% → 4.8% error rate
- ✅ Uses new concise format throughout

### Phase 2: Applicator Enhancement (3-4 hours) ✅ COMPLETE
1. ✅ Created `_apply_operation_intelligent()` method
2. ✅ Uses Claude to read files and make changes
3. ✅ Added validation hooks in changeset structure
4. ✅ Maintains backward compatibility with legacy format

**Implementation**:
- New method reads current file content
- Builds prompts for Claude to apply strategic guidance
- Handles CODE_FIX, PROMPT_ENHANCEMENT, CONFIG_UPDATE, NEW_MODULE
- Temperature 0.2 for precise code changes

### Phase 3: Generic Scripts (1 hour) ✅ COMPLETE
1. ✅ Already done: `run_curator_generic.py`
2. ⏸️ Create: `apply_changeset_generic.py` (deferred - can use Curator's apply_changeset method)
3. ✅ Dry-run mode available in `apply_changeset()` method

### Phase 4: End-to-End Test (2 hours) ⏸️ READY TO RUN
1. ✅ Run Reflector on V10 (already done)
2. ✅ Run improved Curator (just completed - 12 operations)
3. ⏸️ Apply changes with improved Applicator (next step)
4. ⏸️ Validate V11 is better

**Status**: Meta-ACE improvements committed to repository (commit: 4940cf0)

---

## 🎯 Expected Outcomes

**Before (Current)**:
- Curator hits 16K token limit ❌
- JSON gets truncated ❌
- Can't apply changes ❌

**After (Improved)**:
- Curator generates <5K token JSON ✅
- Complete, valid JSON always ✅
- Applicator intelligently applies changes ✅
- True autonomous ACE cycle ✅

---

## 💡 Key Insight

**The ACE paper's principle**: "Treat contexts as evolving playbooks"

We were violating this by having the Curator try to rewrite the entire playbook. Instead:

- **Curator**: Strategic advisor (WHAT + WHY)
- **Applicator**: Tactical executor (HOW)
- **Reflector**: Quality analyst (FEEDBACK)

This separation of concerns is the true ACE architecture.

---

## 🚀 Next Steps

1. **Immediate**: Implement Phase 1 (Curator redesign)
2. **Short-term**: Implement Phase 2 (Applicator enhancement)
3. **Medium-term**: Complete Phase 3-4 (testing)
4. **Long-term**: Document patterns for future meta-improvements

The system should be able to improve itself at any level - including improving the improvement agents themselves. This is meta-ACE.

