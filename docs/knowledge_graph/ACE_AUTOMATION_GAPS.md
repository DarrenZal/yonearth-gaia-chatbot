# ACE System Automation Gaps

## Issues Identified (2025-10-14)

### ❌ Gap #1: Extraction Script Not Auto-Generated

**Problem**: After Curator creates V12 prompts, the extraction script must be created manually.

**Current State**:
- Curator generates: `pass1_extraction_v12.txt`, `pass2_evaluation_v12.txt`
- Missing: `extract_kg_v12_book.py`
- Manual workaround required: Copy V11.2.2 script → update paths

**Why This Matters**:
- Breaks full automation of ACE cycle
- Requires human intervention
- Error-prone (manual path updates)

**Proposed Fix**:
- Curator should include `NEW_SCRIPT` operation type in changesets
- Applicator should generate extraction scripts from template
- Template should parameterize: `{version}`, `{prompt_paths}`, `{output_dir}`

**Better Solution**:
- Create single `extract_kg_generic.py --version=12` script
- No more version-specific scripts needed
- Just update version parameter

---

### ❌ Gap #2: No Targeted Testing on Problematic Chunks

**Problem**: After changes applied, system jumps straight to full extraction (slow, no immediate feedback).

**Current State**:
- V11.2.2 Reflector identifies 70 specific problematic chunks
- V12 changes claim to fix these issues
- No verification before full extraction
- Full extraction takes ~30 minutes

**Why This Matters**:
- Wastes time if fixes don't work
- No early validation
- Can't iterate quickly

**Proposed Fix**:
- Run `test_v12_on_issues.py` FIRST (created 2025-10-14)
- Re-extract only the 70 problematic chunks with V12
- Quick validation (~5 minutes vs 30 minutes)
- Only proceed to full extraction if targeted test passes

**Workflow**:
1. Reflector identifies issues → stores evidence text
2. V12 created → test on problematic chunks FIRST
3. If targeted test shows improvement → full extraction
4. If targeted test fails → iterate on fixes

---

## Implementation Priority

**High Priority** (implement next):
1. ✅ Targeted testing script (completed 2025-10-14)
2. Generic extraction script with version parameter
3. Auto-generation of extraction scripts in Curator

**Medium Priority**:
4. Validation that V12 fixes actually address Reflector issues
5. Regression testing framework (ensure fixes don't break other things)

**Low Priority**:
6. Full end-to-end automation (no human intervention)

---

## Meta-ACE Insight

**This conversation revealed a meta-ACE learning**: 
The Curator improved itself (file path scanner) but didn't consider **what else should be automated**.

Future Curator improvements should include:
- Extraction script generation
- Test script generation
- Validation scripts
- Documentation updates

The ACE agents should be able to improve the ACE framework itself - **true meta-ACE**.
