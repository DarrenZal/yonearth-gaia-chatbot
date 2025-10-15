#!/usr/bin/env python3
"""
Run Curator on V14 to generate V14.1 improvements.

Based on V14 test results and root cause analysis.
Target: A-grade (< 5% issue rate, 90%+ test pass rate)
"""

import os
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Paths
PROMPTS_DIR = Path("/home/claudeuser/yonearth-gaia-chatbot/kg_extraction_playbook/prompts")
ANALYSIS_DIR = Path("/home/claudeuser/yonearth-gaia-chatbot/kg_extraction_playbook/analysis_reports")
CHANGESETS_DIR = Path("/home/claudeuser/yonearth-gaia-chatbot/kg_extraction_playbook/changesets")

V14_PASS1 = PROMPTS_DIR / "pass1_extraction_v14.txt"
V14_PASS2 = PROMPTS_DIR / "pass2_evaluation_v14.txt"
V14_ANALYSIS = ANALYSIS_DIR / "v14_analysis_for_curator.md"

# Load files
print("="*80)
print("🎨 CURATOR: V14 → V14.1")
print("="*80)
print()

print("📂 Loading files...")
v14_pass1 = V14_PASS1.read_text()
v14_pass2 = V14_PASS2.read_text()
v14_analysis = V14_ANALYSIS.read_text()
print(f"✅ Loaded V14 Pass 1 ({len(v14_pass1)} chars)")
print(f"✅ Loaded V14 Pass 2 ({len(v14_pass2)} chars)")
print(f"✅ Loaded V14 Analysis ({len(v14_analysis)} chars)")
print()

# Curator prompt
CURATOR_PROMPT = f"""You are the CURATOR in a Meta-ACE (Analyze-Curate-Extract) framework for knowledge graph extraction.

Your role: Generate targeted improvements based on performance analysis.

## Context

**Current Version**: V14
**Target Version**: V14.1
**Goal**: Achieve A-grade (< 5% issue rate, 90%+ test pass rate on V13.1 issues)

## V14 Performance Analysis

{v14_analysis}

## Current V14 Prompts

### Pass 1 Extraction Prompt (V14)
```
{v14_pass1}
```

### Pass 2 Evaluation Prompt (V14)
```
{v14_pass2}
```

## Your Task

Generate V14.1 improvements that address the identified issues while maintaining V14's successful filtering.

**Key Requirements**:

1. **REDUCE confidence penalties** as specified in the analysis:
   - TESTABLE_CLAIM: -0.05 (was -0.15)
   - OPINION: -0.10 (was -0.25)
   - NORMATIVE: -0.15 (was -0.30)
   - PHILOSOPHICAL_CLAIM: -0.30 (was -0.50)

2. **Refine entity specificity constraints** to allow resolvable demonstratives
   - Don't auto-reject "this", "that" - extract and let postprocessing resolve
   - Allow compound phrases
   - Add guidance on when demonstratives are acceptable

3. **Maintain classification_flags** (they work correctly)
   - Keep FACTUAL, TESTABLE_CLAIM, PHILOSOPHICAL_CLAIM, NORMATIVE, OPINION
   - The detection logic is excellent; only penalties need adjustment

4. **Keep all other V14 improvements**
   - Semantic predicate validation
   - Contextual awareness
   - Abstract pattern detection

## Output Format

Provide your response as a JSON changeset with the following structure:

```json
{{
  "version": "v14.1",
  "source_version": "v14",
  "curator_analysis": "Brief explanation of changes and rationale",
  "target_metrics": {{
    "test_pass_rate": "90%+",
    "issue_rate": "<5%",
    "pipeline_reduction": "30-40%"
  }},
  "changes": [
    {{
      "prompt": "pass1_extraction",
      "change_type": "refinement",
      "description": "Specific change description",
      "before": "Text to find in V14 prompt",
      "after": "Text to replace with in V14.1"
    }},
    {{
      "prompt": "pass2_evaluation",
      "change_type": "adjustment",
      "description": "Specific change description",
      "before": "Text to find in V14 prompt",
      "after": "Text to replace with in V14.1"
    }}
  ]
}}
```

**Important**:
- Make specific, surgical changes (not wholesale rewrites)
- Include exact "before" and "after" text for each change
- Ensure all changes align with the analysis recommendations
- Balance filtering vs preservation

Generate the V14.1 changeset now.
"""

print("🎨 Running Curator (GPT-4)...")
print("   Generating V14.1 improvements...")
print()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": CURATOR_PROMPT}
    ],
    temperature=0.3,
    response_format={"type": "json_object"}
)

changeset_json = response.choices[0].message.content

# Parse and save changeset
changeset = json.loads(changeset_json)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
changeset_file = CHANGESETS_DIR / f"v14_1_changeset_{timestamp}.json"

with open(changeset_file, 'w') as f:
    json.dump(changeset, f, indent=2)

print("="*80)
print("✅ CURATOR COMPLETE")
print("="*80)
print()
print(f"📁 Changeset saved: {changeset_file}")
print()
print("📊 V14.1 CHANGESET SUMMARY")
print("="*80)
print()
print(f"**Version**: {changeset.get('version', 'v14.1')}")
print(f"**Source**: {changeset.get('source_version', 'v14')}")
print()
print("**Curator Analysis**:")
print(changeset.get('curator_analysis', 'N/A'))
print()
print(f"**Changes**: {len(changeset.get('changes', []))} modifications")
for i, change in enumerate(changeset.get('changes', []), 1):
    print(f"  {i}. [{change['prompt']}] {change['description']}")
print()

print("📋 Target Metrics:")
target = changeset.get('target_metrics', {})
for key, value in target.items():
    print(f"  - {key}: {value}")
print()

print("="*80)
print("🎯 NEXT STEPS")
print("="*80)
print()
print("1. Review changeset:")
print(f"   cat {changeset_file}")
print()
print("2. Apply changeset:")
print(f"   python3 scripts/apply_changeset_generic.py {changeset_file}")
print()
print("3. Test V14.1 on V13.1 issues:")
print("   python3 scripts/test_v14_on_v13_1_issues.py")
print()
print("4. If test pass rate ≥ 90%:")
print("   python3 scripts/run_v14_1_from_v12_checkpoint.py")
print()
print("="*80)
