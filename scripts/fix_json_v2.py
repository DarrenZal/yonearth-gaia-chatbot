#!/usr/bin/env python3
import json
from pathlib import Path

input_path = Path("/home/claudeuser/yonearth-gaia-chatbot/kg_extraction_playbook/analysis_reports/reflection_v7_meta_ace_20251012_224756.json")
output_path = input_path.parent / "reflection_v7_meta_ace_20251012_224756_fixed.json"

with open(input_path) as f:
    data = json.load(f)

raw = data['raw_response']
start = raw.find('```json') + 7
end = raw.find('```', start)
json_str = raw[start:end].strip()

print("Fixing JSON with proper escaping...")
print()

# The key insight: in the JSON string, we need \" not \\"
# So when doing Python string replacement, we use r'\"' or '\\"'

# But actually, let me check what we're replacing:
# Original JSON has: "evidence_text": "...text with "nested" quotes...",
# We want: "evidence_text": "...text with \"nested\" quotes...",

# In Python strings:
# Original: '"25x\'25""'  (this is: "25x'25"" with the single quote escaped for Python)
# Want: '\"25x\'25\"'  (this is: \"25x'25\" with quotes escaped for JSON)

# Use raw string to avoid confusion
json_str = json_str.replace(
    '"25x\'25""',
    r'\"25x\'25\"'
)
print("✓ Fixed: 25x'25 quotes")

json_str = json_str.replace(
    '"evidence_text": ""Aaron Perry',
    r'"evidence_text": "\"Aaron Perry'
)
json_str = json_str.replace(
    'ways.",',
    r'ways.\"",')
print("✓ Fixed: Aaron Perry quotes")

json_str = json_str.replace(
    '"evidence_text": ""Soil is the answer',
    r'"evidence_text": "\"Soil is the answer'
)
json_str = json_str.replace(
    'together.",',
    r'together.\"",',
)
print("✓ Fixed: Soil quotes")

# Try parsing
try:
    parsed = json.loads(json_str)
    print("\n✅✅✅ SUCCESS!!! ✅✅✅\n")

    with open(output_path, 'w') as f:
        json.dump(parsed, f, indent=2)

    print(f"✅ Saved: {output_path}\n")
    print(f"📊 Full V7 Reflector Analysis Content:")
    print(f"   - Total issues: {parsed['quality_summary']['total_issues']}")
    print(f"   - Critical: {parsed['quality_summary']['critical_issues']}")
    print(f"   - High: {parsed['quality_summary']['high_priority_issues']}")
    print(f"   - Medium: {parsed['quality_summary']['medium_priority_issues']}")
    print(f"   - Mild: {parsed['quality_summary']['mild_issues']}")
    print(f"   - Recommendations: {len(parsed.get('improvement_recommendations', []))}")
    print(f"   - Issue categories: {len(parsed.get('issue_categories', []))}")
    print(f"   - Novel error patterns: {len(parsed.get('novel_error_patterns', []))}")

    total_examples = sum(len(cat.get('examples', [])) for cat in parsed.get('issue_categories', []))
    print(f"   - Total detailed examples: {total_examples}")
    print()
    print("✅ Ready for Curator!")

except json.JSONDecodeError as e:
    print(f"\n❌ Error at line {e.lineno}, col {e.colno}: {e.msg}")
    lines = json_str.split('\n')
    for i in range(max(0, e.lineno - 3), min(len(lines), e.lineno + 2)):
        marker = ' >>> ' if i == e.lineno - 1 else '     '
        print(f'{marker}{i+1:3d}: {lines[i][:120]}')
