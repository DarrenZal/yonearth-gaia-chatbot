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

print("Fixing JSON - escaping double quotes only")
print()

# Split into lines
lines = json_str.split('\n')

# Fix specific problematic patterns
for i in range(len(lines)):
    line = lines[i]

    # Pattern 1: "25x'25"" -> \"25x'25\"
    if 'Founding Director, "25x\'25""' in line:
        lines[i] = line.replace('"25x\'25""', '\\"25x\'25\\"')
        print(f"✓ Fixed line {i+1}: 25x'25 quotes")

    # Pattern 2: ""Aaron... -> \"Aaron...
    if '"evidence_text": ""Aaron' in line:
        lines[i] = line.replace('"evidence_text": ""Aaron', '"evidence_text": "\\"Aaron')
        lines[i] = lines[i].replace('ways.",', 'ways.\\"",')
        print(f"✓ Fixed line {i+1}: Aaron Perry quotes")

    # Pattern 3: ""Soil... -> \"Soil...
    if '"evidence_text": ""Soil is the answer' in line:
        lines[i] = line.replace('"evidence_text": ""Soil', '"evidence_text": "\\"Soil')
        lines[i] = lines[i].replace('together.",', 'together.\\"",')
        print(f"✓ Fixed line {i+1}: Soil quotes")

json_str_fixed = '\n'.join(lines)

# Try parsing
try:
    parsed = json.loads(json_str_fixed)
    print("\n✅✅✅ SUCCESS! JSON PARSED ✅✅✅\n")

    with open(output_path, 'w') as f:
        json.dump(parsed, f, indent=2)

    print(f"✅ Saved: {output_path}\n")
    print(f"📊 Content:")
    print(f"   - Total issues: {parsed['quality_summary']['total_issues']}")
    print(f"   - Recommendations: {len(parsed.get('improvement_recommendations', []))}")
    print(f"   - Issue categories: {len(parsed.get('issue_categories', []))}")
    print(f"   - Examples: {sum(len(cat.get('examples', [])) for cat in parsed.get('issue_categories', []))}")

except json.JSONDecodeError as e:
    print(f"\n❌ Error at line {e.lineno}, col {e.colno}: {e.msg}")
    lines_fixed = json_str_fixed.split('\n')
    for i in range(max(0, e.lineno - 3), min(len(lines_fixed), e.lineno + 2)):
        marker = ' >>> ' if i == e.lineno - 1 else '     '
        print(f'{marker}{i+1:3d}: {lines_fixed[i][:150]}')
