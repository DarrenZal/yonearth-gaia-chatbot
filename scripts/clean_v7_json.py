#!/usr/bin/env python3
import json
from pathlib import Path

# Load
input_path = Path("/home/claudeuser/yonearth-gaia-chatbot/kg_extraction_playbook/analysis_reports/reflection_v7_meta_ace_20251012_224756.json")
output_path = input_path.parent / "reflection_v7_meta_ace_20251012_224756_cleaned.json"

with open(input_path) as f:
    data = json.load(f)

raw = data['raw_response']

# Extract JSON from markdown
start = raw.find('```json') + 7
end = raw.find('```', start)
json_str = raw[start:end].strip()

# Fix the malformed string
json_str = json_str.replace('"25x\'25""', '"25x\'25"')

# Parse
parsed = json.loads(json_str)

# Save cleaned version
with open(output_path, 'w') as f:
    json.dump(parsed, f, indent=2)

print(f"✅ Fixed and saved to: {output_path}")
print(f"   Total issues: {parsed['quality_summary']['total_issues']}")
print(f"   Recommendations: {len(parsed.get('improvement_recommendations', []))}")
