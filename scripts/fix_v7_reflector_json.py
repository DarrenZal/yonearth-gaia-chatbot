#!/usr/bin/env python3
"""Fix JSON parse error in V7 Reflector output"""

import json
import sys
from pathlib import Path

# Load the file with parse error
reflector_path = Path("/home/claudeuser/yonearth-gaia-chatbot/kg_extraction_playbook/analysis_reports/reflection_v7_meta_ace_20251012_224756.json")

with open(reflector_path) as f:
    data = json.load(f)

if "raw_response" in data:
    raw = data["raw_response"]

    # Extract JSON from markdown
    start = raw.find("```json") + 7
    end = raw.find("```", start)
    json_str = raw[start:end].strip()

    # Try to parse and identify error
    try:
        parsed = json.loads(json_str)
        print("✅ SUCCESS - JSON is valid")

        # Save cleaned version
        output_path = reflector_path.parent / "reflection_v7_meta_ace_20251012_224756_cleaned.json"
        with open(output_path, 'w') as f:
            json.dump(parsed, f, indent=2)

        print(f"✅ Saved cleaned JSON to: {output_path}")

    except json.JSONDecodeError as e:
        print(f"❌ ERROR at line {e.lineno}, column {e.colno}: {e.msg}")
        print(f"Position: {e.pos}")
        print("\nContext around error:")
        lines = json_str.split('\n')
        start_line = max(0, e.lineno - 5)
        end_line = min(len(lines), e.lineno + 3)
        for i in range(start_line, end_line):
            marker = ' >>> ' if i == e.lineno - 1 else '     '
            print(f'{marker}{i+1:3d}: {lines[i]}')

        print(f"\nCharacter at error position: '{json_str[e.pos:e.pos+10]}'")

        # Try to find the malformed part
        print("\nLet me try to find and fix the error...")

        # Common JSON error: missing comma or quote
        # The error is at line 37, column 153 (char 1788)
        print(f"\nSearching around character {e.pos}...")
        context_start = max(0, e.pos - 100)
        context_end = min(len(json_str), e.pos + 100)
        print(f"Context: ...{json_str[context_start:context_end]}...")
