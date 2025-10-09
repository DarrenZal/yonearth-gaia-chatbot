#!/usr/bin/env python3
"""
Wrapper script to run relationship extraction with proper environment loading
"""
import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
project_root = Path(__file__).parent.parent
env_path = project_root / ".env"

print(f"Loading environment from: {env_path}")
load_dotenv(env_path)

# Verify API key is loaded
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("ERROR: OPENAI_API_KEY not found in environment!")
    sys.exit(1)

print(f"API key loaded: {api_key[:10]}...")

# Run the extraction script
script_path = project_root / "scripts" / "extract_all_relationships_comprehensive.py"
print(f"\nRunning: python3 {script_path} --yes\n")

result = subprocess.run(
    [sys.executable, str(script_path), "--yes"],
    env=os.environ.copy()
)

sys.exit(result.returncode)
