#!/usr/bin/env python3
"""
Run ACE Reflection Cycle

Convenient script to run the ACE (Autonomous Cognitive Entity) framework
for autonomous system evolution.

Usage:
    python scripts/run_ace_cycle.py                  # Interactive mode
    python scripts/run_ace_cycle.py --auto-apply     # Auto-apply low-risk changes
    python scripts/run_ace_cycle.py --summary        # Show evolution summary
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ace.orchestrator import main

if __name__ == "__main__":
    main()
