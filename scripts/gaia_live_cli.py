#!/usr/bin/env python3
"""
Simple CLI to talk to the real Gaia LLM using your local .env configuration.

Usage:
  # From the project root, with your venv active and .env configured:
  python scripts/gaia_live_cli.py "Tell me about regenerative agriculture"

If no message is provided on the command line, the script will prompt you
interactively.
"""

import os
import sys
from pathlib import Path


def main() -> int:
    # Ensure project root is on sys.path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    # Import here so that settings loads .env once path is set
    from src.character.gaia import GaiaCharacter  # type: ignore[import]

    if len(sys.argv) > 1:
        user_input = " ".join(sys.argv[1:])
    else:
        try:
            user_input = input("You: ").strip()
        except EOFError:
            return 0
        if not user_input:
            print("No input provided.")
            return 0

    print("Initializing Gaia with real LLM (using .env)...")
    gaia = GaiaCharacter()

    response = gaia.generate_response(
        user_input=user_input,
        retrieved_docs=[],
        session_id="cli-session",
    )

    print("\nGaia:")
    print(response.get("response", "No response"))

    # Optionally show basic citation info if present
    citations = response.get("citations") or []
    if citations:
        print("\nCitations:")
        for c in citations:
            ep = c.get("episode_number", "unknown")
            title = c.get("title", "Unknown Episode")
            guest = c.get("guest_name", "Guest")
            print(f"  - Episode {ep}: {title} (with {guest})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

