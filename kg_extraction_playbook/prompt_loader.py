"""
Prompt Loader for ACE-Managed KG Extraction

Loads version-controlled prompts from the playbook/prompts directory.
This allows the ACE system (Reflector + Curator) to analyze and modify prompts.
"""

from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class PromptLoader:
    """
    Loads and manages version-controlled extraction prompts.

    Prompts are stored in kg_extraction_playbook/prompts/ and can be
    modified by the ACE Curator to improve extraction quality.
    """

    def __init__(self, playbook_path: str = None):
        if playbook_path is None:
            # Default to playbook directory
            playbook_path = Path(__file__).parent

        self.playbook_path = Path(playbook_path)
        self.prompts_dir = self.playbook_path / "prompts"

        if not self.prompts_dir.exists():
            raise FileNotFoundError(f"Prompts directory not found: {self.prompts_dir}")

        self._cache = {}

    def load_prompt(self, prompt_name: str, version: Optional[str] = None) -> str:
        """
        Load a prompt by name and optional version.

        Args:
            prompt_name: Name of the prompt (e.g., "pass1_extraction", "pass2_evaluation")
            version: Optional version string (e.g., "v7"). If None, loads latest.

        Returns:
            Prompt text with {placeholders} for formatting

        Example:
            loader = PromptLoader()
            prompt = loader.load_prompt("pass1_extraction", "v7")
            formatted = prompt.format(text=chunk_text)
        """
        # Build filename
        if version:
            filename = f"{prompt_name}_{version}.txt"
        else:
            # Find latest version
            matching_files = sorted(self.prompts_dir.glob(f"{prompt_name}_*.txt"))
            if not matching_files:
                raise FileNotFoundError(f"No prompts found matching: {prompt_name}")
            filename = matching_files[-1].name

        prompt_path = self.prompts_dir / filename

        # Check cache
        cache_key = str(prompt_path)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Load from file
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

        with open(prompt_path, 'r') as f:
            prompt_text = f.read()

        # Cache for performance
        self._cache[cache_key] = prompt_text

        logger.info(f"✅ Loaded prompt: {filename}")

        return prompt_text

    def get_available_prompts(self) -> Dict[str, list]:
        """
        Get all available prompts grouped by name.

        Returns:
            Dictionary mapping prompt names to list of versions

        Example:
            {
                "pass1_extraction": ["v5", "v6", "v7"],
                "pass2_evaluation": ["v5", "v6", "v7"]
            }
        """
        prompts = {}

        for prompt_file in self.prompts_dir.glob("*.txt"):
            # Parse filename: "pass1_extraction_v7.txt" -> name="pass1_extraction", version="v7"
            parts = prompt_file.stem.rsplit('_', 1)
            if len(parts) == 2:
                name, version = parts
                if name not in prompts:
                    prompts[name] = []
                prompts[name].append(version)

        # Sort versions
        for name in prompts:
            prompts[name] = sorted(prompts[name])

        return prompts

    def save_prompt(self, prompt_name: str, version: str, content: str) -> Path:
        """
        Save a new prompt version (used by ACE Curator).

        Args:
            prompt_name: Name of the prompt (e.g., "pass1_extraction")
            version: Version string (e.g., "v8")
            content: Prompt text content

        Returns:
            Path to saved prompt file
        """
        filename = f"{prompt_name}_{version}.txt"
        prompt_path = self.prompts_dir / filename

        with open(prompt_path, 'w') as f:
            f.write(content)

        # Invalidate cache
        cache_key = str(prompt_path)
        if cache_key in self._cache:
            del self._cache[cache_key]

        logger.info(f"✅ Saved prompt: {filename}")

        return prompt_path

    def get_prompt_metadata(self, prompt_name: str, version: str) -> Dict:
        """
        Get metadata about a prompt (line count, character count, etc.)

        Useful for ACE Reflector analysis.
        """
        prompt_text = self.load_prompt(prompt_name, version)

        lines = prompt_text.split('\n')

        # Count placeholder variables
        import re
        placeholders = set(re.findall(r'\{(\w+)\}', prompt_text))

        return {
            "prompt_name": prompt_name,
            "version": version,
            "line_count": len(lines),
            "char_count": len(prompt_text),
            "word_count": len(prompt_text.split()),
            "placeholders": list(placeholders),
            "has_examples": "example:" in prompt_text.lower() or "✅" in prompt_text or "❌" in prompt_text
        }


# Global instance for convenience
_global_loader = None

def get_prompt_loader(playbook_path: str = None) -> PromptLoader:
    """Get or create global PromptLoader instance."""
    global _global_loader
    if _global_loader is None:
        _global_loader = PromptLoader(playbook_path)
    return _global_loader


if __name__ == "__main__":
    # Test the loader
    loader = PromptLoader()

    print("Available prompts:")
    for name, versions in loader.get_available_prompts().items():
        print(f"  {name}: {', '.join(versions)}")

    print("\nLoading pass1_extraction_v7:")
    prompt = loader.load_prompt("pass1_extraction", "v7")
    print(f"  Length: {len(prompt)} characters")

    metadata = loader.get_prompt_metadata("pass1_extraction", "v7")
    print(f"\nMetadata: {metadata}")
