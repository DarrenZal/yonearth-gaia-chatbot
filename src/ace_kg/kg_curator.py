"""
KG Curator Agent

Uses Claude Sonnet 4.5 to transform Reflector insights into actionable
code/prompt/config changes for the knowledge graph extraction pipeline.
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path

import anthropic


class KGCuratorAgent:
    """
    The KG Curator transforms Reflector insights into executable changes:
    - Organizes recommendations by type and priority
    - Creates specific code modifications
    - Proposes prompt enhancements
    - Designs configuration tweaks
    - Manages system evolution and version control

    Uses Claude Sonnet 4.5 for strategic planning and code generation.
    """

    def __init__(
        self,
        playbook_path: str = "/home/claudeuser/yonearth-gaia-chatbot/kg_extraction_playbook",
        model: str = "claude-sonnet-4-5-20250929"
    ):
        self.playbook_path = Path(playbook_path)
        self.model = model

        # Initialize Anthropic client
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable must be set")

        self.client = anthropic.Anthropic(api_key=api_key)

    def curate_improvements(
        self,
        reflector_report: Dict[str, Any],
        current_version: int,
        playbook_state: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Transform Reflector insights into actionable changeset.

        Args:
            reflector_report: Output from KG Reflector analysis
            current_version: Current extraction version number
            playbook_state: Current Playbook files/configs (optional)

        Returns:
            Changeset with specific file operations to evolve system
        """
        # Load current Playbook state if not provided
        if playbook_state is None:
            playbook_state = self._load_playbook_state()

        # Build curation prompt
        prompt = self._build_curation_prompt(
            reflector_report,
            current_version,
            playbook_state
        )

        # Run Claude Sonnet 4.5 curation
        response = self.client.messages.create(
            model=self.model,
            max_tokens=16000,  # Allow for detailed changesets
            temperature=0.4,  # Slightly creative for solutions
            system=self._get_curator_system_prompt(),
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        # Parse response
        curation_text = response.content[0].text

        # Extract JSON from response
        try:
            if "```json" in curation_text:
                json_start = curation_text.find("```json") + 7
                json_end = curation_text.find("```", json_start)
                json_str = curation_text[json_start:json_end].strip()
            else:
                json_str = curation_text

            changeset = json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"⚠️ Failed to parse JSON from Claude response: {e}")
            print(f"Response text: {curation_text[:500]}...")
            changeset = {
                "error": "json_parse_failed",
                "raw_response": curation_text
            }

        # Enhance with metadata
        changeset["metadata"] = {
            "curation_date": datetime.now().isoformat(),
            "source_version": current_version,
            "target_version": current_version + 1,
            "reflector_analysis_id": reflector_report.get("metadata", {}).get("analysis_date"),
            "curator_version": "1.0_claude",
            "model_used": self.model
        }

        # Save changeset
        self._save_changeset(changeset, current_version)

        return changeset

    def _get_curator_system_prompt(self) -> str:
        """System prompt for Claude Sonnet 4.5 Curator - CONCISE STRATEGIC OUTPUT."""
        return """You are a strategic technical advisor for a knowledge graph extraction system. Your role is to provide CONCISE, HIGH-LEVEL guidance on what to change and why, not HOW to implement it.

**CRITICAL PRINCIPLE**: You provide STRATEGIC direction (WHAT + WHY), not tactical implementation (HOW).

You are curating improvements for a multi-stage KG extraction pipeline:
- **Pass 1**: LLM extraction of entities and relationships from book text
- **Pass 2**: Dual-signal validation (text confidence + knowledge plausibility)
- **Pass 2.5**: Post-processing with 7 modules:
  1. BibliographicCitationParser (fix reversed authorship)
  2. ListSplitter (split comma-separated targets)
  3. PronounResolver (resolve He/She/We to entities)
  4. ContextEnricher (expand vague concepts)
  5. TitleCompletenessValidator (flag incomplete titles)
  6. PredicateValidator (validate semantic compatibility)
  7. FigurativeLanguageFilter (detect metaphors)

THE ENTIRE PIPELINE IS YOUR "PLAYBOOK" - YOU CAN MODIFY ANY PART:
- Python code files (.py)
- Extraction prompts (.txt)
- Configuration files (.json)
- Vocabulary lists, regular expressions, confidence thresholds

CHANGE TYPES YOU CAN PROPOSE:
- **CODE_FIX**: Modify Python code (function logic, regex, algorithms)
- **PROMPT_ENHANCEMENT**: Update extraction prompts (instructions, examples, constraints)
- **CONFIG_UPDATE**: Change configuration values (thresholds, vocabularies)
- **NEW_MODULE**: Add new post-processing module

PROMPT vs CODE: When to Choose Each
- Use **PROMPT_ENHANCEMENT** when errors occur in Pass 1 extraction (before code processing)
- Use **CODE_FIX** when patterns are systematic and detectable with rules
- Use **BOTH** when prompt prevents + code catches any that slip through

⚠️ **CRITICAL OUTPUT RULES** ⚠️

1. **DO NOT include full code blocks or complete file contents**
2. **DO NOT provide old_content/new_content with 100+ lines**
3. **DO provide concise descriptions of WHAT to change and WHY**
4. **DO provide specific guidance on HOW to approach the fix**
5. **Keep JSON output under 5,000 tokens MAXIMUM**

The Applicator agent will read actual files and implement your strategic guidance.

Output ONLY valid JSON with this CONCISE structure:
{
  "changeset_metadata": {
    "source_version": 9,
    "target_version": 10,
    "total_changes": 8,
    "estimated_impact": "Reduces critical issues by 100%, high priority by 60%"
  },
  "file_operations": [
    {
      "operation_id": "change_001",
      "operation_type": "CODE_FIX",
      "file_path": "modules/pass2_5_postprocessing/dedication_parser.py",
      "priority": "CRITICAL",
      "rationale": "Dedication parser creates 6+ malformed relationships per dedication by concatenating multiple parsing strategies instead of choosing one",
      "risk_level": "low",
      "affected_issue_category": "Dedication Parsing Failure",
      "expected_improvement": "Fixes all 12 critical dedication parsing errors",

      "change_description": "Rewrite DedicationParser.process_batch() to use a single, consistent parsing strategy instead of concatenating results from multiple strategies",
      "affected_function": "DedicationParser.process_batch",
      "change_type": "function_rewrite",

      "guidance": {
        "current_issue": "Parser runs 3 different strategies (comma-split, and-split, full-target) and concatenates all results, creating 6+ relationships per dedication",
        "fix_approach": "Use ONLY comma-splitting strategy. Remove full-target relationship append. Add deduplication before returning. Ensure max 2-3 relationships per dedication.",
        "test_with": "Example: 'dedicated to my two children, Osha and Hunter' should create 2 relationships: (author, dedicated to, Osha) and (author, dedicated to, Hunter). NOT 6+.",
        "key_changes": [
          "Remove concatenation of multiple strategy results",
          "Use only comma-based splitting",
          "Add deduplication logic",
          "Validate output has ≤3 relationships per dedication"
        ]
      },

      "validation": {
        "test_cases": [
          "Single target: 'dedicated to my mother' → 1 relationship",
          "Comma list: 'dedicated to Osha and Hunter' → 2 relationships",
          "Long list: 'dedicated to A, B, and C' → 3 relationships"
        ],
        "success_criteria": "No dedication generates >3 relationships"
      }
    },
    {
      "operation_id": "change_002",
      "operation_type": "PROMPT_ENHANCEMENT",
      "file_path": "prompts/pass1_extraction_v9.txt",
      "priority": "HIGH",
      "rationale": "Pass 1 prompt doesn't explicitly prohibit possessive pronouns ('my people', 'our tradition') as entity sources",
      "risk_level": "low",
      "affected_issue_category": "Possessive Pronoun Sources",
      "expected_improvement": "Reduces possessive pronoun issues by 80%",

      "change_description": "Add explicit prohibition against possessive pronouns and demonstrative pronouns as entity sources",
      "target_section": "ENTITY RESOLUTION RULES",

      "guidance": {
        "current_issue": "Prompt doesn't mention possessive pronouns specifically, so LLM extracts 'my people', 'our tradition' as entities",
        "fix_approach": "Add a new section after entity type definitions with explicit examples of prohibited patterns",
        "insertion_point": "Before the OUTPUT FORMAT section",
        "content_to_add": {
          "heading": "⚠️ PROHIBITED ENTITY PATTERNS",
          "examples": [
            "❌ Possessive pronouns: 'my people', 'our tradition', 'their practices'",
            "❌ Demonstrative pronouns: 'this', 'that', 'these', 'those'",
            "❌ Vague references: 'the process', 'the way', 'the solution'",
            "✅ Instead: Identify the specific entity being referenced from context"
          ],
          "instruction": "Always resolve pronouns and vague references to specific named entities before extraction"
        }
      },

      "validation": {
        "test_prompt_with": "Sample text containing 'my people', 'our ancestors', 'this practice'",
        "success_criteria": "No possessive/demonstrative pronouns in extracted entities",
        "prompt_version": "v10"
      }
    },
    {
      "operation_id": "change_003",
      "operation_type": "CODE_FIX",
      "file_path": "modules/pass2_5_postprocessing/deduplicator.py",
      "priority": "MEDIUM",
      "rationale": "43 duplicate relationships exist, indicating deduplication module is not working correctly",
      "risk_level": "low",
      "affected_issue_category": "Duplicate Relationships",
      "expected_improvement": "Eliminates all 43 duplicate relationships",

      "change_description": "Fix deduplication logic to properly normalize and detect case-insensitive duplicates",
      "affected_function": "Deduplicator.remove_duplicates",
      "change_type": "bug_fix",

      "guidance": {
        "current_issue": "Deduplicator not catching case-insensitive duplicates like ('Bill Mollison', 'authored', 'Permaculture Manual') vs ('bill mollison', 'authored', 'permaculture manual')",
        "fix_approach": "Normalize source/relationship/target to lowercase and strip whitespace before comparison. Use set of tuples to track seen relationships.",
        "key_changes": [
          "Add .lower().strip() normalization before duplicate detection",
          "Use (source, relationship, target) tuple as dedup key",
          "Preserve original casing in output, only use normalized for comparison"
        ]
      },

      "validation": {
        "test_cases": [
          "Exact duplicates: ('A', 'rel', 'B') vs ('A', 'rel', 'B')",
          "Case variants: ('A', 'rel', 'B') vs ('a', 'rel', 'b')",
          "Whitespace variants: ('A ', 'rel', ' B') vs ('A', 'rel', 'B')"
        ],
        "success_criteria": "Zero duplicates in output"
      }
    }
  ],
  "expected_impact": {
    "issues_fixed": 67,
    "critical_fixed": 12,
    "high_fixed": 38,
    "medium_fixed": 17,
    "estimated_error_rate": "13.6% → 5.5%",
    "target_grade": "B- → B+",
    "primary_improvements": [
      "Dedication parsing: 12 errors eliminated",
      "Possessive pronouns: 15 errors reduced to ~3",
      "Duplicates: 43 eliminated"
    ]
  },
  "priorities": {
    "immediate": [
      "change_001: Fix dedication parser (CRITICAL - 12 errors)",
      "change_002: Add possessive pronoun prohibition (HIGH - 15 errors)"
    ],
    "short_term": [
      "change_003: Fix deduplicator (MEDIUM - 43 errors)"
    ]
  },
  "testing_strategy": {
    "validation_approach": "Run V10 extraction and compare to V9 quality metrics",
    "success_criteria": [
      "Critical issues: 12 → 0",
      "Total issues: 117 → <50",
      "Error rate: 13.6% → <8%"
    ]
  }
}

**OUTPUT GUIDELINES**:
- Each operation should be 15-30 lines of JSON (NOT 200+ lines)
- Use `change_description` (1-2 sentences) instead of `old_content`/`new_content` (500 lines)
- Use `guidance` object with structured sub-fields for clarity
- Be specific about WHAT to change, but don't provide complete implementations
- Total JSON output should be 3,000-5,000 tokens maximum

Be strategic, concise, and actionable. Every change must be traceable to a Reflector insight and have clear expected impact."""

    def _build_curation_prompt(
        self,
        reflector_report: Dict[str, Any],
        current_version: int,
        playbook_state: Dict[str, Any]
    ) -> str:
        """Build the curation prompt for Claude."""

        prompt = f"""Transform the following Reflector analysis into an executable changeset.

## REFLECTOR ANALYSIS

{json.dumps(reflector_report, indent=2)}

## CURRENT SYSTEM STATE

Version: V{current_version}
Target Version: V{current_version + 1}

Current Playbook Structure:
{json.dumps(playbook_state, indent=2)}

## YOUR TASK

Based on the Reflector's analysis, create a changeset that:

1. **Addresses Critical Issues First**: Focus on issues that affect >5% of relationships
2. **Targets Root Causes**: Don't just fix symptoms - fix the underlying module/prompt/config
3. **Provides Exact Changes**: Specify old_content and new_content for every edit
4. **Estimates Impact**: Predict how many relationships each change will fix
5. **Manages Risk**: Mark changes as low/medium/high risk

PRIORITIES:
- **CRITICAL**: Reversed authorship, broken extraction logic → Fix immediately
- **HIGH**: Pronouns, lists, vague entities → High-impact patterns
- **MEDIUM**: Incomplete titles, wrong predicates → Quality improvements
- **LOW**: Metaphorical language, minor edge cases → Polish

For each Reflector recommendation, propose:
- Specific file path to modify
- Exact code/prompt/config changes
- Rationale connecting to the issue
- Expected improvement (quantified if possible)
- Risk level and rollback plan

Output your changeset as JSON matching the schema in your system prompt."""

        return prompt

    def _load_playbook_state(self) -> Dict[str, Any]:
        """Load current Playbook structure and key files - SCANS ACTUAL FILESYSTEM."""
        state = {
            "modules": [],
            "prompts": [],
            "configs": [],
            "vocabularies": []
        }

        # Define actual directory paths (not hypothetical)
        project_root = Path("/home/claudeuser/yonearth-gaia-chatbot")
        postprocessing_dir = project_root / "src" / "knowledge_graph" / "postprocessing"
        prompts_dir = self.playbook_path / "prompts"
        configs_dir = self.playbook_path / "config"

        # Scan ACTUAL postprocessing modules
        if postprocessing_dir.exists():
            # Get all Python files and make paths relative for readability
            module_files = list(postprocessing_dir.rglob("*.py"))
            # Return paths relative to project root for clarity
            state["modules"] = [str(f.relative_to(project_root)) for f in module_files]
        else:
            print(f"⚠️  Warning: Postprocessing directory not found: {postprocessing_dir}")

        # Scan prompts (this was already correct)
        if prompts_dir.exists():
            prompt_files = list(prompts_dir.rglob("*.txt"))
            state["prompts"] = [str(f.relative_to(self.playbook_path)) for f in prompt_files]
        else:
            print(f"⚠️  Warning: Prompts directory not found: {prompts_dir}")

        # Scan configs
        if configs_dir.exists():
            config_files = list(configs_dir.rglob("*.json"))
            state["configs"] = [str(f.relative_to(self.playbook_path)) for f in config_files]

        # Debug output
        print(f"📂 Scanned filesystem:")
        print(f"   - Modules found: {len(state['modules'])}")
        print(f"   - Prompts found: {len(state['prompts'])}")
        print(f"   - Configs found: {len(state['configs'])}")

        return state

    def _save_changeset(self, changeset: Dict[str, Any], version: int) -> None:
        """Save changeset for review and application."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        changesets_dir = self.playbook_path / "changesets"
        changesets_dir.mkdir(parents=True, exist_ok=True)

        output_path = changesets_dir / f"changeset_v{version}_to_v{version+1}_{timestamp}.json"

        with open(output_path, 'w') as f:
            json.dump(changeset, f, indent=2)

        print(f"✅ Changeset saved to: {output_path}")

    def apply_changeset(
        self,
        changeset: Dict[str, Any],
        dry_run: bool = False,
        auto_apply_low_risk: bool = True
    ) -> Dict[str, Any]:
        """
        Apply changeset to evolve Playbook.

        Args:
            changeset: Output from curate_improvements()
            dry_run: If True, show changes but don't apply
            auto_apply_low_risk: Automatically apply low-risk changes

        Returns:
            Application results with success/failure status
        """
        results = {
            "applied": [],
            "skipped": [],
            "failed": [],
            "requires_approval": []
        }

        print(f"\n{'='*60}")
        print(f"APPLYING CHANGESET: V{changeset['metadata']['source_version']} → V{changeset['metadata']['target_version']}")
        print(f"{'='*60}\n")

        for operation in changeset.get("file_operations", []):
            op_id = operation.get("operation_id", "unknown")
            priority = operation.get("priority", "MEDIUM")
            risk = operation.get("risk_level", "medium")
            file_path = operation.get("file_path", "unknown")

            print(f"\n[{op_id}] {operation.get('operation_type')} - {file_path}")
            print(f"  Priority: {priority}, Risk: {risk}")
            print(f"  Rationale: {operation.get('rationale', 'N/A')}")

            # Decide whether to apply
            should_apply = False

            if dry_run:
                print(f"  ⏸️  DRY RUN - Would apply")
                results["skipped"].append(operation)
                continue

            if auto_apply_low_risk and risk == "low":
                should_apply = True
                print(f"  ⚡ AUTO-APPLYING (low risk)")
            else:
                print(f"  ⏸️  REQUIRES APPROVAL (risk: {risk})")
                results["requires_approval"].append(operation)
                continue

            if should_apply:
                try:
                    self._apply_file_operation(operation)
                    results["applied"].append(operation)
                    print(f"  ✅ APPLIED")
                except Exception as e:
                    print(f"  ❌ FAILED: {e}")
                    results["failed"].append({
                        "operation": operation,
                        "error": str(e)
                    })

        # Summary
        print(f"\n{'='*60}")
        print(f"CHANGESET APPLICATION COMPLETE")
        print(f"{'='*60}")
        print(f"✅ Applied: {len(results['applied'])}")
        print(f"⏸️  Requires Approval: {len(results['requires_approval'])}")
        print(f"⏭️  Skipped: {len(results['skipped'])}")
        print(f"❌ Failed: {len(results['failed'])}")

        return results

    def _apply_file_operation(self, operation: Dict[str, Any]) -> None:
        """Apply a single file operation using intelligent Applicator."""
        op_type = operation.get("operation_type")
        file_path_str = operation.get("file_path")

        # Smart path resolution: Check if path is relative to project root or playbook
        project_root = Path("/home/claudeuser/yonearth-gaia-chatbot")

        if file_path_str.startswith("src/") or file_path_str.startswith("/"):
            # Module paths (src/knowledge_graph/...) are relative to project root
            file_path = project_root / file_path_str if not file_path_str.startswith("/") else Path(file_path_str)
        else:
            # Prompt/config paths (prompts/..., config/...) are relative to playbook
            file_path = self.playbook_path / file_path_str

        # Check if we have the new concise format (change_description + guidance)
        # vs old verbose format (old_content + new_content)
        has_change_description = "change_description" in operation
        has_guidance = "guidance" in operation

        if has_change_description and has_guidance:
            # NEW FORMAT: Use intelligent Applicator
            self._apply_operation_intelligent(operation, file_path)
        else:
            # OLD FORMAT: Use legacy string replacement (for backward compatibility)
            self._apply_operation_legacy(operation, file_path)

    def _apply_operation_intelligent(self, operation: Dict[str, Any], file_path: Path) -> None:
        """
        Intelligent Applicator: Reads files and uses Claude to make strategic changes.

        This implements the true ACE architecture where:
        - Curator provides STRATEGIC guidance (WHAT + WHY)
        - Applicator provides TACTICAL execution (HOW)
        """
        op_type = operation.get("operation_type")
        change_desc = operation.get("change_description")
        guidance = operation.get("guidance", {})
        affected_function = operation.get("affected_function")
        change_type = operation.get("change_type", "modification")

        if op_type == "CODE_FIX":
            # Read current file
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            with open(file_path, 'r') as f:
                current_content = f.read()

            # Build prompt for Claude to make the change
            fix_prompt = f"""You are an expert Python developer. I need you to modify a file based on strategic guidance.

**File**: {file_path.name}
**Function to modify**: {affected_function}
**Change type**: {change_type}

**Current file content**:
```python
{current_content}
```

**Change needed**:
{change_desc}

**Detailed guidance**:
- **Current issue**: {guidance.get('current_issue', 'See change description')}
- **Fix approach**: {guidance.get('fix_approach', 'Apply the change described above')}
- **Key changes**: {json.dumps(guidance.get('key_changes', []), indent=2)}

**Test with**: {guidance.get('test_with', 'N/A')}

Please provide the COMPLETE modified file content. Output ONLY the Python code, no explanations."""

            # Use Claude to generate the fix
            print(f"    🤖 Using Claude to apply intelligent fix to {file_path.name}...")
            response = self.client.messages.create(
                model=self.model,
                max_tokens=8000,
                temperature=0.2,  # Low temperature for precise code changes
                messages=[
                    {
                        "role": "user",
                        "content": fix_prompt
                    }
                ]
            )

            modified_content = response.content[0].text

            # Extract code from markdown if wrapped
            if "```python" in modified_content:
                code_start = modified_content.find("```python") + 9
                code_end = modified_content.find("```", code_start)
                modified_content = modified_content[code_start:code_end].strip()
            elif "```" in modified_content:
                code_start = modified_content.find("```") + 3
                code_end = modified_content.find("```", code_start)
                modified_content = modified_content[code_start:code_end].strip()

            # Write modified content
            with open(file_path, 'w') as f:
                f.write(modified_content)

            print(f"    ✅ Applied intelligent fix to {file_path.name}")

        elif op_type == "PROMPT_ENHANCEMENT":
            # Read current prompt
            if not file_path.exists():
                raise FileNotFoundError(f"Prompt file not found: {file_path}")

            with open(file_path, 'r') as f:
                current_content = f.read()

            # Build prompt for Claude to enhance the extraction prompt
            target_section = operation.get("target_section", "N/A")
            content_to_add = guidance.get("content_to_add", {})
            insertion_point = guidance.get("insertion_point", "At the end")

            enhance_prompt = f"""You are an expert prompt engineer. I need you to enhance an extraction prompt.

**Current prompt**:
```
{current_content}
```

**Enhancement needed**:
{change_desc}

**Target section**: {target_section}
**Insertion point**: {insertion_point}

**Content to add**:
{json.dumps(content_to_add, indent=2)}

**Detailed guidance**:
- **Current issue**: {guidance.get('current_issue', 'See change description')}
- **Fix approach**: {guidance.get('fix_approach', 'Apply the enhancement described above')}

Please provide the COMPLETE enhanced prompt. Output ONLY the prompt text, no explanations."""

            # Use Claude to generate the enhanced prompt
            print(f"    🤖 Using Claude to enhance prompt {file_path.name}...")
            response = self.client.messages.create(
                model=self.model,
                max_tokens=8000,
                temperature=0.3,
                messages=[
                    {
                        "role": "user",
                        "content": enhance_prompt
                    }
                ]
            )

            enhanced_content = response.content[0].text

            # Save as new version if specified
            new_version = operation.get("validation", {}).get("prompt_version")
            if new_version:
                # Extract prompt name from path
                prompt_name = file_path.stem.rsplit('_', 1)[0]
                new_file_path = self.playbook_path / "prompts" / f"{prompt_name}_{new_version}.txt"

                with open(new_file_path, 'w') as f:
                    f.write(enhanced_content)

                print(f"    📝 Created new prompt version: {new_file_path.name}")
            else:
                # Update in place
                with open(file_path, 'w') as f:
                    f.write(enhanced_content)

                print(f"    ✅ Enhanced prompt {file_path.name}")

        elif op_type == "CONFIG_UPDATE":
            # CONFIG_UPDATE can use the same logic as before
            self._apply_config_update(operation, file_path)

        elif op_type == "NEW_MODULE":
            # NEW_MODULE needs intelligent generation
            self._apply_new_module_intelligent(operation, file_path)

        else:
            raise ValueError(f"Unknown operation type: {op_type}")

    def _apply_operation_legacy(self, operation: Dict[str, Any], file_path: Path) -> None:
        """
        Legacy Applicator: Uses old string replacement approach.
        Kept for backward compatibility with old changeset format.
        """
        op_type = operation.get("operation_type")

        if op_type == "CODE_FIX" or op_type == "PROMPT_ENHANCEMENT":
            # Edit existing file with old_content/new_content approach
            edit_details = operation.get("edit_details", {})
            old_content = edit_details.get("old_content", "")
            new_content = edit_details.get("new_content", "")

            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            with open(file_path, 'r') as f:
                content = f.read()

            if old_content not in content:
                raise ValueError(f"Old content not found in {file_path}")

            updated_content = content.replace(old_content, new_content, 1)

            # Handle prompt versioning for PROMPT_ENHANCEMENT
            if op_type == "PROMPT_ENHANCEMENT":
                new_version = edit_details.get("prompt_version")
                if new_version:
                    prompt_name = file_path.stem.rsplit('_', 1)[0]
                    new_file_path = self.playbook_path / "prompts" / f"{prompt_name}_{new_version}.txt"
                    with open(new_file_path, 'w') as f:
                        f.write(updated_content)
                    print(f"  📝 Created new prompt version: {new_file_path.name}")
                    return

            with open(file_path, 'w') as f:
                f.write(updated_content)

        elif op_type == "CONFIG_UPDATE":
            self._apply_config_update(operation, file_path)

        elif op_type == "NEW_MODULE":
            # Create new file with provided content
            create_details = operation.get("create_details", {})
            content = create_details.get("content", "")

            file_path.parent.mkdir(parents=True, exist_ok=True)

            with open(file_path, 'w') as f:
                f.write(content)

        else:
            raise ValueError(f"Unknown operation type: {op_type}")

    def _apply_config_update(self, operation: Dict[str, Any], file_path: Path) -> None:
        """Apply config update (same for both legacy and intelligent)."""
        edit_details = operation.get("edit_details", {})
        json_path = edit_details.get("json_path", "")
        new_value = edit_details.get("new_value")

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, 'r') as f:
            config = json.load(f)

        # Navigate to setting
        keys = json_path.split('.')
        current = config
        for key in keys[:-1]:
            current = current[key]

        # Update value
        current[keys[-1]] = new_value

        with open(file_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"    ✅ Updated config: {json_path} = {new_value}")

    def _apply_new_module_intelligent(self, operation: Dict[str, Any], file_path: Path) -> None:
        """Intelligently create a new module based on strategic guidance."""
        change_desc = operation.get("change_description")
        guidance = operation.get("guidance", {})
        create_details = operation.get("create_details", {})

        # Build prompt for Claude to generate the module
        module_prompt = f"""You are an expert Python developer. I need you to create a new module.

**Module path**: {file_path}
**Purpose**: {change_desc}

**Guidance**:
{json.dumps(guidance, indent=2)}

**Additional details**:
{json.dumps(create_details, indent=2)}

Please provide the COMPLETE module code. Output ONLY the Python code, no explanations."""

        # Use Claude to generate the module
        print(f"    🤖 Using Claude to generate new module {file_path.name}...")
        response = self.client.messages.create(
            model=self.model,
            max_tokens=8000,
            temperature=0.3,
            messages=[
                {
                    "role": "user",
                    "content": module_prompt
                }
            ]
        )

        module_content = response.content[0].text

        # Extract code from markdown if wrapped
        if "```python" in module_content:
            code_start = module_content.find("```python") + 9
            code_end = module_content.find("```", code_start)
            module_content = module_content[code_start:code_end].strip()
        elif "```" in module_content:
            code_start = module_content.find("```") + 3
            code_end = module_content.find("```", code_start)
            module_content = module_content[code_start:code_end].strip()

        # Create directory if needed
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Write module
        with open(file_path, 'w') as f:
            f.write(module_content)

        print(f"    ✅ Created new module {file_path.name}")


if __name__ == "__main__":
    # Example usage
    curator = KGCuratorAgent()

    # Load sample Reflector report
    sample_report = {
        "critical_issues": 15,
        "recommendations": [
            {
                "priority": "CRITICAL",
                "type": "CODE_FIX",
                "target_file": "modules/bibliographic_parser.py"
            }
        ]
    }

    changeset = curator.curate_improvements(
        reflector_report=sample_report,
        current_version=5
    )

    print(json.dumps(changeset, indent=2))
