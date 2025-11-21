"""
Curator Agent

Organizes Reflector insights into actionable knowledge and updates the Playbook
to evolve the YonEarth chatbot system.
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path

from openai import OpenAI


class CuratorAgent:
    """
    The Curator transforms Reflector insights into actionable changes:
    - Organizes recommendations by type and priority
    - Creates/updates Playbook entries
    - Designs experiments for uncertain changes
    - Manages system evolution and version control
    """

    def __init__(self, playbook_path: str = "/home/claudeuser/yonearth-gaia-chatbot/data/playbook"):
        self.playbook_path = Path(playbook_path)
        self.client = OpenAI()
        self.model = "gpt-4o"  # Using GPT-4o for strategic planning

    def curate_insights(
        self,
        reflector_insights: Dict[str, Any],
        playbook_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Transform Reflector insights into actionable Playbook updates.

        Args:
            reflector_insights: Output from Reflector agent
            playbook_state: Current Playbook configuration

        Returns:
            Curation plan with updates, experiments, and version notes
        """
        # Load system context
        system_context = self._load_system_context()

        # Build curation prompt
        prompt = self._build_curation_prompt(
            reflector_insights,
            playbook_state,
            system_context
        )

        # Run GPT-4 curation
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": self._get_curator_system_prompt()
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.4  # Slightly higher for creative solutions
        )

        # Parse curation plan
        curation_plan = json.loads(response.choices[0].message.content)

        # Enhance with metadata
        curation_plan["metadata"] = {
            "curation_date": datetime.now().isoformat(),
            "reflection_id": reflector_insights.get("metadata", {}).get("analysis_date"),
            "curator_version": "1.0",
            "model_used": self.model
        }

        # Save curation plan
        self._save_curation_plan(curation_plan)

        return curation_plan

    def _get_curator_system_prompt(self) -> str:
        """Get the system prompt for the Curator agent."""
        return """You are a Curator Agent responsible for evolving the YonEarth Gaia chatbot
from V5 to V6 based on insights from conversation analysis.

Your role is to transform insights into action:
1. Organize recommendations by type (prompt, search, content, feature, configuration)
2. Create specific Playbook updates with clear rationale
3. Design A/B tests for uncertain changes
4. Plan version evolution with migration steps

You must output a JSON object with this structure:
{
  "playbook_updates": [
    {
      "file_path": "/data/playbook/path/to/file.json",
      "change_type": "create|update|delete",
      "content": "actual file content or changes",
      "rationale": "why this change addresses the insight",
      "impact_estimate": "expected effect on user experience",
      "risk_level": "low|medium|high",
      "requires_approval": true|false
    }
  ],
  "prompt_improvements": [
    {
      "target": "gaia_core|search|citation",
      "current_version": "existing prompt text",
      "proposed_version": "improved prompt text",
      "improvements": ["improvement1", "improvement2"],
      "testing_strategy": "how to validate improvement"
    }
  ],
  "configuration_changes": [
    {
      "setting_path": "search_weights.category_match",
      "current_value": 0.60,
      "proposed_value": 0.65,
      "rationale": "why this adjustment helps",
      "rollback_plan": "how to revert if needed"
    }
  ],
  "experiments": [
    {
      "name": "experiment_name",
      "hypothesis": "what we expect to happen",
      "configuration": {
        "variant_a": "current behavior",
        "variant_b": "new behavior"
      },
      "success_metrics": ["metric1", "metric2"],
      "duration": "1 week",
      "rollout_percentage": 50
    }
  ],
  "version_notes": {
    "changes_summary": "high-level description of V5→V6 evolution",
    "breaking_changes": ["change1", "change2"],
    "migration_steps": ["step1", "step2"],
    "rollout_plan": "how to deploy these changes"
  },
  "priorities": {
    "immediate": ["high priority items needing quick action"],
    "short_term": ["1-2 week timeline items"],
    "long_term": ["strategic improvements for future versions"]
  }
}

Be specific about file paths, content changes, and implementation details.
Focus on high-impact, low-risk improvements first."""

    def _build_curation_prompt(
        self,
        reflector_insights: Dict[str, Any],
        playbook_state: Dict[str, Any],
        system_context: Dict[str, Any]
    ) -> str:
        """Build the user prompt with insights and context."""

        prompt = f"""Transform the following Reflector insights into actionable Playbook updates:

## REFLECTOR INSIGHTS

{json.dumps(reflector_insights, indent=2)}

## CURRENT SYSTEM STATE

Version: {system_context.get('current_version', '5.0')} → {system_context.get('target_version', '6.0')}

Capabilities:
{json.dumps(system_context.get('capabilities', {}), indent=2)}

## CURRENT PLAYBOOK STATE

{json.dumps(playbook_state, indent=2)}

## V6 EVOLUTION GOALS

Priority 1: Enhanced Search Intelligence
- Dynamic search weight optimization
- Conversation context integration

Priority 2: Improved User Experience
- Personalized response styles
- Progressive conversation depth

Priority 3: Content Intelligence
- Cross-content synthesis (episodes + books)
- Timeline-aware responses
- Knowledge graph integration

Priority 4: Self-Improvement
- Automated insight extraction
- Prompt evolution based on feedback
- A/B testing framework

---

Based on the Reflector insights, create:
1. Specific Playbook file updates to address identified gaps
2. Prompt improvements to fix response quality issues
3. Configuration tweaks to optimize search behavior
4. Experiments to test uncertain improvements
5. Version evolution plan with clear priorities

Focus on changes that will have immediate positive impact on user satisfaction."""

        return prompt

    def _load_system_context(self) -> Dict[str, Any]:
        """Load current system state and evolution history."""
        context = {}

        # Load version info
        version_path = self.playbook_path / "meta" / "version.json"
        if version_path.exists():
            with open(version_path) as f:
                context.update(json.load(f))

        # Load capabilities
        capabilities_path = self.playbook_path / "meta" / "capability_matrix.json"
        if capabilities_path.exists():
            with open(capabilities_path) as f:
                context["capabilities"] = json.load(f)

        # Load evolution history
        evolution_path = self.playbook_path / "meta" / "evolution_log.json"
        if evolution_path.exists():
            with open(evolution_path) as f:
                context["evolution_history"] = json.load(f)

        return context

    def _save_curation_plan(self, plan: Dict[str, Any]) -> None:
        """Save curation plan for review and implementation."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plans_dir = self.playbook_path / "insights" / "improvement_opportunities"
        plans_dir.mkdir(parents=True, exist_ok=True)

        output_path = plans_dir / f"curation_plan_{timestamp}.json"

        with open(output_path, 'w') as f:
            json.dump(plan, f, indent=2)

        print(f"✅ Curation plan saved to: {output_path}")

    def apply_approved_updates(
        self,
        curation_plan: Dict[str, Any],
        auto_apply_low_risk: bool = True
    ) -> Dict[str, Any]:
        """
        Apply approved Playbook updates from curation plan.

        Args:
            curation_plan: Output from curate_insights()
            auto_apply_low_risk: Whether to auto-apply low-risk changes

        Returns:
            Application results with success/failure status
        """
        results = {
            "applied": [],
            "skipped": [],
            "failed": [],
            "requires_approval": []
        }

        # Process playbook updates
        for update in curation_plan.get("playbook_updates", []):
            risk_level = update.get("risk_level", "medium")
            requires_approval = update.get("requires_approval", True)

            # Auto-apply if low risk and allowed
            if auto_apply_low_risk and risk_level == "low" and not requires_approval:
                try:
                    self._apply_playbook_update(update)
                    results["applied"].append(update)
                except Exception as e:
                    results["failed"].append({
                        "update": update,
                        "error": str(e)
                    })
            else:
                results["requires_approval"].append(update)

        # Process configuration changes
        for config_change in curation_plan.get("configuration_changes", []):
            try:
                self._apply_configuration_change(config_change)
                results["applied"].append(config_change)
            except Exception as e:
                results["failed"].append({
                    "change": config_change,
                    "error": str(e)
                })

        # Log evolution
        self._log_evolution(curation_plan, results)

        return results

    def _apply_playbook_update(self, update: Dict[str, Any]) -> None:
        """Apply a single Playbook file update."""
        file_path = Path(update["file_path"])
        change_type = update["change_type"]
        content = update["content"]

        if change_type == "create":
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w') as f:
                if isinstance(content, dict) or isinstance(content, list):
                    json.dump(content, f, indent=2)
                else:
                    f.write(content)

        elif change_type == "update":
            if file_path.suffix == '.json':
                with open(file_path) as f:
                    existing = json.load(f)
                # Merge content
                if isinstance(existing, dict) and isinstance(content, dict):
                    existing.update(content)
                    with open(file_path, 'w') as f:
                        json.dump(existing, f, indent=2)
            else:
                with open(file_path, 'w') as f:
                    f.write(content)

        elif change_type == "delete":
            if file_path.exists():
                file_path.unlink()

        print(f"✅ Applied update: {change_type} {file_path}")

    def _apply_configuration_change(self, change: Dict[str, Any]) -> None:
        """Apply a configuration change to Playbook."""
        setting_path = change["setting_path"]
        new_value = change["proposed_value"]

        # Parse setting path (e.g., "search_weights.category_match")
        parts = setting_path.split('.')
        file_name = parts[0] + '.json'
        file_path = self.playbook_path / "configurations" / file_name

        if file_path.exists():
            with open(file_path) as f:
                config = json.load(f)

            # Navigate to setting
            current = config
            for part in parts[1:-1]:
                current = current[part]

            # Update value
            current[parts[-1]] = new_value

            # Save
            with open(file_path, 'w') as f:
                json.dump(config, f, indent=2)

            print(f"✅ Updated config: {setting_path} = {new_value}")

    def _log_evolution(self, curation_plan: Dict[str, Any], results: Dict[str, Any]) -> None:
        """Log evolution changes to history."""
        evolution_path = self.playbook_path / "meta" / "evolution_log.json"

        with open(evolution_path) as f:
            evolution_log = json.load(f)

        # Add reflection cycle entry
        cycle_entry = {
            "cycle_number": len(evolution_log.get("reflection_cycles", [])) + 1,
            "date": datetime.now().isoformat(),
            "insights_processed": len(curation_plan.get("playbook_updates", [])),
            "changes_applied": len(results["applied"]),
            "changes_pending": len(results["requires_approval"]),
            "changes_failed": len(results["failed"]),
            "version_notes": curation_plan.get("version_notes", {})
        }

        if "reflection_cycles" not in evolution_log:
            evolution_log["reflection_cycles"] = []

        evolution_log["reflection_cycles"].append(cycle_entry)

        with open(evolution_path, 'w') as f:
            json.dump(evolution_log, f, indent=2)

        print(f"✅ Logged evolution cycle #{cycle_entry['cycle_number']}")


if __name__ == "__main__":
    # Example usage
    curator = CuratorAgent()

    # Load latest Reflector insights
    insights_dir = curator.playbook_path / "insights" / "conversation_patterns"
    latest_insights_file = sorted(insights_dir.glob("reflection_*.json"))[-1]

    print(f"📖 Loading insights from: {latest_insights_file}")

    with open(latest_insights_file) as f:
        reflector_insights = json.load(f)

    # Load playbook state
    playbook_state = curator._load_system_context()

    # Run curation
    print("🎨 Running Curator analysis...")
    curation_plan = curator.curate_insights(
        reflector_insights,
        playbook_state
    )

    print("\n📋 CURATION COMPLETE")
    print(f"Playbook updates: {len(curation_plan.get('playbook_updates', []))}")
    print(f"Prompt improvements: {len(curation_plan.get('prompt_improvements', []))}")
    print(f"Configuration changes: {len(curation_plan.get('configuration_changes', []))}")
    print(f"Experiments: {len(curation_plan.get('experiments', []))}")
