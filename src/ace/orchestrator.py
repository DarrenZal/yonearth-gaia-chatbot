"""
ACE Orchestrator

Coordinates the reflection-curation cycle for autonomous system evolution.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from .reflector import ReflectorAgent
from .curator import CuratorAgent


class ACEOrchestrator:
    """
    The Orchestrator coordinates the ACE (Autonomous Cognitive Entity) framework:

    WORKFLOW:
    1. COLLECT: Gather new conversations and feedback
    2. REFLECT: Run Reflector agent on collected data
    3. CURATE: Run Curator agent on Reflector insights
    4. REVIEW: Present proposed changes for approval
    5. APPLY: Execute approved changes
    6. MONITOR: Track impact of changes
    7. ITERATE: Repeat cycle
    """

    def __init__(
        self,
        playbook_path: str = "/home/claudeuser/yonearth-gaia-chatbot/data/playbook",
        feedback_dir: str = "/home/claudeuser/yonearth-gaia-chatbot/data/feedback"
    ):
        self.playbook_path = Path(playbook_path)
        self.feedback_dir = Path(feedback_dir)

        self.reflector = ReflectorAgent(str(playbook_path))
        self.curator = CuratorAgent(str(playbook_path))

        self.cycle_count = self._get_current_cycle_count()

    def run_reflection_cycle(
        self,
        auto_apply_low_risk: bool = True,
        interactive_approval: bool = True
    ) -> Dict[str, Any]:
        """
        Execute a complete reflection cycle.

        Args:
            auto_apply_low_risk: Automatically apply low-risk changes
            interactive_approval: Prompt for approval of other changes

        Returns:
            Cycle results with metrics and applied changes
        """
        print(f"\n{'='*60}")
        print(f"🔄 ACE REFLECTION CYCLE #{self.cycle_count + 1}")
        print(f"{'='*60}\n")

        cycle_start = datetime.now()
        results = {
            "cycle_number": self.cycle_count + 1,
            "start_time": cycle_start.isoformat(),
            "stages": {}
        }

        try:
            # Stage 1: COLLECT
            print("📦 STAGE 1: COLLECTING DATA...")
            conversations, feedback_data = self._collect_data()
            results["stages"]["collect"] = {
                "conversations": len(conversations),
                "feedback_items": len(feedback_data),
                "status": "success"
            }
            print(f"   ✅ Collected {len(conversations)} conversations, {len(feedback_data)} feedback items\n")

            if not conversations:
                print("⚠️  No new data to analyze. Exiting cycle.")
                results["status"] = "skipped"
                results["reason"] = "no_new_data"
                return results

            # Stage 2: REFLECT
            print("🔍 STAGE 2: RUNNING REFLECTOR ANALYSIS...")
            playbook_state = self._load_playbook_state()
            insights = self.reflector.analyze_conversation_batch(
                conversations,
                feedback_data,
                playbook_state
            )
            results["stages"]["reflect"] = {
                "patterns": len(insights.get("patterns", [])),
                "gaps": len(insights.get("gaps", [])),
                "recommendations": len(insights.get("recommendations", [])),
                "status": "success"
            }
            print(f"   ✅ Identified {len(insights.get('patterns', []))} patterns, "
                  f"{len(insights.get('gaps', []))} gaps, "
                  f"{len(insights.get('recommendations', []))} recommendations\n")

            # Display key insights
            self._display_insights_summary(insights)

            # Stage 3: CURATE
            print("\n🎨 STAGE 3: RUNNING CURATOR PLANNING...")
            curation_plan = self.curator.curate_insights(
                insights,
                playbook_state
            )
            results["stages"]["curate"] = {
                "playbook_updates": len(curation_plan.get("playbook_updates", [])),
                "prompt_improvements": len(curation_plan.get("prompt_improvements", [])),
                "configuration_changes": len(curation_plan.get("configuration_changes", [])),
                "experiments": len(curation_plan.get("experiments", [])),
                "status": "success"
            }
            print(f"   ✅ Generated {len(curation_plan.get('playbook_updates', []))} playbook updates, "
                  f"{len(curation_plan.get('configuration_changes', []))} config changes\n")

            # Display curation plan summary
            self._display_curation_summary(curation_plan)

            # Stage 4: REVIEW & APPROVE
            print("\n👤 STAGE 4: REVIEW & APPROVAL...")
            if interactive_approval:
                approved_plan = self._interactive_approval(curation_plan)
            else:
                approved_plan = curation_plan
                print("   ⚙️  Auto-approval mode (non-interactive)")

            results["stages"]["review"] = {
                "status": "success",
                "interactive": interactive_approval
            }

            # Stage 5: APPLY
            print("\n⚡ STAGE 5: APPLYING APPROVED CHANGES...")
            application_results = self.curator.apply_approved_updates(
                approved_plan,
                auto_apply_low_risk=auto_apply_low_risk
            )
            results["stages"]["apply"] = {
                "applied": len(application_results["applied"]),
                "skipped": len(application_results["skipped"]),
                "failed": len(application_results["failed"]),
                "pending_approval": len(application_results["requires_approval"]),
                "status": "success"
            }
            print(f"   ✅ Applied {len(application_results['applied'])} changes")
            if application_results["failed"]:
                print(f"   ⚠️  {len(application_results['failed'])} changes failed")
            if application_results["requires_approval"]:
                print(f"   ⏸️  {len(application_results['requires_approval'])} changes require manual approval")

            # Update cycle count
            self._increment_cycle_count()

            # Final results
            cycle_end = datetime.now()
            results["end_time"] = cycle_end.isoformat()
            results["duration_seconds"] = (cycle_end - cycle_start).total_seconds()
            results["status"] = "success"

            print(f"\n{'='*60}")
            print(f"✅ CYCLE COMPLETE in {results['duration_seconds']:.1f}s")
            print(f"{'='*60}\n")

            return results

        except Exception as e:
            print(f"\n❌ CYCLE FAILED: {e}")
            results["status"] = "failed"
            results["error"] = str(e)
            return results

    def _collect_data(self) -> tuple:
        """Collect conversations and feedback from data directory."""
        return self.reflector.load_conversations_from_feedback_files(
            str(self.feedback_dir)
        )

    def _load_playbook_state(self) -> Dict[str, Any]:
        """Load current Playbook configuration."""
        state = {}

        # Load all configuration files
        config_dir = self.playbook_path / "configurations"
        if config_dir.exists():
            for config_file in config_dir.glob("*.json"):
                try:
                    with open(config_file) as f:
                        state[config_file.stem] = json.load(f)
                except Exception as e:
                    print(f"⚠️  Error loading {config_file}: {e}")

        return state

    def _display_insights_summary(self, insights: Dict[str, Any]) -> None:
        """Display a summary of Reflector insights."""
        print("\n   📊 INSIGHTS SUMMARY:")

        # Metrics
        metrics = insights.get("metrics", {})
        print(f"      • Avg Satisfaction: {metrics.get('avg_satisfaction', 0):.2f}/5.0")
        print(f"      • Citation Accuracy: {metrics.get('citation_accuracy_sample', 'N/A')}")

        # Top patterns
        patterns = insights.get("patterns", [])
        if patterns:
            print(f"\n      Top Patterns:")
            for pattern in patterns[:3]:
                print(f"         - {pattern.get('pattern', 'N/A')} "
                      f"(freq: {pattern.get('frequency', 0)}, "
                      f"impact: {pattern.get('impact', 'unknown')})")

        # Critical gaps
        gaps = [g for g in insights.get("gaps", []) if g.get("severity") == "critical"]
        if gaps:
            print(f"\n      Critical Gaps:")
            for gap in gaps:
                print(f"         - {gap.get('topic', 'N/A')}")

        # Top recommendations
        recs = sorted(
            insights.get("recommendations", []),
            key=lambda r: r.get("priority", 0),
            reverse=True
        )
        if recs:
            print(f"\n      Top Recommendations:")
            for rec in recs[:3]:
                print(f"         - [{rec.get('priority', 0)}/10] {rec.get('category', 'N/A')}: "
                      f"{rec.get('suggestion', 'N/A')[:60]}...")

    def _display_curation_summary(self, curation_plan: Dict[str, Any]) -> None:
        """Display a summary of Curator curation plan."""
        print("\n   📋 CURATION SUMMARY:")

        # Priorities
        priorities = curation_plan.get("priorities", {})
        if priorities.get("immediate"):
            print(f"\n      Immediate Actions ({len(priorities['immediate'])}):")
            for item in priorities["immediate"][:3]:
                print(f"         - {item}")

        # Playbook updates
        updates = curation_plan.get("playbook_updates", [])
        if updates:
            print(f"\n      Playbook Updates ({len(updates)}):")
            for update in updates[:3]:
                print(f"         - {update.get('change_type', 'N/A')}: "
                      f"{Path(update.get('file_path', '')).name} "
                      f"(risk: {update.get('risk_level', 'unknown')})")

        # Configuration changes
        config_changes = curation_plan.get("configuration_changes", [])
        if config_changes:
            print(f"\n      Configuration Changes ({len(config_changes)}):")
            for change in config_changes[:3]:
                print(f"         - {change.get('setting_path', 'N/A')}: "
                      f"{change.get('current_value', 'N/A')} → "
                      f"{change.get('proposed_value', 'N/A')}")

    def _interactive_approval(self, curation_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Interactive approval of curation plan.

        In a real implementation, this would show a UI or detailed console
        interface for reviewing and approving changes.
        """
        print("\n   Review the curation plan above.")
        print("   For now, auto-approving all changes (interactive mode TBD)")

        # TODO: Implement actual interactive approval
        # This would involve:
        # - Displaying each change in detail
        # - Allowing user to approve/reject/modify
        # - Saving approval decisions
        # - Returning modified plan

        return curation_plan

    def _get_current_cycle_count(self) -> int:
        """Get the current reflection cycle count."""
        evolution_path = self.playbook_path / "meta" / "evolution_log.json"

        if not evolution_path.exists():
            return 0

        with open(evolution_path) as f:
            evolution_log = json.load(f)

        return len(evolution_log.get("reflection_cycles", []))

    def _increment_cycle_count(self) -> None:
        """Increment the reflection cycle count in version metadata."""
        version_path = self.playbook_path / "meta" / "version.json"

        if version_path.exists():
            with open(version_path) as f:
                version_data = json.load(f)

            version_data["reflection_cycle_count"] = self.cycle_count + 1
            version_data["last_updated"] = datetime.now().isoformat()

            with open(version_path, 'w') as f:
                json.dump(version_data, f, indent=2)

    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get a summary of system evolution across all cycles."""
        evolution_path = self.playbook_path / "meta" / "evolution_log.json"
        version_path = self.playbook_path / "meta" / "version.json"

        summary = {
            "total_cycles": self.cycle_count,
            "cycles": [],
            "current_version": "unknown",
            "evolution_stage": "unknown"
        }

        if evolution_path.exists():
            with open(evolution_path) as f:
                evolution_log = json.load(f)
                summary["cycles"] = evolution_log.get("reflection_cycles", [])

        if version_path.exists():
            with open(version_path) as f:
                version_data = json.load(f)
                summary["current_version"] = version_data.get("current_version")
                summary["evolution_stage"] = version_data.get("evolution_stage")

        return summary


def main():
    """Main entry point for running ACE orchestrator."""
    import argparse

    parser = argparse.ArgumentParser(
        description="ACE Orchestrator - Autonomous system evolution"
    )
    parser.add_argument(
        "--auto-apply",
        action="store_true",
        help="Automatically apply low-risk changes"
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Run without interactive approval"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show evolution summary and exit"
    )

    args = parser.parse_args()

    orchestrator = ACEOrchestrator()

    if args.summary:
        summary = orchestrator.get_evolution_summary()
        print(json.dumps(summary, indent=2))
        return

    # Run reflection cycle
    results = orchestrator.run_reflection_cycle(
        auto_apply_low_risk=args.auto_apply,
        interactive_approval=not args.non_interactive
    )

    # Save results
    results_path = Path("/home/claudeuser/yonearth-gaia-chatbot/data/playbook/insights")
    results_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(results_path / f"cycle_results_{timestamp}.json", 'w') as f:
        json.dump(results, f, indent=2)

    # Exit with appropriate status
    sys.exit(0 if results["status"] == "success" else 1)


if __name__ == "__main__":
    main()
