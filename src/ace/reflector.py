"""
Reflector Agent

Analyzes conversations and feedback to identify improvement opportunities
for the YonEarth chatbot system.
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path

from openai import OpenAI


class ReflectorAgent:
    """
    The Reflector analyzes conversations between users and Gaia to identify:
    - Recurring patterns and user needs
    - Knowledge gaps and content weaknesses
    - Citation accuracy and response quality
    - Improvement opportunities ranked by priority
    """

    def __init__(self, playbook_path: str = "/home/claudeuser/yonearth-gaia-chatbot/data/playbook"):
        self.playbook_path = Path(playbook_path)
        self.client = OpenAI()
        self.model = "gpt-4o"  # Using GPT-4o for advanced analysis

    def analyze_conversation_batch(
        self,
        conversations: List[Dict[str, Any]],
        feedback_data: List[Dict[str, Any]],
        playbook_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze a batch of conversations and generate insight report.

        Args:
            conversations: List of conversation objects with messages
            feedback_data: List of user feedback objects
            playbook_state: Current Playbook configuration

        Returns:
            Insight report with patterns, gaps, and recommendations
        """
        # Load current system state
        system_state = self._load_system_state()

        # Prepare analysis prompt
        prompt = self._build_analysis_prompt(
            conversations,
            feedback_data,
            playbook_state,
            system_state
        )

        # Run GPT-4 analysis
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": self._get_reflector_system_prompt()
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.3  # Lower temperature for analytical tasks
        )

        # Parse and validate insights
        insights = json.loads(response.choices[0].message.content)

        # Enhance insights with metadata
        insights["metadata"] = {
            "analysis_date": datetime.now().isoformat(),
            "conversations_analyzed": len(conversations),
            "feedback_items_analyzed": len(feedback_data),
            "reflector_version": "1.0",
            "model_used": self.model
        }

        # Save insights to Playbook
        self._save_insights(insights)

        return insights

    def _get_reflector_system_prompt(self) -> str:
        """Get the system prompt for the Reflector agent."""
        return """You are a Reflector Agent analyzing conversations between users and Gaia
(an Earth consciousness chatbot powered by 172 podcast episodes and 3 books).

Your role is to identify improvement opportunities by analyzing:
1. User questions and Gaia's responses
2. User feedback (ratings, comments, satisfaction)
3. Citation accuracy (do referenced episodes/books match the question?)
4. Response quality (depth, relevance, personality consistency)
5. Knowledge gaps (topics users ask about that system handles poorly)

You must output a JSON object with this structure:
{
  "patterns": [
    {
      "pattern": "description of recurring pattern",
      "frequency": count,
      "examples": ["conversation_id_1", "conversation_id_2"],
      "impact": "high|medium|low",
      "insight": "what this pattern reveals about user needs"
    }
  ],
  "gaps": [
    {
      "topic": "description of knowledge gap",
      "evidence": ["conversation_id_1", "conversation_id_2"],
      "severity": "critical|important|minor",
      "current_performance": "description",
      "improvement_needed": "specific enhancement"
    }
  ],
  "recommendations": [
    {
      "category": "prompt|search|content|feature|configuration",
      "suggestion": "specific actionable change",
      "rationale": "why this helps users",
      "priority": 1-10,
      "effort": "low|medium|high",
      "expected_impact": "description of benefits"
    }
  ],
  "metrics": {
    "conversations_analyzed": count,
    "avg_satisfaction": 0.0-5.0,
    "citation_accuracy_sample": "X/Y accurate",
    "common_topics": ["topic1", "topic2"],
    "positive_patterns": ["pattern1"],
    "negative_patterns": ["pattern1"]
  },
  "highlights": {
    "top_strength": "what's working well",
    "top_weakness": "what needs most improvement",
    "surprising_insight": "unexpected discovery"
  }
}

Be specific, actionable, and data-driven in your analysis."""

    def _build_analysis_prompt(
        self,
        conversations: List[Dict[str, Any]],
        feedback_data: List[Dict[str, Any]],
        playbook_state: Dict[str, Any],
        system_state: Dict[str, Any]
    ) -> str:
        """Build the user prompt with conversation data."""

        # Summarize conversations
        conv_summary = self._summarize_conversations(conversations)

        # Summarize feedback
        feedback_summary = self._summarize_feedback(feedback_data)

        prompt = f"""Analyze the following data from the YonEarth Gaia chatbot:

## SYSTEM CONTEXT

Current Version: {system_state.get('current_version', '5.0')}
Target Version: {system_state.get('target_version', '6.0')}

Active Capabilities:
{json.dumps(system_state.get('capabilities', {}), indent=2)}

## CONVERSATION DATA

Total Conversations: {len(conversations)}

{conv_summary}

## FEEDBACK DATA

Total Feedback Items: {len(feedback_data)}

{feedback_summary}

## CURRENT PLAYBOOK STATE

{json.dumps(playbook_state, indent=2)}

---

Analyze this data and provide:
1. Recurring patterns in user questions and interactions
2. Knowledge gaps where Gaia struggles to provide good answers
3. Specific, actionable recommendations for V6 improvements
4. Priority ranking based on user impact

Focus on insights that will directly improve user satisfaction and system capabilities."""

        return prompt

    def _summarize_conversations(self, conversations: List[Dict[str, Any]]) -> str:
        """Create a concise summary of conversations for analysis."""
        if not conversations:
            return "No conversations available for analysis."

        summary_parts = []

        for i, conv in enumerate(conversations[:20], 1):  # Limit to first 20 for prompt length
            messages = conv.get('messages', [])
            user_msgs = [m for m in messages if m.get('role') == 'user']
            assistant_msgs = [m for m in messages if m.get('role') == 'assistant']

            summary_parts.append(f"""
Conversation {i} (ID: {conv.get('id', 'unknown')}):
- User Questions: {len(user_msgs)}
- Gaia Responses: {len(assistant_msgs)}
- Sample User Query: "{user_msgs[0].get('content', '')[:200] if user_msgs else 'N/A'}..."
- Sample Gaia Response: "{assistant_msgs[0].get('content', '')[:200] if assistant_msgs else 'N/A'}..."
- Feedback: {conv.get('feedback', 'None')}
""")

        if len(conversations) > 20:
            summary_parts.append(f"\n... and {len(conversations) - 20} more conversations")

        return "\n".join(summary_parts)

    def _summarize_feedback(self, feedback_data: List[Dict[str, Any]]) -> str:
        """Create a concise summary of feedback for analysis."""
        if not feedback_data:
            return "No feedback available for analysis."

        # Calculate statistics
        ratings = [f.get('rating', 0) for f in feedback_data if f.get('rating')]
        thumbs = [f.get('thumbs', '') for f in feedback_data if f.get('thumbs')]
        comments = [f for f in feedback_data if f.get('comment', '').strip()]

        avg_rating = sum(ratings) / len(ratings) if ratings else 0
        thumbs_up_count = sum(1 for t in thumbs if t == 'up')
        thumbs_down_count = sum(1 for t in thumbs if t == 'down')

        summary = f"""
Statistics:
- Average Rating: {avg_rating:.2f}/5.0 ({len(ratings)} ratings)
- Thumbs Up: {thumbs_up_count}
- Thumbs Down: {thumbs_down_count}
- Comments: {len(comments)}

Sample Comments:
"""

        for i, fb in enumerate(comments[:10], 1):
            summary += f"\n{i}. (Rating: {fb.get('rating', 'N/A')}) {fb.get('comment', '')[:150]}..."

        return summary

    def _load_system_state(self) -> Dict[str, Any]:
        """Load current system version and capabilities."""
        version_path = self.playbook_path / "meta" / "version.json"
        capabilities_path = self.playbook_path / "meta" / "capability_matrix.json"

        state = {}

        if version_path.exists():
            with open(version_path) as f:
                state.update(json.load(f))

        if capabilities_path.exists():
            with open(capabilities_path) as f:
                state["capabilities"] = json.load(f)

        return state

    def _save_insights(self, insights: Dict[str, Any]) -> None:
        """Save insights to Playbook for Curator agent."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        insights_dir = self.playbook_path / "insights" / "conversation_patterns"
        insights_dir.mkdir(parents=True, exist_ok=True)

        output_path = insights_dir / f"reflection_{timestamp}.json"

        with open(output_path, 'w') as f:
            json.dump(insights, f, indent=2)

        print(f"✅ Insights saved to: {output_path}")

    def load_conversations_from_feedback_files(
        self,
        feedback_dir: str = "/home/claudeuser/yonearth-gaia-chatbot/data/feedback"
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Load conversations and feedback from feedback JSON files.

        Returns:
            Tuple of (conversations, feedback_data)
        """
        feedback_path = Path(feedback_dir)
        conversations = []
        feedback_data = []

        if not feedback_path.exists():
            print(f"⚠️  Feedback directory not found: {feedback_dir}")
            return conversations, feedback_data

        # Load all feedback JSON files
        for feedback_file in feedback_path.glob("feedback_*.json"):
            try:
                with open(feedback_file) as f:
                    data = json.load(f)

                    # Extract feedback entries
                    if isinstance(data, list):
                        feedback_data.extend(data)
                    elif isinstance(data, dict):
                        feedback_data.append(data)

            except Exception as e:
                print(f"⚠️  Error loading {feedback_file}: {e}")

        # Convert feedback to conversation format
        # (In real implementation, you'd want to fetch actual conversation logs)
        for fb in feedback_data:
            conversations.append({
                "id": fb.get("timestamp", "unknown"),
                "messages": [
                    {"role": "user", "content": fb.get("query", "")},
                    {"role": "assistant", "content": fb.get("response", "")}
                ],
                "feedback": fb
            })

        print(f"✅ Loaded {len(conversations)} conversations from {len(feedback_data)} feedback items")

        return conversations, feedback_data


if __name__ == "__main__":
    # Example usage
    reflector = ReflectorAgent()

    # Load real conversation data from feedback files
    conversations, feedback_data = reflector.load_conversations_from_feedback_files()

    if not conversations:
        print("⚠️  No conversation data found. Add feedback files to /data/feedback/")
    else:
        # Load current playbook state
        playbook_state = reflector._load_system_state()

        # Run analysis
        print("🔍 Running Reflector analysis...")
        insights = reflector.analyze_conversation_batch(
            conversations,
            feedback_data,
            playbook_state
        )

        print("\n📊 ANALYSIS COMPLETE")
        print(f"Patterns found: {len(insights.get('patterns', []))}")
        print(f"Gaps identified: {len(insights.get('gaps', []))}")
        print(f"Recommendations: {len(insights.get('recommendations', []))}")
