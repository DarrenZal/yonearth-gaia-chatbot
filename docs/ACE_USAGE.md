# ACE Framework Usage Guide

## What is ACE?

ACE (Autonomous Cognitive Entity) is a self-reflective framework that enables the YonEarth chatbot to evolve autonomously by:
1. **Analyzing** user conversations and feedback
2. **Identifying** improvement opportunities
3. **Proposing** system changes
4. **Applying** approved updates
5. **Tracking** evolution over time

## Architecture Components

### 1. Reflector Agent (`src/ace/reflector.py`)
Analyzes conversations to identify patterns, gaps, and improvement opportunities.

**What it does:**
- Examines user questions and Gaia's responses
- Analyzes user feedback (ratings, comments, thumbs)
- Identifies recurring patterns and user needs
- Detects knowledge gaps and response quality issues
- Generates prioritized recommendations

**Output:** Insight report with patterns, gaps, and recommendations

### 2. Curator Agent (`src/ace/curator.py`)
Transforms insights into actionable Playbook updates.

**What it does:**
- Organizes recommendations by type and priority
- Creates specific Playbook file updates
- Proposes prompt improvements
- Designs configuration tweaks
- Plans A/B experiments for uncertain changes

**Output:** Curation plan with file updates and evolution strategy

### 3. Orchestrator (`src/ace/orchestrator.py`)
Coordinates the reflection-curation-application cycle.

**What it does:**
- Collects conversation and feedback data
- Runs Reflector and Curator agents
- Presents changes for review
- Applies approved updates
- Tracks evolution history

**Output:** Cycle results with applied changes and metrics

## The Playbook

The Playbook (`/data/playbook/`) is the central knowledge repository that evolves through ACE cycles:

```
/data/playbook/
├── meta/                      # System version and evolution tracking
│   ├── version.json          # Current version and features
│   ├── evolution_log.json    # History of all changes
│   └── capability_matrix.json # Feature status
├── prompts/                   # AI prompts that evolve
│   └── gaia_core_v6.txt      # Gaia's personality prompt
├── configurations/            # System settings
│   ├── search_weights.json   # Search algorithm weights
│   ├── category_thresholds.json
│   └── response_templates.json
├── insights/                  # Agent outputs
│   ├── conversation_patterns/
│   ├── user_needs/
│   └── improvement_opportunities/
└── experiments/               # A/B tests
    ├── a_b_tests/
    └── feature_flags.json
```

## Running ACE Cycles

### Quick Start

```bash
# Run a single reflection cycle (interactive mode)
python scripts/run_ace_cycle.py

# Auto-apply low-risk changes
python scripts/run_ace_cycle.py --auto-apply

# Non-interactive mode (for automation)
python scripts/run_ace_cycle.py --auto-apply --non-interactive

# View evolution summary
python scripts/run_ace_cycle.py --summary
```

### The Reflection Cycle

Each cycle follows these stages:

**1. COLLECT** - Gather conversation data
```
Loads feedback files from /data/feedback/
Extracts conversations and user ratings
```

**2. REFLECT** - Analyze conversations
```
Reflector agent examines:
- User question patterns
- Response quality
- Citation accuracy
- Knowledge gaps

Outputs: Insight report
```

**3. CURATE** - Plan improvements
```
Curator agent creates:
- Playbook updates
- Prompt improvements
- Configuration tweaks
- Experiment proposals

Outputs: Curation plan
```

**4. REVIEW** - Approve changes
```
Interactive: Review and approve each change
Auto-mode: Approve low-risk changes automatically
```

**5. APPLY** - Execute approved updates
```
Updates Playbook files
Modifies configurations
Logs evolution history
```

**6. MONITOR** - Track impact
```
Next cycle measures impact:
- User satisfaction changes
- Citation accuracy improvements
- New patterns emerging
```

## Example Workflow

### Scenario: Users struggling with book references

**Day 1: Collect Feedback**
```bash
# Users interact with Gaia
# Feedback is saved to /data/feedback/feedback_2025-10-12.json
```

**Day 2: Run Reflection Cycle**
```bash
python scripts/run_ace_cycle.py --auto-apply
```

**What Happens:**

1. **Reflector identifies pattern:**
```json
{
  "pattern": "Users ask about book content but responses cite episodes",
  "frequency": 15,
  "impact": "high",
  "insight": "Book search may be underweighted in hybrid retriever"
}
```

2. **Curator proposes fix:**
```json
{
  "file_path": "/data/playbook/configurations/search_weights.json",
  "change_type": "update",
  "content": {
    "book_content_boost": 1.2
  },
  "rationale": "Boost book results for book-specific queries",
  "risk_level": "low"
}
```

3. **Orchestrator applies change:**
```
✅ Updated config: search_weights.book_content_boost = 1.2
```

**Day 3+: Monitor Impact**
```bash
# Next cycle will show:
# - Book citation accuracy improved
# - User satisfaction increased
# - Pattern frequency decreased
```

## Manual Agent Usage

### Run Reflector Independently

```python
from src.ace.reflector import ReflectorAgent

reflector = ReflectorAgent()

# Load data
conversations, feedback = reflector.load_conversations_from_feedback_files()

# Analyze
playbook_state = reflector._load_system_state()
insights = reflector.analyze_conversation_batch(
    conversations,
    feedback,
    playbook_state
)

print(insights)
```

### Run Curator Independently

```python
from src.ace.curator import CuratorAgent
import json

curator = CuratorAgent()

# Load insights from previous Reflector run
with open('/data/playbook/insights/conversation_patterns/reflection_20251012_143022.json') as f:
    insights = json.load(f)

# Generate curation plan
playbook_state = curator._load_system_context()
plan = curator.curate_insights(insights, playbook_state)

# Apply changes
results = curator.apply_approved_updates(plan, auto_apply_low_risk=True)
print(results)
```

## Understanding Results

### Insight Report Structure

```json
{
  "patterns": [
    {
      "pattern": "Users ask multi-step questions",
      "frequency": 23,
      "examples": ["conv_123", "conv_456"],
      "impact": "high",
      "insight": "Need conversational context integration"
    }
  ],
  "gaps": [
    {
      "topic": "Timeline awareness",
      "evidence": ["conv_789"],
      "severity": "important",
      "improvement_needed": "Track when topics evolved across episodes"
    }
  ],
  "recommendations": [
    {
      "category": "feature",
      "suggestion": "Add conversation history to search context",
      "priority": 9,
      "effort": "high",
      "expected_impact": "Better multi-turn conversations"
    }
  ],
  "metrics": {
    "avg_satisfaction": 4.2,
    "citation_accuracy_sample": "47/50 accurate",
    "common_topics": ["biochar", "composting", "regenerative ag"]
  }
}
```

### Curation Plan Structure

```json
{
  "playbook_updates": [
    {
      "file_path": "/data/playbook/prompts/gaia_core_v6.txt",
      "change_type": "update",
      "content": "Updated prompt with conversation context...",
      "rationale": "Addresses multi-turn conversation gap",
      "risk_level": "medium",
      "requires_approval": true
    }
  ],
  "configuration_changes": [
    {
      "setting_path": "search_weights.context_boost",
      "current_value": 0.0,
      "proposed_value": 0.15,
      "rationale": "Boost results matching conversation context"
    }
  ],
  "experiments": [
    {
      "name": "conversation_context_integration",
      "hypothesis": "Using conversation history improves response relevance",
      "success_metrics": ["user_satisfaction", "follow_up_questions"]
    }
  ],
  "priorities": {
    "immediate": ["Fix book citation weighting"],
    "short_term": ["Add conversation context"],
    "long_term": ["Knowledge graph integration"]
  }
}
```

## Best Practices

### 1. Regular Cycles
Run ACE cycles on a regular schedule:
- **Daily**: If actively collecting feedback
- **Weekly**: For stable production systems
- **On-demand**: After major changes or issues

### 2. Review High-Risk Changes
Always manually review changes with:
- `"risk_level": "high"`
- `"requires_approval": true`
- Prompt modifications
- Major feature additions

### 3. Monitor Evolution
Track system evolution over time:

```bash
# View all cycles
python scripts/run_ace_cycle.py --summary

# Check version progression
cat /data/playbook/meta/version.json

# Review evolution history
cat /data/playbook/meta/evolution_log.json
```

### 4. Rollback if Needed
If a change causes issues:

```python
# Revert configuration change
from src.ace.curator import CuratorAgent

curator = CuratorAgent()
curator._apply_configuration_change({
    "setting_path": "search_weights.category_match",
    "proposed_value": 0.60,  # Previous value
    "rollback": True
})
```

## Automation

### Daily Cron Job

```bash
# Add to crontab
0 2 * * * cd /home/claudeuser/yonearth-gaia-chatbot && \
  python scripts/run_ace_cycle.py --auto-apply --non-interactive >> \
  /var/log/ace_cycles.log 2>&1
```

### Continuous Monitoring

```python
# Monitor script (runs continuously)
import time
from src.ace.orchestrator import ACEOrchestrator

orchestrator = ACEOrchestrator()

while True:
    # Check for new feedback
    conversations, feedback = orchestrator._collect_data()

    # Run cycle if enough new data (e.g., 10+ items)
    if len(feedback) >= 10:
        orchestrator.run_reflection_cycle(
            auto_apply_low_risk=True,
            interactive_approval=False
        )

    # Wait 24 hours
    time.sleep(86400)
```

## Troubleshooting

### No Data to Analyze
```
⚠️ No new data to analyze. Exiting cycle.
```
**Solution:** Add feedback data to `/data/feedback/` directory

### Agent API Errors
```
❌ CYCLE FAILED: OpenAI API error
```
**Solution:** Check `OPENAI_API_KEY` environment variable

### File Permission Issues
```
❌ Permission denied: /data/playbook/...
```
**Solution:** Ensure write permissions on Playbook directory

### Invalid JSON in Playbook
```
⚠️ Error loading search_weights.json
```
**Solution:** Validate and fix JSON syntax in configuration files

## V5 → V6 Evolution Goals

The ACE framework is designed to achieve these V6 goals:

**Search Intelligence:**
- ✅ Semantic category matching (V5)
- ✅ Episode diversity (V5)
- 🎯 Dynamic search weights (V6)
- 🎯 Conversation context (V6)

**User Experience:**
- ✅ Voice integration (V5)
- ✅ Configurable thresholds (V5)
- 🎯 Personalized responses (V6)
- 🎯 Progressive depth (V6)

**Content Intelligence:**
- 🎯 Cross-content synthesis (V6)
- 🎯 Timeline awareness (V6)
- 🎯 Knowledge graph integration (V6)

**Self-Improvement:**
- 🎯 Automated insights (V6 - ACE)
- 🎯 Prompt evolution (V6 - ACE)
- 🎯 A/B testing (V6 - ACE)

## Next Steps

1. **Collect Initial Feedback:**
   - Have users interact with Gaia
   - Gather ratings and comments

2. **Run First Cycle:**
   ```bash
   python scripts/run_ace_cycle.py
   ```

3. **Review Insights:**
   - Check `/data/playbook/insights/conversation_patterns/`
   - Understand user needs and patterns

4. **Apply Improvements:**
   - Approve proposed changes
   - Monitor impact in next cycle

5. **Iterate:**
   - Run regular cycles
   - Track version evolution
   - Measure user satisfaction improvements

---

**For More Information:**
- [ACE Framework Design](ACE_FRAMEWORK_DESIGN.md)
- [Playbook Structure](../data/playbook/README.md)
- [YonEarth Main Docs](../CLAUDE.md)
