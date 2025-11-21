# Self-Improving Learning-Loop System for Knowledge Graph Evolution

## Overview

This self-improving learning-loop system (inspired by the ACE framework from Zhang et al. 2024) provides automated quality improvement for knowledge graph extraction. The system uses specialized agents (Reflector and Curator) working in a continuous loop to analyze extraction quality, identify systematic errors, and evolve the system through improvements to **prompts, code, and configurations**—not just context.

**Key Distinction**: Unlike traditional context-only systems, this learning-loop improves:
- **Prompts** (Pass 1 & 2 extraction instructions)
- **Code** (Pass 2.5 post-processing modules)
- **Configurations** (thresholds, vocabularies, detection patterns)

## Architecture

### Primary Learning-Loop (Continuous)

```
┌─────────────────────────────────────────────────────────┐
│              LEARNING-LOOP ORCHESTRATOR                  │
│    (Coordinates reflection cycles and system evolution)  │
└─────────────────────────────────────────────────────────┘
                          │
                          ├──────────────────┬──────────────────┐
                          ▼                  ▼                  ▼
                   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
                   │  REFLECTOR  │   │   CURATOR   │   │  PLAYBOOK   │
                   │    AGENT    │   │    AGENT    │   │  DATABASE   │
                   └─────────────┘   └─────────────┘   └─────────────┘
                          │                  │                  │
                          │                  │                  │
                          ▼                  ▼                  ▼
                   Analyzes         Organizes           Stores
                   extraction       insights into       evolving
                   quality          actionable          system
                                   changesets           knowledge
```

### Meta-Learning-Loop (Optional, Manual)

```
                    ┌────────────────────────────┐
                    │   META-LEARNING-LOOP       │
                    │  (Reviews Reflector/       │
                    │   Curator performance)     │
                    │                            │
                    │  Trigger: Manual or        │
                    │  periodic (every 10-20     │
                    │  cycles or quality drift)  │
                    └────────────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────────┐
                    │  Sample & Validate         │
                    │  - Precision check         │
                    │  - Recall estimation       │
                    │  - Integration constraints │
                    │  - New pattern discovery   │
                    └────────────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────────┐
                    │  Meta-Recommendations      │
                    │  - Reflector improvements  │
                    │  - Curator enhancements    │
                    │  - Detection patterns      │
                    └────────────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────────┐
                    │   HUMAN APPROVAL           │
                    │  (Required before changes) │
                    └────────────────────────────┘
```

## Core Components

### 1. Reflector Agent

**Purpose**: Analyze conversations and feedback to identify improvement opportunities

**Inputs**:
- Chat conversation logs (user questions + Gaia responses)
- User feedback (ratings, comments, thumbs up/down)
- Current system state (Playbook entries, active capabilities)

**Outputs**:
- Insight reports with identified patterns
- Gap analysis (what users need vs. what system provides)
- Improvement suggestions with priority rankings

**Key Responsibilities**:
- Pattern recognition across conversations
- Feedback analysis and sentiment tracking
- Citation accuracy validation
- Response quality assessment
- Knowledge gap identification

### 2. Curator Agent

**Purpose**: Organize insights into actionable knowledge and update the Playbook

**Inputs**:
- Reflector insights and recommendations
- Existing Playbook structure
- System capabilities and constraints

**Outputs**:
- Updated Playbook entries
- New prompt templates
- Configuration changes
- Feature proposals

**Key Responsibilities**:
- Knowledge organization and categorization
- Playbook entry creation and updates
- Prompt engineering and refinement
- System capability evolution tracking
- Version control of knowledge artifacts

### 3. Playbook Database

**Purpose**: Centralized repository of evolving system knowledge

**Structure**:
```
/data/playbook/
├── meta/
│   ├── version.json              # Current system version
│   ├── evolution_log.json        # History of changes
│   └── capability_matrix.json    # Feature status tracking
├── prompts/
│   ├── gaia_core_v6.txt         # Evolved core personality
│   ├── search_optimization.txt   # Search behavior prompts
│   └── citation_templates.txt    # Reference formatting
├── insights/
│   ├── conversation_patterns/    # Reflector outputs
│   ├── user_needs/              # Identified requirements
│   └── improvement_opportunities/
├── configurations/
│   ├── search_weights.json       # BM25/semantic weights
│   ├── category_thresholds.json  # Category matching config
│   └── response_templates.json   # Response formatting
└── experiments/
    ├── a_b_tests/               # A/B test configurations
    └── feature_flags.json       # Experimental features
```

### 4. Learning-Loop Orchestrator

**Purpose**: Coordinate the reflection-curation cycle

**Workflow**:
```python
1. COLLECT: Gather extraction outputs and quality metrics
2. REFLECT: Run Reflector agent on extraction quality
3. CURATE: Run Curator agent on Reflector insights
4. UPDATE: Apply approved changes to Playbook (prompts/code/config)
5. MONITOR: Track impact of changes on extraction quality
6. ITERATE: Repeat cycle with new extractions
```

**Configuration**:
- Reflection frequency (per extraction, periodic, on-demand)
- Approval thresholds (auto-apply vs. human review)
- Rollback mechanisms for failed experiments
- Version control integration

### 5. Meta-Learning-Loop (Optional)

**Purpose**: Validate and improve the Reflector and Curator agents themselves

**When to Run**:
- **Manual trigger**: Operator initiates meta-review at will
- **Periodic**: Every 10-20 extraction cycles
- **Quality drift**: When error rates increase >5 percentage points
- **Major changes**: After updating extraction prompts or adding modules

**Key Activities**:

1. **Precision Validation**
   - Sample 20 relationships flagged by Reflector as issues
   - Validate each is truly a problem (catch false positives)
   - Calculate precision rate (target: >95%)
   - Recommend Reflector prompt adjustments if needed

2. **Recall Estimation**
   - Sample 20 non-flagged relationships
   - Check for missed issues (estimate false negatives)
   - Calculate estimated recall (target: >85%)
   - Propose new detection patterns for missed issues

3. **Integration Constraint Review**
   - Analyze Curator's recommended changes
   - Check for integration risks (e.g., string formatting, JSON escaping)
   - Add constraint warnings and validation tests
   - Document integration requirements

4. **New Pattern Discovery**
   - Review multiple Reflector reports
   - Identify recurring patterns not yet in detection rules
   - Propose new error categories and severity levels
   - Update Reflector prompt with expanded patterns

**Model Choice**:
- **Recommended**: Claude Sonnet 4.5 (same as Reflector/Curator)
- **Strategic Audits**: Claude Opus for quarterly deep reviews
- **Cost Impact**: ~9% increase vs. 640% for continuous Opus review

**Human Approval Required**: All meta-level recommendations must be reviewed and approved before modifying Reflector/Curator agents.

**Example Meta-Recommendations**:
```json
{
  "meta_improvements": [
    {
      "target": "Reflector",
      "issue": "Missing possessive pronoun detection",
      "recommendation": "Add pattern: 'my|our|their + noun' detection",
      "expected_impact": "Catch additional 12% of vague entity issues"
    },
    {
      "target": "Curator",
      "issue": "JSON examples break .format() interpolation",
      "recommendation": "Add integration constraint checks",
      "validation": "Test that changesets don't contain unescaped braces"
    }
  ]
}
```

## Agent Prompts

### Reflector Agent Prompt Structure

```
ROLE: You are a reflective AI analyst examining conversations between users
and Gaia (Earth consciousness chatbot) to identify improvement opportunities.

CONTEXT:
- System version: V5 (current) → V6 (target)
- Current capabilities: [list from Playbook]
- Recent changes: [from evolution log]

INPUTS:
- Conversation batch: {conversation_data}
- User feedback: {feedback_data}
- Current Playbook state: {playbook_snapshot}

ANALYSIS TASKS:
1. Pattern Recognition: Identify recurring user questions, topics, or confusion
2. Citation Accuracy: Verify episode/book references match user questions
3. Response Quality: Assess depth, relevance, and personality consistency
4. Knowledge Gaps: Find topics users ask about that system handles poorly
5. User Satisfaction: Correlate feedback with response characteristics

OUTPUT FORMAT:
{
  "patterns": [
    {
      "pattern": "description",
      "frequency": count,
      "examples": [conversation_ids],
      "impact": "high|medium|low"
    }
  ],
  "gaps": [
    {
      "topic": "description",
      "evidence": [conversation_ids],
      "severity": "critical|important|minor"
    }
  ],
  "recommendations": [
    {
      "category": "prompt|search|content|feature",
      "suggestion": "specific change",
      "rationale": "why this helps",
      "priority": 1-10
    }
  ],
  "metrics": {
    "conversations_analyzed": count,
    "avg_satisfaction": 0-5,
    "citation_accuracy": 0-100%,
    "common_topics": [topic_list]
  }
}
```

### Curator Agent Prompt Structure

```
ROLE: You are a knowledge curator responsible for organizing insights and
updating the Playbook to evolve the YonEarth system from V5 to V6.

CONTEXT:
- System architecture: [technical overview]
- Playbook structure: [directory layout]
- Version goals: [V6 objectives]

INPUTS:
- Reflector insights: {reflector_output}
- Current Playbook: {playbook_state}
- Change history: {evolution_log}

CURATION TASKS:
1. Insight Organization: Categorize recommendations by type and priority
2. Playbook Updates: Create/modify entries based on insights
3. Prompt Engineering: Refine system prompts for identified gaps
4. Configuration Tuning: Adjust search weights, thresholds, etc.
5. Experiment Design: Propose A/B tests for uncertain changes

OUTPUT FORMAT:
{
  "playbook_updates": [
    {
      "file_path": "/data/playbook/...",
      "change_type": "create|update|delete",
      "content": "new/updated content",
      "rationale": "why this change",
      "impact_estimate": "description"
    }
  ],
  "experiments": [
    {
      "name": "experiment name",
      "hypothesis": "what we expect",
      "configuration": {test_config},
      "success_metrics": [metrics]
    }
  ],
  "version_notes": {
    "changes_summary": "high-level description",
    "breaking_changes": [list],
    "migration_steps": [steps]
  }
}
```

## Reflection Cycle Phases

### Phase 1: Data Collection (Automated)
- Pull conversations from `/data/feedback/` directory
- Aggregate user ratings and comments
- Load current Playbook state
- Prepare batch for analysis

### Phase 2: Reflection (Agent-Driven)
- Reflector agent processes conversation batch
- Generates insight report
- Identifies high-priority improvements
- Flags critical issues requiring immediate attention

### Phase 3: Curation (Agent-Driven)
- Curator agent reviews insights
- Proposes Playbook updates
- Designs experiments for uncertain changes
- Generates version evolution plan

### Phase 4: Review & Approval (Human-in-Loop)
- Present proposed changes to developer
- Show impact analysis and rationale
- Allow selective approval/rejection
- Collect feedback on agent reasoning

### Phase 5: Implementation (Automated)
- Apply approved Playbook changes
- Update system configurations
- Deploy new prompts/templates
- Log changes to evolution history

### Phase 6: Monitoring (Continuous)
- Track user satisfaction metrics
- Monitor citation accuracy
- Measure response quality
- Detect regressions or improvements

## V5 → V6 Evolution Goals

Based on the V5 Implementation Plan, V6 should address:

**Priority 1: Enhanced Search Intelligence**
- ✅ Semantic category matching (implemented)
- ✅ Episode diversity algorithm (implemented)
- 🎯 Dynamic search weight optimization
- 🎯 Conversation context integration

**Priority 2: Improved User Experience**
- ✅ Voice integration (implemented)
- ✅ Configurable category thresholds (implemented)
- 🎯 Personalized response styles
- 🎯 Progressive conversation depth

**Priority 3: Content Intelligence**
- 🎯 Cross-content synthesis (episodes + books)
- 🎯 Timeline-aware responses
- 🎯 Contradiction detection
- 🎯 Knowledge graph integration

**Priority 4: Self-Improvement (Learning-Loop)**
- 🎯 Automated quality analysis (Reflector)
- 🎯 Prompt/code/config evolution based on errors (Curator)
- 🎯 Meta-learning-loop for agent calibration (optional)
- 🎯 Continuous quality improvement across versions

## Success Metrics

**User Satisfaction**:
- Average rating: target >4.5/5
- Thumbs up rate: target >80%
- Repeat usage: track returning users

**Citation Accuracy**:
- Episode relevance: target 95%+
- Book chapter precision: target 98%+
- Multi-source synthesis quality

**Response Quality**:
- Answer completeness (user follow-ups as metric)
- Personality consistency (tone analysis)
- Depth appropriateness (context-aware)

**System Evolution (Learning-Loop Metrics)**:
- Issues identified per reflection cycle
- Quality improvement per version (error rate reduction)
- Playbook updates deployed (prompts/code/config)
- Time-to-improvement (issue detection → fix deployment)
- Meta-learning-loop calibration accuracy (precision/recall)

## Implementation Timeline

**Week 1**: Core infrastructure
- Create Playbook directory structure
- Implement data collection scripts
- Build basic orchestrator

**Week 2**: Agent development
- Implement Reflector agent
- Implement Curator agent
- Test on sample conversations

**Week 3**: Integration
- Connect agents to orchestrator
- Build approval/review interface
- Implement change deployment

**Week 4**: Iteration & refinement
- Run first reflection cycle
- Apply improvements
- Measure impact
- Refine agent prompts

## Next Steps

1. Create Playbook directory structure
2. Implement Reflector agent with Claude Sonnet 4.5
3. Implement Curator agent with Claude Sonnet 4.5
4. Build learning-loop orchestrator script
5. Test on extraction outputs
6. Deploy first quality improvements

## Using the Meta-Learning-Loop

The meta-learning-loop is a **standalone operation** that can be triggered manually at will:

```bash
# Manual trigger - review Reflector/Curator performance
python scripts/run_meta_learning_loop.py \
  --reflector-reports /path/to/reports \
  --sample-size 20 \
  --validate-precision \
  --estimate-recall

# Output: Meta-recommendations for improving agents
# Requires: Human review and approval before applying
```

**When to Trigger**:
- After 10-20 extraction cycles (periodic maintenance)
- When quality metrics drift unexpectedly
- After major system changes (new modules, prompt updates)
- Before production deployment (validate agent accuracy)
- On-demand for strategic audits

**Cost Considerations**:
- Uses same model as primary agents (Sonnet 4.5)
- Sample-based validation (20 items), not exhaustive
- ~9% cost increase vs. continuous Opus review (640% increase)
- Human approval prevents runaway meta-optimization

**Example Output**:
```
Meta-Learning-Loop Analysis Complete
=====================================
Reflector Precision: 100% (5/5 flagged issues valid)
Reflector Recall: ~87% (2/15 non-flagged had mild issues)

Recommendations:
1. Add possessive pronoun detection to Reflector
2. Expand Curator integration constraint checks
3. New pattern: Abstract philosophical statements

Approve recommendations? (y/n)
```

## References

- **ACE Framework**: Zhang et al. (2024) - "Autonomous Cognitive Entities: Contexts as Evolving Playbooks"
- **Knowledge Graph Extraction**: Multi-stage pipeline with LLM-based extraction and validation
- **Meta-Learning**: Self-improving systems that learn from their own improvement process
