# Claude Code Prompt: Setup Knowledge Graph Extraction for New Podcast

Copy and paste this entire prompt into Claude Code after cloning the repository:

---

## Context

I've cloned https://github.com/DarrenZal/yonearth-gaia-chatbot which contains a production-ready knowledge graph extraction system (v3.2.2). This system was built for the YonEarth podcast and I want to adapt it for my own podcast.

## My Goal

Set up the complete pipeline for extracting a knowledge graph from my podcast episodes:
1. **Transcribe audio files** with word-level timestamps using Whisper
2. **Extract knowledge graph** using the v3.2.2 three-stage system
3. **Adapt the system** for my podcast's specific needs

## Key Files to Review

Please start by reading these files to understand the existing system:

### Transcription System
1. `docs/TRANSCRIPTION_SETUP.md` - Complete transcription guide
2. `scripts/retranscribe_episodes_lightweight.py` - Whisper transcription script (word-level timestamps)
3. `scripts/retranscribe_episodes_with_timestamps.py` - Alternative transcription approach

### Knowledge Graph Extraction
4. `docs/knowledge_graph/README.md` - KG system overview
5. `docs/knowledge_graph/KG_MASTER_GUIDE_V3.md` - Complete v3.2.2 architecture
6. `scripts/extract_kg_v3_2_2.py` - Core extraction engine
7. `V3_2_2_TEST_RESULTS.md` - Test results and performance metrics

### Quick Start Guides
8. `KG_V3_2_2_QUICK_START.md` - Quick start for extraction
9. `docs/knowledge_graph/KG_V3_2_2_IMPLEMENTATION_GUIDE.md` - Implementation details

## My Podcast Details

**[Fill in your details here]**
- **Podcast name**: [Your podcast name]
- **Number of episodes**: [e.g., 50 episodes]
- **Audio format**: [e.g., MP3, M4A]
- **Audio location**: [e.g., local files, RSS feed, YouTube]
- **Average episode length**: [e.g., 45 minutes]
- **Topics/themes**: [e.g., technology, business, education]

## Tasks I Need Help With

### Phase 1: Transcription Setup (Priority)

1. **Review the existing Whisper transcription approach**
   - Look at `scripts/retranscribe_episodes_lightweight.py`
   - Understand how YonEarth did word-level timestamp transcription
   - Note: They transcribed 172 episodes with word-level timestamps successfully

2. **Help me adapt the transcription script for my podcast**
   - Modify file paths and naming conventions
   - Adjust for my audio file format/location
   - Add YouTube fallback if needed (see how YonEarth handled 14 episodes from YouTube)
   - Ensure output format matches what the KG extraction expects

3. **Create a test transcription workflow**
   - Test on 1-2 episodes first
   - Verify output format is correct
   - Check word-level timestamp quality

4. **Key questions to answer:**
   - What Whisper model should I use? (YonEarth used `base` model)
   - How do I handle episodes with missing/broken audio?
   - What's the expected output JSON format?
   - How long will transcription take for my episodes?
   - What are the costs/requirements?

### Phase 2: KG Extraction Adaptation (Later)

5. **Adapt the extraction system for my podcast**
   - Modify episode loading paths
   - Adjust any YonEarth-specific entity types
   - Update documentation references

6. **Run test extraction**
   - Use the same 1-2 test episodes
   - Verify results quality
   - Compare with expected output format

## Specific Questions

1. **Where are the YonEarth audio files located?** (so I know where to put mine)
2. **What's the exact output format** from the transcription script?
3. **What environment variables** do I need to set?
4. **What are the Python dependencies** for transcription? (I see `requirements.txt`)
5. **How much disk space** do I need for transcripts and KG data?
6. **Can I run this on a laptop** or do I need a server?

## Expected Output

Please provide:
1. **Step-by-step transcription setup guide** tailored to my podcast
2. **Modified transcription script** (or instructions to modify)
3. **Test command** to transcribe 1 episode
4. **Validation checklist** to ensure transcripts are correct
5. **Next steps** for KG extraction after transcription is working

## Installation Already Done

- ✅ Cloned repository
- ✅ Python environment ready
- ⏳ Need to install dependencies
- ⏳ Need to set up OpenAI API key
- ⏳ Need to configure for my podcast

## Notes

- The YonEarth system uses OpenAI's `gpt-4o-mini` for extraction (~$6 for 172 episodes)
- Whisper transcription is local (no API costs)
- The system achieved 95% high-confidence relationships in testing
- Word-level timestamps enable precise audio navigation
- Processing time: ~13 minutes per episode for KG extraction

---

## Action Items

**Start with these tasks:**

1. Read `docs/TRANSCRIPTION_SETUP.md` and explain the YonEarth transcription approach
2. Show me the exact JSON output format expected by the KG extraction system
3. Help me create a modified transcription script for my podcast
4. Create a test workflow to transcribe my first episode

Let's focus on **transcription first** - once that's working, we'll move to knowledge graph extraction.
