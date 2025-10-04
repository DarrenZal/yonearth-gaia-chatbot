# Features and Usage Guide

Complete guide to all features available in the YonEarth Gaia Chatbot.

---

## 🌟 Core Features

### 🧠 Advanced Category-First RAG System

The chatbot uses a sophisticated hybrid search approach that combines multiple search methods:

#### Search Methods

**1. 🌿 Original (Semantic Search)**
- Pure meaning-based understanding
- Uses OpenAI embeddings for context
- Best for conceptual questions
- Example: "regenerative practices" matches similar concepts

**2. 🔍 BM25 (Category-First Hybrid)**
- **Category Matching (60-80%)**: Episodes tagged with relevant categories
- **Semantic Search (15-25%)**: OpenAI embedding similarity
- **Keyword Search (5-15%)**: BM25 term frequency
- **Cross-Encoder Reranking**: MS-MARCO MiniLM for final relevance
- Best for specific topic queries
- Example: "biochar" matches BIOCHAR category episodes

**3. ⚖️ Both (Side-by-Side)**
- Compare results from both methods
- See differences in episode selection
- Helpful for understanding search behavior

#### ✨ Semantic Category Matching

TRUE semantic understanding using OpenAI embeddings:

- **Automatic Topic Matching**: "soil" → BIOCHAR (32.1% similarity)
- **Cached Embeddings**: Fast performance (`/data/processed/category_embeddings.json`)
- **Configurable Thresholds**:
  - **Broad (0.6)**: Explore creative connections
  - **Normal (0.7)**: Balanced matching (default)
  - **Strict (0.8)**: Only very closely related
  - **Disabled (1.1)**: No category filtering

#### Episode Diversity Algorithm

Ensures all relevant episodes appear, not just one with many chunks:
- Limits chunks per episode in results
- Guarantees episode variety
- Example: All 4 biochar episodes (120, 122, 124, 165) appear for "biochar" queries

---

## 🌱 Gaia Personality System

Choose how Gaia responds to you:

### Predefined Personalities

**🤱 Nurturing Mother (warm_mother)**
- Warm, caring, patient guidance
- Encouraging and supportive tone
- Great for beginners
- Example: "Let me gently guide you through..."

**🧙‍♂️ Ancient Sage (wise_guide)**
- Deep wisdom from Earth's timeless perspective
- Contemplative and profound
- Great for philosophical discussions
- Example: "From the depths of geological time..."

**⚡ Earth Guardian (earth_activist)**
- Passionate activist for ecological justice
- Direct and action-oriented
- Great for environmental advocacy
- Example: "We must act now to protect..."

### ✨ Custom Personalities

Create your own personalized Gaia:

1. Click **"Custom"** in personality selector
2. Edit the system prompt template
3. Customize Gaia's:
   - Tone (warm/serious/playful)
   - Focus areas (soil/climate/community)
   - Speaking style (poetic/technical/conversational)
4. Save to browser localStorage

**Example Custom Prompt**:
```
You are Gaia, speaking as a regenerative agriculture expert.
Focus on practical, actionable advice for farmers and gardeners.
Use technical terms but explain them clearly. Reference specific
episodes and books for detailed guidance.
```

---

## 🎤 Voice Integration

Hear Gaia's responses spoken aloud:

### Features
- **Text-to-Speech**: ElevenLabs AI voice technology
- **Custom Voice**: Specially cloned voice for natural speech
- **Toggle Control**: Speaker button to enable/disable
- **Auto-play**: Responses automatically play when enabled
- **Manual Replay**: Audio control button to replay
- **Persistent Settings**: Voice preference saved in browser

### How to Use
1. Click **speaker icon** to enable voice
2. Ask a question
3. Response plays automatically
4. Click **audio button** on response to replay
5. Toggle speaker icon to disable

### Browser Support
- ✅ Chrome/Edge (full support)
- ✅ Firefox (full support)
- ✅ Safari (full support)
- ⚠️ Requires modern browser with audio support

---

## 📚 Multi-Content Search

Search across both podcast episodes AND books simultaneously:

### Integrated Books (3 total)

**1. VIRIDITAS: THE GREAT HEALING**
- 2,029 searchable chunks
- Chapter-level references
- Themes: ecology, health, spirituality
- Links: eBook, audiobook, print editions

**2. Soil Stewardship Handbook**
- 136 searchable chunks
- Practical soil management
- Building healthy soil ecosystems

**3. Y on Earth: Get Smarter, Feel Better, Heal the Planet**
- 2,124 searchable chunks
- Personal and planetary wellbeing
- Sustainable living practices

### Citation Format

**Episodes**:
```
Episode 120: High Plains Biochar with Kelpie Wilson & Troy Cowan
https://yonearth.org/podcast/episode-120
```

**Books**:
```
📖 VIRIDITAS: THE GREAT HEALING - Chapter 30: Gaia Speaks
   eBook | Audiobook | Print
```

---

## 🎯 Smart Conversation Features

### Topic Tracking
- Automatically extracts conversation themes
- Identifies discussed topics
- Updates recommendations dynamically
- Example: "Based on our conversation about: permaculture, soil health..."

### Dynamic Recommendations

**Inline Citations**:
- Episodes referenced in response appear immediately below
- Books cited show chapter and format links

**Recommended Content Section**:
- Shows ALL references from entire conversation
- Evolves as conversation develops
- Prevents duplicate suggestions
- Includes both episodes and books

### Context Evolution
As you chat, recommendations become more relevant:
- Early: General episode suggestions
- Middle: Topic-specific episodes from discussion
- Later: Deep-dive episodes on specific themes

**Example**:
```
User: "What is biochar?"
Gaia: [Explains biochar, cites Episodes 120, 122]

User: "How do I make it?"
Gaia: [Explains process, cites Episode 165]
Recommendations: "Try asking about: other applications of biochar"
```

---

## 💬 User Feedback System

Help improve Gaia's responses:

### Quick Feedback
- 👍 Thumbs up: Good response
- 👎 Thumbs down: Needs improvement

### Detailed Feedback
After rating, provide:
- **Star Rating (1-5)**: Overall quality
- **Episode Correctness**: Were citations accurate?
- **Text Comments**: What could be better?

### Data Storage
- Saved to `/data/feedback/feedback_YYYY-MM-DD.json`
- Organized by date
- View with: `python scripts/view_feedback.py`

---

## 💰 Cost Tracking

Transparent breakdown of API costs:

### What's Tracked
- **Embeddings**: OpenAI embedding API calls
- **Completion**: LLM response generation
- **Voice**: ElevenLabs TTS (if enabled)
- **Total**: Sum of all costs per response

### Display
- Shows in response footer
- Format: "$0.0234 (Embeddings: $0.0012, Completion: $0.0222)"
- Helps understand usage patterns

---

## 🔧 Advanced Features

### Configurable Search Settings

**Max References (1-10)**
- Controls how many sources cited per response
- Default: 3
- More references = more comprehensive but longer

**Category Threshold**
- Controls semantic category matching strictness
- See "Semantic Category Matching" above

**Search Method**
- Choose Original, BM25, or Both
- Compare different approaches

### Browser Cache-Busting

When updating web files, increment version numbers:
```html
<!-- Old -->
<link rel="stylesheet" href="styles.css?v=2">

<!-- New (after update) -->
<link rel="stylesheet" href="styles.css?v=3">
```

This forces browsers to load fresh versions instead of cached files.

---

## 📊 Response Quality

### What Makes a Good Response

**Accurate Citations**:
- ✅ Episodes actually contain referenced content
- ✅ Book chapters correctly identified
- ✅ No hallucinated sources

**Comprehensive Coverage**:
- ✅ Multiple relevant sources (episodes + books)
- ✅ Diverse perspectives
- ✅ Both conceptual and practical info

**Clear Presentation**:
- ✅ Concise but complete answers
- ✅ Organized with sections
- ✅ Actionable recommendations

### Example Quality Response

**Query**: "What is biochar?"

**Good Response**:
```
Biochar is a carbon-rich material created through pyrolysis...

[Detailed explanation with multiple aspects]

Referenced Episodes:
• Episode 120: High Plains Biochar
• Episode 122: Dr. David Laird on Soil Carbon
• Episode 165: Kelpie Wilson on Biochar Production

📖 Soil Stewardship Handbook - Chapter 5: Carbon Sequestration

Recommended Content:
Based on this topic, you might also explore:
• Episode 124: Biochar in Agriculture
• Y on Earth - Chapter 12: Regenerative Practices
```

---

## 🎓 Best Practices

### Getting Great Results

**1. Be Specific**
- ❌ "Tell me about farming"
- ✅ "What are regenerative agriculture practices for small farms?"

**2. Use Context**
- ❌ "What about composting?"
- ✅ "How does composting relate to soil health and carbon sequestration?"

**3. Ask Follow-ups**
- Build on previous answers
- Recommendations improve with conversation depth

**4. Try Both Search Methods**
- Original for conceptual questions
- BM25 for specific topics
- Compare to see differences

**5. Adjust Settings**
- Increase max references for comprehensive answers
- Adjust category threshold for broader/narrower results
- Enable voice for accessibility

---

## 🚀 Use Cases

### For Learners
- Explore regenerative agriculture concepts
- Discover relevant episodes and books
- Build knowledge progressively through conversation

### For Practitioners
- Find specific techniques and practices
- Get episode recommendations for deep dives
- Access both theory (books) and real-world examples (podcasts)

### For Educators
- Find teaching resources
- Identify episode examples for lessons
- Access diverse perspectives on topics

### For Researchers
- Discover connections between concepts
- Find primary source material (episode transcripts)
- Track themes across episodes and books

---

## 📖 Related Documentation

- [Setup and Deployment](SETUP_AND_DEPLOYMENT.md) - Installation and configuration
- [Architecture](ARCHITECTURE.md) - System design and components
- [Development](DEVELOPMENT.md) - Development workflow and APIs

---

**Questions or Issues?** Check the [Troubleshooting](SETUP_AND_DEPLOYMENT.md#troubleshooting) section or open an issue on GitHub.
