# V4 EXTRACTION QUALITY ISSUES REPORT
**Generated**: 2025-10-12 06:16:04  
**Extraction Version**: v4_comprehensive  
**Book**: Soil Stewardship Handbook  
**Total Relationships**: 873  
**Problematic Relationships**: 234 (26.8%)

---

## 📊 EXECUTIVE SUMMARY

The V4 comprehensive extraction successfully increased quantity (873 relationships, 63% page coverage), but introduced significant quality issues that need to be addressed in V5.

### Quality Issues Breakdown

| Issue Type | Count | % of Total |
|------------|-------|------------|
| **Pronoun Sources** | 75 | 8.6% |
| **List Targets** | 100 | 11.5% |
| **Vague Sources** | 36 | 4.1% |
| **Vague Targets** | 20 | 2.3% |
| **Generic Entities** | 2 | 0.2% |
| **Pronoun Targets** | 1 | 0.1% |
| **TOTAL** | 234 | 26.8% |

---

## 🚨 ISSUE #1: PRONOUN SOURCES (75 relationships)

**Problem**: Extracted "He", "We", "They" instead of resolving to actual entity names.

**Why it's wrong**: Pronouns lack specificity and are useless in a knowledge graph without context.

### Examples (First 10):


**1. Pronoun: "we"**
- **Triple**: (we, cultivate, victory gardens)
- **Evidence**: "As we each cultivate our victory gardens, we will establish the kind of liberty and grounded democracy that our forebears envisioned for us."
- **Page**: 3
- **What's wrong**: "we" is a pronoun referring to an unnamed person
- **Should be**: Resolve to actual person's name from context


**2. Pronoun: "we"**
- **Triple**: (we, must take action to, help nourish the soil)
- **Evidence**: "we must all take action now to help nourish the soil."
- **Page**: 3
- **What's wrong**: "we" is a pronoun referring to an unnamed person
- **Should be**: Resolve to actual person's name from context


**3. Pronoun: "we"**
- **Triple**: (we, have, a deep tradition of family farming)
- **Evidence**: "we have a deep tradition of family farming."
- **Page**: 10
- **What's wrong**: "we" is a pronoun referring to an unnamed person
- **Should be**: Resolve to actual person's name from context


**4. Pronoun: "we"**
- **Triple**: (we, connect to, our soil)
- **Evidence**: "We connect to our soil within the loving context of family."
- **Page**: 10
- **What's wrong**: "we" is a pronoun referring to an unnamed person
- **Should be**: Resolve to actual person's name from context


**5. Pronoun: "we"**
- **Triple**: (we, will find, hardly anything is as complex as the living web of interconnectedness)
- **Evidence**: "we will find that hardly anything is as complex as the living web of interconnectedness."
- **Page**: 10
- **What's wrong**: "we" is a pronoun referring to an unnamed person
- **Should be**: Resolve to actual person's name from context


**6. Pronoun: "we"**
- **Triple**: (we, have the choice to, thrive and to heal)
- **Evidence**: "We have the choice to thrive and to heal."
- **Page**: 10
- **What's wrong**: "we" is a pronoun referring to an unnamed person
- **Should be**: Resolve to actual person's name from context


**7. Pronoun: "we"**
- **Triple**: (we, can connect with, the living soil)
- **Evidence**: "by connecting with the living soil."
- **Page**: 10
- **What's wrong**: "we" is a pronoun referring to an unnamed person
- **Should be**: Resolve to actual person's name from context


**8. Pronoun: "we"**
- **Triple**: (we, can cultivate, a deep awareness of the awesome miracle that is life on Earth)
- **Evidence**: "and by cultivating a deep awareness of the awesome miracle that is life on Earth."
- **Page**: 10
- **What's wrong**: "we" is a pronoun referring to an unnamed person
- **Should be**: Resolve to actual person's name from context


**9. Pronoun: "we"**
- **Triple**: (we, are, living descendants of peoples who have long and beautiful traditions of connectedness to the land)
- **Evidence**: "we are all living descendants of peoples who have long and beautiful traditions of connectedness to the land."
- **Page**: 10
- **What's wrong**: "we" is a pronoun referring to an unnamed person
- **Should be**: Resolve to actual person's name from context


**10. Pronoun: "we"**
- **Triple**: (we, are human, being connected to land and soil)
- **Evidence**: "For we are human, and being connected to land and soil is what it means to be human."
- **Page**: 10
- **What's wrong**: "we" is a pronoun referring to an unnamed person
- **Should be**: Resolve to actual person's name from context


**Total pronoun sources**: 75

**Breakdown**:
- `we`: 63 occurrences
- `he`: 10 occurrences
- `it`: 2 occurrences

---

## 🚨 ISSUE #2: LIST TARGETS (100 relationships)

**Problem**: Target contains comma-separated lists like "houseplants, gardens, yards and neighborhood parks".

**Why it's wrong**: Each item in the list should be a separate relationship for proper knowledge graph structure.

### Examples (First 10):


**1. List: "Partnership and Development, REVERB"**
- **Triple**: (Tanner Watt, is the Director of, Partnership and Development, REVERB)
- **Evidence**: "—Tanner Watt Director of Partnership and Development, REVERB."
- **Page**: 3
- **What's wrong**: Target contains 2 comma-separated items
- **Should be**: Split into 2 separate relationships:
  - (Tanner Watt, is the Director of, Partnership)
  - (Tanner Watt, is the Director of, REVERB)


**2. List: "positively influence your families, communities and planet"**
- **Triple**: (Learning about the power of soil, is an exciting way to, positively influence your families, communities and planet)
- **Evidence**: "Learning about the power of soil and engaging in soil-building practices is an exciting way to positively influence your families, communities and planet."
- **Page**: 3
- **What's wrong**: Target contains 2 comma-separated items
- **Should be**: Split into 2 separate relationships:
  - (Learning about the power of soil, is an exciting way to, positively influence your families)
  - (Learning about the power of soil, is an exciting way to, communities)


**3. List: "a vast realm of physical, mental and spiritual growth, health and vitality"**
- **Triple**: (cultivating a personal relationship with soil, unlocks the door to, a vast realm of physical, mental and spiritual growth, health and vitality)
- **Evidence**: "by cultivating a personal relationship with soil, we unlock the door to a vast realm of physical, mental and spiritual growth, health and vitality."
- **Page**: 11
- **What's wrong**: Target contains 3 comma-separated items
- **Should be**: Split into 3 separate relationships:
  - (cultivating a personal relationship with soil, unlocks the door to, a vast realm of physical)
  - (cultivating a personal relationship with soil, unlocks the door to, mental)
  - (cultivating a personal relationship with soil, unlocks the door to, health)


**4. List: "emotional balance, physical health and mental acuity"**
- **Triple**: (connection with soil, is key to, emotional balance, physical health and mental acuity)
- **Evidence**: "Our connection with soil is key to emotional balance, physical health and mental acuity."
- **Page**: 14
- **What's wrong**: Target contains 2 comma-separated items
- **Should be**: Split into 2 separate relationships:
  - (connection with soil, is key to, emotional balance)
  - (connection with soil, is key to, physical health)


**5. List: "clean drinking water, climate stabilization and environmental healing"**
- **Triple**: (healthy soil, is key to, clean drinking water, climate stabilization and environmental healing)
- **Evidence**: "Healthy soil is also key to clean drinking water, climate stabilization and environmental healing world-wide."
- **Page**: 14
- **What's wrong**: Target contains 2 comma-separated items
- **Should be**: Split into 2 separate relationships:
  - (healthy soil, is key to, clean drinking water)
  - (healthy soil, is key to, climate stabilization)


**6. List: "intelligence, health and well-being"**
- **Triple**: (soil, enhances, intelligence, health and well-being)
- **Evidence**: "Through soil, we: Enhance our intelligence, health and well-being."
- **Page**: 14
- **What's wrong**: Target contains 2 comma-separated items
- **Should be**: Split into 2 separate relationships:
  - (soil, enhances, intelligence)
  - (soil, enhances, health)


**7. List: "our food, our fuel, and our shelter"**
- **Triple**: (soil, will grow, our food, our fuel, and our shelter)
- **Evidence**: "Husband it and it will grow our food, our fuel, and our shelter."
- **Page**: 14
- **What's wrong**: Target contains 3 comma-separated items
- **Should be**: Split into 3 separate relationships:
  - (soil, will grow, our food)
  - (soil, will grow, our fuel)
  - (soil, will grow, and our shelter)


**8. List: "heal existing soil, create more living soil, cultivate community, and reverse climate change, enhance our own health and well-being"**
- **Triple**: (global Guild movement, supports, heal existing soil, create more living soil, cultivate community, and reverse climate change, enhance our own health and well-being)
- **Evidence**: "By choosing to join the global Guild movement, we will heal existing soil, create more living soil, cultivate community, and reverse climate change—all while enhancing our own health and well-being!"
- **Page**: 15
- **What's wrong**: Target contains 5 comma-separated items
- **Should be**: Split into 5 separate relationships:
  - (global Guild movement, supports, heal existing soil)
  - (global Guild movement, supports, create more living soil)
  - (global Guild movement, supports, cultivate community)
  - (global Guild movement, supports, and reverse climate change)
  - (global Guild movement, supports, enhance our own health)


**9. List: "made possible by a variety of detritivorous worms, nematodes, and micro-organisms"**
- **Triple**: (process, is-a, made possible by a variety of detritivorous worms, nematodes, and micro-organisms)
- **Evidence**: "The process is made possible by a variety of detritivorous worms, nematodes, and micro-organisms."
- **Page**: 15
- **What's wrong**: Target contains 3 comma-separated items
- **Should be**: Split into 3 separate relationships:
  - (process, is-a, made possible by a variety of detritivorous worms)
  - (process, is-a, nematodes)
  - (process, is-a, and micro-organisms)


**10. List: "Create Compost Guilds, Host Soil Building Parties, Create Soil Installations, Visit Organic Farms and Forests, Establish Community Composting Programs, Organize Tree Planting Parties, Organize Soil Building Flash Mob Parties"**
- **Triple**: (community, work and neighborhood groups, can do, Create Compost Guilds, Host Soil Building Parties, Create Soil Installations, Visit Organic Farms and Forests, Establish Community Composting Programs, Organize Tree Planting Parties, Organize Soil Building Flash Mob Parties)
- **Evidence**: "Th ese are the activities we can do in community, work and neighborhood groups: 022 Create Compost Guilds 022 Host Soil Building Parties 022 Create Soil Installations 022 Visit Organic Farms and Forests 022 Establish Community Composting Programs 022 Organize Tree Planting Parties 022 Organize Soil Building Flash Mob Parties."
- **Page**: 15
- **What's wrong**: Target contains 7 comma-separated items
- **Should be**: Split into 7 separate relationships:
  - (community, work and neighborhood groups, can do, Create Compost Guilds)
  - (community, work and neighborhood groups, can do, Host Soil Building Parties)
  - (community, work and neighborhood groups, can do, Create Soil Installations)
  - (community, work and neighborhood groups, can do, Visit Organic Farms)
  - (community, work and neighborhood groups, can do, Establish Community Composting Programs)
  - (community, work and neighborhood groups, can do, Organize Tree Planting Parties)
  - (community, work and neighborhood groups, can do, Organize Soil Building Flash Mob Parties)


**Total list targets**: 100

---

## 🚨 ISSUE #3: VAGUE SOURCES (36 relationships)

**Problem**: Source entities are vague concepts like "the amount of carbon", "the process", "the way".

**Why it's wrong**: Missing critical context that makes the relationship meaningful.

### Examples (First 10):


**1. Vague: "This Soil Stewardship Handbook"**
- **Triple**: (This Soil Stewardship Handbook, is an excellent tool for, engage with this critical mission and quest)
- **Evidence**: "This Soil Stewardship Handbook is an excellent tool for us to engage with this critical mission and quest."
- **Page**: 3
- **What's wrong**: "This Soil Stewardship Handbook" is incomplete/lacks context
- **Context needed**: Specify what kind of amount/process/way this refers to


**2. Vague: "This is an important little book"**
- **Triple**: (This is an important little book, can be of immediate use to, anyone who wants to help restore a greener Earth)
- **Evidence**: "This is an important little book that can be of immediate use to anyone who wants to help restore a greener Earth."
- **Page**: 3
- **What's wrong**: "This is an important little book" is incomplete/lacks context
- **Context needed**: Specify what kind of amount/process/way this refers to


**3. Vague: "This handbook"**
- **Triple**: (This handbook, provides, a nicely explained overview of the science)
- **Evidence**: "This handbook provides a nicely explained overview of the science with an inclusion of several easy steps."
- **Page**: 3
- **What's wrong**: "This handbook" is incomplete/lacks context
- **Context needed**: Specify what kind of amount/process/way this refers to


**4. Vague: "the way that relationship"**
- **Triple**: (the way that relationship, can affect, so many aspects of life)
- **Evidence**: "the way that relationship can affect so many aspects of life."
- **Page**: 3
- **What's wrong**: "the way that relationship" is incomplete/lacks context
- **Context needed**: Specify what kind of amount/process/way this refers to


**5. Vague: "this Soil Stewardship Handbook"**
- **Triple**: (this Soil Stewardship Handbook, guides us through, daily life practices and decisions to improve our quality of life)
- **Evidence**: "this Soil Stewardship Handbook guides us through daily life practices and decisions to improve our quality of life."
- **Page**: 3
- **What's wrong**: "this Soil Stewardship Handbook" is incomplete/lacks context
- **Context needed**: Specify what kind of amount/process/way this refers to


**6. Vague: "this Soil Stewardship Handbook"**
- **Triple**: (this Soil Stewardship Handbook, helps, heal the planet)
- **Evidence**: "and help heal the planet—wherever we are!"
- **Page**: 3
- **What's wrong**: "this Soil Stewardship Handbook" is incomplete/lacks context
- **Context needed**: Specify what kind of amount/process/way this refers to


**7. Vague: "this handbook"**
- **Triple**: (this handbook, invites us to, re-orient our lifestyle toward ecological justice)
- **Evidence**: "that invites us to re-orient our lifestyle toward ecological justice."
- **Page**: 3
- **What's wrong**: "this handbook" is incomplete/lacks context
- **Context needed**: Specify what kind of amount/process/way this refers to


**8. Vague: "this Soil Stewardship Handbook"**
- **Triple**: (this Soil Stewardship Handbook, is, deceptively small and simple)
- **Evidence**: "This Soil Stewardship Handbook is deceptively small and simple."
- **Page**: 10
- **What's wrong**: "this Soil Stewardship Handbook" is incomplete/lacks context
- **Context needed**: Specify what kind of amount/process/way this refers to


**9. Vague: "this Soil Stewardship Handbook"**
- **Triple**: (this Soil Stewardship Handbook, is full of, some of the most salient insights)
- **Evidence**: "it is chock full of some of the most salient insights."
- **Page**: 10
- **What's wrong**: "this Soil Stewardship Handbook" is incomplete/lacks context
- **Context needed**: Specify what kind of amount/process/way this refers to


**10. Vague: "this Soil Stewardship Handbook"**
- **Triple**: (this Soil Stewardship Handbook, is a, road-map of sorts)
- **Evidence**: "The book before you is a road-map of sorts."
- **Page**: 10
- **What's wrong**: "this Soil Stewardship Handbook" is incomplete/lacks context
- **Context needed**: Specify what kind of amount/process/way this refers to


**Total vague sources**: 36

---

## 🚨 ISSUE #4: VAGUE TARGETS (20 relationships)

**Problem**: Target entities are vague like "this process", "these practices", "some methods".

**Why it's wrong**: Lacks specificity needed for meaningful knowledge graph connections.

### Examples (First 10):


**1. Vague: "so many aspects of life"**
- **Triple**: (the way that relationship, can affect, so many aspects of life)
- **Evidence**: "the way that relationship can affect so many aspects of life."
- **Page**: 3
- **What's wrong**: "so many aspects of life" is too generic


**2. Vague: "some of the most salient insights"**
- **Triple**: (this Soil Stewardship Handbook, is full of, some of the most salient insights)
- **Evidence**: "it is chock full of some of the most salient insights."
- **Page**: 10
- **What's wrong**: "some of the most salient insights" is too generic


**3. Vague: "some of the simplest practices"**
- **Triple**: (the way through and out of these challenges, include, some of the simplest practices)
- **Evidence**: "the way through and out of these challenges include some of the simplest."
- **Page**: 10
- **What's wrong**: "some of the simplest practices" is too generic


**4. Vague: "an awesome adventure together"**
- **Triple**: (we, are embarking on, an awesome adventure together)
- **Evidence**: "We are embarking on an awesome adventure together."
- **Page**: 11
- **What's wrong**: "an awesome adventure together" is too generic


**5. Vague: "many questions on this journey together"**
- **Triple**: (soil, is the answer to, many questions on this journey together)
- **Evidence**: "Soil is the answer. We are asking many questions on this journey together."
- **Page**: 11
- **What's wrong**: "many questions on this journey together" is too generic


**6. Vague: "many group activities"**
- **Triple**: (we, can engage in, many group activities)
- **Evidence**: "we can engage in many group activities to cultivate our experience."
- **Page**: 14
- **What's wrong**: "many group activities" is too generic


**7. Vague: "wealth that can sustain any community, economy or nation"**
- **Triple**: (photosynthetic process, derives, wealth that can sustain any community, economy or nation)
- **Evidence**: "Ultimately, the only wealth that can sustain any community, economy or nation is derived from the photosynthetic process014green plants growing on regenerating soil."
- **Page**: 15
- **What's wrong**: "wealth that can sustain any community, economy or nation" is too generic


**8. Vague: "some of our best insights, ideas and inspiration occur while we’re connecting directly with living soil"**
- **Triple**: (some of us, know from personal experience that, some of our best insights, ideas and inspiration occur while we’re connecting directly with living soil)
- **Evidence**: "Some of us know from personal experience that some of our best insights, ideas and inspiration occur while we’re connecting directly with living soil."
- **Page**: 18
- **What's wrong**: "some of our best insights, ideas and inspiration occur while we’re connecting directly with living soil" is too generic


**9. Vague: "this simple truth"**
- **Triple**: (we, must understand, this simple truth)
- **Evidence**: "it is critical that we understand this simple truth."
- **Page**: 19
- **What's wrong**: "this simple truth" is too generic


**10. Vague: "the amount of carbon in the atmosphere by over 40%"**
- **Triple**: (human activity, has increased, the amount of carbon in the atmosphere by over 40%)
- **Evidence**: "In other words, our human activity has increased the amount of carbon in the atmosphere by over 40%!"
- **Page**: 20
- **What's wrong**: "the amount of carbon in the atmosphere by over 40%" is too generic


**Total vague targets**: 20

---

## 🚨 ISSUE #5: GENERIC ENTITIES (2 relationships)

**Problem**: Overly generic entities like "Publications", "Information", "Data".

**Why it's wrong**: Too broad to be useful in knowledge graph.

### Examples:


**1. Generic: "people" (source)**
- **Triple**: (people, can take, to realize a regenerative future)
- **Evidence**: "that people can take to realize a regenerative future."
- **Page**: 3
- **What's wrong**: "people" is too generic


**2. Generic: "activities" (source)**
- **Triple**: (activities, enhance, impact within our communities)
- **Evidence**: "to cultivate our experience and enhance our impact within our communities."
- **Page**: 14
- **What's wrong**: "activities" is too generic


**Total generic entities**: 2

---

## 📊 PAGE DISTRIBUTION OF PROBLEMS

Let's see which pages have the most quality issues:

| Page | Problem Count |
|------|---------------|
| 10 | 22 |
| 28 | 21 |
| 22 | 19 |
| 20 | 16 |
| 14 | 15 |
| 51 | 14 |
| 3 | 13 |
| 11 | 12 |
| 25 | 9 |
| 18 | 8 |
| 27 | 7 |
| 50 | 7 |
| 19 | 6 |
| 15 | 6 |
| 33 | 6 |
| 40 | 6 |
| 43 | 6 |
| 48 | 6 |
| 29 | 5 |
| 46 | 5 |

---

## 🎯 ROOT CAUSES

### 1. LLM Extracting Too Literally
The comprehensive prompt encouraged extraction of "EVERYTHING", leading to:
- Extracting pronouns without resolution
- Extracting incomplete context
- Not inferring entity names from surrounding text

### 2. No Post-Processing Validation
V4 has type validation and geographic validation, but NO:
- ❌ Pronoun resolution
- ❌ List splitting
- ❌ Context enrichment
- ❌ Vague entity detection

### 3. Insufficient Few-Shot Examples
The 4 few-shot examples didn't show:
- How to resolve pronouns to entities
- How to split list relationships
- How to provide full context for concepts

---

## 🔧 RECOMMENDED FIXES FOR V5

### Priority 1: Post-Processing Pipeline (HIGH)

Add a **Pass 2.5** between evaluation and final output:

1. **Pronoun Resolution**:
   - Detect pronouns (He, She, We, They)
   - Look back in evidence text for antecedent
   - Resolve to actual entity name or flag for review

2. **List Splitting**:
   - Detect comma-separated targets
   - Split into N separate relationships
   - Preserve all metadata and evidence

3. **Context Enrichment**:
   - Detect vague concepts ("the amount", "the process")
   - Expand with context from evidence text
   - E.g., "the amount of carbon" → "atmospheric carbon concentration increase"

4. **Generic Entity Detection**:
   - Flag overly generic entities
   - Either expand with specifics or filter out

### Priority 2: Improved Few-Shot Examples (HIGH)

Add examples showing:

```
BAD: (He, resides in, Colorado)
GOOD: (Aaron William Perry, resides in, Colorado)

BAD: (biochar, is used for, houseplants, gardens, yards)
GOOD: Split into:
  - (biochar, is used for, houseplants)
  - (biochar, is used for, gardens)
  - (biochar, is used for, yards)

BAD: (the amount, is equivalent to, 243 billion tons)
GOOD: (atmospheric carbon concentration increase, is equivalent to, 243 billion tons)
```

### Priority 3: Explicit Prompt Instructions (MEDIUM)

Add to Pass 1 prompt:

```
⚠️  ENTITY RESOLUTION RULES:
- ❌ NEVER use pronouns (He, She, We) as entities
- ✅ ALWAYS resolve to actual names (look back in text)
- ❌ NEVER use vague concepts ("the amount", "the process")
- ✅ ALWAYS include full context ("atmospheric carbon", "composting process")
- ❌ NEVER combine multiple targets with commas
- ✅ ALWAYS create separate relationships for each target
```

### Priority 4: Human Review Interface (LOW)

Create a review tool that:
- Flags problematic relationships automatically
- Allows human reviewer to fix pronouns/lists/context
- Learns patterns for future auto-fixing

---

## 📈 EXPECTED V5 IMPROVEMENTS

If we implement these fixes:

| Metric | V4 Current | V5 Target | Improvement |
|--------|-----------|-----------|-------------|
| Total relationships | 873 | 950+ | +9% (from list splitting) |
| Pronoun issues | 76 (8.7%) | <10 (1%) | -87% |
| List issues | 100 (11.5%) | <5 (0.5%) | -95% |
| Vague entities | 56 (6.4%) | <15 (1.5%) | -73% |
| **Overall quality issues** | 234 (26.8%) | <50 (5%) | **-78%** |
| High confidence | 812 (93%) | 900+ (95%) | +2% |

---

## 🎯 CONCLUSION

V4 successfully increased **quantity** (873 relationships, 63% coverage) but at the cost of **quality** (26.8% problematic relationships).

**The good news**: Most issues are systematic and fixable with:
1. Post-processing pipeline (pronoun resolution, list splitting)
2. Better prompt instructions
3. Improved few-shot examples

**V5 should focus on**:
- ✅ Maintaining V4's high recall (comprehensive extraction)
- ✅ Adding quality post-processing (entity resolution, list splitting, context enrichment)
- ✅ Achieving both high quantity AND high quality

**Grade**: V4 = B+ → V5 target = A++ (with quality fixes)

---

## 📋 APPENDIX: SAMPLE CORRECTIONS

### Example 1: Pronoun Resolution

**V4 Extract** (WRONG):
```json
{
  "source": "He",
  "relationship": "resides in",
  "target": "Colorado",
  "evidence": "He resides in Colorado."
}
```

**V5 Extract** (CORRECT):
```json
{
  "source": "Aaron William Perry",
  "relationship": "resides in",
  "target": "Colorado",
  "evidence": "He resides in Colorado.",
  "resolved_from": "He",
  "resolution_source": "page context: Aaron William Perry biography"
}
```

### Example 2: List Splitting

**V4 Extract** (WRONG):
```json
{
  "source": "biochar",
  "relationship": "is used for",
  "target": "houseplants, gardens, yards and neighborhood parks"
}
```

**V5 Extract** (CORRECT):
```json
[
  {
    "source": "biochar",
    "relationship": "is used for",
    "target": "houseplants"
  },
  {
    "source": "biochar",
    "relationship": "is used for",
    "target": "gardens"
  },
  {
    "source": "biochar",
    "relationship": "is used for",
    "target": "yards"
  },
  {
    "source": "biochar",
    "relationship": "is used for",
    "target": "neighborhood parks"
  }
]
```

### Example 3: Context Enrichment

**V4 Extract** (WRONG):
```json
{
  "source": "the amount of carbon",
  "relationship": "is equivalent to",
  "target": "243 billion tons",
  "evidence": "we've increased the atmospheric concentration of carbon by some 243 billion tons."
}
```

**V5 Extract** (CORRECT):
```json
{
  "source": "human-caused atmospheric carbon concentration increase",
  "relationship": "is equivalent to",
  "target": "243 billion tons",
  "evidence": "we've increased the atmospheric concentration of carbon by some 243 billion tons.",
  "context_enriched_from": "the amount of carbon"
}
```

---

**Generated**: 2025-10-12 06:16:04  
**Next Steps**: Review this report and decide whether to implement V5 improvements or use V4 as-is.
