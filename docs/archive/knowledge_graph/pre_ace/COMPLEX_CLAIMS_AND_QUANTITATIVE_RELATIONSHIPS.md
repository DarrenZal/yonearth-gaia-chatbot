# Complex Claims and Quantitative Relationships

## Problem: Current Extraction Loses Context

### Example: Soil Carbon Sequestration Goal

**Original Text** (Soil Handbook, page 21):
> "The amount of fossil carbon that we need to return to the ground is an amount equal to a 10% increase of the carbon content in soil world-wide."

**Current Extraction** (❌ WRONG):
```
soil → is increased by → 10%
```

**Problems**:
1. ❌ "soil" should be "soil carbon content"
2. ❌ "10%" is meaningless without context
3. ❌ Lost: This is about FOSSIL carbon sequestration goal
4. ❌ Lost: This is WORLDWIDE in scope
5. ❌ Lost: This is an EQUIVALENCE claim (goal = 10% increase)

---

## Solution 1: Multi-Triple Representation (RECOMMENDED)

Break complex claims into multiple connected triples:

```json
[
  {
    "source": "fossil carbon sequestration goal",
    "relationship": "equals",
    "target": "10% increase in global soil carbon",
    "source_type": "Goal",
    "target_type": "Measurement"
  },
  {
    "source": "soil carbon content",
    "relationship": "can increase by",
    "target": "10%",
    "source_type": "Measurement",
    "target_type": "Percentage",
    "scope": "worldwide"
  },
  {
    "source": "10% increase in global soil carbon",
    "relationship": "would sequester",
    "target": "all excess fossil carbon",
    "source_type": "Measurement",
    "target_type": "Amount"
  }
]
```

**Advantages**:
- ✅ Each triple is simple and semantically valid
- ✅ Complex claim is preserved through connections
- ✅ Can query: "What would a 10% soil carbon increase achieve?"
- ✅ Can query: "What is the fossil carbon sequestration goal?"

---

## Solution 2: Reified Relationship (Advanced)

Create a node representing the relationship itself:

```json
{
  "claim_id": "soil_carbon_sequestration_equivalence",
  "claim_type": "Equivalence",
  "subject": "fossil carbon sequestration goal",
  "predicate": "equals",
  "object": "10% increase in global soil carbon",
  "properties": {
    "scope": "worldwide",
    "measurement_type": "percentage",
    "measurement_value": "10%",
    "measured_quantity": "soil carbon content",
    "context": "climate change mitigation"
  },
  "evidence": {
    "doc_id": "Soil Stewardship Handbook",
    "page": 21,
    "quote": "The amount of fossil carbon that we need to return to the ground is an amount equal to a 10% increase of the carbon content in soil world-wide."
  }
}
```

**Advantages**:
- ✅ Captures ALL context in one structure
- ✅ Supports complex queries
- ✅ Preserves quantitative details

**Disadvantages**:
- ⚠️ More complex to extract
- ⚠️ Harder to visualize in graph form

---

## Solution 3: N-ary Relationship Pattern

Use RDF-star / property graphs approach:

```turtle
# Main triple
:soil_carbon :can_increase_by "10%" .

# Metadata about the triple
<< :soil_carbon :can_increase_by "10%" >>
  :scope "worldwide" ;
  :equals :fossil_carbon_sequestration_goal ;
  :context "climate mitigation" ;
  :source_page 21 ;
  :confidence 0.95 .
```

---

## How to Improve Extraction Process

### 1. Better Entity Extraction

**Current**: Single-word entities lose context
```
"soil carbon content" → extracted as "soil" ❌
```

**Improved**: Multi-word concept extraction
```python
# Extraction prompt improvement
"Extract entities as complete concepts, including important modifiers:
- ✅ 'soil carbon content' NOT 'soil'
- ✅ 'organic matter' NOT 'matter'
- ✅ 'fossil fuel emissions' NOT 'emissions'
- ✅ 'global temperature' NOT 'temperature'

Keep adjectives that change meaning:
- 'active carbon' vs 'total carbon' vs 'stable carbon'
- 'short-term goal' vs 'long-term goal'
"
```

### 2. Detect Quantitative Claim Patterns

**Pattern Recognition**:
```python
QUANTITATIVE_PATTERNS = [
    # Equivalence patterns
    r"(amount|quantity) equal to (.+) of (.+)",
    r"(.+) equals (.+)",
    r"same as (.+)",

    # Measurement patterns
    r"(\d+%?) increase (?:in|of) (.+)",
    r"(.+) would (?:increase|decrease) by (\d+%?)",

    # Goal/requirement patterns
    r"need to (.+) by (\d+%?)",
    r"requires (.+) of (.+)",
]
```

**When detected, extract as multi-triple**:
```python
if is_quantitative_claim(text):
    # Extract components
    measurement = extract_measurement(text)  # "10% increase"
    measured_thing = extract_measured_entity(text)  # "soil carbon content"
    equivalence = extract_equivalence(text)  # "fossil carbon sequestration goal"
    scope = extract_scope(text)  # "worldwide"

    # Create multiple triples
    return [
        (measured_thing, "can_change_by", measurement),
        (measurement, "equals", equivalence),
        (measurement, "has_scope", scope)
    ]
```

### 3. Enhanced Prompt for Complex Claims

**Add to extraction prompt**:
```
# Special Instructions for Quantitative Claims

When extracting claims with numbers, percentages, or measurements:

1. **Extract the FULL entity** including the thing being measured
   - ✅ "soil carbon content" NOT "soil"
   - ✅ "atmospheric CO2 concentration" NOT "CO2"

2. **Create MULTIPLE triples** for complex equivalences
   - Example: "Goal X equals a 10% increase in Y"
   - Triple 1: (Goal X, equals, 10% increase in Y)
   - Triple 2: (Y, can increase by, 10%)

3. **Capture SCOPE as property** (worldwide, regional, local, etc.)

4. **Preserve EQUIVALENCE relationships**
   - When text says "amount equal to" or "same as" or "equivalent to"
   - Extract as: (Thing A, equals, Thing B)

5. **Context matters**
   - If the claim is conditional ("if we X, then Y will Z")
   - Extract the precondition: (X, enables, Y changes by Z)
```

### 4. Post-Processing Validation

**Validate extracted triples**:
```python
def validate_quantitative_triple(triple):
    """Check if quantitative triple makes sense"""
    source, rel, target = triple

    # Check 1: If target is a percentage/number, source should be measurable
    if is_number_or_percentage(target):
        if not is_measurable_quantity(source):
            return ValidationError(
                f"Target '{target}' is a measurement but source '{source}' "
                f"is not a measurable quantity. "
                f"Suggestion: Add measured property to source "
                f"(e.g., 'soil' → 'soil carbon content')"
            )

    # Check 2: Increase/decrease relationships need measurable quantities
    if 'increase' in rel or 'decrease' in rel:
        if not has_measurable_property(source):
            return ValidationError(
                f"Relationship '{rel}' requires measurable quantity, "
                f"but '{source}' lacks measurement context"
            )

    return ValidationSuccess()
```

---

## Recommended Approach

**For the Soil Handbook claim**, extract as:

### Option A: Three Simple Triples (RECOMMENDED)
```json
[
  {
    "source": "soil carbon content",
    "relationship": "can increase by",
    "target": "10%",
    "scope": "worldwide",
    "context": "climate mitigation"
  },
  {
    "source": "10% global soil carbon increase",
    "relationship": "equals",
    "target": "fossil carbon sequestration goal"
  },
  {
    "source": "fossil carbon sequestration",
    "relationship": "requires",
    "target": "10% global soil carbon increase"
  }
]
```

### Option B: One Rich Triple with Properties
```json
{
  "source": "fossil carbon sequestration goal",
  "relationship": "equals",
  "target": "10% increase in soil carbon content",
  "source_type": "Goal",
  "target_type": "Measurement",
  "properties": {
    "measurement_value": "10%",
    "measurement_type": "percentage_increase",
    "measured_quantity": "soil carbon content",
    "scope": "worldwide",
    "domain": "climate change mitigation"
  },
  "evidence": {...}
}
```

---

## Implementation Plan

1. **Phase 1**: Update extraction prompts with quantitative claim guidance ✅ CAN DO NOW
2. **Phase 2**: Add pattern detection for measurements, equivalences, scope ✅ CAN DO NOW
3. **Phase 3**: Add post-processing validation to flag nonsensical triples ✅ CAN DO NOW
4. **Phase 4**: Implement multi-triple extraction for complex claims (FUTURE)
5. **Phase 5**: Add reification support for very complex claims (FUTURE)

---

## Examples from Real Text

### Example 1: Current vs Improved

**Text**: "we're only talking about an increase of soil carbon of about 10%"

**Current** ❌:
```
soil → is increased by → 10%
```

**Improved** ✅:
```
soil carbon content → can increase by → 10%
```

### Example 2: Equivalence Claim

**Text**: "The amount of fossil carbon that we need to return to the ground is an amount equal to a 10% increase of the carbon content in soil world-wide."

**Current** ❌:
```
soil → is increased by → 10%
```

**Improved** ✅:
```
Triple 1: fossil carbon sequestration goal → equals → 10% global soil carbon increase
Triple 2: soil carbon content → can increase by → 10% (scope: worldwide)
Triple 3: 10% global soil carbon increase → would sequester → excess fossil carbon
```

### Example 3: Conditional Claim

**Text**: "If soil carbon increases by 10% worldwide, it would sequester all excess fossil carbon."

**Current** ❌:
```
soil → is increased by → 10%
```

**Improved** ✅:
```
Triple 1: 10% increase in global soil carbon → would sequester → all excess fossil carbon
Triple 2: soil carbon content → can increase by → 10% (scope: worldwide)
Triple 3: fossil carbon sequestration → requires → 10% global soil carbon increase
```

---

## Conclusion

**Key Takeaways**:

1. ✅ Extract **complete concepts** ("soil carbon content" not "soil")
2. ✅ Break **complex claims** into multiple connected triples
3. ✅ Capture **scope/context** as properties (worldwide, conditional, etc.)
4. ✅ Detect **equivalence** and **quantitative** patterns
5. ✅ Validate that **measurements** connect to **measurable quantities**

**Next Steps**:
- Update extraction prompts (Phase 1)
- Add pattern detection (Phase 2)
- Add validation rules (Phase 3)
- Re-run extraction on Soil Handbook
