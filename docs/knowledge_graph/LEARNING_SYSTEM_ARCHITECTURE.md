# 🧠 Multi-Level Learning System Architecture (Ultrathought Design)

## The Core Insight: 4 Different Types of "Learning"

After analyzing the biochar example, it's clear we need to separate fundamentally different types of errors:

### Error Type 1: **Schema/Type Violations** (Universal Logic)
**Example**: `International Biochar Initiative --[located_in]--> biochar`

**Why it's wrong**:
- Biochar is type `soil conditioner`/`charcoal` (Wikidata: Q905495)
- Geographic relationships require target of type `Place`/`Location`
- This is a **structural** violation, not factual

**What to learn**:
```python
# NOT this (too specific):
if target == "biochar" and relationship == "located_in":
    delete()

# YES this (universal rule):
SHACL_CONSTRAINT = """
:GeographicRelationship a sh:NodeShape ;
    sh:targetSubjectsOf :located_in, :part_of, :contains ;
    sh:property [
        sh:path :located_in ;
        sh:class :GeographicLocation ;  # Target MUST be a Place type
        sh:message "Geographic relationships require target of type Place" ;
    ] .
"""
```

**Generalization**: Yes! Once learned, applies to ALL geographic relationships
**Source**: Wikidata types, local ontology, SHACL constraints
**Computable**: Yes, by checking entity types

---

### Error Type 2: **Logical Rules** (Computable from Properties)
**Example**: `Boulder --[located_in]--> Lafayette`

**Why it's wrong**:
- Boulder population: 108,000
- Lafayette population: 30,000
- Rule: Smaller places contain larger places? NO!

**What to learn**:
```python
# Universal logical rule:
def validate_geographic_containment(parent, child):
    """Parent must be larger than child"""
    if get_population(child) > get_population(parent) * 1.2:
        return REVERSE_RELATIONSHIP

    if get_area(child) > get_area(parent):
        return REVERSE_RELATIONSHIP

    # Check administrative hierarchy
    if not is_administrative_parent(parent, child):
        return FLAG_FOR_REVIEW
```

**Generalization**: Yes! Applies to all geographic containment
**Source**: External data (GeoNames, population databases)
**Computable**: Yes, if properties are available

---

### Error Type 3: **Instance-Level Corrections** (No Generalization Possible)
**Example**: `John Doe --[lives_in]--> Florida` (actually lives in California)

**Why it's wrong**:
- Factually incorrect
- But structurally valid (Person → lives_in → Place)
- No type violation, no logical rule

**What to "learn"**:
```python
# Can't generalize! Just track:
corrections_log.append({
    'original': ('John Doe', 'lives_in', 'Florida'),
    'corrected': ('John Doe', 'lives_in', 'California'),
    'reasoning': 'Factual error',
    'cannot_generalize': True
})

# This correction teaches us NOTHING about other relationships
# It's just a database update
```

**Generalization**: NO! One-off correction
**Source**: Human knowledge, external verification
**Computable**: No, requires external fact-checking

---

### Error Type 4: **Extraction Quality Patterns** (About the LLM, not the knowledge)
**Example**: LLM often assigns low confidence when uncertain about geographic direction

**Why it matters**:
- Relationships with `relationship_confidence < 0.70` are wrong 40% of the time
- Geographic relationships with `relationship_confidence < 0.75` are wrong 60% of the time
- LLM flags uncertainty correctly!

**What to learn**:
```python
# Learn about the EXTRACTION PROCESS:
extraction_quality_patterns = {
    'geographic_low_confidence': {
        'pattern': 'relationship_type in [located_in, part_of] AND confidence < 0.75',
        'error_rate': 0.60,
        'action': 'FLAG_FOR_VALIDATION'
    },
    'unknown_entities': {
        'pattern': 'target contains "unknown"',
        'error_rate': 0.95,
        'action': 'DELETE'
    }
}
```

**Generalization**: Yes! About the LLM's behavior
**Source**: Analyzing corrections vs. original confidence scores
**Computable**: Yes, by statistical analysis

---

## The Proper Architecture

### Component 1: Type-Based Constraint System (SHACL)
```python
class TypeConstraintLearner:
    """Learns universal type constraints from corrections"""

    def __init__(self):
        self.wikidata_cache = {}
        self.shacl_constraints = []

    def analyze_correction(self, correction):
        """Check if this is a type violation"""
        source_types = self.get_wikidata_types(correction['source'])
        target_types = self.get_wikidata_types(correction['target'])
        relationship = correction['relationship']

        # Check for type mismatch
        if relationship in ['located_in', 'part_of', 'contains']:
            # Geographic relationships
            if not self.is_geographic_type(target_types):
                # LEARN: This relationship requires geographic target
                constraint = self.create_shacl_constraint(
                    relationship=relationship,
                    target_type_required='GeographicLocation',
                    violation_example=correction
                )
                return constraint

        return None

    def get_wikidata_types(self, entity_name):
        """Fetch entity types from Wikidata"""
        # Query Wikidata API
        # Cache results locally
        # Return list of types
        pass

    def is_geographic_type(self, types):
        """Check if entity types include geographic/location types"""
        geographic_types = [
            'geographic location', 'place', 'city', 'country',
            'administrative territorial entity', 'settlement'
        ]
        return any(t in geographic_types for t in types)
```

### Component 2: Logical Rule System
```python
class LogicalRuleEngine:
    """Computable rules based on entity properties"""

    def __init__(self):
        self.rules = [
            PopulationHierarchyRule(),
            AreaHierarchyRule(),
            AdministrativeHierarchyRule(),
            CyclePreventionRule(),
        ]

    def validate(self, relationship):
        """Apply all logical rules"""
        violations = []
        for rule in self.rules:
            if rule.applies_to(relationship):
                result = rule.check(relationship)
                if not result.valid:
                    violations.append(result)
        return violations

class PopulationHierarchyRule:
    """Container must have larger population than contained"""

    def check(self, rel):
        if rel['relationship'] not in ['located_in', 'part_of']:
            return Valid()

        source_pop = get_population(rel['source'])
        target_pop = get_population(rel['target'])

        if source_pop and target_pop:
            if source_pop > target_pop * 1.2:  # 20% tolerance
                return Invalid(
                    reason=f"{rel['source']} (pop: {source_pop}) cannot be in {rel['target']} (pop: {target_pop})",
                    suggested_fix="REVERSE_RELATIONSHIP",
                    confidence=0.95
                )

        return Valid()
```

### Component 3: Correction Log (No Learning)
```python
class InstanceCorrectionLog:
    """Just tracks corrections that can't be generalized"""

    def record(self, correction):
        """Store correction without trying to learn from it"""
        self.log.append({
            'timestamp': now(),
            'original': correction['original'],
            'corrected': correction['corrected'],
            'reasoning': correction['reasoning'],
            'generalizable': False,  # Explicitly mark
            'correction_type': 'instance_level_fact'
        })

        # Update the knowledge graph directly
        self.apply_correction_to_kg(correction)

        # NO pattern extraction
        # NO rule generation
        # Just a database update
```

### Component 4: Extraction Quality Analyzer
```python
class ExtractionQualityLearner:
    """Learns patterns about the LLM's extraction behavior"""

    def analyze_corrections(self, corrections):
        """Find patterns in what the LLM gets wrong"""

        patterns = {}

        # Group by confidence ranges
        for correction in corrections:
            conf = correction['original']['relationship_confidence']
            conf_bucket = int(conf * 10) / 10  # Round to 0.1

            if conf_bucket not in patterns:
                patterns[conf_bucket] = {
                    'total': 0,
                    'errors': 0,
                    'relationship_types': defaultdict(int)
                }

            patterns[conf_bucket]['total'] += 1
            patterns[conf_bucket]['errors'] += 1
            patterns[conf_bucket]['relationship_types'][correction['original']['relationship']] += 1

        # Calculate error rates
        for conf, data in patterns.items():
            data['error_rate'] = data['errors'] / data['total']

            # If error rate > 50% at this confidence, flag it
            if data['error_rate'] > 0.5:
                self.create_validation_rule(
                    f"Flag all relationships with confidence < {conf + 0.1} for review",
                    expected_error_rate=data['error_rate']
                )
```

## The Integrated Workflow

```python
class SmartLearningSystem:
    def __init__(self):
        self.type_constraints = TypeConstraintLearner()
        self.logical_rules = LogicalRuleEngine()
        self.instance_log = InstanceCorrectionLog()
        self.extraction_analyzer = ExtractionQualityLearner()

    def learn_from_correction(self, correction):
        """Route correction to appropriate learning component"""

        # Step 1: Is this a type violation?
        type_constraint = self.type_constraints.analyze_correction(correction)
        if type_constraint:
            self.type_constraints.add_constraint(type_constraint)
            return {
                'learned': True,
                'type': 'SCHEMA_CONSTRAINT',
                'generalizable': True,
                'constraint': type_constraint
            }

        # Step 2: Is this a logical rule violation?
        logical_violation = self.logical_rules.check_if_rule_learnable(correction)
        if logical_violation:
            self.logical_rules.add_rule(logical_violation)
            return {
                'learned': True,
                'type': 'LOGICAL_RULE',
                'generalizable': True,
                'rule': logical_violation
            }

        # Step 3: Track extraction quality
        self.extraction_analyzer.add_correction(correction)

        # Step 4: Otherwise, just a factual correction
        self.instance_log.record(correction)
        return {
            'learned': False,
            'type': 'INSTANCE_CORRECTION',
            'generalizable': False,
            'message': 'Factual correction recorded, no generalizable pattern'
        }

    def validate_relationship(self, rel):
        """Apply all learned knowledge"""

        # Check type constraints (SHACL)
        type_violations = self.type_constraints.validate(rel)
        if type_violations:
            return type_violations

        # Check logical rules
        logical_violations = self.logical_rules.validate(rel)
        if logical_violations:
            return logical_violations

        # Check extraction quality patterns
        quality_flags = self.extraction_analyzer.should_flag(rel)
        if quality_flags:
            return quality_flags

        return Valid()
```

## Examples Applied

### Example 1: Biochar Location Error
```python
correction = {
    'original': {
        'source': 'International Biochar Initiative',
        'relationship': 'located_in',
        'target': 'biochar',
        'relationship_confidence': 0.80
    },
    'action': 'DELETE',
    'reasoning': 'biochar is a product, not a place'
}

result = system.learn_from_correction(correction)
# Returns: {
#     'learned': True,
#     'type': 'SCHEMA_CONSTRAINT',
#     'generalizable': True,
#     'constraint': 'located_in requires target of type GeographicLocation'
# }

# Future extractions automatically catch:
# - "X located_in charcoal" ❌
# - "X located_in soil" ❌
# - "X located_in compost" ❌
# All caught by type checking!
```

### Example 2: Boulder/Lafayette
```python
correction = {
    'original': {
        'source': 'Boulder',
        'relationship': 'located_in',
        'target': 'Lafayette',
        'relationship_confidence': 0.60
    },
    'action': 'REVERSE',
    'reasoning': 'Boulder (pop 108k) cannot be in Lafayette (pop 30k)'
}

result = system.learn_from_correction(correction)
# Returns: {
#     'learned': True,
#     'type': 'LOGICAL_RULE',
#     'generalizable': True,
#     'rule': 'Container must have larger population than contained (with 20% tolerance)'
# }

# Future extractions automatically catch:
# - Any city-in-city with wrong population order
# - Computes from GeoNames data
```

### Example 3: John Doe Lives in Florida
```python
correction = {
    'original': {
        'source': 'John Doe',
        'relationship': 'lives_in',
        'target': 'Florida',
        'relationship_confidence': 0.85
    },
    'corrected': {
        'source': 'John Doe',
        'relationship': 'lives_in',
        'target': 'California',
        'relationship_confidence': 0.95
    },
    'action': 'MODIFY',
    'reasoning': 'Factual error'
}

result = system.learn_from_correction(correction)
# Returns: {
#     'learned': False,
#     'type': 'INSTANCE_CORRECTION',
#     'generalizable': False,
#     'message': 'Factual correction recorded, no generalizable pattern'
# }

# No learning - just updates the database
# Can't generalize "John Doe" errors to other people
```

## Why This Architecture Works

### 1. Separates Concerns
- **Type violations** → SHACL constraints
- **Logical violations** → Computable rules
- **Factual errors** → Just corrections
- **LLM patterns** → Quality filters

### 2. Maximizes Generalization
- Type constraints apply to ALL entities of that type
- Logical rules apply to ALL relationships with those properties
- Only factual errors are one-offs

### 3. Leverages External Knowledge
- Wikidata for types
- GeoNames for geographic properties
- Research papers for constraint patterns

### 4. Aligns with Research
- **SHACL validation** (KG_Research_3.md)
- **Neural-symbolic integration** (KG_Research_2.md)
- **Active learning** but on the RIGHT things

## Implementation Priority

### Phase 1: Type Constraints (Highest ROI)
1. Build Wikidata type checker
2. Create SHACL shapes for common violations
3. Apply to all 15,201 relationships
4. **Expected: Catch 49 biochar-type errors automatically**

### Phase 2: Logical Rules (Medium ROI)
1. Implement population/area checks
2. Add GeoNames integration
3. Cycle detection
4. **Expected: Catch Boulder/Lafayette errors automatically**

### Phase 3: Quality Patterns (Low ROI but easy)
1. Analyze confidence vs. error correlation
2. Create flagging rules
3. **Expected: Reduce human review by 30%**

### Phase 4: Track Instance Corrections (No learning, just logging)
1. Simple correction log
2. Database updates
3. **Expected: Handle one-off factual errors**

## The Key Insight

**Don't try to learn patterns from everything!**

Only 2 types of errors are generalizable:
1. **Type violations** (biochar is not a place)
2. **Logical rules** (smaller doesn't contain larger)

The other 2 are NOT generalizable:
3. **Factual errors** (John lives in CA not FL)
4. **Extraction patterns** (low confidence often wrong)

Focus learning effort on #1 and #2. Just track #3. Use statistics for #4.

This is what the research meant by "schema-aware iterative refinement"!