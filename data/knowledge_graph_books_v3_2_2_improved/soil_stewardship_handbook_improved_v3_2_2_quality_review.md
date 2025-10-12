# Extraction Quality Review

**File**: `data/knowledge_graph_books_v3_2_2_improved/soil_stewardship_handbook_improved_v3_2_2.json`
**Total Relationships**: 493

## Issue Summary

- **missing_critical_context**: 3 cases
- **number_without_context**: 2 cases
- **lost_specificity**: 2 cases
- **semantically_odd**: 1 cases

## Issue Examples (up to 10 per type)

### number_without_context

**Description**: Percentages or numbers without clear context

#### Example 1

**Triple**: `Michael Bowman` → `is the Founding Director of` → `25x’25`

**Evidence** (page 2):
> Founding Director, “25x’25”

**Issue**: Target '25x’25' is a number/percentage without clear meaning on its own

---

#### Example 2

**Triple**: `soil` → `is increased by` → `10%`

**Evidence** (page 21):
> we’re only talking about an increase of soil carbon of about 10%.

**Issue**: Target '10%' is a number/percentage without clear meaning on its own

---

### lost_specificity

**Description**: Extraction lost important details from evidence text

#### Example 1

**Triple**: `soil` → `is crucial for` → `carbon sequestration`

**Evidence** (page 19):
> To sequester atmospheric carbon, increase soil carbon by: 10% increase of the carbon content in soil world-wide.

**Issue**: Extraction lost important context from evidence text

---

#### Example 2

**Triple**: `soil` → `is increased by` → `10%`

**Evidence** (page 21):
> we’re only talking about an increase of soil carbon of about 10%.

**Issue**: Extraction lost important context from evidence text

---

### semantically_odd

**Description**: Relationship doesn't make semantic sense

#### Example 1

**Triple**: `soil` → `is increased by` → `10%`

**Evidence** (page 21):
> we’re only talking about an increase of soil carbon of about 10%.

**Issue**: This triple doesn't make semantic sense as stated

---

### missing_critical_context

**Description**: Critical context from evidence missing in entities

#### Example 1

**Triple**: `Kitchen scraps` → `end up in` → `landfills`

**Evidence** (page 23):
> when we throw away kitchen scraps, paper and other organic, biodegradable 'waste' it ends up in landfills.

**Issue**: Important qualifiers from evidence are missing

---

#### Example 2

**Triple**: `biodynamic products` → `are better for` → `planet Earth`

**Evidence** (page 25):
> Choose organic, biodynamic, and soil-regenerating products—they’re better for you and for our planet Earth!

**Issue**: Important qualifiers from evidence are missing

---

#### Example 3

**Triple**: `communities` → `engage in` → `soil stewardship`

**Evidence** (page 28):
> active, vibrant Soil Stewardship Guilds.

**Issue**: Important qualifiers from evidence are missing

---
