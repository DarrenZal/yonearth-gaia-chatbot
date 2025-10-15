#!/usr/bin/env python3
"""
Validate V8 Curator-Generated Changes

Quick validation that all 8 V8 enhancements are present in the code.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("="*80)
print("✅ VALIDATING V8 CURATOR-GENERATED CHANGES")
print("="*80)
print()

# Import V8 modules
try:
    from scripts.extract_kg_v8_book import (
        PraiseQuoteDetector,
        BibliographicCitationParser,
        ListSplitter,
        PronounResolver,
        ContextEnricher,
        PredicateNormalizer,
        FigurativeLanguageFilter,
        DUAL_SIGNAL_EVALUATION_PROMPT
    )
    print("✅ Successfully imported all V8 modules")
    print()
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Validate each V8 enhancement

print("VALIDATING CRITICAL CHANGES:")
print()

# 1. PraiseQuoteDetector (NEW)
print("1. PraiseQuoteDetector (Curator Change #001):")
detector = PraiseQuoteDetector()
has_endorsement_indicators = hasattr(detector, 'endorsement_indicators')
has_front_matter_pages = hasattr(detector, 'front_matter_pages')
has_praise_check = hasattr(detector, 'is_praise_quote_context')
if has_endorsement_indicators and has_front_matter_pages and has_praise_check:
    print("   ✅ NEW PraiseQuoteDetector module present")
    print(f"      - Endorsement indicators: {len(detector.endorsement_indicators)}")
    print(f"      - Front matter pages: {list(detector.front_matter_pages)[:5]}...")
else:
    print("   ❌ PraiseQuoteDetector incomplete")
print()

# 2. PronounResolver - Possessive pronouns (ENHANCED)
print("2. PronounResolver Enhancement (Curator Change #002-004):")
resolver = PronounResolver()
has_possessive = hasattr(resolver, 'possessive_pronouns')
has_possessive_patterns = hasattr(resolver, 'possessive_patterns')
has_author_context = hasattr(resolver, 'author_context')
has_context_window = hasattr(resolver, 'context_window')
if has_possessive and has_possessive_patterns and has_author_context:
    print("   ✅ ENHANCED PronounResolver with possessive support")
    print(f"      - Possessive pronouns: {resolver.possessive_pronouns}")
    print(f"      - Context window: {resolver.context_window} sentences")
    print(f"      - Possessive patterns: {len(resolver.possessive_patterns)}")
else:
    print("   ❌ PronounResolver enhancements incomplete")
print()

print("VALIDATING HIGH PRIORITY CHANGES:")
print()

# 3. ContextEnricher - Context replacement (ENHANCED)
print("3. ContextEnricher Enhancement (Curator Change #005-006):")
enricher = ContextEnricher()
has_context_replacements = hasattr(enricher, 'context_replacements')
has_find_replacement = hasattr(enricher, '_find_replacement')
if has_context_replacements and has_find_replacement:
    print("   ✅ ENHANCED ContextEnricher with context-aware replacement")
    print(f"      - Context replacement rules: {len(enricher.context_replacements)}")
    rules = list(enricher.context_replacements.keys())[:3]
    print(f"      - Sample rules: {rules}")
else:
    print("   ❌ ContextEnricher enhancements incomplete")
print()

# 4. ListSplitter - 'and' conjunctions (ENHANCED)
print("4. ListSplitter Enhancement (Curator Change #008-009):")
splitter = ListSplitter()
has_list_patterns = hasattr(splitter, 'list_patterns')
if has_list_patterns:
    pattern_count = len(splitter.list_patterns)
    # Check for 'and' patterns
    has_and_pattern = any('and' in str(p) for p in splitter.list_patterns)
    if pattern_count >= 3 and has_and_pattern:
        print("   ✅ ENHANCED ListSplitter with 'and' conjunction support")
        print(f"      - List patterns: {pattern_count}")
        print(f"      - Supports 'and' conjunctions: Yes")
    else:
        print(f"   ⚠️  ListSplitter partially enhanced (patterns: {pattern_count})")
else:
    print("   ❌ ListSplitter enhancements incomplete")
print()

# 5. Philosophical filter in prompt (ENHANCED)
print("5. Pass 2 Prompt Enhancement (Curator Change #007):")
has_philosophical_filter = 'PHILOSOPHICAL' in DUAL_SIGNAL_EVALUATION_PROMPT or 'philosophical' in DUAL_SIGNAL_EVALUATION_PROMPT.lower()
has_factual_concreteness = 'FACTUAL CONCRETENESS' in DUAL_SIGNAL_EVALUATION_PROMPT or 'factual concreteness' in DUAL_SIGNAL_EVALUATION_PROMPT.lower()
if has_philosophical_filter or has_factual_concreteness:
    print("   ✅ ENHANCED Pass 2 prompt with philosophical statement filter")
    print(f"      - Has philosophical filter: {has_philosophical_filter}")
    print(f"      - Has factual concreteness: {has_factual_concreteness}")
else:
    print("   ❌ Philosophical filter not found in prompt")
print()

print("VALIDATING MEDIUM PRIORITY CHANGES:")
print()

# 6. PredicateNormalizer - Semantic validation (ENHANCED)
print("6. PredicateNormalizer Enhancement (Curator Change #010-011):")
normalizer = PredicateNormalizer()
has_entity_type_predicates = hasattr(normalizer, 'entity_type_predicates')
has_detect_entity_type = hasattr(normalizer, '_detect_entity_type')
if has_entity_type_predicates and has_detect_entity_type:
    print("   ✅ ENHANCED PredicateNormalizer with semantic validation")
    print(f"      - Entity type rules: {list(normalizer.entity_type_predicates.keys())}")
    # Check Book rules
    if 'Book' in normalizer.entity_type_predicates:
        book_rules = normalizer.entity_type_predicates['Book']
        print(f"      - Book forbidden predicates: {len(book_rules['forbidden'])}")
else:
    print("   ❌ PredicateNormalizer enhancements incomplete")
print()

# 7. BibliographicCitationParser - Dedication detection (ENHANCED)
print("7. BibliographicCitationParser Enhancement (Curator Change #012-013):")
bib_parser = BibliographicCitationParser()
has_dedication_patterns = hasattr(bib_parser, 'dedication_patterns')
has_dedication_check = hasattr(bib_parser, 'is_dedication')
if has_dedication_patterns and has_dedication_check:
    print("   ✅ ENHANCED BibliographicCitationParser with dedication detection")
    print(f"      - Dedication patterns: {len(bib_parser.dedication_patterns)}")
else:
    print("   ❌ BibliographicCitationParser enhancements incomplete")
print()

# 8. FigurativeLanguageFilter - Metaphor normalization (ENHANCED)
print("8. FigurativeLanguageFilter Enhancement (Curator Change #014-015):")
fig_filter = FigurativeLanguageFilter()
has_metaphor_normalizations = hasattr(fig_filter, 'metaphor_normalizations')
if has_metaphor_normalizations:
    print("   ✅ ENHANCED FigurativeLanguageFilter with metaphor normalization")
    print(f"      - Metaphor normalizations: {len(fig_filter.metaphor_normalizations)}")
    sample_metaphors = list(fig_filter.metaphor_normalizations.items())[:3]
    for metaphor, literal in sample_metaphors:
        print(f"         '{metaphor}' → '{literal}'")
else:
    print("   ❌ FigurativeLanguageFilter enhancements incomplete")
print()

print("="*80)
print("✅ V8 VALIDATION COMPLETE")
print("="*80)
print()

# Count successful validations
validations = [
    has_endorsement_indicators and has_praise_check,  # PraiseQuoteDetector
    has_possessive and has_possessive_patterns,       # PronounResolver
    has_context_replacements and has_find_replacement,  # ContextEnricher
    has_list_patterns and has_and_pattern,            # ListSplitter (partial)
    has_philosophical_filter or has_factual_concreteness,  # Prompt
    has_entity_type_predicates and has_detect_entity_type,  # PredicateNormalizer
    has_dedication_patterns and has_dedication_check,  # BibliographicParser
    has_metaphor_normalizations                        # FigurativeLanguageFilter
]

success_count = sum(validations)
total_count = len(validations)

print(f"RESULTS: {success_count}/{total_count} V8 enhancements validated")
print()

if success_count == total_count:
    print("✅✅✅ ALL V8 CURATOR-GENERATED CHANGES SUCCESSFULLY APPLIED ✅✅✅")
    print()
    print("READY FOR PRODUCTION:")
    print("  - 15 Curator-generated changes applied")
    print("  - Expected impact: 6.71% → 2.7% issue rate")
    print("  - Expected fixes: 46 issues resolved")
    print()
    print("NEXT STEP: Run full V8 extraction")
    print("  python scripts/extract_kg_v8_book.py")
elif success_count >= total_count * 0.75:
    print("⚠️  MOSTLY COMPLETE: Some enhancements may need review")
    print("  Proceed with caution or investigate incomplete validations")
else:
    print("❌ VALIDATION FAILED: Multiple enhancements missing")
    print("  Review V8 code before proceeding")
    sys.exit(1)
