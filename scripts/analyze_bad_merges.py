#!/usr/bin/env python3
"""
Bad Merge Analyzer for Knowledge Graph Entity Merges

Phase 6, Work Chunk 6.2: Mine Bad Merges from History

This script analyzes entity_merges.json to identify suspicious merges
that may have resulted in incorrect entity resolution.

Heuristics used:
1. Type mismatch indicators: Words suggesting different semantic domains
2. Levenshtein-only merges: High string similarity but different meanings
3. Cross-domain merges: Entities from obviously different knowledge domains
4. Geographic confusion: Different locations merged together
5. Person vs generic term: Proper names merged with common words

Output: data/analysis/bad_merge_candidates.json
"""

import json
import os
import re
from datetime import datetime
from typing import Dict, List, Tuple, Set
from collections import defaultdict

# Try to import Levenshtein for edit distance calculation
try:
    from Levenshtein import distance as levenshtein_distance, ratio as levenshtein_ratio
    HAS_LEVENSHTEIN = True
except ImportError:
    try:
        from fuzzywuzzy import fuzz
        def levenshtein_ratio(s1: str, s2: str) -> float:
            return fuzz.ratio(s1, s2) / 100.0
        def levenshtein_distance(s1: str, s2: str) -> int:
            # Approximate distance from ratio
            return int((1 - levenshtein_ratio(s1, s2)) * max(len(s1), len(s2)))
        HAS_LEVENSHTEIN = True
    except ImportError:
        HAS_LEVENSHTEIN = False
        print("Warning: Neither python-Levenshtein nor fuzzywuzzy installed. Using basic ratio.")
        def levenshtein_ratio(s1: str, s2: str) -> float:
            """Simple ratio based on common characters."""
            if not s1 or not s2:
                return 0.0
            common = set(s1.lower()) & set(s2.lower())
            return len(common) / max(len(set(s1.lower())), len(set(s2.lower())))


# Domain indicators for detecting cross-domain merges
DOMAIN_INDICATORS = {
    'geography': {'california', 'europe', 'africa', 'asia', 'america', 'state', 'country',
                  'city', 'town', 'region', 'north', 'south', 'east', 'west', 'montana',
                  'colorado', 'washington', 'oregon', 'texas', 'florida', 'york'},
    'food': {'food', 'cooking', 'eating', 'meal', 'diet', 'nutrition', 'kale', 'corn',
             'berry', 'fruit', 'vegetable', 'cacao', 'chocolate', 'salt'},
    'nature': {'soil', 'water', 'ocean', 'river', 'mountain', 'forest', 'tree', 'plant',
               'flower', 'garden', 'earth', 'sun', 'moon', 'star'},
    'people': {'person', 'people', 'man', 'woman', 'child', 'mother', 'father', 'family',
               'friend', 'teacher', 'doctor', 'farmer', 'scientist'},
    'organization': {'company', 'corporation', 'organization', 'foundation', 'institute',
                     'university', 'school', 'government', 'agency'},
    'time': {'year', 'month', 'day', 'generation', 'era', 'period', 'future', 'past'},
    'emotion': {'mood', 'feeling', 'emotion', 'love', 'fear', 'joy', 'anger', 'peace'},
    'activity': {'tour', 'work', 'study', 'yoga', 'meditation', 'coaching', 'parenting',
                 'gardening', 'farming', 'cooking'},
    'abstract': {'concept', 'idea', 'theory', 'philosophy', 'spirit', 'soul', 'consciousness'},
}

# Known proper names that shouldn't merge with generic terms
PROPER_NAME_INDICATORS = {
    'aaron', 'john', 'jane', 'david', 'michael', 'sarah', 'mary', 'robert',
    'james', 'william', 'richard', 'thomas', 'charles', 'christopher',
    'wendell', 'stephen', 'steven', 'grace', 'berry', 'price', 'bronner',
}

# Known bad merge patterns (regex)
BAD_MERGE_PATTERNS = [
    # Acronyms merged with unrelated words
    (r'^(dia|drc|usa|uk|un)$', r'^[a-z]{4,}$'),
    # Single word merged with company/org name
    (r'^[a-z]{3,6}$', r'^[a-z]+ (inc|corp|llc|foundation|institute)$'),
    # Geographic direction conflicts
    (r'^(north|south|east|west)ern? \w+$', r'^(north|south|east|west)ern? \w+$'),
]


def get_domain(text: str) -> Set[str]:
    """Identify which domains a text belongs to."""
    text_lower = text.lower()
    words = set(text_lower.split())

    domains = set()
    for domain, indicators in DOMAIN_INDICATORS.items():
        if words & indicators:
            domains.add(domain)
        # Also check if any indicator is a substring
        for indicator in indicators:
            if indicator in text_lower:
                domains.add(domain)
                break

    return domains


def is_proper_name(text: str) -> bool:
    """Check if text appears to be a proper name."""
    text_lower = text.lower()
    words = text_lower.split()

    # Check for proper name indicators
    for word in words:
        if word in PROPER_NAME_INDICATORS:
            return True

    # Check for title patterns (Dr., Mr., etc.)
    if re.match(r'^(dr\.?|mr\.?|mrs\.?|ms\.?)\s', text_lower):
        return True

    # Check for capitalized multi-word names in original
    if len(words) >= 2 and all(w[0].isupper() if w else False for w in text.split()):
        return True

    return False


def is_generic_term(text: str) -> bool:
    """Check if text is a generic term (not a proper name)."""
    text_lower = text.lower()

    # Single lowercase word is likely generic
    if len(text.split()) == 1 and text == text_lower:
        return True

    # Check against domain indicators
    for indicators in DOMAIN_INDICATORS.values():
        if text_lower in indicators:
            return True

    return False


def analyze_merge(source: str, target: str) -> Dict:
    """
    Analyze a single merge and return plausibility assessment.

    Returns dict with:
        - plausibility_score: 0-1 (lower = more suspicious)
        - reasons: list of reasons for suspicion
        - recommendation: 'blocklist', 'review', or 'ok'
    """
    source_lower = source.lower().strip()
    target_lower = target.lower().strip()

    reasons = []
    score = 1.0  # Start at maximum plausibility

    # Heuristic 1: Levenshtein analysis
    lev_ratio = levenshtein_ratio(source_lower, target_lower)

    # High similarity but different meanings (suspicious)
    if 0.5 <= lev_ratio <= 0.8:
        # Check if domains are different
        source_domains = get_domain(source_lower)
        target_domains = get_domain(target_lower)

        if source_domains and target_domains and not (source_domains & target_domains):
            score -= 0.4
            reasons.append(f"cross_domain_merge: {source_domains} vs {target_domains}")

    # Very different strings but merged anyway
    if lev_ratio < 0.4:
        score -= 0.3
        reasons.append(f"low_similarity: {lev_ratio:.2f}")

    # Heuristic 2: Cross-domain merges
    source_domains = get_domain(source_lower)
    target_domains = get_domain(target_lower)

    if source_domains and target_domains:
        overlap = source_domains & target_domains
        if not overlap:
            score -= 0.3
            if 'cross_domain_merge' not in ' '.join(reasons):
                reasons.append(f"domain_mismatch: {source_domains} vs {target_domains}")

    # Heuristic 3: Geographic confusion
    geographic_keywords = {'north', 'south', 'east', 'west', 'california', 'europe',
                          'africa', 'asia', 'state', 'square'}
    source_geo = any(kw in source_lower for kw in geographic_keywords)
    target_geo = any(kw in target_lower for kw in geographic_keywords)

    if source_geo and target_geo:
        # Check for directional conflicts
        directions = {'north', 'south', 'east', 'west'}
        source_dir = directions & set(source_lower.split())
        target_dir = directions & set(target_lower.split())

        if source_dir and target_dir and source_dir != target_dir:
            score -= 0.5
            reasons.append(f"geographic_direction_conflict: {source_dir} vs {target_dir}")

    # Heuristic 4: Proper name vs generic term
    source_is_name = is_proper_name(source)
    target_is_name = is_proper_name(target)
    source_is_generic = is_generic_term(source)
    target_is_generic = is_generic_term(target)

    if (source_is_name and target_is_generic) or (source_is_generic and target_is_name):
        score -= 0.3
        reasons.append("proper_name_vs_generic")

    # Heuristic 5: Short word merged with longer compound
    source_words = len(source.split())
    target_words = len(target.split())

    if source_words == 1 and target_words >= 3:
        if not (source_lower in target_lower):  # Unless it's a substring
            score -= 0.2
            reasons.append("single_to_compound_merge")

    # Heuristic 6: High similarity different meaning patterns
    # These are specific patterns we know are problematic
    problematic_pairs = [
        ('mood', 'food'), ('floods', 'food'),
        ('cooking', 'coaching'), ('parenting', 'gardening'),
        ('society', 'soviets'), ('schools', 'scholars'),
        ('water', 'nature'), ('ocean', 'japan'),
        ('mountains', 'montana'), ('mountain', 'montana'),
    ]

    for p1, p2 in problematic_pairs:
        if (source_lower == p1 and target_lower == p2) or \
           (source_lower == p2 and target_lower == p1):
            score = 0.0
            reasons.append("known_bad_pattern")
            break

    # Clamp score
    score = max(0.0, min(1.0, score))

    # Determine recommendation
    if score <= 0.3:
        recommendation = "blocklist"
    elif score <= 0.6:
        recommendation = "review"
    else:
        recommendation = "ok"

    return {
        'source': source,
        'target': target,
        'plausibility_score': round(score, 3),
        'levenshtein_ratio': round(lev_ratio, 3),
        'reasons': reasons,
        'recommendation': recommendation
    }


def load_merges(filepath: str) -> Dict[str, str]:
    """Load merge history from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract merges dict (skip stats)
    return data.get('merges', {})


def main():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)

    merges_path = os.path.join(repo_root, 'data', 'knowledge_graph_unified', 'entity_merges.json')
    output_path = os.path.join(repo_root, 'data', 'analysis', 'bad_merge_candidates.json')

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Loading merges from: {merges_path}")
    merges = load_merges(merges_path)
    print(f"Loaded {len(merges)} merge entries")

    # Analyze all merges
    suspicious_merges = []
    all_analyses = []

    for source, target in merges.items():
        analysis = analyze_merge(source, target)
        all_analyses.append(analysis)

        if analysis['recommendation'] in ('blocklist', 'review'):
            suspicious_merges.append(analysis)

    # Sort by plausibility score (lowest first = most suspicious)
    suspicious_merges.sort(key=lambda x: x['plausibility_score'])

    # Categorize by recommendation
    blocklist_candidates = [m for m in suspicious_merges if m['recommendation'] == 'blocklist']
    review_candidates = [m for m in suspicious_merges if m['recommendation'] == 'review']

    # Generate confirmed blocklist additions (top suspicious with specific patterns)
    confirmed_blocklist = []
    for merge in blocklist_candidates[:50]:  # Top 50 most suspicious
        pair = (merge['source'].lower(), merge['target'].lower())
        if pair not in confirmed_blocklist:
            confirmed_blocklist.append(pair)

    # Build output report
    report = {
        'analysis_date': datetime.now().isoformat()[:10],
        'total_merges_analyzed': len(merges),
        'suspicious_count': len(suspicious_merges),
        'blocklist_candidates_count': len(blocklist_candidates),
        'review_candidates_count': len(review_candidates),
        'suspicious_merges': suspicious_merges[:100],  # Top 100 for review
        'confirmed_blocklist_additions': confirmed_blocklist[:20],  # Top 20 confirmed
        'statistics': {
            'avg_plausibility': round(sum(a['plausibility_score'] for a in all_analyses) / len(all_analyses), 3) if all_analyses else 0,
            'merges_below_0.5': len([a for a in all_analyses if a['plausibility_score'] < 0.5]),
            'merges_below_0.3': len([a for a in all_analyses if a['plausibility_score'] < 0.3]),
            'reason_counts': defaultdict(int)
        }
    }

    # Count reasons
    for analysis in all_analyses:
        for reason in analysis['reasons']:
            reason_type = reason.split(':')[0] if ':' in reason else reason
            report['statistics']['reason_counts'][reason_type] += 1

    report['statistics']['reason_counts'] = dict(report['statistics']['reason_counts'])

    # Save report
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print("BAD MERGE ANALYSIS REPORT")
    print('='*60)
    print(f"Total merges analyzed: {len(merges)}")
    print(f"Suspicious merges identified: {len(suspicious_merges)}")
    print(f"  - Blocklist candidates: {len(blocklist_candidates)}")
    print(f"  - Review candidates: {len(review_candidates)}")
    print(f"\nStatistics:")
    print(f"  - Average plausibility: {report['statistics']['avg_plausibility']}")
    print(f"  - Merges below 0.5 plausibility: {report['statistics']['merges_below_0.5']}")
    print(f"  - Merges below 0.3 plausibility: {report['statistics']['merges_below_0.3']}")
    print(f"\nReason breakdown:")
    for reason, count in sorted(report['statistics']['reason_counts'].items(), key=lambda x: -x[1]):
        print(f"  - {reason}: {count}")
    print(f"\nTop 10 most suspicious merges:")
    for i, merge in enumerate(suspicious_merges[:10], 1):
        print(f"  {i}. '{merge['source']}' -> '{merge['target']}'")
        print(f"     Score: {merge['plausibility_score']}, Reasons: {', '.join(merge['reasons'])}")
    print(f"\nConfirmed blocklist additions (first 10):")
    for i, (s, t) in enumerate(confirmed_blocklist[:10], 1):
        print(f"  {i}. ('{s}', '{t}')")
    print(f"\nReport saved to: {output_path}")
    print('='*60)

    return report


if __name__ == '__main__':
    main()
