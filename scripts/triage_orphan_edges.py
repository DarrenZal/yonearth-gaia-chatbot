#!/usr/bin/env python3
"""
Orphan Edge Triage Script

Analyzes orphan edges to identify root causes and suggest fixes:
1. Tokenization/normalization mismatches
2. Missing aliases
3. Type mismatches
4. Source-specific gaps
"""

import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple
import re

PROJECT_ROOT = Path("/home/claudeuser/yonearth-gaia-chatbot")
UNIFIED_PATH = PROJECT_ROOT / "data/knowledge_graph_unified/unified.json"
STATS_PATH = PROJECT_ROOT / "data/knowledge_graph_unified/stats.json"

def load_data():
    """Load unified graph and stats"""
    print("Loading knowledge graph data...")

    with open(UNIFIED_PATH, 'r') as f:
        unified = json.load(f)

    with open(STATS_PATH, 'r') as f:
        stats = json.load(f)

    return unified, stats

def analyze_orphan_patterns(unified: Dict, stats: Dict):
    """Analyze patterns in orphan edges"""

    orphan_edges = stats.get('orphan_edges', [])
    entities = unified.get('entities', {})
    relationships = unified.get('relationships', [])

    print(f"\n{'='*60}")
    print(f"ORPHAN EDGE TRIAGE REPORT")
    print(f"{'='*60}")
    print(f"Total orphan edges: {len(orphan_edges)}")
    print(f"Total relationships: {len(relationships)}")
    print(f"Orphan rate: {len(orphan_edges)/len(relationships)*100:.2f}%")

    # Categorize orphan types
    missing_source = []
    missing_target = []
    missing_both = []

    for edge in orphan_edges:
        if not edge['source_exists'] and not edge['target_exists']:
            missing_both.append(edge)
        elif not edge['source_exists']:
            missing_source.append(edge)
        else:
            missing_target.append(edge)

    print(f"\nOrphan Categories:")
    print(f"  - Missing source entity: {len(missing_source)}")
    print(f"  - Missing target entity: {len(missing_target)}")
    print(f"  - Missing both entities: {len(missing_both)}")

    return missing_source, missing_target, missing_both

def find_tokenization_issues(orphan_edges: List[Dict], entities: Dict) -> Dict:
    """Identify potential tokenization/normalization mismatches"""

    issues = {
        'case_mismatch': [],
        'punctuation_diff': [],
        'whitespace_diff': [],
        'potential_aliases': []
    }

    # Create normalized entity lookup
    normalized_entities = {}
    for entity_name in entities.keys():
        # Various normalizations
        normalized_variants = [
            entity_name.lower(),
            entity_name.upper(),
            entity_name.replace("-", " "),
            entity_name.replace("_", " "),
            re.sub(r'[^\w\s]', '', entity_name),
            ' '.join(entity_name.split())  # Normalize whitespace
        ]

        for variant in normalized_variants:
            if variant not in normalized_entities:
                normalized_entities[variant] = []
            normalized_entities[variant].append(entity_name)

    # Check orphan edges for normalization matches
    for edge in orphan_edges:
        missing_entity = edge['source'] if not edge.get('source_exists', True) else edge['target']

        # Check various normalizations
        if missing_entity.lower() in normalized_entities:
            canonical = normalized_entities[missing_entity.lower()][0]
            if canonical != missing_entity:
                issues['case_mismatch'].append({
                    'missing': missing_entity,
                    'exists_as': canonical,
                    'predicate': edge['predicate']
                })

        # Check without punctuation
        no_punct = re.sub(r'[^\w\s]', '', missing_entity)
        if no_punct in normalized_entities and no_punct != missing_entity:
            issues['punctuation_diff'].append({
                'missing': missing_entity,
                'normalized': no_punct,
                'candidates': normalized_entities[no_punct][:3]
            })

        # Check with normalized whitespace
        normalized_ws = ' '.join(missing_entity.split())
        if normalized_ws != missing_entity and normalized_ws in entities:
            issues['whitespace_diff'].append({
                'missing': missing_entity,
                'exists_as': normalized_ws
            })

    return issues

def analyze_predicate_patterns(orphan_edges: List[Dict]) -> Dict:
    """Analyze predicates associated with orphan edges"""

    predicate_counts = Counter()
    predicate_examples = defaultdict(list)

    for edge in orphan_edges:
        pred = edge['predicate']
        predicate_counts[pred] += 1

        if len(predicate_examples[pred]) < 3:
            predicate_examples[pred].append({
                'source': edge['source'],
                'target': edge['target']
            })

    return {
        'counts': dict(predicate_counts.most_common(10)),
        'examples': dict(predicate_examples)
    }

def suggest_aliases(orphan_edges: List[Dict], entities: Dict) -> List[Dict]:
    """Suggest potential aliases based on string similarity"""

    suggestions = []
    entity_names = set(entities.keys())

    # Get unique missing entities
    missing_entities = set()
    for edge in orphan_edges:
        if not edge.get('source_exists', True):
            missing_entities.add(edge['source'])
        if not edge.get('target_exists', True):
            missing_entities.add(edge['target'])

    for missing in list(missing_entities)[:20]:  # Limit to top 20 for readability
        # Find close matches using simple heuristics
        candidates = []

        # Check for substring matches
        for entity in entity_names:
            if missing.lower() in entity.lower() or entity.lower() in missing.lower():
                similarity = len(set(missing.lower()) & set(entity.lower())) / max(len(missing), len(entity))
                if similarity > 0.5:
                    candidates.append((entity, similarity))

        # Sort by similarity
        candidates.sort(key=lambda x: x[1], reverse=True)

        if candidates:
            suggestions.append({
                'missing': missing,
                'candidates': [(c[0], f"{c[1]:.2%}") for c in candidates[:3]]
            })

    return suggestions

def generate_fixes(issues: Dict, suggestions: List[Dict]) -> Dict:
    """Generate concrete fixes for the identified issues"""

    fixes = {
        'alias_mappings': {},
        'normalization_rules': [],
        'entity_merges': []
    }

    # Generate alias mappings from case mismatches
    for issue in issues.get('case_mismatch', [])[:10]:
        fixes['alias_mappings'][issue['missing']] = issue['exists_as']

    # Generate normalization rules
    if issues['punctuation_diff']:
        fixes['normalization_rules'].append("Remove punctuation before entity matching")

    if issues['whitespace_diff']:
        fixes['normalization_rules'].append("Normalize whitespace (multiple spaces to single)")

    # Generate entity merge suggestions
    for suggestion in suggestions[:5]:
        if suggestion['candidates'] and float(suggestion['candidates'][0][1].rstrip('%')) > 75:
            fixes['entity_merges'].append({
                'from': suggestion['missing'],
                'to': suggestion['candidates'][0][0],
                'confidence': suggestion['candidates'][0][1]
            })

    return fixes

def main():
    """Main triage analysis"""
    unified, stats = load_data()

    # Get orphan edges
    orphan_edges = stats.get('orphan_edges', [])

    if not orphan_edges:
        print("No orphan edges found! Graph is fully connected.")
        return

    # Analyze orphan patterns
    missing_source, missing_target, missing_both = analyze_orphan_patterns(unified, stats)

    # Find tokenization issues
    print(f"\n{'='*60}")
    print("TOKENIZATION/NORMALIZATION ISSUES")
    print(f"{'='*60}")

    issues = find_tokenization_issues(orphan_edges, unified['entities'])

    print(f"\nCase mismatches found: {len(issues['case_mismatch'])}")
    for issue in issues['case_mismatch'][:5]:
        print(f"  - '{issue['missing']}' exists as '{issue['exists_as']}'")

    print(f"\nPunctuation differences: {len(issues['punctuation_diff'])}")
    for issue in issues['punctuation_diff'][:5]:
        print(f"  - '{issue['missing']}' → {issue['candidates'][:2]}")

    print(f"\nWhitespace differences: {len(issues['whitespace_diff'])}")
    for issue in issues['whitespace_diff'][:5]:
        print(f"  - '{issue['missing']}' → '{issue['exists_as']}'")

    # Analyze predicate patterns
    print(f"\n{'='*60}")
    print("PREDICATE ANALYSIS")
    print(f"{'='*60}")

    pred_analysis = analyze_predicate_patterns(orphan_edges)
    print("\nTop predicates in orphan edges:")
    for pred, count in pred_analysis['counts'].items():
        print(f"  - {pred}: {count} edges")
        if pred in pred_analysis['examples']:
            for ex in pred_analysis['examples'][pred][:2]:
                print(f"      Example: {ex['source'][:30]} → {ex['target'][:30]}")

    # Suggest aliases
    print(f"\n{'='*60}")
    print("ALIAS SUGGESTIONS")
    print(f"{'='*60}")

    suggestions = suggest_aliases(orphan_edges, unified['entities'])

    print(f"\nPotential aliases found: {len(suggestions)}")
    for sugg in suggestions[:10]:
        print(f"\n  Missing: '{sugg['missing']}'")
        print(f"  Candidates:")
        for candidate, score in sugg['candidates']:
            print(f"    - {candidate} ({score})")

    # Generate fixes
    print(f"\n{'='*60}")
    print("RECOMMENDED FIXES")
    print(f"{'='*60}")

    fixes = generate_fixes(issues, suggestions)

    print("\n1. Add these alias mappings:")
    for missing, canonical in list(fixes['alias_mappings'].items())[:10]:
        print(f"   '{missing}': '{canonical}'")

    print("\n2. Apply these normalization rules:")
    for rule in fixes['normalization_rules']:
        print(f"   - {rule}")

    print("\n3. Consider these entity merges:")
    for merge in fixes['entity_merges']:
        print(f"   - Merge '{merge['from']}' → '{merge['to']}' (confidence: {merge['confidence']})")

    # Save detailed report
    report_path = PROJECT_ROOT / "data/knowledge_graph_unified/orphan_triage_report.json"
    report = {
        'summary': {
            'total_orphan_edges': len(orphan_edges),
            'orphan_rate': f"{len(orphan_edges)/len(unified['relationships'])*100:.2f}%",
            'missing_source': len(missing_source),
            'missing_target': len(missing_target),
            'missing_both': len(missing_both)
        },
        'issues': {
            'case_mismatches': len(issues['case_mismatch']),
            'punctuation_diffs': len(issues['punctuation_diff']),
            'whitespace_diffs': len(issues['whitespace_diff'])
        },
        'top_predicates': pred_analysis['counts'],
        'alias_suggestions': suggestions[:20],
        'recommended_fixes': fixes
    }

    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n✅ Detailed report saved to: {report_path}")

if __name__ == "__main__":
    main()