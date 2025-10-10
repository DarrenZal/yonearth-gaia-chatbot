#!/usr/bin/env python3
"""
Correction Feedback Loop System
Based on research insight: Active learning with human corrections to improve extraction

This system:
1. Presents problematic relationships for human review
2. Records corrections with reasoning
3. Builds training data from corrections
4. Learns patterns to improve future extractions
"""

import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import os

DATA_DIR = Path("/home/claudeuser/yonearth-gaia-chatbot/data")
KG_DIR = DATA_DIR / "knowledge_graph_v2"
CORRECTIONS_DIR = DATA_DIR / "corrections"
CORRECTIONS_DIR.mkdir(exist_ok=True)

class CorrectionFeedbackLoop:
    """Manages human corrections and learning from them"""

    def __init__(self):
        self.corrections_file = CORRECTIONS_DIR / "corrections_log.json"
        self.patterns_file = CORRECTIONS_DIR / "learned_patterns.json"
        self.corrections = self.load_corrections()
        self.learned_patterns = self.load_patterns()

    def load_corrections(self):
        """Load existing corrections"""
        if self.corrections_file.exists():
            with open(self.corrections_file) as f:
                return json.load(f)
        return []

    def load_patterns(self):
        """Load learned patterns from corrections"""
        if self.patterns_file.exists():
            with open(self.patterns_file) as f:
                return json.load(f)
        return {'error_patterns': [], 'correction_rules': []}

    def save_corrections(self):
        """Save corrections to file"""
        with open(self.corrections_file, 'w') as f:
            json.dump(self.corrections, f, indent=2)

    def save_patterns(self):
        """Save learned patterns"""
        with open(self.patterns_file, 'w') as f:
            json.dump(self.learned_patterns, f, indent=2)

    def record_correction(self, original_rel, correction_action, corrected_rel=None, reasoning=""):
        """
        Record a human correction

        Args:
            original_rel: The original relationship dict
            correction_action: 'DELETE', 'MODIFY', 'REVERSE', 'RETYPE'
            corrected_rel: The corrected relationship (if not DELETE)
            reasoning: Why this correction was made
        """
        correction = {
            'timestamp': datetime.now().isoformat(),
            'action': correction_action,
            'original': original_rel,
            'corrected': corrected_rel,
            'reasoning': reasoning,
            'episode': original_rel.get('episode_number')
        }

        self.corrections.append(correction)
        self.save_corrections()

        # Learn from this correction
        self._learn_from_correction(correction)

        return correction

    def _learn_from_correction(self, correction):
        """Extract patterns from a correction to improve future extractions"""
        action = correction['action']
        original = correction['original']
        reasoning = correction['reasoning']

        # Pattern 1: Vague entities should be deleted
        if action == 'DELETE':
            if 'unknown' in original['target'].lower() or 'unclear' in original['target'].lower():
                pattern = {
                    'type': 'vague_entity_deletion',
                    'rule': 'DELETE relationships where target contains "unknown", "unclear", or "unspecified"',
                    'confidence_threshold': original.get('relationship_confidence', 0),
                    'examples': [original]
                }
                self._add_or_update_pattern(pattern)

        # Pattern 2: Abstract concepts as locations
        if action == 'DELETE' and 'abstract' in reasoning.lower():
            abstract_terms = ['biochar', 'concept', 'principle']
            for term in abstract_terms:
                if term in original['target'].lower():
                    pattern = {
                        'type': 'abstract_as_location',
                        'rule': f'DELETE geographic relationships where target is abstract concept like "{term}"',
                        'trigger_terms': [term],
                        'examples': [original]
                    }
                    self._add_or_update_pattern(pattern)

        # Pattern 3: Wrong relationship type (part_of vs is)
        if action == 'RETYPE':
            if 'part_of' in original['relationship'] and correction['corrected']:
                if 'is' in correction['corrected']['relationship']:
                    pattern = {
                        'type': 'wrong_relation_type_nationality',
                        'rule': 'Change "part_of" to "is" for nationality relationships',
                        'from_rel': 'part_of',
                        'to_rel': 'is',
                        'context': 'nationality/ethnicity',
                        'examples': [original]
                    }
                    self._add_or_update_pattern(pattern)

        # Pattern 4: Reversed geographic relationships
        if action == 'REVERSE':
            pattern = {
                'type': 'reversed_geographic',
                'rule': f'Reverse relationship: {original["source"]} should not contain {original["target"]}',
                'trigger': f'{original["source"]} --> {original["target"]}',
                'examples': [original]
            }
            self._add_or_update_pattern(pattern)

        self.save_patterns()

    def _add_or_update_pattern(self, new_pattern):
        """Add a new pattern or update existing one with more examples"""
        existing = None
        for p in self.learned_patterns['error_patterns']:
            if p['type'] == new_pattern['type']:
                existing = p
                break

        if existing:
            # Update existing pattern with new example
            if 'examples' not in existing:
                existing['examples'] = []
            existing['examples'].append(new_pattern['examples'][0])
            existing['count'] = existing.get('count', 1) + 1
        else:
            # Add new pattern
            new_pattern['count'] = 1
            self.learned_patterns['error_patterns'].append(new_pattern)

    def apply_learned_patterns(self, relationships):
        """
        Apply learned patterns to automatically correct/flag new relationships

        Returns:
            tuple: (auto_corrected, flagged_for_review)
        """
        auto_corrected = []
        flagged_for_review = []

        for rel in relationships:
            corrected = False

            # Apply each learned pattern
            for pattern in self.learned_patterns['error_patterns']:

                # Pattern: Delete vague entities
                if pattern['type'] == 'vague_entity_deletion':
                    if any(term in rel['target'].lower() for term in ['unknown', 'unclear', 'unspecified']):
                        auto_corrected.append({
                            'action': 'DELETE',
                            'original': rel,
                            'reason': f"Learned pattern: {pattern['rule']}",
                            'pattern_confidence': pattern.get('count', 1) / 10  # Higher count = higher confidence
                        })
                        corrected = True
                        break

                # Pattern: Delete abstract as location
                if pattern['type'] == 'abstract_as_location':
                    if any(term in rel['target'].lower() for term in pattern.get('trigger_terms', [])):
                        if any(geo in rel['relationship'].lower() for geo in ['located_in', 'part_of', 'contains']):
                            auto_corrected.append({
                                'action': 'DELETE',
                                'original': rel,
                                'reason': f"Learned pattern: {pattern['rule']}",
                                'pattern_confidence': pattern.get('count', 1) / 10
                            })
                            corrected = True
                            break

                # Pattern: Fix wrong relationship type
                if pattern['type'] == 'wrong_relation_type_nationality':
                    if pattern['from_rel'] in rel['relationship'].lower():
                        nationality_terms = ['lebanese', 'american', 'european', 'african']
                        if any(term in rel['target'].lower() for term in nationality_terms):
                            auto_corrected.append({
                                'action': 'RETYPE',
                                'original': rel,
                                'corrected': {
                                    **rel,
                                    'relationship': pattern['to_rel']
                                },
                                'reason': f"Learned pattern: {pattern['rule']}",
                                'pattern_confidence': pattern.get('count', 1) / 10
                            })
                            corrected = True
                            break

            # If not auto-corrected but low confidence, flag for review
            if not corrected and rel['relationship_confidence'] < 0.70:
                flagged_for_review.append(rel)

        return auto_corrected, flagged_for_review

    def generate_correction_report(self):
        """Generate a report of corrections and learned patterns"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_corrections': len(self.corrections),
            'corrections_by_action': defaultdict(int),
            'learned_patterns_count': len(self.learned_patterns['error_patterns']),
            'top_patterns': []
        }

        # Count by action type
        for corr in self.corrections:
            report['corrections_by_action'][corr['action']] += 1

        # Get top patterns by frequency
        sorted_patterns = sorted(
            self.learned_patterns['error_patterns'],
            key=lambda p: p.get('count', 0),
            reverse=True
        )

        report['top_patterns'] = [
            {
                'type': p['type'],
                'rule': p['rule'],
                'frequency': p.get('count', 0),
                'example_count': len(p.get('examples', []))
            }
            for p in sorted_patterns[:10]
        ]

        return report

def interactive_correction_session():
    """Interactive session for correcting relationships"""
    loop = CorrectionFeedbackLoop()

    # Load problematic relationships from analysis
    analysis_file = KG_DIR / "quality_analysis_report.json"
    if not analysis_file.exists():
        print("❌ Run analyze_extraction_quality.py first!")
        return

    with open(analysis_file) as f:
        analysis = json.load(f)

    suspicious = analysis['examples']['suspicious']

    print("="*70)
    print("🔧 INTERACTIVE CORRECTION SESSION")
    print("="*70)
    print(f"Found {len(suspicious)} suspicious relationships")
    print("\nCommands:")
    print("  d - DELETE this relationship")
    print("  m - MODIFY this relationship")
    print("  r - REVERSE this relationship")
    print("  t - RETYPE (change relationship type)")
    print("  k - KEEP (it's actually correct)")
    print("  s - SKIP (review later)")
    print("  q - QUIT\n")

    for i, item in enumerate(suspicious[:20], 1):  # Start with 20
        print(f"\n{'='*70}")
        print(f"Relationship {i}/{min(20, len(suspicious))}")
        print(f"{'='*70}")
        print(f"Episode: {item['episode']}")
        print(f"Source: {item['source']}")
        print(f"Relationship: {item['relationship']}")
        print(f"Target: {item['target']}")
        print(f"Confidence: {item['confidence']:.2f}")
        print(f"Issue: {item['issue']}")

        action = input("\nAction (d/m/r/t/k/s/q): ").strip().lower()

        if action == 'q':
            break
        elif action == 's':
            continue
        elif action == 'k':
            print("✓ Marked as correct")
            continue
        elif action == 'd':
            reasoning = input("Why delete? ").strip()
            loop.record_correction(item, 'DELETE', reasoning=reasoning)
            print(f"✓ Deleted and pattern learned")
        elif action == 't':
            new_rel_type = input("New relationship type: ").strip()
            corrected = {
                **item,
                'relationship': new_rel_type
            }
            reasoning = input("Why retype? ").strip()
            loop.record_correction(item, 'RETYPE', corrected, reasoning)
            print(f"✓ Retyped and pattern learned")
        elif action == 'r':
            corrected = {
                'source': item['target'],
                'relationship': item['relationship'],
                'target': item['source']
            }
            reasoning = input("Why reverse? ").strip()
            loop.record_correction(item, 'REVERSE', corrected, reasoning)
            print(f"✓ Reversed and pattern learned")

    # Generate report
    report = loop.generate_correction_report()
    print("\n" + "="*70)
    print("📊 SESSION SUMMARY")
    print("="*70)
    print(f"Total corrections: {report['total_corrections']}")
    print(f"Learned patterns: {report['learned_patterns_count']}")
    print("\nTop patterns:")
    for p in report['top_patterns']:
        print(f"  - {p['type']}: {p['frequency']} occurrences")

    return loop

if __name__ == "__main__":
    interactive_correction_session()
