#!/usr/bin/env python3
"""
Add classification_flags to existing postprocessed episode files.

Episodes were postprocessed with ACE v3.2.2 which didn't populate classification_flags.
This script adds them so episodes can benefit from discourse graph transformation.

Usage:
    python scripts/add_classification_flags_to_episodes.py
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Set, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleRelationshipClassifier:
    """
    Simplified classifier that works with plain JSON dicts (not relationship objects).

    Based on ClaimClassifier but adapted for direct JSON manipulation.
    """

    def __init__(self):
        # Factual predicates - verifiable facts
        self.factual_predicates = {
            'authored', 'wrote', 'published', 'founded', 'located', 'contains',
            'born', 'died', 'established', 'created', 'produced', 'released',
            'named', 'called', 'titled', 'dated', 'measured', 'counted',
            'discovered', 'invented', 'built', 'constructed', 'designed',
            'has_part', 'part_of', 'member_of', 'employed_by', 'works_for',
            'educated_at', 'graduated_from', 'awarded', 'received',
            'published_in', 'appeared_in', 'cited_in', 'referenced_in',
            'is the author of', 'is vice president of', 'sourced ingredients from',
            'is', 'are', 'was', 'were', 'has', 'have', 'had'
        }

        # Philosophical predicates - abstract relationships
        self.philosophical_predicates = {
            'represents', 'symbolizes', 'embodies', 'reflects', 'signifies',
            'manifests', 'expresses', 'conveys', 'demonstrates', 'illustrates',
            'reveals', 'suggests', 'implies', 'indicates', 'means',
            'is_essence_of', 'is_nature_of', 'transcends', 'encompasses'
        }

        # Opinion markers in evidence text
        self.opinion_patterns = [
            r'\bbelieves?\b', r'\bthinks?\b', r'\bfeels?\b', r'\bopines?\b',
            r'\bclaims?\b', r'\bargues?\b', r'\bcontends?\b', r'\bmaintains?\b',
            r'\basserts?\b', r'\bproposes?\b', r'\bsuggests?\b',
            r'\bin my opinion\b', r'\bin his view\b', r'\bin her view\b',
            r'\baccording to\b', r'\bseems? to\b', r'\bappears? to\b'
        ]

        # Recommendation markers in evidence text
        self.recommendation_patterns = [
            r'\bshould\b', r'\bmust\b', r'\bshall\b', r'\bought to\b',
            r'\bneed to\b', r'\bhave to\b', r'\brecommends?\b', r'\badvises?\b',
            r'\bsuggests?\b', r'\burges?\b', r'\bencourages?\b', r'\bcounsels?\b',
            r'\bit is important\b', r'\bit is essential\b', r'\bit is vital\b',
            r'\bit is critical\b', r'\bit is necessary\b',
            r'\bwe should\b', r'\byou should\b', r'\bone should\b',
            r'\bwe must\b', r'\byou must\b', r'\bone must\b'
        ]

    def classify(self, rel: Dict[str, Any]) -> List[str]:
        """
        Classify relationship and return classification flags as lowercase list.

        Args:
            rel: Relationship dict with keys: source, target, relationship, evidence_text

        Returns:
            List of classification flags: ['factual', 'opinion', 'philosophical', 'recommendation']
        """
        classifications = set()

        # Get predicate (handle different key names)
        predicate = (
            rel.get('relationship') or
            rel.get('predicate') or
            rel.get('relationship_type') or
            ''
        ).lower().strip()

        # Check factual
        if predicate in self.factual_predicates:
            classifications.add('factual')

        # Check philosophical
        if predicate in self.philosophical_predicates:
            classifications.add('philosophical')

        # Check evidence text for opinion/recommendation markers
        evidence = rel.get('evidence_text', '') or ''
        if evidence:
            evidence_lower = evidence.lower()

            # Check opinion markers
            for pattern in self.opinion_patterns:
                if re.search(pattern, evidence_lower):
                    classifications.add('opinion')
                    break

            # Check recommendation markers
            for pattern in self.recommendation_patterns:
                if re.search(pattern, evidence_lower):
                    classifications.add('recommendation')
                    break

        # Default to factual if nothing else matched
        if not classifications:
            # Check for abstract/vague concepts
            source = (rel.get('source') or rel.get('source_entity') or '').lower()
            target = (rel.get('target') or rel.get('target_entity') or '').lower()
            abstract_indicators = ['essence', 'nature', 'spirit', 'soul', 'energy', 'consciousness']

            if any(ind in source or ind in target for ind in abstract_indicators):
                classifications.add('philosophical')
            else:
                classifications.add('factual')

        return sorted(list(classifications))


def add_classification_flags(episode_file: Path, classifier: SimpleRelationshipClassifier) -> Dict[str, Any]:
    """
    Add classification_flags to all relationships in an episode file.

    Args:
        episode_file: Path to episode JSON file
        classifier: Relationship classifier

    Returns:
        Statistics dict
    """
    logger.info(f"Processing: {episode_file.name}")

    with open(episode_file, 'r') as f:
        data = json.load(f)

    relationships = data.get('relationships', [])
    original_count = len(relationships)

    # Classification stats
    stats = {
        'factual': 0,
        'philosophical': 0,
        'opinion': 0,
        'recommendation': 0
    }

    # Add classification_flags to each relationship
    for rel in relationships:
        # Classify
        flags = classifier.classify(rel)

        # Add to relationship
        rel['classification_flags'] = flags

        # Update stats
        for flag in flags:
            if flag in stats:
                stats[flag] += 1

    # Update episode data
    data['classification_version'] = 'v1.0.0_retrofit'
    data['classification_date'] = '2025-11-21'

    # Save updated file
    with open(episode_file, 'w') as f:
        json.dump(data, f, indent=2)

    logger.info(f"  ✅ Classified {original_count} relationships:")
    logger.info(f"     Factual: {stats['factual']}, Philosophical: {stats['philosophical']}, "
                f"Opinion: {stats['opinion']}, Recommendation: {stats['recommendation']}")

    return stats


def main():
    project_root = Path("/home/claudeuser/yonearth-gaia-chatbot")
    episodes_dir = project_root / "data/knowledge_graph_unified/episodes_postprocessed"

    if not episodes_dir.exists():
        logger.error(f"Episodes directory not found: {episodes_dir}")
        return

    # Find all episode files
    episode_files = sorted(episodes_dir.glob('episode_*_post.json'))
    logger.info(f"Found {len(episode_files)} episode files")

    if not episode_files:
        logger.error("No episode files found")
        return

    # Create classifier
    classifier = SimpleRelationshipClassifier()

    # Process all episodes
    total_stats = {
        'factual': 0,
        'philosophical': 0,
        'opinion': 0,
        'recommendation': 0,
        'files_processed': 0
    }

    for episode_file in episode_files:
        try:
            stats = add_classification_flags(episode_file, classifier)

            # Accumulate stats
            for key in ['factual', 'philosophical', 'opinion', 'recommendation']:
                total_stats[key] += stats[key]
            total_stats['files_processed'] += 1

        except Exception as e:
            logger.error(f"Error processing {episode_file.name}: {e}")
            continue

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"✅ COMPLETE: Processed {total_stats['files_processed']}/{len(episode_files)} episodes")
    logger.info(f"📊 Total Classifications:")
    logger.info(f"   Factual: {total_stats['factual']}")
    logger.info(f"   Philosophical: {total_stats['philosophical']}")
    logger.info(f"   Opinion: {total_stats['opinion']}")
    logger.info(f"   Recommendation: {total_stats['recommendation']}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
