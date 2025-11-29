#!/usr/bin/env python3
"""
Add classification_flags to existing unified knowledge graph.

The unified_normalized.json contains 172 episodes but lacks classification_flags.
This script adds them so the graph can benefit from discourse graph transformation.

Usage:
    python scripts/add_classification_flags_to_unified_graph.py
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnifiedGraphClassifier:
    """
    Classifier for unified graph format (different from episode format).

    Unified graph uses:
    - 'predicate' instead of 'relationship'
    - 'source' and 'target' instead of 'source_entity'/'target_entity'
    - 'evidence' dict with nested 'text' field
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
            'is', 'are', 'was', 'were', 'has', 'have', 'had',
            'co-founded', 'owns', 'operates', 'manages', 'leads', 'directs',
            'located_in', 'based_in', 'headquartered_in', 'works_at'
        }

        # Philosophical predicates
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

        # Recommendation markers
        self.recommendation_patterns = [
            r'\bshould\b', r'\bmust\b', r'\bshall\b', r'\bought to\b',
            r'\bneed to\b', r'\bhave to\b', r'\brecommends?\b', r'\badvises?\b',
            r'\bsuggests?\b', r'\burges?\b', r'\bencourages?\b', r'\bcounsels?\b',
            r'\bit is important\b', r'\bit is essential\b', r'\bit is vital\b',
            r'\bit is critical\b', r'\bit is necessary\b',
            r'\bwe should\b', r'\byou should\b', r'\bone should\b'
        ]

    def classify(self, rel: Dict[str, Any]) -> List[str]:
        """
        Classify unified graph relationship.

        Args:
            rel: Relationship dict with keys: id, source, target, predicate, evidence, etc.

        Returns:
            List of classification flags: ['factual', 'opinion', 'philosophical', 'recommendation']
        """
        classifications = set()

        # Get predicate (handle variations)
        predicate = (
            rel.get('predicate') or
            rel.get('original_predicate') or
            ''
        ).lower().strip().replace('_', ' ')

        # Check factual
        if predicate in self.factual_predicates:
            classifications.add('factual')

        # Check philosophical
        if predicate in self.philosophical_predicates:
            classifications.add('philosophical')

        # Check evidence text for opinion/recommendation markers
        evidence = rel.get('evidence', {})
        if isinstance(evidence, dict):
            evidence_text = evidence.get('text', '') or evidence.get('evidence_text', '')
        else:
            evidence_text = str(evidence) if evidence else ''

        if evidence_text:
            evidence_lower = evidence_text.lower()

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
            source = rel.get('source', '').lower()
            target = rel.get('target', '').lower()
            abstract_indicators = ['essence', 'nature', 'spirit', 'soul', 'energy', 'consciousness']

            if any(ind in source or ind in target for ind in abstract_indicators):
                classifications.add('philosophical')
            else:
                classifications.add('factual')

        return sorted(list(classifications))


def add_classification_flags_to_unified_graph(
    input_file: Path,
    output_file: Path,
    backup_file: Path
) -> Dict[str, Any]:
    """
    Add classification_flags to unified knowledge graph.

    Args:
        input_file: Path to unified_normalized.json
        output_file: Path to save updated graph
        backup_file: Path to save backup

    Returns:
        Statistics dict
    """
    logger.info(f"Loading unified graph: {input_file}")

    # Backup original
    import shutil
    shutil.copy(input_file, backup_file)
    logger.info(f"  Backup saved: {backup_file}")

    with open(input_file, 'r') as f:
        data = json.load(f)

    relationships = data.get('relationships', [])
    original_count = len(relationships)

    logger.info(f"  Relationships: {original_count}")
    logger.info(f"  Entities: {len(data.get('entities', {}))}")

    # Create classifier
    classifier = UnifiedGraphClassifier()

    # Classification stats
    stats = {
        'factual': 0,
        'philosophical': 0,
        'opinion': 0,
        'recommendation': 0
    }

    # Classify each relationship
    logger.info("Classifying relationships...")

    for i, rel in enumerate(relationships):
        # Classify
        flags = classifier.classify(rel)

        # Add classification_flags
        rel['classification_flags'] = flags

        # Update stats
        for flag in flags:
            if flag in stats:
                stats[flag] += 1

        # Progress update every 10,000 relationships
        if (i + 1) % 10000 == 0:
            logger.info(f"  Processed {i + 1}/{original_count} relationships...")

    # Add metadata about classification
    data['classification_metadata'] = {
        'version': 'v1.0.0_unified_retrofit',
        'date': datetime.now().isoformat(),
        'relationships_classified': original_count,
        'statistics': stats
    }

    # Save updated graph
    logger.info(f"Saving updated graph: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    logger.info(f"✅ Complete!")
    logger.info(f"📊 Classification Statistics:")
    logger.info(f"   Factual: {stats['factual']} ({stats['factual']/original_count*100:.1f}%)")
    logger.info(f"   Philosophical: {stats['philosophical']} ({stats['philosophical']/original_count*100:.1f}%)")
    logger.info(f"   Opinion: {stats['opinion']} ({stats['opinion']/original_count*100:.1f}%)")
    logger.info(f"   Recommendation: {stats['recommendation']} ({stats['recommendation']/original_count*100:.1f}%)")

    return stats


def main():
    project_root = Path("/home/claudeuser/yonearth-gaia-chatbot")
    graph_dir = project_root / "data/knowledge_graph_unified"

    input_file = graph_dir / "unified_normalized.json"
    output_file = graph_dir / "unified_normalized.json"  # Overwrite in place
    backup_file = graph_dir / "backups" / f"unified_normalized_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # Ensure backup directory exists
    backup_file.parent.mkdir(exist_ok=True, parents=True)

    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return

    logger.info("="*60)
    logger.info("Adding classification_flags to unified knowledge graph")
    logger.info("="*60)

    stats = add_classification_flags_to_unified_graph(
        input_file=input_file,
        output_file=output_file,
        backup_file=backup_file
    )

    logger.info("="*60)


if __name__ == "__main__":
    main()
