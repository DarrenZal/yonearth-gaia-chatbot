#!/usr/bin/env python3
"""
Integrate 4 cleaned books into unified knowledge graph.

This script:
1. Loads 4 cleaned book files (7,421 relationships)
2. Adds classification_flags to each book relationship
3. Converts to unified graph format
4. Merges with existing unified_normalized.json (172 episodes, 43,297 rels)
5. Creates final unified graph with episodes + books

Usage:
    python scripts/integrate_books_into_unified_graph.py
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BookRelationshipClassifier:
    """Classifier for book relationships (same logic as unified graph classifier)"""

    def __init__(self):
        self.factual_predicates = {
            'authored', 'wrote', 'published', 'founded', 'located', 'contains',
            'born', 'died', 'established', 'created', 'produced', 'released',
            'named', 'called', 'titled', 'dated', 'measured', 'counted',
            'discovered', 'invented', 'built', 'constructed', 'designed',
            'has_part', 'part_of', 'member_of', 'employed_by', 'works_for',
            'educated_at', 'graduated_from', 'awarded', 'received',
            'published_in', 'appeared_in', 'cited_in', 'referenced_in',
            'is', 'are', 'was', 'were', 'has', 'have', 'had',
            'mentions', 'discusses', 'describes', 'defines', 'explains'
        }

        self.philosophical_predicates = {
            'represents', 'symbolizes', 'embodies', 'reflects', 'signifies',
            'manifests', 'expresses', 'conveys', 'demonstrates', 'illustrates',
            'reveals', 'suggests', 'implies', 'indicates', 'means'
        }

        self.opinion_patterns = [
            r'\bbelieves?\b', r'\bthinks?\b', r'\bfeels?\b', r'\bclaims?\b',
            r'\bargues?\b', r'\bcontends?\b', r'\bmaintains?\b', r'\basserts?\b'
        ]

        self.recommendation_patterns = [
            r'\bshould\b', r'\bmust\b', r'\bshall\b', r'\bought to\b',
            r'\bneed to\b', r'\bhave to\b', r'\brecommends?\b', r'\badvises?\b'
        ]

    def classify(self, rel: Dict[str, Any]) -> List[str]:
        """Classify book relationship"""
        classifications = set()

        predicate = (rel.get('relationship_type') or '').lower().strip().replace('_', ' ')

        if predicate in self.factual_predicates:
            classifications.add('factual')
        if predicate in self.philosophical_predicates:
            classifications.add('philosophical')

        description = rel.get('description', '') or ''
        if description:
            desc_lower = description.lower()
            for pattern in self.opinion_patterns:
                if re.search(pattern, desc_lower):
                    classifications.add('opinion')
                    break
            for pattern in self.recommendation_patterns:
                if re.search(pattern, desc_lower):
                    classifications.add('recommendation')
                    break

        if not classifications:
            source = rel.get('source_entity', '').lower()
            target = rel.get('target_entity', '').lower()
            abstract_indicators = ['essence', 'nature', 'spirit', 'soul', 'energy', 'consciousness']
            if any(ind in source or ind in target for ind in abstract_indicators):
                classifications.add('philosophical')
            else:
                classifications.add('factual')

        return sorted(list(classifications))


def load_and_process_book(
    book_file: Path,
    book_slug: str,
    classifier: BookRelationshipClassifier
) -> tuple[List[Dict], Dict[str, int]]:
    """
    Load book file, add classification_flags, convert to unified format.

    Returns:
        (relationships_list, stats_dict)
    """
    logger.info(f"  Loading: {book_file.name}")

    with open(book_file, 'r') as f:
        data = json.load(f)

    book_title = data.get('title', 'Unknown')
    raw_rels = data.get('relationships', [])

    logger.info(f"    Title: {book_title}")
    logger.info(f"    Relationships: {len(raw_rels)}")

    # Stats
    stats = {'factual': 0, 'philosophical': 0, 'opinion': 0, 'recommendation': 0}

    # Convert to unified format
    unified_rels = []
    for rel in raw_rels:
        # Classify
        flags = classifier.classify(rel)

        # Update stats
        for flag in flags:
            if flag in stats:
                stats[flag] += 1

        # Convert to unified format
        unified_rel = {
            'id': f"{book_slug}_{len(unified_rels)}",
            'source': rel.get('source_entity', ''),
            'target': rel.get('target_entity', ''),
            'predicate': rel.get('relationship_type', '').lower().replace(' ', '_'),
            'original_predicate': rel.get('relationship_type', ''),
            'confidence': 1.0,
            'evidence': {
                'text': rel.get('description', ''),
                'book_slug': book_slug,
                'chunk_id': rel.get('metadata', {}).get('chunk_id')
            },
            'source_type': 'UNKNOWN',
            'target_type': 'UNKNOWN',
            'metadata': {
                'book_slug': book_slug,
                'book_title': book_title,
                'source': 'book_extraction'
            },
            'classification_flags': flags
        }

        unified_rels.append(unified_rel)

    logger.info(f"    Classifications: factual={stats['factual']}, philosophical={stats['philosophical']}, "
                f"opinion={stats['opinion']}, recommendation={stats['recommendation']}")

    return unified_rels, stats


def integrate_books_into_unified_graph():
    """Main integration function"""

    project_root = Path("/home/claudeuser/yonearth-gaia-chatbot")
    books_dir = project_root / "data/knowledge_graph/books"
    graph_dir = project_root / "data/knowledge_graph_unified"

    # Book files
    book_configs = [
        {
            'file': books_dir / "veriditas_ace_v14_3_8_cleaned.json",
            'slug': 'veriditas'
        },
        {
            'file': books_dir / "soil-stewardship-handbook_ace_v14_3_8_cleaned.json",
            'slug': 'soil-stewardship-handbook'
        },
        {
            'file': books_dir / "y-on-earth_ace_v14_3_8_cleaned.json",
            'slug': 'y-on-earth'
        },
        {
            'file': books_dir / "OurBiggestDeal_ace_v14_3_8_cleaned.json",
            'slug': 'our-biggest-deal'
        }
    ]

    # Load existing unified graph
    unified_file = graph_dir / "unified_normalized.json"
    backup_file = graph_dir / "backups" / f"unified_before_books_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    logger.info("="*60)
    logger.info("INTEGRATING BOOKS INTO UNIFIED GRAPH")
    logger.info("="*60)

    # Backup
    backup_file.parent.mkdir(exist_ok=True, parents=True)
    shutil.copy(unified_file, backup_file)
    logger.info(f"✅ Backup saved: {backup_file.name}")

    # Load unified graph
    logger.info(f"\nLoading unified graph: {unified_file.name}")
    with open(unified_file, 'r') as f:
        unified_data = json.load(f)

    original_rel_count = len(unified_data.get('relationships', []))
    logger.info(f"  Current: {len(unified_data.get('entities', {}))} entities, {original_rel_count} relationships")

    # Process all books
    logger.info(f"\nProcessing {len(book_configs)} books:")

    classifier = BookRelationshipClassifier()
    all_book_rels = []
    total_stats = {'factual': 0, 'philosophical': 0, 'opinion': 0, 'recommendation': 0}

    for config in book_configs:
        book_rels, stats = load_and_process_book(
            book_file=config['file'],
            book_slug=config['slug'],
            classifier=classifier
        )

        all_book_rels.extend(book_rels)

        # Accumulate stats
        for key in total_stats:
            total_stats[key] += stats[key]

    logger.info(f"\n✅ Processed {len(book_configs)} books:")
    logger.info(f"  Total book relationships: {len(all_book_rels)}")
    logger.info(f"  Classifications: factual={total_stats['factual']}, philosophical={total_stats['philosophical']}, "
                f"opinion={total_stats['opinion']}, recommendation={total_stats['recommendation']}")

    # Merge with unified graph
    logger.info(f"\nMerging with unified graph...")

    unified_data['relationships'].extend(all_book_rels)

    new_rel_count = len(unified_data['relationships'])
    logger.info(f"  Before: {original_rel_count} relationships")
    logger.info(f"  After:  {new_rel_count} relationships (+{new_rel_count - original_rel_count})")

    # Update metadata
    if 'classification_metadata' in unified_data:
        unified_data['classification_metadata']['books_integrated'] = True
        unified_data['classification_metadata']['books_integrated_date'] = datetime.now().isoformat()
        unified_data['classification_metadata']['book_count'] = len(book_configs)
        unified_data['classification_metadata']['book_relationships'] = len(all_book_rels)

    # Save updated graph
    logger.info(f"\nSaving updated unified graph...")
    with open(unified_file, 'w') as f:
        json.dump(unified_data, f, indent=2)

    logger.info(f"✅ Saved: {unified_file}")

    logger.info("\n" + "="*60)
    logger.info("INTEGRATION COMPLETE")
    logger.info("="*60)
    logger.info(f"Final unified graph:")
    logger.info(f"  Entities: {len(unified_data.get('entities', {}))}")
    logger.info(f"  Relationships: {new_rel_count}")
    logger.info(f"  Episodes: 172")
    logger.info(f"  Books: {len(book_configs)}")
    logger.info("="*60)


if __name__ == "__main__":
    integrate_books_into_unified_graph()
