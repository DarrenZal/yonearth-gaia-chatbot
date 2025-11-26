"""
Normalize entity types to canonical uppercase form.

This fixes the case-sensitivity issue that prevented proper entity deduplication.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT = Path("/home/claudeuser/yonearth-gaia-chatbot")
UNIFIED_PATH = ROOT / "data/knowledge_graph_unified/unified_normalized.json"
BACKUP_PATH = ROOT / "data/knowledge_graph_unified" / f"unified_normalized_backup_pre_normalize_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"


# Type normalization mapping
TYPE_NORMALIZATION = {
    # Concept/abstract
    'concept': 'CONCEPT',
    'Concept': 'CONCEPT',
    'CONCEPT': 'CONCEPT',
    'idea': 'CONCEPT',
    'Idea': 'CONCEPT',

    # People
    'person': 'PERSON',
    'Person': 'PERSON',
    'PERSON': 'PERSON',
    'individual': 'PERSON',
    'Individual': 'PERSON',
    'human': 'PERSON',
    'Human': 'PERSON',

    # Organizations
    'organization': 'ORGANIZATION',
    'Organization': 'ORGANIZATION',
    'ORGANIZATION': 'ORGANIZATION',
    'company': 'ORGANIZATION',
    'Company': 'ORGANIZATION',
    'business': 'ORGANIZATION',
    'Business': 'ORGANIZATION',
    'group': 'ORGANIZATION',
    'Group': 'ORGANIZATION',

    # Locations
    'place': 'PLACE',
    'Place': 'PLACE',
    'PLACE': 'PLACE',
    'location': 'PLACE',
    'Location': 'PLACE',
    'region': 'PLACE',
    'Region': 'PLACE',

    # Products
    'product': 'PRODUCT',
    'Product': 'PRODUCT',
    'PRODUCT': 'PRODUCT',

    # Events
    'event': 'EVENT',
    'Event': 'EVENT',
    'EVENT': 'EVENT',

    # Practices/Activities
    'practice': 'PRACTICE',
    'Practice': 'PRACTICE',
    'PRACTICE': 'PRACTICE',
    'activity': 'PRACTICE',
    'Activity': 'PRACTICE',
    'action': 'PRACTICE',
    'Action': 'PRACTICE',

    # Species/organisms
    'species': 'SPECIES',
    'Species': 'SPECIES',
    'SPECIES': 'SPECIES',
    'plant': 'SPECIES',
    'Plant': 'SPECIES',
    'animal': 'SPECIES',
    'Animal': 'SPECIES',

    # Technology
    'technology': 'TECHNOLOGY',
    'Technology': 'TECHNOLOGY',
    'TECHNOLOGY': 'TECHNOLOGY',
    'website': 'TECHNOLOGY',
    'Website': 'TECHNOLOGY',

    # Ecosystem/environment
    'ecosystem': 'ECOSYSTEM',
    'Ecosystem': 'ECOSYSTEM',
    'ECOSYSTEM': 'ECOSYSTEM',

    # Generic fallback
    'string': 'CONCEPT',
    'entity': 'CONCEPT',
    'object': 'CONCEPT',
    'noun': 'CONCEPT',
}


def normalize_types(data):
    """Normalize all entity types to canonical uppercase form."""
    logger.info("Normalizing entity types...")

    entities = data.get('entities', {})
    normalized_count = 0
    type_changes = {}

    for entity_id, entity in entities.items():
        old_type = entity.get('type', 'UNKNOWN')
        new_type = TYPE_NORMALIZATION.get(old_type, old_type.upper())

        if old_type != new_type:
            entity['type'] = new_type
            normalized_count += 1
            type_changes[old_type] = type_changes.get(old_type, 0) + 1

    logger.info(f"✓ Normalized {normalized_count} entity types")
    logger.info(f"  Type changes:")
    for old_type, count in sorted(type_changes.items(), key=lambda x: -x[1])[:20]:
        new_type = TYPE_NORMALIZATION.get(old_type, old_type.upper())
        logger.info(f"    {old_type} → {new_type}: {count} entities")

    return data


def main():
    logger.info("=" * 80)
    logger.info("Normalize Entity Types")
    logger.info("=" * 80)

    # Load
    logger.info(f"Loading: {UNIFIED_PATH}")
    with open(UNIFIED_PATH, 'r') as f:
        data = json.load(f)

    logger.info(f"  Entities: {len(data.get('entities', {}))}")
    logger.info(f"  Relationships: {len(data.get('relationships', []))}")

    # Backup
    logger.info(f"Creating backup: {BACKUP_PATH}")
    shutil.copy2(UNIFIED_PATH, BACKUP_PATH)

    # Normalize
    data = normalize_types(data)

    # Save
    logger.info(f"Saving normalized graph: {UNIFIED_PATH}")
    with open(UNIFIED_PATH, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info("=" * 80)
    logger.info("✅ Complete!")
    logger.info("=" * 80)
    logger.info("Next steps:")
    logger.info("  1. Re-run deduplication: python3 scripts/build_unified_graph_hybrid.py")
    logger.info("  2. Regenerate discourse graph")
    logger.info("  3. Regenerate GraphRAG hierarchy")


if __name__ == "__main__":
    main()
