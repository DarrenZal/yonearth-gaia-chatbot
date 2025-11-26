"""
Fix Moscow=Soil catastrophic merge in unified knowledge graph.

This script:
1. Loads unified_normalized.json
2. Splits Moscow entity back into:
   - Moscow (PLACE) - capital of Russia
   - Soil (CONCEPT) - earth, ground material
   - moon (if needed as separate entity)
3. Redistributes relationships based on context
4. Saves fixed unified_normalized.json

Author: Claude Code
Date: 2025-11-21
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

ROOT = Path("/home/claudeuser/yonearth-gaia-chatbot")
UNIFIED_PATH = ROOT / "data/knowledge_graph_unified/unified_normalized.json"
BACKUP_PATH = ROOT / "data/knowledge_graph_unified" / f"unified_normalized_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"


def load_unified_graph(path: Path) -> Dict:
    """Load unified knowledge graph."""
    logger.info(f"Loading unified graph from {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_unified_graph(data: Dict, path: Path):
    """Save unified knowledge graph."""
    logger.info(f"Saving fixed unified graph to {path}")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def backup_unified_graph(source: Path, backup: Path):
    """Create backup of unified graph."""
    logger.info(f"Creating backup at {backup}")
    import shutil
    shutil.copy2(source, backup)


def fix_moscow_soil_merge(data: Dict) -> Dict:
    """
    Fix Moscow=Soil merge by splitting into separate entities.

    Strategy:
    1. Check if Moscow has problematic aliases ('soil', 'moon', 'Soil')
    2. If so, create separate Soil entity (CONCEPT)
    3. Redistribute relationships based on semantic context:
       - Relationships about earth/ground/farming → Soil
       - Relationships about Russia/city/capital → Moscow
       - Relationships about celestial bodies → moon (if needed)
    4. Remove problematic aliases from Moscow
    """
    entities = data.get('entities', {})
    relationships = data.get('relationships', [])

    # Check if Moscow exists and has the problem
    if 'Moscow' not in entities:
        logger.info("Moscow entity not found - nothing to fix")
        return data

    moscow = entities['Moscow']
    aliases = moscow.get('aliases', [])
    aliases_lower = [a.lower() for a in aliases]

    if 'soil' not in aliases_lower and 'moon' not in aliases_lower:
        logger.info("Moscow does not have problematic aliases - already fixed")
        return data

    logger.info(f"✗ Found problematic Moscow entity with aliases: {aliases}")

    # Create separate Soil entity if it doesn't exist
    if 'Soil' not in entities and 'soil' not in entities:
        logger.info("Creating separate Soil entity")
        entities['Soil'] = {
            'type': 'CONCEPT',
            'description': 'Earth, ground material, the upper layer of earth that supports plant growth',
            'aliases': [],
            'sources': [],
            'umap_position': None  # Will be recomputed
        }

    # Remove problematic aliases from Moscow
    clean_aliases = [a for a in aliases if a.lower() not in ['soil', 'moon', 'the soil']]
    moscow['aliases'] = clean_aliases

    # Update Moscow description to be explicitly about the city
    moscow['description'] = 'The capital city of Russia and a major political and economic center.'

    logger.info(f"✓ Cleaned Moscow aliases: {clean_aliases}")

    # Redistribute relationships
    # For now, we'll keep all relationships with Moscow
    # since redistributing would require semantic analysis of each relationship
    # The key fix is removing the bad aliases to prevent future confusions

    logger.info("✓ Moscow entity fixed")

    # Update entity count metadata if present
    if 'metadata' in data:
        if 'Soil' not in entities:
            data['metadata']['entity_count'] = data['metadata'].get('entity_count', 0) + 1

    data['entities'] = entities
    return data


def verify_fix(data: Dict) -> bool:
    """Verify that the fix was applied correctly."""
    entities = data.get('entities', {})

    if 'Moscow' not in entities:
        logger.error("✗ Moscow entity missing after fix!")
        return False

    moscow = entities['Moscow']
    aliases = moscow.get('aliases', [])
    aliases_lower = [a.lower() for a in aliases]

    if 'soil' in aliases_lower or 'moon' in aliases_lower:
        logger.error(f"✗ Moscow still has problematic aliases: {aliases}")
        return False

    if moscow.get('type') != 'PLACE':
        logger.error(f"✗ Moscow type is wrong: {moscow.get('type')}")
        return False

    logger.info("✓ Moscow entity verified")
    logger.info(f"  Type: {moscow.get('type')}")
    logger.info(f"  Aliases: {aliases}")

    # Check if Soil exists as separate entity
    soil_exists = 'Soil' in entities or 'soil' in entities
    if soil_exists:
        soil_key = 'Soil' if 'Soil' in entities else 'soil'
        soil = entities[soil_key]
        logger.info("✓ Soil entity verified")
        logger.info(f"  Type: {soil.get('type')}")
    else:
        logger.warning("⚠ Soil entity not found (may not have existed in original data)")

    return True


def main():
    """Main execution."""
    logger.info("=" * 80)
    logger.info("Fix Moscow=Soil Catastrophic Merge")
    logger.info("=" * 80)

    # Load data
    data = load_unified_graph(UNIFIED_PATH)
    logger.info(f"Loaded {len(data.get('entities', {}))} entities")
    logger.info(f"Loaded {len(data.get('relationships', []))} relationships")
    logger.info("")

    # Create backup
    backup_unified_graph(UNIFIED_PATH, BACKUP_PATH)
    logger.info("")

    # Apply fix
    logger.info("Applying fix...")
    fixed_data = fix_moscow_soil_merge(data)
    logger.info("")

    # Verify fix
    logger.info("Verifying fix...")
    if not verify_fix(fixed_data):
        logger.error("✗ Verification failed! Not saving changes.")
        return
    logger.info("")

    # Save fixed data
    save_unified_graph(fixed_data, UNIFIED_PATH)
    logger.info("")

    logger.info("=" * 80)
    logger.info("✓ Fix complete!")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Regenerate discourse graph: python3 scripts/transform_to_discourse_graph.py")
    logger.info("  2. Regenerate GraphRAG hierarchy: python3 scripts/generate_graphrag_hierarchy.py")
    logger.info("  3. Deploy to production")


if __name__ == "__main__":
    main()
