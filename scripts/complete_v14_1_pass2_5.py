#!/usr/bin/env python3
"""
Complete V14.1 extraction from Pass 2 checkpoint.
Runs only Pass 2.5 (15 postprocessing modules) to finish the extraction.
"""

import json
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.knowledge_graph.postprocessing import ProcessingContext
from src.knowledge_graph.postprocessing.pipelines import get_book_pipeline

# Import ModuleRelationship for reconstruction
sys.path.append(str(Path(__file__).parent))
from extract_kg_v14_1_book import ModuleRelationship

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
PLAYBOOK_DIR = Path(__file__).parent.parent / "kg_extraction_playbook"
OUTPUT_DIR = PLAYBOOK_DIR / "output" / "v14_1"
CHECKPOINT_FILE = OUTPUT_DIR / "book_soil_handbook_v14_1_20251014_200456_pass2_checkpoint.json"

logger.info("🚀 V14.1 Pass 2.5 Completion Script")
logger.info(f"Loading checkpoint from: {CHECKPOINT_FILE.name}")

# Load Pass 2 checkpoint
with open(CHECKPOINT_FILE) as f:
    pass2_data = json.load(f)

logger.info(f"✅ Loaded {len(pass2_data)} evaluated relationships")

# Reconstruct ModuleRelationship objects
relationships = [ModuleRelationship.from_dict(rel) for rel in pass2_data]

# Create processing context
context = ProcessingContext(
    content_type='book',
    document_metadata={'title': 'Soil Stewardship Handbook', 'author': 'Aaron Perry'},
    pages_with_text=[],
    run_id='book_soil_handbook_v14_1_20251014_200456',
    extraction_version='v14_1'
)

# Run Pass 2.5
logger.info("🔧 Running Pass 2.5 with FIXED SemanticDeduplicator...")
pipeline = get_book_pipeline()
final_relationships, pp_stats = pipeline.run(relationships, context)

logger.info(f"✅ Pass 2.5 complete: {len(final_relationships)} final relationships")

# Save final results
output_file = OUTPUT_DIR / "soil_stewardship_handbook_v14_1.json"
results = {
    'metadata': {
        'extraction_version': 'v14.1',
        'book_title': 'Soil Stewardship Handbook',
        'extraction_date': datetime.now().isoformat()
    },
    'extraction_stats': {
        'pass2_evaluated': len(pass2_data),
        'pass2_5_final': len(final_relationships)
    },
    'postprocessing_stats': pp_stats,
    'relationships': [rel.to_dict() for rel in final_relationships]
}

with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

logger.info(f"💾 Results saved to: {output_file.name}")
logger.info(f"📊 Final count: {len(final_relationships)} relationships")
