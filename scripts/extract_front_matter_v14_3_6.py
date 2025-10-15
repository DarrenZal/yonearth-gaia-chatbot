#!/usr/bin/env python3
"""
Quick V14.3.6 Front Matter Extraction

Uses V14.3.6 pipeline with:
- BibliographicCitationParser v1.6.0 (new object creation)
- ListSplitter v1.6.0 (min 3 words + safe_mode)
"""

import sys
import json
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.knowledge_graph.postprocessing.pipelines.book_pipeline import get_book_pipeline_v1436
from src.knowledge_graph.postprocessing import ProcessingContext

def main():
    """Run V14.3.6 pipeline on front matter extraction"""
    
    # Input: Front Matter from most recent V14.3.3 extraction
    input_file = Path("kg_extraction_playbook/output/our_biggest_deal/v14_3_3/chapters/front_matter_v14_3_3_20251015_051354.json")
    
    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        sys.exit(1)
    
    print("="*80)
    print("🔧 V14.3.6 POSTPROCESSING - FRONT MATTER")
    print("="*80)
    print(f"Input: {input_file.name}")
    print(f"Pipeline: v14.3.6 (BibCitParser v1.6.0 + ListSplitter v1.6.0)")
    print("="*80)
    print()
    
    # Load extraction
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    relationships = data['relationships']
    print(f"✅ Loaded {len(relationships)} relationships")
    print()
    
    # Convert to ModuleRelationship format (import from incremental script)
    from scripts.extract_kg_v14_3_3_incremental import ModuleRelationship
    
    rels = [ModuleRelationship.from_dict(r) for r in relationships]
    print(f"✅ Converted to ModuleRelationship format")
    print()
    
    # Create context
    context = ProcessingContext(
        content_type='book',
        document_metadata={
            'author': 'Aaron William Perry',
            'title': 'Our Biggest Deal',
            'section': 'front_matter',
            'pages': '1-30'
        },
        pages_with_text=[],
        run_id='v14_3_6_test',
        extraction_version='v14_3_6'
    )
    
    # Run V14.3.6 pipeline
    print("🔧 Running V14.3.6 pipeline...")
    pipeline = get_book_pipeline_v1436()
    processed, stats = pipeline.run(rels, context)
    
    print()
    print("="*80)
    print("✅ V14.3.6 PROCESSING COMPLETE")
    print("="*80)
    print(f"Relationships: {len(rels)} → {len(processed)}")
    print()
    print("Module Statistics:")
    for module, module_stats in stats.items():
        # Skip the initial_count and final_count metadata
        if module in ['initial_count', 'final_count', 'execution_order']:
            continue
        print(f"  {module}: {module_stats}")
    print("="*80)
    
    # Save results
    output_file = Path("kg_extraction_playbook/output/our_biggest_deal/v14_3_3/chapters/front_matter_v14_3_6_postprocessed.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        'metadata': data.get('metadata', {}),
        'metadata': {**data.get('metadata', {}), 'pipeline_version': 'v14_3_6'},
        'relationships': [r.to_dict() for r in processed],
        'postprocessing_stats': stats
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n💾 Saved to: {output_file}")
    print()
    print("NEXT STEP: Run Reflector to check quality")
    print("="*80)

if __name__ == "__main__":
    main()
