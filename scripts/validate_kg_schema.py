#!/usr/bin/env python3
"""Validate knowledge graph data schema before deployment."""

import json
from pathlib import Path

def validate_schema():
    """Check actual data formats match plan assumptions."""

    print("🔍 Validating Knowledge Graph Data Schema")
    print("=" * 50)

    # Load unified graph
    unified_path = Path("data/knowledge_graph_unified/unified_v2.json")
    with open(unified_path) as f:
        unified = json.load(f)

    # Load relationships
    rel_path = Path("data/knowledge_graph_unified/relationships_processed.json")
    with open(rel_path) as f:
        relationships = json.load(f)

    # Check entity sources format
    print("\n1. Entity Sources Format:")
    sample_entity = list(unified['entities'].values())[0]
    sample_sources = sample_entity.get('sources', [])
    print(f"   Sample entity: {list(unified['entities'].keys())[0]}")
    print(f"   Sources format: {sample_sources[:3]}")

    # Verify episode format
    episode_sources = [s for s in sample_sources if 'episode' in s.lower()]
    if episode_sources:
        print(f"   Episode format: {episode_sources[0]}")
        if episode_sources[0].startswith('episode_'):
            print("   ✓ Format matches plan assumption (episode_N)")
        else:
            print("   ⚠️  Format differs from plan! Adjust filter logic.")

    # Check relationship source_id format
    print("\n2. Relationship source_id Format:")
    sample_rel = relationships[0]
    source_id = sample_rel.get('source_id')
    print(f"   Sample relationship: {sample_rel.get('source')} → {sample_rel.get('target')}")
    print(f"   source_id format: '{source_id}' (type: {type(source_id).__name__})")

    if isinstance(source_id, str) and source_id.isdigit():
        print("   ✓ Format matches plan assumption (string number like '5')")
    elif isinstance(source_id, str) and source_id.startswith('episode_'):
        print("   ⚠️  Format is 'episode_N', adjust filter to match")
    else:
        print(f"   ⚠️  Unexpected format! Review filter logic.")

    # Test Episode 5 filtering
    print("\n3. Test Filtering for Episode 5:")

    # Count entities
    episode_5_entities = [
        name for name, data in unified['entities'].items()
        if 'episode_5' in data.get('sources', [])
    ]
    print(f"   Entities in Episode 5: {len(episode_5_entities)}")
    if episode_5_entities:
        print(f"   Sample entities: {episode_5_entities[:5]}")

    # Count relationships
    episode_5_rels = [
        r for r in relationships
        if r.get('source_id') == '5' or r.get('source_id') == 'episode_5'
    ]
    print(f"   Relationships in Episode 5: {len(episode_5_rels)}")
    if episode_5_rels:
        sample_rel = episode_5_rels[0]
        print(f"   Sample: {sample_rel.get('source')} → {sample_rel.get('predicate')} → {sample_rel.get('target')}")

    # Check entity_chunk_map
    print("\n4. Entity-Chunk Map Format:")
    chunk_map_path = Path("data/graph_index/entity_chunk_map.json")
    if chunk_map_path.exists():
        with open(chunk_map_path) as f:
            chunk_map = json.load(f)

        sample_entity_name = list(chunk_map.keys())[0]
        sample_chunks = chunk_map[sample_entity_name]
        print(f"   Sample entity: {sample_entity_name}")
        print(f"   Chunk format: {sample_chunks[:2]}")

        # Check for episode chunk format
        episode_chunks = [c for c in sample_chunks if 'episode' in c.lower()]
        if episode_chunks:
            print(f"   Episode chunk format: {episode_chunks[0]}")
            if episode_chunks[0].startswith('episode_'):
                print("   ✓ Format matches plan assumption (episode_N_chunk_...)")
            else:
                print("   ⚠️  Unexpected chunk ID format")
    else:
        print("   ⚠️  entity_chunk_map.json not found!")

    print("\n" + "=" * 50)
    print("✅ Schema validation complete")
    print("\nIf any warnings appeared, update filter logic in graph_endpoints.py")

if __name__ == "__main__":
    validate_schema()
