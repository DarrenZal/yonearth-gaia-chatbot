#!/usr/bin/env python3
"""
Compile statistics from episodes 132-172 extraction results
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict


def compile_statistics(entities_dir: str, start_episode: int = 132, end_episode: int = 172):
    """Compile statistics from extraction results"""
    
    entities_path = Path(entities_dir)
    
    # Initialize statistics
    stats = {
        "total_episodes_processed": 0,
        "total_entities": 0,
        "total_relationships": 0,
        "total_chunks": 0,
        "entity_types": defaultdict(int),
        "relationship_types": defaultdict(int),
        "episodes_processed": [],
        "episodes_missing": [],
        "entity_names": defaultdict(int),  # Track entity frequency
        "top_entities": [],
        "top_relationships": []
    }
    
    # Process each episode
    for episode_num in range(start_episode, end_episode + 1):
        extraction_file = entities_path / f"episode_{episode_num}_extraction.json"
        
        if not extraction_file.exists():
            stats["episodes_missing"].append(episode_num)
            continue
        
        try:
            with open(extraction_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            stats["episodes_processed"].append(episode_num)
            stats["total_episodes_processed"] += 1
            stats["total_chunks"] += data.get("chunks_processed", 0)
            
            # Count entities
            entities = data.get("entities", [])
            stats["total_entities"] += len(entities)
            
            for entity in entities:
                entity_type = entity.get("type", "UNKNOWN")
                stats["entity_types"][entity_type] += 1
                
                entity_name = entity.get("name", "Unknown")
                stats["entity_names"][entity_name] += 1
            
            # Count relationships
            relationships = data.get("relationships", [])
            stats["total_relationships"] += len(relationships)
            
            for rel in relationships:
                rel_type = rel.get("relationship_type", "UNKNOWN")
                stats["relationship_types"][rel_type] += 1
        
        except Exception as e:
            print(f"Error processing episode {episode_num}: {e}")
            continue
    
    # Calculate top entities
    stats["top_entities"] = sorted(
        stats["entity_names"].items(),
        key=lambda x: x[1],
        reverse=True
    )[:20]
    
    # Convert defaultdicts to regular dicts for JSON serialization
    stats["entity_types"] = dict(sorted(
        stats["entity_types"].items(),
        key=lambda x: x[1],
        reverse=True
    ))
    stats["relationship_types"] = dict(sorted(
        stats["relationship_types"].items(),
        key=lambda x: x[1],
        reverse=True
    ))
    
    # Remove the large entity_names dict before saving
    del stats["entity_names"]
    
    return stats


def print_report(stats: Dict[str, Any]):
    """Print a formatted statistics report"""
    
    print("\n" + "="*70)
    print("KNOWLEDGE GRAPH EXTRACTION STATISTICS")
    print("Episodes 132-172")
    print("="*70)
    
    print(f"\n📊 OVERVIEW")
    print(f"{'─'*70}")
    print(f"Episodes processed:       {stats['total_episodes_processed']}/41")
    print(f"Total chunks processed:   {stats['total_chunks']:,}")
    print(f"Total entities extracted: {stats['total_entities']:,}")
    print(f"Total relationships:      {stats['total_relationships']:,}")
    
    if stats["episodes_missing"]:
        missing_count = len(stats["episodes_missing"])
        print(f"\n⏳ Episodes pending:        {missing_count}")
        if missing_count <= 10:
            print(f"   Pending: {', '.join(map(str, stats['episodes_missing']))}")
        else:
            first_few = ', '.join(map(str, stats["episodes_missing"][:5]))
            print(f"   First pending: {first_few}, ...")
    
    if stats["episodes_processed"]:
        processed_list = stats["episodes_processed"]
        if len(processed_list) <= 10:
            print(f"\n✓ Completed: {', '.join(map(str, processed_list))}")
        else:
            first_few = ', '.join(map(str, processed_list[:5]))
            last_few = ', '.join(map(str, processed_list[-5:]))
            print(f"\n✓ Completed: {first_few} ... {last_few}")
    
    print(f"\n🏷️  ENTITY TYPES")
    print(f"{'─'*70}")
    for entity_type, count in list(stats["entity_types"].items())[:10]:
        percentage = (count / stats["total_entities"] * 100) if stats["total_entities"] > 0 else 0
        print(f"  {entity_type:20s}: {count:5d} ({percentage:5.1f}%)")
    
    print(f"\n🔗 RELATIONSHIP TYPES")
    print(f"{'─'*70}")
    for rel_type, count in list(stats["relationship_types"].items())[:10]:
        percentage = (count / stats["total_relationships"] * 100) if stats["total_relationships"] > 0 else 0
        print(f"  {rel_type:20s}: {count:5d} ({percentage:5.1f}%)")
    
    print(f"\n⭐ TOP ENTITIES (by frequency)")
    print(f"{'─'*70}")
    for entity_name, count in stats["top_entities"][:15]:
        print(f"  {entity_name:40s}: {count:3d} mentions")
    
    print(f"\n{'='*70}\n")


def main():
    entities_dir = "/home/claudeuser/yonearth-gaia-chatbot/data/knowledge_graph/entities"
    
    print("Compiling statistics for episodes 132-172...")
    stats = compile_statistics(entities_dir)
    
    # Print report
    print_report(stats)
    
    # Save statistics to JSON
    output_file = Path(entities_dir) / "extraction_statistics_132_172.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"Statistics saved to: {output_file}")
    
    # Save text report
    report_file = Path(entities_dir) / "extraction_report_132_172.txt"
    # Redirect print to file (simplified approach)
    import sys
    original_stdout = sys.stdout
    with open(report_file, 'w', encoding='utf-8') as f:
        sys.stdout = f
        print_report(stats)
    sys.stdout = original_stdout
    
    print(f"Report saved to: {report_file}")


if __name__ == "__main__":
    main()
