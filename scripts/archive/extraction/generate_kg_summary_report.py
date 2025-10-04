#!/usr/bin/env python3
"""
Generate Knowledge Graph Extraction Summary Report

Analyzes all extracted episodes 0-43 and generates a comprehensive report.
"""

import json
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

project_root = Path(__file__).parent.parent
entities_dir = project_root / "data" / "knowledge_graph" / "entities"

def generate_summary_report(start=0, end=43):
    """Generate comprehensive summary report"""

    # Collect data
    all_entities = []
    all_relationships = []
    episode_stats = []
    entity_type_counts = Counter()
    relationship_type_counts = Counter()
    failed_episodes = []

    for ep_num in range(start, end + 1):
        file_path = entities_dir / f"episode_{ep_num}_extraction.json"

        if file_path.exists():
            try:
                with open(file_path) as f:
                    data = json.load(f)

                # Episode stats
                episode_stats.append({
                    'episode_number': ep_num,
                    'title': data.get('episode_title', ''),
                    'guest': data.get('guest_name', ''),
                    'chunks': data.get('total_chunks', 0),
                    'entities': data.get('summary_stats', {}).get('total_entities', 0),
                    'relationships': data.get('summary_stats', {}).get('total_relationships', 0)
                })

                # Collect entities
                for entity in data.get('entities', []):
                    all_entities.append(entity)
                    entity_type_counts[entity.get('type', 'UNKNOWN')] += 1

                # Collect relationships
                for rel in data.get('relationships', []):
                    all_relationships.append(rel)
                    relationship_type_counts[rel.get('relationship_type', 'UNKNOWN')] += 1

            except Exception as e:
                failed_episodes.append((ep_num, str(e)))

    # Generate report
    report = []
    report.append("=" * 80)
    report.append("KNOWLEDGE GRAPH EXTRACTION SUMMARY REPORT")
    report.append(f"Episodes {start}-{end}")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 80)
    report.append("")

    # Overall Statistics
    report.append("OVERALL STATISTICS")
    report.append("-" * 80)
    report.append(f"Total Episodes Processed: {len(episode_stats)}")
    report.append(f"Total Episodes Failed: {len(failed_episodes)}")
    report.append(f"Total Chunks Processed: {sum(e['chunks'] for e in episode_stats)}")
    report.append(f"Total Entities Extracted: {len(all_entities)}")
    report.append(f"Total Relationships Extracted: {len(all_relationships)}")
    report.append("")

    # Entity Type Distribution
    report.append("ENTITY TYPE DISTRIBUTION")
    report.append("-" * 80)
    for entity_type, count in sorted(entity_type_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(all_entities) * 100) if all_entities else 0
        report.append(f"{entity_type:20s}: {count:5d} ({percentage:5.1f}%)")
    report.append("")

    # Relationship Type Distribution
    report.append("RELATIONSHIP TYPE DISTRIBUTION")
    report.append("-" * 80)
    for rel_type, count in sorted(relationship_type_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(all_relationships) * 100) if all_relationships else 0
        report.append(f"{rel_type:20s}: {count:5d} ({percentage:5.1f}%)")
    report.append("")

    # Top Episodes by Extraction
    report.append("TOP 10 EPISODES BY ENTITIES EXTRACTED")
    report.append("-" * 80)
    top_entities = sorted(episode_stats, key=lambda x: x['entities'], reverse=True)[:10]
    for ep in top_entities:
        report.append(
            f"Episode {ep['episode_number']:3d}: {ep['entities']:3d} entities, "
            f"{ep['relationships']:3d} rels - {ep['title'][:50]}"
        )
    report.append("")

    # Failed Episodes
    if failed_episodes:
        report.append("FAILED EPISODES")
        report.append("-" * 80)
        for ep_num, error in failed_episodes:
            report.append(f"Episode {ep_num}: {error}")
        report.append("")

    # Sample Entities
    report.append("SAMPLE ENTITIES (First 20)")
    report.append("-" * 80)
    for i, entity in enumerate(all_entities[:20], 1):
        report.append(f"{i:2d}. {entity.get('name', 'Unknown'):30s} ({entity.get('type', 'UNKNOWN')})")
        desc = entity.get('description', '')[:70]
        if desc:
            report.append(f"    {desc}")
    report.append("")

    # Sample Relationships
    report.append("SAMPLE RELATIONSHIPS (First 20)")
    report.append("-" * 80)
    for i, rel in enumerate(all_relationships[:20], 1):
        report.append(
            f"{i:2d}. {rel.get('source_entity', 'Unknown')} --[{rel.get('relationship_type', 'UNKNOWN')}]--> "
            f"{rel.get('target_entity', 'Unknown')}"
        )
        desc = rel.get('description', '')[:70]
        if desc:
            report.append(f"    {desc}")
    report.append("")

    report.append("=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)

    # Print report
    report_text = "\n".join(report)
    print(report_text)

    # Save report
    report_file = entities_dir / f"extraction_report_{start}_{end}.txt"
    with open(report_file, 'w') as f:
        f.write(report_text)

    print(f"\nReport saved to: {report_file}")

    # Save JSON summary
    summary = {
        "generated_at": datetime.now().isoformat(),
        "episode_range": f"{start}-{end}",
        "total_episodes_processed": len(episode_stats),
        "total_episodes_failed": len(failed_episodes),
        "total_chunks": sum(e['chunks'] for e in episode_stats),
        "total_entities": len(all_entities),
        "total_relationships": len(all_relationships),
        "entity_type_distribution": dict(entity_type_counts),
        "relationship_type_distribution": dict(relationship_type_counts),
        "episode_stats": episode_stats,
        "failed_episodes": failed_episodes
    }

    summary_file = entities_dir / f"extraction_summary_{start}_{end}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"JSON summary saved to: {summary_file}")

    return summary

if __name__ == "__main__":
    generate_summary_report(0, 43)
