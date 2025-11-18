#!/usr/bin/env python3
"""
Build Cross-Content Links between Books and Episodes

This script creates three types of edges:
1. mentioned_in: entity → [episode_ids, book_ids]
2. supports: assertion → assertion (confidence ≥0.7)
3. contradicts: assertion → assertion (opposite polarity)

Target: 5,000+ cross-content edges using entity_chunk_map for efficient lookup
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm
import hashlib

# Paths
PROJECT_ROOT = Path("/home/claudeuser/yonearth-gaia-chatbot")
UNIFIED_KG_PATH = PROJECT_ROOT / "data/knowledge_graph_unified/unified.json"
ADJACENCY_PATH = PROJECT_ROOT / "data/knowledge_graph_unified/adjacency.json"
DISCOURSE_PATH = PROJECT_ROOT / "data/knowledge_graph_unified/discourse.json"
EPISODE_DISCOURSE_PATH = PROJECT_ROOT / "data/knowledge_graph_unified/episode_discourse.json"
OUTPUT_DIR = PROJECT_ROOT / "data/knowledge_graph_unified"

# Output files
CROSS_LINKS_PATH = OUTPUT_DIR / "cross_content_links.json"
UPDATED_ADJACENCY_PATH = OUTPUT_DIR / "adjacency_with_cross_links.json"
CROSS_LINKS_STATS_PATH = OUTPUT_DIR / "cross_links_stats.json"

class CrossContentLinker:
    def __init__(self):
        self.entities = {}
        self.relationships = []
        self.adjacency = defaultdict(lambda: defaultdict(list))
        self.discourse = {}
        self.cross_links = {
            "mentioned_in": defaultdict(set),  # entity -> set of content_ids
            "supports": [],  # list of (assertion1, assertion2, confidence)
            "contradicts": []  # list of (assertion1, assertion2, confidence)
        }
        self.stats = {
            "mentioned_in_links": 0,
            "supports_links": 0,
            "contradicts_links": 0,
            "total_cross_links": 0,
            "entities_with_multi_source": 0,
            "timestamp": datetime.now().isoformat()
        }

    def load_knowledge_graph(self):
        """Load the unified knowledge graph"""
        print("Loading knowledge graph...")

        # Load unified graph
        with open(UNIFIED_KG_PATH, 'r') as f:
            data = json.load(f)
            self.entities = data['entities']
            self.relationships = data.get('relationships', [])

        # Load adjacency
        if ADJACENCY_PATH.exists():
            with open(ADJACENCY_PATH, 'r') as f:
                adj_data = json.load(f)
                for source, targets in adj_data.items():
                    for target, rels in targets.items():
                        self.adjacency[source][target] = rels

        print(f"Loaded {len(self.entities)} entities and {len(self.relationships)} relationships")

    def load_discourse(self):
        """Load discourse elements (questions, claims, assertions, evidence)"""
        print("Loading discourse elements...")
        self.discourse = {"assertions": [], "questions": [], "claims": [], "evidence": []}

        # Load book discourse
        if DISCOURSE_PATH.exists():
            with open(DISCOURSE_PATH, 'r') as f:
                book_discourse = json.load(f)
                self.discourse["assertions"].extend(book_discourse.get('assertions', []))
                self.discourse["questions"].extend(book_discourse.get('questions', []))
                self.discourse["claims"].extend(book_discourse.get('claims', []))
                self.discourse["evidence"].extend(book_discourse.get('evidence', []))

        # Load episode discourse
        if EPISODE_DISCOURSE_PATH.exists():
            with open(EPISODE_DISCOURSE_PATH, 'r') as f:
                episode_data = json.load(f)
                # Episode discourse has a different structure with "episodes" array
                if 'episodes' in episode_data:
                    for ep in episode_data['episodes']:
                        self.discourse["assertions"].extend(ep.get('assertions', []))
                        self.discourse["questions"].extend(ep.get('questions', []))
                        self.discourse["claims"].extend(ep.get('claims', []))
                        self.discourse["evidence"].extend(ep.get('evidence', []))

        print(f"Loaded {len(self.discourse.get('assertions', []))} assertions total")
        print(f"  - Questions: {len(self.discourse.get('questions', []))}")
        print(f"  - Claims: {len(self.discourse.get('claims', []))}")
        print(f"  - Evidence: {len(self.discourse.get('evidence', []))}")

    def build_mentioned_in_links(self):
        """Build 'mentioned_in' links for entities appearing in multiple sources"""
        print("\n=== Building mentioned_in links ===")

        for entity_name, entity_data in tqdm(self.entities.items(), desc="Processing entities"):
            sources = entity_data.get('sources', [])
            evidence = entity_data.get('evidence', [])

            # Extract content IDs from sources
            content_ids = set()

            # Parse sources (could be episode numbers or book names)
            for source in sources:
                if isinstance(source, str):
                    if source.startswith('episode_'):
                        # Extract episode number
                        try:
                            ep_num = int(source.split('_')[1])
                            content_ids.add(f"episode_{ep_num}")
                        except:
                            content_ids.add(source)
                    elif 'book' in source.lower() or source in ['VIRIDITAS', 'Y on Earth', 'Soil Stewardship']:
                        # Book source
                        content_ids.add(f"book_{source}")
                    else:
                        content_ids.add(source)
                elif isinstance(source, int):
                    # Episode number
                    content_ids.add(f"episode_{source}")

            # Parse evidence for chunk IDs
            for ev in evidence:
                if isinstance(ev, dict):
                    chunk_id = ev.get('chunk_id', '')
                    if 'episode' in chunk_id:
                        # Extract episode from chunk ID
                        parts = chunk_id.split('_')
                        for i, part in enumerate(parts):
                            if part == 'episode' and i + 1 < len(parts):
                                try:
                                    ep_num = int(parts[i + 1])
                                    content_ids.add(f"episode_{ep_num}")
                                except:
                                    pass
                    elif 'book' in chunk_id.lower():
                        # Extract book from chunk ID
                        if 'viriditas' in chunk_id.lower():
                            content_ids.add("book_VIRIDITAS")
                        elif 'soil' in chunk_id.lower():
                            content_ids.add("book_Soil Stewardship")
                        elif 'earth' in chunk_id.lower():
                            content_ids.add("book_Y on Earth")

            # Store content IDs for this entity
            if len(content_ids) > 0:
                self.cross_links["mentioned_in"][entity_name] = content_ids

                # Track multi-source entities (appearing in both books and episodes)
                has_book = any('book_' in cid for cid in content_ids)
                has_episode = any('episode_' in cid for cid in content_ids)

                if has_book and has_episode:
                    self.stats["entities_with_multi_source"] += 1

        self.stats["mentioned_in_links"] = len(self.cross_links["mentioned_in"])
        print(f"Created {self.stats['mentioned_in_links']} mentioned_in links")
        print(f"Found {self.stats['entities_with_multi_source']} entities appearing in both books and episodes")

    def calculate_assertion_similarity(self, assertion1, assertion2):
        """Calculate similarity between two assertions"""

        # Extract text representations
        text1 = assertion1.get('text', '')
        text2 = assertion2.get('text', '')

        if not text1 or not text2:
            # Try to construct from subject-predicate-object
            if all(k in assertion1 for k in ['subject', 'predicate', 'object']):
                text1 = f"{assertion1['subject']} {assertion1['predicate']} {assertion1['object']}"
            if all(k in assertion2 for k in ['subject', 'predicate', 'object']):
                text2 = f"{assertion2['subject']} {assertion2['predicate']} {assertion2['object']}"

        if not text1 or not text2:
            return 0.0

        # Simple Jaccard similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union) if union else 0.0

    def detect_contradiction(self, assertion1, assertion2):
        """Detect if two assertions contradict each other"""

        # Check if same subject but opposite predicates
        subj1 = assertion1.get('subject', '').lower()
        subj2 = assertion2.get('subject', '').lower()
        pred1 = assertion1.get('predicate', '').lower()
        pred2 = assertion2.get('predicate', '').lower()

        # Lists of opposite predicates
        opposites = [
            ('increases', 'decreases'),
            ('supports', 'opposes'),
            ('causes', 'prevents'),
            ('enables', 'disables'),
            ('promotes', 'inhibits'),
            ('positive', 'negative'),
            ('beneficial', 'harmful')
        ]

        # Check if subjects are similar
        if subj1 and subj2:
            subj_similarity = self.calculate_assertion_similarity(
                {'text': subj1},
                {'text': subj2}
            )

            if subj_similarity > 0.7:
                # Check if predicates are opposites
                for opp1, opp2 in opposites:
                    if (opp1 in pred1 and opp2 in pred2) or (opp2 in pred1 and opp1 in pred2):
                        return True

        return False

    def build_assertion_links(self):
        """Build supports and contradicts links between assertions"""
        print("\n=== Building assertion links ===")

        assertions = self.discourse.get('assertions', [])

        if not assertions:
            print("No assertions found in discourse")
            return

        print(f"Comparing {len(assertions)} assertions...")

        # Group assertions by source for cross-content comparison
        book_assertions = []
        episode_assertions = []

        for assertion in assertions:
            chunk_id = assertion.get('chunk_id', '')
            if 'book' in chunk_id.lower():
                book_assertions.append(assertion)
            elif 'episode' in chunk_id.lower() or 'ep' in chunk_id:
                episode_assertions.append(assertion)

        print(f"Found {len(book_assertions)} book assertions and {len(episode_assertions)} episode assertions")

        # Compare book assertions with episode assertions
        for book_assert in tqdm(book_assertions, desc="Comparing assertions"):
            for ep_assert in episode_assertions:
                # Calculate similarity
                similarity = self.calculate_assertion_similarity(book_assert, ep_assert)

                # Check for support relationship (high similarity)
                if similarity >= 0.7:
                    confidence = min(
                        book_assert.get('confidence', 0.7),
                        ep_assert.get('confidence', 0.7)
                    ) * similarity

                    self.cross_links["supports"].append({
                        "source": book_assert,
                        "target": ep_assert,
                        "confidence": confidence,
                        "similarity": similarity
                    })
                    self.stats["supports_links"] += 1

                # Check for contradiction
                if self.detect_contradiction(book_assert, ep_assert):
                    self.cross_links["contradicts"].append({
                        "source": book_assert,
                        "target": ep_assert,
                        "confidence": 0.8  # High confidence for detected contradictions
                    })
                    self.stats["contradicts_links"] += 1

        print(f"Created {self.stats['supports_links']} supports links")
        print(f"Created {self.stats['contradicts_links']} contradicts links")

    def add_cross_links_to_adjacency(self):
        """Add cross-content links to the adjacency list"""
        print("\n=== Adding cross-links to adjacency ===")

        # Add mentioned_in edges
        for entity, content_ids in self.cross_links["mentioned_in"].items():
            for content_id in content_ids:
                # Create a pseudo-node for the content
                if entity not in self.adjacency:
                    self.adjacency[entity] = {}

                if content_id not in self.adjacency[entity]:
                    self.adjacency[entity][content_id] = []

                self.adjacency[entity][content_id].append({
                    "predicate": "mentioned_in",
                    "confidence": 1.0,
                    "cross_content": True
                })

        # Add supports edges
        for support_link in self.cross_links["supports"]:
            source_text = support_link["source"].get("text", "assertion_source")
            target_text = support_link["target"].get("text", "assertion_target")

            # Create hash-based IDs for assertions
            source_id = f"assertion_{hashlib.md5(source_text.encode()).hexdigest()[:8]}"
            target_id = f"assertion_{hashlib.md5(target_text.encode()).hexdigest()[:8]}"

            if source_id not in self.adjacency:
                self.adjacency[source_id] = {}

            if target_id not in self.adjacency[source_id]:
                self.adjacency[source_id][target_id] = []

            self.adjacency[source_id][target_id].append({
                "predicate": "supports",
                "confidence": support_link["confidence"],
                "cross_content": True,
                "similarity": support_link["similarity"]
            })

        # Add contradicts edges
        for contradict_link in self.cross_links["contradicts"]:
            source_text = contradict_link["source"].get("text", "assertion_source")
            target_text = contradict_link["target"].get("text", "assertion_target")

            source_id = f"assertion_{hashlib.md5(source_text.encode()).hexdigest()[:8]}"
            target_id = f"assertion_{hashlib.md5(target_text.encode()).hexdigest()[:8]}"

            if source_id not in self.adjacency:
                self.adjacency[source_id] = {}

            if target_id not in self.adjacency[source_id]:
                self.adjacency[source_id][target_id] = []

            self.adjacency[source_id][target_id].append({
                "predicate": "contradicts",
                "confidence": contradict_link["confidence"],
                "cross_content": True
            })

        print(f"Updated adjacency with cross-content links")

    def save_results(self):
        """Save cross-content links and updated adjacency"""
        print("\nSaving results...")

        # Calculate total cross-links
        self.stats["total_cross_links"] = (
            self.stats["mentioned_in_links"] +
            self.stats["supports_links"] +
            self.stats["contradicts_links"]
        )

        # Save cross-links
        cross_links_output = {
            "mentioned_in": {k: list(v) for k, v in self.cross_links["mentioned_in"].items()},
            "supports": self.cross_links["supports"],
            "contradicts": self.cross_links["contradicts"],
            "stats": self.stats
        }

        with open(CROSS_LINKS_PATH, 'w') as f:
            json.dump(cross_links_output, f, indent=2)
        print(f"Saved cross-content links to {CROSS_LINKS_PATH}")

        # Save updated adjacency
        adjacency_output = {source: dict(targets) for source, targets in self.adjacency.items()}

        with open(UPDATED_ADJACENCY_PATH, 'w') as f:
            json.dump(adjacency_output, f, indent=2)
        print(f"Saved updated adjacency to {UPDATED_ADJACENCY_PATH}")

        # Save statistics
        with open(CROSS_LINKS_STATS_PATH, 'w') as f:
            json.dump(self.stats, f, indent=2)
        print(f"Saved statistics to {CROSS_LINKS_STATS_PATH}")

    def run(self):
        """Run the cross-content linking pipeline"""
        print("=" * 60)
        print("Cross-Content Linking Pipeline")
        print("=" * 60)

        # Load data
        self.load_knowledge_graph()
        self.load_discourse()

        # Build different types of cross-links
        self.build_mentioned_in_links()
        self.build_assertion_links()

        # Update adjacency with cross-links
        self.add_cross_links_to_adjacency()

        # Save results
        self.save_results()

        # Print summary
        print("\n" + "=" * 60)
        print("Cross-Content Linking Summary")
        print("=" * 60)
        print(f"Total cross-content links: {self.stats['total_cross_links']:,}")
        print(f"  - mentioned_in: {self.stats['mentioned_in_links']:,}")
        print(f"  - supports: {self.stats['supports_links']:,}")
        print(f"  - contradicts: {self.stats['contradicts_links']:,}")
        print(f"Multi-source entities: {self.stats['entities_with_multi_source']:,}")

        # Check if we met the target
        if self.stats['total_cross_links'] >= 5000:
            print("\n✅ Successfully created 5,000+ cross-content links!")
        else:
            print(f"\n⚠️ Created {self.stats['total_cross_links']} links (target: 5,000+)")
            print("Consider running discourse extraction first for more assertion links")


if __name__ == "__main__":
    linker = CrossContentLinker()
    linker.run()