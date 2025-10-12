#!/usr/bin/env python3
"""
Emergent Domain Ontology System

This system creates and evolves a domain ontology emergently from the data,
rather than using predefined categories. It discovers semantic patterns,
creates domain types dynamically, and evolves as new data arrives.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict, Counter
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import openai
import os
from dataclasses import dataclass
from datetime import datetime


@dataclass
class DomainType:
    """Emergent domain type discovered from data"""
    id: str
    name: str
    raw_members: List[str]
    centroid_embedding: List[float]
    properties: Dict
    confidence: float
    first_seen: str
    last_updated: str
    frequency: int
    evolution_history: List[Dict]


class EmergentOntologySystem:
    """System for discovering and evolving domain ontology from relationship data"""

    def __init__(self, embedding_model="text-embedding-3-small"):
        """Initialize the emergent ontology system"""
        self.embedding_model = embedding_model
        self.domain_types = {}
        self.raw_to_domain_map = {}
        self.embeddings_cache = {}
        self.evolution_log = []

        # Clustering parameters (can be tuned)
        self.min_cluster_size = 3  # Minimum relationships to form a domain type
        self.similarity_threshold = 0.7  # For assigning new relationships
        self.merge_threshold = 0.85  # For merging similar domain types
        self.novelty_threshold = 0.5  # Below this, create new domain type

        # Load existing ontology if it exists
        self.ontology_path = Path("data/knowledge_graph/emergent_ontology.json")
        self.load_ontology()

    def load_ontology(self):
        """Load existing emergent ontology if available"""
        if self.ontology_path.exists():
            with open(self.ontology_path) as f:
                data = json.load(f)
                self.domain_types = {k: DomainType(**v) for k, v in data["domain_types"].items()}
                self.raw_to_domain_map = data["mappings"]
                self.evolution_log = data.get("evolution_log", [])
                print(f"Loaded {len(self.domain_types)} emergent domain types")

    def save_ontology(self):
        """Save the current emergent ontology"""
        self.ontology_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "domain_types": {k: v.__dict__ for k, v in self.domain_types.items()},
            "mappings": self.raw_to_domain_map,
            "evolution_log": self.evolution_log,
            "metadata": {
                "last_updated": datetime.now().isoformat(),
                "total_raw_types": len(self.raw_to_domain_map),
                "total_domain_types": len(self.domain_types)
            }
        }

        with open(self.ontology_path, 'w') as f:
            json.dump(data, f, indent=2)

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a relationship type"""
        if text in self.embeddings_cache:
            return self.embeddings_cache[text]

        try:
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.embeddings.create(
                input=text,
                model=self.embedding_model
            )
            embedding = response.data[0].embedding
            self.embeddings_cache[text] = embedding
            return embedding
        except Exception as e:
            print(f"Error getting embedding for {text}: {e}")
            return None

    def discover_domain_types(self, raw_relationships: List[str]) -> Dict[str, List[str]]:
        """
        Discover domain types emergently from raw relationship types
        using clustering on embeddings
        """
        print(f"\n🔬 Discovering domain types from {len(raw_relationships)} raw relationships...")

        # Get embeddings for all raw relationships
        embeddings = []
        valid_relationships = []

        for rel in raw_relationships:
            emb = self.get_embedding(rel)
            if emb:
                embeddings.append(emb)
                valid_relationships.append(rel)

        if not embeddings:
            return {}

        embeddings_array = np.array(embeddings)

        # Use DBSCAN for discovering clusters (doesn't require predefined number)
        clusterer = DBSCAN(
            eps=1 - self.similarity_threshold,  # Distance threshold
            min_samples=self.min_cluster_size,
            metric='cosine'
        )

        cluster_labels = clusterer.fit_predict(embeddings_array)

        # Group relationships by cluster
        clusters = defaultdict(list)
        for rel, label in zip(valid_relationships, cluster_labels):
            if label != -1:  # -1 is noise in DBSCAN
                clusters[label].append(rel)

        print(f"  Found {len(clusters)} emergent clusters")

        # Create domain types from clusters
        new_domain_types = {}
        for cluster_id, members in clusters.items():
            domain_type = self._create_domain_type(members, embeddings_array)
            if domain_type:
                new_domain_types[domain_type.id] = domain_type

        return new_domain_types

    def _create_domain_type(self, members: List[str], all_embeddings: np.ndarray) -> Optional[DomainType]:
        """Create a domain type from a cluster of relationships"""
        if len(members) < self.min_cluster_size:
            return None

        # Calculate centroid embedding
        member_embeddings = [self.get_embedding(m) for m in members]
        centroid = np.mean(member_embeddings, axis=0).tolist()

        # Generate name using GPT
        domain_name = self._generate_domain_name(members)

        # Infer properties from members
        properties = self._infer_properties(members)

        # Create domain type
        domain_id = f"DOMAIN_{len(self.domain_types)}"
        domain_type = DomainType(
            id=domain_id,
            name=domain_name,
            raw_members=members,
            centroid_embedding=centroid,
            properties=properties,
            confidence=self._calculate_cluster_confidence(member_embeddings),
            first_seen=datetime.now().isoformat(),
            last_updated=datetime.now().isoformat(),
            frequency=len(members),
            evolution_history=[]
        )

        return domain_type

    def _generate_domain_name(self, members: List[str]) -> str:
        """Use GPT to generate a semantic name for a cluster"""
        try:
            prompt = f"""Given these relationship types, generate a single semantic category name:
            {', '.join(members[:10])}

            Return ONLY the category name in CAPS_WITH_UNDERSCORES format. Examples: MENTORS, FUNDS, CREATES"""

            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
                temperature=0
            )

            name = response.choices[0].message.content.strip()
            # Fallback to pattern-based name if GPT fails
            if not name or len(name) > 30:
                name = self._pattern_based_name(members)
            return name

        except:
            return self._pattern_based_name(members)

    def _pattern_based_name(self, members: List[str]) -> str:
        """Generate name based on common patterns in members"""
        # Find most common words
        words = []
        for member in members:
            words.extend(member.split('_'))

        common = Counter(words).most_common(1)[0][0]
        return f"DOMAIN_{common.upper()}"

    def _infer_properties(self, members: List[str]) -> Dict:
        """Infer semantic properties from cluster members"""
        properties = {}

        # Check for common patterns
        if any('FUND' in m or 'FINANC' in m or 'INVEST' in m for m in members):
            properties['financial'] = True

        if any('TEACH' in m or 'MENTOR' in m or 'EDUCAT' in m for m in members):
            properties['knowledge_transfer'] = True

        if any('CREATE' in m or 'PRODUCE' in m or 'BUILD' in m for m in members):
            properties['creative'] = True

        if any('CAUSE' in m or 'LEAD' in m or 'RESULT' in m for m in members):
            properties['causal'] = True

        if any('HAS_' in m or 'CONTAIN' in m or 'POSSESS' in m for m in members):
            properties['attributive'] = True

        return properties

    def _calculate_cluster_confidence(self, embeddings: List[List[float]]) -> float:
        """Calculate confidence score for a cluster based on cohesion"""
        if len(embeddings) < 2:
            return 1.0

        # Calculate average pairwise similarity
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                similarities.append(sim)

        return float(np.mean(similarities))

    def evolve_with_new_relationships(self, new_relationships: List[str]):
        """
        Evolve the ontology with new relationships from a new episode.
        This is called incrementally as new data arrives.
        """
        print(f"\n🔄 Evolving ontology with {len(new_relationships)} new relationships...")

        evolution_entry = {
            "timestamp": datetime.now().isoformat(),
            "new_relationships": len(new_relationships),
            "actions": []
        }

        for rel in new_relationships:
            if rel in self.raw_to_domain_map:
                # Already mapped, increase frequency
                domain_id = self.raw_to_domain_map[rel]
                if domain_id in self.domain_types:
                    self.domain_types[domain_id].frequency += 1
                continue

            # Get embedding for new relationship
            embedding = self.get_embedding(rel)
            if not embedding:
                continue

            # Find closest domain type
            best_similarity = 0
            best_domain = None

            for domain_id, domain_type in self.domain_types.items():
                similarity = cosine_similarity([embedding], [domain_type.centroid_embedding])[0][0]
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_domain = domain_id

            # Decide action based on similarity
            if best_similarity >= self.similarity_threshold:
                # Assign to existing domain type
                self._add_to_domain(rel, best_domain, embedding)
                evolution_entry["actions"].append({
                    "type": "assigned",
                    "relationship": rel,
                    "domain": best_domain,
                    "similarity": best_similarity
                })

            elif best_similarity >= self.novelty_threshold:
                # Similar but not enough - might need new subcategory
                self._create_subdomain(rel, best_domain, embedding)
                evolution_entry["actions"].append({
                    "type": "subdomain_created",
                    "relationship": rel,
                    "parent_domain": best_domain,
                    "similarity": best_similarity
                })

            else:
                # Very novel - create new domain type
                new_domain = self._create_singleton_domain(rel, embedding)
                evolution_entry["actions"].append({
                    "type": "new_domain",
                    "relationship": rel,
                    "domain": new_domain.id if new_domain else None
                })

        # Check for domain mergers
        self._check_domain_mergers()

        # Log evolution
        self.evolution_log.append(evolution_entry)

        # Save updated ontology
        self.save_ontology()

        print(f"  Evolution complete: {len(evolution_entry['actions'])} changes made")

    def _add_to_domain(self, relationship: str, domain_id: str, embedding: List[float]):
        """Add a relationship to an existing domain type and update centroid"""
        domain = self.domain_types[domain_id]

        # Add to members
        domain.raw_members.append(relationship)
        domain.frequency += 1

        # Update centroid (incremental update for efficiency)
        n = len(domain.raw_members)
        old_centroid = np.array(domain.centroid_embedding)
        new_centroid = ((n - 1) * old_centroid + np.array(embedding)) / n
        domain.centroid_embedding = new_centroid.tolist()

        # Update timestamp
        domain.last_updated = datetime.now().isoformat()

        # Map relationship
        self.raw_to_domain_map[relationship] = domain_id

    def _create_subdomain(self, relationship: str, parent_domain: str, embedding: List[float]):
        """Create a subdomain when similarity is moderate"""
        # For now, create as new domain but track parent relationship
        new_domain = self._create_singleton_domain(relationship, embedding)
        if new_domain:
            new_domain.properties['parent_domain'] = parent_domain

    def _create_singleton_domain(self, relationship: str, embedding: List[float]) -> Optional[DomainType]:
        """Create a new domain type from a single novel relationship"""
        domain_id = f"DOMAIN_{len(self.domain_types)}"

        domain_type = DomainType(
            id=domain_id,
            name=relationship,  # Will be refined as more members join
            raw_members=[relationship],
            centroid_embedding=embedding,
            properties=self._infer_properties([relationship]),
            confidence=0.5,  # Lower confidence for singleton
            first_seen=datetime.now().isoformat(),
            last_updated=datetime.now().isoformat(),
            frequency=1,
            evolution_history=[]
        )

        self.domain_types[domain_id] = domain_type
        self.raw_to_domain_map[relationship] = domain_id

        return domain_type

    def _check_domain_mergers(self):
        """Check if any domain types should be merged based on similarity"""
        merged = []

        for id1, domain1 in self.domain_types.items():
            if id1 in merged:
                continue

            for id2, domain2 in self.domain_types.items():
                if id2 <= id1 or id2 in merged:
                    continue

                similarity = cosine_similarity(
                    [domain1.centroid_embedding],
                    [domain2.centroid_embedding]
                )[0][0]

                if similarity >= self.merge_threshold:
                    # Merge domain2 into domain1
                    self._merge_domains(id1, id2)
                    merged.append(id2)

        # Remove merged domains
        for domain_id in merged:
            del self.domain_types[domain_id]

    def _merge_domains(self, keep_id: str, merge_id: str):
        """Merge two domain types"""
        keep_domain = self.domain_types[keep_id]
        merge_domain = self.domain_types[merge_id]

        # Combine members
        keep_domain.raw_members.extend(merge_domain.raw_members)

        # Recalculate centroid
        all_embeddings = [self.get_embedding(m) for m in keep_domain.raw_members]
        keep_domain.centroid_embedding = np.mean(all_embeddings, axis=0).tolist()

        # Update frequency and properties
        keep_domain.frequency += merge_domain.frequency
        keep_domain.properties.update(merge_domain.properties)

        # Update mappings
        for member in merge_domain.raw_members:
            self.raw_to_domain_map[member] = keep_id

        # Log merger in evolution history
        keep_domain.evolution_history.append({
            "type": "merger",
            "merged_from": merge_id,
            "timestamp": datetime.now().isoformat()
        })

    def generate_report(self):
        """Generate a report on the emergent ontology"""
        report = {
            "total_domain_types": len(self.domain_types),
            "total_raw_relationships": len(self.raw_to_domain_map),
            "coverage": len(self.raw_to_domain_map) / 837 * 100 if self.raw_to_domain_map else 0,
            "top_domains": [],
            "evolution_summary": {
                "total_evolution_steps": len(self.evolution_log),
                "last_evolution": self.evolution_log[-1]["timestamp"] if self.evolution_log else None
            }
        }

        # Find top domains by frequency
        sorted_domains = sorted(
            self.domain_types.values(),
            key=lambda x: x.frequency,
            reverse=True
        )[:10]

        for domain in sorted_domains:
            report["top_domains"].append({
                "name": domain.name,
                "frequency": domain.frequency,
                "members": len(domain.raw_members),
                "confidence": domain.confidence,
                "sample_members": domain.raw_members[:5]
            })

        return report


def main():
    """Main function to run emergent ontology discovery"""
    import argparse

    parser = argparse.ArgumentParser(description="Emergent Domain Ontology System")
    parser.add_argument("--discover", action="store_true",
                       help="Discover domain types from all existing relationships")
    parser.add_argument("--evolve", type=str,
                       help="Path to new relationships file to evolve with")
    parser.add_argument("--report", action="store_true",
                       help="Generate report on current ontology")

    args = parser.parse_args()

    # Initialize system
    system = EmergentOntologySystem()

    if args.discover:
        # Collect all raw relationships from existing extractions
        relationships_dir = Path("data/knowledge_graph/relationships")
        all_relationships = set()

        for file_path in relationships_dir.glob("episode_*_extraction.json"):
            with open(file_path) as f:
                data = json.load(f)
                for rel in data.get("relationships", []):
                    all_relationships.add(rel.get("relationship_type", ""))

        print(f"Found {len(all_relationships)} unique relationship types")

        # Discover domain types
        domain_types = system.discover_domain_types(list(all_relationships))

        # Save to system
        system.domain_types.update(domain_types)
        for domain_id, domain in domain_types.items():
            for member in domain.raw_members:
                system.raw_to_domain_map[member] = domain_id

        system.save_ontology()
        print(f"\n✅ Discovered {len(domain_types)} emergent domain types")

    if args.evolve:
        # Load new relationships and evolve
        with open(args.evolve) as f:
            data = json.load(f)
            new_rels = [rel.get("relationship_type", "") for rel in data.get("relationships", [])]

        system.evolve_with_new_relationships(new_rels)

    if args.report:
        report = system.generate_report()
        print("\n" + "=" * 70)
        print("EMERGENT ONTOLOGY REPORT")
        print("=" * 70)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()