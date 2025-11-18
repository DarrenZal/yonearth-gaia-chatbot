#!/usr/bin/env python3
"""
Build Discourse Layer from Book and Episode Knowledge Graph Extractions

This script extracts discourse elements (questions, claims, evidence) from:
1. Book KG extractions with existing claims and questions
2. Episode KG relationships that imply discourse

Output:
- data/knowledge_graph_unified/discourse.json
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Set, Any, Optional
from collections import defaultdict
import hashlib
from datetime import datetime
import re

# Paths
PROJECT_ROOT = Path("/home/claudeuser/yonearth-gaia-chatbot")
BOOK_KG_DIR = PROJECT_ROOT / "kg_unified_discourse/outputs/book_extractions"
EPISODE_KG_DIR = PROJECT_ROOT / "data/knowledge_graph_v3_2_2"
UNIFIED_KG_PATH = PROJECT_ROOT / "data/knowledge_graph_unified/unified.json"
OUTPUT_DIR = PROJECT_ROOT / "data/knowledge_graph_unified"

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

class DiscourseBuilder:
    def __init__(self):
        self.assertions = []
        self.questions = []
        self.evidence_snippets = []
        self.answers = []
        self.chunk_discourse_map = defaultdict(lambda: {
            "questions": [],
            "assertions": [],
            "evidence": []
        })
        self.stats = {
            "assertion_count": 0,
            "question_count": 0,
            "evidence_count": 0,
            "answer_count": 0,
            "sources": {
                "books": [],
                "episodes": []
            },
            "created_at": datetime.now().isoformat()
        }

    def extract_book_discourse(self):
        """Extract discourse elements from book KG extractions"""
        print("Extracting discourse from book KGs...")

        book_files = [
            "y-on-earth_complete.json",
            "veriditas_complete.json",
            "soil-stewardship-handbook_complete.json",
            "our-biggest-deal_complete.json"
        ]

        for book_file in book_files:
            book_path = BOOK_KG_DIR / book_file
            if not book_path.exists():
                print(f"Warning: Book file {book_file} not found")
                continue

            with open(book_path, 'r') as f:
                data = json.load(f)

            book_name = data.get('book', book_file.replace('_complete.json', ''))
            self.stats['sources']['books'].append(book_name)

            # Extract assertions from relationships with discourse patterns
            for rel in data.get('relationships', []):
                self._extract_assertion_from_relationship(rel, book_name)
                self._extract_question_from_relationship(rel, book_name)
                self._extract_evidence_from_relationship(rel, book_name)

        print(f"Extracted {len(self.assertions)} assertions, {len(self.questions)} questions from books")

    def extract_episode_discourse(self):
        """Extract discourse elements from episode KG extractions"""
        print("Extracting discourse from episode KGs...")

        episode_files = sorted(EPISODE_KG_DIR.glob("episode_*_v3_2_2.json"))

        for episode_file in episode_files[:50]:  # Process first 50 episodes for now
            try:
                with open(episode_file, 'r') as f:
                    data = json.load(f)

                episode_num = data.get('episode', 'unknown')
                self.stats['sources']['episodes'].append(episode_num)

                # Extract discourse from relationships
                for rel in data.get('relationships', []):
                    self._extract_assertion_from_relationship(rel, f"episode_{episode_num}")
                    self._extract_question_from_relationship(rel, f"episode_{episode_num}")
                    self._extract_evidence_from_relationship(rel, f"episode_{episode_num}")

            except Exception as e:
                print(f"Error processing {episode_file}: {e}")
                continue

        print(f"Extracted additional discourse from {len(self.stats['sources']['episodes'])} episodes")

    def _extract_assertion_from_relationship(self, rel: Dict, source: str):
        """Extract assertion from a relationship if it represents a claim"""
        predicate = rel.get('predicate', rel.get('relationship', '')).lower()

        # Patterns that indicate assertions/claims
        claim_patterns = [
            'claims', 'states', 'argues', 'asserts', 'believes', 'maintains',
            'advocates', 'promotes', 'suggests', 'proposes', 'emphasizes',
            'demonstrates', 'shows', 'reveals', 'indicates', 'implies'
        ]

        if any(pattern in predicate for pattern in claim_patterns):
            assertion_id = self._generate_id(f"{rel.get('source')}_{predicate}_{rel.get('target')}")

            assertion = {
                "id": assertion_id,
                "type": "ASSERTION",
                "subject": rel.get('source'),
                "predicate": predicate,
                "object": rel.get('target'),
                "confidence": rel.get('confidence', rel.get('p_true', 0.5)),
                "evidence_text": rel.get('evidence', rel.get('evidence_text', '')),
                "source": source,
                "context": rel.get('context', ''),
                "metadata": {
                    "extracted_from": "relationship",
                    "original_relationship_id": rel.get('id', ''),
                    "source_type": rel.get('source_type'),
                    "target_type": rel.get('target_type')
                }
            }

            self.assertions.append(assertion)

            # Map to chunk if available
            if 'chunk_id' in rel:
                self.chunk_discourse_map[rel['chunk_id']]['assertions'].append(assertion_id)

    def _extract_question_from_relationship(self, rel: Dict, source: str):
        """Extract question from a relationship if it represents an inquiry"""
        predicate = rel.get('predicate', rel.get('relationship', '')).lower()
        evidence = rel.get('evidence', rel.get('evidence_text', ''))

        # Patterns that indicate questions
        question_patterns = [
            'asks', 'questions', 'inquires', 'wonders', 'poses question',
            'raises question', 'explores whether', 'investigates'
        ]

        # Also check for actual question marks in evidence
        has_question = any(pattern in predicate for pattern in question_patterns) or '?' in evidence

        if has_question:
            question_id = self._generate_id(f"question_{rel.get('source')}_{predicate}_{source}")

            # Try to extract actual question text from evidence
            question_text = self._extract_question_text(evidence) or f"{rel.get('source')} {predicate} {rel.get('target')}"

            question = {
                "id": question_id,
                "type": "QUESTION",
                "text": question_text,
                "asked_by": rel.get('source'),
                "topic_entities": [rel.get('target')] if rel.get('target') else [],
                "occurs_in": source,
                "metadata": {
                    "original_predicate": predicate,
                    "confidence": rel.get('confidence', rel.get('p_true', 0.5))
                }
            }

            self.questions.append(question)

            # Map to chunk if available
            if 'chunk_id' in rel:
                self.chunk_discourse_map[rel['chunk_id']]['questions'].append(question_id)

    def _extract_evidence_from_relationship(self, rel: Dict, source: str):
        """Extract evidence snippet from relationship"""
        evidence_text = rel.get('evidence', rel.get('evidence_text', ''))

        if evidence_text and len(evidence_text) > 20:  # Only meaningful evidence
            evidence_id = self._generate_id(f"evidence_{evidence_text[:50]}_{source}")

            evidence_snippet = {
                "id": evidence_id,
                "type": "EVIDENCE_SNIPPET",
                "text": evidence_text,
                "source_chunk": rel.get('chunk_id', source),
                "supports": [],  # Will be linked later
                "contradicts": [],  # Will be linked later
                "metadata": {
                    "confidence": rel.get('confidence', rel.get('p_true', 0.5)),
                    "source": source,
                    "entities_mentioned": [rel.get('source'), rel.get('target')]
                }
            }

            self.evidence_snippets.append(evidence_snippet)

            # Map to chunk if available
            if 'chunk_id' in rel:
                self.chunk_discourse_map[rel['chunk_id']]['evidence'].append(evidence_id)

    def _extract_question_text(self, text: str) -> Optional[str]:
        """Extract actual question from text if present"""
        if not text:
            return None

        # Find sentences with question marks
        sentences = re.split(r'[.!?]', text)
        questions = [s.strip() for s in sentences if '?' in s]

        if questions:
            return questions[0] + '?'

        return None

    def _generate_id(self, content: str) -> str:
        """Generate stable ID for discourse element"""
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def link_evidence_to_assertions(self):
        """Link evidence snippets to assertions they support"""
        print("Linking evidence to assertions...")

        # Simple heuristic: link evidence to assertions from same source
        # with overlapping entities
        for evidence in self.evidence_snippets:
            evidence_entities = set(evidence['metadata'].get('entities_mentioned', []))
            evidence_source = evidence['metadata'].get('source')

            for assertion in self.assertions:
                if assertion['source'] == evidence_source:
                    assertion_entities = {assertion['subject'], assertion['object']}

                    # If entities overlap, consider it supporting evidence
                    if evidence_entities & assertion_entities:
                        evidence['supports'].append(assertion['id'])

    def calculate_stats(self):
        """Calculate statistics about discourse elements"""
        self.stats['assertion_count'] = len(self.assertions)
        self.stats['question_count'] = len(self.questions)
        self.stats['evidence_count'] = len(self.evidence_snippets)
        self.stats['answer_count'] = len(self.answers)

        # Count by source type
        book_assertions = sum(1 for a in self.assertions if not a['source'].startswith('episode_'))
        episode_assertions = sum(1 for a in self.assertions if a['source'].startswith('episode_'))

        book_questions = sum(1 for q in self.questions if not q['occurs_in'].startswith('episode_'))
        episode_questions = sum(1 for q in self.questions if q['occurs_in'].startswith('episode_'))

        print(f"\nDiscourse Statistics:")
        print(f"  - Total Assertions: {self.stats['assertion_count']} (Books: {book_assertions}, Episodes: {episode_assertions})")
        print(f"  - Total Questions: {self.stats['question_count']} (Books: {book_questions}, Episodes: {episode_questions})")
        print(f"  - Total Evidence: {self.stats['evidence_count']}")
        print(f"  - Chunks with discourse: {len(self.chunk_discourse_map)}")

    def save_discourse(self):
        """Save discourse elements to JSON"""
        output_path = OUTPUT_DIR / "discourse.json"

        discourse_data = {
            "metadata": {
                "created_at": self.stats['created_at'],
                "statistics": self.stats
            },
            "assertions": self.assertions,
            "questions": self.questions,
            "evidence": self.evidence_snippets,
            "answers": self.answers,
            "chunk_discourse_map": dict(self.chunk_discourse_map)
        }

        with open(output_path, 'w') as f:
            json.dump(discourse_data, f, indent=2)

        print(f"\n✅ Saved discourse to {output_path}")

def main():
    print("=" * 60)
    print("Discourse Layer Builder")
    print("=" * 60)

    builder = DiscourseBuilder()

    # Extract from books
    builder.extract_book_discourse()

    # Extract from episodes
    builder.extract_episode_discourse()

    # Link evidence to assertions
    builder.link_evidence_to_assertions()

    # Calculate stats
    builder.calculate_stats()

    # Save discourse
    builder.save_discourse()

    print("\n✅ Discourse layer build complete!")

if __name__ == "__main__":
    main()