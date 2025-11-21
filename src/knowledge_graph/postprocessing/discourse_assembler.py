"""
Discourse Assembly Module for Knowledge Graph Postprocessing
Extracts Questions, Claims, and Evidence from relationships
"""

from typing import List, Dict, Any, Optional
import hashlib
import re
from dataclasses import dataclass
from .base import PostProcessingModule

class DiscourseAssembler(PostProcessingModule):
    """
    Assembles discourse elements (questions, claims, evidence) from relationships.
    Creates a discourse overlay on existing knowledge graph.
    """

    name = "DiscourseAssembler"
    version = "1.0.0"

    def __init__(self,
                 min_claim_words: int = 12,
                 min_p_true: float = 0.7,
                 stable_id_length: int = 12):
        """
        Initialize discourse assembler.

        Args:
            min_claim_words: Minimum words for a claim to be considered
            min_p_true: Minimum confidence for discourse elements
            stable_id_length: Length of SHA-256 hash for stable IDs
        """
        super().__init__()
        self.min_claim_words = min_claim_words
        self.min_p_true = min_p_true
        self.stable_id_length = stable_id_length
        self.claim_registry = {}
        self.question_registry = {}

    def process(self, relationships: List[Any], context: Any) -> List[Any]:
        """
        Process relationships to extract discourse elements.

        Args:
            relationships: List of relationship objects
            context: Processing context with metadata

        Returns:
            Enhanced relationships list with discourse elements added
        """
        discourse_relationships = []

        for rel in relationships:
            try:
                # Extract questions
                if self._is_question(rel):
                    question_rel = self._create_question(rel)
                    if question_rel:
                        discourse_relationships.append(question_rel)

                # Extract claims from "said" relationships
                elif self._is_claim_candidate(rel):
                    claim_rel = self._create_claim(rel)
                    if claim_rel:
                        discourse_relationships.append(claim_rel)

                # Extract evidence links
                evidence_link = self._extract_evidence_link(rel)
                if evidence_link:
                    discourse_relationships.append(evidence_link)

            except Exception as e:
                self.logger.debug(f"Error processing relationship: {e}")
                continue

        # Add discourse relationships to original list
        relationships.extend(discourse_relationships)

        # Update statistics
        self._update_stats()

        return relationships

    def _is_question(self, rel) -> bool:
        """Check if relationship represents a question being posed"""
        if hasattr(rel, 'relationship'):
            rel_text = rel.relationship.lower()
        else:
            rel_text = str(rel.get('relationship', '')).lower()

        patterns = ['poses question', 'asks', 'inquires', 'questions', 'wonders']
        return any(pattern in rel_text for pattern in patterns)

    def _is_claim_candidate(self, rel) -> bool:
        """Check if relationship could represent a claim"""
        # Check relationship type
        if hasattr(rel, 'relationship'):
            rel_type = rel.relationship.lower()
        else:
            rel_type = str(rel.get('relationship', '')).lower()

        if rel_type != 'said':
            return False

        # Check target length
        if hasattr(rel, 'target'):
            target_text = rel.target
        else:
            target_text = rel.get('target', '')

        if not target_text or len(target_text.split()) < self.min_claim_words:
            return False

        # Check confidence
        if hasattr(rel, 'p_true'):
            p_true = rel.p_true
        else:
            p_true = rel.get('p_true', 0.5)

        return p_true >= self.min_p_true

    def _create_question(self, rel):
        """Create a question entity from a relationship"""
        if hasattr(rel, 'target'):
            question_text = rel.target
            source = rel.source if hasattr(rel, 'source') else ''
        else:
            question_text = rel.get('target', '')
            source = rel.get('source', '')

        if not question_text:
            return None

        # Generate stable ID
        qid = self._stable_id(question_text, 'Q')

        # Check if we've already seen this question
        if qid in self.question_registry:
            # Add additional source
            if source and source not in self.question_registry[qid]['sources']:
                self.question_registry[qid]['sources'].append(source)
            return None

        # Create new question relationship
        question_data = {
            'source': source,
            'relationship': 'poses',
            'target': question_text,
            'source_type': getattr(rel, 'source_type', 'UNKNOWN'),
            'target_type': 'QUESTION',
            'context': getattr(rel, 'context', ''),
            'page': getattr(rel, 'page', 0),
            'text_confidence': getattr(rel, 'text_confidence', 0.5),
            'p_true': getattr(rel, 'p_true', 0.5),
            'candidate_uid': qid
        }

        # Register question
        self.question_registry[qid] = {
            'id': qid,
            'text': question_text,
            'sources': [source] if source else []
        }

        # Create appropriate return type
        if hasattr(rel, '__class__'):
            return rel.__class__(**question_data)
        else:
            return question_data

    def _create_claim(self, rel):
        """Create or merge a claim entity from a relationship"""
        if hasattr(rel, 'target'):
            claim_text = rel.target
            source = rel.source if hasattr(rel, 'source') else ''
        else:
            claim_text = rel.get('target', '')
            source = rel.get('source', '')

        if not claim_text:
            return None

        # Generate stable ID
        claim_id = self._stable_id(claim_text, 'C')

        # Check if we've already seen this claim
        if claim_id in self.claim_registry:
            # Merge sources
            existing = self.claim_registry[claim_id]
            if source and source not in existing['sources']:
                existing['sources'].append(source)
                # Update context if needed
                if hasattr(rel, 'context'):
                    existing['context'] += f"; also claimed by {source}"
            return None

        # Create new claim relationship
        claim_data = {
            'source': source,
            'relationship': 'claims',
            'target': claim_text,
            'source_type': getattr(rel, 'source_type', 'UNKNOWN'),
            'target_type': 'CLAIM',
            'context': getattr(rel, 'context', ''),
            'page': getattr(rel, 'page', 0),
            'text_confidence': getattr(rel, 'text_confidence', 0.5),
            'p_true': getattr(rel, 'p_true', 0.5),
            'candidate_uid': claim_id,
            'word_count': len(claim_text.split())
        }

        # Register claim
        self.claim_registry[claim_id] = {
            'id': claim_id,
            'text': claim_text,
            'sources': [source] if source else [],
            'context': claim_data['context']
        }

        # Create appropriate return type
        if hasattr(rel, '__class__'):
            return rel.__class__(**{k: v for k, v in claim_data.items() if k != 'word_count'})
        else:
            return claim_data

    def _extract_evidence_link(self, rel) -> Optional[Dict]:
        """Extract evidence links from relationship context"""
        context = getattr(rel, 'context', '') or rel.get('context', '')

        if not context:
            return None

        # Check for evidence patterns
        evidence_patterns = [
            'according to', 'study shows', 'research indicates',
            'data from', 'measured at', '% of', 'survey found'
        ]

        if not any(pattern in context.lower() for pattern in evidence_patterns):
            return None

        # Extract metrics from context
        metrics = self._extract_metrics(context)

        if not metrics:
            return None

        # Find related claim if exists
        target_text = getattr(rel, 'target', '') or rel.get('target', '')
        if target_text and len(target_text.split()) >= self.min_claim_words:
            claim_id = self._stable_id(target_text, 'C')

            # Create evidence link
            evidence_data = {
                'source': 'EVIDENCE',
                'relationship': 'supports_claim',
                'target': claim_id,
                'source_type': 'EVIDENCE',
                'target_type': 'CLAIM',
                'context': context,
                'evidence_properties': metrics
            }

            return evidence_data

        return None

    def _extract_metrics(self, text: str) -> Dict:
        """Extract numeric values, percentages, dates from text"""
        metrics = {}

        try:
            # Percentages
            pct_match = re.search(r'(\d+(?:\.\d+)?)\s*%', text)
            if pct_match:
                metrics['percentage'] = float(pct_match.group(1))

            # Numbers with units
            unit_patterns = r'billion|million|thousand|GT|tons?|kg|USD|dollars|euros'
            num_match = re.search(r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(' + unit_patterns + ')', text, re.IGNORECASE)
            if num_match:
                value_str = num_match.group(1).replace(',', '')
                metrics['value'] = float(value_str) if '.' in value_str else int(value_str)
                metrics['unit'] = num_match.group(2)

            # Years
            year_match = re.search(r'\b(19\d{2}|20\d{2})\b', text)
            if year_match:
                metrics['year'] = int(year_match.group(1))

        except Exception as e:
            self.logger.debug(f"Error extracting metrics: {e}")

        return metrics

    def _stable_id(self, text: str, prefix: str) -> str:
        """Generate stable ID using SHA-256"""
        if not text:
            return f"{prefix}_empty"

        # Normalize text
        normalized = re.sub(r'[^\w\s]', '', text.lower()).strip()
        normalized = ' '.join(normalized.split())

        # Generate hash
        if normalized:
            hash_val = hashlib.sha256(normalized.encode()).hexdigest()[:self.stable_id_length]
        else:
            hash_val = 'empty'

        return f"{prefix}_{hash_val}"

    def _update_stats(self):
        """Update processing statistics"""
        self.stats['questions_extracted'] = len(self.question_registry)
        self.stats['claims_extracted'] = len(self.claim_registry)

        # Calculate claims with multiple sources
        multi_source_claims = sum(
            1 for claim in self.claim_registry.values()
            if len(claim.get('sources', [])) > 1
        )
        self.stats['claims_with_multiple_sources'] = multi_source_claims

        # Calculate deduplication rate
        if self.stats['claims_extracted'] > 0:
            self.stats['claim_deduplication_rate'] = (
                multi_source_claims / self.stats['claims_extracted']
            )
        else:
            self.stats['claim_deduplication_rate'] = 0

    def get_discourse_summary(self) -> Dict:
        """Get summary of extracted discourse elements"""
        return {
            'total_questions': len(self.question_registry),
            'total_claims': len(self.claim_registry),
            'claims_with_multiple_sources': self.stats.get('claims_with_multiple_sources', 0),
            'top_questions': list(self.question_registry.values())[:5],
            'top_claims': sorted(
                self.claim_registry.values(),
                key=lambda x: len(x.get('sources', [])),
                reverse=True
            )[:5]
        }