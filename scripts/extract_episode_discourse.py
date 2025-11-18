#!/usr/bin/env python3
"""
Extract Discourse Elements from Episode Transcripts

This script uses the ACE (Assertion, Claim, Evidence) framework to extract:
- Questions: "asks", "wonders", "inquires", sentences with "?"
- Claims: "states", "argues", "believes", "advocates"
- Evidence: Text snippets supporting claims

Expected output: ~50 assertions, ~30 questions per episode
Minimum confidence: p_true ≥ 0.6
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
from collections import defaultdict
import hashlib
from datetime import datetime
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize
import time
import openai
from openai import OpenAI

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Paths
PROJECT_ROOT = Path("/home/claudeuser/yonearth-gaia-chatbot")
TRANSCRIPTS_DIR = PROJECT_ROOT / "data/transcripts"
OUTPUT_DIR = PROJECT_ROOT / "data/knowledge_graph_unified"
DISCOURSE_OUTPUT_PATH = OUTPUT_DIR / "episode_discourse.json"
MERGED_DISCOURSE_PATH = OUTPUT_DIR / "discourse.json"

# OpenAI setup
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class DiscourseExtractor:
    def __init__(self, pilot_mode=False):
        self.pilot_mode = pilot_mode
        self.discourse_elements = {
            "assertions": [],
            "questions": [],
            "claims": [],
            "evidence": []
        }
        self.stats = {
            "episodes_processed": 0,
            "total_assertions": 0,
            "total_questions": 0,
            "total_claims": 0,
            "total_evidence": 0,
            "extraction_timestamp": datetime.now().isoformat()
        }

        # Patterns for discourse extraction
        self.question_patterns = [
            r'\?$',  # Ends with question mark
            r'^(what|who|when|where|why|how|is|are|can|could|would|should|do|does|did)',
            r'(wonder|wondering|ask|asking|inquire|inquiring|question)',
        ]

        self.claim_patterns = [
            r'(state|states|stating|argue|argues|arguing|believe|believes|believing)',
            r'(advocate|advocates|advocating|claim|claims|claiming)',
            r'(assert|asserts|asserting|maintain|maintains|maintaining)',
            r'(propose|proposes|proposing|suggest|suggests|suggesting)',
            r'(according to|research shows|studies show|evidence suggests)',
        ]

        self.evidence_patterns = [
            r'(for example|for instance|such as|including)',
            r'(data shows|research indicates|study found)',
            r'(percent|percentage|%|\d+\s*(million|billion|thousand))',
            r'(scientific|peer-reviewed|published)',
        ]

    def load_transcript(self, episode_num):
        """Load transcript for a specific episode"""
        transcript_path = TRANSCRIPTS_DIR / f"episode_{episode_num}.json"

        if not transcript_path.exists():
            return None

        with open(transcript_path, 'r') as f:
            data = json.load(f)

        return data.get('full_transcript', '')

    def chunk_transcript(self, text, chunk_size=1000, overlap=100):
        """Chunk transcript into overlapping segments"""
        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)

        return chunks

    def extract_with_patterns(self, text):
        """Extract discourse elements using pattern matching"""
        sentences = sent_tokenize(text)

        questions = []
        claims = []
        evidence = []

        for sent in sentences:
            sent_lower = sent.lower()

            # Check for questions
            if any(re.search(pattern, sent_lower) for pattern in self.question_patterns):
                questions.append(sent)

            # Check for claims
            if any(re.search(pattern, sent_lower) for pattern in self.claim_patterns):
                claims.append(sent)

            # Check for evidence
            if any(re.search(pattern, sent_lower) for pattern in self.evidence_patterns):
                evidence.append(sent)

        return questions, claims, evidence

    def extract_with_llm(self, text, episode_num):
        """Use LLM to extract structured discourse elements"""

        prompt = f"""Extract discourse elements from this podcast transcript segment.

Identify:
1. QUESTIONS: Direct questions asked or topics of inquiry
2. CLAIMS: Statements of fact or belief presented as true
3. ASSERTIONS: Strong declarative statements with confidence
4. EVIDENCE: Specific facts, data, or examples supporting claims

Format as JSON with these fields:
- questions: list of question strings
- claims: list of {{text: string, confidence: float 0-1}}
- assertions: list of {{text: string, subject: string, predicate: string, object: string, confidence: float}}
- evidence: list of {{text: string, supports_claim: string or null}}

Text:
{text[:3000]}  # Limit to avoid token limits

Return only valid JSON."""

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert at discourse analysis and extracting structured information from text."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)
            return result

        except Exception as e:
            print(f"LLM extraction error for episode {episode_num}: {e}")
            return None

    def process_episode(self, episode_num):
        """Process a single episode for discourse extraction"""

        transcript = self.load_transcript(episode_num)
        if not transcript:
            return None

        print(f"Processing episode {episode_num}...")

        episode_discourse = {
            "episode": episode_num,
            "questions": [],
            "claims": [],
            "assertions": [],
            "evidence": []
        }

        # Chunk the transcript
        chunks = self.chunk_transcript(transcript)

        # Process each chunk
        for chunk_idx, chunk in enumerate(tqdm(chunks, desc=f"Episode {episode_num} chunks")):
            # Pattern-based extraction
            questions, claims, evidence = self.extract_with_patterns(chunk)

            # Add pattern-based results
            for q in questions:
                episode_discourse["questions"].append({
                    "text": q,
                    "chunk_id": f"ep{episode_num}_chunk{chunk_idx}",
                    "method": "pattern"
                })

            for c in claims:
                episode_discourse["claims"].append({
                    "text": c,
                    "chunk_id": f"ep{episode_num}_chunk{chunk_idx}",
                    "confidence": 0.7,  # Default confidence for pattern-based
                    "method": "pattern"
                })

            for e in evidence:
                episode_discourse["evidence"].append({
                    "text": e,
                    "chunk_id": f"ep{episode_num}_chunk{chunk_idx}",
                    "method": "pattern"
                })

            # LLM-based extraction (every 3rd chunk to save API costs)
            if chunk_idx % 3 == 0:
                llm_result = self.extract_with_llm(chunk, episode_num)

                if llm_result:
                    # Add LLM questions
                    for q in llm_result.get("questions", []):
                        episode_discourse["questions"].append({
                            "text": q,
                            "chunk_id": f"ep{episode_num}_chunk{chunk_idx}",
                            "method": "llm"
                        })

                    # Add LLM claims
                    for c in llm_result.get("claims", []):
                        if isinstance(c, dict) and c.get("confidence", 0) >= 0.6:
                            episode_discourse["claims"].append({
                                "text": c.get("text", ""),
                                "chunk_id": f"ep{episode_num}_chunk{chunk_idx}",
                                "confidence": c.get("confidence", 0.6),
                                "method": "llm"
                            })

                    # Add LLM assertions
                    for a in llm_result.get("assertions", []):
                        if isinstance(a, dict) and a.get("confidence", 0) >= 0.6:
                            episode_discourse["assertions"].append({
                                "subject": a.get("subject", ""),
                                "predicate": a.get("predicate", ""),
                                "object": a.get("object", ""),
                                "text": a.get("text", ""),
                                "chunk_id": f"ep{episode_num}_chunk{chunk_idx}",
                                "confidence": a.get("confidence", 0.6),
                                "method": "llm"
                            })

                    # Add LLM evidence
                    for e in llm_result.get("evidence", []):
                        if isinstance(e, dict):
                            episode_discourse["evidence"].append({
                                "text": e.get("text", ""),
                                "supports_claim": e.get("supports_claim"),
                                "chunk_id": f"ep{episode_num}_chunk{chunk_idx}",
                                "method": "llm"
                            })

                # Rate limiting for API
                time.sleep(0.5)

        # Deduplicate similar items
        episode_discourse = self.deduplicate_discourse(episode_discourse)

        # Update stats
        self.stats["episodes_processed"] += 1
        self.stats["total_questions"] += len(episode_discourse["questions"])
        self.stats["total_claims"] += len(episode_discourse["claims"])
        self.stats["total_assertions"] += len(episode_discourse["assertions"])
        self.stats["total_evidence"] += len(episode_discourse["evidence"])

        return episode_discourse

    def deduplicate_discourse(self, discourse):
        """Remove duplicate or very similar discourse elements"""

        def similar_texts(text1, text2, threshold=0.9):
            """Check if two texts are similar using Jaccard similarity"""
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())

            if not words1 or not words2:
                return False

            jaccard = len(words1 & words2) / len(words1 | words2)
            return jaccard >= threshold

        # Deduplicate each type
        for field in ["questions", "claims", "evidence"]:
            if field in discourse:
                unique_items = []
                texts_seen = []

                for item in discourse[field]:
                    text = item.get("text", "")

                    # Check if similar text already exists
                    is_duplicate = False
                    for seen_text in texts_seen:
                        if similar_texts(text, seen_text):
                            is_duplicate = True
                            break

                    if not is_duplicate:
                        unique_items.append(item)
                        texts_seen.append(text)

                discourse[field] = unique_items

        # Deduplicate assertions (check subject-predicate-object)
        if "assertions" in discourse:
            unique_assertions = []
            assertions_seen = set()

            for assertion in discourse["assertions"]:
                key = (assertion.get("subject", ""),
                       assertion.get("predicate", ""),
                       assertion.get("object", ""))

                if key not in assertions_seen:
                    unique_assertions.append(assertion)
                    assertions_seen.add(key)

            discourse["assertions"] = unique_assertions

        return discourse

    def merge_with_existing_discourse(self):
        """Merge episode discourse with existing book discourse"""

        # Load existing discourse if it exists
        existing_discourse = {"assertions": [], "questions": [], "claims": [], "evidence": []}

        if MERGED_DISCOURSE_PATH.exists():
            with open(MERGED_DISCOURSE_PATH, 'r') as f:
                existing_discourse = json.load(f)

        # Add episode discourse (handle missing keys gracefully)
        if "assertions" not in existing_discourse:
            existing_discourse["assertions"] = []
        if "questions" not in existing_discourse:
            existing_discourse["questions"] = []
        if "claims" not in existing_discourse:
            existing_discourse["claims"] = []
        if "evidence" not in existing_discourse:
            existing_discourse["evidence"] = []

        existing_discourse["assertions"].extend(self.discourse_elements.get("assertions", []))
        existing_discourse["questions"].extend(self.discourse_elements.get("questions", []))
        existing_discourse["claims"].extend(self.discourse_elements.get("claims", []))
        existing_discourse["evidence"].extend(self.discourse_elements.get("evidence", []))

        # Add metadata
        existing_discourse["metadata"] = {
            "episode_extraction": self.stats,
            "total_elements": {
                "assertions": len(existing_discourse["assertions"]),
                "questions": len(existing_discourse["questions"]),
                "claims": len(existing_discourse["claims"]),
                "evidence": len(existing_discourse["evidence"])
            },
            "last_updated": datetime.now().isoformat()
        }

        # Save merged discourse
        with open(MERGED_DISCOURSE_PATH, 'w') as f:
            json.dump(existing_discourse, f, indent=2)

        print(f"Saved merged discourse to {MERGED_DISCOURSE_PATH}")

        return existing_discourse

    def run(self):
        """Run the discourse extraction pipeline"""
        print("=" * 60)
        print("Episode Discourse Extraction Pipeline")
        print("=" * 60)

        # Determine episode range
        if self.pilot_mode:
            # Pilot on episodes 100-120
            episode_range = range(100, 121)
            print("Running in PILOT MODE: Episodes 100-120")
        else:
            # All episodes except 26 (doesn't exist)
            episode_range = list(range(0, 26)) + list(range(27, 173))
            print(f"Processing all {len(episode_range)} episodes")

        # Process each episode
        all_episode_discourse = []

        for episode_num in tqdm(episode_range, desc="Processing episodes"):
            episode_discourse = self.process_episode(episode_num)

            if episode_discourse:
                all_episode_discourse.append(episode_discourse)

                # Aggregate discourse elements
                self.discourse_elements["questions"].extend(episode_discourse["questions"])
                self.discourse_elements["claims"].extend(episode_discourse["claims"])
                self.discourse_elements["assertions"].extend(episode_discourse["assertions"])
                self.discourse_elements["evidence"].extend(episode_discourse["evidence"])

        # Save episode-specific discourse
        episode_output = {
            "episodes": all_episode_discourse,
            "stats": self.stats,
            "metadata": {
                "pilot_mode": self.pilot_mode,
                "extraction_method": "pattern + llm",
                "min_confidence": 0.6
            }
        }

        with open(DISCOURSE_OUTPUT_PATH, 'w') as f:
            json.dump(episode_output, f, indent=2)

        print(f"\nSaved episode discourse to {DISCOURSE_OUTPUT_PATH}")

        # Merge with existing discourse
        merged = self.merge_with_existing_discourse()

        # Print summary
        print("\n" + "=" * 60)
        print("Extraction Summary")
        print("=" * 60)
        print(f"Episodes processed: {self.stats['episodes_processed']}")
        print(f"Total questions: {self.stats['total_questions']:,}")
        print(f"Total claims: {self.stats['total_claims']:,}")
        print(f"Total assertions: {self.stats['total_assertions']:,}")
        print(f"Total evidence: {self.stats['total_evidence']:,}")
        print(f"\nAverage per episode:")
        if self.stats['episodes_processed'] > 0:
            print(f"  Questions: {self.stats['total_questions'] / self.stats['episodes_processed']:.1f}")
            print(f"  Claims: {self.stats['total_claims'] / self.stats['episodes_processed']:.1f}")
            print(f"  Assertions: {self.stats['total_assertions'] / self.stats['episodes_processed']:.1f}")
            print(f"  Evidence: {self.stats['total_evidence'] / self.stats['episodes_processed']:.1f}")

        print(f"\n✅ Discourse extraction complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract discourse elements from episodes")
    parser.add_argument("--pilot", action="store_true", help="Run pilot on episodes 100-120 only")
    args = parser.parse_args()

    extractor = DiscourseExtractor(pilot_mode=args.pilot)
    extractor.run()