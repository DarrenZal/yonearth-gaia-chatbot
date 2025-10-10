#!/usr/bin/env python3
"""
Comprehensive knowledge graph extraction from PDF books using LandingAI.

This script is designed to extract ALL useful knowledge from books about
economic systems, including entities, relationships, claims, evidence, and
discourse elements.

Usage:
    python3 scripts/landingai_book_extraction.py --pdf <path> --output <dir>
"""

import json
import os
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List
import requests
import io
from dotenv import load_dotenv
import pdfplumber

# Load environment variables
load_dotenv()

# Configuration
LANDINGAI_API_KEY = os.getenv("LANDINGAI_API_KEY")
LANDINGAI_PARSE_URL = "https://api.va.landing.ai/v1/ade/parse"
LANDINGAI_EXTRACT_URL = "https://api.va.landing.ai/v1/ade/extract"


def create_comprehensive_schema() -> Dict[str, Any]:
    """
    Create a comprehensive schema for extracting ALL knowledge from economic books.

    This schema captures:
    - Traditional entities (people, orgs, places, concepts)
    - Economic-specific entities (policies, systems, indicators)
    - Discourse elements (claims, evidence, questions, arguments)
    - Relationships between all elements

    Design philosophy: ERR ON THE SIDE OF EXTRACTING TOO MUCH
    """
    schema = {
        "type": "object",
        "properties": {
            # ===== ENTITIES =====
            "entities": {
                "type": "array",
                "description": "ALL entities mentioned in the text - extract comprehensively, err on the side of including more rather than less",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Entity name or identifier"
                        },
                        "type": {
                            "type": "string",
                            "enum": [
                                # Traditional entities
                                "PERSON",           # People, authors, economists, activists
                                "ORG",              # Organizations, companies, institutions
                                "PLACE",            # Countries, cities, regions
                                "CONCEPT",          # Abstract ideas, theories, principles
                                "EVENT",            # Historical events, conferences, movements
                                "TECHNOLOGY",       # Technologies, tools, platforms
                                "PRODUCT",          # Books, reports, frameworks, tools

                                # Economic-specific entities
                                "ECONOMIC_SYSTEM",  # Capitalism, socialism, degrowth economy, etc.
                                "POLICY",           # Specific policies, regulations, laws
                                "INDICATOR",        # GDP, GPI, wellbeing metrics, etc.
                                "PRACTICE",         # Economic practices, methodologies
                                "INSTITUTION",      # Central banks, government bodies, etc.
                                "MOVEMENT",         # Social movements, campaigns
                                "RESOURCE",         # Natural resources, commons, ecosystem services
                                "FRAMEWORK",        # Theoretical frameworks, models
                                "METRIC",           # Measurements, KPIs, indices

                                # Temporal
                                "TIME_PERIOD",      # Decades, eras, historical periods

                                # Other
                                "DATA_SOURCE",      # Studies, datasets, statistics
                                "CASE_STUDY"        # Examples, case studies, real-world instances
                            ],
                            "description": "Entity type - choose the most specific applicable type"
                        },
                        "description": {
                            "type": "string",
                            "description": "Detailed description of the entity, including context and significance"
                        },
                        "aliases": {
                            "type": "array",
                            "description": "Alternative names or abbreviations (e.g., 'GDP' for 'Gross Domestic Product')",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["name", "type"]
                }
            },

            # ===== DISCOURSE ELEMENTS =====
            "discourse_elements": {
                "type": "array",
                "description": "Claims, evidence, arguments, and questions - extract ALL reasoning and argumentation",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": [
                                "CLAIM",            # Assertions, propositions
                                "EVIDENCE",         # Data, facts, studies supporting claims
                                "QUESTION",         # Questions raised by the text
                                "ARGUMENT",         # Logical arguments or reasoning
                                "COUNTERARGUMENT",  # Rebuttals or opposing views
                                "PRINCIPLE",        # Guiding principles or values
                                "RECOMMENDATION",   # Suggested actions or policies
                                "CRITIQUE",         # Criticisms of existing systems/ideas
                                "PREDICTION",       # Forecasts or scenarios
                                "DEFINITION"        # Definitions of key terms
                            ],
                            "description": "Type of discourse element"
                        },
                        "content": {
                            "type": "string",
                            "description": "The full text of the claim, evidence, question, etc."
                        },
                        "subject": {
                            "type": "string",
                            "description": "What this discourse element is about (entity name or concept)"
                        },
                        "stance": {
                            "type": "string",
                            "enum": ["positive", "negative", "neutral", "mixed"],
                            "description": "Author's stance or position"
                        },
                        "strength": {
                            "type": "string",
                            "enum": ["strong", "moderate", "weak", "uncertain"],
                            "description": "Strength of the claim, evidence, or argument"
                        },
                        "context": {
                            "type": "string",
                            "description": "Surrounding context or conditions"
                        }
                    },
                    "required": ["type", "content", "subject"]
                }
            },

            # ===== RELATIONSHIPS =====
            "relationships": {
                "type": "array",
                "description": "ALL relationships between entities - be comprehensive, extract both explicit and implicit relationships",
                "items": {
                    "type": "object",
                    "properties": {
                        "source": {
                            "type": "string",
                            "description": "Source entity name"
                        },
                        "relationship": {
                            "type": "string",
                            "description": "Type of relationship - use descriptive verbs (e.g., 'critiques', 'proposes', 'replaces', 'causes', 'measures', 'implements')"
                        },
                        "target": {
                            "type": "string",
                            "description": "Target entity name"
                        },
                        "context": {
                            "type": "string",
                            "description": "Text snippet or context explaining this relationship"
                        },
                        "relationship_type": {
                            "type": "string",
                            "enum": [
                                # Social/organizational
                                "authored_by", "works_for", "founded", "leads", "collaborates_with",
                                "member_of", "affiliated_with", "influenced_by",

                                # Economic/policy
                                "implements", "proposes", "critiques", "replaces", "reforms",
                                "regulates", "funds", "measures", "impacts", "causes",
                                "benefits_from", "depends_on", "competes_with",

                                # Conceptual
                                "is_example_of", "is_type_of", "is_part_of", "contrasts_with",
                                "supports", "contradicts", "builds_on", "challenges",

                                # Discourse
                                "claims", "provides_evidence_for", "questions", "argues_for",
                                "argues_against", "recommends", "defines",

                                # Temporal
                                "preceded_by", "followed_by", "occurs_during",

                                # Spatial
                                "located_in", "applies_to", "implemented_in",

                                # Other
                                "references", "cites", "describes", "analyzes"
                            ],
                            "description": "Categorized relationship type for easier querying"
                        },
                        "confidence": {
                            "type": "number",
                            "description": "Confidence in this relationship (0.0 to 1.0)",
                            "minimum": 0,
                            "maximum": 1
                        },
                        "directionality": {
                            "type": "string",
                            "enum": ["directed", "bidirectional", "undirected"],
                            "description": "Whether the relationship is one-way or two-way"
                        }
                    },
                    "required": ["source", "relationship", "target"]
                }
            },

            # ===== KEY THEMES =====
            "key_themes": {
                "type": "array",
                "description": "Major themes and topics discussed in this section",
                "items": {
                    "type": "object",
                    "properties": {
                        "theme": {
                            "type": "string",
                            "description": "Name of the theme (e.g., 'degrowth', 'circular economy', 'wellbeing metrics')"
                        },
                        "description": {
                            "type": "string",
                            "description": "How this theme is discussed in the text"
                        },
                        "importance": {
                            "type": "string",
                            "enum": ["primary", "secondary", "tertiary"],
                            "description": "Importance level of this theme"
                        }
                    },
                    "required": ["theme"]
                }
            },

            # ===== QUOTES =====
            "notable_quotes": {
                "type": "array",
                "description": "Important quotes that capture key ideas",
                "items": {
                    "type": "object",
                    "properties": {
                        "quote": {
                            "type": "string",
                            "description": "The exact quote"
                        },
                        "speaker": {
                            "type": "string",
                            "description": "Who said it (if attributed)"
                        },
                        "significance": {
                            "type": "string",
                            "description": "Why this quote is important"
                        }
                    },
                    "required": ["quote"]
                }
            }
        },
        "required": ["entities", "relationships", "discourse_elements"]
    }

    return schema


def parse_pdf_locally(pdf_path: str, chunk_size: int = 2000) -> Dict[str, Any]:
    """
    Use pdfplumber to extract text from PDF locally (avoiding LandingAI rate limits).

    Args:
        pdf_path: Path to PDF file
        chunk_size: Words per chunk

    Returns:
        dict with 'text', 'chunks', 'metadata'
    """
    print(f"📄 Parsing PDF locally: {pdf_path}")

    all_text = []
    page_count = 0

    try:
        with pdfplumber.open(pdf_path) as pdf:
            page_count = len(pdf.pages)
            print(f"   Pages: {page_count}")

            for i, page in enumerate(pdf.pages, 1):
                if i % 10 == 0:
                    print(f"   Processing page {i}/{page_count}...")

                text = page.extract_text()
                if text:
                    all_text.append(text)

        # Combine all text
        full_text = '\n\n'.join(all_text)

        print(f"✅ PDF parsed successfully")
        print(f"   Total text length: {len(full_text)} characters")

        # Create chunks (by word count)
        words = full_text.split()
        chunks = []

        for i in range(0, len(words), chunk_size):
            chunk_text = ' '.join(words[i:i+chunk_size])
            chunks.append({'text': chunk_text})

        print(f"   Created {len(chunks)} chunks ({chunk_size} words each)")

        return {
            'text': full_text,
            'chunks': chunks,
            'metadata': {
                'page_count': page_count,
                'total_words': len(words),
                'total_chars': len(full_text),
                'chunk_count': len(chunks),
                'chunk_size': chunk_size
            }
        }

    except Exception as e:
        print(f"❌ PDF parsing error: {e}")
        raise


def extract_knowledge_from_chunk(chunk_text: str, schema: Dict[str, Any], chunk_num: int, total_chunks: int) -> Dict[str, Any]:
    """
    Extract knowledge from a single chunk using LandingAI Extract API.
    """
    headers = {
        "Authorization": f"Bearer {LANDINGAI_API_KEY}"
    }

    print(f"\n--- Chunk {chunk_num}/{total_chunks} ---")
    print(f"Preview: {chunk_text[:150]}...")

    # Prepare request
    files = {
        'markdown': ('chunk.txt', io.StringIO(chunk_text), 'text/plain')
    }

    data = {
        'schema': json.dumps(schema)
    }

    try:
        response = requests.post(
            LANDINGAI_EXTRACT_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=120
        )
        response.raise_for_status()

        result = response.json()

        # Debug: print full response
        print(f"📋 Full API response keys: {list(result.keys())}")

        extraction = result.get('extraction', {})

        # Print summary
        entities = extraction.get('entities', [])
        relationships = extraction.get('relationships', [])
        discourse = extraction.get('discourse_elements', [])
        themes = extraction.get('key_themes', [])

        print(f"✅ Extracted:")
        print(f"   Entities: {len(entities)}")
        print(f"   Relationships: {len(relationships)}")
        print(f"   Discourse elements: {len(discourse)}")
        print(f"   Themes: {len(themes)}")

        if entities:
            print(f"   Sample entity: {entities[0].get('name')} ({entities[0].get('type')})")
        if relationships:
            rel = relationships[0]
            print(f"   Sample relationship: {rel.get('source')} --[{rel.get('relationship')}]--> {rel.get('target')}")

        # If empty, print more debug info
        if not entities and not relationships:
            print(f"\n⚠️  Empty extraction. Full response:")
            print(f"   {json.dumps(result, indent=2)[:500]}...")

        return extraction

    except requests.exceptions.RequestException as e:
        print(f"❌ Extraction error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response: {e.response.text}")
        return None


def merge_extractions(extractions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge multiple chunk extractions into a unified knowledge graph.
    Deduplicates entities and relationships.
    """
    print(f"\n{'='*80}")
    print(f"Merging {len(extractions)} chunk extractions...")
    print(f"{'='*80}\n")

    merged = {
        "entities": [],
        "relationships": [],
        "discourse_elements": [],
        "key_themes": [],
        "notable_quotes": []
    }

    # Track unique entities and relationships to deduplicate
    seen_entities = {}  # name -> entity dict
    seen_relationships = set()  # (source, relationship, target) tuples
    seen_discourse = set()  # (type, subject, content[:100]) tuples
    seen_themes = set()  # theme names

    for extraction in extractions:
        if not extraction:
            continue

        # Merge entities (deduplicate by name, keep most detailed description)
        for entity in extraction.get('entities', []):
            name = entity.get('name', '').lower().strip()
            if name:
                if name not in seen_entities or len(entity.get('description', '')) > len(seen_entities[name].get('description', '')):
                    seen_entities[name] = entity

        # Merge relationships (deduplicate by source-relationship-target)
        for rel in extraction.get('relationships', []):
            source = rel.get('source', '').lower().strip()
            relationship = rel.get('relationship', '').lower().strip()
            target = rel.get('target', '').lower().strip()

            if source and relationship and target:
                key = (source, relationship, target)
                if key not in seen_relationships:
                    seen_relationships.add(key)
                    merged['relationships'].append(rel)

        # Merge discourse elements (deduplicate by type + subject + content snippet)
        for discourse in extraction.get('discourse_elements', []):
            dtype = discourse.get('type', '')
            subject = discourse.get('subject', '').lower().strip()
            content = discourse.get('content', '')[:100]

            key = (dtype, subject, content)
            if key not in seen_discourse:
                seen_discourse.add(key)
                merged['discourse_elements'].append(discourse)

        # Merge themes (deduplicate by name)
        for theme in extraction.get('key_themes', []):
            theme_name = theme.get('theme', '').lower().strip()
            if theme_name and theme_name not in seen_themes:
                seen_themes.add(theme_name)
                merged['key_themes'].append(theme)

        # Quotes (keep all - they're usually unique)
        merged['notable_quotes'].extend(extraction.get('notable_quotes', []))

    # Convert entities dict back to list
    merged['entities'] = list(seen_entities.values())

    print(f"📊 Merged Results:")
    print(f"   Unique entities: {len(merged['entities'])}")
    print(f"   Unique relationships: {len(merged['relationships'])}")
    print(f"   Discourse elements: {len(merged['discourse_elements'])}")
    print(f"   Key themes: {len(merged['key_themes'])}")
    print(f"   Notable quotes: {len(merged['notable_quotes'])}")

    return merged


def process_book_pdf(pdf_path: str, output_dir: str):
    """
    Main processing pipeline: Parse PDF -> Extract from chunks -> Merge results.
    """
    print(f"\n{'='*80}")
    print(f"LandingAI Comprehensive Book Knowledge Graph Extraction")
    print(f"{'='*80}\n")
    print(f"📚 Book: {pdf_path}")
    print(f"📂 Output: {output_dir}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Step 1: Parse PDF locally (avoiding LandingAI rate limits)
    parsed = parse_pdf_locally(pdf_path, chunk_size=2000)

    # Save parsed text
    full_text = parsed.get('text', '')
    chunks = parsed.get('chunks', [])

    if full_text:
        text_path = output_path / f"{Path(pdf_path).stem}_parsed.txt"
        with open(text_path, 'w') as f:
            f.write(full_text)
        print(f"\n💾 Saved parsed text: {text_path}")
        print(f"   {parsed['metadata']['total_words']} words, {parsed['metadata']['page_count']} pages")

    # Step 2: Create comprehensive schema
    schema = create_comprehensive_schema()
    print(f"\n📋 Created comprehensive extraction schema")
    print(f"   Entity types: 18 types (PERSON, ORG, ECONOMIC_SYSTEM, POLICY, etc.)")
    print(f"   Discourse types: 10 types (CLAIM, EVIDENCE, QUESTION, etc.)")
    print(f"   Relationship types: 35+ categorized types")

    # Step 3: Try SINGLE-CALL extraction first (entire book at once!)
    print(f"\n{'='*80}")
    print(f"Attempting SINGLE-CALL extraction (entire book at once)")
    print(f"{'='*80}\n")
    print(f"📖 Book text: {parsed['metadata']['total_words']} words, {parsed['metadata']['total_chars']} characters")

    # Try extracting from the full text in one call
    extraction_result = extract_knowledge_from_chunk(
        full_text,
        schema,
        chunk_num=1,
        total_chunks=1
    )

    if extraction_result:
        print(f"\n✅ Single-call extraction succeeded!")
        merged = extraction_result
    else:
        print(f"\n⚠️  Single-call extraction failed, falling back to chunked approach...")

        # Step 3b: Extract from each chunk (fallback)
        extractions = []

        for i, chunk in enumerate(chunks, 1):
            chunk_text = chunk.get('text', '') or chunk.get('content', '')

            if not chunk_text or len(chunk_text.strip()) < 100:
                print(f"\n⚠️  Skipping chunk {i} (too short)")
                continue

            # Extract knowledge
            extraction = extract_knowledge_from_chunk(chunk_text, schema, i, len(chunks))

            if extraction:
                extractions.append(extraction)

            # Rate limiting - be gentle with API
            if i < len(chunks):
                time.sleep(3.0)  # 3 second delay between chunks

        # Step 4: Merge all extractions
        merged = merge_extractions(extractions)

    # Step 5: Save results
    book_name = Path(pdf_path).stem

    # Save merged knowledge graph
    kg_path = output_path / f"{book_name}_knowledge_graph.json"
    with open(kg_path, 'w') as f:
        json.dump({
            "book": book_name,
            "source_pdf": str(pdf_path),
            "extraction_model": "landingai-ade-dpt-2",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "total_chunks": len(chunks),
            "chunks_extracted": len(extractions),
            **merged
        }, f, indent=2)

    print(f"\n{'='*80}")
    print(f"✅ Extraction Complete!")
    print(f"{'='*80}")
    print(f"📊 Final Knowledge Graph:")
    print(f"   Entities: {len(merged['entities'])}")
    print(f"   Relationships: {len(merged['relationships'])}")
    print(f"   Discourse elements: {len(merged['discourse_elements'])}")
    print(f"   Key themes: {len(merged['key_themes'])}")
    print(f"   Notable quotes: {len(merged['notable_quotes'])}")
    print(f"\n💾 Saved to: {kg_path}")

    # Generate summary report
    generate_summary_report(merged, output_path / f"{book_name}_summary.md", book_name)


def generate_summary_report(kg: Dict[str, Any], output_path: Path, book_name: str):
    """Generate a human-readable summary report."""

    report = f"""# Knowledge Graph Summary: {book_name}

## Overview

- **Total Entities:** {len(kg['entities'])}
- **Total Relationships:** {len(kg['relationships'])}
- **Discourse Elements:** {len(kg['discourse_elements'])}
- **Key Themes:** {len(kg['key_themes'])}
- **Notable Quotes:** {len(kg['notable_quotes'])}

## Entity Breakdown

"""

    # Count entities by type
    entity_types = {}
    for entity in kg['entities']:
        etype = entity.get('type', 'UNKNOWN')
        entity_types[etype] = entity_types.get(etype, 0) + 1

    for etype, count in sorted(entity_types.items(), key=lambda x: x[1], reverse=True):
        report += f"- **{etype}**: {count}\n"

    report += f"\n## Key Themes\n\n"
    for theme in kg['key_themes'][:20]:
        report += f"- **{theme.get('theme')}**"
        if theme.get('importance'):
            report += f" ({theme['importance']})"
        if theme.get('description'):
            report += f": {theme['description'][:100]}"
        report += "\n"

    report += f"\n## Sample Entities (Top 20)\n\n"
    for entity in kg['entities'][:20]:
        report += f"### {entity.get('name')} ({entity.get('type')})\n"
        if entity.get('description'):
            report += f"{entity['description']}\n"
        report += "\n"

    report += f"\n## Sample Relationships (Top 30)\n\n"
    for rel in kg['relationships'][:30]:
        report += f"- {rel.get('source')} **{rel.get('relationship')}** {rel.get('target')}\n"

    report += f"\n## Discourse Elements by Type\n\n"
    discourse_types = {}
    for discourse in kg['discourse_elements']:
        dtype = discourse.get('type', 'UNKNOWN')
        discourse_types[dtype] = discourse_types.get(dtype, 0) + 1

    for dtype, count in sorted(discourse_types.items(), key=lambda x: x[1], reverse=True):
        report += f"- **{dtype}**: {count}\n"

    with open(output_path, 'w') as f:
        f.write(report)

    print(f"📄 Summary report: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Extract comprehensive knowledge graph from PDF using LandingAI")
    parser.add_argument("--pdf", required=True, help="Path to PDF file")
    parser.add_argument("--output", default="data/knowledge_graph_landingai_books", help="Output directory")

    args = parser.parse_args()

    if not LANDINGAI_API_KEY:
        print("❌ ERROR: LANDINGAI_API_KEY not found in environment variables")
        return 1

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"❌ ERROR: PDF not found: {pdf_path}")
        return 1

    try:
        process_book_pdf(str(pdf_path), args.output)
        return 0
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
