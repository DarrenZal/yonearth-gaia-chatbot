#!/usr/bin/env python3
"""
Test LandingAI extraction on first 50 pages of OUR BIGGEST DEAL.

This validates the single-call approach on a manageable chunk.
"""

import sys
sys.path.insert(0, '/home/claudeuser/yonearth-gaia-chatbot')

from scripts.landingai_book_extraction import (
    create_comprehensive_schema,
    extract_knowledge_from_chunk,
    generate_summary_report
)
import json
import pdfplumber
from pathlib import Path
import time

# Configuration
PDF_PATH = "/home/claudeuser/yonearth-gaia-chatbot/data/books/OurBiggestDeal/OUR+BIGGEST+DEAL+-+Full+Book+-+Pre-publication+Galley+PDF+to+Share+v2.pdf"
OUTPUT_DIR = Path("data/knowledge_graph_landingai_books")
PAGES_TO_TEST = 50  # Test first 50 pages

def main():
    print(f"\n{'='*80}")
    print(f"LandingAI Test: First {PAGES_TO_TEST} Pages of OUR BIGGEST DEAL")
    print(f"{'='*80}\n")

    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    # Step 1: Extract text from first N pages
    print(f"📄 Extracting text from first {PAGES_TO_TEST} pages...")

    all_text = []
    with pdfplumber.open(PDF_PATH) as pdf:
        total_pages = len(pdf.pages)
        print(f"   Total pages in PDF: {total_pages}")
        print(f"   Extracting pages 1-{min(PAGES_TO_TEST, total_pages)}...")

        for i in range(min(PAGES_TO_TEST, total_pages)):
            if (i + 1) % 10 == 0:
                print(f"   Processing page {i+1}...")

            text = pdf.pages[i].extract_text()
            if text:
                all_text.append(text)

    full_text = '\n\n'.join(all_text)

    print(f"✅ Extracted {len(all_text)} pages with text")
    print(f"   Total characters: {len(full_text):,}")
    print(f"   Estimated words: {len(full_text.split()):,}")

    # Save extracted text
    text_path = OUTPUT_DIR / f"OUR_BIGGEST_DEAL_first_{PAGES_TO_TEST}_pages.txt"
    with open(text_path, 'w') as f:
        f.write(full_text)
    print(f"💾 Saved text: {text_path}")

    # Step 2: Create schema
    schema = create_comprehensive_schema()
    print(f"\n📋 Created comprehensive extraction schema")

    # Step 3: Single-call extraction
    print(f"\n{'='*80}")
    print(f"Attempting SINGLE-CALL extraction")
    print(f"{'='*80}\n")

    extraction = extract_knowledge_from_chunk(
        full_text,
        schema,
        chunk_num=1,
        total_chunks=1
    )

    if not extraction:
        print(f"\n❌ Extraction failed!")
        return 1

    print(f"\n✅ Extraction succeeded!")

    # Step 4: Save results
    output_path = OUTPUT_DIR / f"OUR_BIGGEST_DEAL_first_{PAGES_TO_TEST}_pages_kg.json"

    with open(output_path, 'w') as f:
        json.dump({
            "book": "OUR BIGGEST DEAL",
            "pages_extracted": f"1-{PAGES_TO_TEST}",
            "source_pdf": PDF_PATH,
            "extraction_model": "landingai-ade-dpt-2",
            "single_call": True,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            **extraction
        }, f, indent=2)

    print(f"\n{'='*80}")
    print(f"✅ Test Complete!")
    print(f"{'='*80}")
    print(f"📊 Results:")
    print(f"   Entities: {len(extraction.get('entities', []))}")
    print(f"   Relationships: {len(extraction.get('relationships', []))}")
    print(f"   Discourse elements: {len(extraction.get('discourse_elements', []))}")
    print(f"   Key themes: {len(extraction.get('key_themes', []))}")
    print(f"   Notable quotes: {len(extraction.get('notable_quotes', []))}")
    print(f"\n💾 Saved to: {output_path}")

    # Generate summary
    summary_path = OUTPUT_DIR / f"OUR_BIGGEST_DEAL_first_{PAGES_TO_TEST}_pages_summary.md"
    generate_summary_report(extraction, summary_path, f"OUR BIGGEST DEAL (First {PAGES_TO_TEST} Pages)")
    print(f"📄 Summary: {summary_path}")

    return 0


if __name__ == "__main__":
    exit(main())
