#!/usr/bin/env python3
"""
Comprehensive Knowledge Graph Extraction Review

Reviews ALL extracted relationships AND all pages of the source book to:
1. Identify incorrect relationships with reasons
2. Identify missing knowledge that should have been extracted
3. Generate comprehensive quality report
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict

try:
    import pdfplumber
except ImportError:
    print("ERROR: pdfplumber not installed. Run: pip install pdfplumber")
    sys.exit(1)


class ComprehensiveReviewer:
    """Comprehensive extraction quality reviewer"""

    def __init__(self, extraction_file: Path, pdf_file: Path):
        self.extraction_file = extraction_file
        self.pdf_file = pdf_file

        # Load extraction data
        with open(extraction_file) as f:
            self.extraction_data = json.load(f)

        self.relationships = self.extraction_data['relationships']
        self.total_rels = len(self.relationships)

        # Group relationships by page
        self.rels_by_page = defaultdict(list)
        for rel in self.relationships:
            page = rel['evidence'].get('page_number')
            if page:
                self.rels_by_page[page].append(rel)

        print(f"Loaded {self.total_rels} relationships from {len(self.rels_by_page)} pages")

    def review_relationship_correctness(self, rel: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Review if a relationship is correct
        Returns: (is_correct, reason_if_incorrect)
        """
        source = rel['source']
        relationship = rel['relationship']
        target = rel['target']
        evidence = rel.get('evidence_text', '')

        issues = []

        # Check 1: Number/percentage without context
        if self._is_number_or_percentage(target):
            if not self._has_measurement_context(source, evidence):
                issues.append(
                    f"Target '{target}' is a number/percentage but source '{source}' "
                    f"lacks measurement context (should specify what is being measured)"
                )

        # Check 2: Lost specificity from evidence
        if 'soil carbon' in evidence.lower() and source.lower() == 'soil':
            issues.append(
                f"Lost specificity: evidence mentions 'soil carbon' but extracted as '{source}'"
            )

        # Check 3: Semantically odd (doesn't make sense)
        if self._is_semantically_odd(source, relationship, target):
            issues.append(
                f"Semantically odd: '{source} {relationship} {target}' doesn't make sense"
            )

        # Check 4: Missing important qualifiers from evidence
        missing_qualifiers = self._find_missing_qualifiers(source, target, evidence)
        if missing_qualifiers:
            issues.append(
                f"Missing qualifiers from evidence: {', '.join(missing_qualifiers)}"
            )

        # Check 5: Vague/generic entities
        if self._is_too_generic(source) or self._is_too_generic(target):
            issues.append(
                f"Entity too generic/vague: '{source}' or '{target}'"
            )

        # Check 6: Entity not actually in evidence
        if not self._entity_in_evidence(source, evidence) or not self._entity_in_evidence(target, evidence):
            issues.append(
                f"Entity not found in evidence text"
            )

        if issues:
            return False, " | ".join(issues)
        return True, ""

    def _is_number_or_percentage(self, text: str) -> bool:
        """Check if text is primarily a number or percentage"""
        text = text.strip()
        return '%' in text or (len(text) < 10 and any(c.isdigit() for c in text))

    def _has_measurement_context(self, source: str, evidence: str) -> bool:
        """Check if source has clear measurement context"""
        measurement_indicators = [
            'content', 'level', 'amount', 'quantity', 'rate',
            'concentration', 'percentage', 'fraction', 'ratio'
        ]
        return any(indicator in source.lower() for indicator in measurement_indicators)

    def _is_semantically_odd(self, source: str, rel: str, target: str) -> bool:
        """Check if triple is semantically odd"""
        # Check for nonsensical increase/decrease
        if 'increase' in rel.lower() or 'decrease' in rel.lower():
            if self._is_number_or_percentage(target):
                if not self._has_measurement_context(source, ''):
                    return True
        return False

    def _find_missing_qualifiers(self, source: str, target: str, evidence: str) -> List[str]:
        """Find important qualifiers in evidence missing from entities"""
        important_qualifiers = [
            'organic', 'inorganic', 'natural', 'synthetic',
            'active', 'passive', 'stable', 'labile',
            'total', 'available', 'annual', 'daily',
            'global', 'local', 'regional', 'worldwide'
        ]

        evidence_lower = evidence.lower()
        source_lower = source.lower()
        target_lower = target.lower()

        missing = []
        for qualifier in important_qualifiers:
            if qualifier in evidence_lower:
                if qualifier not in source_lower and qualifier not in target_lower:
                    missing.append(qualifier)

        return missing

    def _is_too_generic(self, entity: str) -> bool:
        """Check if entity is too generic"""
        generic_words = [
            'it', 'this', 'that', 'these', 'those',
            'thing', 'stuff', 'way', 'method'
        ]
        return entity.lower().strip() in generic_words or len(entity) < 3

    def _entity_in_evidence(self, entity: str, evidence: str) -> bool:
        """Check if entity appears in evidence"""
        if not evidence:
            return False
        entity_lower = entity.lower()
        evidence_lower = evidence.lower()

        # Check exact match
        if entity_lower in evidence_lower:
            return True

        # Check word-by-word for multi-word entities
        entity_words = entity_lower.split()
        if len(entity_words) > 1:
            # At least most words should be in evidence
            matches = sum(1 for word in entity_words if word in evidence_lower)
            return matches >= len(entity_words) * 0.7

        return False

    def review_page_for_missing_knowledge(self, page_num: int, page_text: str) -> List[str]:
        """
        Review a page to identify knowledge that should have been extracted but wasn't
        Returns list of missing knowledge items
        """
        missing = []
        extracted_rels = self.rels_by_page.get(page_num, [])

        # Check for key indicators of knowledge
        indicators = {
            'definitions': ['is defined as', 'refers to', 'means', 'is a'],
            'statistics': ['%', 'percent', 'gigatons', 'million', 'billion'],
            'processes': ['process', 'method', 'technique', 'practice'],
            'causes': ['causes', 'leads to', 'results in', 'produces'],
            'benefits': ['benefit', 'advantage', 'helps', 'improves'],
            'problems': ['problem', 'issue', 'challenge', 'threat'],
            'solutions': ['solution', 'approach', 'strategy', 'way to'],
            'organizations': ['organization', 'institute', 'foundation', 'company'],
            'people': ['Dr.', 'Professor', 'founded by', 'created by'],
            'locations': ['located in', 'based in', 'from', 'in'],
        }

        # Count how many knowledge indicators are in the text
        indicators_found = {}
        for category, patterns in indicators.items():
            count = sum(1 for pattern in patterns if pattern.lower() in page_text.lower())
            if count > 0:
                indicators_found[category] = count

        # If page has knowledge indicators but no extractions
        if indicators_found and not extracted_rels:
            missing.append(
                f"Page has knowledge indicators ({', '.join(indicators_found.keys())}) "
                f"but NO relationships were extracted"
            )

        # If page has many indicators but few extractions
        total_indicators = sum(indicators_found.values())
        if total_indicators > 5 and len(extracted_rels) < 3:
            missing.append(
                f"Page has {total_indicators} knowledge indicators but only {len(extracted_rels)} "
                f"relationships extracted - likely missing content"
            )

        # Look for specific patterns that should always be extracted
        if 'founded by' in page_text.lower() or 'founded' in page_text.lower():
            has_founded_rel = any('found' in rel['relationship'].lower() for rel in extracted_rels)
            if not has_founded_rel:
                missing.append("Page mentions 'founded' but no founding relationship extracted")

        if 'located in' in page_text.lower() or 'based in' in page_text.lower():
            has_location_rel = any('locat' in rel['relationship'].lower() for rel in extracted_rels)
            if not has_location_rel:
                missing.append("Page mentions location but no location relationship extracted")

        # Check for lists (often contain rich knowledge)
        list_indicators = ['\n- ', '\n• ', '\n1.', '\n2.', '\n*']
        has_list = any(indicator in page_text for indicator in list_indicators)
        if has_list and len(extracted_rels) < 5:
            missing.append(
                f"Page contains lists but only {len(extracted_rels)} relationships extracted"
            )

        return missing

    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive quality report"""
        print("Generating comprehensive quality report...")

        # Part 1: Review all relationships
        print("  Reviewing all relationships...")
        incorrect_rels = []
        for rel in self.relationships:
            is_correct, reason = self.review_relationship_correctness(rel)
            if not is_correct:
                incorrect_rels.append((rel, reason))

        # Part 2: Review all pages
        print("  Reviewing all pages for missing knowledge...")
        pages_with_missing = []

        with pdfplumber.open(self.pdf_file) as pdf:
            total_pages = len(pdf.pages)
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if not text or len(text) < 100:
                    continue

                missing = self.review_page_for_missing_knowledge(page_num, text)
                if missing:
                    pages_with_missing.append((page_num, text[:500], missing))

                if page_num % 10 == 0:
                    print(f"    Processed {page_num}/{total_pages} pages")

        # Generate report
        report = self._format_comprehensive_report(
            incorrect_rels,
            pages_with_missing,
            total_pages
        )

        return report

    def _format_comprehensive_report(
        self,
        incorrect_rels: List[Tuple[Dict, str]],
        pages_with_missing: List[Tuple[int, str, List[str]]],
        total_pages: int
    ) -> str:
        """Format comprehensive report as markdown"""
        lines = []

        lines.append("# Comprehensive Knowledge Graph Extraction Review")
        lines.append("")
        lines.append(f"**Extraction File**: `{self.extraction_file}`")
        lines.append(f"**Source Book**: `{self.pdf_file.name}`")
        lines.append(f"**Total Pages in Book**: {total_pages}")
        lines.append(f"**Pages with Extractions**: {len(self.rels_by_page)}")
        lines.append(f"**Total Relationships**: {self.total_rels}")
        lines.append("")

        # Summary statistics
        lines.append("## Executive Summary")
        lines.append("")
        lines.append(f"- **Incorrect Relationships**: {len(incorrect_rels)} out of {self.total_rels} ({len(incorrect_rels)/self.total_rels*100:.1f}%)")
        lines.append(f"- **Pages Fully Skipped**: {total_pages - len(self.rels_by_page)} pages with no extractions")
        lines.append(f"- **Pages with Missing Knowledge**: {len(pages_with_missing)} pages likely missing content")
        lines.append(f"- **Coverage**: {len(self.rels_by_page)/total_pages*100:.1f}% of pages have at least one extraction")
        lines.append("")

        # Part 1: Incorrect relationships
        lines.append("## Part 1: Incorrect Relationships")
        lines.append("")
        lines.append(f"Found **{len(incorrect_rels)}** incorrect relationships:")
        lines.append("")

        if not incorrect_rels:
            lines.append("✅ No incorrect relationships found!")
            lines.append("")
        else:
            for i, (rel, reason) in enumerate(incorrect_rels, 1):
                lines.append(f"### {i}. Incorrect Relationship")
                lines.append("")
                lines.append(f"**Triple**: `{rel['source']}` → `{rel['relationship']}` → `{rel['target']}`")
                lines.append("")
                lines.append(f"**Page**: {rel['evidence'].get('page_number', 'unknown')}")
                lines.append("")
                lines.append(f"**Evidence**:")
                lines.append(f"> {rel.get('evidence_text', 'No evidence')[:300]}...")
                lines.append("")
                lines.append(f"**Issue**: {reason}")
                lines.append("")
                lines.append("---")
                lines.append("")

        # Part 2: Missing knowledge by page
        lines.append("## Part 2: Missing Knowledge by Page")
        lines.append("")
        lines.append(f"Found **{len(pages_with_missing)}** pages with likely missing extractions:")
        lines.append("")

        if not pages_with_missing:
            lines.append("✅ No obvious missing knowledge detected!")
            lines.append("")
        else:
            for page_num, text_sample, missing_items in pages_with_missing:
                lines.append(f"### Page {page_num}")
                lines.append("")
                lines.append(f"**Relationships Extracted**: {len(self.rels_by_page.get(page_num, []))}")
                lines.append("")
                lines.append(f"**Issues**:")
                for item in missing_items:
                    lines.append(f"- {item}")
                lines.append("")
                lines.append(f"**Text Sample**:")
                lines.append(f"```")
                lines.append(text_sample[:400])
                lines.append(f"...```")
                lines.append("")
                lines.append("---")
                lines.append("")

        # Part 3: Pages completely skipped
        skipped_pages = set(range(1, total_pages + 1)) - set(self.rels_by_page.keys())
        if skipped_pages:
            lines.append("## Part 3: Pages Completely Skipped")
            lines.append("")
            lines.append(f"**{len(skipped_pages)}** pages had NO extractions at all:")
            lines.append("")

            # Group consecutive pages
            skipped_sorted = sorted(skipped_pages)
            ranges = []
            start = skipped_sorted[0]
            end = start

            for page in skipped_sorted[1:]:
                if page == end + 1:
                    end = page
                else:
                    if start == end:
                        ranges.append(f"{start}")
                    else:
                        ranges.append(f"{start}-{end}")
                    start = page
                    end = page

            # Add last range
            if start == end:
                ranges.append(f"{start}")
            else:
                ranges.append(f"{start}-{end}")

            lines.append(f"**Pages**: {', '.join(ranges)}")
            lines.append("")
            lines.append("**Note**: These pages may contain:")
            lines.append("- Front matter (title, copyright, table of contents)")
            lines.append("- Images/diagrams without extractable text")
            lines.append("- Chapter dividers")
            lines.append("- OR actual content that was missed")
            lines.append("")

        # Recommendations
        lines.append("## Recommendations")
        lines.append("")

        if incorrect_rels:
            lines.append("### Fix Incorrect Relationships")
            lines.append("")
            lines.append(f"1. **Update extraction prompts** to address the {len(incorrect_rels)} incorrect relationships")
            lines.append("2. **Add validation rules** to catch these issues automatically")
            lines.append("3. **Re-extract** after prompt improvements")
            lines.append("")

        if pages_with_missing:
            lines.append("### Improve Coverage")
            lines.append("")
            lines.append(f"1. **Review {len(pages_with_missing)} pages** flagged for missing knowledge")
            lines.append("2. **Adjust chunking strategy** - some pages may have been partially chunked")
            lines.append("3. **Lower extraction threshold** if filtering out too many valid relationships")
            lines.append("")

        if skipped_pages:
            lines.append("### Increase Page Coverage")
            lines.append("")
            lines.append(f"1. **Manually review** a sample of the {len(skipped_pages)} skipped pages")
            lines.append("2. **Verify** if they contain extractable knowledge")
            lines.append("3. **Adjust chunking or extraction logic** if valuable pages were missed")
            lines.append("")

        return '\n'.join(lines)


def main():
    if len(sys.argv) < 3:
        print("Usage: python comprehensive_extraction_review.py <extraction.json> <source.pdf>")
        print("\nExample:")
        print("  python comprehensive_extraction_review.py \\")
        print("    data/knowledge_graph_books_v3_2_2_improved/soil_stewardship_handbook_improved_v3_2_2.json \\")
        print("    data/books/soil-stewardship-handbook/Soil-Stewardship-Handbook-eBook.pdf")
        sys.exit(1)

    extraction_file = Path(sys.argv[1])
    pdf_file = Path(sys.argv[2])

    if not extraction_file.exists():
        print(f"ERROR: Extraction file not found: {extraction_file}")
        sys.exit(1)

    if not pdf_file.exists():
        print(f"ERROR: PDF file not found: {pdf_file}")
        sys.exit(1)

    reviewer = ComprehensiveReviewer(extraction_file, pdf_file)
    report = reviewer.generate_comprehensive_report()

    # Save report
    output_file = extraction_file.parent / f"{extraction_file.stem}_comprehensive_review.md"
    with open(output_file, 'w') as f:
        f.write(report)

    print(f"\n✅ Comprehensive review complete!")
    print(f"📄 Report saved to: {output_file}")


if __name__ == '__main__':
    main()
