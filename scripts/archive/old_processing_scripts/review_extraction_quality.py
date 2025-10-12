#!/usr/bin/env python3
"""
Review knowledge graph extraction quality and flag problematic relationships.

Identifies:
- Relationships that don't make sense on their own
- Missing context (partial information)
- Vague or meaningless targets (e.g., percentages without context)
- Generic relationships that lose specificity
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict


class ExtractionQualityReviewer:
    """Review extraction quality and flag issues"""

    def __init__(self):
        self.issues = defaultdict(list)

    def review_relationship(self, rel: Dict[str, Any]) -> List[str]:
        """
        Review a single relationship and return list of issues

        Returns:
            List of issue types found (e.g., ['vague_target', 'missing_context'])
        """
        issues = []

        source = rel.get('source', '')
        target = rel.get('target', '')
        relationship = rel.get('relationship', '')
        evidence_text = rel.get('evidence_text', '')

        # Issue 1: Percentage/number without context
        if self._is_percentage_or_number(target):
            if not self._has_clear_context(source, relationship, target, evidence_text):
                issues.append('number_without_context')

        # Issue 2: Very generic/vague entities
        if self._is_too_generic(source) or self._is_too_generic(target):
            issues.append('generic_entity')

        # Issue 3: Lost specificity (entity in text is more specific than extracted)
        if self._lost_specificity(source, evidence_text) or self._lost_specificity(target, evidence_text):
            issues.append('lost_specificity')

        # Issue 4: Relationship doesn't make sense semantically
        if self._semantically_odd(source, relationship, target):
            issues.append('semantically_odd')

        # Issue 5: Extracted entity missing critical context from text
        if self._missing_critical_context(source, target, evidence_text):
            issues.append('missing_critical_context')

        return issues

    def _is_percentage_or_number(self, text: str) -> bool:
        """Check if target is just a number or percentage"""
        text = text.strip()
        # Check for percentages
        if '%' in text:
            return True
        # Check for standalone numbers
        if text.replace('.', '').replace(',', '').isdigit():
            return True
        # Check for number with unit but no clear subject
        if len(text) < 10 and any(char.isdigit() for char in text):
            return True
        return False

    def _has_clear_context(self, source: str, rel: str, target: str, evidence: str) -> bool:
        """Check if a numeric relationship has clear meaning"""
        # If the relationship makes the triple self-explanatory, it's OK
        clear_rels = ['has_value', 'equals', 'measures', 'is_valued_at']
        if any(r in rel.lower() for r in clear_rels):
            return True

        # If source is very specific and includes the unit concept, it's OK
        # e.g., "soil carbon content" + "10%" = OK
        # but "soil" + "10%" = NOT OK
        if 'carbon' in source.lower() and 'carbon' not in evidence[:100].lower():
            return False

        return False

    def _is_too_generic(self, entity: str) -> bool:
        """Check if entity is too generic to be useful"""
        generic_words = [
            'it', 'this', 'that', 'these', 'those',
            'thing', 'stuff', 'area', 'place', 'way', 'method'
        ]
        entity_lower = entity.lower().strip()

        # Single generic word
        if entity_lower in generic_words:
            return True

        # Very short entities that are likely incomplete
        if len(entity) < 3:
            return True

        return False

    def _lost_specificity(self, entity: str, evidence: str) -> bool:
        """Check if extraction lost important specificity from evidence"""
        # Example: evidence says "soil carbon" but entity is just "soil"
        evidence_lower = evidence.lower()
        entity_lower = entity.lower()

        # Check for common cases where context was lost
        if entity_lower == 'soil' and 'soil carbon' in evidence_lower:
            return True
        if entity_lower == 'carbon' and 'soil carbon' in evidence_lower:
            return True
        if entity_lower == 'water' and 'water quality' in evidence_lower:
            return True
        if entity_lower == 'compost' and 'compost tea' in evidence_lower:
            return True

        return False

    def _semantically_odd(self, source: str, rel: str, target: str) -> bool:
        """Check if the triple doesn't make semantic sense"""
        # Check for odd combinations
        # e.g., "soil is increased by 10%" (should be "soil carbon content is increased by 10%")

        if 'increase' in rel.lower() or 'decrease' in rel.lower():
            # These relationships need measurable quantities
            if not any(word in source.lower() for word in ['level', 'amount', 'content', 'rate', 'quantity']):
                # And the source is something that should have a measurable property
                if any(word in source.lower() for word in ['soil', 'carbon', 'water', 'nitrogen']):
                    if self._is_percentage_or_number(target):
                        return True

        return False

    def _missing_critical_context(self, source: str, target: str, evidence: str) -> bool:
        """Check if critical context from evidence is missing in entities"""
        evidence_lower = evidence.lower()

        # Check if evidence has important qualifiers not in entities
        important_phrases = [
            'organic', 'inorganic', 'total', 'available',
            'active', 'passive', 'stable', 'labile'
        ]

        for phrase in important_phrases:
            if phrase in evidence_lower:
                if phrase not in source.lower() and phrase not in target.lower():
                    # Important qualifier in evidence but not in extracted entities
                    return True

        return False

    def review_extraction(self, extraction_file: Path) -> Dict[str, Any]:
        """
        Review entire extraction and generate report

        Returns:
            Dict with issues organized by type
        """
        print(f"Loading extraction from: {extraction_file}")
        with open(extraction_file, 'r') as f:
            data = json.load(f)

        relationships = data.get('relationships', [])
        print(f"Reviewing {len(relationships)} relationships...")

        issue_examples = defaultdict(list)

        for rel in relationships:
            issues = self.review_relationship(rel)

            for issue_type in issues:
                # Store example if we don't have many yet
                if len(issue_examples[issue_type]) < 10:
                    issue_examples[issue_type].append({
                        'source': rel['source'],
                        'relationship': rel['relationship'],
                        'target': rel['target'],
                        'evidence_text': rel.get('evidence_text', ''),
                        'page': rel.get('evidence', {}).get('page_number', 'unknown')
                    })

        # Generate report
        report = {
            'extraction_file': str(extraction_file),
            'total_relationships': len(relationships),
            'issue_summary': {
                issue_type: len(examples)
                for issue_type, examples in issue_examples.items()
            },
            'issue_examples': dict(issue_examples)
        }

        return report

    def format_report(self, report: Dict[str, Any]) -> str:
        """Format report as readable markdown"""
        lines = []
        lines.append(f"# Extraction Quality Review")
        lines.append(f"")
        lines.append(f"**File**: `{report['extraction_file']}`")
        lines.append(f"**Total Relationships**: {report['total_relationships']}")
        lines.append(f"")
        lines.append(f"## Issue Summary")
        lines.append(f"")

        for issue_type, count in sorted(report['issue_summary'].items(), key=lambda x: -x[1]):
            lines.append(f"- **{issue_type}**: {count} cases")

        lines.append(f"")
        lines.append(f"## Issue Examples (up to 10 per type)")
        lines.append(f"")

        issue_descriptions = {
            'number_without_context': 'Percentages or numbers without clear context',
            'generic_entity': 'Entities that are too generic or vague',
            'lost_specificity': 'Extraction lost important details from evidence text',
            'semantically_odd': 'Relationship doesn\'t make semantic sense',
            'missing_critical_context': 'Critical context from evidence missing in entities'
        }

        for issue_type, examples in report['issue_examples'].items():
            lines.append(f"### {issue_type}")
            lines.append(f"")
            lines.append(f"**Description**: {issue_descriptions.get(issue_type, 'No description')}")
            lines.append(f"")

            for i, ex in enumerate(examples, 1):
                lines.append(f"#### Example {i}")
                lines.append(f"")
                lines.append(f"**Triple**: `{ex['source']}` → `{ex['relationship']}` → `{ex['target']}`")
                lines.append(f"")
                lines.append(f"**Evidence** (page {ex['page']}):")
                lines.append(f"> {ex['evidence_text']}")
                lines.append(f"")

                # Explain the specific issue
                issue_explanation = ""
                if issue_type == 'number_without_context':
                    issue_explanation = f"Target '{ex['target']}' is a number/percentage without clear meaning on its own"
                elif issue_type == 'lost_specificity':
                    issue_explanation = f"Extraction lost important context from evidence text"
                elif issue_type == 'semantically_odd':
                    issue_explanation = f"This triple doesn't make semantic sense as stated"
                elif issue_type == 'missing_critical_context':
                    issue_explanation = f"Important qualifiers from evidence are missing"
                else:
                    issue_explanation = f"See description above"

                lines.append(f"**Issue**: {issue_explanation}")

                lines.append(f"")
                lines.append(f"---")
                lines.append(f"")

        return '\n'.join(lines)


def main():
    if len(sys.argv) < 2:
        print("Usage: python review_extraction_quality.py <extraction_file.json>")
        print("\nExample:")
        print("  python review_extraction_quality.py data/knowledge_graph_books_v3_2_2_improved/soil_stewardship_handbook_improved_v3_2_2.json")
        sys.exit(1)

    extraction_file = Path(sys.argv[1])

    if not extraction_file.exists():
        print(f"Error: File not found: {extraction_file}")
        sys.exit(1)

    reviewer = ExtractionQualityReviewer()
    report = reviewer.review_extraction(extraction_file)

    # Save report
    output_file = extraction_file.parent / f"{extraction_file.stem}_quality_review.md"
    markdown_report = reviewer.format_report(report)

    with open(output_file, 'w') as f:
        f.write(markdown_report)

    print(f"\n✅ Quality review complete!")
    print(f"📄 Report saved to: {output_file}")
    print(f"\n📊 Issue Summary:")
    for issue_type, count in sorted(report['issue_summary'].items(), key=lambda x: -x[1]):
        print(f"  - {issue_type}: {count} cases")


if __name__ == '__main__':
    main()
