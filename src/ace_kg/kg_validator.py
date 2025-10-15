"""
KG Validator Agent

Validates that Curator fixes work by testing on problematic chunks
BEFORE running full corpus extraction.

This provides fast feedback (~2-5 minutes) vs. full extraction (~52 minutes).
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set
from collections import Counter

import pdfplumber

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KGValidatorAgent:
    """
    The KG Validator tests fixes on problem chunks before full extraction.

    Workflow:
    1. Load Reflector analysis to find error pages
    2. Extract only those pages from PDF
    3. Run extraction pipeline on problem chunks
    4. Check if errors are fixed
    5. Report validation results

    This prevents wasting 52 minutes on full extraction if fixes don't work.
    """

    def __init__(
        self,
        playbook_path: str = "/home/claudeuser/yonearth-gaia-chatbot/kg_extraction_playbook"
    ):
        self.playbook_path = Path(playbook_path)
        self.analysis_dir = self.playbook_path / "analysis_reports"
        self.output_dir = self.playbook_path / "output"

    def extract_problem_pages(
        self,
        reflector_analysis: Dict[str, Any],
        max_pages_per_category: int = 5
    ) -> Set[int]:
        """
        Extract page numbers where errors were found.

        Args:
            reflector_analysis: Output from KGReflectorAgent
            max_pages_per_category: Max pages to extract per error category

        Returns:
            Set of page numbers with errors
        """
        problem_pages = set()

        # Extract from issue_categories
        for category in reflector_analysis.get('issue_categories', []):
            category_pages = 0
            for example in category.get('examples', []):
                if category_pages >= max_pages_per_category:
                    break

                page = example.get('page')
                if page and isinstance(page, int):
                    problem_pages.add(page)
                    category_pages += 1

        # Extract from novel_error_patterns
        for pattern in reflector_analysis.get('novel_error_patterns', []):
            pattern_pages = 0
            for example in pattern.get('examples', []):
                if pattern_pages >= max_pages_per_category:
                    break

                page = example.get('page')
                if page and isinstance(page, int):
                    problem_pages.add(page)
                    pattern_pages += 1

        logger.info(f"📍 Found {len(problem_pages)} unique pages with errors")
        logger.info(f"   Pages: {sorted(list(problem_pages))}")

        return problem_pages

    def extract_chunks_from_pages(
        self,
        pdf_path: Path,
        page_numbers: Set[int],
        context_pages: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Extract text chunks from specific pages in PDF.

        Args:
            pdf_path: Path to PDF file
            page_numbers: Page numbers to extract
            context_pages: Number of surrounding pages to include for context

        Returns:
            List of chunks with text and metadata
        """
        logger.info(f"📖 Extracting chunks from {len(page_numbers)} problem pages...")

        # Add context pages
        expanded_pages = set()
        for page in page_numbers:
            for offset in range(-context_pages, context_pages + 1):
                expanded_page = page + offset
                if expanded_page > 0:  # Valid page number
                    expanded_pages.add(expanded_page)

        logger.info(f"   With context: {len(expanded_pages)} pages")

        chunks = []

        with pdfplumber.open(pdf_path) as pdf:
            for page_num in sorted(expanded_pages):
                if page_num <= 0 or page_num > len(pdf.pages):
                    continue

                page = pdf.pages[page_num - 1]  # 0-indexed
                text = page.extract_text()

                if text and len(text.strip()) > 50:
                    # Create chunk (simpler than full chunking - just page-based)
                    chunks.append({
                        'text': text,
                        'pages': [page_num],
                        'word_count': len(text.split()),
                        'is_error_page': page_num in page_numbers
                    })

        logger.info(f"✅ Extracted {len(chunks)} chunks")
        logger.info(f"   - Error pages: {sum(1 for c in chunks if c['is_error_page'])}")
        logger.info(f"   - Context pages: {sum(1 for c in chunks if not c['is_error_page'])}")

        return chunks

    def validate_fixes(
        self,
        reflector_analysis: Dict[str, Any],
        pdf_path: Path,
        extraction_function,  # Function that runs extraction on chunks
        expected_error_reduction: float = 0.5
    ) -> Dict[str, Any]:
        """
        Validate that fixes work by running extraction on problem chunks.

        Args:
            reflector_analysis: Reflector output from previous version
            pdf_path: Path to book PDF
            extraction_function: Function(chunks) -> relationships
            expected_error_reduction: Expected reduction in error rate (0.5 = 50%)

        Returns:
            Validation report with pass/fail and metrics
        """
        logger.info("="*80)
        logger.info("🧪 VALIDATING FIXES ON PROBLEM CHUNKS")
        logger.info("="*80)

        # Step 1: Extract problem pages
        problem_pages = self.extract_problem_pages(reflector_analysis)

        if not problem_pages:
            logger.warning("⚠️  No problem pages found in Reflector analysis")
            return {
                'validation_status': 'skipped',
                'reason': 'no_problem_pages_found'
            }

        # Step 2: Extract chunks from those pages
        chunks = self.extract_chunks_from_pages(pdf_path, problem_pages, context_pages=1)

        if not chunks:
            logger.warning("⚠️  Could not extract chunks from problem pages")
            return {
                'validation_status': 'skipped',
                'reason': 'chunk_extraction_failed'
            }

        # Step 3: Run extraction on problem chunks
        logger.info("")
        logger.info("🔬 Running extraction on problem chunks...")

        try:
            relationships = extraction_function(chunks)
            logger.info(f"✅ Extracted {len(relationships)} relationships from {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"❌ Extraction failed: {e}")
            return {
                'validation_status': 'failed',
                'reason': 'extraction_error',
                'error': str(e)
            }

        # Step 4: Analyze error patterns in mini-extraction
        error_analysis = self._analyze_error_patterns(
            relationships,
            reflector_analysis
        )

        # Step 5: Compare error rates
        original_issues = reflector_analysis.get('quality_summary', {})
        original_error_rate = original_issues.get('issue_rate_percent', 0) / 100

        validation_error_rate = error_analysis['estimated_error_rate']

        logger.info("")
        logger.info("📊 VALIDATION RESULTS:")
        logger.info(f"   Original error rate: {original_error_rate*100:.1f}%")
        logger.info(f"   Validation error rate: {validation_error_rate*100:.1f}%")
        logger.info(f"   Error reduction: {(1 - validation_error_rate/original_error_rate)*100:.1f}%")
        logger.info("")

        # Determine if validation passed
        actual_reduction = 1 - (validation_error_rate / original_error_rate) if original_error_rate > 0 else 1.0
        validation_passed = actual_reduction >= expected_error_reduction

        if validation_passed:
            logger.info("✅ VALIDATION PASSED - Fixes appear to work!")
            logger.info(f"   Error reduction ({actual_reduction*100:.1f}%) meets target ({expected_error_reduction*100:.1f}%)")
        else:
            logger.warning("❌ VALIDATION FAILED - Fixes did not reduce errors enough")
            logger.warning(f"   Error reduction ({actual_reduction*100:.1f}%) below target ({expected_error_reduction*100:.1f}%)")

        result = {
            'validation_status': 'passed' if validation_passed else 'failed',
            'problem_pages_tested': len(problem_pages),
            'chunks_tested': len(chunks),
            'relationships_extracted': len(relationships),
            'original_error_rate': original_error_rate,
            'validation_error_rate': validation_error_rate,
            'error_reduction': actual_reduction,
            'expected_error_reduction': expected_error_reduction,
            'error_analysis': error_analysis,
            'recommendation': 'proceed_with_full_extraction' if validation_passed else 'refine_fixes_first'
        }

        # Save validation report
        self._save_validation_report(result, reflector_analysis.get('metadata', {}))

        return result

    def _analyze_error_patterns(
        self,
        relationships: List[Dict[str, Any]],
        original_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze relationships from validation extraction for error patterns.

        This is a simplified version of Reflector analysis focused on
        checking if known errors are still present.
        """
        total = len(relationships)
        if total == 0:
            return {'estimated_error_rate': 0, 'errors_found': []}

        errors_found = []

        # Check for pronoun sources/targets
        pronoun_patterns = ['he', 'she', 'we', 'they', 'it', 'his', 'her', 'their', 'my', 'our']
        for rel in relationships:
            source_lower = rel.get('source', '').lower()
            target_lower = rel.get('target', '').lower()

            if any(source_lower.startswith(p) for p in pronoun_patterns):
                errors_found.append({
                    'type': 'pronoun_source',
                    'relationship': rel
                })
            if any(target_lower.startswith(p) for p in pronoun_patterns):
                errors_found.append({
                    'type': 'pronoun_target',
                    'relationship': rel
                })

        # Check for list targets (commas in target)
        for rel in relationships:
            target = rel.get('target', '')
            if ',' in target and len(target.split(',')) >= 3:
                errors_found.append({
                    'type': 'list_target',
                    'relationship': rel
                })

        # Check for vague entities
        vague_patterns = ['the amount', 'the process', 'this handbook', 'this book', 'the way', 'the answer']
        for rel in relationships:
            source = rel.get('source', '').lower()
            target = rel.get('target', '').lower()

            if any(vague in source for vague in vague_patterns):
                errors_found.append({
                    'type': 'vague_source',
                    'relationship': rel
                })
            if any(vague in target for vague in vague_patterns):
                errors_found.append({
                    'type': 'vague_target',
                    'relationship': rel
                })

        # Check for duplicate relationships
        rel_signatures = {}
        for rel in relationships:
            sig = (rel.get('source', '').lower().strip(),
                   rel.get('relationship', '').lower().strip(),
                   rel.get('target', '').lower().strip())
            rel_signatures[sig] = rel_signatures.get(sig, 0) + 1

        duplicates = {sig: count for sig, count in rel_signatures.items() if count > 1}
        duplicate_count = sum(count - 1 for count in duplicates.values())

        if duplicate_count > 0:
            errors_found.append({
                'type': 'duplicates',
                'count': duplicate_count
            })

        # Estimate error rate
        error_count = len(errors_found)
        estimated_error_rate = error_count / total if total > 0 else 0

        # Group errors by type
        error_types = Counter(e['type'] for e in errors_found)

        return {
            'total_relationships': total,
            'errors_found': len(errors_found),
            'estimated_error_rate': estimated_error_rate,
            'error_breakdown': dict(error_types),
            'error_examples': errors_found[:10]  # First 10 examples
        }

    def _save_validation_report(
        self,
        validation_result: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> None:
        """Save validation report for review."""
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version = metadata.get('extraction_version', 'unknown')

        validation_dir = self.playbook_path / "validation_reports"
        validation_dir.mkdir(parents=True, exist_ok=True)

        output_path = validation_dir / f"validation_{version}_{timestamp}.json"

        with open(output_path, 'w') as f:
            json.dump(validation_result, f, indent=2)

        logger.info(f"📁 Validation report saved to: {output_path}")


if __name__ == "__main__":
    # Example usage
    validator = KGValidatorAgent()

    # Load Reflector analysis
    analysis_path = Path("/path/to/reflection_v11_2_1_*.json")
    with open(analysis_path) as f:
        reflector_analysis = json.load(f)

    # Define extraction function
    def extraction_func(chunks):
        # This would call the actual extraction pipeline
        # For now, just a placeholder
        return []

    # Run validation
    pdf_path = Path("/path/to/book.pdf")
    result = validator.validate_fixes(
        reflector_analysis=reflector_analysis,
        pdf_path=pdf_path,
        extraction_function=extraction_func,
        expected_error_reduction=0.5  # Expect 50% reduction
    )

    print(json.dumps(result, indent=2))
