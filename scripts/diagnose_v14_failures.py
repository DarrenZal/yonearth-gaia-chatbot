#!/usr/bin/env python3
"""
Diagnostic analysis of V14 test failures.

For each failed test case, captures:
- What V14 actually extracted at each stage
- Why expected relationships were filtered/not found
- Root cause categorization for next ACE cycle
"""

import json
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load V14 prompts
PROMPTS_DIR = Path("/home/claudeuser/yonearth-gaia-chatbot/kg_extraction_playbook/prompts")
PASS1_PROMPT_PATH = PROMPTS_DIR / "pass1_extraction_v14.txt"
PASS2_PROMPT_PATH = PROMPTS_DIR / "pass2_evaluation_v14.txt"

# Import postprocessing modules
from src.knowledge_graph.postprocessing.universal.pronoun_resolver import PronounResolver
from src.knowledge_graph.postprocessing.universal.predicate_normalizer import PredicateNormalizer
from src.knowledge_graph.postprocessing.universal.generic_isa_filter import GenericIsAFilter
from src.knowledge_graph.postprocessing.universal.vague_entity_blocker import VagueEntityBlocker
from src.knowledge_graph.postprocessing.content_specific.books.bibliographic_citation_parser import BibliographicCitationParser
from src.knowledge_graph.postprocessing.content_specific.books.praise_quote_detector import PraiseQuoteDetector
from src.knowledge_graph.postprocessing.base import ProcessingContext

# Pydantic models
class ExtractedRelationship(BaseModel):
    source: str
    relationship: str
    target: str
    source_type: str
    target_type: str
    context: str
    page: int
    entity_specificity_score: Optional[float] = None

class ExtractionResult(BaseModel):
    relationships: List[ExtractedRelationship] = Field(default_factory=list)

class EvaluatedRelationship(BaseModel):
    source: str
    relationship: str
    target: str
    source_type: str
    target_type: str
    context: str
    page: int
    p_true: float
    text_confidence: float
    knowledge_plausibility: float
    entity_specificity_score: Optional[float] = None
    classification_flags: List[str] = Field(default_factory=list)

class EvaluationResult(BaseModel):
    relationships: List[EvaluatedRelationship] = Field(default_factory=list)

# Simple wrapper class for postprocessing modules
class RelationshipObject:
    """Simple object wrapper for relationship data that modules can work with"""
    def __init__(self, data: Dict[str, Any]):
        self.source = data.get('source', '')
        self.relationship = data.get('relationship', '')
        self.target = data.get('target', '')
        self.source_type = data.get('source_type', '')
        self.target_type = data.get('target_type', '')
        self.context = data.get('context', '')
        self.evidence_text = data.get('context', '')
        self.page = data.get('page', 0)
        self.p_true = data.get('p_true', 0.0)
        self.text_confidence = data.get('text_confidence', 0.0)
        self.knowledge_plausibility = data.get('knowledge_plausibility', 0.0)
        self.entity_specificity_score = data.get('entity_specificity_score')
        self.classification_flags = data.get('classification_flags', [])
        self.flags = data.get('flags', {})
        self.evidence = data.get('evidence', {'page_number': data.get('page', 0)})
        self.source_surface = data.get('source', '')
        self.target_surface = data.get('target', '')

    def to_dict(self) -> Dict[str, Any]:
        """Convert back to dict"""
        return {
            'source': self.source,
            'relationship': self.relationship,
            'target': self.target,
            'source_type': self.source_type,
            'target_type': self.target_type,
            'context': self.context,
            'page': self.page,
            'p_true': self.p_true,
            'text_confidence': self.text_confidence,
            'knowledge_plausibility': self.knowledge_plausibility,
            'entity_specificity_score': self.entity_specificity_score,
            'classification_flags': self.classification_flags,
            'flags': self.flags
        }


def load_v13_1_test_cases():
    """Load test cases from V13.1 issues"""
    test_cases_file = Path("/tmp/v14_test_cases.json")
    with open(test_cases_file) as f:
        test_data = json.load(f)
    return test_data['test_cases']


def pass1_extract(text: str, page: int, prompt: str) -> List[ExtractedRelationship]:
    """Pass 1: Extract relationships using V14 extraction prompts"""
    try:
        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt.format(text=text)}],
            response_format=ExtractionResult,
            temperature=0.3
        )

        result = response.choices[0].message.parsed
        for rel in result.relationships:
            rel.page = page

        return result.relationships
    except Exception as e:
        print(f"   ❌ Pass 1 error: {e}")
        return []


def pass2_evaluate(relationships: List[ExtractedRelationship], text: str, prompt: str) -> List[EvaluatedRelationship]:
    """Pass 2: Evaluate relationships using V14 evaluation prompts"""
    if not relationships:
        return []

    try:
        rels_json = json.dumps([{
            'source': r.source,
            'relationship': r.relationship,
            'target': r.target,
            'source_type': r.source_type,
            'target_type': r.target_type,
            'context': r.context,
            'entity_specificity_score': r.entity_specificity_score
        } for r in relationships], indent=2)

        batch_size = len(relationships)

        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt.format(
                batch_size=batch_size,
                relationships_json=rels_json
            )}],
            response_format=EvaluationResult,
            temperature=0.0
        )

        result = response.choices[0].message.parsed

        # Preserve page numbers and entity_specificity_score
        for i, eval_rel in enumerate(result.relationships):
            if i < len(relationships):
                eval_rel.page = relationships[i].page
                if relationships[i].entity_specificity_score:
                    eval_rel.entity_specificity_score = relationships[i].entity_specificity_score

        return result.relationships
    except Exception as e:
        print(f"   ❌ Pass 2 error: {e}")
        return []


def pass2_5_postprocess(relationships: List[EvaluatedRelationship]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Pass 2.5: Apply V14 postprocessing modules

    Returns:
        (before_filtering, after_filtering) - both lists of relationship dicts
    """
    if not relationships:
        return [], []

    # Convert to dict format
    rels_dict = [{
        'source': r.source,
        'relationship': r.relationship,
        'target': r.target,
        'source_type': r.source_type,
        'target_type': r.target_type,
        'context': r.context,
        'page': r.page,
        'p_true': r.p_true,
        'text_confidence': r.text_confidence,
        'knowledge_plausibility': r.knowledge_plausibility,
        'entity_specificity_score': r.entity_specificity_score,
        'classification_flags': r.classification_flags
    } for r in relationships]

    # Convert to objects for postprocessing
    rels_objects = [RelationshipObject(r) for r in rels_dict]

    # Initialize postprocessing modules
    pronoun_resolver = PronounResolver()
    predicate_normalizer = PredicateNormalizer()
    vague_entity_blocker = VagueEntityBlocker()
    generic_isa_filter = GenericIsAFilter()
    praise_detector = PraiseQuoteDetector()
    bib_parser = BibliographicCitationParser()

    context = ProcessingContext(
        content_type="book",
        document_metadata={"title": "Soil Stewardship Handbook"}
    )

    # Apply modules
    rels_objects = pronoun_resolver.process_batch(rels_objects, context)
    rels_objects = predicate_normalizer.process_batch(rels_objects, context)
    rels_objects = vague_entity_blocker.process_batch(rels_objects, context)

    # GenericIsAFilter takes dicts only
    rels_dict = [r.to_dict() for r in rels_objects]
    rels_dict = generic_isa_filter.process_batch(rels_dict)
    rels_objects = [RelationshipObject(r) for r in rels_dict]

    rels_objects = praise_detector.process_batch(rels_objects, context)
    rels_objects = bib_parser.process_batch(rels_objects, context)

    # Convert back to dicts
    before_filtering = [r.to_dict() for r in rels_objects]

    # Apply p_true threshold
    after_filtering = [r for r in before_filtering if r.get('p_true', 0) >= 0.5]

    return before_filtering, after_filtering


def diagnose_failure(test_case: Dict, pass1_rels: List, pass2_rels: List,
                     before_filter: List[Dict], after_filter: List[Dict]) -> Dict:
    """
    Diagnose why a test case failed.

    Returns detailed diagnostic info including root cause
    """
    test_id = test_case['test_id']
    category = test_case['category']
    expected = test_case['expected_v14_output']
    v13_output = test_case['v13_1_output']

    diagnosis = {
        'test_id': test_id,
        'category': category,
        'severity': test_case['severity'],
        'v13_output': v13_output,
        'expected_v14': expected,
        'actual_v14_pass1': [{'source': r.source, 'rel': r.relationship, 'target': r.target} for r in pass1_rels],
        'actual_v14_pass2': [{'source': r.source, 'rel': r.relationship, 'target': r.target,
                               'p_true': r.p_true, 'flags': r.classification_flags} for r in pass2_rels],
        'actual_v14_before_filter': [{'source': r['source'], 'rel': r['relationship'], 'target': r['target'],
                                        'p_true': r['p_true'], 'flags': r.get('flags', {})} for r in before_filter],
        'actual_v14_after_filter': [{'source': r['source'], 'rel': r['relationship'], 'target': r['target']}
                                     for r in after_filter],
        'root_cause': None,
        'stage_lost': None
    }

    # Determine root cause
    if isinstance(expected, dict) and expected.get('action') == 'REJECT':
        # Expected to filter, check if still present
        v13_source = v13_output.get('source', '')
        v13_target = v13_output.get('target', '')

        found_in_final = any(r['source'] == v13_source and r['target'] == v13_target
                             for r in after_filter)

        if found_in_final:
            diagnosis['root_cause'] = 'FILTERING_INSUFFICIENT'
            diagnosis['stage_lost'] = 'NOT_FILTERED'
        else:
            diagnosis['root_cause'] = 'SUCCESS'
            diagnosis['stage_lost'] = 'N/A'
    else:
        # Expected correction/preservation
        expected_source = expected.get('source', '')
        expected_target = expected.get('target', '')
        expected_rel = expected.get('relationship', '')

        # Check each stage
        found_pass1 = any(r.source == expected_source and r.target == expected_target
                          for r in pass1_rels)
        found_pass2 = any(r.source == expected_source and r.target == expected_target
                          for r in pass2_rels)
        found_before = any(r['source'] == expected_source and r['target'] == expected_target
                           for r in before_filter)
        found_after = any(r['source'] == expected_source and r['target'] == expected_target
                          for r in after_filter)

        if not found_pass1:
            diagnosis['root_cause'] = 'NOT_EXTRACTED'
            diagnosis['stage_lost'] = 'PASS1'
        elif not found_pass2:
            diagnosis['root_cause'] = 'FILTERED_IN_EVALUATION'
            diagnosis['stage_lost'] = 'PASS2'
            # Check p_true of original in pass2
            for r in pass2_rels:
                if r.source == v13_output.get('source') or r.target == v13_output.get('target'):
                    diagnosis['pass2_p_true'] = r.p_true
                    diagnosis['pass2_flags'] = r.classification_flags
        elif not found_before:
            diagnosis['root_cause'] = 'FILTERED_IN_POSTPROCESSING'
            diagnosis['stage_lost'] = 'PASS2.5_MODULES'
        elif not found_after:
            diagnosis['root_cause'] = 'FILTERED_BY_P_TRUE_THRESHOLD'
            diagnosis['stage_lost'] = 'PASS2.5_FILTER'
            # Find the p_true value
            for r in before_filter:
                if r['source'] == expected_source and r['target'] == expected_target:
                    diagnosis['p_true_value'] = r['p_true']
                    diagnosis['flags'] = r.get('flags', {})
        else:
            diagnosis['root_cause'] = 'SUCCESS'
            diagnosis['stage_lost'] = 'N/A'

    return diagnosis


def main():
    print("="*80)
    print("🔬 V14 DIAGNOSTIC ANALYSIS")
    print("="*80)
    print()

    # Load prompts
    PASS1_PROMPT = PASS1_PROMPT_PATH.read_text()
    PASS2_PROMPT = PASS2_PROMPT_PATH.read_text()

    # Load test cases
    test_cases = load_v13_1_test_cases()

    # Focus on failed tests only
    print("Analyzing failed test cases in detail...")
    print()

    all_diagnostics = []

    for tc in test_cases:
        test_id = tc['test_id']
        category = tc['category']
        evidence = tc['evidence_text']
        page = tc['page']

        print(f"🔍 {test_id}: {category}")

        # Run full pipeline
        pass1_rels = pass1_extract(evidence, page, PASS1_PROMPT)
        pass2_rels = pass2_evaluate(pass1_rels, evidence, PASS2_PROMPT)
        before_filter, after_filter = pass2_5_postprocess(pass2_rels)

        # Diagnose
        diagnosis = diagnose_failure(tc, pass1_rels, pass2_rels, before_filter, after_filter)
        all_diagnostics.append(diagnosis)

        print(f"   Root cause: {diagnosis['root_cause']}")
        print(f"   Lost at: {diagnosis['stage_lost']}")
        if 'p_true_value' in diagnosis:
            print(f"   p_true: {diagnosis['p_true_value']:.3f} (threshold: 0.5)")
        if 'pass2_p_true' in diagnosis:
            print(f"   Pass 2 p_true: {diagnosis['pass2_p_true']:.3f}")
            print(f"   Pass 2 flags: {diagnosis['pass2_flags']}")
        print()

    # Aggregate root cause statistics
    root_causes = {}
    for d in all_diagnostics:
        cause = d['root_cause']
        root_causes[cause] = root_causes.get(cause, 0) + 1

    print("="*80)
    print("📊 ROOT CAUSE SUMMARY")
    print("="*80)
    print()
    for cause, count in sorted(root_causes.items(), key=lambda x: -x[1]):
        print(f"{cause}: {count} test cases")
    print()

    # Save detailed diagnostics
    output_file = Path("/home/claudeuser/yonearth-gaia-chatbot/kg_extraction_playbook/analysis_reports/v14_diagnostics.json")
    with open(output_file, 'w') as f:
        json.dump({
            'diagnostics': all_diagnostics,
            'root_cause_summary': root_causes,
            'analysis_date': '2025-10-14'
        }, f, indent=2)

    print(f"📁 Detailed diagnostics saved to: {output_file}")
    print()

    # Generate recommendations for next ACE cycle
    print("="*80)
    print("💡 RECOMMENDATIONS FOR V14.1 (NEXT ACE CYCLE)")
    print("="*80)
    print()

    if root_causes.get('NOT_EXTRACTED', 0) > 0:
        print(f"✅ Pass 1 Issue: {root_causes['NOT_EXTRACTED']} cases not extracted")
        print("   → V14 extraction prompts may be too conservative")
        print("   → Consider relaxing entity specificity constraints")
        print()

    if root_causes.get('FILTERED_BY_P_TRUE_THRESHOLD', 0) > 0:
        print(f"✅ Pass 2 Issue: {root_causes['FILTERED_BY_P_TRUE_THRESHOLD']} cases filtered by p_true < 0.5")
        print("   → V14 confidence penalties too aggressive")
        print("   → Consider reducing penalty amounts or adjusting threshold")
        print()

    if root_causes.get('FILTERED_IN_POSTPROCESSING', 0) > 0:
        print(f"✅ Pass 2.5 Issue: {root_causes['FILTERED_IN_POSTPROCESSING']} cases filtered by postprocessing")
        print("   → Postprocessing modules may be too strict")
        print("   → Review vague_entity_blocker threshold (0.90)")
        print()

    print("="*80)


if __name__ == "__main__":
    main()
