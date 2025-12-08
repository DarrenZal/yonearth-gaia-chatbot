#!/usr/bin/env python3
"""
GraphRAG Production Smoke Test

Validates the GraphRAG backend before/after deployment:
- Health check
- Representative queries (global, local, grounded)
- Performance metrics
- KG boost effectiveness

Usage:
    python scripts/smoke_test_graphrag.py [--api-url http://127.0.0.1:8001]
"""

import argparse
import json
import sys
import time
from typing import Dict, Any, List, Tuple

import requests


# Test queries covering different query types
SMOKE_TEST_QUERIES = [
    # (query, expected_type, description)
    ("What are the main themes discussed in the podcast?", "global", "Broad thematic question"),
    ("Who is Aaron Perry?", "local", "Entity-specific question"),
    ("What episode should I listen to about permaculture?", "grounded", "Citation-focused question"),
    ("What is biochar?", "drift", "Concept question with KG entities"),
]

# Thresholds for pass/fail
MAX_LATENCY_SECONDS = 15.0
MIN_ENTITIES_FOR_LOCAL = 1
MIN_COMMUNITIES_FOR_GLOBAL = 1


def check_health(api_url: str) -> Tuple[bool, Dict[str, Any]]:
    """Check GraphRAG health endpoint"""
    try:
        response = requests.get(f"{api_url}/api/graphrag/health", timeout=10)
        data = response.json()

        checks = {
            'initialized': data.get('initialized', False),
            'community_search_ready': data.get('community_search_ready', False),
            'local_search_ready': data.get('local_search_ready', False),
            'vectorstore_ready': data.get('vectorstore_ready', False),
            'gaia_ready': data.get('gaia_ready', False),
        }

        all_healthy = all(checks.values())
        return all_healthy, checks

    except Exception as e:
        return False, {'error': str(e)}


def run_query(api_url: str, query: str, search_mode: str = "auto") -> Dict[str, Any]:
    """Run a single GraphRAG query"""
    try:
        start = time.time()
        response = requests.post(
            f"{api_url}/api/graphrag/chat",
            json={
                "message": query,
                "search_mode": search_mode,
                "personality": "warm_mother"
            },
            timeout=60
        )
        latency = time.time() - start

        data = response.json()
        data['client_latency'] = latency
        return data

    except Exception as e:
        return {'success': False, 'error': str(e), 'client_latency': 0}


def run_comparison(api_url: str, query: str) -> Dict[str, Any]:
    """Run comparison endpoint to check overlap metrics"""
    try:
        response = requests.post(
            f"{api_url}/api/graphrag/compare",
            json={"message": query},
            timeout=120
        )
        return response.json()
    except Exception as e:
        return {'error': str(e)}


def evaluate_result(result: Dict[str, Any], expected_type: str) -> Tuple[bool, List[str]]:
    """Evaluate if a query result passes quality checks"""
    issues = []

    # Check success
    if not result.get('success', False):
        issues.append(f"Query failed: {result.get('error', 'Unknown error')}")
        return False, issues

    # Check latency
    latency = result.get('processing_time', result.get('client_latency', 0))
    if latency > MAX_LATENCY_SECONDS:
        issues.append(f"Latency too high: {latency:.2f}s > {MAX_LATENCY_SECONDS}s")

    # Check response exists
    response = result.get('response', '')
    if not response or len(response) < 50:
        issues.append(f"Response too short: {len(response)} chars")

    # Type-specific checks
    entities = result.get('entities_matched', [])
    communities = result.get('communities_used', [])

    if expected_type == 'local' and len(entities) < MIN_ENTITIES_FOR_LOCAL:
        issues.append(f"Expected entities for local query, got {len(entities)}")

    if expected_type == 'global' and len(communities) < MIN_COMMUNITIES_FOR_GLOBAL:
        issues.append(f"Expected communities for global query, got {len(communities)}")

    # Check for entity noise (short names)
    short_entities = [e['name'] for e in entities if len(e.get('name', '')) <= 2]
    if short_entities:
        issues.append(f"Entity noise detected: {short_entities}")

    # Check relationship types aren't all generic
    relationships = result.get('relationships', [])
    if relationships:
        predicates = set(r.get('predicate', '') for r in relationships)
        if predicates == {'RELATED_TO'}:
            issues.append("All relationships are generic RELATED_TO")

    passed = len(issues) == 0
    return passed, issues


def run_smoke_tests(api_url: str) -> bool:
    """Run all smoke tests and return overall pass/fail"""
    print("=" * 70)
    print("GRAPHRAG SMOKE TEST")
    print(f"API: {api_url}")
    print("=" * 70)
    print()

    all_passed = True
    results_summary = []

    # 1. Health Check
    print("1. HEALTH CHECK")
    print("-" * 40)
    healthy, health_data = check_health(api_url)

    if healthy:
        print("   [PASS] All components healthy")
    else:
        print("   [FAIL] Health check failed:")
        for key, value in health_data.items():
            status = "OK" if value else "FAILED"
            print(f"      - {key}: {status}")
        all_passed = False

    print()

    # 2. Query Tests
    print("2. QUERY TESTS")
    print("-" * 40)

    for query, expected_type, description in SMOKE_TEST_QUERIES:
        print(f"\n   Testing: {description}")
        print(f"   Query: \"{query[:50]}...\"")
        print(f"   Expected type: {expected_type}")

        result = run_query(api_url, query, search_mode="auto")
        passed, issues = evaluate_result(result, expected_type)

        latency = result.get('processing_time', result.get('client_latency', 0))
        entities_count = len(result.get('entities_matched', []))
        communities_count = len(result.get('communities_used', []))

        print(f"   Latency: {latency:.2f}s | Entities: {entities_count} | Communities: {communities_count}")

        if passed:
            print("   [PASS]")
        else:
            print("   [FAIL]")
            for issue in issues:
                print(f"      - {issue}")
            all_passed = False

        results_summary.append({
            'query': query[:40],
            'type': expected_type,
            'passed': passed,
            'latency': latency,
            'entities': entities_count,
            'communities': communities_count
        })

    print()

    # 3. Comparison/Overlap Test
    print("3. COMPARISON METRICS")
    print("-" * 40)

    comparison = run_comparison(api_url, "What is biochar?")
    metrics = comparison.get('comparison_metrics', {})

    if metrics:
        bm25_eps = metrics.get('bm25_episodes', [])
        gr_eps = metrics.get('graphrag_episodes', [])
        overlap = metrics.get('episode_overlap_ratio', 0)

        print(f"   BM25 episodes: {len(bm25_eps)}")
        print(f"   GraphRAG episodes: {len(gr_eps)}")
        print(f"   Overlap ratio: {overlap:.1%}")

        if len(gr_eps) == 0:
            print("   [WARN] GraphRAG returned no episodes - check source extraction")
        else:
            print("   [PASS] Comparison metrics working")
    else:
        print("   [FAIL] Could not get comparison metrics")
        all_passed = False

    print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed_count = sum(1 for r in results_summary if r['passed'])
    total_count = len(results_summary)
    avg_latency = sum(r['latency'] for r in results_summary) / total_count if total_count else 0

    print(f"Queries passed: {passed_count}/{total_count}")
    print(f"Average latency: {avg_latency:.2f}s")
    print(f"Health check: {'PASS' if healthy else 'FAIL'}")
    print()

    if all_passed:
        print("[OVERALL: PASS] - Safe to deploy")
    else:
        print("[OVERALL: FAIL] - Fix issues before deploying")

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="GraphRAG Production Smoke Test")
    parser.add_argument('--api-url', default='http://127.0.0.1:8001', help='API base URL')
    parser.add_argument('--json', action='store_true', help='Output results as JSON')

    args = parser.parse_args()

    passed = run_smoke_tests(args.api_url)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
