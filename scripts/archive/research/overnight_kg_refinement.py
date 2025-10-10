#!/usr/bin/env python3
"""
Overnight Knowledge Graph Refinement Processing
Runs multiple refinement tasks in parallel to maximize overnight compute time.
Based on our research synthesis - focuses on the highest-impact operations.
"""

import json
import logging
import time
from pathlib import Path
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'kg_refinement_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path("/home/claudeuser/yonearth-gaia-chatbot/data")
KG_DIR = DATA_DIR / "knowledge_graph"
OUTPUT_DIR = DATA_DIR / "refinement_output"
OUTPUT_DIR.mkdir(exist_ok=True)

def load_knowledge_graph():
    """Load the current knowledge graph"""
    logger.info("Loading knowledge graph...")

    # Load entities
    entities_path = KG_DIR / "graph" / "unified_knowledge_graph.json"
    if entities_path.exists():
        with open(entities_path) as f:
            kg_data = json.load(f)
            entities = kg_data.get('entities', [])
            relationships = kg_data.get('relationships', [])
            logger.info(f"Loaded {len(entities)} entities and {len(relationships)} relationships")
            return entities, relationships
    else:
        logger.error(f"Knowledge graph not found at {entities_path}")
        return [], []

def train_embeddings_overnight(entities, relationships):
    """Train multiple embedding models for ensemble validation"""
    logger.info("Starting embedding training (this will take 15-30 minutes)...")

    try:
        from pykeen.pipeline import pipeline
        from pykeen.triples import TriplesFactory
        import pandas as pd
        import torch

        # Convert to triples format
        triples = []
        for rel in relationships:
            if all(k in rel for k in ['source', 'relationship', 'target']):
                triples.append([
                    rel['source'],
                    rel['relationship'],
                    rel['target']
                ])

        if not triples:
            logger.warning("No valid triples found for embedding training")
            return None

        logger.info(f"Training on {len(triples)} triples")

        # Train multiple models for ensemble
        models_to_train = ['TransE', 'RotatE', 'ComplEx']
        results = {}

        for model_name in models_to_train:
            logger.info(f"Training {model_name} model...")
            start_time = time.time()

            try:
                # Create triples factory
                tf = TriplesFactory.from_labeled_triples(
                    triples=pd.DataFrame(triples, columns=['h', 'r', 't']).values
                )

                # Train model
                result = pipeline(
                    training=tf,
                    model=model_name,
                    epochs=50 if model_name == 'TransE' else 100,  # TransE trains faster
                    embedding_dim=64,  # Smaller for overnight training
                    device='cpu',  # Use CPU for stability
                    random_seed=42
                )

                # Score all triples to find anomalies
                model = result.model
                scores = []
                for triple in triples[:1000]:  # Score first 1000 for speed
                    h, r, t = triple
                    try:
                        score = model.score_hrt_manual(h, r, t).item()
                        scores.append((triple, score))
                    except:
                        continue

                # Find suspicious triples (top 5% worst scores)
                scores.sort(key=lambda x: x[1], reverse=True)
                suspicious = scores[:max(50, len(scores)//20)]

                results[model_name] = {
                    'model': model_name,
                    'training_time': time.time() - start_time,
                    'suspicious_triples': suspicious[:20],  # Top 20 most suspicious
                    'metrics': result.metric_results.to_dict() if hasattr(result, 'metric_results') else {}
                }

                logger.info(f"{model_name} training complete in {time.time()-start_time:.1f}s")
                logger.info(f"Found {len(suspicious)} suspicious triples")

            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                continue

        # Save results
        output_path = OUTPUT_DIR / "embedding_validation_results.json"
        with open(output_path, 'w') as f:
            # Convert to JSON-serializable format
            json_results = {}
            for model_name, data in results.items():
                json_results[model_name] = {
                    'training_time': data['training_time'],
                    'suspicious_count': len(data['suspicious_triples']),
                    'top_suspicious': [
                        {'triple': list(t), 'score': float(s)}
                        for t, s in data['suspicious_triples'][:10]
                    ]
                }
            json.dump(json_results, f, indent=2)

        logger.info(f"Embedding results saved to {output_path}")
        return results

    except ImportError:
        logger.error("PyKEEN not installed. Run: pip install pykeen")
        return None

def find_entity_duplicates(entities):
    """Find potential duplicate entities using multiple string similarity metrics"""
    logger.info("Starting entity deduplication analysis...")

    try:
        import pandas as pd
        from difflib import SequenceMatcher
        import re

        # Extract entity names
        entity_names = []
        for e in entities:
            if isinstance(e, dict):
                name = e.get('name', e.get('entity', ''))
            else:
                name = str(e)
            if name:
                entity_names.append(name)

        logger.info(f"Analyzing {len(entity_names)} entities for duplicates")

        # Find potential duplicates
        duplicates = []
        processed = set()

        for i, name1 in enumerate(entity_names):
            if i % 500 == 0:
                logger.info(f"Processed {i}/{len(entity_names)} entities")

            if name1 in processed:
                continue

            # Normalize for comparison
            norm1 = name1.lower().replace(' ', '').replace('-', '').replace('_', '')

            matches = []
            for j, name2 in enumerate(entity_names[i+1:], i+1):
                if name2 in processed:
                    continue

                norm2 = name2.lower().replace(' ', '').replace('-', '').replace('_', '')

                # Multiple similarity checks
                checks = {
                    'exact_normalized': norm1 == norm2,
                    'substring': norm1 in norm2 or norm2 in norm1,
                    'sequence_similarity': SequenceMatcher(None, name1, name2).ratio(),
                    'token_overlap': len(set(name1.split()) & set(name2.split())) / max(len(name1.split()), len(name2.split())) if name1.split() else 0
                }

                # Check for YonEarth specific patterns
                if ('yonearth' in norm1 and 'yonearth' in norm2) or \
                   ('yon' in norm1 and 'earth' in norm1 and 'yon' in norm2 and 'earth' in norm2):
                    checks['yonearth_variant'] = True

                # High confidence duplicate
                if checks['exact_normalized'] or \
                   checks['sequence_similarity'] > 0.9 or \
                   (checks['substring'] and checks['token_overlap'] > 0.7):
                    matches.append({
                        'entity1': name1,
                        'entity2': name2,
                        'confidence': checks['sequence_similarity'],
                        'checks': checks
                    })
                    processed.add(name2)

            if matches:
                duplicates.append({
                    'canonical': name1,
                    'duplicates': matches
                })

        logger.info(f"Found {len(duplicates)} entity clusters with potential duplicates")

        # Save results
        output_path = OUTPUT_DIR / "entity_duplicates.json"
        with open(output_path, 'w') as f:
            json.dump(duplicates[:100], f, indent=2)  # Top 100 clusters

        logger.info(f"Duplicate analysis saved to {output_path}")
        return duplicates

    except Exception as e:
        logger.error(f"Error in entity deduplication: {e}")
        return []

def validate_geographic_logic(relationships):
    """Check for geographic logic errors like Boulder LOCATED_IN Lafayette"""
    logger.info("Validating geographic relationships...")

    location_errors = []

    # Build location graph
    location_graph = {}
    populations = {
        'boulder': 108000,
        'lafayette': 30000,
        'denver': 715000,
        'colorado': 5800000,
        'california': 39500000
    }

    for rel in relationships:
        if not isinstance(rel, dict):
            continue

        rel_type = rel.get('relationship', '').lower()
        if any(term in rel_type for term in ['located', 'in', 'contains', 'part_of']):
            source = str(rel.get('source', '')).lower()
            target = str(rel.get('target', '')).lower()

            # Check for known problematic patterns
            source_key = source.split()[0] if source else ''
            target_key = target.split()[0] if target else ''

            # Population logic check
            if source_key in populations and target_key in populations:
                if populations[source_key] > populations[target_key] * 1.5:
                    error = {
                        'type': 'population_logic_error',
                        'relationship': rel,
                        'issue': f"{source} (pop: {populations[source_key]}) cannot be located in {target} (pop: {populations[target_key]})",
                        'suggested_fix': 'REVERSE_RELATIONSHIP',
                        'confidence': 0.95
                    }
                    location_errors.append(error)
                    logger.warning(f"Geographic error found: {error['issue']}")

            # Track for cycle detection
            if source not in location_graph:
                location_graph[source] = set()
            location_graph[source].add(target)

    # Detect cycles
    def has_cycle(node, visited, rec_stack):
        visited.add(node)
        rec_stack.add(node)

        for neighbor in location_graph.get(node, []):
            if neighbor not in visited:
                if has_cycle(neighbor, visited, rec_stack):
                    return True
            elif neighbor in rec_stack:
                return True

        rec_stack.remove(node)
        return False

    visited = set()
    for node in location_graph:
        if node not in visited:
            if has_cycle(node, visited, set()):
                location_errors.append({
                    'type': 'cycle_detected',
                    'node': node,
                    'issue': f"Location cycle detected involving {node}",
                    'confidence': 0.9
                })

    logger.info(f"Found {len(location_errors)} geographic logic errors")

    # Save results
    output_path = OUTPUT_DIR / "geographic_errors.json"
    with open(output_path, 'w') as f:
        json.dump(location_errors, f, indent=2)

    logger.info(f"Geographic validation saved to {output_path}")
    return location_errors

def calculate_quality_metrics(entities, relationships):
    """Calculate baseline quality metrics for the knowledge graph"""
    logger.info("Calculating quality metrics...")

    metrics = {
        'timestamp': datetime.now().isoformat(),
        'basic_stats': {
            'total_entities': len(entities),
            'total_relationships': len(relationships),
            'unique_relationship_types': len(set(r.get('relationship', '') for r in relationships if isinstance(r, dict)))
        }
    }

    # Entity statistics
    entity_types = {}
    for e in entities:
        if isinstance(e, dict):
            e_type = e.get('type', 'unknown')
        else:
            e_type = 'string'
        entity_types[e_type] = entity_types.get(e_type, 0) + 1

    metrics['entity_types'] = entity_types

    # Relationship statistics
    rel_types = {}
    confidence_scores = []
    for r in relationships:
        if isinstance(r, dict):
            r_type = r.get('relationship', 'unknown')
            rel_types[r_type] = rel_types.get(r_type, 0) + 1

            if 'confidence' in r:
                confidence_scores.append(r['confidence'])

    metrics['relationship_types'] = dict(sorted(rel_types.items(), key=lambda x: x[1], reverse=True)[:20])

    if confidence_scores:
        metrics['confidence_stats'] = {
            'mean': sum(confidence_scores) / len(confidence_scores),
            'min': min(confidence_scores),
            'max': max(confidence_scores),
            'high_confidence_ratio': len([c for c in confidence_scores if c > 0.8]) / len(confidence_scores)
        }

    # Graph density
    possible_edges = len(entities) * (len(entities) - 1)
    metrics['graph_density'] = len(relationships) / possible_edges if possible_edges > 0 else 0

    logger.info(f"Metrics calculated: {len(entities)} entities, {len(relationships)} relationships")

    # Save metrics
    output_path = OUTPUT_DIR / "quality_metrics_baseline.json"
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Metrics saved to {output_path}")
    return metrics

def run_overnight_pipeline():
    """Main pipeline orchestrator"""
    start_time = time.time()
    logger.info("="*50)
    logger.info("Starting overnight KG refinement pipeline")
    logger.info("="*50)

    # Load knowledge graph
    entities, relationships = load_knowledge_graph()

    if not entities or not relationships:
        logger.error("Failed to load knowledge graph. Exiting.")
        return

    # Run tasks in parallel where possible
    with ThreadPoolExecutor(max_workers=3) as executor:
        # Start all analysis tasks
        futures = {
            'duplicates': executor.submit(find_entity_duplicates, entities),
            'geographic': executor.submit(validate_geographic_logic, relationships),
            'metrics': executor.submit(calculate_quality_metrics, entities, relationships)
        }

        # Embeddings need more resources, run separately
        embedding_result = train_embeddings_overnight(entities, relationships)

        # Collect results
        results = {}
        for name, future in futures.items():
            try:
                results[name] = future.result(timeout=600)  # 10 minute timeout
                logger.info(f"Task '{name}' completed successfully")
            except Exception as e:
                logger.error(f"Task '{name}' failed: {e}")
                results[name] = None

    # Generate summary report
    summary = {
        'execution_time': time.time() - start_time,
        'timestamp': datetime.now().isoformat(),
        'tasks_completed': sum(1 for r in results.values() if r is not None),
        'summary': {
            'duplicates_found': len(results.get('duplicates', [])),
            'geographic_errors': len(results.get('geographic', [])),
            'embeddings_trained': embedding_result is not None
        }
    }

    # Save summary
    summary_path = OUTPUT_DIR / "overnight_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info("="*50)
    logger.info(f"Pipeline completed in {summary['execution_time']:.1f} seconds")
    logger.info(f"Results saved to {OUTPUT_DIR}")
    logger.info("="*50)

    # Print actionable findings
    logger.info("\n🎯 KEY FINDINGS TO REVIEW IN THE MORNING:")

    if results.get('geographic'):
        logger.info(f"\n📍 Found {len(results['geographic'])} geographic errors")
        for error in results['geographic'][:3]:
            if isinstance(error, dict):
                logger.info(f"  - {error.get('issue', error)}")

    if results.get('duplicates'):
        logger.info(f"\n👥 Found {len(results['duplicates'])} potential duplicate entity clusters")
        for dup in results['duplicates'][:3]:
            if isinstance(dup, dict) and 'canonical' in dup:
                logger.info(f"  - {dup['canonical']} has {len(dup.get('duplicates', []))} potential duplicates")

    if embedding_result:
        logger.info(f"\n🧠 Embedding validation complete")
        for model_name, data in embedding_result.items():
            if isinstance(data, dict) and 'suspicious_triples' in data:
                logger.info(f"  - {model_name} found {len(data['suspicious_triples'])} suspicious triples")

    logger.info(f"\n📊 All detailed results saved to: {OUTPUT_DIR}")
    logger.info("\nGood night! Check the results in the morning for actionable refinements. 🌙")

if __name__ == "__main__":
    # Check dependencies
    required_packages = ['pandas', 'numpy']
    optional_packages = ['pykeen', 'torch', 'pyshacl', 'splink']

    missing_required = []
    missing_optional = []

    for pkg in required_packages:
        try:
            __import__(pkg)
        except ImportError:
            missing_required.append(pkg)

    for pkg in optional_packages:
        try:
            __import__(pkg)
        except ImportError:
            missing_optional.append(pkg)

    if missing_required:
        logger.error(f"Missing required packages: {missing_required}")
        logger.error("Install with: pip install " + " ".join(missing_required))
        exit(1)

    if missing_optional:
        logger.warning(f"Missing optional packages (some features disabled): {missing_optional}")
        logger.warning("For full functionality: pip install " + " ".join(missing_optional))

    # Run the pipeline
    run_overnight_pipeline()