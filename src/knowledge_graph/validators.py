"""
Knowledge Graph Validation & Refinement Module

Implements validation and pattern recognition for knowledge graph quality:
- Geographic validation (3-tier: admin hierarchy, population, distance)
- Pattern priors with Laplace smoothing
- Confidence calibration

Based on KG_MASTER_GUIDE_V3.md specifications.
"""

import math
from typing import Dict, Optional, Tuple, List
from collections import defaultdict, Counter


# ============================================================================
# GEOGRAPHIC VALIDATION (3-Tier System)
# ============================================================================

class GeographicValidator:
    """
    Three-tier geographic validation system:
    1. Admin hierarchy (most decisive)
    2. Population sanity check
    3. Distance as fallback
    """

    def __init__(self, geo_cache: Dict = None):
        self.geo_cache = geo_cache or {}

    def get_geo_data(self, entity: str) -> Optional[Dict]:
        """Get geographic data from cache or API"""
        # In production, this would call GeoNames API
        # For now, return cached data if available
        return self.geo_cache.get(entity)

    def is_admin_parent(self, parent_path: List[str], child_path: List[str]) -> bool:
        """Check if parent is in child's administrative hierarchy"""
        if not parent_path or not child_path:
            return True  # Can't verify, assume ok

        # Each element in parent path should match start of child path
        for i, parent_level in enumerate(parent_path):
            if i >= len(child_path) or child_path[i] != parent_level:
                return False
        return True

    def haversine_distance(self, coords1: Tuple[float, float],
                          coords2: Tuple[float, float]) -> float:
        """Calculate distance between two coordinates in km"""
        lat1, lon1 = coords1
        lat2, lon2 = coords2

        R = 6371  # Earth radius in km

        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)

        a = (math.sin(dlat / 2) ** 2 +
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
             math.sin(dlon / 2) ** 2)

        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    def validate_geographic_relationship(self, source: str, relationship: str,
                                        target: str) -> Dict:
        """
        Validate geographic relationship with 3-tier system

        Returns:
            {
                'valid': True/False/None,
                'reason': str,
                'confidence_penalty': float,
                'suggested_correction': dict
            }
        """
        if relationship != "located_in":
            return {'valid': True}

        # Get geo data
        src = self.get_geo_data(source)
        tgt = self.get_geo_data(target)

        if not src or not tgt:
            return {
                'valid': None,
                'reason': 'missing_geo_data',
                'confidence_penalty': 0.05,
                'geo_lookup_needed': True
            }

        # Tier 1: Admin hierarchy check (most decisive)
        src_admin = src.get('admin_path', [])
        tgt_admin = tgt.get('admin_path', [])

        if not self.is_admin_parent(tgt_admin, src_admin):
            return {
                'valid': False,
                'reason': 'admin_hierarchy_mismatch',
                'confidence_penalty': 0.7,
                'source_admin': src_admin,
                'target_admin': tgt_admin
            }

        # Tier 2: Population sanity check (catches reversals)
        src_pop = src.get('population')
        tgt_pop = tgt.get('population')

        if src_pop and tgt_pop:
            if src_pop > 1.2 * tgt_pop:  # 20% tolerance
                return {
                    'valid': False,
                    'reason': 'population_hierarchy_violation',
                    'confidence_penalty': 0.6,
                    'suggested_correction': {
                        'source': target,
                        'relationship': 'located_in',
                        'target': source
                    },
                    'source_population': src_pop,
                    'target_population': tgt_pop
                }

        # Tier 3: Distance check (fallback)
        src_coords = src.get('coords')
        tgt_coords = tgt.get('coords')

        if src_coords and tgt_coords:
            distance_km = self.haversine_distance(src_coords, tgt_coords)

            # Type-specific distance thresholds
            src_type = src.get('type', 'default')
            tgt_type = tgt.get('type', 'default')

            max_distance = {
                ('City', 'State'): 500,
                ('City', 'County'): 100,
                ('Building', 'City'): 50,
            }.get((src_type, tgt_type), 50)

            if distance_km > max_distance:
                return {
                    'valid': False,
                    'reason': f'too_far:{int(distance_km)}km (max:{max_distance}km)',
                    'confidence_penalty': 0.3,
                    'distance_km': distance_km,
                    'max_distance_km': max_distance
                }

        return {'valid': True}


# ============================================================================
# PATTERN PRIORS WITH LAPLACE SMOOTHING
# ============================================================================

class SmoothedPatternPriors:
    """
    Pattern frequency with Laplace smoothing
    Prevents overfitting by smoothing rare patterns
    """

    def __init__(self, existing_relationships: List[Dict] = None, alpha: int = 3):
        """
        Args:
            existing_relationships: List of existing graph relationships
            alpha: Laplace smoothing parameter (default: 3)
        """
        self.alpha = alpha
        self.pattern_counts = defaultdict(int)
        self.total_relationships = 0
        self.num_unique_patterns = 0

        # Entity type cache (to avoid re-resolution)
        self.entity_types = {}

        if existing_relationships:
            self._build_priors(existing_relationships)

    def _get_entity_type(self, entity: str, entity_type: Optional[str] = None) -> str:
        """Get entity type from cache or provided value"""
        if entity in self.entity_types:
            return self.entity_types[entity]

        if entity_type:
            self.entity_types[entity] = entity_type
            return entity_type

        # In production, would resolve via API
        # For now, return UNKNOWN
        return "UNKNOWN"

    def _build_priors(self, relationships: List[Dict]):
        """Build pattern counts from existing relationships"""
        for rel in relationships:
            # Get types
            src_type = self._get_entity_type(
                rel['source'],
                rel.get('source_type')
            )
            tgt_type = self._get_entity_type(
                rel['target'],
                rel.get('target_type')
            )

            # Create pattern tuple
            pattern = (src_type, rel['relationship'], tgt_type)
            self.pattern_counts[pattern] += 1
            self.total_relationships += 1

        self.num_unique_patterns = len(self.pattern_counts)

    def get_prior(self, source: str, relationship: str, target: str,
                  source_type: str = None, target_type: str = None) -> float:
        """
        Get smoothed prior probability for a pattern

        Returns value between 0 and 1, smoothed by Laplace parameter alpha
        """
        if self.total_relationships == 0:
            return 0.5  # Uninformed prior

        # Get types
        src_type = self._get_entity_type(source, source_type)
        tgt_type = self._get_entity_type(target, target_type)

        # Get pattern count
        pattern = (src_type, relationship, tgt_type)
        count = self.pattern_counts.get(pattern, 0)

        # Laplace smoothing
        # P(pattern) = (count + alpha) / (total + alpha * num_patterns)
        numerator = count + self.alpha
        denominator = self.total_relationships + (self.alpha * self.num_unique_patterns)

        prior = numerator / denominator

        # Cap influence at 50% (prevent over-reliance on priors)
        prior = min(prior, 0.5)

        return prior

    def get_stats(self) -> Dict:
        """Get pattern statistics"""
        top_patterns = sorted(
            self.pattern_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        return {
            'total_relationships': self.total_relationships,
            'unique_patterns': self.num_unique_patterns,
            'alpha': self.alpha,
            'top_patterns': [
                {
                    'pattern': f"{p[0]} → {p[1]} → {p[2]}",
                    'count': c
                }
                for p, c in top_patterns
            ]
        }


# ============================================================================
# CALIBRATED CONFIDENCE COMPUTATION
# ============================================================================

def compute_p_true(text_confidence: float, knowledge_plausibility: float,
                  pattern_prior: float, signals_conflict: bool) -> float:
    """
    Calibrated probability combiner (logistic regression with fixed coefficients)

    Args:
        text_confidence: How clearly text states relationship (0-1)
        knowledge_plausibility: How plausible given world knowledge (0-1)
        pattern_prior: How common is this pattern (0-1)
        signals_conflict: Do text and knowledge disagree?

    Returns:
        Calibrated probability that relationship is correct (0-1)
    """
    z = (-1.2 +
         2.1 * text_confidence +
         0.9 * knowledge_plausibility +
         0.6 * pattern_prior -
         0.8 * int(signals_conflict))

    p_true = 1 / (1 + math.exp(-z))

    return p_true


def apply_geo_adjustment(p_true: float, geo_validation: Dict) -> float:
    """Apply geographic validation penalty to p_true"""
    if geo_validation.get('valid') is False:
        penalty = geo_validation.get('confidence_penalty', 0.0)
        p_true = max(0.0, p_true - penalty)
    elif geo_validation.get('valid') is None:
        # Missing geo data - small penalty
        p_true = max(0.0, p_true - 0.05)

    return p_true


# ============================================================================
# EXPECTED CALIBRATION ERROR (ECE)
# ============================================================================

def calculate_ece(predictions: List[Tuple[float, bool]], num_bins: int = 10) -> float:
    """
    Calculate Expected Calibration Error

    Args:
        predictions: List of (p_true, is_correct) tuples
        num_bins: Number of bins for calibration (default: 10)

    Returns:
        ECE value (0-1, lower is better, ≤ 0.07 is well-calibrated)
    """
    if not predictions:
        return 0.0

    # Sort into bins by predicted probability
    bins = [[] for _ in range(num_bins)]

    for p_pred, is_correct in predictions:
        bin_idx = min(int(p_pred * num_bins), num_bins - 1)
        bins[bin_idx].append((p_pred, is_correct))

    # Calculate ECE
    ece = 0.0
    n_total = len(predictions)

    for bin_preds in bins:
        if not bin_preds:
            continue

        # Average predicted probability in bin
        avg_pred = sum(p for p, _ in bin_preds) / len(bin_preds)

        # Actual accuracy in bin
        avg_acc = sum(1 for _, correct in bin_preds if correct) / len(bin_preds)

        # Weighted contribution to ECE
        bin_weight = len(bin_preds) / n_total
        ece += bin_weight * abs(avg_pred - avg_acc)

    return ece


# ============================================================================
# INTEGRATION EXAMPLE
# ============================================================================

def refine_relationship(rel: Dict, geo_validator: GeographicValidator,
                       pattern_priors: SmoothedPatternPriors) -> Dict:
    """
    Complete refinement pipeline for a single relationship

    Args:
        rel: Relationship dict with text_confidence, knowledge_plausibility, etc.
        geo_validator: Configured geographic validator
        pattern_priors: Trained pattern priors model

    Returns:
        Refined relationship with updated p_true and flags
    """
    # Get pattern prior
    pattern_prior = pattern_priors.get_prior(
        rel['source'],
        rel['relationship'],
        rel['target'],
        rel.get('source_type'),
        rel.get('target_type')
    )

    # Compute base p_true
    p_true = compute_p_true(
        rel.get('text_confidence', 0.0),
        rel.get('knowledge_plausibility', 0.0),
        pattern_prior,
        rel.get('signals_conflict', False)
    )

    # Apply geographic validation
    geo_validation = geo_validator.validate_geographic_relationship(
        rel['source'],
        rel['relationship'],
        rel['target']
    )

    p_true = apply_geo_adjustment(p_true, geo_validation)

    # Update relationship
    rel['pattern_prior'] = pattern_prior
    rel['p_true'] = p_true
    rel['geo_validation'] = geo_validation

    # Add flags
    if geo_validation.get('geo_lookup_needed'):
        if 'flags' not in rel:
            rel['flags'] = {}
        rel['flags']['GEO_LOOKUP_NEEDED'] = True

    if geo_validation.get('suggested_correction'):
        rel['suggested_correction'] = geo_validation['suggested_correction']

    return rel
