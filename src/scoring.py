"""
Centralized scoring utilities to eliminate code repetition.
"""

from typing import Dict, Any


class ScoreInterpreter:
    """Centralized score interpretation to avoid repetitive code."""

    # Interpretation thresholds
    THRESHOLDS = {
        'excellent': 0.8,
        'good': 0.6,
        'fair': 0.4,
        'poor': 0.2
    }

    @classmethod
    def interpret_score(cls, score: float, score_type: str, additional_context: str = "") -> str:
        """
        Universal score interpretation function.

        Args:
            score: Score between 0.0 and 1.0
            score_type: Type of score (readability, uniqueness, etc.)
            additional_context: Additional context for interpretation

        Returns:
            Human-readable interpretation string
        """
        if score >= cls.THRESHOLDS['excellent']:
            prefix = "Excellent" if not additional_context else f"Optimal {additional_context}"
        elif score >= cls.THRESHOLDS['good']:
            prefix = "Good" if not additional_context else f"Good {additional_context}"
        elif score >= cls.THRESHOLDS['fair']:
            prefix = "Fair" if not additional_context else f"Moderate {additional_context}"
        elif score >= cls.THRESHOLDS['poor']:
            prefix = "Poor" if not additional_context else f"Low {additional_context}"
        else:
            prefix = "Very poor" if not additional_context else f"Very {additional_context.lower()}"

        return f"{prefix} {score_type}"


class OptimalRangeScorer:
    """Centralized optimal range scoring to eliminate repetitive scoring logic."""

    @staticmethod
    def score_with_optimal_range(value: float, min_optimal: float, max_optimal: float,
                               penalty_factor: float = 1.0) -> float:
        """
        Score a value based on optimal range with standardized penalty calculation.

        Args:
            value: Value to score
            min_optimal: Minimum optimal value
            max_optimal: Maximum optimal value
            penalty_factor: How harshly to penalize values outside optimal range

        Returns:
            Score between 0.0 and 1.0
        """
        if min_optimal <= value <= max_optimal:
            return 1.0
        elif value < min_optimal:
            return max(0.0, value / min_optimal)
        else:  # value > max_optimal
            return max(0.0, 1.0 - (value - max_optimal) / (max_optimal * penalty_factor))


class QualityMetrics:
    """Centralized quality thresholds and calculations."""

    # Standard quality ranges for e-commerce content
    READABILITY_RANGES = {
        'word_length': {'min': 4.0, 'max': 6.0},
        'sentence_length': {'min': 10.0, 'max': 20.0}
    }

    KEYWORD_DENSITY_RANGES = {
        'optimal_min': 2.0,
        'optimal_max': 8.0,
        'penalty_factor': 1.5  # Stricter penalty for over-optimization
    }

    # Weighted scoring for overall quality
    SCORE_WEIGHTS = {
        'readability': 0.4,
        'keyword_density': 0.3,
        'uniqueness': 0.3
    }

    @classmethod
    def calculate_weighted_score(cls, scores: Dict[str, float]) -> float:
        """Calculate weighted overall score from component scores."""
        total = sum(scores.get(component, 0) * weight
                   for component, weight in cls.SCORE_WEIGHTS.items())
        return round(total, 3)


class RecommendationEngine:
    """Centralized recommendation logic to avoid repetitive recommendation patterns."""

    @staticmethod
    def generate_readability_recommendations(avg_word_length: float, avg_sentence_length: float,
                                           score: float) -> list:
        """Generate readability-specific recommendations."""
        recommendations = []
        ranges = QualityMetrics.READABILITY_RANGES

        if score < ScoreInterpreter.THRESHOLDS['good']:
            if avg_word_length > ranges['word_length']['max']:
                recommendations.append("Use simpler, shorter words to improve readability")
            if avg_sentence_length > ranges['sentence_length']['max']:
                recommendations.append("Break long sentences into shorter ones")
            if avg_sentence_length < ranges['sentence_length']['min']:
                recommendations.append("Combine short sentences for better flow")

        return recommendations

    @staticmethod
    def generate_keyword_recommendations(keyword_densities: Dict[str, float],
                                       total_density: float, score: float) -> list:
        """Generate keyword density specific recommendations."""
        recommendations = []
        ranges = QualityMetrics.KEYWORD_DENSITY_RANGES

        if score < ScoreInterpreter.THRESHOLDS['good']:
            if total_density < ranges['optimal_min']:
                missing_keywords = [kw for kw, density in keyword_densities.items() if density == 0]
                if missing_keywords:
                    recommendations.append(
                        f"Consider including keywords: {', '.join(missing_keywords[:2])}"
                    )
            elif total_density > ranges['optimal_max']:
                recommendations.append("Reduce keyword repetition to avoid over-optimization")

        return recommendations

    @staticmethod
    def generate_uniqueness_recommendations(score: float) -> list:
        """Generate uniqueness specific recommendations."""
        if score < ScoreInterpreter.THRESHOLDS['fair']:
            return ["Make content more unique by adding specific details or benefits"]
        return []

    @staticmethod
    def generate_overall_recommendations(overall_score: float) -> list:
        """Generate overall quality recommendations."""
        if overall_score < ScoreInterpreter.THRESHOLDS['fair']:
            return ["Consider rewriting content with focus on clarity and uniqueness"]
        return []