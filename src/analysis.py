"""
Text analysis engine for calculating readability, keyword density, and uniqueness scores.
"""

from typing import List, Dict, Any, Optional
from .utils import (
    tokenize_text,
    calculate_average_word_length,
    calculate_average_sentence_length,
    calculate_keyword_density,
    calculate_uniqueness_score
)
from .scoring import (
    ScoreInterpreter,
    OptimalRangeScorer,
    QualityMetrics,
    RecommendationEngine
)


class TextAnalysisEngine:
    """Main engine for analyzing product descriptions."""

    def __init__(self, target_keywords: Optional[List[str]] = None):
        """
        Initialize the text analysis engine.

        Args:
            target_keywords: List of keywords to analyze for density.
                           Defaults to e-commerce relevant keywords.
        """
        self.target_keywords = target_keywords or [
            "eco-friendly", "sustainable", "premium", "luxury"
        ]

    def calculate_readability_score(self, text: str) -> Dict[str, float]:
        """
        Calculate readability score based on word and sentence length.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary with readability metrics and overall score
        """
        words, sentences = tokenize_text(text)

        if not words or not sentences:
            return {
                'avg_word_length': 0.0,
                'avg_sentence_length': 0.0,
                'readability_score': 0.0,
                'interpretation': 'No readable content'
            }

        avg_word_length = calculate_average_word_length(words)
        avg_sentence_length = calculate_average_sentence_length(sentences)

        # Use centralized optimal range scoring
        ranges = QualityMetrics.READABILITY_RANGES
        word_score = OptimalRangeScorer.score_with_optimal_range(
            avg_word_length, ranges['word_length']['min'], ranges['word_length']['max']
        )
        sentence_score = OptimalRangeScorer.score_with_optimal_range(
            avg_sentence_length, ranges['sentence_length']['min'], ranges['sentence_length']['max']
        )

        readability_score = (word_score + sentence_score) / 2
        interpretation = ScoreInterpreter.interpret_score(readability_score, "readability")

        return {
            'avg_word_length': round(avg_word_length, 2),
            'avg_sentence_length': round(avg_sentence_length, 2),
            'readability_score': round(readability_score, 3),
            'interpretation': interpretation
        }


    def calculate_keyword_density_score(self, text: str) -> Dict[str, Any]:
        """
        Calculate keyword density scores for target keywords.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary with keyword densities and overall score
        """
        keyword_densities = calculate_keyword_density(text, self.target_keywords)
        total_density = sum(keyword_densities.values())

        # Use centralized optimal range scoring
        ranges = QualityMetrics.KEYWORD_DENSITY_RANGES
        density_score = OptimalRangeScorer.score_with_optimal_range(
            total_density, ranges['optimal_min'], ranges['optimal_max'], ranges['penalty_factor']
        )

        # Create context-aware interpretation
        if density_score >= ScoreInterpreter.THRESHOLDS['good']:
            interpretation = f"Optimal keyword usage ({total_density}%)"
        elif total_density < ranges['optimal_min']:
            interpretation = f"Under-optimized keywords ({total_density}%)"
        else:
            interpretation = f"Over-optimized keywords ({total_density}%)"

        return {
            'keyword_densities': keyword_densities,
            'total_density': round(total_density, 2),
            'density_score': round(density_score, 3),
            'interpretation': interpretation
        }


    def calculate_uniqueness_score(self, text: str, reference_texts: List[str]) -> Dict[str, float]:
        """
        Calculate uniqueness score compared to reference texts.

        Args:
            text: Input text to evaluate
            reference_texts: List of texts to compare against

        Returns:
            Dictionary with uniqueness metrics
        """
        uniqueness = calculate_uniqueness_score(text, reference_texts)

        # Use centralized interpretation with context
        if uniqueness >= 0.9:
            interpretation = "Highly unique content"
        elif uniqueness >= 0.7:
            interpretation = "Good uniqueness"
        elif uniqueness >= 0.5:
            interpretation = "Moderate uniqueness"
        elif uniqueness >= 0.3:
            interpretation = "Low uniqueness"
        else:
            interpretation = "Very similar to existing content"

        return {
            'uniqueness_score': round(uniqueness, 3),
            'interpretation': interpretation
        }

    def analyze_single_text(self, text: str, reference_texts: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform complete analysis on a single text.

        Args:
            text: Text to analyze
            reference_texts: Optional list of texts for uniqueness comparison

        Returns:
            Complete analysis results
        """
        reference_texts = reference_texts or []

        readability = self.calculate_readability_score(text)
        keyword_density = self.calculate_keyword_density_score(text)
        uniqueness = self.calculate_uniqueness_score(text, reference_texts)

        # Use centralized weighted scoring
        component_scores = {
            'readability': readability['readability_score'],
            'keyword_density': keyword_density['density_score'],
            'uniqueness': uniqueness['uniqueness_score']
        }
        overall_score = QualityMetrics.calculate_weighted_score(component_scores)
        overall_interpretation = ScoreInterpreter.interpret_score(overall_score, "content quality")

        return {
            'text': text,
            'readability': readability,
            'keyword_density': keyword_density,
            'uniqueness': uniqueness,
            'overall_score': overall_score,
            'overall_interpretation': overall_interpretation
        }


    def analyze_multiple_texts(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze multiple texts with cross-comparison for uniqueness.

        Args:
            texts: List of texts to analyze

        Returns:
            List of analysis results for each text
        """
        results = []

        for i, text in enumerate(texts):
            # Use all other texts as reference for uniqueness
            reference_texts = texts[:i] + texts[i+1:]
            analysis = self.analyze_single_text(text, reference_texts)
            analysis['text_index'] = i
            results.append(analysis)

        return results

    def generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """
        Generate improvement recommendations based on analysis results.

        Args:
            analysis: Analysis results from analyze_single_text

        Returns:
            List of actionable recommendations
        """
        recommendations = []

        # Use centralized recommendation engine
        readability = analysis['readability']
        recommendations.extend(RecommendationEngine.generate_readability_recommendations(
            readability['avg_word_length'],
            readability['avg_sentence_length'],
            readability['readability_score']
        ))

        keyword_density = analysis['keyword_density']
        recommendations.extend(RecommendationEngine.generate_keyword_recommendations(
            keyword_density['keyword_densities'],
            keyword_density['total_density'],
            keyword_density['density_score']
        ))

        uniqueness = analysis['uniqueness']
        recommendations.extend(RecommendationEngine.generate_uniqueness_recommendations(
            uniqueness['uniqueness_score']
        ))

        recommendations.extend(RecommendationEngine.generate_overall_recommendations(
            analysis['overall_score']
        ))

        return recommendations if recommendations else ["Content meets quality standards"]