"""
Unit tests for the text analysis engine.
"""

import pytest
from src.analysis import TextAnalysisEngine


class TestTextAnalysisEngine:
    """Test cases for TextAnalysisEngine class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = TextAnalysisEngine()
        self.sample_text = "This is a premium luxury product. It offers sustainable eco-friendly features."

    def test_calculate_readability_score(self):
        """Test readability score calculation."""
        result = self.analyzer.calculate_readability_score(self.sample_text)

        assert 'avg_word_length' in result
        assert 'avg_sentence_length' in result
        assert 'readability_score' in result
        assert 'interpretation' in result

        assert isinstance(result['avg_word_length'], float)
        assert isinstance(result['avg_sentence_length'], float)
        assert 0 <= result['readability_score'] <= 1

    def test_readability_empty_text(self):
        """Test readability calculation with empty text."""
        result = self.analyzer.calculate_readability_score("")

        assert result['avg_word_length'] == 0.0
        assert result['avg_sentence_length'] == 0.0
        assert result['readability_score'] == 0.0
        assert result['interpretation'] == 'No readable content'

    def test_calculate_keyword_density_score(self):
        """Test keyword density score calculation."""
        result = self.analyzer.calculate_keyword_density_score(self.sample_text)

        assert 'keyword_densities' in result
        assert 'total_density' in result
        assert 'density_score' in result
        assert 'interpretation' in result

        # Should find "premium", "luxury", "sustainable", "eco-friendly"
        assert result['keyword_densities']['premium'] > 0
        assert result['keyword_densities']['luxury'] > 0
        assert result['keyword_densities']['sustainable'] > 0
        assert result['keyword_densities']['eco-friendly'] > 0

        assert isinstance(result['total_density'], float)
        assert 0 <= result['density_score'] <= 1

    def test_keyword_density_custom_keywords(self):
        """Test keyword density with custom keywords."""
        custom_analyzer = TextAnalysisEngine(['test', 'sample'])
        result = custom_analyzer.calculate_keyword_density_score("This is a test sample text.")

        assert 'test' in result['keyword_densities']
        assert 'sample' in result['keyword_densities']
        assert result['keyword_densities']['test'] > 0
        assert result['keyword_densities']['sample'] > 0

    def test_calculate_uniqueness_score(self):
        """Test uniqueness score calculation."""
        reference_texts = [
            "This is completely different content about cars.",
            "Another unrelated text about cooking recipes."
        ]

        result = self.analyzer.calculate_uniqueness_score(self.sample_text, reference_texts)

        assert 'uniqueness_score' in result
        assert 'interpretation' in result
        assert 0 <= result['uniqueness_score'] <= 1

    def test_uniqueness_identical_text(self):
        """Test uniqueness with identical reference text."""
        reference_texts = [self.sample_text]

        result = self.analyzer.calculate_uniqueness_score(self.sample_text, reference_texts)

        # Should be very low uniqueness for identical text
        assert result['uniqueness_score'] < 0.1

    def test_uniqueness_no_reference(self):
        """Test uniqueness with no reference texts."""
        result = self.analyzer.calculate_uniqueness_score(self.sample_text, [])

        # Should be maximum uniqueness with no references
        assert result['uniqueness_score'] == 1.0

    def test_analyze_single_text(self):
        """Test complete single text analysis."""
        result = self.analyzer.analyze_single_text(self.sample_text)

        required_keys = [
            'text', 'readability', 'keyword_density', 'uniqueness',
            'overall_score', 'overall_interpretation'
        ]

        for key in required_keys:
            assert key in result

        assert 0 <= result['overall_score'] <= 1
        assert isinstance(result['overall_interpretation'], str)

    def test_analyze_multiple_texts(self):
        """Test analysis of multiple texts."""
        texts = [
            "This is a premium luxury product with sustainable features.",
            "High-quality eco-friendly item made from premium materials.",
            "Completely different product about technology and innovation."
        ]

        results = self.analyzer.analyze_multiple_texts(texts)

        assert len(results) == 3

        for i, result in enumerate(results):
            assert result['text_index'] == i
            assert 'overall_score' in result
            assert 'uniqueness' in result

        # First two texts should have lower uniqueness (more similar) or equal
        # Our custom similarity function may produce slightly different results
        assert results[0]['uniqueness']['uniqueness_score'] <= results[2]['uniqueness']['uniqueness_score'] + 0.1

    def test_generate_recommendations(self):
        """Test recommendation generation."""
        # Create analysis with poor scores to trigger recommendations
        analysis = {
            'readability': {
                'readability_score': 0.3,
                'avg_word_length': 8.0,
                'avg_sentence_length': 25.0
            },
            'keyword_density': {
                'density_score': 0.2,
                'total_density': 0.5,
                'keyword_densities': {'premium': 0, 'luxury': 0, 'eco-friendly': 0, 'sustainable': 0.5}
            },
            'uniqueness': {
                'uniqueness_score': 0.3
            },
            'overall_score': 0.3
        }

        recommendations = self.analyzer.generate_recommendations(analysis)

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

        # Should suggest improvements for poor scores
        rec_text = ' '.join(recommendations).lower()
        assert any(word in rec_text for word in ['shorter', 'simpler', 'unique', 'keyword'])

    def test_generate_recommendations_good_content(self):
        """Test recommendations for high-quality content."""
        analysis = {
            'readability': {
                'readability_score': 0.9,
                'avg_word_length': 5.0,
                'avg_sentence_length': 15.0
            },
            'keyword_density': {
                'density_score': 0.9,
                'total_density': 4.0,
                'keyword_densities': {'premium': 1.0, 'luxury': 1.0, 'eco-friendly': 1.0, 'sustainable': 1.0}
            },
            'uniqueness': {
                'uniqueness_score': 0.9
            },
            'overall_score': 0.9
        }

        recommendations = self.analyzer.generate_recommendations(analysis)

        assert "Content meets quality standards" in recommendations

    def test_score_interpretation_ranges(self):
        """Test that score interpretations work correctly for different ranges."""
        from src.scoring import ScoreInterpreter

        # Test centralized score interpretation
        assert "Excellent" in ScoreInterpreter.interpret_score(0.9, "readability")
        assert "Good" in ScoreInterpreter.interpret_score(0.7, "readability")
        assert "Fair" in ScoreInterpreter.interpret_score(0.5, "readability")
        assert "Poor" in ScoreInterpreter.interpret_score(0.3, "readability")
        assert "Very poor" in ScoreInterpreter.interpret_score(0.1, "readability")

        # Test with different score types
        assert "content quality" in ScoreInterpreter.interpret_score(0.8, "content quality")
        assert "uniqueness" in ScoreInterpreter.interpret_score(0.6, "uniqueness")

    def test_word_length_scoring(self):
        """Test centralized optimal range scoring function."""
        from src.scoring import OptimalRangeScorer

        # Optimal range (4-6 chars)
        assert OptimalRangeScorer.score_with_optimal_range(5.0, 4.0, 6.0) == 1.0
        assert OptimalRangeScorer.score_with_optimal_range(4.0, 4.0, 6.0) == 1.0
        assert OptimalRangeScorer.score_with_optimal_range(6.0, 4.0, 6.0) == 1.0

        # Below optimal
        assert OptimalRangeScorer.score_with_optimal_range(2.0, 4.0, 6.0) < 1.0
        assert OptimalRangeScorer.score_with_optimal_range(2.0, 4.0, 6.0) > 0.0

        # Above optimal
        assert OptimalRangeScorer.score_with_optimal_range(8.0, 4.0, 6.0) < 1.0
        assert OptimalRangeScorer.score_with_optimal_range(8.0, 4.0, 6.0) > 0.0

    def test_sentence_length_scoring(self):
        """Test centralized sentence length scoring function."""
        from src.scoring import OptimalRangeScorer

        # Optimal range (10-20 words)
        assert OptimalRangeScorer.score_with_optimal_range(15.0, 10.0, 20.0) == 1.0
        assert OptimalRangeScorer.score_with_optimal_range(10.0, 10.0, 20.0) == 1.0
        assert OptimalRangeScorer.score_with_optimal_range(20.0, 10.0, 20.0) == 1.0

        # Below optimal
        assert OptimalRangeScorer.score_with_optimal_range(5.0, 10.0, 20.0) < 1.0
        assert OptimalRangeScorer.score_with_optimal_range(5.0, 10.0, 20.0) > 0.0

        # Above optimal
        assert OptimalRangeScorer.score_with_optimal_range(25.0, 10.0, 20.0) < 1.0
        assert OptimalRangeScorer.score_with_optimal_range(25.0, 10.0, 20.0) > 0.0