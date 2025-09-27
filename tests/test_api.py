"""
Unit tests for the FastAPI application.
"""

import pytest
from fastapi.testclient import TestClient
from src.api import app


class TestAPI:
    """Test cases for FastAPI endpoints."""

    def setup_method(self):
        """Set up test fixtures."""
        self.client = TestClient(app)
        self.sample_input = {
            "descriptions": [
                "Premium handcrafted leather bag with elegant design.",
                "Sustainable eco-friendly water bottle made from recycled materials."
            ]
        }

    def test_root_endpoint(self):
        """Test the root health check endpoint."""
        response = self.client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "endpoints" in data

    def test_analyze_endpoint_valid_input(self):
        """Test /analyze endpoint with valid input."""
        response = self.client.post("/analyze", json=self.sample_input)

        assert response.status_code == 200
        data = response.json()

        assert "results" in data
        assert "summary" in data
        assert len(data["results"]) == 2

        # Check structure of results
        for result in data["results"]:
            assert "text" in result
            assert "readability" in result
            assert "keyword_density" in result
            assert "uniqueness" in result
            assert "overall_score" in result
            assert "text_index" in result

        # Check summary structure
        summary = data["summary"]
        assert "total_descriptions" in summary
        assert "average_scores" in summary
        assert "quality_distribution" in summary

    def test_analyze_endpoint_custom_keywords(self):
        """Test /analyze endpoint with custom keywords."""
        input_with_keywords = {
            "descriptions": ["Premium handcrafted leather bag with elegant design."],
            "keywords": ["premium", "handcrafted", "elegant"]
        }

        response = self.client.post("/analyze", json=input_with_keywords)

        assert response.status_code == 200
        data = response.json()

        result = data["results"][0]
        keyword_densities = result["keyword_density"]["keyword_densities"]

        # Should include custom keywords
        assert "premium" in keyword_densities
        assert "handcrafted" in keyword_densities
        assert "elegant" in keyword_densities

    def test_analyze_endpoint_empty_descriptions(self):
        """Test /analyze endpoint with empty descriptions list."""
        invalid_input = {"descriptions": []}

        response = self.client.post("/analyze", json=invalid_input)

        assert response.status_code == 422  # Validation error

    def test_analyze_endpoint_missing_descriptions(self):
        """Test /analyze endpoint with missing descriptions field."""
        invalid_input = {"keywords": ["test"]}

        response = self.client.post("/analyze", json=invalid_input)

        assert response.status_code == 422  # Validation error

    def test_recommend_endpoint_valid_input(self):
        """Test /recommend endpoint with valid input."""
        response = self.client.post("/recommend", json=self.sample_input)

        assert response.status_code == 200
        data = response.json()

        assert "recommendations" in data
        assert len(data["recommendations"]) == 2

        # Check structure of recommendations
        for rec in data["recommendations"]:
            assert "description_index" in rec
            assert "description" in rec
            assert "overall_score" in rec
            assert "recommendations" in rec
            assert isinstance(rec["recommendations"], list)

    def test_recommend_endpoint_custom_keywords(self):
        """Test /recommend endpoint with custom keywords."""
        input_with_keywords = {
            "descriptions": ["Short text."],  # Should trigger recommendations
            "keywords": ["missing", "keyword"]
        }

        response = self.client.post("/recommend", json=input_with_keywords)

        assert response.status_code == 200
        data = response.json()

        recommendations = data["recommendations"][0]["recommendations"]
        assert len(recommendations) > 0

    def test_sample_analysis_endpoint(self):
        """Test /sample-analysis endpoint."""
        response = self.client.get("/sample-analysis")

        assert response.status_code == 200
        data = response.json()

        assert "message" in data
        assert "results" in data
        assert "summary" in data
        assert len(data["results"]) == 5  # First 5 products

    def test_docs_endpoint(self):
        """Test that API documentation is accessible."""
        response = self.client.get("/docs")
        assert response.status_code == 200

        response = self.client.get("/openapi.json")
        assert response.status_code == 200

    def test_input_validation_types(self):
        """Test input validation for wrong data types."""
        # Wrong type for descriptions
        invalid_input = {"descriptions": "not a list"}
        response = self.client.post("/analyze", json=invalid_input)
        assert response.status_code == 422

        # Wrong type for keywords
        invalid_input = {
            "descriptions": ["test"],
            "keywords": "not a list"
        }
        response = self.client.post("/analyze", json=invalid_input)
        assert response.status_code == 422

    def test_error_handling(self):
        """Test error handling for various edge cases."""
        # Very long description (should still work)
        long_description = "word " * 10000
        long_input = {"descriptions": [long_description]}

        response = self.client.post("/analyze", json=long_input)
        assert response.status_code == 200

        # Empty string descriptions
        empty_input = {"descriptions": [""]}
        response = self.client.post("/analyze", json=empty_input)
        assert response.status_code == 200

    def test_response_data_types(self):
        """Test that response data types are correct."""
        response = self.client.post("/analyze", json=self.sample_input)
        data = response.json()

        result = data["results"][0]

        # Check numeric types
        assert isinstance(result["overall_score"], (int, float))
        assert isinstance(result["readability"]["readability_score"], (int, float))
        assert isinstance(result["keyword_density"]["density_score"], (int, float))
        assert isinstance(result["uniqueness"]["uniqueness_score"], (int, float))

        # Check string types
        assert isinstance(result["text"], str)
        assert isinstance(result["readability"]["interpretation"], str)
        assert isinstance(result["keyword_density"]["interpretation"], str)
        assert isinstance(result["uniqueness"]["interpretation"], str)

    def test_score_ranges(self):
        """Test that all scores are within expected ranges."""
        response = self.client.post("/analyze", json=self.sample_input)
        data = response.json()

        for result in data["results"]:
            # All scores should be between 0 and 1
            assert 0 <= result["overall_score"] <= 1
            assert 0 <= result["readability"]["readability_score"] <= 1
            assert 0 <= result["keyword_density"]["density_score"] <= 1
            assert 0 <= result["uniqueness"]["uniqueness_score"] <= 1

            # Keyword density percentages should be non-negative
            for density in result["keyword_density"]["keyword_densities"].values():
                assert density >= 0

    def test_multiple_descriptions_uniqueness(self):
        """Test that uniqueness scores work correctly with multiple descriptions."""
        similar_descriptions = [
            "Premium luxury handbag with elegant design.",
            "Premium luxury handbag with elegant styling.",  # Very similar
            "Eco-friendly water bottle made from recycled plastic."  # Different
        ]

        input_data = {"descriptions": similar_descriptions}
        response = self.client.post("/analyze", json=input_data)

        assert response.status_code == 200
        data = response.json()

        results = data["results"]

        # The third description should be more unique
        assert results[2]["uniqueness"]["uniqueness_score"] > results[0]["uniqueness"]["uniqueness_score"]
        assert results[2]["uniqueness"]["uniqueness_score"] > results[1]["uniqueness"]["uniqueness_score"]