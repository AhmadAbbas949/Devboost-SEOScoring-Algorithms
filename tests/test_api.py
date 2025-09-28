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
        assert len(data["results"]) == 2

        # Check structure of results
        for result in data["results"]:
            assert "description" in result
            assert "readability" in result
            assert "keyword_density" in result
            assert "similarity" in result

            # Check readability structure
            readability = result["readability"]
            assert "avg_word_length" in readability
            assert "avg_sentence_length" in readability

            # Check keyword density structure
            keyword_density = result["keyword_density"]
            assert "eco_friendly" in keyword_density
            assert "sustainable" in keyword_density
            assert "premium" in keyword_density
            assert "luxury" in keyword_density

            # Check similarity structure
            similarity = result["similarity"]
            assert "uniqueness_score" in similarity
            assert "is_duplicate" in similarity
            assert "max_similarity" in similarity

    def test_analyze_endpoint_custom_keywords(self):
        """Test /analyze endpoint - note: now uses fixed keywords."""
        input_data = {
            "descriptions": ["Premium eco-friendly sustainable luxury handbag."]
        }

        response = self.client.post("/analyze", json=input_data)

        assert response.status_code == 200
        data = response.json()

        result = data["results"][0]
        keyword_density = result["keyword_density"]

        # Should use fixed keywords and detect them
        assert keyword_density["premium"] > 0
        assert keyword_density["eco_friendly"] > 0
        assert keyword_density["sustainable"] > 0
        assert keyword_density["luxury"] > 0

    def test_analyze_endpoint_empty_descriptions(self):
        """Test /analyze endpoint with empty descriptions list - now defaults to products.json."""
        input_data = {"descriptions": []}

        response = self.client.post("/analyze", json=input_data)

        assert response.status_code == 200  # Now successfully processes products.json
        data = response.json()

        assert "results" in data
        assert len(data["results"]) == 50  # Should process all 50 products from products.json

    def test_analyze_endpoint_missing_descriptions(self):
        """Test /analyze endpoint with missing descriptions field - now defaults to products.json."""
        input_data = {"keywords": ["test"]}

        response = self.client.post("/analyze", json=input_data)

        assert response.status_code == 200  # Now successfully processes products.json
        data = response.json()

        assert "results" in data
        assert len(data["results"]) == 50  # Should process all 50 products from products.json

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
            assert "recommendations" in rec
            assert isinstance(rec["recommendations"], list)
            assert len(rec["recommendations"]) == 3  # Should always be exactly 3 suggestions

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
        assert len(data["results"]) == 50  # All products from products.json

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

        # Check numeric types for readability
        assert isinstance(result["readability"]["avg_word_length"], (int, float))
        assert isinstance(result["readability"]["avg_sentence_length"], (int, float))

        # Check numeric types for keyword density
        assert isinstance(result["keyword_density"]["eco_friendly"], (int, float))
        assert isinstance(result["keyword_density"]["sustainable"], (int, float))
        assert isinstance(result["keyword_density"]["premium"], (int, float))
        assert isinstance(result["keyword_density"]["luxury"], (int, float))

        # Check numeric types for similarity
        assert isinstance(result["similarity"]["uniqueness_score"], (int, float))
        assert isinstance(result["similarity"]["max_similarity"], (int, float))
        assert isinstance(result["similarity"]["is_duplicate"], bool)

        # Check string types
        assert isinstance(result["description"], str)

    def test_score_ranges(self):
        """Test that all scores are within expected ranges."""
        response = self.client.post("/analyze", json=self.sample_input)
        data = response.json()

        for result in data["results"]:
            # Similarity scores should be between 0 and 1
            assert 0 <= result["similarity"]["uniqueness_score"] <= 1
            assert 0 <= result["similarity"]["max_similarity"] <= 1

            # Readability values should be positive
            assert result["readability"]["avg_word_length"] > 0
            assert result["readability"]["avg_sentence_length"] > 0

            # Keyword density percentages should be non-negative
            assert result["keyword_density"]["eco_friendly"] >= 0
            assert result["keyword_density"]["sustainable"] >= 0
            assert result["keyword_density"]["premium"] >= 0
            assert result["keyword_density"]["luxury"] >= 0

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
        assert results[2]["similarity"]["uniqueness_score"] > results[0]["similarity"]["uniqueness_score"]
        assert results[2]["similarity"]["uniqueness_score"] > results[1]["similarity"]["uniqueness_score"]

        # First two should be flagged as duplicates (high similarity)
        assert results[0]["similarity"]["is_duplicate"] or results[1]["similarity"]["is_duplicate"]

        # Third should not be a duplicate
        assert not results[2]["similarity"]["is_duplicate"]