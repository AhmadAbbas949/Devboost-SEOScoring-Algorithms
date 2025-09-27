"""
Unit tests for the data ingestion module.
"""

import pytest
import json
import tempfile
from pathlib import Path

from src.ingest import ProductDataIngestion


class TestProductDataIngestion:
    """Test cases for ProductDataIngestion class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sample_products = [
            {
                "id": 1,
                "title": "Test Product 1",
                "description": "A sample product description."
            },
            {
                "id": 2,
                "title": "Test Product 2",
                "description": "Another sample product description."
            }
        ]

    def test_load_valid_products(self):
        """Test loading valid product data."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.sample_products, f)
            temp_path = f.name

        try:
            ingestion = ProductDataIngestion(temp_path)
            products = ingestion.load_products()

            assert len(products) == 2
            assert products[0]['id'] == 1
            assert products[0]['title'] == "Test Product 1"
            assert products[1]['id'] == 2
        finally:
            Path(temp_path).unlink()

    def test_load_nonexistent_file(self):
        """Test handling of nonexistent data file."""
        ingestion = ProductDataIngestion("nonexistent.json")

        with pytest.raises(FileNotFoundError):
            ingestion.load_products()

    def test_load_invalid_json(self):
        """Test handling of invalid JSON format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            temp_path = f.name

        try:
            ingestion = ProductDataIngestion(temp_path)

            with pytest.raises(ValueError, match="Invalid JSON format"):
                ingestion.load_products()
        finally:
            Path(temp_path).unlink()

    def test_validate_product_valid(self):
        """Test validation of valid product data."""
        ingestion = ProductDataIngestion()
        valid_product = {
            "id": 1,
            "title": "Test Product",
            "description": "Test description"
        }

        result = ingestion._validate_product(valid_product)

        assert result is not None
        assert result['id'] == 1
        assert result['title'] == "Test Product"
        assert result['description'] == "Test description"

    def test_validate_product_missing_fields(self):
        """Test validation of product with missing required fields."""
        ingestion = ProductDataIngestion()

        # Missing description
        invalid_product = {"id": 1, "title": "Test"}
        result = ingestion._validate_product(invalid_product)
        assert result is None

        # Missing title
        invalid_product = {"id": 1, "description": "Test"}
        result = ingestion._validate_product(invalid_product)
        assert result is None

    def test_validate_product_wrong_types(self):
        """Test validation of product with wrong data types."""
        ingestion = ProductDataIngestion()

        # Non-string title
        invalid_product = {"id": 1, "title": 123, "description": "Test"}
        result = ingestion._validate_product(invalid_product)
        assert result is not None  # Should convert to string

        # Non-dict input
        result = ingestion._validate_product("not a dict")
        assert result is None

    def test_get_descriptions(self):
        """Test extraction of product descriptions."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.sample_products, f)
            temp_path = f.name

        try:
            ingestion = ProductDataIngestion(temp_path)
            descriptions = ingestion.get_descriptions()

            assert len(descriptions) == 2
            assert descriptions[0] == "A sample product description."
            assert descriptions[1] == "Another sample product description."
        finally:
            Path(temp_path).unlink()

    def test_get_products_with_metadata(self):
        """Test getting products with computed metadata."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.sample_products, f)
            temp_path = f.name

        try:
            ingestion = ProductDataIngestion(temp_path)
            products = ingestion.get_products_with_metadata()

            assert len(products) == 2
            assert 'word_count' in products[0]
            assert 'char_count' in products[0]
            assert 'sentence_count' in products[0]

            # Check calculated values
            desc = "A sample product description."
            assert products[0]['word_count'] == len(desc.split())
            assert products[0]['char_count'] == len(desc)
        finally:
            Path(temp_path).unlink()

    def test_data_cleaning(self):
        """Test that product data is properly cleaned."""
        dirty_products = [
            {
                "id": "1",  # String ID should be converted
                "title": "  Test Product  ",  # Should be stripped
                "description": "  Test description.  "  # Should be stripped
            }
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(dirty_products, f)
            temp_path = f.name

        try:
            ingestion = ProductDataIngestion(temp_path)
            products = ingestion.load_products()

            assert products[0]['id'] == 1  # Converted to int
            assert products[0]['title'] == "Test Product"  # Stripped
            assert products[0]['description'] == "Test description."  # Stripped
        finally:
            Path(temp_path).unlink()