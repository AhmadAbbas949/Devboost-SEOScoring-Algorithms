"""
Data ingestion module for loading and preparing Shopify product data.
"""

import json
from pathlib import Path
from typing import List, Dict, Any


class ProductDataIngestion:
    """Handles loading and validation of Shopify product data."""

    def __init__(self, data_path: str = "data/products.json"):
        """Initialize with path to product data file."""
        self.data_path = Path(data_path)

    def load_products(self) -> List[Dict[str, Any]]:
        """
        Load products from JSON file and validate structure.

        Returns:
            List of product dictionaries with id, title, and description fields.

        Raises:
            FileNotFoundError: If the data file doesn't exist.
            ValueError: If the data format is invalid.
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Product data file not found: {self.data_path}")

        try:
            with open(self.data_path, 'r', encoding='utf-8') as file:
                products = json.load(file)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in {self.data_path}: {e}")

        if not isinstance(products, list):
            raise ValueError("Product data must be a list of products")

        validated_products = []
        for product in products:
            validated_product = self._validate_product(product)
            if validated_product:
                validated_products.append(validated_product)

        return validated_products

    def _validate_product(self, product: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and clean individual product data.

        Args:
            product: Raw product dictionary

        Returns:
            Validated product dictionary or None if invalid
        """
        required_fields = ['id', 'title', 'description']

        if not isinstance(product, dict):
            return None

        for field in required_fields:
            if field not in product:
                return None
            if not isinstance(product[field], (str, int)):
                return None

        return {
            'id': int(product['id']),
            'title': str(product['title']).strip(),
            'description': str(product['description']).strip()
        }

    def get_descriptions(self) -> List[str]:
        """
        Extract just the product descriptions for analysis.

        Returns:
            List of product description strings.
        """
        products = self.load_products()
        return [product['description'] for product in products]

    def get_products_with_metadata(self) -> List[Dict[str, Any]]:
        """
        Get products with additional metadata for analysis.

        Returns:
            List of products with computed metadata fields.
        """
        products = self.load_products()

        for product in products:
            description = product['description']
            product['word_count'] = len(description.split())
            product['char_count'] = len(description)
            product['sentence_count'] = len([s for s in description.split('.') if s.strip()])

        return products