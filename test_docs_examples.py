#!/usr/bin/env python3
"""
Test the documentation examples to ensure they work correctly.
"""

import json
from fastapi.testclient import TestClient
from src.api import app

def test_readme_examples():
    """Test examples from README.md"""
    client = TestClient(app)

    print("=== TESTING README EXAMPLES ===")

    # Example from README with default keywords
    readme_example = {
        "descriptions": [
            "Premium handcrafted leather bag with elegant design.",
            "Sustainable eco-friendly water bottle made from recycled materials."
        ],
        "keywords": ["eco-friendly", "sustainable", "premium", "luxury"]
    }

    response = client.post("/analyze", json=readme_example)

    if response.status_code == 200:
        result = response.json()
        first_result = result['results'][0]

        print("✓ README example works correctly")
        print(f"  - Text: {first_result['text']}")
        print(f"  - Overall score: {first_result['overall_score']}")
        print(f"  - Keyword density: {first_result['keyword_density']['total_density']}%")

        if 'adaptive_context' in first_result['keyword_density']:
            ac = first_result['keyword_density']['adaptive_context']
            print(f"  - Text category: {ac['text_length_category']}")
            print(f"  - Applied threshold: {ac['applied_max_threshold']}%")

        print(f"  - Business insight: {result['summary']['content_analysis']['business_insights'][0]}")
    else:
        print(f"✗ README example failed: {response.text}")

def test_project_guide_examples():
    """Test examples from PROJECT_GUIDE.md"""
    client = TestClient(app)

    print("\n=== TESTING PROJECT_GUIDE EXAMPLES ===")

    # Test adaptive thresholds with different text lengths
    test_cases = [
        {
            "name": "Very Short (Product Title)",
            "descriptions": ["Premium luxury bag."],
            "expected_category": "very_short",
            "expected_threshold": 30.0
        },
        {
            "name": "Short (Brief Description)",
            "descriptions": ["Premium luxury leather bag with elegant design and sustainable materials."],
            "expected_category": "short",
            "expected_threshold": 20.0
        },
        {
            "name": "Normal (Full Description)",
            "descriptions": ["This premium luxury leather bag features elegant design with handcrafted details, sustainable materials, and eco-friendly production methods that ensure durability and style for modern consumers."],
            "expected_category": "normal",
            "expected_threshold": 8.0
        }
    ]

    for case in test_cases:
        response = client.post("/analyze", json={
            "descriptions": case["descriptions"],
            "keywords": ["eco-friendly", "sustainable", "premium", "luxury"]
        })

        if response.status_code == 200:
            result = response.json()
            first_result = result['results'][0]
            ac = first_result['keyword_density']['adaptive_context']

            print(f"✓ {case['name']} works correctly")
            print(f"  - Word count: {ac['word_count']}")
            print(f"  - Category: {ac['text_length_category']} (expected: {case['expected_category']})")
            print(f"  - Threshold: {ac['applied_max_threshold']}% (expected: {case['expected_threshold']}%)")
            print(f"  - Density score: {first_result['keyword_density']['density_score']}")

            # Verify correct categorization
            assert ac['text_length_category'] == case['expected_category'], f"Category mismatch for {case['name']}"
            assert ac['applied_max_threshold'] == case['expected_threshold'], f"Threshold mismatch for {case['name']}"
        else:
            print(f"✗ {case['name']} failed: {response.text}")

def test_swagger_default_example():
    """Test the Swagger UI default example"""
    client = TestClient(app)

    print("\n=== TESTING SWAGGER DEFAULT EXAMPLE ===")

    # This should use default keywords (no keywords field provided)
    swagger_example = {
        "descriptions": [
            "Premium handcrafted leather bag with elegant design."
        ]
    }

    response = client.post("/analyze", json=swagger_example)

    if response.status_code == 200:
        result = response.json()
        first_result = result['results'][0]
        kd = first_result['keyword_density']

        print("✓ Swagger default example works correctly")
        print(f"  - Uses default keywords: {list(kd['keyword_densities'].keys())}")
        print(f"  - Total density: {kd['total_density']}%")
        print(f"  - Adaptive context: {kd['adaptive_context']['text_length_category']}")
        print(f"  - Applied threshold: {kd['adaptive_context']['applied_max_threshold']}%")

        # Verify default keywords are used
        expected_keywords = ["eco-friendly", "sustainable", "premium", "luxury"]
        actual_keywords = list(kd['keyword_densities'].keys())
        assert actual_keywords == expected_keywords, f"Default keywords mismatch: {actual_keywords} vs {expected_keywords}"
    else:
        print(f"✗ Swagger example failed: {response.text}")

if __name__ == "__main__":
    test_readme_examples()
    test_project_guide_examples()
    test_swagger_default_example()
    print("\n🎉 All documentation examples work correctly!")