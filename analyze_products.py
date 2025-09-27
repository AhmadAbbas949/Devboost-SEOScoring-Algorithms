#!/usr/bin/env python3
"""
Standalone script to analyze all product descriptions from products.json

This script loads all products from the data/products.json file and performs
the 3 core analyses:
1. Readability Score (average word length, average sentence length)
2. Keyword Density for ["eco-friendly", "sustainable", "premium", "luxury"]
3. Cosine Similarity comparison to detect duplicates across entire catalog

Usage:
    python analyze_products.py

Output:
    - Prints summary statistics to console
    - Saves detailed results to analysis_results.json
    - Saves duplicate pairs to duplicates_found.json
"""

import json
import sys
from pathlib import Path
import numpy as np

# Add src directory to path so we can import our modules
sys.path.append(str(Path(__file__).parent / "src"))

from src.ingest import ProductDataIngestion
from src.analysis import TextAnalysisEngine
from src.utils import calculate_text_similarity_matrix


def main():
    """Main analysis function."""
    print("Product Description Analysis Tool")
    print("=" * 50)

    try:
        # Initialize components
        ingestion = ProductDataIngestion()
        analyzer = TextAnalysisEngine()

        # Set target keywords
        target_keywords = ["eco-friendly", "sustainable", "premium", "luxury"]
        analyzer.target_keywords = target_keywords

        print(f"Loading products from {ingestion.data_path}...")
        products = ingestion.load_products()

        if not products:
            print("No products found in data file!")
            return

        print(f"Loaded {len(products)} products")
        print("\nCalculating similarity matrix...")

        # Extract all descriptions
        all_descriptions = [p["description"] for p in products]

        # Calculate similarity matrix for ALL descriptions
        similarity_matrix = calculate_text_similarity_matrix(all_descriptions)
        print(f"Similarity matrix calculated: {similarity_matrix.shape}")

        print("\nAnalyzing each product...")

        # Process each product description
        analysis_results = []
        duplicate_pairs = []

        for i, (product, description) in enumerate(zip(products, all_descriptions)):
            if (i + 1) % 10 == 0:
                print(f"   Processing product {i + 1}/{len(products)}...")

            # 1. Readability Score
            readability_data = analyzer.calculate_readability_score(description)

            # 2. Keyword Density
            keyword_data = analyzer.calculate_keyword_density_score(description)
            keyword_densities = keyword_data["keyword_densities"]

            # 3. Cosine Similarity & Duplicate Detection
            if len(all_descriptions) > 1:
                similarities = similarity_matrix[i]
                other_similarities = np.concatenate([similarities[:i], similarities[i+1:]])
                max_similarity = float(np.max(other_similarities)) if len(other_similarities) > 0 else 0.0
            else:
                max_similarity = 0.0

            uniqueness_score = 1.0 - max_similarity
            is_duplicate = max_similarity >= 0.79

            # Track duplicate pairs (avoid duplicate entries)
            if is_duplicate:
                # Find which product it's most similar to for duplicate pair tracking
                max_idx = np.argmax(similarities)
                if max_idx == i:  # If it's pointing to itself, find the second highest
                    similarities_copy = similarities.copy()
                    similarities_copy[i] = -1  # Mark self as very low
                    max_idx = np.argmax(similarities_copy)
                    max_similarity = similarities[max_idx]

                if max_similarity > 0.1:  # Only if meaningfully similar
                    most_similar_product_id = products[max_idx]["id"]

                    # Check if this pair already exists (in reverse)
                    pair_exists = any(
                        (dp["product_1"]["id"] == most_similar_product_id and dp["product_2"]["id"] == product["id"])
                        for dp in duplicate_pairs
                    )

                    if not pair_exists:
                        duplicate_pairs.append({
                            "product_1": {"id": product["id"], "title": product["title"]},
                            "product_2": {
                                "id": most_similar_product_id,
                                "title": next(p["title"] for p in products if p["id"] == most_similar_product_id)
                            },
                            "similarity": round(max_similarity, 3)
                        })

            analysis_results.append({
                "product_id": product["id"],
                "title": product["title"],
                "description": description,
                "readability": {
                    "avg_word_length": readability_data["avg_word_length"],
                    "avg_sentence_length": readability_data["avg_sentence_length"]
                },
                "keyword_density": {
                    "eco_friendly": keyword_densities.get("eco-friendly", 0.0),
                    "sustainable": keyword_densities.get("sustainable", 0.0),
                    "premium": keyword_densities.get("premium", 0.0),
                    "luxury": keyword_densities.get("luxury", 0.0)
                },
                "similarity": {
                    "uniqueness_score": round(uniqueness_score, 3),
                    "is_duplicate": bool(is_duplicate),
                    "max_similarity": round(max_similarity, 3)
                }
            })

        # Calculate summary statistics
        total_products = len(analysis_results)
        duplicate_count = sum(1 for r in analysis_results if r["similarity"]["is_duplicate"])

        # Keyword usage statistics
        keyword_stats = {
            "eco_friendly": sum(1 for r in analysis_results if r["keyword_density"]["eco_friendly"] > 0),
            "sustainable": sum(1 for r in analysis_results if r["keyword_density"]["sustainable"] > 0),
            "premium": sum(1 for r in analysis_results if r["keyword_density"]["premium"] > 0),
            "luxury": sum(1 for r in analysis_results if r["keyword_density"]["luxury"] > 0)
        }

        # Readability statistics
        avg_word_lengths = [r["readability"]["avg_word_length"] for r in analysis_results]
        avg_sentence_lengths = [r["readability"]["avg_sentence_length"] for r in analysis_results]

        # Print summary to console
        print("\n" + "=" * 50)
        print("ANALYSIS SUMMARY")
        print("=" * 50)
        print(f"Total Products Analyzed: {total_products}")
        print(f"Duplicates Found: {duplicate_count}")
        print(f"Duplicate Percentage: {round((duplicate_count / total_products) * 100, 1)}%")
        print(f"Unique Products: {total_products - duplicate_count}")

        print(f"\nKEYWORD USAGE:")
        for keyword, count in keyword_stats.items():
            percentage = round((count / total_products) * 100, 1)
            print(f"   {keyword.replace('_', '-').title()}: {count} products ({percentage}%)")

        print(f"\nREADABILITY AVERAGES:")
        print(f"   Average Word Length: {round(np.mean(avg_word_lengths), 2)} characters")
        print(f"   Average Sentence Length: {round(np.mean(avg_sentence_lengths), 2)} words")

        if duplicate_pairs:
            print(f"\nDUPLICATE PAIRS FOUND ({len(duplicate_pairs)}):")
            for i, pair in enumerate(duplicate_pairs[:5], 1):  # Show first 5
                print(f"   {i}. '{pair['product_1']['title']}' <-> '{pair['product_2']['title']}' ({pair['similarity']} similarity)")
            if len(duplicate_pairs) > 5:
                print(f"   ... and {len(duplicate_pairs) - 5} more pairs")

        # Save detailed results to JSON files
        print(f"\nSaving results...")

        # Save full analysis results
        with open("analysis_results.json", "w", encoding="utf-8") as f:
            json.dump({
                "summary": {
                    "total_products_analyzed": total_products,
                    "duplicates_found": duplicate_count,
                    "duplicate_percentage": round((duplicate_count / total_products) * 100, 1),
                    "unique_products": total_products - duplicate_count,
                    "keyword_usage": keyword_stats,
                    "readability_averages": {
                        "avg_word_length": round(np.mean(avg_word_lengths), 2),
                        "avg_sentence_length": round(np.mean(avg_sentence_lengths), 2)
                    }
                },
                "results": analysis_results,
                "analysis_type": "3 Core Metrics: Readability + Keyword Density + Cosine Similarity"
            }, f, indent=2, ensure_ascii=False)

        # Save duplicate pairs separately
        with open("duplicates_found.json", "w", encoding="utf-8") as f:
            json.dump({
                "total_duplicate_pairs": len(duplicate_pairs),
                "duplicate_pairs": duplicate_pairs
            }, f, indent=2, ensure_ascii=False)

        print(f"Results saved to:")
        print(f"   analysis_results.json - Complete analysis results")
        print(f"   duplicates_found.json - Duplicate pairs found")

        print(f"\nAnalysis complete!")

    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()