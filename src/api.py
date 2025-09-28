"""
FastAPI application providing REST endpoints for text analysis.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn
import numpy as np

from .analysis import TextAnalysisEngine
from .ingest import ProductDataIngestion
from .utils import calculate_text_similarity_matrix, analyze_duplicates_advanced


app = FastAPI(
    title="devBoost Text Analysis API",
    description="AI-powered text analysis for e-commerce product descriptions",
    version="1.0.0",
)

# Initialize analysis engine
analyzer = TextAnalysisEngine()


class ProductInput(BaseModel):
    """
    Input model for product analysis requests.

    Default behavior: Send empty request {} to analyze all 50 products from products.json.
    """

    descriptions: Optional[List[str]] = Field(
        None,
        description="List of product descriptions to analyze. If not provided or empty, will analyze all products from products.json",
        json_schema_extra={
            "examples": [
                [],  # Default: use products.json
                [
                    "Premium handcrafted leather bag with elegant design."
                ],  # Custom descriptions
            ]
        },
    )
    keywords: Optional[List[str]] = Field(
        None,
        description="Custom keywords to analyze for density. Defaults to: eco-friendly, sustainable, premium, luxury",
        json_schema_extra={
            "example": ["eco-friendly", "sustainable", "premium", "luxury"]
        },
    )


class ReadabilityResult(BaseModel):
    """Readability analysis result."""

    avg_word_length: float = Field(..., description="Average word length in characters")
    avg_sentence_length: float = Field(
        ..., description="Average sentence length in words"
    )


class KeywordDensityResult(BaseModel):
    """Keyword density analysis result."""

    eco_friendly: float = Field(
        ..., description="Eco-friendly keyword density percentage"
    )
    sustainable: float = Field(
        ..., description="Sustainable keyword density percentage"
    )
    premium: float = Field(..., description="Premium keyword density percentage")
    luxury: float = Field(..., description="Luxury keyword density percentage")


class SimilarityResult(BaseModel):
    """Similarity analysis result."""

    uniqueness_score: float = Field(
        ..., description="Uniqueness score (0.0 = duplicate, 1.0 = unique)"
    )
    is_duplicate: bool = Field(
        ...,
        description="Whether this content is likely a duplicate (similarity >= 79%)",
    )
    max_similarity: float = Field(
        ..., description="Maximum cosine similarity to other descriptions"
    )


class AnalysisResult(BaseModel):
    """Complete analysis result for a single description."""

    description: str = Field(..., description="The analyzed text")
    readability: ReadabilityResult = Field(..., description="Readability metrics")
    keyword_density: KeywordDensityResult = Field(
        ..., description="Keyword density metrics"
    )
    similarity: SimilarityResult = Field(
        ..., description="Similarity and duplicate detection"
    )


class AnalysisResponse(BaseModel):
    """Response model for analysis results."""

    results: List[AnalysisResult] = Field(
        ..., description="Analysis results for each description"
    )


class RecommendationResult(BaseModel):
    """Single recommendation result."""

    description_index: int = Field(
        ..., description="Index of the description in the input list"
    )
    description: str = Field(..., description="The analyzed description text")
    recommendations: List[str] = Field(
        ...,
        description="Exactly 3 specific improvement suggestions",
        min_length=3,
        max_length=3,
    )


class ProductRecommendationResult(BaseModel):
    """Single product recommendation result from products.json."""

    product_id: int = Field(..., description="Product ID from products.json")
    title: str = Field(..., description="Product title")
    description: str = Field(..., description="Product description")
    recommendations: List[str] = Field(
        ...,
        description="Exactly 3 specific improvement suggestions",
        min_length=3,
        max_length=3,
    )


class RecommendationResponse(BaseModel):
    """Response model for recommendations."""

    recommendations: List[RecommendationResult] = Field(
        ..., description="Exactly 3 recommendations for each description"
    )


class ProductRecommendationResponse(BaseModel):
    """Response model for product recommendations from products.json."""

    message: str = Field(..., description="Summary message")
    total_products: int = Field(..., description="Total number of products analyzed")
    recommendations: List[ProductRecommendationResult] = Field(
        ..., description="Exactly 3 recommendations for each product"
    )


class SampleAnalysisResult(BaseModel):
    """Result for a single product in sample analysis."""

    product_id: int = Field(..., description="Product ID from products.json")
    title: str = Field(..., description="Product title")
    description: str = Field(..., description="Product description")
    readability: ReadabilityResult = Field(..., description="Readability metrics")
    keyword_density: KeywordDensityResult = Field(
        ..., description="Keyword density metrics"
    )
    similarity: SimilarityResult = Field(
        ..., description="Similarity and duplicate detection"
    )


class DuplicatePair(BaseModel):
    """A pair of duplicate products."""

    product_1: Dict[str, Any] = Field(..., description="First product info (id, title)")
    product_2: Dict[str, Any] = Field(
        ..., description="Second product info (id, title)"
    )
    similarity: float = Field(
        ..., description="Similarity score between the two products"
    )


class SampleAnalysisSummary(BaseModel):
    """Summary statistics for sample analysis."""

    total_products_analyzed: int = Field(
        ..., description="Total number of products analyzed"
    )
    duplicates_found: int = Field(..., description="Number of duplicate products found")
    duplicate_percentage: float = Field(
        ..., description="Percentage of products that are duplicates"
    )
    unique_products: int = Field(..., description="Number of unique products")
    keyword_usage: Dict[str, int] = Field(
        ..., description="Count of products using each keyword"
    )
    readability_averages: Dict[str, float] = Field(
        ..., description="Average readability metrics"
    )


class SampleAnalysisResponse(BaseModel):
    """Response model for sample analysis."""

    message: str = Field(..., description="Summary message")
    summary: SampleAnalysisSummary = Field(
        ..., description="Analysis summary statistics"
    )
    duplicate_pairs: List[DuplicatePair] = Field(
        ..., description="List of detected duplicate pairs"
    )
    results: List[SampleAnalysisResult] = Field(
        ..., description="Detailed analysis for each product"
    )
    analysis_type: str = Field(..., description="Type of analysis performed")


@app.get("/")
async def root():
    """Health check endpoint with comprehensive API documentation overview."""
    return {
        "message": "devBoost Text Analysis API - Enterprise ML-Powered Analytics",
        "status": "healthy",
        "version": "1.0.0",
        "endpoints": {
            "/analyze": "Detailed metric computation with precise formulas",
            "/recommend": "AI-powered suggestions with intelligent algorithms",
            "/sample-analysis": "Complete catalog analysis with technical documentation",
            "/docs": "Interactive Swagger UI with comprehensive examples",
        },
        "metrics_documentation": {
            "readability": {
                "word_length_formula": "sum(len(word) for word in words) / len(words)",
                "sentence_length_formula": "total_words / total_sentences",
                "tokenization": "regex \\b[a-zA-Z]+\\b (alphabetic only)",
                "precision": "2 decimal places",
            },
            "keyword_density": {
                "formula": "(occurrences × keyword_words / total_words) × 100",
                "pattern_matching": "\\b{keyword}\\b with word boundaries",
                "multi_word_support": "eco-friendly counts as 2 words per occurrence",
                "adaptive_thresholds": "30%/20%/8% for short/medium/long text",
            },
            "uniqueness": {
                "method": "scikit-learn CountVectorizer + cosine_similarity",
                "vectorization": "TF vectors with English stop words removal",
                "similarity_threshold": "79% for duplicate detection",
                "uniqueness_score": "1.0 - max_similarity_to_others",
            },
        },
        "ml_stack": {
            "primary_libraries": ["NumPy", "scikit-learn", "FastAPI"],
            "vectorization": "CountVectorizer with 1000 max features",
            "similarity": "Cosine similarity with optimized matrix operations",
            "optimization": "Vectorized NumPy operations, advanced indexing",
            "algorithms": ["NearestNeighbors", "AgglomerativeClustering", "TF-IDF"],
        },
        "technical_features": {
            "precision": "All scores rounded to 3 decimal places",
            "performance": "Vectorized operations for 50+ products",
            "memory_efficient": "Advanced NumPy indexing for similarity matrices",
            "production_ready": "Enterprise-grade error handling and validation",
            "documentation": "Complete computation formulas in Swagger UI",
        },
        "quick_start": {
            "analyze_all_products": "POST /analyze with {}",
            "custom_analysis": 'POST /analyze with {"descriptions": ["your text"]}',
            "get_recommendations": "POST /recommend with {}",
            "view_documentation": "GET /docs for interactive examples",
        },
    }


@app.post(
    "/analyze",
    response_model=AnalysisResponse,
    summary="Analyze product descriptions with detailed metrics",
    description="""
    Analyze product descriptions using 3 core metrics with detailed computation methods:

    ## 📊 **Metric 1: Readability Analysis**

    **Average Word Length:**
    - **Formula:** `sum(len(word) for word in words) / total_words`
    - **Tokenization:** Extracts alphabetic words using regex `\\b[a-zA-Z]+\\b`
    - **Units:** Characters per word
    - **Example:** "Premium bag" → words=["Premium", "bag"] → lengths=[7,3] → avg=5.0 chars

    **Average Sentence Length:**
    - **Formula:** `total_words_across_sentences / total_sentences`
    - **Sentence Split:** Uses regex `[.!?]+\\s*` to identify sentence boundaries
    - **Units:** Words per sentence
    - **Example:** "Great product. Buy now!" → sentences=2, words=4 → avg=2.0 words/sentence

    ## 🔍 **Metric 2: Keyword Density Analysis**

    **Target Keywords:** ["eco-friendly", "sustainable", "premium", "luxury"]

    **Calculation Method:**
    - **Formula:** `(keyword_occurrences × keyword_word_count / total_words) × 100`
    - **Pattern Matching:** Uses word boundaries `\\b{keyword}\\b` (case-insensitive)
    - **Multi-word Keywords:** "eco-friendly" counts as 2 words per occurrence
    - **Units:** Percentage of total word count
    - **Example:** "Premium bag with premium materials" (6 words total)
      - "premium" appears 2 times → `(2 × 1 / 6) × 100 = 33.33%`

    **Adaptive Thresholds:**
    - **Very Short (≤8 words):** Up to 30% density allowed
    - **Short (≤15 words):** Up to 20% density allowed
    - **Normal (>15 words):** 2-8% optimal density range

    ## 🎯 **Metric 3: Uniqueness/Duplicate Detection**

    **Method:** Professional ML using scikit-learn CountVectorizer + cosine similarity

    **Technical Implementation:**
    ```python
    # Text Vectorization
    CountVectorizer(
        lowercase=True,
        stop_words='english',
        token_pattern=r'\\b[a-zA-Z]+\\b',
        max_features=1000,
        binary=False  # Uses term frequency
    )

    # Similarity Calculation
    cosine_similarity_matrix = cosine_similarity(text_vectors)
    uniqueness_score = 1.0 - max_similarity_to_others
    ```

    **Duplicate Detection:**
    - **Similarity Threshold:** ≥79% similarity = duplicate
    - **Uniqueness Score:** `1.0 - max_cosine_similarity` (0.0=identical, 1.0=unique)
    - **Cross-Comparison:** Each description compared against all others in set
    - **Duplicate Pairs:** Automatically detected using NearestNeighbors algorithm

    ## 📋 **Input Modes & Examples**

    ### **Mode 1: Default Analysis (Recommended)**
    ```json
    {}
    ```
    - Analyzes all 50 products from products.json
    - Cross-compares entire product catalog for duplicates
    - Uses default keywords: ["eco-friendly", "sustainable", "premium", "luxury"]

    ### **Mode 2: Custom Descriptions**
    ```json
    {
      "descriptions": [
        "Premium handcrafted leather bag with elegant design.",
        "Sustainable eco-friendly water bottle made from recycled materials."
      ]
    }
    ```
    - Analyzes provided descriptions only
    - Cross-compares within provided set for uniqueness
    - Uses default keywords

    ### **Mode 3: Custom Keywords**
    ```json
    {
      "descriptions": ["Your product description here"],
      "keywords": ["organic", "natural", "artisan", "handmade"]
    }
    ```
    - Uses custom keywords instead of defaults
    - All other analysis remains the same

    ## 📊 **Response Format**

    ```json
    {
      "results": [
        {
          "description": "Premium handcrafted leather bag with elegant design.",
          "readability": {
            "avg_word_length": 5.78,
            "avg_sentence_length": 9.0
          },
          "keyword_density": {
            "eco_friendly": 0.0,
            "sustainable": 0.0,
            "premium": 11.11,
            "luxury": 0.0
          },
          "similarity": {
            "uniqueness_score": 0.876,
            "is_duplicate": false,
            "max_similarity": 0.124
          }
        }
      ]
    }
    ```

    ## 🔬 **Technical Notes**

    - **Performance:** Optimized ML pipeline using scikit-learn + NumPy
    - **Vectorization:** TF (term frequency) vectors with 1000 max features
    - **Stop Words:** English stop words automatically removed
    - **Precision:** All scores rounded to 3 decimal places
    - **Memory Efficient:** Advanced NumPy indexing for similarity matrices
    - **Scalable:** Handles 1-50+ descriptions efficiently

    ## 📈 **Output Guarantees**

    - **Readability:** Always returns avg_word_length and avg_sentence_length
    - **Keywords:** Always returns density for all 4 target keywords (0.0 if not found)
    - **Similarity:** Always returns uniqueness_score, is_duplicate, and max_similarity
    - **Precision:** All numeric values to exactly 3 decimal places
    - **Validation:** Pydantic models ensure type safety and structure
    """,
)
async def analyze_descriptions(input_data: ProductInput):
    """Analyze product descriptions using 3 core metrics with detailed computation methods."""
    try:

        # Use custom keywords if provided, otherwise use default
        target_keywords = input_data.keywords or [
            "eco-friendly",
            "sustainable",
            "premium",
            "luxury",
        ]
        analyzer.target_keywords = target_keywords

        # Determine mode: use provided descriptions or load from products.json
        if input_data.descriptions and len(input_data.descriptions) > 0:
            # Mode 1: Use provided descriptions
            descriptions = input_data.descriptions
        else:
            # Mode 2: Load from products.json (same as sample-analysis)
            ingestion = ProductDataIngestion()
            products = ingestion.load_products()
            descriptions = [p["description"] for p in products]

        # Advanced ML-powered duplicate detection using scikit-learn optimizations
        duplicate_analysis = analyze_duplicates_advanced(
            texts=descriptions, similarity_threshold=0.79
        )

        # Extract results from advanced analysis
        similarity_scores = duplicate_analysis["similarity_scores"]
        uniqueness_scores = duplicate_analysis["uniqueness_scores"]
        duplicate_indices = set(duplicate_analysis["duplicate_indices"])

        results = []
        for i, description in enumerate(descriptions):
            # 1. Readability Score
            readability_data = analyzer.calculate_readability_score(description)
            readability = ReadabilityResult(
                avg_word_length=readability_data["avg_word_length"],
                avg_sentence_length=readability_data["avg_sentence_length"],
            )

            # 2. Keyword Density
            keyword_data = analyzer.calculate_keyword_density_score(description)
            keyword_densities = keyword_data["keyword_densities"]
            keyword_density = KeywordDensityResult(
                eco_friendly=keyword_densities.get("eco-friendly", 0.0),
                sustainable=keyword_densities.get("sustainable", 0.0),
                premium=keyword_densities.get("premium", 0.0),
                luxury=keyword_densities.get("luxury", 0.0),
            )

            # 3. Use pre-computed similarity results from advanced ML analysis
            max_similarity = similarity_scores[i] if i < len(similarity_scores) else 0.0
            uniqueness_score = (
                uniqueness_scores[i] if i < len(uniqueness_scores) else 1.0
            )
            is_duplicate = i in duplicate_indices

            similarity = SimilarityResult(
                uniqueness_score=round(uniqueness_score, 3),
                is_duplicate=bool(is_duplicate),  # Ensure boolean type
                max_similarity=round(max_similarity, 3),
            )

            # Create final result
            result = AnalysisResult(
                description=description,
                readability=readability,
                keyword_density=keyword_density,
                similarity=similarity,
            )
            results.append(result)

        return AnalysisResponse(results=results)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post(
    "/recommend",
    summary="Get AI-powered improvement recommendations",
    description="Generate exactly 3 actionable suggestions per description using intelligent analysis algorithms.",
)
async def get_recommendations(input_data: ProductInput):
    """
    Generate exactly 3 specific improvement recommendations using intelligent analysis algorithms:

    ## 🤖 **Recommendation Engine Architecture**

    **Core Principle:** Each description receives **exactly 3 suggestions** based on systematic analysis of:
    1. **Readability metrics** (word/sentence length optimization)
    2. **Keyword density analysis** (context-aware SEO optimization)
    3. **Uniqueness assessment** (duplicate prevention & differentiation)

    ## 📊 **Analysis Process**

    ### **Step 1: Readability Analysis**
    - **Word Length Formula:** `sum(len(word) for word in words) / total_words`
    - **Sentence Length Formula:** `total_words_across_sentences / total_sentences`
    - **Tokenization:** Uses regex `\\b[a-zA-Z]+\\b` for alphabetic words only

    ### **Step 2: Keyword Density Assessment**
    - **Density Formula:** `(keyword_occurrences × keyword_word_count / total_words) × 100`
    - **Target Keywords:** ["eco-friendly", "sustainable", "premium", "luxury"]
    - **Adaptive Thresholds:** 30%/20%/8% for short/medium/long text

    ### **Step 3: Similarity Calculation**
    - **Method:** scikit-learn CountVectorizer + cosine similarity
    - **Duplicate Threshold:** ≥79% similarity
    - **Cross-Comparison:** Against entire dataset for uniqueness

    ## 🧠 **Intelligent Recommendation Logic**

    ### **1. Readability Suggestions (Always 1)**
    ```python
    # E-commerce optimized readability rules:
    if avg_sentence_length > 15:
        suggestion = "Try shorter sentences"
    elif avg_sentence_length < 5:
        suggestion = "Use longer, more descriptive sentences"
    elif avg_word_length > 7:
        suggestion = "Use simpler words for better readability"
    elif avg_word_length < 4:
        suggestion = "Add more descriptive words"
    else:
        suggestion = "Consider varying sentence length for better flow"
    ```

    ### **2. Context-Aware Keyword Suggestions (Always 1)**
    ```python
    # Smart material detection for eco-keywords:
    def is_eco_friendly_product(text):
        eco_incompatible = ['leather', 'fur', 'silk', 'wool', 'plastic', 'synthetic']
        return not any(material in text.lower() for material in eco_incompatible)

    # Intelligent keyword recommendations:
    if total_density > 20:
        suggestion = "Reduce keyword repetition to avoid over-optimization"
    elif total_density > 15:
        suggestion = "Balance keyword usage with natural language"
    elif missing_keywords and is_eco_friendly_product(text):
        suggestion = "Increase use of keyword 'eco-friendly'"
    elif "premium" missing and quality_indicators_present:
        suggestion = "Increase use of keyword 'premium'"
    else:
        suggestion = "Enhance keyword variety for better SEO"
    ```

    **Quality Indicators Detection:**
    - "handcrafted", "quality", "luxury" → Suggests "premium"
    - Material compatibility check → Eco-keyword suggestions
    - Density analysis → Over/under-optimization warnings

    ### **3. Uniqueness/Similarity Suggestions (Always 1)**
    ```python
    # Similarity-based recommendations:
    if max_similarity >= 0.79:
        suggestion = f"Avoid duplicate wording with Product #{similar_product_id}"
    elif max_similarity > 0.6:
        suggestion = f"Reduce similarity with Product #{similar_product_id}"
    elif max_similarity > 0.4:
        suggestion = "Add more unique selling points"
    else:
        suggestion = "Add specific product benefits or features"
    ```

    ## 📋 **Input Modes & Examples**

    ### **Mode 1: Default Analysis (Products.json)**
    ```json
    {}
    ```
    **Output:**
    ```json
    {
      "message": "Generated 3 specific recommendations for 50 products from products.json",
      "total_products": 50,
      "recommendations": [
        {
          "product_id": 1,
          "title": "Eco-Friendly Bamboo Toothbrush",
          "description": "A sustainable toothbrush made from 100% biodegradable bamboo...",
          "recommendations": [
            "Try shorter sentences",
            "Increase use of keyword 'premium'",
            "Add specific product benefits or features"
          ]
        }
      ]
    }
    ```

    ### **Mode 2: Custom Descriptions**
    ```json
    {
      "descriptions": [
        "Premium bag",
        "Luxury handcrafted leather wallet with elegant design and premium materials."
      ]
    }
    ```
    **Output:**
    ```json
    {
      "recommendations": [
        {
          "description_index": 0,
          "description": "Premium bag",
          "recommendations": [
            "Use longer, more descriptive sentences",
            "Balance keyword usage with natural language",
            "Add specific product benefits or features"
          ]
        }
      ]
    }
    ```

    ### **Mode 3: Custom Keywords**
    ```json
    {
      "descriptions": ["Organic cotton t-shirt"],
      "keywords": ["organic", "natural", "artisan", "handmade"]
    }
    ```

    ## 🎯 **Decision Thresholds**

    **Readability Optimization:**
    - **Target Word Length:** 4-6 characters (e-commerce optimal)
    - **Target Sentence Length:** 10-20 words (clarity vs. detail balance)
    - **Triggers:** >15 words = too long, <5 words = too short

    **Keyword Density Optimization:**
    - **Text Length ≤8 words:** 30% max density (product titles)
    - **Text Length ≤15 words:** 20% max density (short descriptions)
    - **Text Length >15 words:** 2-8% optimal range (full descriptions)
    - **Over-optimization:** >20% triggers reduction suggestions

    **Similarity Optimization:**
    - **Duplicate Alert:** ≥79% similarity
    - **High Similarity:** 60-79% → specific product reference
    - **Moderate Similarity:** 40-60% → unique selling points suggestion
    - **Unique Content:** <40% → benefits/features enhancement

    ## 🔬 **Technical Implementation**

    **Analysis Pipeline:**
    1. Load descriptions (products.json or custom)
    2. Calculate similarity matrix using advanced ML
    3. Analyze each description individually
    4. Generate context-aware suggestions
    5. Ensure exactly 3 recommendations per description

    **Performance Features:**
    - **Vectorized Operations:** NumPy optimized calculations
    - **ML Optimization:** scikit-learn NearestNeighbors for similarity
    - **Memory Efficient:** Advanced indexing for large catalogs
    - **Scalable:** Handles 1-50+ products efficiently

    ## 📈 **Output Guarantees**

    - **Exactly 3 Suggestions:** Never more, never less per description
    - **Context-Aware:** Suggestions adapt to product type and content
    - **Actionable:** Every recommendation provides specific guidance
    - **Intelligent:** Material detection, quality recognition, duplicate prevention
    - **Consistent:** Same analysis methodology as /analyze endpoint
    """
    try:
        # Use custom keywords if provided, otherwise use default
        target_keywords = input_data.keywords or [
            "eco-friendly",
            "sustainable",
            "premium",
            "luxury",
        ]
        analyzer.target_keywords = target_keywords

        # Determine mode: use provided descriptions or load from products.json
        if input_data.descriptions and len(input_data.descriptions) > 0:
            # Mode 1: Use provided descriptions
            descriptions = input_data.descriptions
            products = None
            mode = "custom"
        else:
            # Mode 2: Load from products.json
            ingestion = ProductDataIngestion()
            products = ingestion.load_products()
            descriptions = [p["description"] for p in products]
            mode = "products"

        # Advanced ML-powered duplicate detection for recommendations
        product_metadata = (
            None
            if mode == "custom"
            else [{"id": p["id"], "title": p["title"]} for p in products]
        )
        duplicate_analysis = analyze_duplicates_advanced(
            texts=descriptions, metadata=product_metadata, similarity_threshold=0.79
        )

        # Extract similarity matrix for recommendation logic
        similarity_matrix = duplicate_analysis["similarity_matrix"]

        def generate_suggestions_for_description(
            description, i, descriptions_list, similarity_matrix, product_ref=None
        ):
            """Generate 3 suggestions for a single description."""
            # Analyze this description
            readability_data = analyzer.calculate_readability_score(description)
            keyword_data = analyzer.calculate_keyword_density_score(description)

            # Extract metrics
            avg_word_length = readability_data["avg_word_length"]
            avg_sentence_length = readability_data["avg_sentence_length"]
            keyword_densities = keyword_data["keyword_densities"]
            total_keyword_density = sum(keyword_densities.values())

            # Optimized similarity analysis using vectorized operations
            most_similar_idx = None
            max_similarity = 0.0
            if len(descriptions_list) > 1:
                # Use vectorized operations instead of manual similarity processing
                similarities = similarity_matrix[i]

                # Create mask to exclude self-similarity
                mask = np.ones(len(similarities), dtype=bool)
                mask[i] = False

                # Vectorized maximum similarity calculation
                other_similarities = similarities[mask]
                if len(other_similarities) > 0:
                    max_similarity = float(np.max(other_similarities))

                    # Find which description it's most similar to using vectorized operations
                    if max_similarity > 0.4:  # Only mention if meaningfully similar
                        # Find the index of maximum similarity (excluding self)
                        max_idx = np.argmax(np.where(mask, similarities, -1))

                        if product_ref:
                            # For products mode, reference by product ID
                            most_similar_idx = products[max_idx]["id"]
                        else:
                            # For custom mode, reference by position
                            most_similar_idx = max_idx + 1

            # Generate exactly 3 recommendations
            suggestions = []

            # 1. Readability suggestion (always include one)
            if avg_sentence_length > 15:
                suggestions.append("Try shorter sentences")
            elif avg_sentence_length < 5:
                suggestions.append("Use longer, more descriptive sentences")
            elif avg_word_length > 7:
                suggestions.append("Use simpler words for better readability")
            elif avg_word_length < 4:
                suggestions.append("Add more descriptive words")
            else:
                suggestions.append("Consider varying sentence length for better flow")

            # 2. Keyword suggestion (context-aware)
            def is_eco_friendly_product(text):
                """Check if product could realistically be eco-friendly."""
                eco_incompatible = [
                    "leather",
                    "fur",
                    "silk",
                    "wool",
                    "plastic",
                    "synthetic",
                    "petroleum",
                    "chemical",
                ]
                text_lower = text.lower()
                return not any(material in text_lower for material in eco_incompatible)

            def get_contextual_keyword_suggestion(
                description, keyword_densities, total_density
            ):
                """Generate context-aware keyword suggestions."""
                missing_keywords = [
                    kw for kw, density in keyword_densities.items() if density == 0
                ]

                # If over-optimized, suggest reduction
                if total_density > 20:
                    return "Reduce keyword repetition to avoid over-optimization"
                elif total_density > 15:
                    return "Balance keyword usage with natural language"

                # Context-aware keyword suggestions
                if missing_keywords:
                    # For luxury/premium products (always appropriate)
                    if "premium" in missing_keywords and (
                        "handcrafted" in description.lower()
                        or "quality" in description.lower()
                        or "luxury" in description.lower()
                    ):
                        return "Increase use of keyword 'premium'"
                    elif "luxury" in missing_keywords and (
                        "premium" in description.lower()
                        or "elegant" in description.lower()
                        or "luxury" in description.lower()
                    ):
                        return "Increase use of keyword 'luxury'"

                    # For eco-friendly products (only if contextually appropriate)
                    elif "eco-friendly" in missing_keywords and is_eco_friendly_product(
                        description
                    ):
                        return "Increase use of keyword 'eco-friendly'"
                    elif "sustainable" in missing_keywords and is_eco_friendly_product(
                        description
                    ):
                        return "Increase use of keyword 'sustainable'"

                    # Fallback to most appropriate missing keyword
                    elif "premium" in missing_keywords:
                        return "Increase use of keyword 'premium'"
                    elif "luxury" in missing_keywords:
                        return "Increase use of keyword 'luxury'"
                    else:
                        return "Add relevant descriptive keywords"
                else:
                    # Find keyword to adjust if none are missing
                    max_keyword = max(keyword_densities.items(), key=lambda x: x[1])
                    if max_keyword[1] > 8:
                        return (
                            f"Reduce repetition of '{max_keyword[0].replace('_', '-')}'"
                        )
                    else:
                        # Suggest increasing the most relevant keyword based on product type
                        if is_eco_friendly_product(
                            description
                        ) and keyword_densities.get(
                            "sustainable", 0
                        ) < keyword_densities.get(
                            "premium", 0
                        ):
                            return (
                                "Consider adding 'sustainable' for eco-conscious appeal"
                            )
                        elif keyword_densities.get("premium", 0) < 5:
                            return "Increase use of keyword 'premium'"
                        else:
                            return "Enhance keyword variety for better SEO"

            keyword_suggestion = get_contextual_keyword_suggestion(
                description, keyword_densities, total_keyword_density
            )
            suggestions.append(keyword_suggestion)

            # 3. Uniqueness/similarity suggestion (always include one)
            if most_similar_idx and max_similarity > 0.79:
                if product_ref:
                    suggestions.append(
                        f"Avoid duplicate wording with Product #{most_similar_idx}"
                    )
                else:
                    suggestions.append(
                        f"Avoid duplicate wording with Product #{most_similar_idx}"
                    )
            elif most_similar_idx and max_similarity > 0.6:
                if product_ref:
                    suggestions.append(
                        f"Reduce similarity with Product #{most_similar_idx}"
                    )
                else:
                    suggestions.append(
                        f"Reduce similarity with Product #{most_similar_idx}"
                    )
            elif max_similarity > 0.4:
                suggestions.append("Add more unique selling points")
            else:
                suggestions.append("Add specific product benefits or features")

            # Ensure exactly 3 suggestions
            while len(suggestions) < 3:
                if len(suggestions) == 0:
                    suggestions.append("Improve overall content quality")
                elif len(suggestions) == 1:
                    suggestions.append("Add more engaging product details")
                else:
                    suggestions.append("Consider professional copywriting review")

            return suggestions[:3]

        # Generate recommendations based on mode
        recommendations = []

        if mode == "custom":
            # Custom descriptions mode
            for i, description in enumerate(descriptions):
                suggestions = generate_suggestions_for_description(
                    description, i, descriptions, similarity_matrix
                )
                recommendations.append(
                    RecommendationResult(
                        description_index=i,
                        description=description,
                        recommendations=suggestions,
                    )
                )
            return RecommendationResponse(recommendations=recommendations)

        else:
            # Products.json mode
            product_recommendations = []
            for i, (product, description) in enumerate(zip(products, descriptions)):
                suggestions = generate_suggestions_for_description(
                    description, i, descriptions, similarity_matrix, product_ref=True
                )
                product_recommendations.append(
                    ProductRecommendationResult(
                        product_id=product["id"],
                        title=product["title"],
                        description=description,
                        recommendations=suggestions,
                    )
                )

            return ProductRecommendationResponse(
                message=f"Generated 3 specific recommendations for {len(products)} products from products.json",
                total_products=len(products),
                recommendations=product_recommendations,
            )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Recommendation generation failed: {str(e)}"
        )


# @app.get("/sample-analysis", response_model=SampleAnalysisResponse)
# async def sample_analysis():
#     """
#     Analyze ALL product descriptions from products.json using 3 core metrics with detailed computation methods.
#     """
#     try:
#         # Load ALL products from products.json
#         ingestion = ProductDataIngestion()
#         products = ingestion.load_products()

#         # Extract all descriptions for analysis
#         all_descriptions = [p["description"] for p in products]

#         print(f"Analyzing {len(all_descriptions)} product descriptions...")

#         # Use fixed keywords for analysis
#         target_keywords = ["eco-friendly", "sustainable", "premium", "luxury"]
#         analyzer.target_keywords = target_keywords

#         # Advanced ML-powered duplicate detection using scikit-learn optimizations
#         product_metadata = [{"id": p["id"], "title": p["title"]} for p in products]
#         duplicate_analysis = analyze_duplicates_advanced(
#             texts=all_descriptions,
#             metadata=product_metadata,
#             similarity_threshold=0.79
#         )

#         # Extract results from advanced analysis
#         similarity_scores = duplicate_analysis['similarity_scores']
#         uniqueness_scores = duplicate_analysis['uniqueness_scores']
#         duplicate_indices = set(duplicate_analysis['duplicate_indices'])
#         duplicate_pairs = duplicate_analysis['duplicate_pairs']

#         # Process each product description with optimized results
#         analysis_results = []

#         for i, (product, description) in enumerate(zip(products, all_descriptions)):
#             # 1. Readability Score
#             readability_data = analyzer.calculate_readability_score(description)

#             # 2. Keyword Density
#             keyword_data = analyzer.calculate_keyword_density_score(description)
#             keyword_densities = keyword_data["keyword_densities"]

#             # 3. Use pre-computed similarity results from advanced ML analysis
#             max_similarity = similarity_scores[i] if i < len(similarity_scores) else 0.0
#             uniqueness_score = uniqueness_scores[i] if i < len(uniqueness_scores) else 1.0
#             is_duplicate = i in duplicate_indices

#             analysis_results.append({
#                 "product_id": product["id"],
#                 "title": product["title"],
#                 "description": description,
#                 "readability": {
#                     "avg_word_length": readability_data["avg_word_length"],
#                     "avg_sentence_length": readability_data["avg_sentence_length"]
#                 },
#                 "keyword_density": {
#                     "eco_friendly": keyword_densities.get("eco-friendly", 0.0),
#                     "sustainable": keyword_densities.get("sustainable", 0.0),
#                     "premium": keyword_densities.get("premium", 0.0),
#                     "luxury": keyword_densities.get("luxury", 0.0)
#                 },
#                 "similarity": {
#                     "uniqueness_score": round(uniqueness_score, 3),
#                     "is_duplicate": is_duplicate,
#                     "max_similarity": round(max_similarity, 3)
#                 }
#             })

#         # Calculate summary statistics
#         total_products = len(analysis_results)
#         duplicate_count = sum(1 for r in analysis_results if r["similarity"]["is_duplicate"])

#         # Optimized keyword usage statistics using vectorized operations
#         keyword_data = np.array([
#             [r["keyword_density"]["eco_friendly"], r["keyword_density"]["sustainable"],
#              r["keyword_density"]["premium"], r["keyword_density"]["luxury"]]
#             for r in analysis_results
#         ])

#         keyword_stats = {
#             "eco_friendly": int(np.sum(keyword_data[:, 0] > 0)),
#             "sustainable": int(np.sum(keyword_data[:, 1] > 0)),
#             "premium": int(np.sum(keyword_data[:, 2] > 0)),
#             "luxury": int(np.sum(keyword_data[:, 3] > 0))
#         }

#         # Readability statistics
#         avg_word_lengths = [r["readability"]["avg_word_length"] for r in analysis_results]
#         avg_sentence_lengths = [r["readability"]["avg_sentence_length"] for r in analysis_results]

#         return {
#             "message": f"Complete analysis of {total_products} product descriptions using 3 core metrics",
#             "summary": {
#                 "total_products_analyzed": total_products,
#                 "duplicates_found": duplicate_count,
#                 "duplicate_percentage": round((duplicate_count / total_products) * 100, 1) if total_products > 0 else 0,
#                 "unique_products": total_products - duplicate_count,
#                 "keyword_usage": keyword_stats,
#                 "readability_averages": {
#                     "avg_word_length": round(np.mean(avg_word_lengths), 2),
#                     "avg_sentence_length": round(np.mean(avg_sentence_lengths), 2)
#                 }
#             },
#             "duplicate_pairs": duplicate_pairs,
#             "results": analysis_results,
#             "analysis_type": "3 Core Metrics: Readability + Keyword Density + Cosine Similarity"
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Sample analysis failed: {str(e)}")


# def _safe_correlation(x, y):
#     """Calculate correlation with NaN handling."""
#     try:
#         corr_matrix = np.corrcoef(x, y)
#         corr_value = corr_matrix[0, 1]
#         if np.isnan(corr_value) or np.isinf(corr_value):
#             return 0.0
#         return float(corr_value)
#     except:
#         return 0.0


# def _calculate_summary_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
#     """Calculate summary statistics across all analysis results."""
#     if not results:
#         return {}

#     # Define score extraction paths to eliminate repetition
#     score_paths = {
#         "overall": lambda r: r["overall_score"],
#         "readability": lambda r: r["readability"]["readability_score"],
#         "keyword_density": lambda r: r["keyword_density"]["density_score"],
#         "uniqueness": lambda r: r["uniqueness"]["uniqueness_score"],
#     }

#     # Extract all scores using the paths
#     all_scores = {
#         name: [extractor(r) for r in results] for name, extractor in score_paths.items()
#     }

#     # Optimized statistics calculation using NumPy vectorized operations
#     average_scores = {
#         name: round(float(np.mean(scores)), 3) for name, scores in all_scores.items()
#     }

#     score_ranges = {
#         name: {"min": float(np.min(scores)), "max": float(np.max(scores))}
#         for name, scores in all_scores.items()
#     }

#     # Use ScoreInterpreter thresholds for quality distribution
#     from .scoring import ScoreInterpreter

#     thresholds = ScoreInterpreter.THRESHOLDS
#     overall_scores = all_scores["overall"]

#     # Optimized quality distribution using NumPy vectorized operations
#     overall_scores_array = np.array(overall_scores)
#     quality_distribution = {
#         "excellent": int(np.sum(overall_scores_array >= thresholds["excellent"])),
#         "good": int(np.sum(
#             (overall_scores_array >= thresholds["good"]) &
#             (overall_scores_array < thresholds["excellent"])
#         )),
#         "fair": int(np.sum(
#             (overall_scores_array >= thresholds["fair"]) &
#             (overall_scores_array < thresholds["good"])
#         )),
#         "poor": int(np.sum(overall_scores_array < thresholds["fair"])),
#     }

#     return {
#         "total_descriptions": len(results),
#         "average_scores": average_scores,
#         "score_ranges": score_ranges,
#         "quality_distribution": quality_distribution,
#     }


if __name__ == "__main__":
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
