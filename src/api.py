"""
FastAPI application providing REST endpoints for text analysis.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn

from .analysis import TextAnalysisEngine
from .ingest import ProductDataIngestion


app = FastAPI(
    title="devBoost Text Analysis API",
    description="AI-powered text analysis for e-commerce product descriptions",
    version="1.0.0",
)

# Initialize analysis engine
analyzer = TextAnalysisEngine()


class ProductInput(BaseModel):
    """Input model for product analysis requests."""

    descriptions: List[str] = Field(
        ...,
        description="List of product descriptions to analyze",
        min_length=1,
        json_schema_extra={
            "example": ["Premium handcrafted leather bag with elegant design."]
        },
    )
    keywords: Optional[List[str]] = Field(
        None,
        description="Custom keywords to analyze for density",
        json_schema_extra={"example": ["premium", "luxury", "handcrafted"]},
    )


class AnalysisResponse(BaseModel):
    """Response model for analysis results."""

    results: List[Dict[str, Any]] = Field(
        ..., description="Analysis results for each description"
    )
    summary: Dict[str, Any] = Field(
        ..., description="Summary statistics across all descriptions"
    )


class RecommendationResponse(BaseModel):
    """Response model for recommendations."""

    recommendations: List[Dict[str, Any]] = Field(
        ..., description="Recommendations for each description"
    )


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": "devBoost Text Analysis API",
        "status": "healthy",
        "endpoints": ["/analyze", "/recommend", "/advanced-analysis", "/sample-analysis", "/docs"],
        "ml_libraries": ["NumPy", "scikit-learn", "FastAPI", "Pydantic"]
    }


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_descriptions(input_data: ProductInput):
    """
    Analyze product descriptions for readability, keyword density, and uniqueness.

    This endpoint accepts a list of product descriptions and returns comprehensive
    analysis including:
    - Readability scores (word length, sentence length)
    - Keyword density for target keywords
    - Uniqueness scores (similarity to other descriptions)
    - Overall quality scores and interpretations
    """
    try:
        # Update analyzer with custom keywords if provided
        if input_data.keywords:
            analyzer.target_keywords = input_data.keywords

        # Analyze all descriptions
        results = analyzer.analyze_multiple_texts(input_data.descriptions)

        # Calculate summary statistics
        summary = _calculate_summary_stats(results)

        return AnalysisResponse(results=results, summary=summary)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(input_data: ProductInput):
    """
    Get improvement recommendations for product descriptions.

    This endpoint analyzes product descriptions and provides actionable
    recommendations for improvement in areas such as:
    - Readability optimization
    - Keyword usage optimization
    - Content uniqueness enhancement
    """
    try:
        # Update analyzer with custom keywords if provided
        if input_data.keywords:
            analyzer.target_keywords = input_data.keywords

        # Analyze descriptions and generate recommendations
        results = analyzer.analyze_multiple_texts(input_data.descriptions)

        recommendations = []
        for i, result in enumerate(results):
            recs = analyzer.generate_recommendations(result)
            recommendations.append(
                {
                    "description_index": i,
                    "description": result["text"],
                    "overall_score": result["overall_score"],
                    "recommendations": recs,
                }
            )

        return RecommendationResponse(recommendations=recommendations)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Recommendation generation failed: {str(e)}"
        )


@app.get("/sample-analysis")
async def sample_analysis():
    """
    Demonstrate API functionality with sample product data.

    This endpoint loads sample products and returns analysis results
    to showcase the API capabilities.
    """
    try:
        # Load sample data
        ingestion = ProductDataIngestion()
        products = ingestion.load_products()

        # Take first 5 products for demo
        sample_descriptions = [p["description"] for p in products[:5]]

        # Analyze sample data
        results = analyzer.analyze_multiple_texts(sample_descriptions)
        summary = _calculate_summary_stats(results)

        return {
            "message": "Sample analysis of first 5 products",
            "results": results,
            "summary": summary,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sample analysis failed: {str(e)}")


@app.post("/advanced-analysis")
async def advanced_analysis(input_data: ProductInput):
    """
    Advanced text analysis using NumPy and scikit-learn for enhanced insights.

    This endpoint showcases professional ML library usage with:
    - scikit-learn's CountVectorizer for text vectorization
    - NumPy arrays for efficient statistical computations
    - Advanced similarity metrics and clustering insights
    """
    try:
        from .utils import calculate_statistical_features, calculate_similarity_metrics, calculate_text_similarity_matrix
        import numpy as np

        # Update analyzer with custom keywords if provided
        if input_data.keywords:
            analyzer.target_keywords = input_data.keywords

        # Standard analysis
        results = analyzer.analyze_multiple_texts(input_data.descriptions)

        # Advanced NumPy-powered statistical analysis
        try:
            statistical_features = calculate_statistical_features(input_data.descriptions)
        except Exception:
            statistical_features = {}

        # scikit-learn powered similarity analysis
        similarity_matrix = calculate_text_similarity_matrix(input_data.descriptions)
        similarity_metrics = calculate_similarity_metrics(similarity_matrix)

        # Enhanced insights using NumPy operations
        if len(input_data.descriptions) > 1:
            # Cluster analysis using similarity patterns
            similarity_scores = []
            for i in range(len(input_data.descriptions)):
                for j in range(i + 1, len(input_data.descriptions)):
                    similarity_scores.append(similarity_matrix[i, j])

            clustering_insights = {
                "similarity_distribution": {
                    "mean": float(np.mean(similarity_scores)),
                    "std": float(np.std(similarity_scores)),
                    "min": float(np.min(similarity_scores)),
                    "max": float(np.max(similarity_scores)),
                    "percentiles": {
                        "25th": float(np.percentile(similarity_scores, 25)),
                        "50th": float(np.percentile(similarity_scores, 50)),
                        "75th": float(np.percentile(similarity_scores, 75))
                    }
                },
                "duplicate_risk_pairs": [
                    {
                        "description_1_index": int(i),
                        "description_2_index": int(j),
                        "similarity_score": float(similarity_matrix[i, j])
                    }
                    for i in range(len(input_data.descriptions))
                    for j in range(i + 1, len(input_data.descriptions))
                    if similarity_matrix[i, j] > 0.7  # High similarity threshold
                ]
            }
        else:
            clustering_insights = {"message": "Need multiple descriptions for clustering analysis"}

        # Convert NumPy arrays to JSON-serializable format
        def convert_numpy_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            return obj

        return {
            "standard_analysis": results,
            "statistical_features": convert_numpy_types(statistical_features) if statistical_features else {},
            "similarity_metrics": convert_numpy_types(similarity_metrics),
            "clustering_insights": convert_numpy_types(clustering_insights),
            "ml_library_usage": {
                "sklearn_vectorization": "Used CountVectorizer for professional text vectorization",
                "numpy_operations": "Leveraged NumPy for efficient statistical computations",
                "matrix_operations": "Applied linear algebra for similarity analysis"
            }
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Advanced analysis failed: {str(e)}"
        )


def _calculate_summary_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate summary statistics across all analysis results."""
    if not results:
        return {}

    # Define score extraction paths to eliminate repetition
    score_paths = {
        "overall": lambda r: r["overall_score"],
        "readability": lambda r: r["readability"]["readability_score"],
        "keyword_density": lambda r: r["keyword_density"]["density_score"],
        "uniqueness": lambda r: r["uniqueness"]["uniqueness_score"],
    }

    # Extract all scores using the paths
    all_scores = {
        name: [extractor(r) for r in results] for name, extractor in score_paths.items()
    }

    # Calculate statistics using list comprehension to reduce repetition
    average_scores = {
        name: round(sum(scores) / len(scores), 3) for name, scores in all_scores.items()
    }

    score_ranges = {
        name: {"min": min(scores), "max": max(scores)}
        for name, scores in all_scores.items()
    }

    # Use ScoreInterpreter thresholds for quality distribution
    from .scoring import ScoreInterpreter

    thresholds = ScoreInterpreter.THRESHOLDS
    overall_scores = all_scores["overall"]

    quality_distribution = {
        "excellent": len([s for s in overall_scores if s >= thresholds["excellent"]]),
        "good": len(
            [
                s
                for s in overall_scores
                if thresholds["good"] <= s < thresholds["excellent"]
            ]
        ),
        "fair": len(
            [s for s in overall_scores if thresholds["fair"] <= s < thresholds["good"]]
        ),
        "poor": len([s for s in overall_scores if s < thresholds["fair"]]),
    }

    return {
        "total_descriptions": len(results),
        "average_scores": average_scores,
        "score_ranges": score_ranges,
        "quality_distribution": quality_distribution,
    }


if __name__ == "__main__":
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
