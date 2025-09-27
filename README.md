# devBoost Text Analysis API

A comprehensive Shopify App for analyzing e-commerce product descriptions with AI-powered insights. This application provides text analysis capabilities including readability scoring, keyword density analysis, and uniqueness detection.

## 🚀 Features

- **Readability Analysis**: Calculate readability scores based on word length and sentence structure
- **Keyword Density**: Analyze keyword usage for SEO optimization (eco-friendly, sustainable, premium, luxury)
- **Uniqueness Detection**: Identify similar content using cosine similarity analysis
- **REST API**: Clean FastAPI endpoints with automatic documentation
- **Comprehensive Testing**: Full test coverage with pytest
- **Production Ready**: Modular code structure with proper error handling

## 📚 Libraries and Why They're Used

| Library                | Purpose                             | Why in This Project                                                |
| ---------------------- | ----------------------------------- | ------------------------------------------------------------------ |
| **FastAPI**            | Build REST APIs quickly and cleanly | Provides `/analyze` and `/recommend` endpoints with automatic docs |
| **Pydantic**           | Data validation & type enforcement  | Ensures incoming JSON matches `ProductInput` model                 |
| **scikit-learn**       | Machine learning utilities          | `CountVectorizer` and `cosine_similarity` for uniqueness scoring   |
| **re** (built-in)      | Regular expressions                 | Tokenizing text into words/sentences & keyword counting            |
| **json** (built-in)    | Parse JSON files                    | Load Shopify-style JSON into Python dictionaries                   |
| **pathlib** (built-in) | Clean file paths                    | Cross-platform safe file path handling for `data/products.json`    |
| **pytest**             | Unit testing                        | Quick tests for `ingest.py`, `analysis.py`, `api.py`               |
| **uvicorn**            | ASGI server for FastAPI             | Runs the FastAPI app locally with hot reload                       |

## 🛠️ Setup & Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Installation Steps

1. **Clone the repository** (or navigate to project directory)

   ```bash
   cd devboost-app
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```bash
   python -c "from src.api import app; print('Installation successful!')"
   ```

## 🏃‍♂️ Running the Application

### Start the API Server

```bash
uvicorn src.api:app --reload
```

The API will be available at:

- **Main API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs

### Run Tests

```bash
pytest
```

### Run Tests with Coverage

```bash
pytest --cov=src tests/
```

## 📖 API Documentation

### Endpoints

#### `GET /`

Health check endpoint

```json
{
  "message": "devBoost Text Analysis API",
  "status": "healthy",
  "endpoints": ["/analyze", "/recommend", "/docs"]
}
```

#### `POST /analyze`

Analyze product descriptions for readability, keyword density, and uniqueness.

**Request Body:**

```json
{
  "descriptions": [
    "Premium handcrafted leather bag with elegant design.",
    "Sustainable eco-friendly water bottle made from recycled materials."
  ],
  "keywords": ["premium", "luxury", "eco-friendly", "sustainable"]
}
```

**Response:**

```json
{
  "results": [
    {
      "text": "Premium handcrafted leather bag with elegant design.",
      "text_index": 0,
      "readability": {
        "avg_word_length": 6.25,
        "avg_sentence_length": 8.0,
        "readability_score": 0.875,
        "interpretation": "Excellent readability"
      },
      "keyword_density": {
        "keyword_densities": {
          "premium": 12.5,
          "luxury": 0.0,
          "eco-friendly": 0.0,
          "sustainable": 0.0
        },
        "total_density": 12.5,
        "density_score": 0.542,
        "interpretation": "Over-optimized keywords (12.5%)"
      },
      "uniqueness": {
        "uniqueness_score": 0.892,
        "interpretation": "Highly unique content"
      },
      "overall_score": 0.769,
      "overall_interpretation": "Good content quality"
    }
  ],
  "summary": {
    "total_descriptions": 2,
    "average_scores": {
      "overall": 0.735,
      "readability": 0.825,
      "keyword_density": 0.456,
      "uniqueness": 0.724
    },
    "quality_distribution": {
      "excellent": 0,
      "good": 2,
      "fair": 0,
      "poor": 0
    }
  }
}
```

#### `POST /recommend`

Get improvement recommendations for product descriptions.

**Request Body:**

```json
{
  "descriptions": [
    "Short text.",
    "This is a very long sentence with many words that makes it difficult to read and understand what the product is about."
  ]
}
```

**Response:**

```json
{
  "recommendations": [
    {
      "description_index": 0,
      "description": "Short text.",
      "overall_score": 0.234,
      "recommendations": [
        "Consider including keywords: eco-friendly, sustainable",
        "Consider rewriting content with focus on clarity and uniqueness"
      ]
    },
    {
      "description_index": 1,
      "description": "This is a very long sentence...",
      "overall_score": 0.412,
      "recommendations": [
        "Break long sentences into shorter ones",
        "Consider including keywords: premium, luxury"
      ]
    }
  ]
}
```

#### `GET /sample-analysis`

Demonstrate API functionality with sample product data.

## 🧪 Testing

The project includes comprehensive unit tests covering:

- **Data Ingestion**: File loading, validation, error handling
- **Text Analysis**: All scoring algorithms and edge cases
- **API Endpoints**: Request/response validation, error scenarios

### Running Specific Tests

```bash
# Test data ingestion
pytest tests/test_ingest.py -v

# Test analysis engine
pytest tests/test_analysis.py -v

# Test API endpoints
pytest tests/test_api.py -v
```

## 📁 Project Structure

```
devboost-app/
│
├── src/
│   ├── __init__.py
│   ├── ingest.py          # Data ingestion/cleaning
│   ├── analysis.py        # Text analysis engine (3 scores)
│   ├── api.py            # FastAPI app exposing /analyze and /recommend
│   └── utils.py          # Helpers (tokenize, cosine similarity)
│
├── tests/
│   ├── test_ingest.py    # Tests for data ingestion
│   ├── test_analysis.py  # Tests for analysis engine
│   └── test_api.py       # Tests for API endpoints
│
├── data/
│   └── products.json     # Sample Shopify-like product data
│
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🔍 Analysis Metrics

### Readability Score

- **Optimal Word Length**: 4-6 characters
- **Optimal Sentence Length**: 10-20 words
- **Score Range**: 0.0 (poor) to 1.0 (excellent)

### Keyword Density

- **Target Keywords**: eco-friendly, sustainable, premium, luxury
- **Optimal Density**: 2-8% total
- **Calculation**: (keyword occurrences × keyword words / total words) × 100

### Uniqueness Score

- **Method**: Cosine similarity using CountVectorizer
- **Score Range**: 0.0 (identical) to 1.0 (completely unique)
- **Comparison**: Against all other provided descriptions

## 🚀 Example Usage

### Python SDK Style Usage

```python
from src.analysis import TextAnalysisEngine
from src.ingest import ProductDataIngestion

# Initialize analyzer
analyzer = TextAnalysisEngine()

# Load sample data
ingestion = ProductDataIngestion()
products = ingestion.load_products()
descriptions = [p['description'] for p in products[:5]]

# Analyze multiple descriptions
results = analyzer.analyze_multiple_texts(descriptions)

for result in results:
    print(f"Overall Score: {result['overall_score']:.3f}")
    print(f"Interpretation: {result['overall_interpretation']}")
```

### cURL Examples

```bash
# Analyze descriptions
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "descriptions": [
      "Premium handcrafted leather bag with elegant design."
    ]
  }'

# Get recommendations
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{
    "descriptions": [
      "Short text with no keywords."
    ]
  }'

# Sample analysis
curl -X GET "http://localhost:8000/sample-analysis"
```

## 🏗️ Architecture

### Core Components

1. **Data Ingestion (`ingest.py`)**

   - Loads and validates JSON product data
   - Handles file errors and data cleaning
   - Provides metadata enrichment

2. **Text Analysis Engine (`analysis.py`)**

   - Implements three scoring algorithms
   - Generates actionable recommendations
   - Supports custom keyword configuration

3. **Utility Functions (`utils.py`)**

   - Text tokenization and processing
   - Similarity calculations using scikit-learn
   - Score normalization helpers

4. **REST API (`api.py`)**
   - FastAPI application with automatic docs
   - Input validation using Pydantic models
   - Comprehensive error handling

### Design Principles

- **Modularity**: Clear separation of concerns
- **Testability**: Comprehensive unit test coverage
- **Scalability**: Efficient algorithms for large datasets
- **Maintainability**: Clean code with type hints and documentation

## 🎯 Business Value

This tool helps e-commerce brands:

- **Optimize SEO**: Improve keyword density for better search rankings
- **Enhance Readability**: Make product descriptions more customer-friendly
- **Ensure Uniqueness**: Avoid duplicate content penalties
- **Save Time**: Automated analysis replaces manual content review
- **Increase Conversions**: Better descriptions lead to higher sales

## 📊 Performance Considerations

- **Memory Efficient**: Processes texts individually to handle large datasets
- **Fast Analysis**: Vectorized operations using NumPy and scikit-learn
- **Concurrent Ready**: Stateless design supports multiple simultaneous requests
- **Scalable Storage**: JSON-based data format for easy integration

## 🛡️ Error Handling

The API includes robust error handling for:

- Invalid JSON input
- Missing required fields
- File system errors
- Analysis computation failures
- Network timeouts

All errors return proper HTTP status codes with descriptive messages.

## 🔄 Future Enhancements

Potential improvements for production deployment:

- **Database Integration**: Replace JSON with PostgreSQL/MongoDB
- **Caching**: Redis for frequently analyzed content
- **Authentication**: API key management for enterprise customers
- **Rate Limiting**: Prevent API abuse
- **Batch Processing**: Handle large product catalogs efficiently
- **Advanced NLP**: Sentiment analysis and readability formulas
- **Multi-language**: Support for international product descriptions

---

Built with ❤️ for devBoost - Making e-commerce content analysis simple and powerful.
