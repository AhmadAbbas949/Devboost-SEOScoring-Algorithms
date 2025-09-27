"""
Utility functions for text processing and analysis.
"""

import re
from typing import List, Tuple, Dict
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def tokenize_text(text: str) -> Tuple[List[str], List[str]]:
    """
    Tokenize text into words and sentences.

    Args:
        text: Input text to tokenize

    Returns:
        Tuple of (words, sentences) as lists
    """
    # Clean and normalize text
    text = text.strip()

    # Extract sentences (split by periods, exclamation marks, question marks)
    sentence_pattern = r'[.!?]+\s*'
    sentences = [s.strip() for s in re.split(sentence_pattern, text) if s.strip()]

    # Extract words (alphanumeric sequences)
    word_pattern = r'\b[a-zA-Z]+\b'
    words = re.findall(word_pattern, text.lower())

    return words, sentences


def calculate_average_word_length(words: List[str]) -> float:
    """
    Calculate average word length in characters.

    Args:
        words: List of words

    Returns:
        Average word length as float
    """
    if not words:
        return 0.0

    total_length = sum(len(word) for word in words)
    return total_length / len(words)


def calculate_average_sentence_length(sentences: List[str]) -> float:
    """
    Calculate average sentence length in words.

    Args:
        sentences: List of sentences

    Returns:
        Average sentence length as float
    """
    if not sentences:
        return 0.0

    total_words = 0
    for sentence in sentences:
        words, _ = tokenize_text(sentence)
        total_words += len(words)

    return total_words / len(sentences)


def count_keyword_occurrences(text: str, keywords: List[str]) -> dict:
    """
    Count occurrences of specific keywords in text.

    Args:
        text: Input text to search
        keywords: List of keywords to count

    Returns:
        Dictionary mapping keywords to their counts
    """
    text_lower = text.lower()
    keyword_counts = {}

    for keyword in keywords:
        # Use word boundaries to match whole words/phrases
        pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
        matches = re.findall(pattern, text_lower)
        keyword_counts[keyword] = len(matches)

    return keyword_counts


def calculate_keyword_density(text: str, keywords: List[str]) -> dict:
    """
    Calculate keyword density as percentage of total words.

    Args:
        text: Input text to analyze
        keywords: List of keywords to calculate density for

    Returns:
        Dictionary mapping keywords to their density percentages
    """
    words, _ = tokenize_text(text)
    total_words = len(words)

    if total_words == 0:
        return {keyword: 0.0 for keyword in keywords}

    keyword_counts = count_keyword_occurrences(text, keywords)
    keyword_densities = {}

    for keyword, count in keyword_counts.items():
        # For multi-word keywords, count the number of keyword instances
        keyword_words = len(keyword.split())
        density = (count * keyword_words / total_words) * 100
        keyword_densities[keyword] = round(density, 2)

    return keyword_densities


def calculate_text_similarity_matrix(texts: List[str]) -> np.ndarray:
    """
    Calculate cosine similarity matrix using scikit-learn for professional ML implementation.

    Args:
        texts: List of text strings to compare

    Returns:
        NumPy array with similarity scores between all text pairs
    """
    if len(texts) < 2:
        return np.array([[1.0]])

    # Handle empty texts gracefully
    if not any(text.strip() for text in texts):
        return np.zeros((len(texts), len(texts)))

    try:
        # Use scikit-learn's CountVectorizer for professional text vectorization
        vectorizer = CountVectorizer(
            lowercase=True,
            stop_words='english',  # Remove common English stop words
            token_pattern=r'\b[a-zA-Z]+\b',  # Only alphabetic tokens
            max_features=1000,  # Limit feature space for efficiency
            binary=False  # Use term frequency, not just binary presence
        )

        # Transform texts to sparse matrix representation
        text_vectors = vectorizer.fit_transform(texts)

        # Calculate cosine similarity matrix using scikit-learn's optimized implementation
        similarity_matrix = cosine_similarity(text_vectors)

        return similarity_matrix

    except ValueError:
        # Fallback for edge cases (all empty texts, no valid tokens, etc.)
        return np.zeros((len(texts), len(texts)))


def calculate_uniqueness_score(text: str, reference_texts: List[str]) -> float:
    """
    Calculate uniqueness score for a text compared to reference texts.

    Args:
        text: Text to evaluate for uniqueness
        reference_texts: List of texts to compare against

    Returns:
        Uniqueness score (0.0 = identical, 1.0 = completely unique)
    """
    if not reference_texts:
        return 1.0

    all_texts = [text] + reference_texts
    similarity_matrix = calculate_text_similarity_matrix(all_texts)

    # Get similarities between target text (index 0) and all reference texts
    similarities = similarity_matrix[0][1:]

    # Uniqueness is 1 minus the maximum similarity to any reference text
    max_similarity = max(similarities) if len(similarities) > 0 else 0.0
    uniqueness = 1.0 - max_similarity

    return max(0.0, min(1.0, uniqueness))  # Clamp between 0 and 1




def calculate_statistical_features(texts: List[str]) -> Dict[str, np.ndarray]:
    """
    Calculate advanced statistical features using NumPy for enhanced text analysis.

    Args:
        texts: List of text strings to analyze

    Returns:
        Dictionary containing various statistical measures as NumPy arrays
    """
    if not texts:
        return {}

    # Vectorized calculations using NumPy for performance
    word_lengths = []
    sentence_lengths = []
    char_counts = []

    for text in texts:
        words, sentences = tokenize_text(text)
        word_lengths.append([len(word) for word in words] if words else [0])
        sentence_lengths.append(len(words) / max(len(sentences), 1))
        char_counts.append(len(text))

    # Convert to NumPy arrays for vectorized operations
    features = {
        'word_length_stats': {
            'mean': np.array([np.mean(lengths) for lengths in word_lengths]),
            'std': np.array([np.std(lengths) for lengths in word_lengths]),
            'median': np.array([np.median(lengths) for lengths in word_lengths])
        },
        'sentence_lengths': np.array(sentence_lengths),
        'char_counts': np.array(char_counts),
        'text_complexity': np.array([
            np.mean(lengths) * np.std(lengths) if len(lengths) > 1 else 0
            for lengths in word_lengths
        ])
    }

    return features


def calculate_similarity_metrics(similarity_matrix: np.ndarray) -> Dict[str, float]:
    """
    Calculate advanced similarity metrics using NumPy linear algebra operations.

    Args:
        similarity_matrix: NumPy array of similarity scores

    Returns:
        Dictionary with various similarity metrics
    """
    if similarity_matrix.size == 0:
        return {}

    # Use NumPy for efficient matrix operations
    # Exclude diagonal (self-similarity) for meaningful statistics
    mask = ~np.eye(similarity_matrix.shape[0], dtype=bool)
    off_diagonal = similarity_matrix[mask]

    metrics = {
        'mean_similarity': float(np.mean(off_diagonal)),
        'max_similarity': float(np.max(off_diagonal)) if off_diagonal.size > 0 else 0.0,
        'similarity_variance': float(np.var(off_diagonal)),
        'similarity_std': float(np.std(off_diagonal)),
        'duplicate_threshold_exceeded': int(np.sum(off_diagonal > 0.8))  # Count high similarities
    }

    return metrics