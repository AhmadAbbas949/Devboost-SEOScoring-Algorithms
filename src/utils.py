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

    # Optimized with NumPy vectorized operations
    word_lengths = np.array([len(word) for word in words])
    return float(np.mean(word_lengths))


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

    # Optimized with vectorized word counting
    sentence_word_counts = np.array([len(tokenize_text(sentence)[0]) for sentence in sentences])
    return float(np.mean(sentence_word_counts))


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


class AdvancedDuplicateDetector:
    """
    Professional ML-powered duplicate detection using advanced scikit-learn techniques.

    Optimizes the manual similarity processing with:
    - NearestNeighbors for efficient similarity queries
    - AgglomerativeClustering for automatic grouping
    - Vectorized operations for performance
    """

    def __init__(self, similarity_threshold: float = 0.79, min_similarity_for_pairs: float = 0.1):
        """
        Initialize the duplicate detector.

        Args:
            similarity_threshold: Threshold for marking items as duplicates
            min_similarity_for_pairs: Minimum similarity to create a duplicate pair
        """
        self.similarity_threshold = similarity_threshold
        self.min_similarity_for_pairs = min_similarity_for_pairs
        self.similarity_matrix = None

    def analyze_duplicates(self, texts: List[str], metadata: List[Dict] = None) -> Dict:
        """
        Advanced duplicate analysis using scikit-learn optimizations.

        Args:
            texts: List of text descriptions to analyze
            metadata: Optional metadata (e.g., product info) for each text

        Returns:
            Dictionary with duplicate analysis results
        """
        from sklearn.neighbors import NearestNeighbors
        from sklearn.cluster import AgglomerativeClustering

        if len(texts) < 2:
            # For single text, create a 1x1 identity matrix
            self.similarity_matrix = np.array([[1.0]]) if texts else np.array([])
            return {
                'duplicate_indices': [],
                'duplicate_pairs': [],
                'similarity_scores': [0.0] if texts else [],
                'uniqueness_scores': [1.0] if texts else [],
                'similarity_matrix': self.similarity_matrix
            }

        # Calculate similarity matrix using existing optimized function
        self.similarity_matrix = calculate_text_similarity_matrix(texts)

        # Method 1: Vectorized similarity analysis (replaces manual loops)
        duplicate_indices, similarity_scores, uniqueness_scores = self._vectorized_duplicate_analysis()

        # Method 2: Advanced duplicate pair detection using NearestNeighbors
        duplicate_pairs = self._find_duplicate_pairs_ml(texts, metadata)

        return {
            'duplicate_indices': duplicate_indices,
            'duplicate_pairs': duplicate_pairs,
            'similarity_scores': similarity_scores,
            'uniqueness_scores': uniqueness_scores,
            'similarity_matrix': self.similarity_matrix
        }

    def _vectorized_duplicate_analysis(self) -> Tuple[List[int], List[float], List[float]]:
        """
        Vectorized analysis replacing manual similarity processing.
        Uses NumPy operations instead of loops for better performance.
        """
        n_texts = self.similarity_matrix.shape[0]

        # Create mask to exclude diagonal (self-similarity)
        mask = ~np.eye(n_texts, dtype=bool)

        # Vectorized maximum similarity calculation
        masked_similarities = np.where(mask, self.similarity_matrix, -1)
        max_similarities = np.max(masked_similarities, axis=1)

        # Vectorized duplicate detection
        duplicate_mask = max_similarities >= self.similarity_threshold
        duplicate_indices = np.where(duplicate_mask)[0].tolist()

        # Calculate uniqueness scores
        uniqueness_scores = (1.0 - max_similarities).tolist()

        return duplicate_indices, max_similarities.tolist(), uniqueness_scores

    def _find_duplicate_pairs_ml(self, texts: List[str], metadata: List[Dict] = None) -> List[Dict]:
        """
        Advanced duplicate pair detection using NearestNeighbors.
        Replaces manual pair tracking with ML-optimized approach.
        """
        from sklearn.neighbors import NearestNeighbors

        # Use similarity matrix as feature space for NearestNeighbors
        # Convert similarity to distance: distance = 1 - similarity
        # Clamp negative similarities to 0 to avoid negative distances
        clamped_similarities = np.clip(self.similarity_matrix, 0.0, 1.0)
        distance_matrix = 1.0 - clamped_similarities

        # Initialize NearestNeighbors with precomputed distance matrix
        nn = NearestNeighbors(
            n_neighbors=min(3, len(texts)),  # Find top 2 nearest neighbors (excluding self)
            metric='precomputed'
        )
        nn.fit(distance_matrix)

        # Find nearest neighbors for each text
        distances, indices = nn.kneighbors(distance_matrix)

        duplicate_pairs = []
        processed_pairs = set()  # Track processed pairs to avoid duplicates

        for i, (neighbor_distances, neighbor_indices) in enumerate(zip(distances, indices)):
            # Skip self (first neighbor is always self with distance 0)
            for j in range(1, len(neighbor_indices)):
                neighbor_idx = neighbor_indices[j]
                similarity = 1.0 - neighbor_distances[j]  # Convert back to similarity

                if similarity >= self.similarity_threshold and similarity > self.min_similarity_for_pairs:
                    # Create ordered pair to avoid duplicates (smaller index first)
                    pair_key = (min(i, neighbor_idx), max(i, neighbor_idx))

                    if pair_key not in processed_pairs:
                        processed_pairs.add(pair_key)

                        # Create pair information
                        pair_info = {
                            'indices': [i, neighbor_idx],
                            'similarity': round(similarity, 3)
                        }

                        # Add metadata if provided
                        if metadata:
                            pair_info.update({
                                'product_1': {
                                    'id': metadata[i].get('id', i),
                                    'title': metadata[i].get('title', f'Text {i+1}')
                                },
                                'product_2': {
                                    'id': metadata[neighbor_idx].get('id', neighbor_idx),
                                    'title': metadata[neighbor_idx].get('title', f'Text {neighbor_idx+1}')
                                }
                            })

                        duplicate_pairs.append(pair_info)

        return duplicate_pairs

    def cluster_similar_texts(self, texts: List[str], n_clusters: int = None) -> Dict:
        """
        Advanced clustering of similar texts using AgglomerativeClustering.
        Provides additional insights beyond simple duplicate detection.
        """
        from sklearn.cluster import AgglomerativeClustering

        if len(texts) < 2:
            return {'cluster_labels': [0] if texts else [], 'n_clusters': 1 if texts else 0}

        # Convert similarity to distance for clustering
        # Clamp negative similarities to 0 to avoid negative distances
        clamped_similarities = np.clip(self.similarity_matrix, 0.0, 1.0)
        distance_matrix = 1.0 - clamped_similarities

        # Determine optimal number of clusters if not specified
        if n_clusters is None:
            # Use a heuristic: aim for clusters with average similarity > threshold
            n_clusters = max(1, int(len(texts) * (1 - self.similarity_threshold)))

        # Perform hierarchical clustering
        clustering = AgglomerativeClustering(
            n_clusters=min(n_clusters, len(texts)),
            metric='precomputed',
            linkage='average'
        )

        cluster_labels = clustering.fit_predict(distance_matrix)

        return {
            'cluster_labels': cluster_labels.tolist(),
            'n_clusters': len(set(cluster_labels)),
            'clusters_info': self._analyze_clusters(cluster_labels, texts)
        }

    def _analyze_clusters(self, cluster_labels: np.ndarray, texts: List[str]) -> List[Dict]:
        """Analyze cluster composition and statistics."""
        clusters_info = []
        unique_labels = set(cluster_labels)

        for label in unique_labels:
            cluster_indices = np.where(cluster_labels == label)[0]
            cluster_size = len(cluster_indices)

            # Optimized intra-cluster similarity calculation using NumPy advanced indexing
            if cluster_size > 1:
                # Use NumPy advanced indexing to get all pairwise similarities at once
                cluster_similarity_submatrix = self.similarity_matrix[np.ix_(cluster_indices, cluster_indices)]

                # Extract upper triangle (excluding diagonal) for pairwise similarities
                upper_triangle_mask = np.triu(np.ones_like(cluster_similarity_submatrix, dtype=bool), k=1)
                cluster_similarities = cluster_similarity_submatrix[upper_triangle_mask]

                avg_similarity = float(np.mean(cluster_similarities))
            else:
                avg_similarity = 1.0

            clusters_info.append({
                'cluster_id': int(label),
                'size': int(cluster_size),
                'indices': cluster_indices.tolist(),
                'avg_intra_similarity': round(avg_similarity, 3),
                'is_duplicate_cluster': avg_similarity >= self.similarity_threshold
            })

        return clusters_info


def analyze_duplicates_advanced(texts: List[str], metadata: List[Dict] = None,
                              similarity_threshold: float = 0.79) -> Dict:
    """
    Convenience function for advanced duplicate detection.

    Args:
        texts: List of text descriptions
        metadata: Optional metadata for each text
        similarity_threshold: Threshold for duplicate detection

    Returns:
        Complete duplicate analysis results
    """
    detector = AdvancedDuplicateDetector(similarity_threshold=similarity_threshold)
    results = detector.analyze_duplicates(texts, metadata)

    # Add clustering analysis for additional insights
    clustering_results = detector.cluster_similar_texts(texts)
    results['clustering'] = clustering_results

    return results
