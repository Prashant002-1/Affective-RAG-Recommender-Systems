"""
Evaluation Metrics for Affective-RAG
Implements standard IR metrics and affective-specific metrics
"""

import numpy as np
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
import json
from datetime import datetime


@dataclass
class RetrievalMetrics:
    """Standard retrieval metrics"""
    precision_at_1: float = 0.0
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    ndcg_at_5: float = 0.0
    ndcg_at_10: float = 0.0
    mrr: float = 0.0
    map_score: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            'P@1': self.precision_at_1,
            'P@5': self.precision_at_5,
            'P@10': self.precision_at_10,
            'R@5': self.recall_at_5,
            'R@10': self.recall_at_10,
            'NDCG@5': self.ndcg_at_5,
            'NDCG@10': self.ndcg_at_10,
            'MRR': self.mrr,
            'MAP': self.map_score
        }


@dataclass
class AffectiveMetrics:
    """Affective-specific evaluation metrics"""
    emotion_alignment: float = 0.0
    affective_coherence: float = 0.0
    emotional_diversity: float = 0.0
    user_emotion_satisfaction: float = 0.0
    affective_precision_at_5: float = 0.0
    affective_precision_at_10: float = 0.0
    affective_displacement_error: float = 0.0
    semantic_recall_at_5: float = 0.0
    semantic_recall_at_10: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            'emotion_alignment': self.emotion_alignment,
            'affective_coherence': self.affective_coherence,
            'emotional_diversity': self.emotional_diversity,
            'user_emotion_satisfaction': self.user_emotion_satisfaction,
            'AP@5': self.affective_precision_at_5,
            'AP@10': self.affective_precision_at_10,
            'ADE': self.affective_displacement_error,
            'semantic_recall@5': self.semantic_recall_at_5,
            'semantic_recall@10': self.semantic_recall_at_10
        }


@dataclass
class ExperimentResult:
    """Complete experiment result with all metrics"""
    experiment_name: str
    config: Dict[str, Any]
    retrieval_metrics: RetrievalMetrics
    affective_metrics: AffectiveMetrics
    num_queries: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'experiment_name': self.experiment_name,
            'config': self.config,
            'retrieval_metrics': self.retrieval_metrics.to_dict(),
            'affective_metrics': self.affective_metrics.to_dict(),
            'num_queries': self.num_queries,
            'timestamp': self.timestamp,
            'notes': self.notes
        }

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


def compute_precision_at_k(
    retrieved: List[str],
    relevant: List[str],
    k: int
) -> float:
    """
    Compute Precision@K

    Args:
        retrieved: List of retrieved item IDs (ranked)
        relevant: List of relevant item IDs (ground truth)
        k: Number of top results to consider

    Returns:
        Precision@K score
    """
    if k <= 0 or not retrieved:
        return 0.0

    retrieved_at_k = retrieved[:k]
    relevant_set = set(relevant)
    hits = sum(1 for item in retrieved_at_k if item in relevant_set)

    return hits / k


def compute_recall_at_k(
    retrieved: List[str],
    relevant: List[str],
    k: int
) -> float:
    """
    Compute Recall@K

    Args:
        retrieved: List of retrieved item IDs (ranked)
        relevant: List of relevant item IDs (ground truth)
        k: Number of top results to consider

    Returns:
        Recall@K score
    """
    if not relevant or k <= 0:
        return 0.0

    retrieved_at_k = retrieved[:k]
    relevant_set = set(relevant)
    hits = sum(1 for item in retrieved_at_k if item in relevant_set)

    return hits / len(relevant)


def compute_ndcg(
    retrieved: List[str],
    relevant: List[str],
    k: int,
    relevance_scores: Optional[Dict[str, float]] = None
) -> float:
    """
    Compute Normalized Discounted Cumulative Gain (NDCG@K)

    Args:
        retrieved: List of retrieved item IDs (ranked)
        relevant: List of relevant item IDs
        k: Number of top results to consider
        relevance_scores: Optional relevance scores for items

    Returns:
        NDCG@K score
    """
    if k <= 0 or not retrieved:
        return 0.0

    relevant_set = set(relevant)

    # Calculate DCG
    dcg = 0.0
    for i, item in enumerate(retrieved[:k]):
        if item in relevant_set:
            rel = relevance_scores.get(item, 1.0) if relevance_scores else 1.0
            dcg += rel / np.log2(i + 2)

    # Calculate IDCG (ideal DCG)
    if relevance_scores:
        ideal_rels = sorted(
            [relevance_scores.get(item, 1.0) for item in relevant],
            reverse=True
        )[:k]
    else:
        ideal_rels = [1.0] * min(len(relevant), k)

    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_rels))

    if idcg == 0:
        return 0.0

    return dcg / idcg


def compute_mrr(
    retrieved: List[str],
    relevant: List[str]
) -> float:
    """
    Compute Mean Reciprocal Rank (MRR)

    Args:
        retrieved: List of retrieved item IDs (ranked)
        relevant: List of relevant item IDs

    Returns:
        MRR score
    """
    relevant_set = set(relevant)

    for i, item in enumerate(retrieved):
        if item in relevant_set:
            return 1.0 / (i + 1)

    return 0.0


def compute_map(
    retrieved: List[str],
    relevant: List[str]
) -> float:
    """
    Compute Mean Average Precision (MAP)

    Args:
        retrieved: List of retrieved item IDs (ranked)
        relevant: List of relevant item IDs

    Returns:
        MAP score
    """
    if not relevant:
        return 0.0

    relevant_set = set(relevant)
    num_relevant = 0
    precision_sum = 0.0

    for i, item in enumerate(retrieved):
        if item in relevant_set:
            num_relevant += 1
            precision_sum += num_relevant / (i + 1)

    if not relevant_set:
        return 0.0

    return precision_sum / len(relevant_set)


def compute_emotion_alignment(
    user_emotions: Dict[str, float],
    item_emotions: Dict[str, float]
) -> float:
    """
    Compute alignment between user and item emotion profiles

    Args:
        user_emotions: User emotion profile
        item_emotions: Item emotion profile

    Returns:
        Alignment score (0-1)
    """
    if not user_emotions or not item_emotions:
        return 0.0

    all_emotions = set(user_emotions.keys()) | set(item_emotions.keys())

    user_vec = np.array([user_emotions.get(e, 0.0) for e in all_emotions])
    item_vec = np.array([item_emotions.get(e, 0.0) for e in all_emotions])

    # Cosine similarity
    norm_user = np.linalg.norm(user_vec)
    norm_item = np.linalg.norm(item_vec)

    if norm_user == 0 or norm_item == 0:
        return 0.0

    return float(np.dot(user_vec, item_vec) / (norm_user * norm_item))


def compute_affective_coherence(
    item_emotions_list: List[Dict[str, float]]
) -> float:
    """
    Compute affective coherence across retrieved items
    Measures how emotionally consistent the recommendations are

    Args:
        item_emotions_list: List of item emotion profiles

    Returns:
        Coherence score (0-1)
    """
    if len(item_emotions_list) < 2:
        return 1.0

    all_emotions = set()
    for emotions in item_emotions_list:
        all_emotions.update(emotions.keys())

    vectors = []
    for emotions in item_emotions_list:
        vec = np.array([emotions.get(e, 0.0) for e in all_emotions])
        vectors.append(vec)

    # Compute pairwise similarities
    similarities = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            norm_i = np.linalg.norm(vectors[i])
            norm_j = np.linalg.norm(vectors[j])
            if norm_i > 0 and norm_j > 0:
                sim = np.dot(vectors[i], vectors[j]) / (norm_i * norm_j)
                similarities.append(sim)

    if not similarities:
        return 0.0

    return float(np.mean(similarities))


def compute_emotional_diversity(
    item_emotions_list: List[Dict[str, float]]
) -> float:
    """
    Compute emotional diversity of retrieved items
    Higher diversity indicates variety in emotional profiles

    Args:
        item_emotions_list: List of item emotion profiles

    Returns:
        Diversity score (0-1)
    """
    if not item_emotions_list:
        return 0.0

    all_emotions = set()
    for emotions in item_emotions_list:
        all_emotions.update(emotions.keys())

    # Compute mean emotion vector
    vectors = []
    for emotions in item_emotions_list:
        vec = np.array([emotions.get(e, 0.0) for e in all_emotions])
        vectors.append(vec)

    mean_vec = np.mean(vectors, axis=0)

    # Compute variance from mean
    variances = []
    for vec in vectors:
        variance = np.sum((vec - mean_vec) ** 2)
        variances.append(variance)

    avg_variance = np.mean(variances)

    # Normalize to 0-1 range
    return float(min(1.0, avg_variance / len(all_emotions)))


def compute_affective_precision_at_k(
    retrieved_ids: List[str],
    movie_affective_signatures: Dict[str, np.ndarray],
    target_emotion_vector: np.ndarray,
    k: int
) -> float:
    """
    Compute Affective Precision at K (AP@K).

    Measures the average cosine similarity between retrieved movies'
    affective signatures and the target emotion vector.

    Args:
        retrieved_ids: List of retrieved content IDs (ranked)
        movie_affective_signatures: Dict mapping content_id to 6-dim affective signature
        target_emotion_vector: User's target emotion vector (6-dim)
        k: Number of top results to consider

    Returns:
        Average cosine similarity (0-1, higher is better)
    """
    if k <= 0 or not retrieved_ids:
        return 0.0

    target_norm = np.linalg.norm(target_emotion_vector)
    if target_norm < 1e-8:
        return 0.0

    cosine_similarities = []

    for content_id in retrieved_ids[:k]:
        if content_id in movie_affective_signatures:
            c_m = movie_affective_signatures[content_id]
            c_m_norm = np.linalg.norm(c_m)

            if c_m_norm > 1e-8:
                cosine_sim = np.dot(c_m, target_emotion_vector) / (c_m_norm * target_norm)
                cosine_similarities.append(cosine_sim)

    if not cosine_similarities:
        return 0.0

    return float(np.mean(cosine_similarities))


def compute_affective_displacement_error(
    retrieved_ids: List[str],
    movie_affective_signatures: Dict[str, np.ndarray],
    target_emotion_vector: np.ndarray,
    k: int
) -> float:
    """
    Compute Affective Displacement Error (ADE).

    Measures the Mean Absolute Error between requested emotion intensities
    and retrieved movies' intensities. Lower is better.

    Args:
        retrieved_ids: List of retrieved content IDs (ranked)
        movie_affective_signatures: Dict mapping content_id to 6-dim affective signature
        target_emotion_vector: User's target emotion vector (6-dim)
        k: Number of top results to consider

    Returns:
        ADE score (lower is better, 0 is perfect match)
    """
    if k <= 0 or not retrieved_ids:
        return float('inf')

    errors = []
    for content_id in retrieved_ids[:k]:
        if content_id in movie_affective_signatures:
            c_m = movie_affective_signatures[content_id]
            mae = np.mean(np.abs(c_m - target_emotion_vector))
            errors.append(mae)

    if not errors:
        return float('inf')

    return float(np.mean(errors))


def compute_faithfulness_necessity_score(
    original_score: float,
    perturbed_score: float
) -> float:
    """
    Compute Faithfulness Necessity Score (FNS).

    Paper formula: FNS = (S_orig - S_perturbed) / S_orig

    Measures sensitivity to missing graph evidence.
    High FNS indicates recommendation was causally driven by graph structure.
    Low FNS implies model relied on latent correlations rather than explicit evidence.

    Args:
        original_score: Score before perturbation
        perturbed_score: Score after removing graph evidence

    Returns:
        FNS score (0-1, higher = more faithful to graph evidence)
    """
    if original_score <= 0:
        return 0.0

    return max(0.0, (original_score - perturbed_score) / original_score)


def compute_semantic_recall_at_k(
    retrieved_ids: List[str],
    semantic_relevant_ids: List[str],
    k: int
) -> float:
    """
    Compute Semantic Recall at K.

    Measures the proportion of narrative-relevant items retrieved,
    ensuring the system maintains utility as a standard search engine.

    Args:
        retrieved_ids: List of retrieved content IDs
        semantic_relevant_ids: List of semantically relevant content IDs
        k: Number of top results to consider

    Returns:
        Semantic Recall@K score (0-1)
    """
    return compute_recall_at_k(retrieved_ids, semantic_relevant_ids, k)


def compute_all_retrieval_metrics(
    retrieved: List[str],
    relevant: List[str],
    relevance_scores: Optional[Dict[str, float]] = None
) -> RetrievalMetrics:
    """
    Compute all standard retrieval metrics

    Args:
        retrieved: List of retrieved item IDs
        relevant: List of relevant item IDs
        relevance_scores: Optional relevance scores

    Returns:
        RetrievalMetrics object
    """
    return RetrievalMetrics(
        precision_at_1=compute_precision_at_k(retrieved, relevant, 1),
        precision_at_5=compute_precision_at_k(retrieved, relevant, 5),
        precision_at_10=compute_precision_at_k(retrieved, relevant, 10),
        recall_at_5=compute_recall_at_k(retrieved, relevant, 5),
        recall_at_10=compute_recall_at_k(retrieved, relevant, 10),
        ndcg_at_5=compute_ndcg(retrieved, relevant, 5, relevance_scores),
        ndcg_at_10=compute_ndcg(retrieved, relevant, 10, relevance_scores),
        mrr=compute_mrr(retrieved, relevant),
        map_score=compute_map(retrieved, relevant)
    )


def compute_all_affective_metrics(
    user_emotions: Dict[str, float],
    item_emotions_list: List[Dict[str, float]]
) -> AffectiveMetrics:
    """
    Compute all affective metrics

    Args:
        user_emotions: User emotion profile
        item_emotions_list: List of retrieved item emotion profiles

    Returns:
        AffectiveMetrics object
    """
    alignments = []
    for item_emotions in item_emotions_list:
        alignment = compute_emotion_alignment(user_emotions, item_emotions)
        alignments.append(alignment)

    return AffectiveMetrics(
        emotion_alignment=float(np.mean(alignments)) if alignments else 0.0,
        affective_coherence=compute_affective_coherence(item_emotions_list),
        emotional_diversity=compute_emotional_diversity(item_emotions_list),
        user_emotion_satisfaction=float(np.mean(alignments[:3])) if len(alignments) >= 3 else 0.0
    )
