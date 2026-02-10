"""
Synthetic Test Set Generator for Affective-RAG Evaluation

Generates explicit queries with target emotion vectors and ground truth sets
based on semantic and affective thresholds.

Paper Section IV.A - Experimental Setup
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import random

from ..core.embeddings import ContentEmbedder
from ..core.emotion_detection import EMOTION_LABELS, EmotionProfile


@dataclass
class QueryTestCase:
    """
    Test case for evaluation.

    Attributes:
        query_id: Unique identifier
        query_text: Natural language intent (q_txt)
        target_emotions: Target affective vector (q_emo) as dict
        relevant_items: Ground truth set of relevant content IDs
        semantic_relevant: Items meeting semantic threshold only
        affective_relevant: Items meeting affective threshold only
    """
    query_id: str
    query_text: str
    target_emotions: Dict[str, float]
    relevant_items: List[str]
    semantic_relevant: List[str] = None
    affective_relevant: List[str] = None

    def get_emotion_vector(self) -> np.ndarray:
        """Convert target_emotions dict to 6-dim numpy array."""
        return np.array([
            self.target_emotions.get(e, 0.0) for e in EMOTION_LABELS
        ])


QUERY_TEMPLATES = [
    "A movie about {topic} that makes me feel {emotion}",
    "Looking for a {emotion} film about {topic}",
    "I want to watch something {emotion} involving {topic}",
    "Recommend a movie with {topic} that evokes {emotion}",
    "Find me a {adjective} {topic} movie",
]

TOPICS = [
    "space exploration", "family relationships", "war", "love and romance",
    "crime and mystery", "supernatural events", "coming of age", "survival",
    "technology", "nature and wildlife", "historical events", "sports",
    "music and art", "adventure and journey", "political intrigue",
    "friendship", "betrayal", "redemption", "loss and grief", "discovery"
]

EMOTION_DESCRIPTIONS = {
    'happiness': ['happy', 'joyful', 'uplifting', 'heartwarming', 'cheerful'],
    'sadness': ['sad', 'melancholic', 'emotional', 'touching', 'poignant'],
    'anger': ['intense', 'powerful', 'provocative', 'gripping', 'fierce'],
    'fear': ['scary', 'thrilling', 'suspenseful', 'terrifying', 'chilling'],
    'surprise': ['surprising', 'unexpected', 'mind-bending', 'twist-filled', 'shocking'],
    'disgust': ['disturbing', 'uncomfortable', 'unsettling', 'dark', 'gritty']
}


class SyntheticTestSetGenerator:
    """
    Generate synthetic test set for Affective-RAG evaluation.

    Paper Section IV.A:
    - Natural Language Intent (q_txt): Narrative descriptions
    - Target Affective Vector (q_emo): Specific intensity distribution
    - Relevance Criteria: semantic similarity > 0.7 AND affective fidelity > 0.85
    """

    def __init__(
        self,
        content_embedder: ContentEmbedder,
        semantic_threshold: float = 0.7,
        affective_threshold: float = 0.85,
        seed: int = 42
    ):
        """
        Initialize generator.

        Args:
            content_embedder: Embedder for computing semantic similarity
            semantic_threshold: Minimum semantic similarity for relevance
            affective_threshold: Minimum affective fidelity for relevance
            seed: Random seed for reproducibility
        """
        self.content_embedder = content_embedder
        self.semantic_threshold = semantic_threshold
        self.affective_threshold = affective_threshold
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def generate_test_set(
        self,
        content_items: List[Any],
        content_embeddings: np.ndarray,
        movie_affective_signatures: Dict[str, np.ndarray],
        num_queries: int = 1000,
        min_relevant: int = 5
    ) -> List[QueryTestCase]:
        """
        Generate synthetic test queries with ground truth.

        Args:
            content_items: List of ContentItem objects
            content_embeddings: Precomputed content embeddings (N x dim)
            movie_affective_signatures: Dict mapping content_id to 6-dim signature
            num_queries: Number of queries to generate
            min_relevant: Minimum relevant items per query (skip if fewer)

        Returns:
            List of QueryTestCase objects
        """
        test_cases = []
        content_ids = [str(item.id) for item in content_items]

        attempts = 0
        max_attempts = num_queries * 3

        while len(test_cases) < num_queries and attempts < max_attempts:
            attempts += 1

            target_emotions = self._sample_emotion_vector()
            query_text = self._generate_query_text(target_emotions)
            query_embedding = self.content_embedder.embed_content(query_text)

            semantic_relevant, affective_relevant, both_relevant = self._find_ground_truth(
                query_embedding,
                target_emotions,
                content_ids,
                content_embeddings,
                movie_affective_signatures
            )

            if len(both_relevant) >= min_relevant:
                test_case = QueryTestCase(
                    query_id=f"synthetic_{len(test_cases):04d}",
                    query_text=query_text,
                    target_emotions=dict(zip(EMOTION_LABELS, target_emotions)),
                    relevant_items=both_relevant,
                    semantic_relevant=semantic_relevant,
                    affective_relevant=affective_relevant
                )
                test_cases.append(test_case)

        return test_cases

    def _sample_emotion_vector(self) -> np.ndarray:
        """
        Sample a target emotion vector with realistic distribution.

        Returns:
            6-dim emotion vector with 1-2 dominant emotions
        """
        emotion_vector = np.zeros(6)

        num_dominant = random.choice([1, 2])
        dominant_indices = random.sample(range(6), num_dominant)

        for idx in dominant_indices:
            emotion_vector[idx] = random.uniform(0.6, 1.0)

        for idx in range(6):
            if idx not in dominant_indices:
                emotion_vector[idx] = random.uniform(0.0, 0.3)

        return emotion_vector

    def _generate_query_text(self, emotion_vector: np.ndarray) -> str:
        """
        Generate natural language query from emotion vector.

        Args:
            emotion_vector: 6-dim emotion vector

        Returns:
            Natural language query string
        """
        dominant_idx = np.argmax(emotion_vector)
        dominant_emotion = EMOTION_LABELS[dominant_idx]

        template = random.choice(QUERY_TEMPLATES)
        topic = random.choice(TOPICS)
        emotion_desc = random.choice(EMOTION_DESCRIPTIONS[dominant_emotion])

        query = template.format(
            topic=topic,
            emotion=emotion_desc,
            adjective=emotion_desc
        )

        return query

    def _find_ground_truth(
        self,
        query_embedding: np.ndarray,
        target_emotions: np.ndarray,
        content_ids: List[str],
        content_embeddings: np.ndarray,
        movie_signatures: Dict[str, np.ndarray]
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Find ground truth items meeting thresholds.

        Args:
            query_embedding: Query embedding vector
            target_emotions: Target emotion vector (6-dim)
            content_ids: List of content IDs
            content_embeddings: Content embedding matrix
            movie_signatures: Dict of affective signatures

        Returns:
            Tuple of (semantic_relevant, affective_relevant, both_relevant)
        """
        semantic_relevant = []
        affective_relevant = []
        both_relevant = []

        query_norm = np.linalg.norm(query_embedding)
        target_norm = np.linalg.norm(target_emotions)

        if query_norm < 1e-8 or target_norm < 1e-8:
            return [], [], []

        for i, content_id in enumerate(content_ids):
            content_emb = content_embeddings[i]
            content_norm = np.linalg.norm(content_emb)

            if content_norm < 1e-8:
                continue

            sem_sim = np.dot(query_embedding, content_emb) / (query_norm * content_norm)

            movie_sig = movie_signatures.get(content_id, np.zeros(6))
            movie_norm = np.linalg.norm(movie_sig)

            if movie_norm > 1e-8:
                aff_sim = np.dot(target_emotions, movie_sig) / (target_norm * movie_norm)
            else:
                aff_sim = 0.0

            meets_semantic = sem_sim >= self.semantic_threshold
            meets_affective = aff_sim >= self.affective_threshold

            if meets_semantic:
                semantic_relevant.append(content_id)
            if meets_affective:
                affective_relevant.append(content_id)
            if meets_semantic and meets_affective:
                both_relevant.append(content_id)

        return semantic_relevant, affective_relevant, both_relevant

    def generate_dissonance_queries(
        self,
        content_items: List[Any],
        content_embeddings: np.ndarray,
        movie_affective_signatures: Dict[str, np.ndarray],
        num_queries: int = 100
    ) -> List[QueryTestCase]:
        """
        Generate Affective-Semantic Dissonance queries.

        Paper Section IV.B: Scenarios where semantic content conflicts
        with desired emotional tone (e.g., "A happy movie about war").

        Args:
            content_items: List of ContentItem objects
            content_embeddings: Precomputed content embeddings
            movie_affective_signatures: Dict of affective signatures
            num_queries: Number of dissonance queries

        Returns:
            List of dissonance QueryTestCase objects
        """
        dissonance_pairs = [
            ("war", "happiness"),
            ("death", "happiness"),
            ("comedy", "fear"),
            ("romance", "disgust"),
            ("children", "fear"),
            ("family", "anger"),
        ]

        test_cases = []
        content_ids = [str(item.id) for item in content_items]

        for i in range(num_queries):
            topic, emotion = random.choice(dissonance_pairs)

            target_emotions = np.zeros(6)
            emotion_idx = EMOTION_LABELS.index(emotion)
            target_emotions[emotion_idx] = random.uniform(0.7, 1.0)

            query_text = f"A {emotion} movie about {topic}"
            query_embedding = self.content_embedder.embed_content(query_text)

            semantic_relevant, affective_relevant, both_relevant = self._find_ground_truth(
                query_embedding,
                target_emotions,
                content_ids,
                content_embeddings,
                movie_affective_signatures
            )

            test_case = QueryTestCase(
                query_id=f"dissonance_{i:04d}",
                query_text=query_text,
                target_emotions=dict(zip(EMOTION_LABELS, target_emotions)),
                relevant_items=both_relevant if both_relevant else affective_relevant[:10],
                semantic_relevant=semantic_relevant,
                affective_relevant=affective_relevant
            )
            test_cases.append(test_case)

        return test_cases


def save_test_set(test_cases: List[QueryTestCase], path: str):
    """Save test set to JSON file."""
    import json

    data = []
    for tc in test_cases:
        data.append({
            'query_id': tc.query_id,
            'query_text': tc.query_text,
            'target_emotions': tc.target_emotions,
            'relevant_items': tc.relevant_items,
            'semantic_relevant': tc.semantic_relevant,
            'affective_relevant': tc.affective_relevant
        })

    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_test_set(path: str) -> List[QueryTestCase]:
    """Load test set from JSON file."""
    import json

    with open(path, 'r') as f:
        data = json.load(f)

    test_cases = []
    for item in data:
        test_cases.append(QueryTestCase(
            query_id=item['query_id'],
            query_text=item['query_text'],
            target_emotions=item['target_emotions'],
            relevant_items=item['relevant_items'],
            semantic_relevant=item.get('semantic_relevant'),
            affective_relevant=item.get('affective_relevant')
        ))

    return test_cases


@dataclass
class LOOCVTestCase:
    """
    LOOCV Test Case per Section IV.A of the paper.

    For user history S = {M_1, ..., M_n}:
    - Ground Truth Target: M_n (final item)
    - Semantic Context: Aggregate of history {M_1, ..., M_{n-1}}
    - Target Affective Vector: Emotion distribution of M_n
    """
    test_id: str
    user_id: str
    history_movie_ids: List[str]
    ground_truth_id: str
    ground_truth_title: str
    history_embedding: np.ndarray
    target_emotions: Dict[str, float]
    metadata: Dict[str, Any] = None

    def get_emotion_vector(self) -> np.ndarray:
        """Convert target_emotions dict to 6-dim numpy array."""
        return np.array([
            self.target_emotions.get(e, 0.0) for e in EMOTION_LABELS
        ])


class LOOCVTestSetGenerator:
    """
    Leave-One-Out Cross-Validation Test Set Generator.

    From Section IV.A of the paper:
    "For each user history sequence S = {M_1, ..., M_n}, we extract the final
    item M_n as the Ground Truth Target. The system is then probed with a tuple:
    - Semantic Context (C_sem): An aggregate vector of {M_1, ..., M_{n-1}}
    - Target Affective Vector (V_aff): Emotion intensity of target M_n"
    """

    def __init__(
        self,
        content_embedder: ContentEmbedder,
        min_history_length: int = 5,
        seed: int = 42
    ):
        """
        Initialize LOOCV generator.

        Args:
            content_embedder: Embedder for computing semantic embeddings
            min_history_length: Minimum history length to include user
            seed: Random seed for reproducibility
        """
        self.content_embedder = content_embedder
        self.min_history_length = min_history_length
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def generate_from_user_ratings(
        self,
        user_ratings_df,
        content_items: List[Any],
        content_embeddings: np.ndarray,
        movie_affective_signatures: Dict[str, np.ndarray],
        max_users: int = None,
        min_rating: float = 3.5
    ) -> List[LOOCVTestCase]:
        """
        Generate LOOCV test cases from user rating data.

        Args:
            user_ratings_df: DataFrame with columns [userId, movieId, rating, timestamp]
            content_items: List of ContentItem objects
            content_embeddings: Precomputed content embeddings (N x dim)
            movie_affective_signatures: Dict mapping content_id to 6-dim signature
            max_users: Maximum number of users to process (None for all)
            min_rating: Minimum rating to include in history (positive signal)

        Returns:
            List of LOOCVTestCase objects
        """
        test_cases = []

        content_id_to_idx = {str(item.id): i for i, item in enumerate(content_items)}
        content_id_to_title = {str(item.id): item.title for item in content_items}

        filtered_ratings = user_ratings_df[user_ratings_df['rating'] >= min_rating].copy()
        filtered_ratings = filtered_ratings.sort_values(['userId', 'timestamp'])

        user_groups = filtered_ratings.groupby('userId')

        user_ids = list(user_groups.groups.keys())
        if max_users:
            user_ids = random.sample(user_ids, min(max_users, len(user_ids)))

        for user_id in user_ids:
            user_data = user_groups.get_group(user_id)
            movie_ids = [str(mid) for mid in user_data['movieId'].tolist()]

            valid_movie_ids = [mid for mid in movie_ids if mid in content_id_to_idx]

            if len(valid_movie_ids) < self.min_history_length:
                continue

            history_ids = valid_movie_ids[:-1]
            ground_truth_id = valid_movie_ids[-1]

            history_embeddings = []
            for mid in history_ids:
                if mid in content_id_to_idx:
                    idx = content_id_to_idx[mid]
                    history_embeddings.append(content_embeddings[idx])

            if not history_embeddings:
                continue

            history_embedding = np.mean(history_embeddings, axis=0)

            target_emotions = movie_affective_signatures.get(
                ground_truth_id, np.zeros(6)
            )
            target_emotions_dict = dict(zip(EMOTION_LABELS, target_emotions))

            test_case = LOOCVTestCase(
                test_id=f"loocv_{user_id}_{ground_truth_id}",
                user_id=str(user_id),
                history_movie_ids=history_ids,
                ground_truth_id=ground_truth_id,
                ground_truth_title=content_id_to_title.get(ground_truth_id, ''),
                history_embedding=history_embedding,
                target_emotions=target_emotions_dict,
                metadata={
                    'history_length': len(history_ids),
                    'total_ratings': len(valid_movie_ids)
                }
            )
            test_cases.append(test_case)

        return test_cases

    def generate_from_user_manager(
        self,
        user_manager,
        content_items: List[Any],
        content_embeddings: np.ndarray,
        movie_affective_signatures: Dict[str, np.ndarray],
        max_users: int = 100
    ) -> List[LOOCVTestCase]:
        """
        Generate LOOCV test cases from UserManager.

        Args:
            user_manager: UserManager instance with loaded ratings
            content_items: List of ContentItem objects
            content_embeddings: Precomputed content embeddings
            movie_affective_signatures: Dict of affective signatures
            max_users: Maximum number of users to sample

        Returns:
            List of LOOCVTestCase objects
        """
        test_cases = []

        content_id_to_idx = {str(item.id): i for i, item in enumerate(content_items)}
        content_id_to_title = {str(item.id): item.title for item in content_items}

        users_with_history = user_manager.get_users_with_min_ratings(
            self.min_history_length
        )

        if len(users_with_history) > max_users:
            users_with_history = random.sample(users_with_history, max_users)

        for user_id in users_with_history:
            user_profile = user_manager.get_user(user_id)
            if not user_profile or not user_profile.watched_movies:
                continue

            movie_ids = list(user_profile.watched_movies)
            valid_movie_ids = [mid for mid in movie_ids if mid in content_id_to_idx]

            if len(valid_movie_ids) < self.min_history_length:
                continue

            random.shuffle(valid_movie_ids)
            history_ids = valid_movie_ids[:-1]
            ground_truth_id = valid_movie_ids[-1]

            history_embeddings = [
                content_embeddings[content_id_to_idx[mid]]
                for mid in history_ids
                if mid in content_id_to_idx
            ]

            if not history_embeddings:
                continue

            history_embedding = np.mean(history_embeddings, axis=0)

            target_emotions = movie_affective_signatures.get(
                ground_truth_id, np.zeros(6)
            )
            target_emotions_dict = dict(zip(EMOTION_LABELS, target_emotions))

            test_case = LOOCVTestCase(
                test_id=f"loocv_{user_id}_{ground_truth_id}",
                user_id=str(user_id),
                history_movie_ids=history_ids,
                ground_truth_id=ground_truth_id,
                ground_truth_title=content_id_to_title.get(ground_truth_id, ''),
                history_embedding=history_embedding,
                target_emotions=target_emotions_dict,
                metadata={
                    'history_length': len(history_ids),
                    'total_movies': len(valid_movie_ids)
                }
            )
            test_cases.append(test_case)

        return test_cases


def save_loocv_test_set(test_cases: List[LOOCVTestCase], path: str):
    """Save LOOCV test set to JSON file."""
    import json

    data = []
    for tc in test_cases:
        data.append({
            'test_id': tc.test_id,
            'user_id': tc.user_id,
            'history_movie_ids': tc.history_movie_ids,
            'ground_truth_id': tc.ground_truth_id,
            'ground_truth_title': tc.ground_truth_title,
            'history_embedding': tc.history_embedding.tolist(),
            'target_emotions': tc.target_emotions,
            'metadata': tc.metadata
        })

    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_loocv_test_set(path: str) -> List[LOOCVTestCase]:
    """Load LOOCV test set from JSON file."""
    import json

    with open(path, 'r') as f:
        data = json.load(f)

    test_cases = []
    for item in data:
        test_cases.append(LOOCVTestCase(
            test_id=item['test_id'],
            user_id=item['user_id'],
            history_movie_ids=item['history_movie_ids'],
            ground_truth_id=item['ground_truth_id'],
            ground_truth_title=item['ground_truth_title'],
            history_embedding=np.array(item['history_embedding']),
            target_emotions=item['target_emotions'],
            metadata=item.get('metadata')
        ))

    return test_cases
