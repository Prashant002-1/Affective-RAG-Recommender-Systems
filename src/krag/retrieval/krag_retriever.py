"""
K-RAG Enhanced Retrieval Engine
Combines semantic, emotion, and knowledge graph retrieval for improved recommendations
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass
from abc import ABC, abstractmethod

from ..core.embeddings import EmotionProfile, QueryEmbedder
from ..core.knowledge_graph import (
    ContentKnowledgeGraph, KRAGSubgraphRetriever,
    AdaptiveRetrievalPolicy, KRAGEncoder, GraphTransformerEncoder
)
from ..storage.vector_store import KRAGVectorStore


@dataclass
class RetrievalResult:
    """Single retrieval result with multiple similarity scores"""
    content_id: str
    title: str
    semantic_score: float
    emotion_score: float
    knowledge_score: float
    combined_score: float
    metadata: Dict[str, Any]
    explanation: str = ""


@dataclass
class QueryContext:
    """
    Context for a user query.
    
    Supports user-scoped retrieval via allowed_content_ids:
    - If set, only content in this set will be returned
    - This enables per-user recommendation scope (e.g., only recommend from movies they've rated)
    """
    query_text: str
    user_emotions: EmotionProfile
    query_embedding: np.ndarray
    emotion_embedding: np.ndarray
    user_preferences: Optional[Dict] = None
    filters: Optional[Dict] = None
    # User-scoped retrieval: only return content in this set
    allowed_content_ids: Optional[Set[str]] = None
    user_id: Optional[str] = None


class BaseRetriever(ABC):
    """Abstract base class for retrieval strategies"""

    @abstractmethod
    def retrieve(self, query_context: QueryContext, k: int = 10) -> List[RetrievalResult]:
        """Retrieve relevant content items"""
        pass


class SemanticRetriever(BaseRetriever):
    """
    Baseline semantic-only retriever (Vector-only RAG).
    
    Uses only semantic embeddings for retrieval, no emotion or knowledge graph.
    Useful as baseline for comparison experiments.
    """

    def __init__(self, vector_store: KRAGVectorStore):
        self.vector_store = vector_store

    def retrieve(self, query_context: QueryContext, k: int = 10) -> List[RetrievalResult]:
        """Retrieve using semantic similarity only"""
        # Request more candidates if user-scoped (many will be filtered)
        n_candidates = k * 5 if query_context.allowed_content_ids else k
        
        results = self.vector_store.semantic_search(
            query_context.query_embedding,
            n_results=n_candidates,
            where_filter=query_context.filters
        )

        retrieval_results = []
        for i, content_id in enumerate(results['ids'][0]):
            # User-scoped filtering
            if query_context.allowed_content_ids is not None and content_id not in query_context.allowed_content_ids:
                continue
                
            distance = results['distances'][0][i]
            metadata = results['metadatas'][0][i]

            # Convert distance to similarity score
            semantic_score = max(0.0, 1.0 - distance)

            result = RetrievalResult(
                content_id=content_id,
                title=metadata.get('title', ''),
                semantic_score=semantic_score,
                emotion_score=0.0,
                knowledge_score=0.0,
                combined_score=semantic_score,
                metadata=metadata,
                explanation=f"Semantic similarity: {semantic_score:.3f}"
            )
            retrieval_results.append(result)
            
            if len(retrieval_results) >= k:
                break

        return retrieval_results
    
    def explain_retrieval(self, result: RetrievalResult, query_context: QueryContext) -> str:
        """Explain why this item was retrieved"""
        return f"Semantic similarity score: {result.semantic_score:.3f}"


class EmotionRetriever(BaseRetriever):
    """
    Emotion-only retriever.
    
    Uses only emotion embeddings for retrieval.
    Useful as baseline for understanding emotion signal contribution.
    """

    def __init__(self, vector_store: KRAGVectorStore):
        self.vector_store = vector_store

    def retrieve(self, query_context: QueryContext, k: int = 10) -> List[RetrievalResult]:
        """Retrieve using emotion similarity only"""
        # Request more candidates if user-scoped
        n_candidates = k * 5 if query_context.allowed_content_ids else k
        
        results = self.vector_store.emotion_search(
            query_context.emotion_embedding,
            n_results=n_candidates,
            where_filter=query_context.filters
        )

        retrieval_results = []
        for i, content_id in enumerate(results['ids'][0]):
            # User-scoped filtering
            if query_context.allowed_content_ids is not None and content_id not in query_context.allowed_content_ids:
                continue
                
            distance = results['distances'][0][i]
            metadata = results['metadatas'][0][i]

            emotion_score = max(0.0, 1.0 - distance)

            result = RetrievalResult(
                content_id=content_id,
                title=metadata.get('title', ''),
                semantic_score=0.0,
                emotion_score=emotion_score,
                knowledge_score=0.0,
                combined_score=emotion_score,
                metadata=metadata,
                explanation=f"Emotion similarity: {emotion_score:.3f}"
            )
            retrieval_results.append(result)
            
            if len(retrieval_results) >= k:
                break

        return retrieval_results
    
    def explain_retrieval(self, result: RetrievalResult, query_context: QueryContext) -> str:
        """Explain why this item was retrieved"""
        return f"Emotion similarity score: {result.emotion_score:.3f}"


class SemanticEmotionRetriever(BaseRetriever):
    """
    Semantic + Emotion retriever (Vector-only RAG without Knowledge Graph).
    
    Combines semantic and emotion embeddings but NO knowledge graph.
    This is the key baseline for Graph+Vector vs Vector-only experiments.
    """
    
    def __init__(
        self, 
        vector_store: KRAGVectorStore,
        semantic_weight: float = 0.6,
        emotion_weight: float = 0.4
    ):
        self.vector_store = vector_store
        self.semantic_weight = semantic_weight
        self.emotion_weight = emotion_weight

    def retrieve(self, query_context: QueryContext, k: int = 10) -> List[RetrievalResult]:
        """Retrieve using semantic + emotion (no knowledge graph)"""
        # Request more candidates for fusion and filtering
        n_candidates = k * 5 if query_context.allowed_content_ids else k * 2
        
        # Semantic search
        semantic_results = self.vector_store.semantic_search(
            query_context.query_embedding,
            n_results=n_candidates,
            where_filter=query_context.filters
        )
        
        # Emotion search
        emotion_results = self.vector_store.emotion_search(
            query_context.emotion_embedding,
            n_results=n_candidates,
            where_filter=query_context.filters
        )
        
        # Fuse results
        score_dict = {}
        
        # Process semantic results
        for i, content_id in enumerate(semantic_results['ids'][0]):
            # User-scoped filtering
            if query_context.allowed_content_ids is not None and content_id not in query_context.allowed_content_ids:
                continue
            if content_id not in score_dict:
                score_dict[content_id] = {'semantic': 0.0, 'emotion': 0.0, 'metadata': {}}
            distance = semantic_results['distances'][0][i]
            score_dict[content_id]['semantic'] = max(0.0, 1.0 - distance)
            score_dict[content_id]['metadata'] = semantic_results['metadatas'][0][i]
        
        # Process emotion results
        for i, content_id in enumerate(emotion_results['ids'][0]):
            # User-scoped filtering
            if query_context.allowed_content_ids is not None and content_id not in query_context.allowed_content_ids:
                continue
            if content_id not in score_dict:
                score_dict[content_id] = {'semantic': 0.0, 'emotion': 0.0, 'metadata': {}}
            distance = emotion_results['distances'][0][i]
            score_dict[content_id]['emotion'] = max(0.0, 1.0 - distance)
            if not score_dict[content_id]['metadata']:
                score_dict[content_id]['metadata'] = emotion_results['metadatas'][0][i]
        
        # Calculate combined scores
        retrieval_results = []
        for content_id, scores in score_dict.items():
            combined = (
                self.semantic_weight * scores['semantic'] +
                self.emotion_weight * scores['emotion']
            )
            
            result = RetrievalResult(
                content_id=content_id,
                title=scores['metadata'].get('title', ''),
                semantic_score=scores['semantic'],
                emotion_score=scores['emotion'],
                knowledge_score=0.0,  # No knowledge graph
                combined_score=combined,
                metadata=scores['metadata'],
                explanation=f"Vector-only: semantic={scores['semantic']:.3f}, emotion={scores['emotion']:.3f}"
            )
            retrieval_results.append(result)
        
        # Sort and return top-k
        retrieval_results.sort(key=lambda x: x.combined_score, reverse=True)
        return retrieval_results[:k]
    
    def explain_retrieval(self, result: RetrievalResult, query_context: QueryContext) -> str:
        """Explain why this item was retrieved"""
        return (f"Vector-only retrieval (no knowledge graph):\n"
                f"  Semantic: {result.semantic_score:.3f} (weight: {self.semantic_weight})\n"
                f"  Emotion: {result.emotion_score:.3f} (weight: {self.emotion_weight})\n"
                f"  Combined: {result.combined_score:.3f}")


class KRAGRetriever(BaseRetriever):
    """
    Affective-RAG retriever implementing the scoring used in this project:

    Score = α·[λ·Semantic + (1-λ)·Graph] - (1-α)·AffectiveRMSE

    Where AffectiveRMSE is the normalized L2 distance between the user's target
    emotion vector and the content affective signature.
    """

    # Max possible Euclidean distance for 6-dim unit vectors: sqrt(6) ≈ 2.449
    MAX_EMOTION_DISTANCE = np.sqrt(6)

    def __init__(self,
                 vector_store: KRAGVectorStore,
                 knowledge_graph: ContentKnowledgeGraph,
                 krag_encoder: KRAGEncoder,
                 alpha: float = 0.7,
                 lambda_weight: float = 0.7,
                 candidate_pool: str = "semantic",
                 subgraph_retriever: Optional[KRAGSubgraphRetriever] = None):
        """
        Args:
            vector_store: Vector store for semantic search
            knowledge_graph: Knowledge graph with EVOKES edges
            krag_encoder: GNN encoder for subgraph embeddings
            alpha: Balance between (semantic+graph) vs affective
            lambda_weight: Balance between semantic vs graph within relevance term
            subgraph_retriever: Optional shared subgraph retriever from system
        """
        self.vector_store = vector_store
        self.knowledge_graph = knowledge_graph
        self.krag_encoder = krag_encoder
        self.alpha = alpha
        self.lambda_weight = lambda_weight
        # Candidate pool controls which items are considered before reranking:
        # - "semantic": semantic_search only (previous behavior)
        # - "semantic+emotion": union of semantic_search and emotion_search results
        self.candidate_pool = candidate_pool

        # Initialize adaptive retrieval policy
        self.adaptive_policy = AdaptiveRetrievalPolicy()

        # Use provided subgraph retriever or create new one
        if subgraph_retriever is not None:
            self.subgraph_retriever = subgraph_retriever
        else:
            self.subgraph_retriever = KRAGSubgraphRetriever(
                self.knowledge_graph,
                self.krag_encoder
            )

        # Cache for GNN-smoothed emotion vectors (populated after GNN training)
        self.smoothed_emotions: Dict[str, np.ndarray] = {}

    def retrieve(self, query_context: QueryContext, k: int = 10) -> List[RetrievalResult]:
        """
        Affective-RAG retrieval using nested formula:
        Score = α·[λ·Semantic + (1-λ)·Graph] - (1-α)·AffectiveRMSE

        Args:
            query_context: Query context with embeddings
            k: Number of results to return

        Returns:
            Ranked list of retrieval results
        """
        if query_context.allowed_content_ids is not None:
            candidate_multiplier = max(5, len(query_context.allowed_content_ids) // k) if k > 0 else 5
            n_candidates = min(k * candidate_multiplier, 500)
        else:
            n_candidates = k * 3

        semantic_results = self.vector_store.semantic_search(
            query_context.query_embedding,
            n_results=n_candidates,
            where_filter=query_context.filters
        )

        candidate_ids = set(semantic_results['ids'][0])

        if self.candidate_pool == "semantic+emotion":
            emotion_results = self.vector_store.emotion_search(
                query_context.emotion_embedding,
                n_results=n_candidates,
                where_filter=query_context.filters
            )
            candidate_ids |= set(emotion_results['ids'][0])
        user_emotion_vector = query_context.user_emotions.to_vector()

        affective_rmse_scores = {}
        graph_similarity_scores = {}

        for content_id in candidate_ids:
            affective_rmse_scores[content_id] = self._compute_affective_rmse(
                content_id, user_emotion_vector
            )
            graph_similarity_scores[content_id] = self._compute_graph_similarity(
                content_id, query_context.query_embedding
            )

        fused_results = self._fusion_ranking(
            semantic_results,
            affective_rmse_scores,
            graph_similarity_scores,
            query_context,
            k
        )

        return fused_results

    def _compute_affective_rmse(self, content_id: str, user_emotion_vector: np.ndarray) -> float:
        """
        Compute ||e_target - e_m||_2: Euclidean distance between user's target emotion
        vector and movie's affective signature.

        Uses GNN-smoothed emotions if available, otherwise falls back to EVOKES weights.

        Args:
            content_id: Movie ID
            user_emotion_vector: 6-dim target emotion vector [happiness, sadness, anger, fear, surprise, disgust]

        Returns:
            Normalized RMSE score in [0, 1] where 0 is perfect match
        """
        # Use GNN-smoothed emotions if available
        if content_id in self.smoothed_emotions:
            movie_emotion = self.smoothed_emotions[content_id]
        else:
            movie_emotion = self._get_movie_affective_signature(content_id)

        distance = np.linalg.norm(user_emotion_vector - movie_emotion)
        normalized_rmse = distance / self.MAX_EMOTION_DISTANCE

        return float(normalized_rmse)

    def _compute_graph_similarity(self, content_id: str, query_embedding: np.ndarray) -> float:
        """
        Compute sim(q_txt, h_Gm): Cosine similarity between query embedding and
        GNN-encoded subgraph embedding.

        Paper formula component: lambda3 * sim(q_txt, h_Gm)

        Args:
            content_id: Movie ID
            query_embedding: Query text embedding (768-dim)

        Returns:
            Similarity score in [0, 1]
        """
        if not hasattr(self.subgraph_retriever, 'subgraph_embeddings'):
            return 0.0

        subgraph_emb = self.subgraph_retriever.subgraph_embeddings.get(content_id)
        if subgraph_emb is None:
            return 0.0

        subgraph_emb = np.array(subgraph_emb)

        if len(query_embedding) != len(subgraph_emb):
            if len(subgraph_emb) > len(query_embedding):
                subgraph_emb = subgraph_emb[:len(query_embedding)]
            else:
                padded = np.zeros(len(subgraph_emb))
                padded[:len(query_embedding)] = query_embedding[:len(padded)]
                query_embedding = padded

        norm_q = np.linalg.norm(query_embedding)
        norm_g = np.linalg.norm(subgraph_emb)

        if norm_q < 1e-8 or norm_g < 1e-8:
            return 0.0

        sim = float(np.dot(query_embedding, subgraph_emb) / (norm_q * norm_g))
        return max(0.0, (sim + 1.0) / 2.0)

    def _get_movie_affective_signature(self, content_id: str) -> np.ndarray:
        """
        Get movie's affective signature C_m from EVOKES edge weights.
        Returns 6-dim vector [happiness, sadness, anger, fear, surprise, disgust].
        """
        emotion_idx = {
            'happiness': 0, 'sadness': 1, 'anger': 2,
            'fear': 3, 'surprise': 4, 'disgust': 5
        }
        signature = np.zeros(6)

        if content_id not in self.knowledge_graph.graph:
            return signature

        for _, target, data in self.knowledge_graph.graph.out_edges(content_id, data=True):
            if data.get('relation') == 'evokes' and target.startswith('emotion_'):
                emotion_name = target.replace('emotion_', '')
                if emotion_name in emotion_idx:
                    signature[emotion_idx[emotion_name]] = data.get('weight', 0.0)

        return signature

    def _fusion_ranking(
        self,
        semantic_results: Dict,
        affective_rmse_scores: Dict[str, float],
        graph_similarity_scores: Dict[str, float],
        query_context: QueryContext,
        k: int
    ) -> List[RetrievalResult]:
        """
        Nested scoring formula:
        Score = α·[λ·Semantic + (1-λ)·Graph] - (1-α)·AffectiveRMSE

        Args:
            semantic_results: Results from semantic search
            affective_rmse_scores: Normalized RMSE scores {content_id: score}
            graph_similarity_scores: Graph similarity scores {content_id: score}
            query_context: Original query context
            k: Number of final results

        Returns:
            Fused and ranked results
        """
        retrieval_results = []

        for i, content_id in enumerate(semantic_results['ids'][0]):
            if query_context.allowed_content_ids is not None and content_id not in query_context.allowed_content_ids:
                continue

            distance = semantic_results['distances'][0][i]
            semantic_score = max(0.0, 1.0 - distance)
            metadata = semantic_results['metadatas'][0][i]

            affective_rmse = affective_rmse_scores.get(content_id, 1.0)
            graph_similarity = graph_similarity_scores.get(content_id, 0.0)

            relevance_score = (
                self.lambda_weight * semantic_score +
                (1 - self.lambda_weight) * graph_similarity
            )

            combined_score = (
                self.alpha * relevance_score -
                (1 - self.alpha) * affective_rmse
            )

            explanation = self._generate_explanation(
                semantic_score, graph_similarity, affective_rmse, combined_score
            )

            result = RetrievalResult(
                content_id=content_id,
                title=metadata.get('title', ''),
                semantic_score=semantic_score,
                emotion_score=1.0 - affective_rmse,
                knowledge_score=graph_similarity,
                combined_score=combined_score,
                metadata=metadata,
                explanation=explanation
            )
            retrieval_results.append(result)

        retrieval_results.sort(key=lambda x: x.combined_score, reverse=True)
        return retrieval_results[:k]

    def _generate_explanation(
        self,
        semantic_score: float,
        graph_similarity: float,
        affective_rmse: float,
        combined_score: float
    ) -> str:
        """Generate explanation for nested formula scoring."""
        parts = []

        if semantic_score > 0.7:
            parts.append(f"High semantic ({semantic_score:.3f})")
        elif semantic_score > 0.4:
            parts.append(f"Moderate semantic ({semantic_score:.3f})")

        if graph_similarity > 0.5:
            parts.append(f"Strong graph ({graph_similarity:.3f})")
        elif graph_similarity > 0.3:
            parts.append(f"Moderate graph ({graph_similarity:.3f})")

        affective_match = 1.0 - affective_rmse
        if affective_match > 0.8:
            parts.append(f"Excellent affective (RMSE: {affective_rmse:.3f})")
        elif affective_match > 0.6:
            parts.append(f"Good affective (RMSE: {affective_rmse:.3f})")

        relevance = self.lambda_weight * semantic_score + (1 - self.lambda_weight) * graph_similarity
        formula_str = (f"Score = {self.alpha:.2f}*[{self.lambda_weight:.2f}*{semantic_score:.3f} + "
                      f"{1-self.lambda_weight:.2f}*{graph_similarity:.3f}] - "
                      f"{1-self.alpha:.2f}*{affective_rmse:.3f}")

        if parts:
            return "; ".join(parts) + f" [{formula_str} = {combined_score:.3f}]"
        return f"{formula_str} = {combined_score:.3f}"

    def get_knowledge_context(self, content_id: str) -> str:
        """Get knowledge context for content item using K-RAG"""
        return self.subgraph_retriever.get_knowledge_context(content_id)

    def explain_retrieval(self, result: RetrievalResult, query_context: QueryContext) -> str:
        """Detailed explanation of why this item was retrieved"""
        explanation = f"Retrieved '{result.title}' because:\n"
        explanation += f"• Semantic similarity (λ={self.lambda_weight:.2f}): {result.semantic_score:.3f}\n"
        explanation += f"• Graph similarity (1-λ={1-self.lambda_weight:.2f}): {result.knowledge_score:.3f}\n"

        user_emotions = query_context.user_emotions.to_dict()
        top_user_emotions = sorted(user_emotions.items(), key=lambda x: x[1], reverse=True)[:2]
        emotion_text = ", ".join([f"{emotion} ({score:.2f})" for emotion, score in top_user_emotions])
        affective_rmse = 1.0 - result.emotion_score
        explanation += f"• Affective match for [{emotion_text}]: RMSE={affective_rmse:.3f}\n"

        knowledge_context = self.get_knowledge_context(result.content_id)
        if knowledge_context:
            explanation += f"• Graph context: {knowledge_context}\n"

        relevance = self.lambda_weight * result.semantic_score + (1 - self.lambda_weight) * result.knowledge_score
        explanation += (f"• Combined: {self.alpha:.2f}*[{self.lambda_weight:.2f}*{result.semantic_score:.3f} + "
                       f"{1-self.lambda_weight:.2f}*{result.knowledge_score:.3f}] - "
                       f"{1-self.alpha:.2f}*{affective_rmse:.3f} = {result.combined_score:.3f}")

        return explanation


class AdaptiveKRAGRetriever(KRAGRetriever):
    """
    Adaptive retriever that adjusts alpha based on query characteristics.
    Higher emotion intensity -> lower alpha (more weight on affective component)
    """

    def retrieve(self, query_context: QueryContext, k: int = 10) -> List[RetrievalResult]:
        """
        Adaptive retrieval with dynamic alpha adjustment

        Args:
            query_context: Query context
            k: Number of results

        Returns:
            Adaptively retrieved and ranked results
        """
        emotion_intensity = self._calculate_emotion_intensity(query_context.user_emotions)
        adapted_alpha = self._adapt_alpha(emotion_intensity)

        original_alpha = self.alpha
        self.alpha = adapted_alpha

        try:
            results = super().retrieve(query_context, k)

            for result in results:
                result.explanation += f" [Adapted α={adapted_alpha:.2f} for emotion_intensity={emotion_intensity:.2f}]"

            return results

        finally:
            self.alpha = original_alpha

    def _calculate_emotion_intensity(self, emotions: EmotionProfile) -> float:
        """Calculate overall emotion intensity (0-1 scale)"""
        emotion_vector = emotions.to_vector()
        return float(np.linalg.norm(emotion_vector) / np.sqrt(len(emotion_vector)))

    def _adapt_alpha(self, emotion_intensity: float) -> float:
        """
        Adapt alpha based on emotion intensity.
        High emotion intensity -> lower alpha (prioritize affective match)
        Low emotion intensity -> higher alpha (prioritize semantic match)

        Args:
            emotion_intensity: Intensity of user emotions (0-1)

        Returns:
            Adapted alpha value
        """
        base_alpha = self.alpha

        if emotion_intensity > 0.7:
            return max(0.2, base_alpha - 0.2)
        elif emotion_intensity > 0.5:
            return max(0.3, base_alpha - 0.1)
        elif emotion_intensity < 0.2:
            return min(0.8, base_alpha + 0.2)

        return base_alpha


class RetrieverFactory:
    """Factory for creating different types of retrievers"""

    @staticmethod
    def create_retriever(retriever_type: str,
                        vector_store: KRAGVectorStore,
                        knowledge_graph: Optional[ContentKnowledgeGraph] = None,
                        krag_encoder: Optional[KRAGEncoder] = None,
                        content_items: Optional[List] = None,
                        **kwargs) -> BaseRetriever:
        """
        Create retriever based on type

        Args:
            retriever_type: Type of retriever ('semantic', 'emotion', 'krag', 'adaptive_krag', 'bm25')
            vector_store: Vector store instance
            knowledge_graph: Knowledge graph (for K-RAG variants)
            krag_encoder: K-RAG encoder with dual GNNs (for K-RAG variants)
            content_items: Content items (for BM25)
            **kwargs: Additional arguments

        Returns:
            Configured retriever instance
        """
        if retriever_type == "semantic":
            return SemanticRetriever(vector_store)

        elif retriever_type == "emotion":
            return EmotionRetriever(vector_store)

        elif retriever_type == "bm25":
            if not content_items:
                raise ValueError("BM25 retriever requires content_items")
            from .bm25_retriever import create_bm25_retriever_from_content_items
            return create_bm25_retriever_from_content_items(content_items)

        elif retriever_type == "krag":
            if not all([knowledge_graph, krag_encoder]):
                raise ValueError("K-RAG retriever requires knowledge_graph and krag_encoder")

            return KRAGRetriever(
                vector_store=vector_store,
                knowledge_graph=knowledge_graph,
                krag_encoder=krag_encoder,
                **kwargs
            )

        elif retriever_type == "adaptive_krag":
            if not all([knowledge_graph, krag_encoder]):
                raise ValueError("Adaptive K-RAG retriever requires knowledge_graph and krag_encoder")

            return AdaptiveKRAGRetriever(
                vector_store=vector_store,
                knowledge_graph=knowledge_graph,
                krag_encoder=krag_encoder,
                **kwargs
            )

        elif retriever_type == "vector_only":
            return SemanticEmotionRetriever(
                vector_store=vector_store,
                semantic_weight=kwargs.get('semantic_weight', 0.6),
                emotion_weight=kwargs.get('emotion_weight', 0.4)
            )

        elif retriever_type == "zero_shot_llm":
            from .llm_retriever import ZeroShotLLMRetriever, LLMConfig
            config = LLMConfig(
                model_name=kwargs.get('model_name', 'gemini-3-flash-preview'),
                project_id=kwargs.get('project_id')
            )
            retriever = ZeroShotLLMRetriever(config=config)
            if content_items:
                retriever.set_content_items(content_items)
            return retriever

        else:
            raise ValueError(
                f"Unknown retriever type: {retriever_type}. "
                f"Available: semantic, emotion, bm25, vector_only, krag, adaptive_krag, zero_shot_llm"
            )