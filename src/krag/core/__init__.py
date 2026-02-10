"""
Core Module for Affective-RAG
Provides embeddings, emotion detection, and knowledge graph components
"""

from .embeddings import (
    ContentItem,
    ContentEmbedder,
    EmotionEmbedder,
    HybridEmbedder,
    QueryEmbedder
)

from .emotion_detection import (
    EmotionProfile,
    UserEmotionProcessor,
    EMOTION_LABELS
)

from .knowledge_graph import (
    KnowledgeTriple,
    ContentKnowledgeGraph,
    GraphTransformerEncoder,
    KRAGEncoder,
    AdaptiveRetrievalPolicy,
    KRAGSubgraphRetriever
)

__all__ = [
    # Embeddings
    'ContentItem',
    'ContentEmbedder',
    'EmotionEmbedder',
    'HybridEmbedder',
    'QueryEmbedder',
    # Emotion
    'EmotionProfile',
    'UserEmotionProcessor',
    'EMOTION_LABELS',
    # Knowledge Graph
    'KnowledgeTriple',
    'ContentKnowledgeGraph',
    'GraphTransformerEncoder',
    'KRAGEncoder',
    'AdaptiveRetrievalPolicy',
    'KRAGSubgraphRetriever'
]

