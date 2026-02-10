"""
Retrieval Module for Affective-RAG

Provides multiple retrieval strategies for experimentation:
- SemanticRetriever: Semantic-only baseline
- EmotionRetriever: Emotion-only retrieval
- SemanticEmotionRetriever: Vector-only (semantic + emotion, no KG)
- KRAGRetriever: Full K-RAG with knowledge graph
- AdaptiveKRAGRetriever: K-RAG with adaptive weight tuning
- ZeroShotLLMRetriever: LLM parametric baseline (no retrieval)
"""

from .krag_retriever import (
    QueryContext,
    RetrievalResult,
    BaseRetriever,
    SemanticRetriever,
    EmotionRetriever,
    SemanticEmotionRetriever,
    KRAGRetriever,
    AdaptiveKRAGRetriever,
    RetrieverFactory
)

from .llm_retriever import (
    ZeroShotLLMRetriever,
    ZeroShotLLMRetrieverFactory,
    LLMConfig
)

__all__ = [
    'QueryContext',
    'RetrievalResult',
    'BaseRetriever',
    'SemanticRetriever',
    'EmotionRetriever',
    'SemanticEmotionRetriever',
    'KRAGRetriever',
    'AdaptiveKRAGRetriever',
    'RetrieverFactory',
    'ZeroShotLLMRetriever',
    'ZeroShotLLMRetrieverFactory',
    'LLMConfig'
]

