"""
Affective-RAG: Emotion-Aware Retrieval-Augmented Generation for Recommendations

Main package providing the complete Affective-RAG system with:
- GCS data loading with pre-computed emotions
- Knowledge graph construction
- Multi-modal retrieval (semantic + emotion + knowledge)
- GNN training with self-supervised denoising
- LLM-powered response generation via Vertex AI
"""

from .system import ARAGSystem, ARAGSystemConfig

# Backward compatibility aliases
KRAGSystem = ARAGSystem
KRAGSystemConfig = ARAGSystemConfig

__all__ = [
    'ARAGSystem',
    'ARAGSystemConfig',
    # Backward compatibility
    'KRAGSystem',
    'KRAGSystemConfig'
]

__version__ = '1.0.0'
