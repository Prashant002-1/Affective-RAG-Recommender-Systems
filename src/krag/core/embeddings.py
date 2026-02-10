"""
Content and Semantic Embedding Generation for Affective-RAG
Handles text embeddings and emotion vector representations

Embedding Dimensions:
- Semantic (SentenceBERT): 768
- Emotion (Ekman 6): 6
- Knowledge (GNN): 1024
"""

import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from .emotion_detection import EmotionProfile, EMOTION_LABELS


@dataclass
class ContentItem:
    """Represents a content item (movie) with metadata"""
    id: str
    title: str
    description: str
    genres: List[str]
    year: Optional[int] = None
    emotions: Optional[EmotionProfile] = None
    metadata: Optional[Dict] = None


class ContentEmbedder:
    """
    Generate semantic embeddings for content using sentence transformers.
    Optimized for Affective-RAG content representation.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self.embedding_dim = 768  # Default for all-mpnet-base-v2
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def initialize(self):
        """Load the sentence transformer model"""
        print(f"Loading content embedding model: {self.model_name}")
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            print(f"  Loaded successfully. Dimension: {self.embedding_dim}")
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            print("Falling back to all-MiniLM-L6-v2")
            self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def create_content_representation(self, content_item: ContentItem) -> str:
        """
        Create a comprehensive text representation of content for embedding.
        
        Args:
            content_item: ContentItem with title, description, genres, etc.
            
        Returns:
            Combined text representation
        """
        parts = []

        # Title (most important)
        if content_item.title:
            parts.append(f"Title: {content_item.title}")

        # Description/plot
        if content_item.description:
            parts.append(f"Description: {content_item.description}")

        # Genres
        if content_item.genres:
            genre_text = ", ".join(content_item.genres)
            parts.append(f"Genres: {genre_text}")

        # Year
        if content_item.year:
            parts.append(f"Year: {content_item.year}")

        # Emotional context
        if content_item.emotions:
            emotion_desc = self._emotions_to_text(content_item.emotions)
            if emotion_desc:
                parts.append(f"Emotional tone: {emotion_desc}")

        return ". ".join(parts)

    def _emotions_to_text(self, emotions: EmotionProfile) -> str:
        """Convert emotion profile to descriptive text"""
        descriptions = []
        emotion_dict = emotions.to_dict()

        # Map emotion scores to descriptions
        # Uses 'happiness' (not 'joy') to match GCS dataset
        emotion_descriptors = {
            'happiness': [
                (0.7, "joyful and uplifting"),
                (0.4, "cheerful"),
                (0.2, "somewhat pleasant")
            ],
            'sadness': [
                (0.7, "melancholic and emotional"),
                (0.4, "touching and poignant"),
                (0.2, "somewhat somber")
            ],
            'anger': [
                (0.7, "intense and dramatic"),
                (0.4, "tense and confrontational"),
                (0.2, "mildly intense")
            ],
            'fear': [
                (0.7, "thrilling and suspenseful"),
                (0.4, "tense and exciting"),
                (0.2, "mildly suspenseful")
            ],
            'surprise': [
                (0.7, "full of unexpected twists"),
                (0.4, "surprising and unpredictable"),
                (0.2, "has some surprises")
            ],
            'disgust': [
                (0.7, "provocative and disturbing"),
                (0.4, "edgy and unsettling"),
                (0.2, "somewhat dark")
            ]
        }

        for emotion, score in emotion_dict.items():
            if emotion in emotion_descriptors:
                for threshold, description in emotion_descriptors[emotion]:
                    if score >= threshold:
                        descriptions.append(description)
                        break

        return ", ".join(descriptions) if descriptions else ""

    def embed_content(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        if not self.model:
            raise ValueError("Model not initialized. Call initialize() first.")

        try:
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            return embedding
        except Exception as e:
            print(f"Error embedding text: {e}")
            return np.zeros(self.embedding_dim)

    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Efficiently embed multiple texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            
        Returns:
            Array of embeddings with shape (N, embedding_dim)
        """
        if not self.model:
            raise ValueError("Model not initialized. Call initialize() first.")

        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            return embeddings
        except Exception as e:
            print(f"Error in batch embedding: {e}")
            return np.zeros((len(texts), self.embedding_dim))

    def embed_content_items(
        self,
        content_items: List[ContentItem],
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Embed a list of content items.
        
        Args:
            content_items: List of ContentItem objects
            batch_size: Batch size for processing
            
        Returns:
            Array of embeddings
        """
        texts = [self.create_content_representation(item) for item in content_items]
        return self.embed_batch(texts, batch_size=batch_size)


class EmotionEmbedder:
    """
    Convert emotion profiles to vectors for similarity search.
    Uses Ekman's 6 basic emotions aligned with GCS data.
    """

    def __init__(self, emotion_labels: Optional[List[str]] = None):
        self.emotion_labels = emotion_labels or EMOTION_LABELS
        self.emotion_dim = len(self.emotion_labels)

    def emotions_to_vector(self, emotions: EmotionProfile) -> np.ndarray:
        """
        Convert EmotionProfile to fixed-dimension vector.
        
        Args:
            emotions: EmotionProfile object
            
        Returns:
            Numpy array with emotion scores
        """
        emotion_dict = emotions.to_dict()
        vector = np.array([
            emotion_dict.get(label, 0.0) for label in self.emotion_labels
        ])
        return vector

    def normalize_emotion_vector(self, emotion_vector: np.ndarray) -> np.ndarray:
        """Normalize emotion vector for cosine similarity"""
        norm = np.linalg.norm(emotion_vector)
        if norm > 0:
            return emotion_vector / norm
        return emotion_vector

    def batch_emotions_to_vectors(self, emotion_profiles: List[EmotionProfile]) -> np.ndarray:
        """
        Convert multiple emotion profiles to vectors.
        
        Args:
            emotion_profiles: List of EmotionProfile objects
            
        Returns:
            Array of shape (N, emotion_dim)
        """
        vectors = np.array([
            self.emotions_to_vector(emotions) for emotions in emotion_profiles
        ])
        return vectors


class HybridEmbedder:
    """
    Combines semantic and emotion embeddings for Affective-RAG retrieval.
    """

    def __init__(
        self,
        content_embedder: ContentEmbedder,
        emotion_embedder: EmotionEmbedder
    ):
        self.content_embedder = content_embedder
        self.emotion_embedder = emotion_embedder

    def create_hybrid_embedding(
        self,
        content_item: ContentItem
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create separate semantic and emotion embeddings for content.
        
        Args:
            content_item: ContentItem object
            
        Returns:
            Tuple of (semantic_embedding, emotion_embedding)
        """
        # Semantic embedding
        content_text = self.content_embedder.create_content_representation(content_item)
        semantic_embedding = self.content_embedder.embed_content(content_text)

        # Emotion embedding
        if content_item.emotions:
            emotion_embedding = self.emotion_embedder.emotions_to_vector(content_item.emotions)
            emotion_embedding = self.emotion_embedder.normalize_emotion_vector(emotion_embedding)
        else:
            emotion_embedding = np.zeros(self.emotion_embedder.emotion_dim)

        return semantic_embedding, emotion_embedding

    def batch_create_hybrid_embeddings(
        self,
        content_items: List[ContentItem],
        batch_size: int = 32
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create embeddings for multiple content items.
        
        IMPORTANT: This creates 768-dim TEXT-BASED emotion embeddings to match
        the Colab-generated embeddings stored in GCS. This ensures consistency
        between local fallback generation and GCS-stored embeddings.
        
        Args:
            content_items: List of ContentItem objects
            batch_size: Batch size for processing
            
        Returns:
            Tuple of (semantic_embeddings, emotion_embeddings) arrays
            Both are 768-dim embeddings for ChromaDB compatibility.
        """
        # Semantic embeddings (768-dim)
        semantic_embeddings = self.content_embedder.embed_content_items(
            content_items, batch_size=batch_size
        )

        # Emotion embeddings (768-dim TEXT-BASED, matching Colab format)
        # NOT 6-dim raw vectors - those cause dimension mismatch with ChromaDB!
        emotion_texts = []
        for item in content_items:
            text_repr = self.content_embedder.create_content_representation(item)
            emotion_desc = self._emotions_to_text(item.emotions) if item.emotions else "neutral emotional tone"
            emotion_texts.append(f"{text_repr} {emotion_desc}")
        
        emotion_embeddings = self.content_embedder.embed_batch(
            emotion_texts, batch_size=batch_size
        )

        return semantic_embeddings, emotion_embeddings
    
    def _emotions_to_text(self, emotions: EmotionProfile) -> str:
        """
        Convert EmotionProfile to text description for embedding.
        Must match Colab script format for consistency.
        """
        emotion_dict = emotions.to_dict()
        
        # Sort by intensity and describe top emotions
        top_emotions = sorted(emotion_dict.items(), key=lambda x: -x[1])[:3]
        desc_parts = []
        for name, score in top_emotions:
            if score > 0.3:
                desc_parts.append(f"{name} ({score:.0%})")
        
        if desc_parts:
            return f"Emotions: {', '.join(desc_parts)}"
        return "neutral emotional tone"


class QueryEmbedder:
    """
    Embed user queries for retrieval.
    """

    def __init__(
        self,
        content_embedder: ContentEmbedder,
        emotion_embedder: EmotionEmbedder
    ):
        self.content_embedder = content_embedder
        self.emotion_embedder = emotion_embedder

    def embed_query(
        self,
        query_text: str,
        user_emotions: EmotionProfile
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Embed user query and emotions.
        
        Args:
            query_text: User's search query
            user_emotions: User's emotional state
            
        Returns:
            Tuple of (query_embedding, emotion_embedding)
            Both are 768-dim text-based embeddings for ChromaDB compatibility.
        """
        # Semantic query embedding (768-dim)
        query_embedding = self.content_embedder.embed_content(query_text)

        # Emotion embedding (768-dim text-based, NOT 6-dim raw vector!)
        # Must match the format used by Colab script for ChromaDB indexing
        emotion_text = self._emotions_to_text(query_text, user_emotions)
        emotion_embedding = self.content_embedder.embed_content(emotion_text)

        return query_embedding, emotion_embedding
    
    def _emotions_to_text(self, query_text: str, emotions: EmotionProfile) -> str:
        """
        Convert query + emotions to text for embedding.
        Must match Colab script format for consistency.
        """
        emotion_dict = emotions.to_dict()
        
        # Sort by intensity and describe top emotions
        top_emotions = sorted(emotion_dict.items(), key=lambda x: -x[1])[:3]
        emotion_parts = []
        for name, score in top_emotions:
            if score > 0.3:
                emotion_parts.append(f"{name} ({score:.0%})")
        
        if emotion_parts:
            emotion_desc = f"Emotions: {', '.join(emotion_parts)}"
        else:
            emotion_desc = "neutral emotional tone"
        
        return f"{query_text} {emotion_desc}"
