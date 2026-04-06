"""
Emotion Detection Module for Affective-RAG
Handles emotion profiles and user emotion processing

Emotion Labels (from GCS dataset):
- happiness, sadness, anger, fear, surprise, disgust
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


# Standard emotion labels matching GCS dataset columns
# These correspond to: happiness_score, sadness_score, anger_score, fear_score, surprise_score, disgust_score
EMOTION_LABELS = ['happiness', 'sadness', 'anger', 'fear', 'surprise', 'disgust']


@dataclass
class EmotionProfile:
    """
    Represents emotion scores for content or user state.
    Uses 6 emotions matching the pre-computed GCS data.
    """
    happiness: float = 0.0
    sadness: float = 0.0
    anger: float = 0.0
    fear: float = 0.0
    surprise: float = 0.0
    disgust: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            'happiness': self.happiness,
            'sadness': self.sadness,
            'anger': self.anger,
            'fear': self.fear,
            'surprise': self.surprise,
            'disgust': self.disgust
        }

    def to_vector(self) -> np.ndarray:
        """Convert to numpy vector for similarity calculations"""
        return np.array([
            self.happiness, self.sadness, self.anger,
            self.fear, self.surprise, self.disgust
        ])

    @classmethod
    def from_dict(cls, emotion_dict: Dict[str, float]) -> 'EmotionProfile':
        return cls(
            happiness=emotion_dict.get('happiness', 0.0),
            sadness=emotion_dict.get('sadness', 0.0),
            anger=emotion_dict.get('anger', 0.0),
            fear=emotion_dict.get('fear', 0.0),
            surprise=emotion_dict.get('surprise', 0.0),
            disgust=emotion_dict.get('disgust', 0.0)
        )

    @classmethod
    def from_vector(cls, vector: np.ndarray) -> 'EmotionProfile':
        """Create from numpy vector (ordered: happiness, sadness, anger, fear, surprise, disgust)"""
        if len(vector) != 6:
            raise ValueError(f"Expected 6 emotions, got {len(vector)}")
        return cls(
            happiness=float(vector[0]),
            sadness=float(vector[1]),
            anger=float(vector[2]),
            fear=float(vector[3]),
            surprise=float(vector[4]),
            disgust=float(vector[5])
        )

    def dominant_emotion(self) -> tuple[str, float]:
        """Get the dominant emotion and its score"""
        emotion_dict = self.to_dict()
        dominant = max(emotion_dict.items(), key=lambda x: x[1])
        return dominant

    def normalize(self) -> 'EmotionProfile':
        """Return L2-normalized emotion profile"""
        vec = self.to_vector()
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return EmotionProfile.from_vector(vec)


class UserEmotionProcessor:
    """
    Process user emotion inputs from sliders/interface.
    Convert user emotion preferences to searchable vectors.
    """

    def __init__(self):
        self.emotion_labels = EMOTION_LABELS

    def process_emotion_sliders(self, emotion_sliders: Dict[str, int]) -> EmotionProfile:
        """
        Convert emotion slider values (0-10) to EmotionProfile.

        Args:
            emotion_sliders: Dict with emotion names and values 0-10
                           e.g., {"happiness": 8, "sadness": 2, "fear": 3}

        Returns:
            Normalized EmotionProfile with values 0-1
        """
        normalized_emotions = {}

        for emotion in self.emotion_labels:
            # Also support 'joy' as alias for 'happiness' for user convenience
            slider_value = emotion_sliders.get(emotion, 0)
            if emotion == 'happiness' and slider_value == 0:
                slider_value = emotion_sliders.get('joy', 0)
            # Normalize from 0-10 scale to 0-1 scale
            normalized_emotions[emotion] = slider_value / 10.0

        return EmotionProfile.from_dict(normalized_emotions)

    def calculate_emotion_similarity(
        self,
        emotion1: EmotionProfile,
        emotion2: EmotionProfile
    ) -> float:
        """
        Calculate cosine similarity between two emotion profiles.

        Args:
            emotion1: First emotion profile
            emotion2: Second emotion profile

        Returns:
            Cosine similarity score (0-1)
        """
        vec1 = emotion1.to_vector()
        vec2 = emotion2.to_vector()

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)
        return max(0.0, similarity)
