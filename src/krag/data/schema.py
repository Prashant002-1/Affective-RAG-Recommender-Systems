"""
Data Schema Definitions

This module defines expected column names for the CSV files consumed by the
pipeline. Keep dataset locations/configuration outside the repository.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum


class ColumnFormat(Enum):
    """Column naming format used in CSV files"""
    NEO4J = "neo4j"      # Uses :ID, :START_ID, :END_ID, :TYPE, :LABEL
    STANDARD = "standard"  # Uses userId, movieId, etc.


# =============================================================================
# NEO4J NODE FILES
# =============================================================================

@dataclass
class MoviesNodeSchema:
    """neo4j_nodes/movies.csv"""
    format: ColumnFormat = ColumnFormat.NEO4J
    
    # Columns (in order)
    movie_id: str = "movieId:ID"
    title: str = "title"
    overview: str = "overview"
    genres: str = "genres"
    label: str = ":LABEL"  # Always "Movie"
    
    @classmethod
    def id_column(cls) -> str:
        return "movieId:ID"


@dataclass
class UsersNodeSchema:
    """neo4j_nodes/users.csv"""
    format: ColumnFormat = ColumnFormat.NEO4J
    
    # Columns (in order)
    user_id: str = "userId:ID"
    num_ratings: str = "num_ratings:INT"
    avg_rating: str = "avg_rating:FLOAT"
    emotion_diversity: str = "emotion_diversity:FLOAT"
    dominant_emotion: str = "dominant_emotion"
    label: str = ":LABEL"  # Always "User"
    
    @classmethod
    def id_column(cls) -> str:
        return "userId:ID"


@dataclass
class GenresNodeSchema:
    """neo4j_nodes/genres.csv"""
    format: ColumnFormat = ColumnFormat.NEO4J
    
    name: str = "name:ID"
    label: str = ":LABEL"  # Always "Genre"


@dataclass
class EmotionsNodeSchema:
    """neo4j_nodes/emotions.csv"""
    format: ColumnFormat = ColumnFormat.NEO4J
    
    name: str = "name:ID"
    label: str = ":LABEL"  # Always "Emotion"


# =============================================================================
# NEO4J RELATIONSHIP FILES
# =============================================================================

@dataclass
class UserRatedMovieSchema:
    """neo4j_relationships/user_rated_movie.csv
    
    IMPORTANT: Uses :START_ID/:END_ID, NOT userId/movieId!
    """
    format: ColumnFormat = ColumnFormat.NEO4J
    
    start_id: str = ":START_ID"   # userId
    end_id: str = ":END_ID"       # movieId  
    rating: str = "rating:FLOAT"
    timestamp: str = "timestamp:INT"
    rel_type: str = ":TYPE"       # Always "RATED"
    
    @classmethod
    def user_column(cls) -> str:
        return ":START_ID"
    
    @classmethod
    def movie_column(cls) -> str:
        return ":END_ID"


@dataclass
class UserPrefersEmotionSchema:
    """neo4j_relationships/user_prefers_emotion.csv"""
    format: ColumnFormat = ColumnFormat.NEO4J
    
    start_id: str = ":START_ID"    # userId
    end_id: str = ":END_ID"        # emotion name
    strength: str = "strength:FLOAT"
    rel_type: str = ":TYPE"        # Always "PREFERS"


@dataclass
class MovieBelongsToGenreSchema:
    """neo4j_relationships/movie_belongs_to_genre.csv"""
    format: ColumnFormat = ColumnFormat.NEO4J
    
    start_id: str = ":START_ID"   # movieId
    end_id: str = ":END_ID"       # genre name
    rel_type: str = ":TYPE"       # Always "BELONGS_TO"


@dataclass
class MovieExpressesEmotionSchema:
    """neo4j_relationships/movie_expresses_emotion.csv"""
    format: ColumnFormat = ColumnFormat.NEO4J
    
    start_id: str = ":START_ID"      # movieId
    end_id: str = ":END_ID"          # emotion name
    confidence: str = "confidence:FLOAT"
    intensity: str = "intensity"     # "low", "medium", "high"
    rel_type: str = ":TYPE"          # Always "EXPRESSES"


@dataclass
class MovieSimilarEmotionsSchema:
    """neo4j_relationships/movie_similar_emotions.csv"""
    format: ColumnFormat = ColumnFormat.NEO4J
    
    start_id: str = ":START_ID"      # movieId1
    end_id: str = ":END_ID"          # movieId2
    similarity: str = "similarity:FLOAT"
    rel_type: str = ":TYPE"          # Always "SIMILAR_EMOTIONS"


# =============================================================================
# PROCESSED DATA FILES
# =============================================================================

@dataclass
class MoviesVectorReadySchema:
    """Primary source for movie data with emotions (standard columns)."""
    format: ColumnFormat = ColumnFormat.STANDARD
    
    movie_id: str = "movieId"
    title: str = "title"
    overview: str = "overview"
    genres: str = "genres"
    
    # Emotion scores (z-scored, need sigmoid normalization)
    anger_score: str = "anger_score"
    disgust_score: str = "disgust_score"
    fear_score: str = "fear_score"
    happiness_score: str = "happiness_score"
    sadness_score: str = "sadness_score"
    surprise_score: str = "surprise_score"
    
    # Precomputed emotion vector (JSON array)
    emotion_vector: str = "emotion_vector"
    
    @classmethod
    def id_column(cls) -> str:
        return "movieId"
    
    @classmethod
    def emotion_columns(cls) -> List[str]:
        return [
            "anger_score", "disgust_score", "fear_score",
            "happiness_score", "sadness_score", "surprise_score"
        ]
    
    @classmethod
    def emotion_column_to_label(cls) -> Dict[str, str]:
        """Maps CSV column names to EmotionProfile label names"""
        return {
            "anger_score": "anger",
            "disgust_score": "disgust",
            "fear_score": "fear",
            "happiness_score": "happiness",
            "sadness_score": "sadness",
            "surprise_score": "surprise"
        }


@dataclass
class UserEmotionProfilesSchema:
    """Primary source for user emotion profiles (standard columns)."""
    format: ColumnFormat = ColumnFormat.STANDARD
    
    user_id: str = "userId"
    
    # Frequency columns (how often user watches movies with this emotion)
    anger_freq: str = "anger_freq"
    disgust_freq: str = "disgust_freq"
    fear_freq: str = "fear_freq"
    happiness_freq: str = "happiness_freq"
    sadness_freq: str = "sadness_freq"
    surprise_freq: str = "surprise_freq"
    
    # Average columns (average emotion intensity in watched movies)
    anger_avg: str = "anger_avg"
    disgust_avg: str = "disgust_avg"
    fear_avg: str = "fear_avg"
    happiness_avg: str = "happiness_avg"
    sadness_avg: str = "sadness_avg"
    surprise_avg: str = "surprise_avg"
    
    # User metadata
    avg_rating: str = "avg_rating"
    rating_std: str = "rating_std"
    num_ratings: str = "num_ratings"
    num_unique_movies: str = "num_unique_movies"
    dominant_emotion: str = "dominant_emotion"
    emotion_diversity: str = "emotion_diversity"
    
    # Adjusted scores
    anger_adj: str = "anger_adj"
    disgust_adj: str = "disgust_adj"
    fear_adj: str = "fear_adj"
    happiness_adj: str = "happiness_adj"
    sadness_adj: str = "sadness_adj"
    surprise_adj: str = "surprise_adj"
    dominant_emotion_adj: str = "dominant_emotion_adj"
    emotion_diversity_adj: str = "emotion_diversity_adj"
    
    @classmethod
    def id_column(cls) -> str:
        return "userId"
    
    @classmethod
    def freq_columns(cls) -> List[str]:
        """Frequency columns - good for user emotion preferences"""
        return [
            "anger_freq", "disgust_freq", "fear_freq",
            "happiness_freq", "sadness_freq", "surprise_freq"
        ]
    
    @classmethod
    def freq_column_to_label(cls) -> Dict[str, str]:
        """Maps freq column to emotion label"""
        return {
            "anger_freq": "anger",
            "disgust_freq": "disgust",
            "fear_freq": "fear",
            "happiness_freq": "happiness",
            "sadness_freq": "sadness",
            "surprise_freq": "surprise"
        }


@dataclass
class UserEmotionSensitivitiesSchema:
    """User emotion sensitivity table (standard columns)."""
    format: ColumnFormat = ColumnFormat.STANDARD
    
    # Columns in actual order
    anger_sensitivity: str = "anger_sensitivity"
    disgust_sensitivity: str = "disgust_sensitivity"
    fear_sensitivity: str = "fear_sensitivity"
    happiness_sensitivity: str = "happiness_sensitivity"
    sadness_sensitivity: str = "sadness_sensitivity"
    surprise_sensitivity: str = "surprise_sensitivity"
    user_id: str = "userId"  # LAST column!
    
    @classmethod
    def id_column(cls) -> str:
        return "userId"
    
    @classmethod
    def sensitivity_columns(cls) -> List[str]:
        return [
            "anger_sensitivity", "disgust_sensitivity", "fear_sensitivity",
            "happiness_sensitivity", "sadness_sensitivity", "surprise_sensitivity"
        ]


@dataclass
class UserComparisonPairsSchema:
    """User comparison pairs (for experiments)."""
    format: ColumnFormat = ColumnFormat.STANDARD
    
    user_a: str = "user_a"
    user_b: str = "user_b"
    cluster_a: str = "cluster_a"
    cluster_b: str = "cluster_b"


# =============================================================================
# ROOT LEVEL FILES
# =============================================================================

@dataclass 
class RatingsSchema:
    """Ratings table (standard columns)."""
    format: ColumnFormat = ColumnFormat.STANDARD
    
    user_id: str = "userId"
    movie_id: str = "movieId"
    rating: str = "rating"
    timestamp: str = "timestamp"


# =============================================================================
# SCHEMA REGISTRY
# =============================================================================

SCHEMAS = {
    # Neo4j nodes
    "neo4j_nodes/movies": MoviesNodeSchema,
    "neo4j_nodes/users": UsersNodeSchema,
    "neo4j_nodes/genres": GenresNodeSchema,
    "neo4j_nodes/emotions": EmotionsNodeSchema,
    
    # Neo4j relationships
    "neo4j_relationships/user_rated_movie": UserRatedMovieSchema,
    "neo4j_relationships/user_prefers_emotion": UserPrefersEmotionSchema,
    "neo4j_relationships/movie_belongs_to_genre": MovieBelongsToGenreSchema,
    "neo4j_relationships/movie_expresses_emotion": MovieExpressesEmotionSchema,
    "neo4j_relationships/movie_similar_emotions": MovieSimilarEmotionsSchema,
    
    # Processed files
    "movies_vector_ready": MoviesVectorReadySchema,
    "user_emotion_profiles": UserEmotionProfilesSchema,
    "user_emotion_sensitivities": UserEmotionSensitivitiesSchema,
    "user_comparison_pairs": UserComparisonPairsSchema,
    
    # Original MovieLens
    "ratings": RatingsSchema,
}


# =============================================================================
# EMOTION LABELS (canonical list)
# =============================================================================

EMOTION_LABELS = ["happiness", "sadness", "anger", "fear", "surprise", "disgust"]
"""
Canonical emotion labels used throughout the system.
Based on Ekman's 6 basic emotions.
All emotion-related code should use these exact labels.
"""


def get_schema(name: str):
    """Get schema by name"""
    return SCHEMAS.get(name)


def validate_dataframe_columns(df, schema_class) -> List[str]:
    """
    Validate that a DataFrame has the expected columns from a schema.
    
    Returns:
        List of missing columns (empty if all present)
    """
    expected = []
    for field_name, field_value in schema_class.__dataclass_fields__.items():
        if field_name != 'format':
            expected.append(field_value.default)
    
    missing = [col for col in expected if col not in df.columns]
    return missing

