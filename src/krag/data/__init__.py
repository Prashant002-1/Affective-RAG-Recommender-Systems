"""
Data Module for Affective-RAG
Provides data loading from GCS and processing pipelines
"""

from .ingestion import (
    IngestionConfig,
    MovieDataLoader,
    UserProfileLoader,
    UserProfile,
    UserManager,
    KnowledgeGraphBuilder,
    DataProcessor,
    generate_node_embeddings_for_kg
)

from .adapters import (
    GCSAdapter,
    DatasetPath,
    DataSchema,
    MOVIE_SCHEMA,
    USER_SCHEMA,
    BUCKET_NAME,
    BASE_PATH,
    get_adapter
)

from .schema import (
    EMOTION_LABELS,
    MoviesVectorReadySchema,
    UserEmotionProfilesSchema,
    UserRatedMovieSchema,
    SCHEMAS,
    get_schema,
    validate_dataframe_columns
)

__all__ = [
    # Ingestion
    'IngestionConfig',
    'MovieDataLoader',
    'UserProfileLoader',
    'UserProfile',
    'UserManager',
    'KnowledgeGraphBuilder',
    'DataProcessor',
    'generate_node_embeddings_for_kg',
    # Adapters
    'GCSAdapter',
    'DatasetPath',
    'DataSchema',
    'MOVIE_SCHEMA',
    'USER_SCHEMA',
    'BUCKET_NAME',
    'BASE_PATH',
    'get_adapter',
    # Schema
    'EMOTION_LABELS',
    'MoviesVectorReadySchema',
    'UserEmotionProfilesSchema',
    'UserRatedMovieSchema',
    'SCHEMAS',
    'get_schema',
    'validate_dataframe_columns'
]
