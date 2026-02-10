"""
Data Ingestion Pipeline for Affective-RAG
Loads and processes content data from Google Cloud Storage
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Generator, Set, Any
from dataclasses import dataclass, asdict, field
from pathlib import Path
import json

from .adapters import GCSAdapter, DatasetPath, get_adapter
from ..core.embeddings import ContentItem
from ..core.emotion_detection import EmotionProfile, EMOTION_LABELS
from ..core.knowledge_graph import ContentKnowledgeGraph, KnowledgeTriple


def generate_node_embeddings_for_kg(
    knowledge_graph: ContentKnowledgeGraph,
    content_embedder
) -> Dict[str, np.ndarray]:
    """
    Generate PLM embeddings for all nodes in a knowledge graph.
    
    Per K-RagRec paper Section 3.2: z_n = PLM(x_n)
    
    This is a standalone function that can be called from system.py
    after the knowledge graph is built.
    
    Args:
        knowledge_graph: The ContentKnowledgeGraph instance
        content_embedder: ContentEmbedder instance for generating embeddings
        
    Returns:
        Dict mapping node_id -> embedding (768-dim)
    """
    print("Generating PLM embeddings for knowledge graph nodes...", flush=True)
    
    # Emotion descriptions for better embeddings
    emotion_descriptions = {
        'happiness': 'a positive emotional state of joy and contentment',
        'sadness': 'a negative emotional state of sorrow and grief',
        'anger': 'an intense emotional state of displeasure and hostility',
        'fear': 'an emotional response to perceived danger or threat',
        'surprise': 'a brief emotional state caused by unexpected events',
        'disgust': 'an emotional response to something offensive or unpleasant'
    }
    
    # Collect text representations for each node
    node_texts = {}
    
    for node_id, data in knowledge_graph.graph.nodes(data=True):
        node_type = data.get('node_type', 'unknown')
        text = None
        
        if node_type == 'content':
            title = data.get('title', '')
            year = data.get('year', '')
            text = f"{title} ({year})" if year else title
            
        elif node_type == 'genre':
            name = data.get('name', node_id.replace('genre_', '').replace('_', ' '))
            text = f"Genre: {name} - A category or type of movie"
            
        elif node_type == 'emotion':
            name = data.get('name', node_id.replace('emotion_', ''))
            desc = emotion_descriptions.get(name.lower(), 'an emotional state')
            text = f"Emotion: {name} - {desc}"
            
        if text:
            node_texts[node_id] = text
    
    if not node_texts:
        print("  Warning: No nodes to embed", flush=True)
        return {}
    
    # Batch embed all node texts
    node_ids = list(node_texts.keys())
    texts = [node_texts[nid] for nid in node_ids]
    
    print(f"  Embedding {len(texts)} nodes...", flush=True)
    embeddings = content_embedder.embed_batch(texts, batch_size=128)
    
    # Create mapping
    node_embeddings = {
        node_id: embeddings[i]
        for i, node_id in enumerate(node_ids)
    }
    
    print(f"  Generated embeddings for {len(node_embeddings)} nodes (768-dim)", flush=True)
    return node_embeddings


@dataclass
class IngestionConfig:
    """Configuration for data ingestion pipeline"""
    bucket_name: str = "your-gcs-bucket"
    base_path: str = "Dataset"
    
    # Processing options
    chunk_size: int = 5000
    max_items: Optional[int] = None  # None = load all
    
    # Feature flags
    load_precomputed_emotions: bool = True  # Use pre-computed emotions from GCS
    build_knowledge_graph: bool = True
    
    # Output paths (local)
    cache_dir: str = "./data/cache"
    

class MovieDataLoader:
    """
    Loads movie data from GCS with pre-computed emotions.
    """

    def __init__(self, adapter: GCSAdapter):
        self.adapter = adapter

    def load_movies(
        self,
        max_items: Optional[int] = None,
        chunk_size: int = 5000
    ) -> Generator[List[ContentItem], None, None]:
        """
        Load movies from GCS as ContentItem objects.
        - Pre-computed emotion scores (joy, sadness, anger, fear, surprise, disgust)

        Args:
            max_items: Maximum items to load (None = all)
            chunk_size: Rows per chunk
            
        Yields:
            Lists of ContentItem objects
        """
        print(f"Loading movies from GCS: {DatasetPath.MOVIES_VECTOR_READY.value}")
        
        total_loaded = 0
        
        for chunk in self.adapter.load_csv(
            DatasetPath.MOVIES_VECTOR_READY,
            chunk_size=chunk_size
        ):
            content_items = []
            
            for _, row in chunk.iterrows():
                if max_items and total_loaded >= max_items:
                    if content_items:
                        yield content_items
                    return
                
                item = self._row_to_content_item(row)
                if item:
                    content_items.append(item)
                    total_loaded += 1
            
            if content_items:
                print(f"  Loaded {total_loaded} movies...")
                yield content_items
        
        print(f"Total movies loaded: {total_loaded}")

    def _row_to_content_item(self, row: pd.Series) -> Optional[ContentItem]:
        """Convert a DataFrame row to ContentItem"""
        try:
            movie_id = str(row.get('movieId', ''))
            title = str(row.get('title', ''))
            
            if not movie_id or not title:
                return None
            
            # Description/overview
            overview = str(row.get('overview', '') or '')
            
            # Genres (pipe-separated in MovieLens format)
            genres = []
            genre_val = row.get('genres', '')
            if pd.notna(genre_val) and genre_val:
                genres = [g.strip() for g in str(genre_val).split('|') if g.strip()]
            
            # Year (extract from title or use release_year column)
            year = None
            if 'release_year' in row and pd.notna(row['release_year']):
                try:
                    year = int(row['release_year'])
                except (ValueError, TypeError):
                    pass
            
            # Pre-computed emotions
            emotions = self._extract_emotions(row)
            
            # Additional metadata
            metadata = {}
            for col in ['imdbId', 'tmdbId', 'vote_average', 'vote_count', 'popularity']:
                if col in row and pd.notna(row[col]):
                    metadata[col] = row[col]
            
            return ContentItem(
                id=movie_id,
                title=title,
                description=overview,
                genres=genres,
                year=year,
                emotions=emotions,
                metadata=metadata
            )
            
        except Exception as e:
            print(f"Error parsing row: {e}")
            return None

    def _extract_emotions(self, row: pd.Series) -> Optional[EmotionProfile]:
        """
        Extract pre-computed emotion scores from row.
        
        GCS columns: happiness_score, sadness_score, anger_score, fear_score, surprise_score, disgust_score
        Values are z-scored (mean~0, range~[-2, 2]).
        We convert to [0, 1] using sigmoid: 1 / (1 + exp(-z))
        """
        emotion_dict = {}
        
        # Direct mapping: emotion label -> GCS column name
        # EMOTION_LABELS = ['happiness', 'sadness', 'anger', 'fear', 'surprise', 'disgust']
        for emotion in EMOTION_LABELS:
            col_name = f'{emotion}_score'
            if col_name in row and pd.notna(row[col_name]):
                try:
                    z_score = float(row[col_name])
                    # Convert z-score to [0, 1] using sigmoid
                    normalized = 1.0 / (1.0 + np.exp(-z_score))
                    emotion_dict[emotion] = normalized
                except (ValueError, TypeError):
                    pass
        
        if emotion_dict:
            return EmotionProfile.from_dict(emotion_dict)
        return None

    def load_all_movies(self, max_items: Optional[int] = None) -> List[ContentItem]:
        """Load all movies into memory (convenience method)"""
        all_items = []
        for batch in self.load_movies(max_items=max_items):
            all_items.extend(batch)
        return all_items


class UserProfileLoader:
    """Loads user emotion profiles from GCS"""

    def __init__(self, adapter: GCSAdapter):
        self.adapter = adapter

    def load_user_profiles(self) -> pd.DataFrame:
        """Load user emotion profiles"""
        print(f"Loading user profiles from GCS: {DatasetPath.USER_EMOTION_PROFILES.value}")
        return self.adapter.load_csv(DatasetPath.USER_EMOTION_PROFILES)

    def load_user_sensitivities(self) -> pd.DataFrame:
        """Load user emotion sensitivities"""
        print(f"Loading user sensitivities from GCS: {DatasetPath.USER_EMOTION_SENSITIVITIES.value}")
        return self.adapter.load_csv(DatasetPath.USER_EMOTION_SENSITIVITIES)

    def get_user_emotion_profile(self, user_id: str, profiles_df: pd.DataFrame) -> Optional[EmotionProfile]:
        """
        Get emotion profile for a specific user.
        
        Uses user_emotion_profiles.csv which has columns:
        - {emotion}_freq: Frequency of watching movies with this emotion
        - {emotion}_avg: Average emotion intensity in watched movies
        - {emotion}_adj: Adjusted scores
        
        We use _freq columns as they represent user preference strength.
        """
        # Try integer lookup first, then string
        try:
            user_row = profiles_df[profiles_df['userId'] == int(user_id)]
        except (ValueError, TypeError):
            user_row = pd.DataFrame()
            
        if user_row.empty:
            user_row = profiles_df[profiles_df['userId'] == str(user_id)]
        
        if user_row.empty:
            return None
        
        row = user_row.iloc[0]
        emotion_dict = {}
        
        for emotion in EMOTION_LABELS:
            # Priority order based on actual user_emotion_profiles.csv columns:
            # 1. _freq columns (user preference frequency)
            # 2. _avg columns (average intensity)
            # 3. _adj columns (adjusted scores)
            col_candidates = [
                f'{emotion}_freq',   # e.g., happiness_freq
                f'{emotion}_avg',    # e.g., happiness_avg
                f'{emotion}_adj',    # e.g., happiness_adj
            ]
            
            for col in col_candidates:
                if col in row and pd.notna(row[col]):
                    try:
                        value = float(row[col])
                        # _freq values are already 0-1 range
                        # _avg and _adj might be z-scored, normalize if needed
                        if '_avg' in col or '_adj' in col:
                            value = 1 / (1 + np.exp(-value))  # Sigmoid for z-scores
                        emotion_dict[emotion] = value
                        break
                    except (ValueError, TypeError):
                        pass
        
        return EmotionProfile.from_dict(emotion_dict) if emotion_dict else None


@dataclass
class UserProfile:
    """
    Complete user profile for personalized recommendations.
    
    Loaded from GCS user data files:
    - user_emotion_profiles.csv: Pre-computed emotion preferences
    - user_emotion_sensitivities.csv: Emotion sensitivity weights
    - neo4j_relationships/user_rated_movie.csv: Movies user has watched
    
    The watched_movies set is crucial for user-scoped retrieval:
    - Each user only sees recommendations from movies they have rated
    - This enables fair evaluation and personalized experiments
    """
    user_id: str
    emotions: EmotionProfile
    watched_movies: Set[str] = field(default_factory=set)  # Movie IDs user has rated
    sensitivities: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'user_id': self.user_id,
            'emotions': self.emotions.to_dict(),
            'watched_movies_count': len(self.watched_movies),
            'sensitivities': self.sensitivities,
            'metadata': self.metadata
        }
    
    @property
    def movie_count(self) -> int:
        """Number of movies this user has watched/rated"""
        return len(self.watched_movies)


class UserManager:
    """
    Manages user profiles for personalized recommendations.
    
    Provides:
    - User profile loading from GCS
    - User emotion profile lookup
    - User-movie rating mapping (for user-scoped retrieval)
    - Caching for performance
    
    Usage:
        user_mgr = UserManager(adapter)
        user_mgr.load_users(load_ratings=True)  # Enable user-scoped retrieval
        profile = user_mgr.get_user("12345")
        emotions = profile.emotions
        allowed_movies = profile.watched_movies  # Movies this user can access
    """
    
    def __init__(self, adapter: GCSAdapter):
        self.adapter = adapter
        self.loader = UserProfileLoader(adapter)
        
        # Cached dataframes
        self._user_profiles_df: Optional[pd.DataFrame] = None
        self._user_sensitivities_df: Optional[pd.DataFrame] = None
        
        # User-movie ratings: user_id -> set of movie_ids
        self._user_movies: Dict[str, Set[str]] = {}
        self._ratings_loaded = False
        
        # User cache
        self._user_cache: Dict[str, UserProfile] = {}
        self._loaded = False
    
    def load_users(
        self, 
        load_sensitivities: bool = False,
        load_ratings: bool = True
    ) -> int:
        """
        Load user data from GCS.
        
        Args:
            load_sensitivities: Whether to load sensitivity data (larger file)
            load_ratings: Whether to load user-movie ratings for scoped retrieval
            
        Returns:
            Number of users loaded
        """
        print("Loading user profiles from GCS...", flush=True)
        
        try:
            self._user_profiles_df = self.loader.load_user_profiles()
            print(f"  Loaded {len(self._user_profiles_df)} user profiles", flush=True)
        except Exception as e:
            print(f"  Warning: Could not load user profiles: {e}", flush=True)
            self._user_profiles_df = pd.DataFrame()
        
        if load_sensitivities:
            try:
                self._user_sensitivities_df = self.loader.load_user_sensitivities()
                print(f"  Loaded user sensitivities", flush=True)
            except Exception as e:
                print(f"  Warning: Could not load sensitivities: {e}", flush=True)
        
        # Load user-movie ratings for scoped retrieval
        if load_ratings:
            self._load_user_ratings()
        
        self._loaded = True
        return len(self._user_profiles_df) if self._user_profiles_df is not None else 0
    
    def _load_user_ratings(self) -> None:
        """
        Load user-movie ratings to enable user-scoped retrieval.
        
        Strategy:
        1. Try to load precomputed mapping from GCS (fast, ~10MB)
        2. Fall back to loading full ratings file (slow, ~700MB)
        """
        # Option 1: Try precomputed mapping first (fast!)
        try:
            if self.adapter.exists(DatasetPath.USER_MOVIE_MAPPING):
                print("  Loading precomputed user-movie mapping from GCS...", flush=True)
                mapping_data = self.adapter.load_pickle(DatasetPath.USER_MOVIE_MAPPING)
                self._user_movies = mapping_data
                print(f"  ✓ Loaded mapping for {len(self._user_movies):,} users", flush=True)
                self._ratings_loaded = True
                self._user_cache.clear()
                return
        except Exception as e:
            print(f"  Precomputed mapping not available: {e}", flush=True)
        
        # Option 2: Load full ratings file (slow)
        print("  Loading user-movie ratings (this may take 1-2 minutes)...", flush=True)
        print("  TIP: Generate precomputed mapping with: python -m krag.data.precompute_mappings", flush=True)
        
        try:
            # Only load the columns we need to save memory
            # Note: Neo4j export uses :START_ID and :END_ID instead of userId and movieId
            print("    Downloading from GCS...", flush=True)
            ratings_df = self.adapter.load_csv(
                DatasetPath.REL_USER_RATING,
                usecols=[':START_ID', ':END_ID'],  # Neo4j format column names
                dtype={':START_ID': str, ':END_ID': str}  # Read as strings directly
            )
            # Rename columns to standard names
            ratings_df = ratings_df.rename(columns={':START_ID': 'userId', ':END_ID': 'movieId'})
            print(f"    Downloaded {len(ratings_df):,} ratings", flush=True)
            
            # Use pandas groupby for fast aggregation (much faster than iterrows)
            print("    Building user-movie mapping...", flush=True)
            grouped = ratings_df.groupby('userId')['movieId'].apply(set).to_dict()
            self._user_movies = grouped
            
            total_ratings = len(ratings_df)
            print(f"  ✓ Loaded {total_ratings:,} ratings for {len(self._user_movies):,} users", flush=True)
            self._ratings_loaded = True
            
            # IMPORTANT: Clear user cache so profiles get rebuilt with watched_movies
            self._user_cache.clear()
            
            # Free memory
            del ratings_df
            
        except Exception as e:
            print(f"  Warning: Could not load user ratings: {e}", flush=True)
            print("  User-scoped retrieval will be disabled.", flush=True)
            self._ratings_loaded = False
    
    def get_user_watched_movies(self, user_id: str) -> Set[str]:
        """
        Get watched movies for a user directly (bypasses profile caching).
        Use this when you need just the movie IDs without full profile.
        """
        if not self._ratings_loaded:
            return set()
        return self._user_movies.get(user_id, set())
    
    def get_user(self, user_id: str) -> Optional[UserProfile]:
        """
        Get a user profile by ID.
        
        Args:
            user_id: The user ID to look up
            
        Returns:
            UserProfile or None if not found
        """
        if not self._loaded:
            raise RuntimeError("Users not loaded. Call load_users() first.")
        
        # Check cache
        if user_id in self._user_cache:
            return self._user_cache[user_id]
        
        # Build profile
        profile = self._build_user_profile(user_id)
        if profile:
            self._user_cache[user_id] = profile
        
        return profile
    
    def _build_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Build a UserProfile from loaded data"""
        if self._user_profiles_df is None or self._user_profiles_df.empty:
            return None
        
        # Get emotion profile
        emotions = self.loader.get_user_emotion_profile(user_id, self._user_profiles_df)
        if not emotions:
            return None
        
        # Get watched movies (for user-scoped retrieval)
        watched_movies = self._user_movies.get(user_id, set())
        
        # Get sensitivities
        sensitivities = None
        if self._user_sensitivities_df is not None:
            sensitivities = self._get_user_sensitivities(user_id)
        
        return UserProfile(
            user_id=user_id,
            emotions=emotions,
            watched_movies=watched_movies,
            sensitivities=sensitivities
        )
    
    def _get_user_sensitivities(self, user_id: str) -> Optional[Dict[str, float]]:
        """Get emotion sensitivities for a user"""
        if self._user_sensitivities_df is None:
            return None
        
        # Try integer lookup first, then string
        try:
            user_row = self._user_sensitivities_df[
                self._user_sensitivities_df['userId'] == int(user_id)
            ]
        except (ValueError, TypeError):
            user_row = pd.DataFrame()
            
        if user_row.empty:
            user_row = self._user_sensitivities_df[
                self._user_sensitivities_df['userId'] == str(user_id)
            ]
        
        if user_row.empty:
            return None
        
        row = user_row.iloc[0]
        sensitivities = {}
        for emotion in EMOTION_LABELS:
            col_name = f'{emotion}_sensitivity'
            if col_name in row and pd.notna(row[col_name]):
                sensitivities[emotion] = float(row[col_name])
        
        return sensitivities if sensitivities else None
    
    def list_users(self, limit: int = 100) -> List[str]:
        """List available user IDs"""
        if self._user_profiles_df is None or self._user_profiles_df.empty:
            return []
        return [str(uid) for uid in self._user_profiles_df['userId'].head(limit).tolist()]
    
    def get_user_movies(self, user_id: str) -> Set[str]:
        """
        Get the set of movie IDs that a user has rated/watched.
        
        Args:
            user_id: The user ID
            
        Returns:
            Set of movie IDs (empty if user not found or ratings not loaded)
        """
        return self._user_movies.get(user_id, set())
    
    def get_users_with_min_ratings(self, min_ratings: int = 50) -> List[str]:
        """
        Get users who have rated at least min_ratings movies.
        Useful for selecting users with sufficient data for experiments.
        
        Args:
            min_ratings: Minimum number of ratings required
            
        Returns:
            List of user IDs
        """
        return [
            uid for uid, movies in self._user_movies.items()
            if len(movies) >= min_ratings
        ]
    
    @property
    def user_count(self) -> int:
        """Number of users loaded"""
        return len(self._user_profiles_df) if self._user_profiles_df is not None else 0
    
    @property
    def ratings_loaded(self) -> bool:
        """Whether user-movie ratings have been loaded"""
        return self._ratings_loaded
    
    @property
    def is_loaded(self) -> bool:
        """Whether user data has been loaded"""
        return self._loaded


class KnowledgeGraphBuilder:
    """Build knowledge graph from GCS data"""

    def __init__(self, adapter: GCSAdapter):
        self.adapter = adapter
        self.kg = ContentKnowledgeGraph()

    def build_from_neo4j_exports(self) -> ContentKnowledgeGraph:
        """
        Build knowledge graph from Neo4j export CSVs in GCS.
        
        Uses:
        - neo4j_nodes/: movies.csv, emotions.csv, genres.csv, users.csv
        - neo4j_relationships/: movie_belongs_to_genre.csv, movie_expresses_emotion.csv, etc.
        """
        print("Building knowledge graph from Neo4j exports...")
        
        # Load and add nodes
        self._load_emotion_nodes()
        self._load_genre_nodes()
        self._load_movie_nodes()
        
        # Load and add relationships
        self._load_movie_genre_edges()
        self._load_movie_emotion_edges()
        self._load_movie_similarity_edges()
        
        print(f"Knowledge graph built: {len(self.kg.graph.nodes())} nodes, {len(self.kg.graph.edges())} edges")
        return self.kg

    def _load_emotion_nodes(self):
        """Load emotion nodes"""
        try:
            df = self.adapter.load_neo4j_nodes('emotions')
            for _, row in df.iterrows():
                emotion_name = row.get('name:ID', row.get('name', row.get('emotionId', '')))
                # Normalize to lowercase to match edge creation in _load_movie_emotion_edges
                node_id = f"emotion_{str(emotion_name).lower()}"
                self.kg.add_node(node_id, 'emotion', {
                    'name': emotion_name  # Preserve original casing in metadata
                })
            print(f"  Loaded {len(df)} emotion nodes")
        except Exception as e:
            print(f"  Warning: Could not load emotion nodes: {e}")

    def _load_genre_nodes(self):
        """Load genre nodes"""
        try:
            df = self.adapter.load_neo4j_nodes('genres')
            for _, row in df.iterrows():
                genre_name = row.get('name:ID', row.get('name', row.get('genre', '')))
                node_id = f"genre_{genre_name.lower().replace(' ', '_')}"
                self.kg.add_node(node_id, 'genre', {'name': genre_name})
            print(f"  Loaded {len(df)} genre nodes")
        except Exception as e:
            print(f"  Warning: Could not load genre nodes: {e}")

    def _load_movie_nodes(self):
        """Load movie nodes"""
        try:
            df = self.adapter.load_neo4j_nodes('movies')
            for _, row in df.iterrows():
                movie_id = str(row.get('movieId:ID', row.get('movieId', '')))
                self.kg.add_node(movie_id, 'content', {
                    'title': row.get('title', ''),
                    'year': row.get('year', None)
                })
            print(f"  Loaded {len(df)} movie nodes")
        except Exception as e:
            print(f"  Warning: Could not load movie nodes: {e}")

    def _load_movie_genre_edges(self):
        """Load movie-genre relationships"""
        try:
            df = self.adapter.load_neo4j_relationships('movie_genre')
            for _, row in df.iterrows():
                movie_id = str(row.get('movieId', row.get(':START_ID', '')))
                genre = row.get('genre', row.get(':END_ID', ''))
                genre_id = f"genre_{str(genre).lower().replace(' ', '_')}"
                
                triple = KnowledgeTriple(
                    head=movie_id,
                    relation='belongs_to_genre',
                    tail=genre_id,
                    weight=1.0
                )
                self.kg.add_triple(triple)
            print(f"  Loaded {len(df)} movie-genre edges")
        except Exception as e:
            print(f"  Warning: Could not load movie-genre edges: {e}")

    def _load_movie_emotion_edges(self):
        """Load movie-emotion relationships with intensity weights"""
        try:
            df = self.adapter.load_neo4j_relationships('movie_emotion')
            for _, row in df.iterrows():
                movie_id = str(row.get('movieId', row.get(':START_ID', '')))
                emotion = row.get('emotion', row.get(':END_ID', ''))
                emotion_id = f"emotion_{str(emotion).lower()}"
                # Use confidence score as weight (normalize to 0-1 range)
                raw_weight = float(row.get('confidence:FLOAT', row.get('weight', row.get('score', 1.0))))
                weight = min(1.0, raw_weight)  # Cap at 1.0 since some values exceed 1
                
                triple = KnowledgeTriple(
                    head=movie_id,
                    relation='evokes',
                    tail=emotion_id,
                    weight=weight
                )
                self.kg.add_triple(triple)
            print(f"  Loaded {len(df)} movie-emotion edges")
        except Exception as e:
            print(f"  Warning: Could not load movie-emotion edges: {e}")

    def _load_movie_similarity_edges(self):
        """Load movie-movie similarity relationships"""
        try:
            df = self.adapter.load_neo4j_relationships('movie_similar')
            for _, row in df.iterrows():
                movie1_id = str(row.get('movieId1', row.get(':START_ID', '')))
                movie2_id = str(row.get('movieId2', row.get(':END_ID', '')))
                weight = float(row.get('similarity', row.get('weight', 0.5)))
                
                if weight > 0.3:  # Only significant similarities
                    triple = KnowledgeTriple(
                        head=movie1_id,
                        relation='similar_to',
                        tail=movie2_id,
                        weight=weight
                    )
                    self.kg.add_triple(triple)
            print(f"  Loaded movie similarity edges")
        except Exception as e:
            print(f"  Warning: Could not load movie similarity edges: {e}")

    def generate_node_embeddings(self, content_embedder) -> Dict[str, np.ndarray]:
        """
        Generate PLM embeddings for all nodes in the knowledge graph.
        
        Per K-RagRec paper Section 3.2: z_n = PLM(x_n)

        Args:
            content_embedder: ContentEmbedder instance for generating embeddings

        Returns:
            Dict mapping node_id -> embedding (768-dim from PLM, will be projected to 1024 by GNN)
        """
        print("Generating node embeddings for knowledge graph...")
        
        # Collect text representations for each node
        node_texts = {}
        
        for node_id, data in self.kg.graph.nodes(data=True):
            node_type = data.get('node_type', 'unknown')
            text = self._get_node_text(node_id, node_type, data)
            if text:
                node_texts[node_id] = text
        
        if not node_texts:
            print("  Warning: No nodes to embed")
            return {}
        
        # Batch embed all node texts
        node_ids = list(node_texts.keys())
        texts = [node_texts[nid] for nid in node_ids]
        
        print(f"  Embedding {len(texts)} nodes...")
        embeddings = content_embedder.embed_batch(texts, batch_size=128)
        
        # Create mapping
        node_embeddings = {
            node_id: embeddings[i]
            for i, node_id in enumerate(node_ids)
        }
        
        print(f"  Generated embeddings for {len(node_embeddings)} nodes")
        return node_embeddings

    def _get_node_text(self, node_id: str, node_type: str, data: dict) -> str:
        """Generate text representation for a node based on its type."""
        
        if node_type == 'content':
            # Movie: "Title (Year) - Description"
            title = data.get('title', '')
            year = data.get('year', '')
            desc = data.get('description', '')[:200] if data.get('description') else ''
            
            if year:
                text = f"{title} ({year})"
            else:
                text = title
            if desc:
                text += f" - {desc}"
            return text
            
        elif node_type == 'genre':
            # Genre: "Genre: Comedy - A genre of film"
            name = data.get('name', node_id.replace('genre_', '').replace('_', ' '))
            return f"Genre: {name} - A category or type of movie"
            
        elif node_type == 'emotion':
            # Emotion: "Emotion: Happiness - A positive emotional state"
            name = data.get('name', node_id.replace('emotion_', ''))
            emotion_descriptions = {
                'happiness': 'a positive emotional state of joy and contentment',
                'sadness': 'a negative emotional state of sorrow and grief',
                'anger': 'an intense emotional state of displeasure and hostility',
                'fear': 'an emotional response to perceived danger or threat',
                'surprise': 'a brief emotional state caused by unexpected events',
                'disgust': 'an emotional response to something offensive or unpleasant'
            }
            desc = emotion_descriptions.get(name.lower(), 'an emotional state')
            return f"Emotion: {name} - {desc}"
            
        elif node_type == 'user':
            # Users are typically not embedded as they don't have text
            return None
            
        else:
            # Unknown type: just use the node ID
            return str(node_id)

    def build_from_content_items(self, content_items: List[ContentItem]) -> ContentKnowledgeGraph:
        """
        Build knowledge graph from ContentItems.
        Fallback method when Neo4j exports aren't available.
        """
        print(f"Building knowledge graph from {len(content_items)} content items...")

        for item in content_items:
            # Add content node
            self.kg.add_node(item.id, 'content', {
                'title': item.title,
                'description': item.description[:500] if item.description else '',
                'year': item.year
            })

            # Add genre relationships
            for genre in item.genres:
                genre_id = f"genre_{genre.lower().replace(' ', '_')}"
                self.kg.add_node(genre_id, 'genre', {'name': genre})

                triple = KnowledgeTriple(
                    head=item.id,
                    relation='belongs_to_genre',
                    tail=genre_id,
                    weight=1.0
                )
                self.kg.add_triple(triple)

            # Add emotion relationships
            if item.emotions:
                emotions_dict = item.emotions.to_dict()
                for emotion_name, score in emotions_dict.items():
                    if score > 0.3:  # Only significant emotions
                        # Normalize to lowercase for consistency with Neo4j-based graph building
                        emotion_id = f"emotion_{str(emotion_name).lower()}"
                        self.kg.add_node(emotion_id, 'emotion', {'name': emotion_name})

                        triple = KnowledgeTriple(
                            head=item.id,
                            relation='evokes',
                            tail=emotion_id,
                            weight=score
                        )
                        self.kg.add_triple(triple)

        # Add content similarity edges
        self._add_similarity_edges(content_items)

        print(f"Knowledge graph: {len(self.kg.graph.nodes())} nodes, {len(self.kg.graph.edges())} edges")
        return self.kg

    def _add_similarity_edges(self, content_items: List[ContentItem], max_items: int = 10000):
        """Add similarity edges based on shared attributes"""
        # Only process first max_items to avoid O(n²) explosion
        items = content_items[:max_items]
        
        for i, item1 in enumerate(items):
            for item2 in items[i+1:]:
                similarity = self._calculate_similarity(item1, item2)
                if similarity > 0.5:
                    triple = KnowledgeTriple(
                        head=item1.id,
                        relation='similar_to',
                        tail=item2.id,
                        weight=similarity
                    )
                    self.kg.add_triple(triple)

    def _calculate_similarity(self, item1: ContentItem, item2: ContentItem) -> float:
        """Calculate similarity between content items"""
        similarity = 0.0

        # Genre overlap (Jaccard)
        if item1.genres and item2.genres:
            intersection = set(item1.genres) & set(item2.genres)
            union = set(item1.genres) | set(item2.genres)
            if union:
                similarity += 0.4 * len(intersection) / len(union)

        # Year proximity
        if item1.year and item2.year:
            year_diff = abs(item1.year - item2.year)
            similarity += 0.2 * max(0, 1 - year_diff / 20.0)

        # Emotion similarity (cosine)
        if item1.emotions and item2.emotions:
            vec1 = item1.emotions.to_vector()
            vec2 = item2.emotions.to_vector()
            dot = np.dot(vec1, vec2)
            norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
            if norm1 > 0 and norm2 > 0:
                similarity += 0.4 * max(0, dot / (norm1 * norm2))

        return min(1.0, similarity)


class DataProcessor:
    """
    Main data processing coordinator for Affective-RAG.
    Orchestrates loading from GCS and building knowledge structures.
    """

    def __init__(self, config: Optional[IngestionConfig] = None):
        self.config = config or IngestionConfig()
        self.adapter = get_adapter(
            bucket_name=self.config.bucket_name,
            base_path=self.config.base_path
        )
        self.movie_loader = MovieDataLoader(self.adapter)
        self.user_loader = UserProfileLoader(self.adapter)
        self.kg_builder = KnowledgeGraphBuilder(self.adapter)
        
        # Create cache directory
        Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)

    def process(self) -> Tuple[List[ContentItem], ContentKnowledgeGraph]:
        """
        Full data processing pipeline.

        Returns:
            Tuple of (content_items, knowledge_graph)
        """
        print("=" * 60)
        print("Starting Affective-RAG Data Ingestion Pipeline")
        print("=" * 60)
        print(f"GCS Bucket: gs://{self.config.bucket_name}/{self.config.base_path}/")
        print()

        # Step 1: Load movies
        print("[1/3] Loading movie data...")
        content_items = self.movie_loader.load_all_movies(
            max_items=self.config.max_items
        )
        print(f"      Loaded {len(content_items)} movies with pre-computed emotions")
        print()

        # Step 2: Build knowledge graph
        print("[2/3] Building knowledge graph...")
        if self.config.build_knowledge_graph:
            try:
                # Try to build from Neo4j exports first
                knowledge_graph = self.kg_builder.build_from_neo4j_exports()
            except Exception as e:
                print(f"      Neo4j exports failed ({e}), building from content items...")
                self.kg_builder = KnowledgeGraphBuilder(self.adapter)  # Reset
                knowledge_graph = self.kg_builder.build_from_content_items(content_items)
        else:
            knowledge_graph = ContentKnowledgeGraph()
        print()

        # Step 3: Summary
        print("[3/3] Processing complete!")
        print("-" * 40)
        print(f"Content items: {len(content_items)}")
        print(f"KG nodes: {len(knowledge_graph.graph.nodes())}")
        print(f"KG edges: {len(knowledge_graph.graph.edges())}")
        
        items_with_emotions = sum(1 for item in content_items if item.emotions)
        if len(content_items) > 0:
            pct = 100 * items_with_emotions / len(content_items)
            print(f"Items with emotions: {items_with_emotions} ({pct:.1f}%)")
        else:
            print(f"Items with emotions: 0 (no items loaded)")
        print("=" * 60)

        return content_items, knowledge_graph

    def load_ratings(self, chunk_size: int = 100000) -> Generator[pd.DataFrame, None, None]:
        """
        Load ratings data in chunks (large file).
        
        Yields:
            DataFrame chunks of ratings data
        """
        print(f"Loading ratings from GCS (chunked)...")
        return self.adapter.load_ratings(chunk_size=chunk_size)

    def get_user_profiles(self) -> pd.DataFrame:
        """Load user emotion profiles"""
        return self.user_loader.load_user_profiles()

    def save_processed_data(
        self,
        content_items: List[ContentItem],
                          knowledge_graph: ContentKnowledgeGraph,
        output_dir: Optional[str] = None
    ):
        """Save processed data locally for caching"""
        output_path = Path(output_dir or self.config.cache_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save content items
        content_data = []
        for item in content_items:
            item_dict = {
                'id': item.id,
                'title': item.title,
                'description': item.description,
                'genres': item.genres,
                'year': item.year,
                'metadata': item.metadata
            }
            if item.emotions:
                item_dict['emotions'] = item.emotions.to_dict()
            content_data.append(item_dict)

        with open(output_path / 'content_items.json', 'w') as f:
            json.dump(content_data, f)

        # Save knowledge graph structure
        kg_data = {
            'nodes': [
                {'id': n, 'type': d.get('node_type', 'unknown'), 'attrs': dict(d)}
                for n, d in knowledge_graph.graph.nodes(data=True)
            ],
            'edges': [
                {'source': u, 'target': v, 'relation': d.get('relation', 'unknown'), 'weight': d.get('weight', 1.0)}
                for u, v, d in knowledge_graph.graph.edges(data=True)
            ]
        }

        with open(output_path / 'knowledge_graph.json', 'w') as f:
            json.dump(kg_data, f)

        print(f"Processed data saved to {output_path}")

    def load_cached_data(
        self, 
        cache_dir: Optional[str] = None
    ) -> Optional[Tuple[List[ContentItem], ContentKnowledgeGraph]]:
        """Load previously cached data if available"""
        cache_path = Path(cache_dir or self.config.cache_dir)
        
        content_file = cache_path / 'content_items.json'
        kg_file = cache_path / 'knowledge_graph.json'
        
        if not content_file.exists() or not kg_file.exists():
            return None

        try:
            # Load content items
            with open(content_file) as f:
                content_data = json.load(f)
            
            content_items = []
            for data in content_data:
                emotions = None
                if 'emotions' in data:
                    emotions = EmotionProfile.from_dict(data['emotions'])
                
                item = ContentItem(
                    id=data['id'],
                    title=data['title'],
                    description=data.get('description', ''),
                    genres=data.get('genres', []),
                    year=data.get('year'),
                    emotions=emotions,
                    metadata=data.get('metadata', {})
                )
                content_items.append(item)

            # Load knowledge graph
            with open(kg_file) as f:
                kg_data = json.load(f)
            
            kg = ContentKnowledgeGraph()
            for node in kg_data['nodes']:
                attrs = node.get('attrs', {}).copy()
                if 'node_type' in attrs:
                    del attrs['node_type']
                kg.add_node(node['id'], node['type'], attrs)
            
            for edge in kg_data['edges']:
                triple = KnowledgeTriple(
                    head=edge['source'],
                    relation=edge['relation'],
                    tail=edge['target'],
                    weight=edge.get('weight', 1.0)
                )
                kg.add_triple(triple)

            print(f"Loaded cached data: {len(content_items)} items, {len(kg.graph.nodes())} KG nodes")
            return content_items, kg

        except Exception as e:
            print(f"Error loading cached data: {e}")
            return None
