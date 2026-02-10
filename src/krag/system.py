"""
Affective-RAG System Integration
Main system class that orchestrates all components for end-to-end functionality

Usage:
    from krag.system import ARAGSystem, ARAGSystemConfig
    
    config = ARAGSystemConfig()
    system = ARAGSystem(config)
    system.initialize()
    system.load_and_index_data()
    result = system.query("uplifting comedy movies", {"joy": 8, "sadness": 2})
"""

import os
import torch
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import json
from dataclasses import dataclass, field
from dotenv import load_dotenv

from .core.emotion_detection import (
    EmotionProfile, UserEmotionProcessor, EMOTION_LABELS
)
from .core.embeddings import (
    ContentEmbedder, EmotionEmbedder, HybridEmbedder, QueryEmbedder, ContentItem
)
from .core.knowledge_graph import (
    ContentKnowledgeGraph, KRAGEncoder,
    KRAGSubgraphRetriever, AdaptiveRetrievalPolicy
)
from .data.ingestion import (
    DataProcessor, IngestionConfig, generate_node_embeddings_for_kg,
    UserManager, UserProfile
)
from .storage.vector_store import KRAGVectorStore, VectorStoreManager
from .retrieval.krag_retriever import (
    KRAGRetriever, AdaptiveKRAGRetriever, RetrieverFactory,
    QueryContext, RetrievalResult
)
from .llm.response_generator import (
    KRAGResponseGenerator, ResponseConfig, VertexAISetupHelper
)

# Load environment variables
load_dotenv()


@dataclass
class ARAGSystemConfig:
    """
    Configuration for the Affective-RAG system.
    """
    
    # GCS Configuration
    gcs_bucket: str = field(default_factory=lambda: os.getenv('GCS_BUCKET', 'your-gcs-bucket'))
    gcs_base_path: str = "Dataset"
    
    # Local paths
    data_dir: str = "./data"
    model_cache_dir: str = "./models"
    vector_db_path: str = "./data/vector_db"
    cache_dir: str = "./data/cache"
    
    # Model configurations
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    
    # Emotion detection - disabled by default since we use pre-computed emotions from GCS
    # Set to model name to enable live detection for queries
    emotion_model: Optional[str] = None  # e.g., "j-hartmann/emotion-english-distilroberta-base"
    
    # Graph Transformer parameters (matches SentenceBERT 768-dim)
    gnn_embedding_dim: int = 768
    gnn_layers: int = 3
    gnn_heads: int = 4
    
    # Retrieval scoring (see `KRAGRetriever` for details)
    alpha: float = 0.7  # Balance between (semantic+graph) vs affective
    lambda_weight: float = 0.7  # Balance between semantic vs graph within relevance term
    
    # LLM configuration
    vertex_ai_project: Optional[str] = field(
        default_factory=lambda: os.getenv('GOOGLE_CLOUD_PROJECT')
    )
    vertex_ai_location: str = "us-central1"
    model_name: str = "gemini-3-flash-preview"
    temperature: float = 0.7
    
    # Processing parameters
    batch_size: int = 32
    max_content_items: Optional[int] = None  # None = load all
    use_cached_data: bool = True  # Try to load from cache first
    
    # Retriever type for experiments
    # Options: "adaptive_krag" (default), "krag", "vector_only", "semantic", "emotion"
    retriever_type: str = "adaptive_krag"
    
    def __post_init__(self):
        """Validate configuration"""
        if not self.vertex_ai_project:
            print("Warning: GOOGLE_CLOUD_PROJECT not set. LLM features will be limited.")


# Alias for backward compatibility
KRAGSystemConfig = ARAGSystemConfig


class ARAGSystem:
    """
    Complete Affective-RAG system orchestrating all components.
    
    This class provides the main interface for:
    1. Data loading from GCS
    2. Model initialization
    3. Content indexing with embeddings
    4. Query processing and retrieval
    5. Response generation
    """

    def __init__(self, config: Optional[ARAGSystemConfig] = None):
        self.config = config or ARAGSystemConfig()
        self.initialized = False
        self.data_loaded = False

        # Core components (initialized lazily)
        self.content_embedder: Optional[ContentEmbedder] = None
        self.emotion_embedder: Optional[EmotionEmbedder] = None
        self.hybrid_embedder: Optional[HybridEmbedder] = None
        self.query_embedder: Optional[QueryEmbedder] = None

        # K-RAG components
        self.knowledge_graph: Optional[ContentKnowledgeGraph] = None
        self.krag_encoder: Optional[KRAGEncoder] = None
        self.subgraph_retriever: Optional[KRAGSubgraphRetriever] = None

        # Storage and retrieval
        self.vector_store: Optional[KRAGVectorStore] = None
        self.vector_manager: Optional[VectorStoreManager] = None
        self.retriever: Optional[AdaptiveKRAGRetriever] = None

        # LLM and response generation
        self.response_generator: Optional[KRAGResponseGenerator] = None

        # Data processing
        self.data_processor: Optional[DataProcessor] = None
        self.content_items: List[ContentItem] = []

        # User management (for user-scoped retrieval)
        self.user_manager: Optional[UserManager] = None
        
        # User emotion processor
        self.user_emotion_processor = UserEmotionProcessor()

        # Create directories
        self._create_directories()

    def _create_directories(self):
        """Create necessary directories"""
        for path in [
            self.config.data_dir,
            self.config.model_cache_dir,
            self.config.vector_db_path,
            self.config.cache_dir
        ]:
            Path(path).mkdir(parents=True, exist_ok=True)

    def initialize(self, credentials_path: Optional[str] = None):
        """
        Initialize all system components.
        
        Args:
            credentials_path: Optional path to GCP service account JSON.
                             If not provided, uses GOOGLE_APPLICATION_CREDENTIALS env var.
        """
        print("=" * 60, flush=True)
        print("Initializing Affective-RAG System", flush=True)
        print("=" * 60, flush=True)

        try:
            # 1. Setup GCP credentials
            self._setup_gcp(credentials_path)

            # 2. Initialize core components
            self._initialize_core_components()

            # 3. Initialize K-RAG components
            self._initialize_krag_components()

            # 4. Initialize storage
            self._initialize_storage()

            # 5. Initialize response generation
            self._initialize_response_generation()

            # 6. Initialize data processor
            self._initialize_data_processor()

            self.initialized = True
            print(flush=True)
            print("✓ System initialization complete!", flush=True)
            print("=" * 60, flush=True)

        except Exception as e:
            print(f"✗ Error during initialization: {e}", flush=True)
            raise

    def _setup_gcp(self, credentials_path: Optional[str] = None):
        """Setup Google Cloud Platform credentials"""
        print("\n[1/6] Setting up GCP credentials...", flush=True)
        
        if credentials_path:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
            
        if self.config.vertex_ai_project:
            VertexAISetupHelper.setup_credentials(
                self.config.vertex_ai_project,
                credentials_path
            )
            print(f"      Project: {self.config.vertex_ai_project}", flush=True)
        else:
            print("      Warning: No GCP project configured", flush=True)

    def _initialize_core_components(self):
        """Initialize core ML components"""
        print("\n[2/6] Initializing core components...", flush=True)

        # Content embeddings (this downloads ~400MB on first run)
        print(f"      Loading embedding model: {self.config.embedding_model}", flush=True)
        print("      (First run downloads ~400MB, please wait...)", flush=True)
        self.content_embedder = ContentEmbedder(self.config.embedding_model)
        self.content_embedder.initialize()

        # Emotion embeddings
        self.emotion_embedder = EmotionEmbedder(emotion_labels=EMOTION_LABELS)

        # Combined embedders
        self.hybrid_embedder = HybridEmbedder(self.content_embedder, self.emotion_embedder)
        self.query_embedder = QueryEmbedder(self.content_embedder, self.emotion_embedder)
        
        print("      Using pre-computed emotions from GCS", flush=True)

    def _initialize_krag_components(self):
        """Initialize K-RAG specific components"""
        print("\n[3/6] Initializing K-RAG components...", flush=True)

        # Knowledge graph (populated during data loading)
        self.knowledge_graph = ContentKnowledgeGraph()

        # K-RAG encoder with dual GNNs
        self.krag_encoder = KRAGEncoder(
            embedding_dim=self.config.gnn_embedding_dim,
            num_layers=self.config.gnn_layers,
            num_heads=self.config.gnn_heads,
            dropout=0.1
        )
        
        # Load trained weights if available
        self._load_trained_weights()
        
        print(f"      GNN: {self.config.gnn_layers} layers, {self.config.gnn_heads} heads, dim={self.config.gnn_embedding_dim}")

    def _load_trained_weights(self):
        """Load trained weights for the GNN encoder if available"""
        models_dir = Path(self.config.model_cache_dir)
        encoder_path = models_dir / "krag_encoder.pt"
        
        if encoder_path.exists():
            print(f"      Loading trained weights from {models_dir}...", flush=True)
            try:
                self.krag_encoder.load_state_dict(torch.load(encoder_path, map_location='cpu'))
                print("      ✓ Weights loaded successfully", flush=True)
            except Exception as e:
                print(f"      Warning: Failed to load weights: {e}", flush=True)
        else:
            print("      Using random initialization", flush=True)

    def _initialize_storage(self):
        """Initialize vector storage"""
        print("\n[4/6] Initializing vector storage...")

        self.vector_store = KRAGVectorStore(self.config.vector_db_path)
        self.vector_store.initialize()
        self.vector_manager = VectorStoreManager(self.vector_store)
        
        stats = self.vector_store.get_stats()
        print(f"      Vector DB at: {self.config.vector_db_path}")
        print(f"      Existing items: semantic={stats.get('semantic_count', 0)}, emotion={stats.get('emotion_count', 0)}")

    def _initialize_response_generation(self):
        """Initialize LLM response generation"""
        print("\n[5/6] Initializing response generation...")

        response_config = ResponseConfig(
            model_name=self.config.model_name,
            temperature=self.config.temperature,
            project_id=self.config.vertex_ai_project,
            location=self.config.vertex_ai_location
        )

        self.response_generator = KRAGResponseGenerator(response_config)
        print(f"      LLM: {self.config.model_name}")

    def _initialize_data_processor(self):
        """Initialize data processing components"""
        print("\n[6/6] Initializing data processor...")
        
        ingestion_config = IngestionConfig(
            bucket_name=self.config.gcs_bucket,
            base_path=self.config.gcs_base_path,
            max_items=self.config.max_content_items,
            cache_dir=self.config.cache_dir
        )
        
        self.data_processor = DataProcessor(ingestion_config)
        print(f"      GCS: gs://{self.config.gcs_bucket}/{self.config.gcs_base_path}/")

    def load_and_index_data(self, force_reload: bool = False) -> Dict[str, Any]:
        """
        Load data from GCS and index it.
        
        Args:
            force_reload: If True, reload from GCS even if cache exists
            
        Returns:
            Dictionary with processing statistics
        """
        if not self.initialized:
            raise ValueError("System not initialized. Call initialize() first.")

        print()
        print("=" * 60)
        print("Loading and Indexing Data")
        print("=" * 60)

        try:
            # Try to load from cache first
            if self.config.use_cached_data and not force_reload:
                cached = self.data_processor.load_cached_data()
                if cached:
                    self.content_items, self.knowledge_graph = cached
                    print("Loaded from cache!")
                else:
                    print("No cache found, loading from GCS...")
                    self.content_items, self.knowledge_graph = self.data_processor.process()
                    self.data_processor.save_processed_data(self.content_items, self.knowledge_graph)
            else:
                print("Loading from GCS...")
                self.content_items, self.knowledge_graph = self.data_processor.process()
                self.data_processor.save_processed_data(self.content_items, self.knowledge_graph)

            # Limit items if configured
            if self.config.max_content_items and len(self.content_items) > self.config.max_content_items:
                self.content_items = self.content_items[:self.config.max_content_items]
                print(f"Limited to {len(self.content_items)} items")

            # Load user profiles (for user-scoped retrieval)
            print("\nLoading user profiles...", flush=True)
            self.user_manager = UserManager(self.data_processor.adapter)
            # Don't load ratings here - it's 706MB and very slow
            # Ratings will be loaded on-demand when user-scoped query is made
            user_count = self.user_manager.load_users(load_ratings=False)
            print(f"  Users loaded: {user_count}", flush=True)
            print(f"  User-scoped retrieval: AVAILABLE (ratings loaded on-demand)", flush=True)

            # Load embeddings: GCS (primary) → Local cache (fallback) → Generate
            from .data.adapters import DatasetPath
            
            semantic_embeddings = None
            emotion_embeddings = None
            
            # Option 1: Load from GCS (preferred - single source of truth)
            if not force_reload:
                try:
                    if self.data_processor.adapter.exists(DatasetPath.EMBEDDINGS):
                        print("\nLoading embeddings from GCS...", flush=True)
                        cached_emb = self.data_processor.adapter.load_numpy(DatasetPath.EMBEDDINGS)
                        semantic_embeddings = cached_emb['semantic']
                        emotion_embeddings = cached_emb['emotion']
                        print(f"  ✓ Loaded from gs://{self.config.gcs_bucket}/Dataset/{DatasetPath.EMBEDDINGS.value}", flush=True)
                        print(f"  Semantic: {semantic_embeddings.shape}, Emotion: {emotion_embeddings.shape}", flush=True)
                except Exception as e:
                    print(f"  Could not load from GCS: {e}", flush=True)
            
            # Option 2: Load from local cache (fallback for offline work)
            if semantic_embeddings is None and not force_reload:
                local_cache = Path(self.config.cache_dir) / "embeddings.npz"
                if local_cache.exists():
                    print("\nLoading embeddings from local cache...", flush=True)
                    cached_emb = np.load(local_cache)
                    semantic_embeddings = cached_emb['semantic']
                    emotion_embeddings = cached_emb['emotion']
                    print(f"  ✓ Loaded from {local_cache}", flush=True)
                    print(f"  Semantic: {semantic_embeddings.shape}, Emotion: {emotion_embeddings.shape}", flush=True)
            
            # Option 3: Generate embeddings (slow - use Colab instead!)
            if semantic_embeddings is None:
                print("\n⚠️  No precomputed embeddings found!", flush=True)
                print("  Generating locally (SLOW - ~1 hour on CPU)...", flush=True)
                print("  TIP: Run colab_generate_embeddings.py on Colab GPU instead!", flush=True)
                print("  Then upload to your bucket under: Dataset/precomputed/embeddings_v1.npz", flush=True)
                print()
                
                semantic_embeddings, emotion_embeddings = self.hybrid_embedder.batch_create_hybrid_embeddings(
                    self.content_items,
                    batch_size=self.config.batch_size
                )
                print(f"  Semantic: {semantic_embeddings.shape}, Emotion: {emotion_embeddings.shape}")
                
                # Save to local cache (for next run)
                local_cache = Path(self.config.cache_dir) / "embeddings.npz"
                np.savez(local_cache, semantic=semantic_embeddings, emotion=emotion_embeddings)
                print(f"  Saved to {local_cache}", flush=True)
                print("  Upload this file to your bucket under: Dataset/precomputed/embeddings_v1.npz", flush=True)

            # Load node embeddings: GCS → Local → Generate
            node_embeddings = None
            
            if not force_reload:
                # Try GCS first
                try:
                    if self.data_processor.adapter.exists(DatasetPath.NODE_EMBEDDINGS):
                        print("\nLoading node embeddings from GCS...", flush=True)
                        node_embeddings = self.data_processor.adapter.load_pickle(DatasetPath.NODE_EMBEDDINGS)
                        print(f"  ✓ Loaded {len(node_embeddings)} node embeddings from GCS", flush=True)
                except Exception as e:
                    print(f"  Could not load node embeddings from GCS: {e}", flush=True)
            
            if node_embeddings is None:
                print("\nGenerating node embeddings (this may take a while)...", flush=True)
                print("  TIP: Run colab_generate_embeddings.py to pre-generate!", flush=True)
                node_embeddings = generate_node_embeddings_for_kg(
                    self.knowledge_graph,
                    self.content_embedder
                )
                print(f"  Generated {len(node_embeddings)} node embeddings", flush=True)
            
            # Check if vector store already has data indexed
            vs_stats = self.vector_store.get_stats()
            already_indexed = vs_stats.get('semantic_count', 0) >= len(self.content_items)
            
            # Initialize subgraph retriever
            print("\nInitializing knowledge graph retriever...", flush=True)
            self.subgraph_retriever = KRAGSubgraphRetriever(
                self.knowledge_graph,
                self.krag_encoder,
                embedding_dim=self.config.gnn_embedding_dim
            )
            self.subgraph_retriever.set_node_embeddings(node_embeddings)
            
            # Always index subgraphs (needed for knowledge scoring, fast operation)
            print("Indexing knowledge graph subgraphs...", flush=True)
            content_ids = [item.id for item in self.content_items]
            self.subgraph_retriever.index_subgraphs(content_ids)
            
            if already_indexed and not force_reload:
                print("✓ Vector store already indexed, skipping vector indexing...", flush=True)
            else:
                # Get knowledge embeddings for vector store
                knowledge_embeddings = None
                if self.subgraph_retriever.subgraph_embeddings:
                    knowledge_embeddings = np.array([
                        self.subgraph_retriever.subgraph_embeddings.get(
                            item.id, 
                            np.zeros(self.config.gnn_embedding_dim)
                        )
                        for item in self.content_items
                    ])
                    print(f"  Knowledge embeddings: {knowledge_embeddings.shape}")

                # Index in vector store
                print("\nIndexing in vector store...", flush=True)
                self.vector_manager.index_content_items(
                    self.content_items,
                    semantic_embeddings,
                    emotion_embeddings,
                    knowledge_embeddings,
                    batch_size=self.config.batch_size
                )

            # Initialize retriever based on config
            print(f"\nInitializing retriever: {self.config.retriever_type}", flush=True)
            self.retriever = RetrieverFactory.create_retriever(
                self.config.retriever_type,
                self.vector_store,
                self.knowledge_graph,
                self.krag_encoder,
                alpha=self.config.alpha,
                lambda_weight=self.config.lambda_weight,
                subgraph_retriever=self.subgraph_retriever
            )

            self.data_loaded = True

            stats = {
                "content_items": len(self.content_items),
                "knowledge_graph_nodes": len(self.knowledge_graph.graph.nodes()),
                "knowledge_graph_edges": len(self.knowledge_graph.graph.edges()),
                "vector_store": self.vector_store.get_stats(),
                "subgraph_embeddings": len(self.subgraph_retriever.subgraph_embeddings),
                "retriever_type": self.config.retriever_type
            }

            print()
            print("=" * 60)
            print("✓ Data loading and indexing complete!")
            print(f"  Content items: {stats['content_items']}")
            print(f"  KG nodes: {stats['knowledge_graph_nodes']}")
            print(f"  KG edges: {stats['knowledge_graph_edges']}")
            print(f"  Retriever: {stats['retriever_type']}")
            print("=" * 60)

            return stats

        except Exception as e:
            print(f"✗ Error in data processing: {e}")
            raise

    def query(
        self,
        query_text: str,
        emotion_sliders: Optional[Dict[str, int]] = None,
        user_id: Optional[str] = None,
        max_results: int = 5,
        include_explanation: bool = True,
        use_user_scope: bool = True
    ) -> Dict[str, Any]:
        """
        Process a user query and return recommendations.
        
        Args:
            query_text: User's query text
            emotion_sliders: Emotion slider values (0-10 scale), e.g., {"happiness": 8, "sadness": 2}
                           If user_id is provided, this is optional (uses user's profile emotions)
            user_id: Optional user ID for personalized recommendations.
                    If provided:
                    - Uses user's pre-computed emotion profile (unless emotion_sliders override)
                    - Filters results to movies the user has rated (if use_user_scope=True)
            max_results: Maximum number of recommendations
            include_explanation: Whether to include detailed explanations
            use_user_scope: If True and user_id is provided, only recommend from user's rated movies
            
        Returns:
            Dictionary with recommendations and metadata
        """
        if not self.initialized or not self.retriever:
            raise ValueError("System not ready. Call initialize() and load_and_index_data() first.")

        try:
            # Get user profile if user_id provided
            user_profile: Optional[UserProfile] = None
            allowed_content_ids: Optional[set] = None
            
            if user_id and self.user_manager:
                # Load ratings on-demand if needed for user-scoped retrieval
                if use_user_scope and not self.user_manager.ratings_loaded:
                    print("Loading user ratings for scoped retrieval (first time only)...", flush=True)
                    self.user_manager._load_user_ratings()
                
                user_profile = self.user_manager.get_user(user_id)
                if user_profile:
                    # Use user's allowed movies for scoped retrieval
                    if use_user_scope and user_profile.watched_movies:
                        allowed_content_ids = user_profile.watched_movies
                else:
                    print(f"Warning: User {user_id} not found, using ad-hoc query mode")
            
            # Determine emotion profile to use
            if emotion_sliders:
                # Explicit emotion sliders override user profile
                user_emotions = self.user_emotion_processor.process_emotion_sliders(emotion_sliders)
            elif user_profile:
                # Use user's pre-computed emotion profile
                user_emotions = user_profile.emotions
            else:
                # Default neutral emotions
                user_emotions = EmotionProfile()

            # Generate query embeddings
            query_embedding, emotion_embedding = self.query_embedder.embed_query(
                query_text, user_emotions
            )

            # Create query context with user scope
            query_context = QueryContext(
                query_text=query_text,
                user_emotions=user_emotions,
                query_embedding=query_embedding,
                emotion_embedding=emotion_embedding,
                allowed_content_ids=allowed_content_ids,
                user_id=user_id
            )

            # Retrieve recommendations
            retrieval_results = self.retriever.retrieve(query_context, k=max_results)

            # Generate response
            response_text = self.response_generator.generate_response(
                query_context, retrieval_results, max_recommendations=max_results
            )

            # Prepare result
            result = {
                "query": query_text,
                "user_id": user_id,
                "user_emotions": user_emotions.to_dict(),
                "user_scope": {
                    "enabled": allowed_content_ids is not None,
                    "available_movies": len(allowed_content_ids) if allowed_content_ids else "all"
                },
                "response": response_text,
                "recommendations": [],
                "metadata": {
                    "num_results": len(retrieval_results),
                    "retrieval_method": "adaptive_krag",
                    "alpha": self.config.alpha
                }
            }

            # Add detailed recommendations
            for i, res in enumerate(retrieval_results):
                rec = {
                    "rank": i + 1,
                    "title": res.title,
                    "content_id": res.content_id,
                    "scores": {
                        "semantic": res.semantic_score,
                        "emotion": res.emotion_score,
                        "knowledge": res.knowledge_score,
                        "combined": res.combined_score
                    },
                    "metadata": res.metadata
                }

                if include_explanation:
                    rec["explanation"] = self.retriever.explain_retrieval(res, query_context)

                result["recommendations"].append(rec)

            return result

        except Exception as e:
            print(f"Error processing query: {e}")
            return {
                "query": query_text,
                "error": str(e),
                "recommendations": []
            }

    def get_content_details(self, content_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a content item"""
        content_info = self.vector_store.get_content_by_id(content_id)

        if content_info and self.subgraph_retriever:
            knowledge_context = self.subgraph_retriever.get_knowledge_context(content_id)
            content_info["knowledge_context"] = knowledge_context

        return content_info

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        stats = {
            "system_status": "ready" if (self.initialized and self.data_loaded) else "not_ready",
            "initialized": self.initialized,
            "data_loaded": self.data_loaded,
            "config": {
                "gcs_bucket": self.config.gcs_bucket,
                "embedding_model": self.config.embedding_model,
                "emotion_model": self.config.emotion_model,
                "llm_model": self.config.model_name,
                "alpha": self.config.alpha
            }
        }

        if self.data_loaded:
            stats["content_items"] = len(self.content_items)
            
        if self.vector_store:
            stats["vector_store"] = self.vector_store.get_stats()

        if self.knowledge_graph:
            stats["knowledge_graph"] = {
                "nodes": len(self.knowledge_graph.graph.nodes()),
                "edges": len(self.knowledge_graph.graph.edges())
            }

        if self.subgraph_retriever:
            stats["subgraph_embeddings"] = len(self.subgraph_retriever.subgraph_embeddings)

        if self.user_manager:
            stats["users"] = {
                "total": self.user_manager.user_count,
                "ratings_loaded": self.user_manager.ratings_loaded
            }

        return stats

    # =========================================================================
    # User Management Methods
    # =========================================================================

    def get_user(self, user_id: str) -> Optional[UserProfile]:
        """
        Get a user profile by ID.
        
        Args:
            user_id: The user ID to look up
            
        Returns:
            UserProfile or None if not found
        """
        if not self.user_manager:
            return None
        return self.user_manager.get_user(user_id)

    def list_users(self, limit: int = 100) -> List[str]:
        """List available user IDs"""
        if not self.user_manager:
            return []
        return self.user_manager.list_users(limit)

    def get_users_with_min_ratings(self, min_ratings: int = 50) -> List[str]:
        """Get users who have rated at least min_ratings movies"""
        if not self.user_manager:
            return []
        return self.user_manager.get_users_with_min_ratings(min_ratings)

    # =========================================================================
    # Experiment Methods
    # =========================================================================

    def set_retriever_type(self, retriever_type: str) -> None:
        """
        Switch retriever type at runtime for A/B experiments.
        
        Args:
            retriever_type: One of:
                - "adaptive_krag": Full K-RAG with adaptive weights (default)
                - "krag": Standard K-RAG
                - "vector_only": Semantic + Emotion, no knowledge graph
                - "semantic": Semantic similarity only
                - "emotion": Emotion similarity only
                
        Example:
            # Compare Graph+Vector vs Vector-only
            system.set_retriever_type("adaptive_krag")
            result_with_kg = system.query(query, user_id=user_id)
            
            system.set_retriever_type("vector_only")
            result_without_kg = system.query(query, user_id=user_id)
        """
        if not self.data_loaded:
            raise ValueError("Data not loaded. Call load_and_index_data() first.")
        
        self.config.retriever_type = retriever_type
        self.retriever = RetrieverFactory.create_retriever(
            retriever_type,
            self.vector_store,
            self.knowledge_graph,
            self.krag_encoder,
            alpha=self.config.alpha,
            lambda_weight=self.config.lambda_weight,
            subgraph_retriever=self.subgraph_retriever
        )
        print(f"Retriever switched to: {retriever_type}")

    def get_available_retriever_types(self) -> List[str]:
        """Get list of available retriever types for experiments"""
        return [
            "adaptive_krag",  # Full K-RAG with adaptive weights
            "krag",           # Standard K-RAG  
            "vector_only",    # Semantic + Emotion (no KG)
            "semantic",       # Semantic only
            "emotion"         # Emotion only
        ]

    def health_check(self) -> bool:
        """Perform system health check"""
        try:
            checks = []

            # Check initialization
            if not self.initialized:
                print("✗ System not initialized")
                return False
            checks.append("initialized")

            # Check data loaded
            if not self.data_loaded:
                print("✗ Data not loaded")
                return False
            checks.append("data_loaded")

            # Check vector store
            if self.vector_manager and self.vector_manager.health_check():
                checks.append("vector_store")
            else:
                print("✗ Vector store unhealthy")
                return False

            # Check LLM connection
            if self.response_generator:
                checks.append("llm_ready")

            print(f"✓ Health check passed: {', '.join(checks)}")
            return True

        except Exception as e:
            print(f"✗ Health check failed: {e}")
            return False

    # =========================================================================
    # GNN Training Methods
    # =========================================================================

    def train_gnn(
        self,
        num_epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        patience: int = 10,
        checkpoint_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Train the GNN encoder with self-supervised denoising objective.

        Args:
            num_epochs: Maximum training epochs
            batch_size: Training batch size
            learning_rate: Adam learning rate
            patience: Early stopping patience
            checkpoint_dir: Directory for saving checkpoints

        Returns:
            Training results with metrics and history
        """
        if not self.data_loaded:
            raise ValueError("Data not loaded. Call load_and_index_data() first.")

        from .training import GNNTrainer, TrainingConfig
        from .training.gnn_trainer import prepare_emotion_ground_truth
        from .data.adapters import DatasetPath

        checkpoint_dir = checkpoint_dir or self.config.model_cache_dir

        config = TrainingConfig(
            embedding_dim=self.config.gnn_embedding_dim,
            num_layers=self.config.gnn_layers,
            num_heads=self.config.gnn_heads,
            learning_rate=learning_rate,
            batch_size=batch_size,
            num_epochs=num_epochs,
            patience=patience,
            checkpoint_dir=checkpoint_dir
        )

        print("=" * 60)
        print("GNN Training: Self-Supervised Denoising")
        print("=" * 60)

        # Load ground truth emotions from GCS
        print("\nLoading ground truth emotions from GCS...")
        movies_df = self.data_processor.adapter.load_movies(vector_ready=True)
        emotion_ground_truth = prepare_emotion_ground_truth(movies_df)
        print(f"  Loaded emotion ground truth for {len(emotion_ground_truth)} movies")

        # Load node embeddings
        node_embeddings = None
        try:
            if self.data_processor.adapter.exists(DatasetPath.NODE_EMBEDDINGS):
                node_embeddings = self.data_processor.adapter.load_pickle(DatasetPath.NODE_EMBEDDINGS)
                print(f"  Loaded {len(node_embeddings)} node embeddings from GCS")
        except Exception as e:
            print(f"  Could not load node embeddings: {e}")

        if node_embeddings is None:
            from .data.ingestion import generate_node_embeddings_for_kg
            print("  Generating node embeddings...")
            node_embeddings = generate_node_embeddings_for_kg(
                self.knowledge_graph,
                self.content_embedder
            )

        # Initialize trainer
        trainer = GNNTrainer(
            gnn_encoder=self.krag_encoder,
            config=config,
            knowledge_graph=self.knowledge_graph
        )

        # Prepare datasets
        train_dataset, val_dataset = trainer.prepare_training_data(
            content_items=self.content_items,
            node_embeddings=node_embeddings,
            emotion_ground_truth=emotion_ground_truth
        )

        # Train
        results = trainer.train(train_dataset, val_dataset)

        # Save trained encoder weights
        encoder_path = Path(checkpoint_dir) / "krag_encoder.pt"
        torch.save(self.krag_encoder.state_dict(), encoder_path)
        print(f"\nSaved trained encoder to: {encoder_path}")

        # Extract and save smoothed emotions
        print("\nExtracting graph-smoothed emotions...")
        smoothed_emotions = trainer.extract_smoothed_emotions(
            self.content_items,
            node_embeddings
        )

        # Save smoothed emotions
        smoothed_path = Path(checkpoint_dir) / "smoothed_emotions.npz"
        trainer.save_smoothed_emotions(smoothed_emotions, str(smoothed_path))

        # Update retriever with smoothed emotions
        if hasattr(self.retriever, 'smoothed_emotions'):
            self.retriever.smoothed_emotions = smoothed_emotions
            print(f"  Updated retriever with {len(smoothed_emotions)} smoothed emotion vectors")

        print("\n" + "=" * 60)
        print("GNN Training Complete!")
        print(f"  Best validation loss: {results['best_val_loss']:.4f}")
        print(f"  Final epoch: {results['final_epoch']}")
        print("=" * 60)

        return results

    def load_smoothed_emotions(self, path: Optional[str] = None) -> int:
        """
        Load pre-computed smoothed emotions into the retriever.

        Args:
            path: Path to smoothed_emotions.npz file.
                  Default: {model_cache_dir}/smoothed_emotions.npz

        Returns:
            Number of emotions loaded
        """
        if path is None:
            path = Path(self.config.model_cache_dir) / "smoothed_emotions.npz"
        else:
            path = Path(path)

        if not path.exists():
            print(f"No smoothed emotions found at {path}")
            return 0

        data = np.load(path, allow_pickle=True)
        content_ids = data['content_ids']
        emotions = data['emotions']

        smoothed_emotions = {
            cid: emot for cid, emot in zip(content_ids, emotions)
        }

        if hasattr(self.retriever, 'smoothed_emotions'):
            self.retriever.smoothed_emotions = smoothed_emotions
            print(f"Loaded {len(smoothed_emotions)} smoothed emotion vectors")

        return len(smoothed_emotions)


# Alias for backward compatibility
KRAGSystem = ARAGSystem
