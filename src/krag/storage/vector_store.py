"""
K-RAG Vector Storage System
Dual vector storage for semantic and emotion embeddings with K-RAG integration
"""

import chromadb
from chromadb.config import Settings
import numpy as np
import torch
from typing import List, Dict, Optional, Tuple, Any
import json
from pathlib import Path
import uuid

from ..core.embeddings import ContentItem, EmotionProfile
from ..core.knowledge_graph import ContentKnowledgeGraph, KRAGSubgraphRetriever


class KRAGVectorStore:
    """
    Enhanced vector store for K-RAG system
    Manages semantic embeddings, emotion embeddings, and knowledge graph embeddings
    """

    def __init__(self, persist_directory: str = "./data/vector_db"):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(exist_ok=True, parents=True)

        self.client = None
        self.semantic_collection = None
        self.emotion_collection = None
        self.knowledge_collection = None  # For K-RAG subgraph embeddings

    def initialize(self):
        """Initialize ChromaDB with persistent storage"""
        print(f"Initializing K-RAG vector store at {self.persist_directory}")

        try:
            self.client = chromadb.PersistentClient(path=str(self.persist_directory))

            # Semantic embeddings collection (content similarity)
            self.semantic_collection = self.client.get_or_create_collection(
                name="content_semantic",
                metadata={"hnsw:space": "cosine", "description": "Semantic content embeddings"}
            )

            # Emotion embeddings collection (emotional similarity)
            self.emotion_collection = self.client.get_or_create_collection(
                name="content_emotion",
                metadata={"hnsw:space": "cosine", "description": "Emotion profile embeddings"}
            )

            # Knowledge graph subgraph embeddings (K-RAG)
            self.knowledge_collection = self.client.get_or_create_collection(
                name="content_knowledge",
                metadata={"hnsw:space": "cosine", "description": "K-RAG subgraph embeddings"}
            )

            print("Vector store initialized successfully")

        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise

    def add_content_item(self, content_item: ContentItem,
                        semantic_embedding: np.ndarray,
                        emotion_embedding: np.ndarray,
                        knowledge_embedding: Optional[np.ndarray] = None):
        """
        Add a single content item with all its embeddings

        Args:
            content_item: ContentItem object
            semantic_embedding: Semantic embedding vector
            emotion_embedding: Emotion embedding vector
            knowledge_embedding: Optional K-RAG subgraph embedding
        """
        content_id = content_item.id

        # Prepare metadata
        base_metadata = {
            "title": content_item.title,
            "description": content_item.description[:500],  # Truncate for storage
            "genres": ",".join(content_item.genres) if content_item.genres else "",
            "year": content_item.year or 0,
            "source": content_item.metadata.get('source', '') if content_item.metadata else ""
        }

        # Add to semantic collection
        self.semantic_collection.add(
            ids=[content_id],
            embeddings=[semantic_embedding.tolist()],
            metadatas=[base_metadata]
        )

        # Add to emotion collection with emotion scores
        emotion_metadata = base_metadata.copy()
        if content_item.emotions:
            emotion_dict = content_item.emotions.to_dict()
            for emotion, score in emotion_dict.items():
                emotion_metadata[f"emotion_{emotion}"] = score

        self.emotion_collection.add(
            ids=[content_id],
            embeddings=[emotion_embedding.tolist()],
            metadatas=[emotion_metadata]
        )

        # Add to knowledge collection if embedding available
        if knowledge_embedding is not None:
            knowledge_metadata = base_metadata.copy()
            knowledge_metadata["has_knowledge"] = True

            self.knowledge_collection.add(
                ids=[content_id],
                embeddings=[knowledge_embedding.tolist()],
                metadatas=[knowledge_metadata]
            )

    def batch_add_content_items(self, content_items: List[ContentItem],
                               semantic_embeddings: np.ndarray,
                               emotion_embeddings: np.ndarray,
                               knowledge_embeddings: Optional[np.ndarray] = None):
        """
        Efficiently add multiple content items

        Args:
            content_items: List of ContentItem objects
            semantic_embeddings: Array of semantic embeddings
            emotion_embeddings: Array of emotion embeddings
            knowledge_embeddings: Optional array of K-RAG embeddings
        """
        print(f"Adding {len(content_items)} items to vector store...")

        # Prepare data for batch insertion
        ids = [item.id for item in content_items]

        # Semantic data
        semantic_embeddings_list = semantic_embeddings.tolist()
        semantic_metadatas = []

        # Emotion data
        emotion_embeddings_list = emotion_embeddings.tolist()
        emotion_metadatas = []

        # Knowledge data (if available)
        knowledge_embeddings_list = None
        knowledge_metadatas = []
        knowledge_ids = []

        for i, item in enumerate(content_items):
            base_metadata = {
                "title": item.title,
                "description": item.description[:500],
                "genres": ",".join(item.genres) if item.genres else "",
                "year": item.year or 0,
                "source": item.metadata.get('source', '') if item.metadata else ""
            }

            # Semantic metadata
            semantic_metadatas.append(base_metadata.copy())

            # Emotion metadata
            emotion_metadata = base_metadata.copy()
            if item.emotions:
                emotion_dict = item.emotions.to_dict()
                for emotion, score in emotion_dict.items():
                    emotion_metadata[f"emotion_{emotion}"] = score
            emotion_metadatas.append(emotion_metadata)

            # Knowledge metadata (if embedding exists)
            if knowledge_embeddings is not None and i < len(knowledge_embeddings):
                knowledge_metadata = base_metadata.copy()
                knowledge_metadata["has_knowledge"] = True
                knowledge_metadatas.append(knowledge_metadata)
                knowledge_ids.append(item.id)

        # Batch insert
        try:
            # Semantic collection
            self.semantic_collection.add(
                ids=ids,
                embeddings=semantic_embeddings_list,
                metadatas=semantic_metadatas
            )

            # Emotion collection
            self.emotion_collection.add(
                ids=ids,
                embeddings=emotion_embeddings_list,
                metadatas=emotion_metadatas
            )

            # Knowledge collection (if data available)
            if knowledge_embeddings is not None and len(knowledge_ids) > 0:
                if knowledge_embeddings_list is None:
                    knowledge_embeddings_list = knowledge_embeddings.tolist()

                self.knowledge_collection.add(
                    ids=knowledge_ids,
                    embeddings=knowledge_embeddings_list[:len(knowledge_ids)],
                    metadatas=knowledge_metadatas
                )

            print(f"Successfully added {len(content_items)} items to vector store")

        except Exception as e:
            print(f"Error in batch add: {e}")
            raise

    def semantic_search(self, query_embedding: np.ndarray,
                       n_results: int = 10,
                       where_filter: Optional[Dict] = None) -> Dict:
        """
        Search by semantic similarity

        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            where_filter: Optional metadata filter

        Returns:
            Search results from ChromaDB
        """
        try:
            results = self.semantic_collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
                where=where_filter
            )
            return results
        except Exception as e:
            print(f"Error in semantic search: {e}")
            return {"ids": [[]], "distances": [[]], "metadatas": [[]]}

    def emotion_search(self, emotion_embedding: np.ndarray,
                      n_results: int = 10,
                      where_filter: Optional[Dict] = None) -> Dict:
        """
        Search by emotional similarity

        Args:
            emotion_embedding: Emotion embedding vector
            n_results: Number of results to return
            where_filter: Optional metadata filter

        Returns:
            Search results from ChromaDB
        """
        try:
            results = self.emotion_collection.query(
                query_embeddings=[emotion_embedding.tolist()],
                n_results=n_results,
                where=where_filter
            )
            return results
        except Exception as e:
            print(f"Error in emotion search: {e}")
            return {"ids": [[]], "distances": [[]], "metadatas": [[]]}

    def knowledge_search(self, knowledge_embedding: np.ndarray,
                        n_results: int = 10,
                        where_filter: Optional[Dict] = None) -> Dict:
        """
        Search by K-RAG knowledge similarity

        Args:
            knowledge_embedding: Knowledge subgraph embedding
            n_results: Number of results to return
            where_filter: Optional metadata filter

        Returns:
            Search results from ChromaDB
        """
        try:
            results = self.knowledge_collection.query(
                query_embeddings=[knowledge_embedding.tolist()],
                n_results=n_results,
                where=where_filter
            )
            return results
        except Exception as e:
            print(f"Error in knowledge search: {e}")
            return {"ids": [[]], "distances": [[]], "metadatas": [[]]}

    def get_content_by_id(self, content_id: str) -> Optional[Dict]:
        """Get content metadata by ID"""
        try:
            result = self.semantic_collection.get(ids=[content_id])
            if result['ids'] and len(result['ids']) > 0:
                return {
                    'id': result['ids'][0],
                    'metadata': result['metadatas'][0] if result['metadatas'] else {}
                }
            return None
        except Exception as e:
            print(f"Error retrieving content {content_id}: {e}")
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            stats = {
                "semantic_count": self.semantic_collection.count(),
                "emotion_count": self.emotion_collection.count(),
                "knowledge_count": self.knowledge_collection.count(),
                "collections": ["content_semantic", "content_emotion", "content_knowledge"]
            }
            return stats
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {"error": str(e)}

    def delete_all(self):
        """Clear all collections (use with caution)"""
        print("Warning: Deleting all data from vector store")
        try:
            if self.client:
                self.client.delete_collection("content_semantic")
                self.client.delete_collection("content_emotion")
                self.client.delete_collection("content_knowledge")
                self.initialize()  # Recreate collections
                print("All collections cleared and recreated")
        except Exception as e:
            print(f"Error clearing collections: {e}")

    def export_data(self, output_path: str):
        """Export vector store data for backup"""
        output_dir = Path(output_path)
        output_dir.mkdir(exist_ok=True, parents=True)

        collections = {
            "semantic": self.semantic_collection,
            "emotion": self.emotion_collection,
            "knowledge": self.knowledge_collection
        }

        for name, collection in collections.items():
            try:
                # Get all data from collection
                result = collection.get()

                export_data = {
                    "ids": result.get("ids", []),
                    "embeddings": result.get("embeddings", []),
                    "metadatas": result.get("metadatas", [])
                }

                # Save to JSON
                with open(output_dir / f"{name}_collection.json", 'w') as f:
                    json.dump(export_data, f, indent=2)

                print(f"Exported {name} collection with {len(export_data['ids'])} items")

            except Exception as e:
                print(f"Error exporting {name} collection: {e}")

    def search_with_filters(self, query_embedding: np.ndarray,
                          collection_type: str = "semantic",
                          genre_filter: Optional[str] = None,
                          year_range: Optional[Tuple[int, int]] = None,
                          emotion_threshold: Optional[Dict[str, float]] = None,
                          n_results: int = 10) -> Dict:
        """
        Advanced search with multiple filters

        Args:
            query_embedding: Query embedding
            collection_type: "semantic", "emotion", or "knowledge"
            genre_filter: Filter by genre
            year_range: Filter by year range (min_year, max_year)
            emotion_threshold: Filter by minimum emotion scores
            n_results: Number of results

        Returns:
            Filtered search results
        """
        # Build where filter
        where_filter = {}

        if genre_filter:
            where_filter["genres"] = {"$contains": genre_filter}

        if year_range:
            where_filter["year"] = {"$gte": year_range[0], "$lte": year_range[1]}

        if emotion_threshold:
            for emotion, threshold in emotion_threshold.items():
                where_filter[f"emotion_{emotion}"] = {"$gte": threshold}

        # Select collection and search
        if collection_type == "semantic":
            return self.semantic_search(query_embedding, n_results, where_filter)
        elif collection_type == "emotion":
            return self.emotion_search(query_embedding, n_results, where_filter)
        elif collection_type == "knowledge":
            return self.knowledge_search(query_embedding, n_results, where_filter)
        else:
            raise ValueError(f"Unknown collection type: {collection_type}")


class VectorStoreManager:
    """
    High-level manager for K-RAG vector operations
    Integrates with data processing and retrieval pipelines
    """

    def __init__(self, vector_store: KRAGVectorStore):
        self.vector_store = vector_store

    def index_content_items(self, content_items: List[ContentItem],
                          semantic_embeddings: np.ndarray,
                          emotion_embeddings: np.ndarray,
                          knowledge_embeddings: Optional[np.ndarray] = None,
                          batch_size: int = 100):
        """
        Index content items in batches

        Args:
            content_items: List of content items
            semantic_embeddings: Semantic embeddings array
            emotion_embeddings: Emotion embeddings array
            knowledge_embeddings: Optional K-RAG embeddings
            batch_size: Batch size for processing
        """
        total_items = len(content_items)
        print(f"Indexing {total_items} content items in batches of {batch_size}")

        for i in range(0, total_items, batch_size):
            end_idx = min(i + batch_size, total_items)

            batch_items = content_items[i:end_idx]
            batch_semantic = semantic_embeddings[i:end_idx]
            batch_emotion = emotion_embeddings[i:end_idx]

            batch_knowledge = None
            if knowledge_embeddings is not None:
                batch_knowledge = knowledge_embeddings[i:end_idx]

            self.vector_store.batch_add_content_items(
                batch_items, batch_semantic, batch_emotion, batch_knowledge
            )

            print(f"Indexed batch {i//batch_size + 1}/{(total_items-1)//batch_size + 1}")

        print(f"Indexing complete. Stats: {self.vector_store.get_stats()}")

    def create_backup(self, backup_path: str):
        """Create backup of vector store"""
        print(f"Creating backup at {backup_path}")
        self.vector_store.export_data(backup_path)

    def health_check(self) -> bool:
        """Check if vector store is healthy"""
        try:
            stats = self.vector_store.get_stats()
            return all(count > 0 for count in [
                stats.get('semantic_count', 0),
                stats.get('emotion_count', 0)
            ])
        except Exception as e:
            print(f"Health check failed: {e}")
            return False