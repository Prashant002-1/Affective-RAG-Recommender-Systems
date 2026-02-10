"""
K-RAG Knowledge Graph Implementation
Based on K-RagRec paper: Knowledge Graph Retrieval-Augmented Generation for LLM-based Recommendation
arXiv:2501.02226
"""

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_mean_pool
from torch_geometric.data import Data, Batch
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class KnowledgeTriple:
    """Represents a knowledge triple: (head, relation, tail)"""
    head: str
    relation: str
    tail: str
    weight: float = 1.0


class ContentKnowledgeGraph:
    """
    Knowledge graph for content relationships.
    Follows K-RagRec: G = {(n, e, n') | n, n' in N, e in E}
    """

    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.node_to_id = {}
        self.id_to_node = {}
        self.relation_to_id = {}
        self.id_to_relation = {}
        self.node_types = {}
        self.node_text = {}  # Store text attributes for PLM encoding

    def add_node(self, node_id: str, node_type: str, attributes: Dict = None):
        """Add a node to the knowledge graph"""
        if node_id not in self.node_to_id:
            idx = len(self.node_to_id)
            self.node_to_id[node_id] = idx
            self.id_to_node[idx] = node_id
            self.node_types[node_id] = node_type

        attrs = attributes or {}
        self.graph.add_node(node_id, node_type=node_type, **attrs)

        # Store text for encoding
        if 'title' in attrs:
            self.node_text[node_id] = attrs.get('title', '') + '. ' + attrs.get('description', '')
        elif 'name' in attrs:
            self.node_text[node_id] = attrs['name']
        else:
            self.node_text[node_id] = node_id

    def add_triple(self, triple: KnowledgeTriple):
        """Add a knowledge triple to the graph"""
        if triple.head not in self.node_to_id:
            self.add_node(triple.head, 'content')
        if triple.tail not in self.node_to_id:
            self.add_node(triple.tail, 'attribute')

        if triple.relation not in self.relation_to_id:
            idx = len(self.relation_to_id)
            self.relation_to_id[triple.relation] = idx
            self.id_to_relation[idx] = triple.relation

        self.graph.add_edge(
            triple.head,
            triple.tail,
            relation=triple.relation,
            weight=triple.weight
        )

    def get_neighbors(self, node_id: str, relation: Optional[str] = None, hops: int = 1) -> List[str]:
        """Get neighbors of a node within specified hops"""
        if node_id not in self.graph:
            return []

        if hops == 1:
            if relation:
                neighbors = []
                for neighbor in self.graph.neighbors(node_id):
                    edge_data = self.graph.get_edge_data(node_id, neighbor)
                    if edge_data and any(data.get('relation') == relation for data in edge_data.values()):
                        neighbors.append(neighbor)
                return neighbors
            else:
                return list(self.graph.neighbors(node_id))
        else:
            visited = set()
            current_level = {node_id}

            for _ in range(hops):
                next_level = set()
                for node in current_level:
                    neighbors = self.get_neighbors(node, relation, hops=1)
                    next_level.update(set(neighbors) - visited)
                visited.update(current_level)
                current_level = next_level

            visited.update(current_level)
            visited.discard(node_id)
            return list(visited)

    def extract_subgraph(self, central_node: str, hops: int = 2) -> nx.Graph:
        """Extract a subgraph around a central node"""
        if central_node not in self.graph:
            return nx.MultiDiGraph()

        nodes_to_include = {central_node}
        nodes_to_include.update(self.get_neighbors(central_node, hops=hops))

        return self.graph.subgraph(nodes_to_include).copy()

    def to_pytorch_geometric(
        self,
        subgraph: nx.Graph,
        node_embeddings: Dict[str, np.ndarray] = None,
        target_dim: int = 768
    ) -> Data:
        """
        Convert graph to PyTorch Geometric format.

        Args:
            subgraph: NetworkX subgraph
            node_embeddings: Pre-computed node embeddings from PLM (typically 768-dim)
            target_dim: Target dimension for GNN input (default 1024)
        """
        if len(subgraph.nodes()) == 0:
            # Return empty data
            return Data(
                x=torch.zeros(1, target_dim),
                edge_index=torch.zeros(2, 0, dtype=torch.long)
            )

        nodes = list(subgraph.nodes())
        node_map = {node: i for i, node in enumerate(nodes)}

        edges = []
        edge_attrs = []

        for u, v, data in subgraph.edges(data=True):
            edges.append([node_map[u], node_map[v]])
            relation_id = self.relation_to_id.get(data.get('relation', 'unknown'), 0)
            edge_attrs.append([relation_id, data.get('weight', 1.0)])

        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long)
            edge_attr = torch.zeros(0, 2)

        if not node_embeddings:
            raise ValueError(
                "node_embeddings is required for to_pytorch_geometric. "
                "Training without node embeddings produces meaningless results."
            )

        node_features = []
        missing_count = 0
        for node in nodes:
            if node in node_embeddings:
                emb = node_embeddings[node]
                if len(emb) != target_dim:
                    raise ValueError(
                        f"Embedding dimension mismatch for node {node}: "
                        f"got {len(emb)}, expected {target_dim}"
                    )
                node_features.append(emb)
            else:
                missing_count += 1
                node_features.append(np.zeros(target_dim))

        x = torch.tensor(np.array(node_features), dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


class GraphTransformerEncoder(nn.Module):
    """
    Graph Transformer encoder for denoising affective signatures.
    Uses TransformerConv layers with multi-head attention weighted by EVOKES edge intensities.

    Default: 768-dim to match SentenceBERT embeddings
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 768,
        output_dim: int = 768,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        num_relations: int = 10,
        edge_dim: int = 16
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.edge_dim = edge_dim

        # Edge feature processing: embed relation_id + combine with weight
        self.relation_embedding = nn.Embedding(num_relations, edge_dim - 1)
        self.edge_proj = nn.Linear(edge_dim, edge_dim)

        self.layers = nn.ModuleList()

        # Input projection if dimensions differ
        if input_dim != hidden_dim:
            self.input_proj = nn.Linear(input_dim, hidden_dim)
        else:
            self.input_proj = nn.Identity()

        # Graph Transformer layers with edge features
        for i in range(num_layers):
            in_dim = hidden_dim
            out_dim = hidden_dim if i < num_layers - 1 else output_dim

            self.layers.append(
                TransformerConv(
                    in_dim,
                    out_dim // num_heads,
                    heads=num_heads,
                    concat=True,
                    dropout=dropout,
                    edge_dim=edge_dim
                )
            )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)

    def _process_edge_features(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Process edge attributes [relation_id, weight] into edge embeddings.
        Relation_id is embedded, then concatenated with weight, then projected.
        """
        if edge_attr is None or edge_attr.size(0) == 0:
            return None

        relation_ids = edge_attr[:, 0].long().clamp(0, self.relation_embedding.num_embeddings - 1)
        weights = edge_attr[:, 1:2]

        relation_emb = self.relation_embedding(relation_ids)
        edge_features = torch.cat([relation_emb, weights], dim=-1)
        edge_features = self.edge_proj(edge_features)

        return edge_features

    def forward(self, data: Data) -> torch.Tensor:
        """
        Encode a knowledge subgraph to a fixed-size vector.
        Uses edge weights (EVOKES intensities) in attention mechanism.

        Returns:
            Subgraph embedding of shape [output_dim]
        """
        x, edge_index = data.x, data.edge_index
        edge_attr = getattr(data, 'edge_attr', None)

        # Input projection
        x = self.input_proj(x)

        # Handle case with no edges: just return mean of projected node features
        if edge_index.size(1) == 0:
            x = self.layer_norm(x)
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            graph_embedding = global_mean_pool(x, batch)
            return graph_embedding.squeeze(0)

        # Process edge features for attention
        edge_features = self._process_edge_features(edge_attr)

        if edge_features is None:
            raise RuntimeError(
                f"Edge feature processing failed in GNN forward pass. "
                f"edge_attr: {edge_attr.shape if edge_attr is not None else None}, "
                f"edge_index: {edge_index.shape}. Check graph construction."
            )

        # Apply Graph Transformer layers with edge features
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_attr=edge_features)
            if i < len(self.layers) - 1:
                x = F.gelu(x)
                x = self.dropout(x)

        x = self.layer_norm(x)

        # Global mean pooling for graph-level representation
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        graph_embedding = global_mean_pool(x, batch)

        return graph_embedding.squeeze(0)


class KRAGEncoder(nn.Module):
    """
    Graph Transformer encoder for Affective-RAG:
    - GNN_Indexing: For building subgraph representations during indexing
    - GNN_Encoding: For encoding retrieved subgraphs during inference
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        # GNN for indexing (phi_1 in paper)
        self.gnn_indexing = GraphTransformerEncoder(
            input_dim=embedding_dim,
            hidden_dim=embedding_dim,
            output_dim=embedding_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )

        # GNN for encoding retrieved subgraphs (phi_2 in paper)
        self.gnn_encoding = GraphTransformerEncoder(
            input_dim=embedding_dim,
            hidden_dim=embedding_dim,
            output_dim=embedding_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )

    def index_subgraph(self, data: Data) -> torch.Tensor:
        """Create index embedding for a subgraph"""
        return self.gnn_indexing(data)

    def encode_subgraph(self, data: Data) -> torch.Tensor:
        """Encode a retrieved subgraph"""
        return self.gnn_encoding(data)


class AdaptiveRetrievalPolicy:
    """
    Adaptive retrieval policy from K-RagRec paper.
    Retrieve if popularity < threshold (paper uses 50%)
    """

    def __init__(self, popularity_threshold: float = 0.5):
        self.popularity_threshold = popularity_threshold

    def should_retrieve(self, item_popularity: float) -> bool:
        """
        Decide whether to retrieve knowledge for this item.
        Paper: "retrieve if popularity < threshold p"
        """
        return item_popularity < self.popularity_threshold


class KRAGSubgraphRetriever:
    """
    Knowledge Sub-graph Retrieval and Re-ranking.
    Follows K-RagRec pipeline:
    1. Create query embedding
    2. Retrieve top-K similar subgraphs
    3. Re-rank using recommendation context
    """

    def __init__(
        self,
        knowledge_graph: ContentKnowledgeGraph,
        encoder: KRAGEncoder,
        embedding_dim: int = 768,
        top_k_retrieve: int = 3,
        top_n_rerank: int = 5
    ):
        self.kg = knowledge_graph
        self.encoder = encoder
        self.embedding_dim = embedding_dim
        self.top_k = top_k_retrieve
        self.top_n = top_n_rerank

        # Cache for indexed subgraph embeddings
        self.subgraph_embeddings = {}
        self.node_embeddings = {}

    def set_node_embeddings(self, embeddings: Dict[str, np.ndarray]):
        """Set PLM embeddings for nodes"""
        self.node_embeddings = embeddings

    def index_subgraphs(self, content_ids: List[str], hops: int = 2):
        """
        Pre-compute index embeddings for all content subgraphs.
        Uses GNN_Indexing (phi_1).
        """
        print(f"Indexing subgraphs for {len(content_ids)} items...")

        self.encoder.eval()
        device = next(self.encoder.parameters()).device
        with torch.no_grad():
            for content_id in content_ids:
                if content_id in self.subgraph_embeddings:
                    continue

                subgraph = self.kg.extract_subgraph(content_id, hops=hops)
                if len(subgraph.nodes()) > 0:
                    pyg_data = self.kg.to_pytorch_geometric(subgraph, self.node_embeddings).to(device)
                    embedding = self.encoder.index_subgraph(pyg_data)
                    self.subgraph_embeddings[content_id] = embedding.cpu().numpy()

    def retrieve_subgraphs(
        self,
        query_embedding: np.ndarray,
        exclude_ids: List[str] = None
    ) -> List[Tuple[str, float]]:
        """
        Retrieve top-K most similar subgraphs based on query.
        Paper: G_j = argtopK sim(q_j, z_g*)
        """
        if not self.subgraph_embeddings:
            return []

        exclude_ids = exclude_ids or []
        similarities = []

        for content_id, embedding in self.subgraph_embeddings.items():
            if content_id in exclude_ids:
                continue

            # Cosine similarity
            sim = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding) + 1e-8
            )
            similarities.append((content_id, float(sim)))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:self.top_k]

    def rerank_subgraphs(
        self,
        retrieved: List[Tuple[str, float]],
        context_embedding: np.ndarray
    ) -> List[Tuple[str, float]]:
        """
        Re-rank retrieved subgraphs using recommendation context.
        Paper: G_hat = argtopN sim(p, z_g*')
        """
        if not retrieved:
            return []

        reranked = []
        for content_id, _ in retrieved:
            if content_id in self.subgraph_embeddings:
                embedding = self.subgraph_embeddings[content_id]
                sim = np.dot(context_embedding, embedding) / (
                    np.linalg.norm(context_embedding) * np.linalg.norm(embedding) + 1e-8
                )
                reranked.append((content_id, float(sim)))

        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:self.top_n]

    def encode_retrieved_subgraphs(
        self,
        content_ids: List[str],
        hops: int = 2
    ) -> torch.Tensor:
        """
        Encode retrieved subgraphs using GNN_Encoding (phi_2).
        Returns concatenated embeddings for MLP projection.
        """
        self.encoder.eval()
        embeddings = []

        with torch.no_grad():
            for content_id in content_ids:
                subgraph = self.kg.extract_subgraph(content_id, hops=hops)
                pyg_data = self.kg.to_pytorch_geometric(subgraph, self.node_embeddings)
                embedding = self.encoder.encode_subgraph(pyg_data)
                embeddings.append(embedding)

        if embeddings:
            return torch.stack(embeddings)
        else:
            return torch.zeros(1, self.embedding_dim)

    def get_knowledge_context(self, content_id: str) -> str:
        """Generate textual knowledge context from subgraph"""
        subgraph = self.kg.extract_subgraph(content_id, hops=2)

        context_parts = []
        for u, v, data in subgraph.edges(data=True):
            relation = data.get('relation', 'related_to')
            weight = data.get('weight', 1.0)

            if weight > 0.3:
                context_parts.append(f"{u} {relation} {v}")

        return "; ".join(context_parts[:10])


# Backward compatibility alias
HopFieldGNNEncoder = GraphTransformerEncoder
