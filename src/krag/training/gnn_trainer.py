"""
GNN Trainer for Affective-RAG

Self-supervised denoising training for Graph Transformer encoder.
Trains GNN to produce denoised emotion vectors from graph structure.

Training Objective:
- Input: Node features (semantic embeddings) + graph structure + EVOKES edge weights
- Target: Ground truth emotion vectors from movies_vector_ready.csv
- Loss: MSE between predicted and actual emotion intensities
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import json
import torch.nn.functional as F


@dataclass
class TrainingConfig:
    """Configuration for GNN training"""
    embedding_dim: int = 768
    emotion_dim: int = 6
    hidden_dim: int = 768
    num_layers: int = 3
    num_heads: int = 4
    dropout: float = 0.1
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 32
    num_epochs: int = 50
    patience: int = 10
    k_hops: int = 2
    k_hops_list: List[int] = field(default_factory=lambda: [1, 2, 3])
    contrastive_weight: float = 0.3
    alignment_weight: float = 0.3
    checkpoint_dir: str = "./checkpoints"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class EmotionProjector(nn.Module):
    """
    Projects GNN output to 6-dim emotion space.
    Used during training to predict denoised emotion vectors.
    """
    def __init__(self, input_dim: int = 768, emotion_dim: int = 6, hidden_dim: int = 256):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, emotion_dim),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)


class SubgraphDataset(Dataset):
    """
    Dataset of movie subgraphs with target emotion vectors.
    """
    def __init__(
        self,
        subgraph_data: List[Data],
        emotion_targets: torch.Tensor,
        content_ids: List[str]
    ):
        self.subgraph_data = subgraph_data
        self.emotion_targets = emotion_targets
        self.content_ids = content_ids

    def __len__(self):
        return len(self.subgraph_data)

    def __getitem__(self, idx):
        return self.subgraph_data[idx], self.emotion_targets[idx], self.content_ids[idx]


def collate_subgraphs(batch):
    """Custom collate function for batching PyG Data objects"""
    graphs, targets, content_ids = zip(*batch)
    batched_graphs = Batch.from_data_list(graphs)
    batched_targets = torch.stack(targets)
    return batched_graphs, batched_targets, list(content_ids)


class MultiKSubgraphDataset(Dataset):
    """Dataset that returns multi-k subgraphs for each content item."""

    def __init__(
        self,
        multi_k_data: Dict[str, Dict[int, Data]],
        emotion_targets: Dict[str, torch.Tensor],
        k_list: List[int]
    ):
        self.content_ids = list(multi_k_data.keys())
        self.multi_k_data = multi_k_data
        self.emotion_targets = emotion_targets
        self.k_list = k_list

    def __len__(self):
        return len(self.content_ids)

    def __getitem__(self, idx):
        content_id = self.content_ids[idx]
        return {
            'content_id': content_id,
            'k_subgraphs': self.multi_k_data[content_id],
            'target': self.emotion_targets[content_id]
        }


def collate_multi_k(batch, k_list):
    """Collate function that batches each k-hop graph set separately."""
    batched = {}
    for k in k_list:
        graphs_k = [item['k_subgraphs'][k] for item in batch if k in item['k_subgraphs']]
        if graphs_k:
            batched[k] = Batch.from_data_list(graphs_k)
    batched['targets'] = torch.stack([item['target'] for item in batch])
    batched['content_ids'] = [item['content_id'] for item in batch]
    return batched


class GNNTrainer:
    """
    Trainer for Graph Transformer encoder with self-supervised denoising.

    The training process:
    1. For each movie, extract k-hop subgraph from knowledge graph
    2. Pass through GNN to get graph-level embedding
    3. Project embedding to 6-dim emotion space
    4. Compute MSE loss against ground truth emotions
    """

    def __init__(
        self,
        gnn_encoder: nn.Module,
        config: TrainingConfig,
        knowledge_graph=None,
        node_embeddings=None
    ):
        self.config = config
        self.device = torch.device(config.device)

        self.gnn_encoder = gnn_encoder.to(self.device)
        self.gnn_encoder_indexing = self.gnn_encoder.gnn_indexing
        self.node_embeddings = node_embeddings

        if config.alignment_weight > 0 and node_embeddings is None:
            raise ValueError(
                f"alignment_weight={config.alignment_weight} but node_embeddings not provided. "
                "Pass node_embeddings to GNNTrainer or set alignment_weight=0."
            )

        self.emotion_projector = EmotionProjector(
            input_dim=config.embedding_dim,
            emotion_dim=config.emotion_dim
        ).to(self.device)

        self.knowledge_graph = knowledge_graph

        self.optimizer = optim.AdamW(
            list(self.gnn_encoder_indexing.parameters()) +
            list(self.emotion_projector.parameters()),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

        self.criterion = nn.MSELoss()

        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_loss = float('inf')
        self.epochs_without_improvement = 0
        self.training_history = []

    def _get_semantic_targets(self, content_ids: List[str]) -> Optional[torch.Tensor]:
        if self.node_embeddings is None:
            return None
        targets = []
        for cid in content_ids:
            if cid not in self.node_embeddings:
                raise KeyError(
                    f"Content ID '{cid}' not found in node_embeddings. "
                    "Ensure node_embeddings covers all training content IDs."
                )
            emb = self.node_embeddings[cid]
            targets.append(torch.tensor(emb, dtype=torch.float))
        return torch.stack(targets).to(self.device)

    def contrastive_loss(
        self,
        emb_k1: torch.Tensor,
        emb_k2: torch.Tensor,
        emb_k3: torch.Tensor,
        margin: float = 0.2
    ) -> torch.Tensor:
        """
        Structural contrastive loss encouraging k-dependent embeddings.

        Goal: k=1, k=2, k=3 should produce distinct embeddings for the same content.
        - k=1 (local context) should differ from k=3 (broad context)
        - k=2 should be intermediate

        Uses triplet-style margin loss:
        - Push k=1 and k=3 apart (they represent different structural views)
        - Encourage ordering: sim(k1,k2) > sim(k1,k3) (k2 is closer to k1 than k3 is)
        """
        z1 = F.normalize(emb_k1, dim=-1)
        z2 = F.normalize(emb_k2, dim=-1)
        z3 = F.normalize(emb_k3, dim=-1)

        sim_1_2 = (z1 * z2).sum(dim=-1)
        sim_1_3 = (z1 * z3).sum(dim=-1)
        sim_2_3 = (z2 * z3).sum(dim=-1)

        ordering_loss = (
            torch.relu(sim_1_3 - sim_1_2 + margin).mean() +
            torch.relu(sim_1_3 - sim_2_3 + margin).mean()
        )

        diversity_loss = torch.relu(sim_1_3 - 0.5).mean()

        return ordering_loss + diversity_loss

    def prepare_training_data(
        self,
        content_items: List,
        node_embeddings: Dict[str, np.ndarray],
        emotion_ground_truth: Dict[str, np.ndarray]
    ) -> Tuple[SubgraphDataset, SubgraphDataset]:
        """
        Prepare training and validation datasets.

        Args:
            content_items: List of ContentItem objects
            node_embeddings: Dict mapping node_id -> 768-dim embedding
            emotion_ground_truth: Dict mapping content_id -> 6-dim emotion vector

        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        subgraph_data = []
        emotion_targets = []
        content_ids = []

        print(f"Preparing training data for {len(content_items)} items...")

        for item in content_items:
            content_id = str(item.id)

            if content_id not in emotion_ground_truth:
                continue

            subgraph = self.knowledge_graph.extract_subgraph(
                content_id,
                hops=self.config.k_hops
            )

            if len(subgraph.nodes()) == 0:
                continue

            pyg_data = self.knowledge_graph.to_pytorch_geometric(
                subgraph,
                node_embeddings,
                target_dim=self.config.embedding_dim
            )

            emotion_vector = emotion_ground_truth[content_id]
            emotion_tensor = torch.tensor(emotion_vector, dtype=torch.float)

            subgraph_data.append(pyg_data)
            emotion_targets.append(emotion_tensor)
            content_ids.append(content_id)

        print(f"Prepared {len(subgraph_data)} valid training samples")

        n_total = len(subgraph_data)
        n_train = int(0.9 * n_total)

        indices = np.random.permutation(n_total)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

        train_dataset = SubgraphDataset(
            [subgraph_data[i] for i in train_indices],
            torch.stack([emotion_targets[i] for i in train_indices]),
            [content_ids[i] for i in train_indices]
        )

        val_dataset = SubgraphDataset(
            [subgraph_data[i] for i in val_indices],
            torch.stack([emotion_targets[i] for i in val_indices]),
            [content_ids[i] for i in val_indices]
        )

        return train_dataset, val_dataset

    def prepare_multi_k_training_data(
        self,
        content_items: List,
        node_embeddings: Dict[str, np.ndarray],
        emotion_ground_truth: Dict[str, np.ndarray]
    ) -> Tuple[MultiKSubgraphDataset, MultiKSubgraphDataset]:
        """
        Prepare training and validation datasets with multi-k subgraphs.
        Each content item has subgraphs for k=1, k=2, k=3.
        """
        multi_k_data = {}
        emotion_targets = {}

        print(f"Preparing multi-k training data for {len(content_items)} items...")
        print(f"  k values: {self.config.k_hops_list}")

        skipped_no_emotion = 0
        skipped_missing_k = 0

        for item in content_items:
            content_id = str(item.id)

            if content_id not in emotion_ground_truth:
                skipped_no_emotion += 1
                continue

            k_subgraphs = {}
            for k in self.config.k_hops_list:
                subgraph = self.knowledge_graph.extract_subgraph(content_id, hops=k)
                if len(subgraph.nodes()) > 0:
                    pyg_data = self.knowledge_graph.to_pytorch_geometric(
                        subgraph,
                        node_embeddings,
                        target_dim=self.config.embedding_dim
                    )
                    k_subgraphs[k] = pyg_data

            if len(k_subgraphs) == len(self.config.k_hops_list):
                multi_k_data[content_id] = k_subgraphs
                emotion_targets[content_id] = torch.tensor(
                    emotion_ground_truth[content_id], dtype=torch.float
                )
            else:
                skipped_missing_k += 1

        print(f"  Items with all k-hop subgraphs: {len(multi_k_data)}")
        if skipped_no_emotion > 0:
            print(f"  Skipped (no emotion ground truth): {skipped_no_emotion}")
        if skipped_missing_k > 0:
            print(f"  Skipped (missing k-hop subgraphs): {skipped_missing_k}")

        if len(multi_k_data) == 0:
            raise ValueError(
                "No valid training samples found. Check emotion ground truth and knowledge graph."
            )

        content_ids = list(multi_k_data.keys())
        n_total = len(content_ids)
        n_train = int(0.9 * n_total)

        indices = np.random.permutation(n_total)
        train_ids = [content_ids[i] for i in indices[:n_train]]
        val_ids = [content_ids[i] for i in indices[n_train:]]

        train_multi_k = {cid: multi_k_data[cid] for cid in train_ids}
        train_targets = {cid: emotion_targets[cid] for cid in train_ids}

        val_multi_k = {cid: multi_k_data[cid] for cid in val_ids}
        val_targets = {cid: emotion_targets[cid] for cid in val_ids}

        train_dataset = MultiKSubgraphDataset(train_multi_k, train_targets, self.config.k_hops_list)
        val_dataset = MultiKSubgraphDataset(val_multi_k, val_targets, self.config.k_hops_list)

        return train_dataset, val_dataset

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Run one training epoch"""
        self.gnn_encoder_indexing.train()
        self.emotion_projector.train()

        total_loss = 0.0
        num_batches = 0

        for batch_graphs, batch_targets, _ in train_loader:
            batch_graphs = batch_graphs.to(self.device)
            batch_targets = batch_targets.to(self.device)

            self.optimizer.zero_grad()

            graph_embeddings = self._encode_batch(batch_graphs)
            predicted_emotions = self.emotion_projector(graph_embeddings)

            loss = self.criterion(predicted_emotions, batch_targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.gnn_encoder_indexing.parameters()) +
                list(self.emotion_projector.parameters()),
                max_norm=1.0
            )
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def train_epoch_multi_k(self, train_loader: DataLoader) -> Tuple[float, float, float]:
        """Run one training epoch with multi-k contrastive + alignment learning."""
        self.gnn_encoder_indexing.train()
        self.emotion_projector.train()

        total_mse_loss = 0.0
        total_contrastive_loss = 0.0
        total_alignment_loss = 0.0
        num_batches = 0
        skipped_batches = 0

        for batch_data in train_loader:
            self.optimizer.zero_grad()

            embeddings_by_k = {}
            for k in self.config.k_hops_list:
                if k in batch_data:
                    batch_graphs_k = batch_data[k].to(self.device)
                    embeddings_by_k[k] = self._encode_batch(batch_graphs_k)

            if len(embeddings_by_k) < len(self.config.k_hops_list):
                skipped_batches += 1
                continue

            primary_k = self.config.k_hops
            graph_embeddings = embeddings_by_k[primary_k]
            predicted_emotions = self.emotion_projector(graph_embeddings)
            batch_targets = batch_data['targets'].to(self.device)

            mse_loss = self.criterion(predicted_emotions, batch_targets)

            c_loss = self.contrastive_loss(
                embeddings_by_k[1],
                embeddings_by_k[2],
                embeddings_by_k[3]
            )

            total_loss = mse_loss + self.config.contrastive_weight * c_loss

            semantic_targets = self._get_semantic_targets(batch_data['content_ids'])
            if semantic_targets is not None:
                align_loss = (1 - F.cosine_similarity(graph_embeddings, semantic_targets, dim=-1)).mean()
                total_loss = total_loss + self.config.alignment_weight * align_loss
                total_alignment_loss += align_loss.item()

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.gnn_encoder_indexing.parameters()) +
                list(self.emotion_projector.parameters()),
                max_norm=1.0
            )
            self.optimizer.step()

            total_mse_loss += mse_loss.item()
            total_contrastive_loss += c_loss.item()
            num_batches += 1

        if num_batches == 0:
            raise RuntimeError(
                f"No batches processed in epoch. {skipped_batches} batches skipped. "
                "Check that all k-hop subgraphs exist for training items."
            )

        if skipped_batches > 0:
            print(f"  Warning: {skipped_batches} batches skipped (missing k-hop data)")

        avg_mse = total_mse_loss / num_batches
        avg_contrastive = total_contrastive_loss / num_batches
        avg_alignment = total_alignment_loss / num_batches
        return avg_mse, avg_contrastive, avg_alignment

    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Run validation"""
        self.gnn_encoder_indexing.eval()
        self.emotion_projector.eval()

        total_loss = 0.0
        total_cosine_sim = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_graphs, batch_targets, _ in val_loader:
                batch_graphs = batch_graphs.to(self.device)
                batch_targets = batch_targets.to(self.device)

                graph_embeddings = self._encode_batch(batch_graphs)
                predicted_emotions = self.emotion_projector(graph_embeddings)

                loss = self.criterion(predicted_emotions, batch_targets)
                total_loss += loss.item()

                cosine_sim = nn.functional.cosine_similarity(
                    predicted_emotions, batch_targets, dim=1
                ).mean()
                total_cosine_sim += cosine_sim.item()

                num_batches += 1

        if num_batches == 0:
            raise RuntimeError("validate: zero batches processed")

        avg_loss = total_loss / num_batches
        avg_cosine = total_cosine_sim / num_batches

        return avg_loss, avg_cosine

    def _encode_batch(self, batch_graphs: Batch) -> torch.Tensor:
        """Encode a batch of graphs and return per-graph embeddings"""
        x, edge_index = batch_graphs.x, batch_graphs.edge_index
        edge_attr = getattr(batch_graphs, 'edge_attr', None)
        batch = batch_graphs.batch

        node_embeddings = self.gnn_encoder_indexing.input_proj(x)

        # Handle case with no edges
        if edge_index.size(1) == 0:
            node_embeddings = self.gnn_encoder_indexing.layer_norm(node_embeddings)
            from torch_geometric.nn import global_mean_pool
            return global_mean_pool(node_embeddings, batch)

        # Process edge features
        edge_features = self.gnn_encoder_indexing._process_edge_features(edge_attr)

        if edge_features is None:
            raise RuntimeError(
                f"Edge feature processing failed. edge_attr shape: {edge_attr.shape if edge_attr is not None else None}, "
                f"edge_index shape: {edge_index.shape}. Check knowledge graph edge construction."
            )

        for i, layer in enumerate(self.gnn_encoder_indexing.layers):
            node_embeddings = layer(node_embeddings, edge_index, edge_attr=edge_features)
            if i < len(self.gnn_encoder_indexing.layers) - 1:
                node_embeddings = nn.functional.gelu(node_embeddings)
                node_embeddings = self.gnn_encoder_indexing.dropout(node_embeddings)

        node_embeddings = self.gnn_encoder_indexing.layer_norm(node_embeddings)

        from torch_geometric.nn import global_mean_pool
        graph_embeddings = global_mean_pool(node_embeddings, batch)

        return graph_embeddings

    def train(
        self,
        train_dataset: SubgraphDataset,
        val_dataset: SubgraphDataset
    ) -> Dict:
        """
        Run full training loop.

        Returns:
            Dict with training history and best metrics
        """
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_subgraphs
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collate_subgraphs
        )

        print(f"\nStarting GNN training...")
        print(f"  Training samples: {len(train_dataset)}")
        print(f"  Validation samples: {len(val_dataset)}")
        print(f"  Device: {self.device}")
        print(f"  Epochs: {self.config.num_epochs}")
        print(f"  Batch size: {self.config.batch_size}")
        print()

        for epoch in range(self.config.num_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_cosine = self.validate(val_loader)

            self.scheduler.step(val_loss)

            self.training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_cosine': val_cosine
            })

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.epochs_without_improvement = 0
                self.save_checkpoint('best_model.pt')
                marker = '*'
            else:
                self.epochs_without_improvement += 1
                marker = ''

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1:3d} | Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | Val Cosine: {val_cosine:.4f} {marker}")

            if self.epochs_without_improvement >= self.config.patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

        self.load_checkpoint('best_model.pt')

        return {
            'best_val_loss': self.best_loss,
            'final_epoch': len(self.training_history),
            'history': self.training_history
        }

    def train_multi_k(
        self,
        train_dataset: MultiKSubgraphDataset,
        val_dataset: MultiKSubgraphDataset
    ) -> Dict:
        """
        Run full training loop with multi-k contrastive learning.
        """
        from functools import partial

        collate_fn = partial(collate_multi_k, k_list=self.config.k_hops_list)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )

        print(f"\nStarting multi-k GNN training with contrastive + alignment loss...")
        print(f"  Training samples: {len(train_dataset)}")
        print(f"  Validation samples: {len(val_dataset)}")
        print(f"  k values: {self.config.k_hops_list}")
        print(f"  Contrastive weight: {self.config.contrastive_weight}")
        print(f"  Alignment weight: {self.config.alignment_weight}")
        print(f"  Device: {self.device}")
        print(f"  Epochs: {self.config.num_epochs}")
        print()

        for epoch in range(self.config.num_epochs):
            mse_loss, contrastive_loss, alignment_loss = self.train_epoch_multi_k(train_loader)
            val_loss, val_cosine, val_alignment = self.validate_multi_k(val_loader)

            self.scheduler.step(val_loss)

            combined_train_loss = (
                mse_loss
                + self.config.contrastive_weight * contrastive_loss
                + self.config.alignment_weight * alignment_loss
            )

            self.training_history.append({
                'epoch': epoch + 1,
                'mse_loss': mse_loss,
                'contrastive_loss': contrastive_loss,
                'alignment_loss': alignment_loss,
                'train_loss': combined_train_loss,
                'val_loss': val_loss,
                'val_cosine': val_cosine,
                'val_alignment': val_alignment
            })

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.epochs_without_improvement = 0
                self.save_checkpoint('best_model.pt')
                marker = '*'
            else:
                self.epochs_without_improvement += 1
                marker = ''

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1:3d} | MSE: {mse_loss:.4f} | Contr: {contrastive_loss:.4f} | "
                      f"Align: {alignment_loss:.4f} | Val: {val_loss:.4f} | Cos: {val_cosine:.4f} {marker}")

            if self.epochs_without_improvement >= self.config.patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

        self.load_checkpoint('best_model.pt')

        return {
            'best_val_loss': self.best_loss,
            'final_epoch': len(self.training_history),
            'history': self.training_history
        }

    def validate_multi_k(self, val_loader: DataLoader) -> Tuple[float, float, float]:
        """Run validation on multi-k data (uses k=2 for MSE validation)."""
        self.gnn_encoder_indexing.eval()
        self.emotion_projector.eval()

        total_loss = 0.0
        total_cosine_sim = 0.0
        total_alignment_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_data in val_loader:
                primary_k = self.config.k_hops
                if primary_k not in batch_data:
                    continue

                batch_graphs = batch_data[primary_k].to(self.device)
                batch_targets = batch_data['targets'].to(self.device)

                graph_embeddings = self._encode_batch(batch_graphs)
                predicted_emotions = self.emotion_projector(graph_embeddings)

                loss = self.criterion(predicted_emotions, batch_targets)
                total_loss += loss.item()

                cosine_sim = nn.functional.cosine_similarity(
                    predicted_emotions, batch_targets, dim=1
                ).mean()
                total_cosine_sim += cosine_sim.item()

                semantic_targets = self._get_semantic_targets(batch_data['content_ids'])
                if semantic_targets is not None:
                    align_loss = (1 - F.cosine_similarity(graph_embeddings, semantic_targets, dim=-1)).mean()
                    total_alignment_loss += align_loss.item()

                num_batches += 1

        if num_batches == 0:
            raise RuntimeError("validate_multi_k: zero batches processed")

        avg_loss = total_loss / num_batches
        avg_cosine = total_cosine_sim / num_batches
        avg_alignment = total_alignment_loss / num_batches

        return avg_loss, avg_cosine, avg_alignment

    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'gnn_encoder_state': self.gnn_encoder.state_dict(),
            'emotion_projector_state': self.emotion_projector.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'config': self.config.__dict__,
            'best_loss': self.best_loss
        }
        torch.save(checkpoint, self.checkpoint_dir / filename)

    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        checkpoint_path = self.checkpoint_dir / filename
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.gnn_encoder.load_state_dict(checkpoint['gnn_encoder_state'])
            self.emotion_projector.load_state_dict(checkpoint['emotion_projector_state'])
            self.best_loss = checkpoint.get('best_loss', float('inf'))
            print(f"Loaded checkpoint from {checkpoint_path}")

    def extract_smoothed_emotions(
        self,
        content_items: List,
        node_embeddings: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Extract graph-smoothed emotion vectors for all content items.
        Used after training to populate the retriever's smoothed_emotions dict.

        Returns:
            Dict mapping content_id -> 6-dim smoothed emotion vector
        """
        self.gnn_encoder_indexing.eval()
        self.emotion_projector.eval()

        smoothed_emotions = {}

        with torch.no_grad():
            for item in content_items:
                content_id = str(item.id)

                subgraph = self.knowledge_graph.extract_subgraph(
                    content_id,
                    hops=self.config.k_hops
                )

                if len(subgraph.nodes()) == 0:
                    continue

                pyg_data = self.knowledge_graph.to_pytorch_geometric(
                    subgraph,
                    node_embeddings,
                    target_dim=self.config.embedding_dim
                )

                pyg_data = pyg_data.to(self.device)

                graph_embedding = self.gnn_encoder.index_subgraph(pyg_data)
                emotion_vector = self.emotion_projector(graph_embedding.unsqueeze(0))

                smoothed_emotions[content_id] = emotion_vector.squeeze(0).cpu().numpy()

        return smoothed_emotions

    def save_smoothed_emotions(
        self,
        smoothed_emotions: Dict[str, np.ndarray],
        output_path: str
    ):
        """Save smoothed emotions to disk"""
        np.savez(
            output_path,
            content_ids=list(smoothed_emotions.keys()),
            emotions=np.array(list(smoothed_emotions.values()))
        )
        print(f"Saved smoothed emotions to {output_path}")


def prepare_emotion_ground_truth(
    movies_df,
    emotion_columns: List[str] = None
) -> Dict[str, np.ndarray]:
    """
    Extract ground truth emotion vectors from movies DataFrame.

    Args:
        movies_df: DataFrame with movie data including emotion scores
        emotion_columns: List of column names for emotion scores
                        Default: happiness_score, sadness_score, etc.

    Returns:
        Dict mapping movie_id -> 6-dim emotion vector
    """
    if emotion_columns is None:
        emotion_columns = [
            'happiness_score', 'sadness_score', 'anger_score',
            'fear_score', 'surprise_score', 'disgust_score'
        ]

    missing_cols = [c for c in emotion_columns if c not in movies_df.columns]
    if missing_cols:
        raise ValueError(f"Missing emotion columns in DataFrame: {missing_cols}")

    emotion_ground_truth = {}
    nan_counts = {col: 0 for col in emotion_columns}
    out_of_range_counts = {col: 0 for col in emotion_columns}

    for _, row in movies_df.iterrows():
        movie_id = str(row['movieId'])

        emotion_values = []
        for col in emotion_columns:
            val = row[col]
            if np.isnan(val):
                nan_counts[col] += 1
                emotion_values.append(0.0)
            else:
                if val < 0 or val > 1:
                    out_of_range_counts[col] += 1
                    val = 1 / (1 + np.exp(-val))
                emotion_values.append(val)

        emotion_ground_truth[movie_id] = np.array(emotion_values, dtype=np.float32)

    total_nan = sum(nan_counts.values())
    total_out_of_range = sum(out_of_range_counts.values())

    if total_nan > 0:
        print(f"  Warning: {total_nan} NaN values imputed as 0.0 in emotion ground truth")
    if total_out_of_range > 0:
        print(f"  Warning: {total_out_of_range} out-of-range values normalized via sigmoid")

    return emotion_ground_truth
