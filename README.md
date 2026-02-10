# ARAG (Affective-RAG)

## Status

This repository is a **research codebase** accompanying the Affective-RAG paper. Experiment notebooks are provided under `notebooks/` for full reproducibility.

## Overview

ARAG explores emotion-aware retrieval-augmented generation for recommendations by combining:

- **Semantic retrieval** (text embeddings)
- **Affective alignment** (emotion profile similarity)
- **Graph-based structure** (a heterogeneous knowledge graph and subgraph encoders)
- **Optional LLM response generation** (Vertex AI-backed in the current implementation)

The goal is to support recommendations that are not only relevant, but also explainable in terms of *why* a given item matches a user’s stated emotional intent.

## Problem statement (research gaps)

Two gaps motivate this work:

- **Affective gap**: Existing affective recommenders frequently treat emotions as static metadata or weak implicit signals. They often do not model emotional preferences as a structured, multi-dimensional space with mechanisms to reason about *why* certain content satisfies an affective intent (e.g., “catharsis”).
- **Explainability gap**: Many recommenders remain opaque. Vector retrieval can produce mathematically similar matches without providing a structured explanation of semantic validity or how emotional inputs affected the ranking, which undermines auditability and user trust.

## Research gap

The intersection remains underexplored:

- Affective systems often lack **structural reasoning** (e.g., knowledge-graph traversal) and treat emotions as isolated features.
- GraphRAG-style systems often focus on factual/semantic entities while ignoring the **affective dimension** common in entertainment consumption.
- Existing explainable AI approaches (path-based and counterfactual explanations) are typically not integrated into a single emotion-aware graph retrieval framework.

ARAG treats emotions as first-class graph entities and targets multi-hop affective reasoning that is explanation-oriented and compatible with counterfactual analysis.

## Contributions (project scope)

This project is organized around the following contributions:

- **Heterogeneous affective knowledge graph**: A schema that models content, genres, and emotions with explicit relationships suitable for inspection and explanation.
- **Graph Transformer-based affective encoding**: Subgraph encoding using graph transformer components to capture local affective topology beyond pure text similarity.
- **Explainable retrieval and counterfactual readiness**: An interpretable scoring pipeline with decomposable terms and hooks for causal/counterfactual analysis.
- **Evaluation beyond accuracy**: Retrieval metrics (Precision/Recall/NDCG/MRR/MAP) alongside affective and interpretability-oriented metrics (e.g., affective coherence and displacement-style measures).

## Method summary

At a high level, the pipeline is:

1. **Ingest** content and user-related data (external; not included in this repo).
2. **Build** a heterogeneous knowledge graph (content/genre/emotion and optional user-related relations).
3. **Embed**:
   - content semantics (text)
   - affective signals (emotion-conditioned representations)
   - graph substructures (subgraph embeddings)
4. **Retrieve and rank** candidates using a decomposable scoring function.
5. **Explain** recommendations via score breakdowns and graph context (and optionally generate natural language responses).

### Scoring

The core retrieval logic uses a nested score of the form:

`Score = α * [λ * Semantic + (1-λ) * Graph] - (1-α) * AffectiveRMSE`

where **AffectiveRMSE** is a normalized L2 distance between the user's target emotion vector and the item's affective signature.

Implementation reference: `src/krag/retrieval/krag_retriever.py`.

## Repository layout

- `notebooks/`: Colab notebooks for reproducing all experiments (see below)
- `colab_generate_embeddings.py`: GPU-oriented embedding generation script for Colab
- `src/krag/`: main library code
  - `system.py`: end-to-end orchestration (`ARAGSystem`)
  - `data/`: ingestion, adapters, and expected schema (`DatasetPath`, column specs)
  - `core/`: embeddings, emotion processing, knowledge graph, graph encoders
  - `retrieval/`: retrievers and fusion/scoring logic
  - `storage/`: vector store (ChromaDB) and indexing utilities
  - `evaluation/`: metric definitions and evaluators
  - `experiments/`: reproducibility helpers (seed control, experiment IDs)
  - `training/`: GNN encoder training (MSE + contrastive + alignment losses)
  - `llm/`: response generation (optional, Vertex AI-based)

## Data

The dataset will be made publicly available. Details and download links will be added here once hosting is finalized.

The emotion labels used in this project were inspired by the methodology in [Emotion Recognition in Movie Abstracts](https://github.com/dimi-fn/Emotion-Recognition-in-Movie-Abstracts), which generates multi-label emotion annotations for movie content.

- **Paths**: dataset paths are defined in `src/krag/data/adapters.py` (`DatasetPath`) and can be adjusted by changing that mapping or by using a different adapter/config.
- **Schemas**: expected column names and formats are defined in `src/krag/data/schema.py`.

## Configuration

The code reads configuration primarily from environment variables (and will also attempt to load them via `python-dotenv` if present).

Common variables:

- `GCS_BUCKET`: bucket name used by the storage adapter
- `GCS_BASE_PATH`: base prefix for dataset paths (defaults to `Dataset`)
- `GOOGLE_CLOUD_PROJECT`: project id used for optional Vertex AI integration
- `GOOGLE_APPLICATION_CREDENTIALS`: credentials path if running with explicit service account credentials (optional; environment-specific)

## Setup

### Requirements

- Python 3.10+ recommended
- Dependencies in `requirements.txt`

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducing results

All experiments run on Google Colab with a GPU runtime. Each notebook clones this repo, installs dependencies, and loads data from GCS.

### Execution order

1. **`notebooks/01_train_gnn_encoder.ipynb`** -- Trains the GNN encoder with MSE + contrastive + alignment losses. Sweeps alignment weight over {0.05, 0.1, 0.3, 0.5}. Downloads encoder checkpoints.
2. Upload the best encoder checkpoint to your GCS bucket under `models/krag_encoder.pt`.
3. The following two notebooks can run **in parallel** (independent runtimes):
   - **`notebooks/02_k_sweep_evaluation.ipynb`** -- Evaluates structural sensitivity across hop depths k=1..5.
   - **`notebooks/03_comparative_analysis.ipynb`** -- Compares BM25, Vector-RAG, and KRAG at three alpha values on agreement and dissonance query sets.

### Prerequisites

- A GCP project with a GCS bucket containing the dataset (see Data expectations below)
- GPU runtime in Colab
- Set `PROJECT_ID` in each notebook's auth cell

### Programmatic usage

```python
import os
from krag.system import ARAGSystem, ARAGSystemConfig

os.environ["GCS_BUCKET"] = "your-gcs-bucket"
os.environ["GCS_BASE_PATH"] = "Dataset"

config = ARAGSystemConfig(retriever_type="adaptive_krag")
system = ARAGSystem(config)
system.initialize()
system.load_and_index_data()

result = system.query(
    "uplifting comedy movies",
    emotion_sliders={"happiness": 8, "sadness": 2}
)
print(result["recommendations"][0])
```

### Embedding generation (Colab)

`colab_generate_embeddings.py` generates precomputed embeddings on a GPU environment. Parameterized via environment variables (`GCS_BUCKET`, `GCS_BASE_PATH`).

