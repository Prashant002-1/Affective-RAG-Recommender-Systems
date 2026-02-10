"""
Affective-RAG: Generate ALL Embeddings on GPU (Colab)


WHAT THIS GENERATES:
1. semantic_embeddings - Movie content embeddings (56k × 768)
2. emotion_embeddings - Emotion-augmented embeddings (56k × 768)  
3. node_embeddings - Knowledge graph node embeddings (56k+ × 768)


This script expects input/output locations to be configured via environment
variables (Colab secrets or runtime env). No repository-specific cloud IDs are
hardcoded here.
"""


import torch
import os
print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU!")

BUCKET = os.getenv("GCS_BUCKET", "your-gcs-bucket")
BASE_PATH = os.getenv("GCS_BASE_PATH", "Dataset")

MOVIES_CSV_PATH = os.getenv("MOVIES_CSV_PATH", f"{BASE_PATH}/movies.csv")
NODES_EMOTIONS_CSV_PATH = os.getenv("NODES_EMOTIONS_CSV_PATH", f"{BASE_PATH}/graph/nodes/emotions.csv")
NODES_GENRES_CSV_PATH = os.getenv("NODES_GENRES_CSV_PATH", f"{BASE_PATH}/graph/nodes/genres.csv")
USER_RATED_MOVIE_CSV_PATH = os.getenv("USER_RATED_MOVIE_CSV_PATH", f"{BASE_PATH}/graph/relationships/user_rated_movie.csv")

from google.cloud import storage
import pandas as pd
import numpy as np
import io

print(f"Loading from gs://{BUCKET}/{MOVIES_CSV_PATH}...")
client = storage.Client()
bucket = client.bucket(BUCKET)
blob = bucket.blob(MOVIES_CSV_PATH)
content = blob.download_as_text()
movies_df = pd.read_csv(io.StringIO(content))
print(f"Loaded {len(movies_df)} movies")

def prepare_text(row):
    title = str(row.get('title', ''))
    overview = str(row.get('overview', '') or '')
    genres = str(row.get('genres', '') or '')
    return f"{title}. {overview} Genres: {genres}"

texts = [prepare_text(row) for _, row in movies_df.iterrows()]
print(f"Prepared {len(texts)} texts")

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
if torch.cuda.is_available():
    model = model.to('cuda')
print(f"Model loaded on: {next(model.parameters()).device}")

# === STEP 7: Generate SEMANTIC embeddings ===
print("Generating semantic embeddings...")
semantic_embeddings = model.encode(
    texts, 
    batch_size=64,  # Higher batch size for GPU
    show_progress_bar=True,
    convert_to_numpy=True
)
print(f"Semantic embeddings shape: {semantic_embeddings.shape}")

# Generate EMOTION embeddings ===
EMOTION_COLS = ['happiness_score', 'sadness_score', 'anger_score', 
                'fear_score', 'surprise_score', 'disgust_score']

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def emotions_to_text(row):
    emotions = {}
    for col in EMOTION_COLS:
        if col in row and pd.notna(row[col]):
            name = col.replace('_score', '')
            emotions[name] = sigmoid(float(row[col]))
    
    if not emotions:
        return "neutral emotional tone"
    
    top = sorted(emotions.items(), key=lambda x: -x[1])[:3]
    desc = [f"{name} ({score:.0%})" for name, score in top if score > 0.3]
    return f"Emotions: {', '.join(desc)}" if desc else "neutral emotional tone"

emotion_texts = [f"{texts[i]} {emotions_to_text(row)}" 
                 for i, (_, row) in enumerate(movies_df.iterrows())]

print("Generating emotion embeddings...")
emotion_embeddings = model.encode(
    emotion_texts, 
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True
)
print(f"Emotion embeddings shape: {emotion_embeddings.shape}")

# === Generate KNOWLEDGE GRAPH NODE embeddings ===
print("\n" + "="*50)
print("STEP 9: Generating Knowledge Graph Node Embeddings")
print("="*50)

# Load KG nodes: emotions, genres
print("Loading KG node data from GCS...")

kg_node_texts = {}

# Emotion nodes
try:
    blob = bucket.blob(NODES_EMOTIONS_CSV_PATH)
    emotions_df = pd.read_csv(io.StringIO(blob.download_as_text()))
    for _, row in emotions_df.iterrows():
        node_id = f"emotion_{row.get('emotionId', row.get('name', ''))}"
        text = f"Emotion: {row.get('name', '')}. {row.get('description', '')}"
        kg_node_texts[node_id] = text
    print(f"  Loaded {len(emotions_df)} emotion nodes")
except Exception as e:
    print(f"  Warning: Could not load emotions: {e}")

# Genre nodes
try:
    blob = bucket.blob(NODES_GENRES_CSV_PATH)
    genres_df = pd.read_csv(io.StringIO(blob.download_as_text()))
    for _, row in genres_df.iterrows():
        node_id = f"genre_{row.get('genreId', row.get('name', ''))}"
        text = f"Genre: {row.get('name', '')}. Movies in this genre typically feature {row.get('name', '')} themes."
        kg_node_texts[node_id] = text
    print(f"  Loaded {len(genres_df)} genre nodes")
except Exception as e:
    print(f"  Warning: Could not load genres: {e}")

# Movie nodes 
for i, (_, row) in enumerate(movies_df.iterrows()):
    movie_id = str(row.get('movieId', i))
    kg_node_texts[movie_id] = texts[i]
print(f"  Added {len(movies_df)} movie nodes")

print(f"\nTotal KG nodes: {len(kg_node_texts)}")

# Generate embeddings for all KG nodes
print("Generating KG node embeddings...")
node_ids = list(kg_node_texts.keys())
node_texts_list = [kg_node_texts[nid] for nid in node_ids]

node_embeddings_array = model.encode(
    node_texts_list,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True
)
print(f"Node embeddings shape: {node_embeddings_array.shape}")

# Create a mapping dict for saving
node_embeddings_dict = {nid: node_embeddings_array[i] for i, nid in enumerate(node_ids)}

print("\n" + "="*50)
print("STEP 10: Saving All Embeddings")
print("="*50)

import pickle

# Save main embeddings as npz (arrays)
print("Saving semantic and emotion embeddings...")
np.savez('embeddings.npz', 
         semantic=semantic_embeddings, 
         emotion=emotion_embeddings)

# Save node embeddings separately (dict format)
print("Saving node embeddings...")
with open('node_embeddings.pkl', 'wb') as f:
    pickle.dump(node_embeddings_dict, f)

# Verify
loaded = np.load('embeddings.npz')
print(f"✓ Semantic: {loaded['semantic'].shape}")
print(f"✓ Emotion: {loaded['emotion'].shape}")
print(f"✓ Node embeddings: {len(node_embeddings_dict)} nodes")

print("\n" + "="*50)
print("STEP 11: Generating User-Movie Mapping")
print("="*50)

print("Loading user ratings from GCS...")
ratings_blob = bucket.blob(USER_RATED_MOVIE_CSV_PATH)
ratings_df = pd.read_csv(
    io.StringIO(ratings_blob.download_as_text()),
    usecols=[':START_ID', ':END_ID'],
    dtype={':START_ID': str, ':END_ID': str}
)
ratings_df = ratings_df.rename(columns={':START_ID': 'userId', ':END_ID': 'movieId'})
print(f"  Loaded {len(ratings_df):,} ratings")

print("Building user-movie mapping...")
user_movie_mapping = ratings_df.groupby('userId')['movieId'].apply(set).to_dict()
print(f"  Created mapping for {len(user_movie_mapping):,} users")

print("Saving user-movie mapping...")
with open('user_movie_mapping.pkl', 'wb') as f:
    pickle.dump(user_movie_mapping, f)
print("✓ Saved user_movie_mapping.pkl")

del ratings_df  # Free memory

# === Upload to GCS ===
print("\n" + "="*50)
print("STEP 12: Uploading to GCS")
print("="*50)

PRECOMPUTED_PREFIX = os.getenv("PRECOMPUTED_PREFIX", f"{BASE_PATH}/precomputed")
GCS_PATH_EMB = os.getenv("EMBEDDINGS_NPZ_PATH", f"{PRECOMPUTED_PREFIX}/embeddings_v1.npz")
GCS_PATH_NODES = os.getenv("NODE_EMBEDDINGS_PKL_PATH", f"{PRECOMPUTED_PREFIX}/node_embeddings_v1.pkl")
GCS_PATH_MAPPING = os.getenv("USER_MOVIE_MAPPING_PKL_PATH", f"{PRECOMPUTED_PREFIX}/user_movie_mapping_v1.pkl")

print(f"Uploading embeddings to gs://{BUCKET}/{GCS_PATH_EMB}...")
upload_blob = bucket.blob(GCS_PATH_EMB)
upload_blob.upload_from_filename('embeddings.npz')
print(f"✓ Uploaded embeddings")

print(f"Uploading node embeddings to gs://{BUCKET}/{GCS_PATH_NODES}...")
upload_blob = bucket.blob(GCS_PATH_NODES)
upload_blob.upload_from_filename('node_embeddings.pkl')
print(f"✓ Uploaded node embeddings")

print(f"Uploading user-movie mapping to gs://{BUCKET}/{GCS_PATH_MAPPING}...")
upload_blob = bucket.blob(GCS_PATH_MAPPING)
upload_blob.upload_from_filename('user_movie_mapping.pkl')
print(f"✓ Uploaded user-movie mapping")

print("\n" + "="*60)
print("DATA GENERATED AND UPLOADED!")
print("="*60)
print(f"\nGCS Locations:")
print(f"  gs://{BUCKET}/{GCS_PATH_EMB}")
print(f"  gs://{BUCKET}/{GCS_PATH_NODES}")
print(f"  gs://{BUCKET}/{GCS_PATH_MAPPING}")
print(f"\nContents:")
print(f"  - Semantic embeddings: {semantic_embeddings.shape}")
print(f"  - Emotion embeddings: {emotion_embeddings.shape}")
print(f"  - Node embeddings: {len(node_embeddings_dict)} nodes")
print(f"  - User-movie mapping: {len(user_movie_mapping):,} users")
