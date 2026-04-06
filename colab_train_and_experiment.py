# %% [markdown]
# # Affective-RAG: GNN Training & Experiments
# Run this notebook on Colab with GPU runtime.
#
# This notebook includes:
# 1. System initialization and data loading
# 2. GNN training (optional)
# 3. **Threshold sensitivity analysis** - Find optimal thresholds for your dataset
# 4. **Dual sample-size experiments** - Run at 50 and 100+ queries for statistical validity

# %% [markdown]
# ## Configuration

# %%
FULL_DATASET = True
MAX_ITEMS = None if FULL_DATASET else 500
NUM_EPOCHS = 100 if FULL_DATASET else 50
BATCH_SIZE = 64 if FULL_DATASET else 32

# Experiment configuration
NUM_QUERIES_LOW = 50     # Lower sample size run
NUM_QUERIES_HIGH = 150   # Higher sample size run (for statistical power)
MIN_TEST_CASES = 30      # Minimum required for valid results

print(f"Mode: {'FULL DATASET' if FULL_DATASET else 'SUBSET (500 items)'}")
print(f"Training epochs: {NUM_EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Experiment queries: {NUM_QUERIES_LOW} (low) / {NUM_QUERIES_HIGH} (high)")
print(f"Min test cases required: {MIN_TEST_CASES}")

# %% [markdown]
# ## Step 1: Clone Repository & Install Dependencies

# %%
import os

REPO_URL = os.environ.get("KRAG_REPO_URL", "https://github.com/<owner>/<repo>.git")
REPO_DIR = "KRAG"
!git clone {REPO_URL} {REPO_DIR}
%cd {REPO_DIR}

# %% [markdown]
# ### Fix Google Cloud AI Platform Version Conflict
# This is required to resolve the `aiplatform.models` AttributeError.
# We upgrade packages to sync versions instead of downgrading.

# %%
# Upgrade google-cloud-aiplatform and vertexai to latest compatible versions
# This keeps them in sync and avoids the models attribute error
!pip install -q --upgrade google-cloud-aiplatform vertexai

# Install langchain packages that work with the current aiplatform version
!pip install -q langchain-google-vertexai langchain-core --upgrade

# Install remaining dependencies
!pip install -q torch torch-geometric sentence-transformers google-cloud-storage python-dotenv networkx chromadb rank_bm25 matplotlib scipy


# %% [markdown]
# ## Step 2: Authenticate with Google Cloud

# %%
from google.colab import auth
auth.authenticate_user()

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "YOUR_PROJECT_ID")
!gcloud config set project {PROJECT_ID}

# %% [markdown]
# ## Step 3: Verify Package Versions
# Run this to confirm the version conflict is resolved

# %%
import importlib
import pkg_resources

packages_to_check = ['google-cloud-aiplatform', 'vertexai', 'langchain-google-vertexai']
print("Package versions:")
for pkg in packages_to_check:
    try:
        version = pkg_resources.get_distribution(pkg).version
        print(f"  OK {pkg}: {version}")
    except pkg_resources.DistributionNotFound:
        print(f"  MISSING {pkg}: NOT INSTALLED")

# Quick sanity check - test the actual import that was failing
print("\nTesting imports...")
try:
    # This is the exact import chain that was failing
    from langchain_google_vertexai import VertexAI
    print("langchain_google_vertexai imports successfully.")
except AttributeError as e:
    if "models" in str(e):
        print("Still getting aiplatform.models error.")
        print("Restart the runtime, then skip Step 1.")
    else:
        print(f"AttributeError: {e}")
except Exception as e:
    print(f"Import error: {type(e).__name__}: {e}")
    print("Restart the runtime, then skip Step 1.")

# %% [markdown]
# ## Step 4: Initialize System & Load Data

# %%
import sys
sys.path.insert(0, 'src')

from krag.system import ARAGSystem, ARAGSystemConfig

config = ARAGSystemConfig(
    max_content_items=MAX_ITEMS,
    vertex_ai_project=PROJECT_ID
)

system = ARAGSystem(config)
system.initialize()
system.load_and_index_data()

print(f"\nLoaded {len(system.content_items)} movies")
print(f"KG nodes: {len(system.knowledge_graph.graph.nodes())}")
print(f"KG edges: {len(system.knowledge_graph.graph.edges())}")

# Verify EVOKES edges exist (critical for affective metrics!)
evokes_count = sum(1 for u, v, d in system.knowledge_graph.graph.edges(data=True) 
                   if d.get('relation') == 'evokes')
print(f"EVOKES edges: {evokes_count}")
if evokes_count == 0:
    print("Warning: No EVOKES edges found. Affective metrics will be invalid.")
    print("   Check that movie_emotion.csv exists in GCS neo4j_relationships/")

# %% [markdown]
# ## Step 5: Train GNN (Self-Supervised Denoising)
# **Optional**: Skip this if you have pre-trained weights

# %%
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

training_results = system.train_gnn(
    num_epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=1e-4,
    patience=15 if FULL_DATASET else 10
)

print(f"\nTraining complete!")
print(f"Best validation loss: {training_results['best_val_loss']:.4f}")
print(f"Final epoch: {training_results['final_epoch']}")

# %% [markdown]
# ## Step 6: Save Trained Model

# %%
import os
from pathlib import Path

models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

model_name = "gnn_full_dataset.pt" if FULL_DATASET else "gnn_subset.pt"
checkpoint_path = Path("checkpoints/best_model.pt")

if checkpoint_path.exists():
    import shutil
    shutil.copy(checkpoint_path, models_dir / model_name)
    print(f"Model saved to: models/{model_name}")
else:
    print("Warning: Checkpoint not found")

# %% [markdown]
# ## Step 7: Threshold Sensitivity Analysis
# **IMPORTANT**: Run this FIRST to find optimal thresholds for your dataset.
# This creates a heatmap showing which threshold combinations produce enough test cases.

# %%
max_items_arg = "" if MAX_ITEMS is None else f"--max-items {MAX_ITEMS}"

print("=" * 60)
print("STEP 7: THRESHOLD SENSITIVITY ANALYSIS")
print("=" * 60)
print("Finding optimal semantic/affective thresholds for your dataset...")
print()

!python run_experiments.py --experiment threshold-sensitivity {max_items_arg} --num-queries {NUM_QUERIES_HIGH} --output-dir ./results

# %% [markdown]
# ### View Threshold Analysis Results

# %%
import json
from pathlib import Path

threshold_file = Path("results/threshold_analysis.json")
if threshold_file.exists():
    with open(threshold_file) as f:
        threshold_data = json.load(f)
    
    print("=" * 60)
    print("THRESHOLD SENSITIVITY RESULTS")
    print("=" * 60)
    
    # Show recommended thresholds
    rec = threshold_data.get("recommended_threshold", {})
    print(f"\nRECOMMENDED THRESHOLDS:")
    print(f"  Semantic: {rec.get('semantic', 'N/A')}")
    print(f"  Affective: {rec.get('affective', 'N/A')}")
    print(f"  Expected test cases: {rec.get('expected_test_cases', 'N/A')}")
    print(f"  Reason: {rec.get('reason', 'N/A')}")
    
    # Summary
    summary = threshold_data.get("summary", {})
    print(f"\nSummary:")
    print(f"  Total combinations tested: {summary.get('total_combinations_tested', 'N/A')}")
    print(f"  Valid combinations (>= {MIN_TEST_CASES} cases): {summary.get('valid_combinations', 'N/A')}")
    print(f"  Max test cases achieved: {summary.get('max_test_cases', 'N/A')}")
    
    # Store recommended thresholds for use in experiments
    RECOMMENDED_SEM_THRESHOLD = rec.get('semantic', 0.5)
    RECOMMENDED_AFF_THRESHOLD = rec.get('affective', 0.6)
    
    print(f"\nUsing thresholds: semantic={RECOMMENDED_SEM_THRESHOLD}, affective={RECOMMENDED_AFF_THRESHOLD}")
else:
    print("Threshold analysis file not found. Using defaults.")
    RECOMMENDED_SEM_THRESHOLD = 0.5
    RECOMMENDED_AFF_THRESHOLD = 0.6

# Display heatmap if available
heatmap_file = Path("results/threshold_heatmap.png")
if heatmap_file.exists():
    from IPython.display import Image, display
    print("\nThreshold Heatmap (red box = recommended):")
    display(Image(filename=str(heatmap_file)))

# %% [markdown]
# ## Step 8: Run Comparative Experiments (Dual Sample Sizes)
# Run experiments at both 50 and 100+ queries to demonstrate statistical robustness.

# %% [markdown]
# ### 8a. Low Sample Size Run (50 queries)

# %%
print("=" * 60)
print(f"COMPARATIVE ANALYSIS: {NUM_QUERIES_LOW} QUERIES")
print("=" * 60)

!python run_experiments.py --experiment comparative \
    {max_items_arg} \
    --num-queries {NUM_QUERIES_LOW} \
    --semantic-threshold {RECOMMENDED_SEM_THRESHOLD} \
    --affective-threshold {RECOMMENDED_AFF_THRESHOLD} \
    --min-test-cases {MIN_TEST_CASES} \
    --output-suffix _t{NUM_QUERIES_LOW} \
    --output-dir ./results

# %% [markdown]
# ### 8b. High Sample Size Run (100+ queries)

# %%
print("=" * 60)
print(f"COMPARATIVE ANALYSIS: {NUM_QUERIES_HIGH} QUERIES")
print("=" * 60)

!python run_experiments.py --experiment comparative \
    {max_items_arg} \
    --num-queries {NUM_QUERIES_HIGH} \
    --semantic-threshold {RECOMMENDED_SEM_THRESHOLD} \
    --affective-threshold {RECOMMENDED_AFF_THRESHOLD} \
    --min-test-cases {MIN_TEST_CASES} \
    --output-suffix _t{NUM_QUERIES_HIGH} \
    --output-dir ./results

# %% [markdown]
# ## Step 9: Run Other Experiments

# %%
# Experiment IV.C: Structural Sensitivity (k-hop ablation)
print("=" * 60)
print("STRUCTURAL SENSITIVITY (k-hop)")
print("=" * 60)
!python run_experiments.py --experiment sensitivity {max_items_arg} --num-queries {NUM_QUERIES_HIGH}

# %%
# Experiment IV.D: Causal Necessity Analysis
print("=" * 60)
print("CAUSAL NECESSITY ANALYSIS")
print("=" * 60)
!python run_experiments.py --experiment causal {max_items_arg} --num-queries {NUM_QUERIES_HIGH}

# %% [markdown]
# ## Step 10: View All Results

# %%
import json
from pathlib import Path

results_dir = Path("results")

print("\n" + "=" * 80)
print("ALL EXPERIMENT RESULTS")
print("=" * 80)

for result_file in sorted(results_dir.glob("*.json")):
    print(f"\n{'─'*80}")
    print(f"FILE: {result_file.name}")
    print('─'*80)
    with open(result_file) as f:
        data = json.load(f)
        
        # Pretty print with key info highlighted
        if "experiment_config" in data:
            config = data["experiment_config"]
            print(f"\nExperiment Config:")
            print(f"   Thresholds: semantic={config.get('semantic_threshold')}, affective={config.get('affective_threshold')}")
            print(f"   Test cases: {config.get('num_test_cases_generated')} (required: {config.get('min_test_cases_required')})")
            print(f"   Status: {config.get('status')}")
            if config.get('fallback_applied'):
                print(f"   Threshold adjustment: {config.get('fallback_reason')}")
        
        if "methods" in data:
            print(f"\nResults:")
            for method, metrics in data["methods"].items():
                print(f"   {method}:")
                print(f"      ADE: {metrics.get('ADE', 'N/A'):.4f}  |  AP@5: {metrics.get('AP@5', 'N/A'):.4f}  |  SR@10: {metrics.get('Semantic_Recall@10', 'N/A'):.4f}")
        
        if "statistical_analysis" in data and "comparison_affective_vs_semantic" in data["statistical_analysis"]:
            comp = data["statistical_analysis"]["comparison_affective_vs_semantic"]
            print(f"\nStatistical Comparison (Affective vs Semantic):")
            print(f"   Effect size (Cohen's d): {comp.get('effect_size_cohens_d', 'N/A')}")
            print(f"   p-value: {comp.get('p_value', 'N/A')}")
            print(f"   Result: {comp.get('interpretation', 'N/A')}")
        
        if "warnings" in data and data["warnings"]:
            print(f"\nWarnings:")
            for w in data["warnings"]:
                print(f"   - {w}")

# %% [markdown]
# ## Step 11: Compare Sample Sizes

# %%
import json
from pathlib import Path

print("=" * 80)
print("SAMPLE SIZE COMPARISON")
print("=" * 80)

results_50 = Path(f"results/comparative_analysis_t{NUM_QUERIES_LOW}.json")
results_150 = Path(f"results/comparative_analysis_t{NUM_QUERIES_HIGH}.json")

if results_50.exists() and results_150.exists():
    with open(results_50) as f:
        data_50 = json.load(f)
    with open(results_150) as f:
        data_150 = json.load(f)
    
    print(f"\n{'Method':<20} {'ADE (n=50)':>12} {'ADE (n=150)':>12} {'Δ':>8}")
    print("-" * 60)
    
    methods_50 = data_50.get("methods", {})
    methods_150 = data_150.get("methods", {})
    
    for method in methods_50:
        ade_50 = methods_50[method].get("ADE", float('inf'))
        ade_150 = methods_150.get(method, {}).get("ADE", float('inf'))
        delta = ade_150 - ade_50
        print(f"{method:<20} {ade_50:>12.4f} {ade_150:>12.4f} {delta:>+8.4f}")
    
    # Check consistency
    print("\n" + "-" * 60)
    print("Statistical Consistency Check:")
    
    stats_50 = data_50.get("statistical_analysis", {})
    stats_150 = data_150.get("statistical_analysis", {})
    
    if "comparison_affective_vs_semantic" in stats_50 and "comparison_affective_vs_semantic" in stats_150:
        p_50 = stats_50["comparison_affective_vs_semantic"].get("p_value", 1.0)
        p_150 = stats_150["comparison_affective_vs_semantic"].get("p_value", 1.0)
        
        print(f"  p-value (n=50):  {p_50:.4f} {'Significant' if p_50 < 0.05 else 'Not significant'}")
        print(f"  p-value (n=150): {p_150:.4f} {'Significant' if p_150 < 0.05 else 'Not significant'}")
        
        if p_50 < 0.05 and p_150 < 0.05:
            print("\n  Results are consistent across sample sizes.")
        elif p_150 < 0.05:
            print("\n  Higher sample size reveals significance not visible at n=50.")
else:
    print("Could not compare: one or both result files are missing.")

# %% [markdown]
# ## Step 12: Download Results & Trained Model

# %%
!mkdir -p export
!cp -r results/ export/
!cp -r models/ export/
!cp -r checkpoints/ export/

dataset_suffix = "full" if FULL_DATASET else "subset"
!zip -r affective_rag_{dataset_suffix}.zip export/

from google.colab import files
files.download(f'affective_rag_{dataset_suffix}.zip')

# %% [markdown]
# ## (Optional) Run with Strict Thresholds
# Use this if you want to enforce only the provided thresholds.
# The experiment will FAIL if insufficient test cases are generated.

# %%
# Strict run - will abort if thresholds produce < MIN_TEST_CASES
# Uncomment to run:
# !python run_experiments.py --experiment comparative \
#     {max_items_arg} \
#     --num-queries 100 \
#     --semantic-threshold 0.7 \
#     --affective-threshold 0.85 \
#     --min-test-cases 30 \
#     --output-suffix _strict \
#     --output-dir ./results

# %% [markdown]
# ## Upload to GCS (Optional)

# %%
from google.cloud import storage

def upload_to_gcs(local_path, bucket_name, blob_prefix):
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    local_path = Path(local_path)
    for file_path in local_path.rglob("*"):
        if file_path.is_file():
            blob_name = f"{blob_prefix}/{file_path.relative_to(local_path)}"
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(str(file_path))
            print(f"Uploaded: {blob_name}")

# Uncomment to upload:
# upload_to_gcs("export", "your-bucket-name", "affective_rag_trained")
