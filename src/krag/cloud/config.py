"""
Cloud Configuration for Affective-RAG
Centralized configuration for Google Cloud Platform resources

Environment Variables:
- GOOGLE_CLOUD_PROJECT: GCP project ID (required)
- GOOGLE_APPLICATION_CREDENTIALS: Path to service account JSON (optional for local dev)
- GCS_BUCKET: GCS bucket name
- GOOGLE_CLOUD_REGION: Region (default: us-central1)
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path


# =============================================================================
# GCS Dataset Configuration
# =============================================================================

# Default bucket name is intentionally a placeholder for public repos.
DEFAULT_GCS_BUCKET = "your-gcs-bucket"
DEFAULT_GCS_BASE_PATH = "Dataset"

# Dataset file paths (relative to base path).
# Keep this mapping generic in the public repo; configure concrete paths externally.
GCS_DATASET_PATHS = {
    "movies": "movies.csv",
    "ratings": "ratings.csv",
    "embeddings": "precomputed/embeddings_v1.npz",
    "node_embeddings": "precomputed/node_embeddings_v1.pkl",
}


# =============================================================================
# Vertex AI Configuration
# =============================================================================

@dataclass
class VertexAIConfig:
    """Configuration for Vertex AI services"""
    project_id: str
    location: str = "us-central1"
    model_name: str = "gemini-3-flash-preview"
    embedding_model: str = "textembedding-gecko@003"
    temperature: float = 0.7
    max_output_tokens: int = 1024
    top_p: float = 0.9
    credentials_path: Optional[str] = None

    def validate(self) -> bool:
        """Validate configuration"""
        if not self.project_id:
            raise ValueError("project_id is required")
        return True

    def to_env_dict(self) -> Dict[str, str]:
        """Convert to environment variables"""
        env = {
            'GOOGLE_CLOUD_PROJECT': self.project_id,
            'GOOGLE_CLOUD_REGION': self.location,
            'VERTEX_AI_MODEL': self.model_name
        }
        if self.credentials_path:
            env['GOOGLE_APPLICATION_CREDENTIALS'] = self.credentials_path
        return env

    def apply_to_environment(self):
        """Set environment variables"""
        for key, value in self.to_env_dict().items():
            os.environ[key] = value


# =============================================================================
# GCS Storage Configuration
# =============================================================================

@dataclass
class GCSConfig:
    """Configuration for Google Cloud Storage"""
    bucket_name: str = DEFAULT_GCS_BUCKET
    base_path: str = DEFAULT_GCS_BASE_PATH
    
    # Output paths for results
    results_prefix: str = "results"
    models_prefix: str = "models"
    cache_prefix: str = "cache"

    def get_data_uri(self, dataset_key: str) -> str:
        """Get full GCS URI for a dataset"""
        if dataset_key in GCS_DATASET_PATHS:
            path = GCS_DATASET_PATHS[dataset_key]
        else:
            path = dataset_key
        return f"gs://{self.bucket_name}/{self.base_path}/{path}"

    def get_results_uri(self, filename: str) -> str:
        """Get GCS URI for results output"""
        return f"gs://{self.bucket_name}/{self.results_prefix}/{filename}"

    def get_model_uri(self, model_name: str) -> str:
        """Get GCS URI for model artifacts"""
        return f"gs://{self.bucket_name}/{self.models_prefix}/{model_name}"


# =============================================================================
# Combined Cloud Configuration
# =============================================================================

@dataclass
class CloudConfig:
    """Main cloud configuration for Affective-RAG"""
    vertex_ai: VertexAIConfig
    gcs: GCSConfig = field(default_factory=GCSConfig)
    
    # Local fallback paths
    local_data_dir: str = "./data"
    local_model_dir: str = "./models"
    local_results_dir: str = "./results"
    local_cache_dir: str = "./data/cache"

    def __post_init__(self):
        """Create local directories"""
        for dir_path in [
            self.local_data_dir,
            self.local_model_dir,
            self.local_results_dir,
            self.local_cache_dir
        ]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    def apply_to_environment(self):
        """Apply configuration to environment variables"""
        self.vertex_ai.apply_to_environment()
        os.environ['GCS_BUCKET'] = self.gcs.bucket_name


# =============================================================================
# Configuration Loaders
# =============================================================================

def load_config_from_env() -> CloudConfig:
    """
    Load cloud configuration from environment variables.
    
    Required:
    - GOOGLE_CLOUD_PROJECT: GCP project ID
    
    Optional:
    - GOOGLE_CLOUD_REGION: Region (default: us-central1)
    - GOOGLE_APPLICATION_CREDENTIALS: Path to service account key
    - GCS_BUCKET: Cloud Storage bucket name
    - VERTEX_AI_MODEL: Model name (default: gemini-3-flash-preview)
    """
    project_id = os.environ.get('GOOGLE_CLOUD_PROJECT')
    if not project_id:
        raise ValueError(
            "GOOGLE_CLOUD_PROJECT environment variable is required. "
            "Set it in your environment."
        )

    vertex_config = VertexAIConfig(
        project_id=project_id,
        location=os.environ.get('GOOGLE_CLOUD_REGION', 'us-central1'),
        model_name=os.environ.get('VERTEX_AI_MODEL', 'gemini-3-flash-preview'),
        credentials_path=os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    )

    gcs_config = GCSConfig(
        bucket_name=os.environ.get('GCS_BUCKET', DEFAULT_GCS_BUCKET),
        base_path=os.environ.get('GCS_BASE_PATH', DEFAULT_GCS_BASE_PATH)
    )

    return CloudConfig(
        vertex_ai=vertex_config,
        gcs=gcs_config
    )


def create_config(
    project_id: str,
    bucket_name: str = DEFAULT_GCS_BUCKET,
    location: str = "us-central1",
    credentials_path: Optional[str] = None
) -> CloudConfig:
    """
    Create a cloud configuration programmatically.
    
    Args:
        project_id: GCP project ID
        bucket_name: GCS bucket name
        location: Vertex AI location
        credentials_path: Optional path to service account key
        
    Returns:
        CloudConfig instance
    """
    vertex_config = VertexAIConfig(
        project_id=project_id,
        location=location,
        credentials_path=credentials_path
    )

    gcs_config = GCSConfig(bucket_name=bucket_name)

    return CloudConfig(
        vertex_ai=vertex_config,
        gcs=gcs_config
    )


# =============================================================================
# Configuration Templates
# =============================================================================

def local_dev_config(project_id: str, credentials_path: str) -> CloudConfig:
    """
    Configuration for local development.
    Requires explicit credentials file.
    """
    return create_config(
        project_id=project_id,
        credentials_path=credentials_path
    )


def colab_config(project_id: str) -> CloudConfig:
    """
    Configuration for Google Colab.
    Uses ADC (Application Default Credentials) automatically.
    """
    return create_config(project_id=project_id)


def production_config(
    project_id: str,
    bucket_name: str = DEFAULT_GCS_BUCKET,
    location: str = "us-central1"
) -> CloudConfig:
    """
    Configuration for production deployment.
    Uses cloud storage for all data.
    """
    return create_config(
        project_id=project_id,
        bucket_name=bucket_name,
        location=location
    )


# =============================================================================
# Backward Compatibility
# =============================================================================

# Old names for backward compatibility
StorageConfig = GCSConfig

def create_default_config(
    project_id: str,
    location: str = "us-central1",
    bucket_name: Optional[str] = None,
    credentials_path: Optional[str] = None
) -> CloudConfig:
    """Backward compatible config creator"""
    return create_config(
        project_id=project_id,
        bucket_name=bucket_name or DEFAULT_GCS_BUCKET,
        location=location,
        credentials_path=credentials_path
    )
