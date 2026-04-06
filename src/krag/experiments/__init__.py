"""
Experiment Infrastructure for Affective-RAG Research
Provides reproducible experiment runners and result organization
"""

from .runner import (
    ExperimentRunner,
    ExperimentConfig,
    run_rq1_retrieval_quality,
    run_rq2_affective_coherence,
    run_rq3_multihop_depth
)

from .reproducibility import (
    set_seed,
    get_experiment_id,
    save_experiment_config
)

__all__ = [
    'ExperimentRunner',
    'ExperimentConfig',
    'run_rq1_retrieval_quality',
    'run_rq2_affective_coherence',
    'run_rq3_multihop_depth',
    'set_seed',
    'get_experiment_id',
    'save_experiment_config'
]
