"""
Evaluation Module for Affective-RAG
Provides metrics and evaluation utilities for research experiments
"""

from .metrics import (
    RetrievalMetrics,
    AffectiveMetrics,
    ExperimentResult,
    compute_precision_at_k,
    compute_ndcg,
    compute_mrr,
    compute_affective_precision_at_k,
    compute_affective_displacement_error,
    compute_faithfulness_necessity_score,
    compute_semantic_recall_at_k
)

from .evaluator import (
    Evaluator,
    BaselineComparator,
    AblationStudy,
    QueryTestCase,
    RetrievalOutput,
    default_weight_configs
)

from .synthetic_testset import (
    SyntheticTestSetGenerator,
    LOOCVTestSetGenerator,
    LOOCVTestCase,
    save_loocv_test_set,
    load_loocv_test_set
)

from .ragas_evaluator import (
    RAGASFaithfulnessEvaluator,
    FaithfulnessResult,
    build_graph_context,
    build_full_subgraph_context,
    run_faithfulness_evaluation
)

__all__ = [
    'RetrievalMetrics',
    'AffectiveMetrics',
    'ExperimentResult',
    'compute_precision_at_k',
    'compute_ndcg',
    'compute_mrr',
    'compute_affective_precision_at_k',
    'compute_affective_displacement_error',
    'compute_faithfulness_necessity_score',
    'compute_semantic_recall_at_k',
    'Evaluator',
    'BaselineComparator',
    'AblationStudy',
    'QueryTestCase',
    'RetrievalOutput',
    'default_weight_configs',
    'SyntheticTestSetGenerator',
    'LOOCVTestSetGenerator',
    'LOOCVTestCase',
    'save_loocv_test_set',
    'load_loocv_test_set',
    'RAGASFaithfulnessEvaluator',
    'FaithfulnessResult',
    'build_graph_context',
    'build_full_subgraph_context',
    'run_faithfulness_evaluation'
]
