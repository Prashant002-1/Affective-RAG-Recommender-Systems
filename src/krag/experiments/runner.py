"""
Experiment Runner for Affective-RAG Research
Implements experiments for RQ1, RQ2, and RQ3
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field, asdict

from ..evaluation import (
    Evaluator,
    BaselineComparator,
    AblationStudy,
    QueryTestCase,
    RetrievalOutput,
    RetrievalMetrics,
    AffectiveMetrics,
    ExperimentResult,
    default_weight_configs
)
from ..retrieval.krag_retriever import QueryContext
from .reproducibility import set_seed, get_experiment_id, save_experiment_config


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run"""
    name: str
    description: str
    seed: int = 42
    output_dir: str = "./results"

    # Dataset configuration
    dataset_name: str = "movielens-25m"  # Dataset from GCS
    num_test_queries: int = 100

    # Model configuration (768-dim matches SentenceBERT)
    gnn_embedding_dim: int = 768
    gnn_layers: int = 3
    gnn_heads: int = 4

    # Retrieval configuration: Score(m) = α·semantic - (1-α)·affective_rmse
    alpha: float = 0.5
    top_k: int = 10

    # Experiment-specific
    research_question: str = "RQ1"
    ablation_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentResults:
    """Complete results from an experiment"""
    config: ExperimentConfig
    experiment_id: str
    timestamp: str
    retrieval_metrics: Dict[str, float]
    affective_metrics: Dict[str, float]
    baseline_comparison: Optional[Dict[str, Any]] = None
    ablation_results: Optional[List[Dict[str, Any]]] = None
    raw_results: Optional[List[Dict[str, Any]]] = None

    def save(self, path: str):
        """Save results to JSON file"""
        data = {
            'config': self.config.to_dict(),
            'experiment_id': self.experiment_id,
            'timestamp': self.timestamp,
            'retrieval_metrics': self.retrieval_metrics,
            'affective_metrics': self.affective_metrics,
            'baseline_comparison': self.baseline_comparison,
            'ablation_results': self.ablation_results
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Results saved to {path}")

    def print_summary(self):
        """Print summary of results"""
        print("\n" + "=" * 60)
        print(f"EXPERIMENT: {self.config.name}")
        print(f"ID: {self.experiment_id}")
        print("=" * 60)

        print("\nRetrieval Metrics:")
        for metric, value in self.retrieval_metrics.items():
            print(f"  {metric}: {value:.4f}")

        print("\nAffective Metrics:")
        for metric, value in self.affective_metrics.items():
            print(f"  {metric}: {value:.4f}")

        if self.baseline_comparison:
            print("\nBaseline Comparison:")
            for approach, metrics in self.baseline_comparison.items():
                print(f"  {approach}:")
                print(f"    NDCG@10: {metrics.get('NDCG@10', 0):.4f}")
                print(f"    MRR: {metrics.get('MRR', 0):.4f}")


class ExperimentRunner:
    """
    Main experiment runner for Affective-RAG research.

    Supports:
    - RQ1: Retrieval quality comparison (semantic vs emotion vs K-RAG)
    - RQ2: Affective coherence analysis
    - RQ3: Multi-hop depth ablation study
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.experiment_id = get_experiment_id()
        self.output_dir = Path(config.output_dir) / self.experiment_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set reproducibility
        set_seed(config.seed)

        # Save config
        save_experiment_config(config.to_dict(), str(self.output_dir / "config.json"))

        # Initialize evaluator
        self.evaluator = Evaluator(output_dir=str(self.output_dir))

    def run_baseline_comparison(
        self,
        test_cases: List[QueryTestCase],
        semantic_retriever: Callable,
        emotion_retriever: Callable,
        krag_retriever: Callable
    ) -> ExperimentResults:
        """
        Run RQ1: Compare baseline approaches against Affective-RAG.

        Args:
            test_cases: List of test cases with ground truth
            semantic_retriever: Semantic-only retrieval function
            emotion_retriever: Emotion-only retrieval function
            krag_retriever: Full Affective-RAG retrieval function

        Returns:
            ExperimentResults with comparison metrics
        """
        comparator = BaselineComparator(self.evaluator)

        results = comparator.run_baseline_comparison(
            test_cases,
            semantic_retriever,
            emotion_retriever,
            krag_retriever
        )

        # Extract metrics
        krag_result = results['affective_rag']

        baseline_comparison = {}
        for name, result in results.items():
            baseline_comparison[name] = {
                'retrieval': result.retrieval_metrics.to_dict(),
                'affective': result.affective_metrics.to_dict()
            }

        experiment_results = ExperimentResults(
            config=self.config,
            experiment_id=self.experiment_id,
            timestamp=datetime.now().isoformat(),
            retrieval_metrics=krag_result.retrieval_metrics.to_dict(),
            affective_metrics=krag_result.affective_metrics.to_dict(),
            baseline_comparison=baseline_comparison
        )

        # Save results
        experiment_results.save(str(self.output_dir / "results.json"))

        # Generate report
        report = comparator.generate_comparison_report(results)
        with open(self.output_dir / "comparison_report.txt", 'w') as f:
            f.write(report)

        return experiment_results

    def run_weight_ablation(
        self,
        test_cases: List[QueryTestCase],
        retriever_factory: Callable[[Dict[str, float]], Callable],
        weight_configs: Optional[List[Dict[str, float]]] = None
    ) -> ExperimentResults:
        """
        Run RQ2: Weight sensitivity analysis for affective coherence.

        Args:
            test_cases: List of test cases
            retriever_factory: Function that creates retriever given weights
            weight_configs: List of weight configurations to test

        Returns:
            ExperimentResults with ablation analysis
        """
        if weight_configs is None:
            weight_configs = default_weight_configs()

        ablation = AblationStudy(self.evaluator)

        results = ablation.weight_sensitivity_analysis(
            test_cases,
            retriever_factory,
            weight_configs
        )

        # Find best configuration
        best_result = max(results, key=lambda r: r.retrieval_metrics.ndcg_at_10)

        ablation_data = []
        for result in results:
            ablation_data.append({
                'config': result.config,
                'retrieval': result.retrieval_metrics.to_dict(),
                'affective': result.affective_metrics.to_dict()
            })

        experiment_results = ExperimentResults(
            config=self.config,
            experiment_id=self.experiment_id,
            timestamp=datetime.now().isoformat(),
            retrieval_metrics=best_result.retrieval_metrics.to_dict(),
            affective_metrics=best_result.affective_metrics.to_dict(),
            ablation_results=ablation_data
        )

        # Save results
        experiment_results.save(str(self.output_dir / "results.json"))

        # Generate report
        report = ablation.generate_ablation_report(results, "Weight Sensitivity")
        with open(self.output_dir / "ablation_report.txt", 'w') as f:
            f.write(report)

        return experiment_results

    def run_multihop_ablation(
        self,
        test_cases: List[QueryTestCase],
        retriever_factory: Callable[[int], Callable],
        hop_depths: List[int] = [1, 2, 3]
    ) -> ExperimentResults:
        """
        Run RQ3: Multi-hop depth analysis.

        Args:
            test_cases: List of test cases
            retriever_factory: Function that creates retriever given hop depth
            hop_depths: List of hop depths to test

        Returns:
            ExperimentResults with multi-hop analysis
        """
        ablation = AblationStudy(self.evaluator)

        results = ablation.multi_hop_analysis(
            test_cases,
            retriever_factory,
            hop_depths
        )

        # Find optimal depth
        best_result = max(results, key=lambda r: r.retrieval_metrics.ndcg_at_10)

        ablation_data = []
        for result in results:
            ablation_data.append({
                'hop_depth': result.config.get('hop_depth'),
                'retrieval': result.retrieval_metrics.to_dict(),
                'affective': result.affective_metrics.to_dict()
            })

        experiment_results = ExperimentResults(
            config=self.config,
            experiment_id=self.experiment_id,
            timestamp=datetime.now().isoformat(),
            retrieval_metrics=best_result.retrieval_metrics.to_dict(),
            affective_metrics=best_result.affective_metrics.to_dict(),
            ablation_results=ablation_data
        )

        # Save results
        experiment_results.save(str(self.output_dir / "results.json"))

        # Generate report
        report = ablation.generate_ablation_report(results, "Multi-hop Depth")
        with open(self.output_dir / "ablation_report.txt", 'w') as f:
            f.write(report)

        return experiment_results


def run_rq1_retrieval_quality(
    test_cases: List[QueryTestCase],
    semantic_retriever: Callable,
    emotion_retriever: Callable,
    krag_retriever: Callable,
    output_dir: str = "./results",
    seed: int = 42
) -> ExperimentResults:
    """
    Convenience function for RQ1: Does K-RAG improve retrieval quality?

    Compares:
    - Semantic-only baseline
    - Emotion-only baseline
    - Full Affective-RAG

    Args:
        test_cases: Test queries with ground truth
        semantic_retriever: Semantic retrieval function
        emotion_retriever: Emotion retrieval function
        krag_retriever: Full K-RAG retrieval function
        output_dir: Output directory for results
        seed: Random seed for reproducibility

    Returns:
        ExperimentResults with comparison
    """
    config = ExperimentConfig(
        name="RQ1_Retrieval_Quality",
        description="Compare retrieval quality: semantic vs emotion vs Affective-RAG",
        research_question="RQ1",
        seed=seed,
        output_dir=output_dir,
        num_test_queries=len(test_cases)
    )

    runner = ExperimentRunner(config)
    results = runner.run_baseline_comparison(
        test_cases,
        semantic_retriever,
        emotion_retriever,
        krag_retriever
    )

    results.print_summary()
    return results


def run_rq2_affective_coherence(
    test_cases: List[QueryTestCase],
    retriever_factory: Callable[[Dict[str, float]], Callable],
    output_dir: str = "./results",
    seed: int = 42
) -> ExperimentResults:
    """
    Convenience function for RQ2: How does emotion weight affect affective coherence?

    Tests various weight configurations to analyze emotion signal contribution.

    Args:
        test_cases: Test queries with ground truth
        retriever_factory: Function to create retriever from weights
        output_dir: Output directory
        seed: Random seed

    Returns:
        ExperimentResults with weight analysis
    """
    config = ExperimentConfig(
        name="RQ2_Affective_Coherence",
        description="Analyze emotion weight impact on affective coherence",
        research_question="RQ2",
        ablation_type="weight_sensitivity",
        seed=seed,
        output_dir=output_dir,
        num_test_queries=len(test_cases)
    )

    runner = ExperimentRunner(config)
    results = runner.run_weight_ablation(test_cases, retriever_factory)

    results.print_summary()
    return results


def run_rq3_multihop_depth(
    test_cases: List[QueryTestCase],
    retriever_factory: Callable[[int], Callable],
    hop_depths: List[int] = [1, 2, 3],
    output_dir: str = "./results",
    seed: int = 42
) -> ExperimentResults:
    """
    Convenience function for RQ3: What is the optimal multi-hop depth?

    Tests different hop depths in knowledge graph traversal.

    Args:
        test_cases: Test queries with ground truth
        retriever_factory: Function to create retriever from hop depth
        hop_depths: Depths to test
        output_dir: Output directory
        seed: Random seed

    Returns:
        ExperimentResults with depth analysis
    """
    config = ExperimentConfig(
        name="RQ3_Multihop_Depth",
        description="Analyze optimal knowledge graph traversal depth",
        research_question="RQ3",
        ablation_type="multihop",
        seed=seed,
        output_dir=output_dir,
        num_test_queries=len(test_cases)
    )

    runner = ExperimentRunner(config)
    results = runner.run_multihop_ablation(test_cases, retriever_factory, hop_depths)

    results.print_summary()
    return results


def run_comparative_retrieval_analysis(
    test_cases: List['QueryTestCase'],
    bm25_retriever: Callable,
    semantic_retriever: Callable,
    balanced_retriever: Callable,
    affective_retriever: Callable,
    movie_affective_signatures: Dict[str, 'np.ndarray'],
    content_embedder: Optional[Any] = None,
    output_dir: str = "./results",
    seed: int = 42,
    k: int = 10
) -> Dict[str, Dict[str, float]]:
    """
    Comparative retrieval analysis.

    Evaluates behavior under affective-semantic agreement and dissonance.

    Args:
        test_cases: Test queries with target emotion vectors
        bm25_retriever: BM25 lexical baseline
        semantic_retriever: Semantic-only (alpha=1.0)
        balanced_retriever: Balanced (alpha=0.5)
        affective_retriever: Affective-focused (alpha=0.3)
        movie_affective_signatures: Dict of movie affective signatures
        content_embedder: Embedder for computing query embeddings
        output_dir: Output directory
        seed: Random seed
        k: Number of results to evaluate

    Returns:
        Dict mapping retriever name to metrics dict
    """
    import numpy as np
    from ..evaluation.metrics import (
        compute_semantic_recall_at_k,
        compute_affective_precision_at_k,
        compute_affective_displacement_error
    )

    set_seed(seed)

    retrievers = {
        'BM25': bm25_retriever,
        'Semantic (α=1.0)': semantic_retriever,
        'Balanced (α=0.5)': balanced_retriever,
        'Affective (α=0.3)': affective_retriever
    }

    results = {}

    for name, retriever in retrievers.items():
        semantic_recalls_5 = []
        semantic_recalls_10 = []
        affective_precisions_5 = []
        affective_precisions_10 = []
        ade_scores = []

        for test_case in test_cases:
            if content_embedder is not None:
                query_embedding = content_embedder.embed_content(test_case.query_text)
            else:
                query_embedding = np.zeros(768)

            query_context = QueryContext(
                query_text=test_case.query_text,
                user_emotions=_dict_to_emotion_profile(test_case.target_emotions),
                query_embedding=query_embedding,
                emotion_embedding=query_embedding
            )

            retrieved = retriever(query_context, k=k)
            retrieved_ids = [r.content_id for r in retrieved]

            target_vector = np.array([
                test_case.target_emotions.get(e, 0.0)
                for e in ['happiness', 'sadness', 'anger', 'fear', 'surprise', 'disgust']
            ])

            if test_case.semantic_relevant:
                semantic_recalls_5.append(
                    compute_semantic_recall_at_k(retrieved_ids, test_case.semantic_relevant, 5)
                )
                semantic_recalls_10.append(
                    compute_semantic_recall_at_k(retrieved_ids, test_case.semantic_relevant, 10)
                )

            affective_precisions_5.append(
                compute_affective_precision_at_k(
                    retrieved_ids, movie_affective_signatures, target_vector, 5
                )
            )
            affective_precisions_10.append(
                compute_affective_precision_at_k(
                    retrieved_ids, movie_affective_signatures, target_vector, 10
                )
            )

            ade = compute_affective_displacement_error(
                retrieved_ids, movie_affective_signatures, target_vector, k
            )
            if ade != float('inf'):
                ade_scores.append(ade)

        results[name] = {
            'Semantic_Recall@5': float(np.mean(semantic_recalls_5)) if semantic_recalls_5 else 0.0,
            'Semantic_Recall@10': float(np.mean(semantic_recalls_10)) if semantic_recalls_10 else 0.0,
            'AP@5': float(np.mean(affective_precisions_5)),
            'AP@10': float(np.mean(affective_precisions_10)),
            'ADE': float(np.mean(ade_scores)) if ade_scores else float('inf')
        }

        print(f"\n{name}:")
        for metric, value in results[name].items():
            print(f"  {metric}: {value:.4f}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "comparative_analysis.json", 'w') as f:
        json.dump(results, f, indent=2)

    return results


def run_ablation_study(
    test_cases: List['QueryTestCase'],
    retrievers: Dict[str, Callable],
    movie_affective_signatures: Dict[str, 'np.ndarray'],
    content_embedder: Optional[Any] = None,
    output_dir: str = "./results",
    seed: int = 42,
    k: int = 10
) -> Dict[str, Dict[str, float]]:
    """
    Run ablation study comparing different retriever configurations.

    Args:
        test_cases: Test queries with ground truth
        retrievers: Dict mapping name -> retriever function
        movie_affective_signatures: Dict of movie affective signatures
        content_embedder: Embedder for computing query embeddings
        output_dir: Output directory
        seed: Random seed
        k: Number of results to evaluate

    Returns:
        Dict mapping retriever name to metrics dict
    """
    import numpy as np
    from ..evaluation.metrics import (
        compute_semantic_recall_at_k,
        compute_affective_precision_at_k,
        compute_affective_displacement_error
    )

    set_seed(seed)

    results = {}

    for name, retriever in retrievers.items():
        semantic_recalls_5 = []
        semantic_recalls_10 = []
        affective_precisions_5 = []
        affective_precisions_10 = []
        ade_scores = []

        for test_case in test_cases:
            if content_embedder is not None:
                query_embedding = content_embedder.embed_content(test_case.query_text)
            else:
                query_embedding = np.zeros(768)

            query_context = QueryContext(
                query_text=test_case.query_text,
                user_emotions=_dict_to_emotion_profile(test_case.target_emotions),
                query_embedding=query_embedding,
                emotion_embedding=query_embedding
            )

            retrieved = retriever(query_context, k=k)
            retrieved_ids = [r.content_id for r in retrieved]

            target_vector = np.array([
                test_case.target_emotions.get(e, 0.0)
                for e in ['happiness', 'sadness', 'anger', 'fear', 'surprise', 'disgust']
            ])

            if test_case.semantic_relevant:
                semantic_recalls_5.append(
                    compute_semantic_recall_at_k(retrieved_ids, test_case.semantic_relevant, 5)
                )
                semantic_recalls_10.append(
                    compute_semantic_recall_at_k(retrieved_ids, test_case.semantic_relevant, 10)
                )

            affective_precisions_5.append(
                compute_affective_precision_at_k(
                    retrieved_ids, movie_affective_signatures, target_vector, 5
                )
            )
            affective_precisions_10.append(
                compute_affective_precision_at_k(
                    retrieved_ids, movie_affective_signatures, target_vector, 10
                )
            )

            ade = compute_affective_displacement_error(
                retrieved_ids, movie_affective_signatures, target_vector, k
            )
            if ade != float('inf'):
                ade_scores.append(ade)

        results[name] = {
            'Semantic_Recall@5': float(np.mean(semantic_recalls_5)) if semantic_recalls_5 else 0.0,
            'Semantic_Recall@10': float(np.mean(semantic_recalls_10)) if semantic_recalls_10 else 0.0,
            'AP@5': float(np.mean(affective_precisions_5)),
            'AP@10': float(np.mean(affective_precisions_10)),
            'ADE': float(np.mean(ade_scores)) if ade_scores else float('inf')
        }

        print(f"\n{name}:")
        for metric, value in results[name].items():
            print(f"  {metric}: {value:.4f}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "comparative_analysis.json", 'w') as f:
        json.dump(results, f, indent=2)

    return results


def run_ablation_study_enhanced(
    test_cases: List['QueryTestCase'],
    retrievers: Dict[str, Callable],
    movie_affective_signatures: Dict[str, 'np.ndarray'],
    content_embedder: Optional[Any] = None,
    seed: int = 42,
    k: int = 10
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, Any]]]:
    """
    Enhanced ablation study that returns both aggregated metrics AND per-query scores.
    
    This enables statistical analysis (confidence intervals, effect sizes, p-values)
    in the calling code.

    Args:
        test_cases: Test queries with ground truth
        retrievers: Dict mapping name -> retriever function
        movie_affective_signatures: Dict of movie affective signatures
        content_embedder: Embedder for computing query embeddings
        seed: Random seed
        k: Number of results to evaluate

    Returns:
        Tuple of:
        - results: Dict mapping retriever name to aggregated metrics dict
        - per_query_scores: Dict mapping retriever name to query-aligned per-query scores.
          Scores are aligned by `test_case.query_id` and use NaN for undefined values.
    """
    import numpy as np
    from ..evaluation.metrics import (
        compute_semantic_recall_at_k,
        compute_affective_precision_at_k,
        compute_affective_displacement_error
    )

    set_seed(seed)

    results = {}
    per_query_scores = {}

    for name, retriever in retrievers.items():
        query_ids: List[str] = []
        sr5_scores: List[float] = []
        sr10_scores: List[float] = []
        ap5_scores: List[float] = []
        ap10_scores: List[float] = []
        ade_scores: List[float] = []

        for test_case in test_cases:
            query_ids.append(test_case.query_id)
            if content_embedder is not None:
                query_embedding = content_embedder.embed_content(test_case.query_text)
            else:
                query_embedding = np.zeros(768)

            # Build a proper 768-dim emotion embedding (text-based) so KRAG can
            # use emotion_search candidates meaningfully when configured.
            # This matches the system's QueryEmbedder format (query + emotion description).
            emotion_profile = _dict_to_emotion_profile(test_case.target_emotions)
            top_emotions = sorted(emotion_profile.to_dict().items(), key=lambda x: -x[1])[:3]
            emotion_parts = [f"{name} ({score:.0%})" for name, score in top_emotions if score > 0.3]
            emotion_desc = f"Emotions: {', '.join(emotion_parts)}" if emotion_parts else "neutral emotional tone"
            emotion_embedding = (
                content_embedder.embed_content(f"{test_case.query_text} {emotion_desc}")
                if content_embedder is not None
                else query_embedding
            )

            query_context = QueryContext(
                query_text=test_case.query_text,
                user_emotions=emotion_profile,
                query_embedding=query_embedding,
                emotion_embedding=emotion_embedding
            )

            retrieved = retriever(query_context, k=k)
            retrieved_ids = [r.content_id for r in retrieved]

            target_vector = np.array([
                test_case.target_emotions.get(e, 0.0)
                for e in ['happiness', 'sadness', 'anger', 'fear', 'surprise', 'disgust']
            ])

            # Semantic recall may be undefined depending on test-case type.
            if test_case.semantic_relevant:
                sr5_scores.append(
                    float(compute_semantic_recall_at_k(retrieved_ids, test_case.semantic_relevant, 5))
                )
                sr10_scores.append(
                    float(compute_semantic_recall_at_k(retrieved_ids, test_case.semantic_relevant, 10))
                )
            else:
                sr5_scores.append(float('nan'))
                sr10_scores.append(float('nan'))

            ap5_scores.append(
                float(compute_affective_precision_at_k(
                    retrieved_ids, movie_affective_signatures, target_vector, 5
                ))
            )
            ap10_scores.append(
                float(compute_affective_precision_at_k(
                    retrieved_ids, movie_affective_signatures, target_vector, 10
                ))
            )

            ade = float(compute_affective_displacement_error(
                retrieved_ids, movie_affective_signatures, target_vector, k
            ))
            # Keep query alignment: use NaN for undefined ADE rather than dropping.
            ade_scores.append(float('nan') if ade == float('inf') else ade)

        sr5_arr = np.array(sr5_scores, dtype=float)
        sr10_arr = np.array(sr10_scores, dtype=float)
        ap5_arr = np.array(ap5_scores, dtype=float)
        ap10_arr = np.array(ap10_scores, dtype=float)
        ade_arr = np.array(ade_scores, dtype=float)

        # Aggregated results
        results[name] = {
            'Semantic_Recall@5': float(np.nanmean(sr5_arr)) if not np.all(np.isnan(sr5_arr)) else 0.0,
            'Semantic_Recall@10': float(np.nanmean(sr10_arr)) if not np.all(np.isnan(sr10_arr)) else 0.0,
            'AP@5': float(np.nanmean(ap5_arr)) if len(ap5_arr) else 0.0,
            'AP@10': float(np.nanmean(ap10_arr)) if len(ap10_arr) else 0.0,
            'ADE': float(np.nanmean(ade_arr)) if not np.all(np.isnan(ade_arr)) else float('inf')
        }
        
        # Per-query scores for statistical analysis
        per_query_scores[name] = {
            'query_ids': query_ids,
            'sr5_scores': sr5_scores,
            'sr10_scores': sr10_scores,
            'ap5_scores': ap5_scores,
            'ap10_scores': ap10_scores,
            'ade_scores': ade_scores
        }

        print(f"\n{name}:")
        for metric, value in results[name].items():
            print(f"  {metric}: {value:.4f}")

    return results, per_query_scores


def _dict_to_emotion_profile(emotion_dict: Dict[str, float]) -> 'EmotionProfile':
    """Convert emotion dict to EmotionProfile object."""
    from ..core.emotion_detection import EmotionProfile
    return EmotionProfile(
        happiness=emotion_dict.get('happiness', 0.0),
        sadness=emotion_dict.get('sadness', 0.0),
        anger=emotion_dict.get('anger', 0.0),
        fear=emotion_dict.get('fear', 0.0),
        surprise=emotion_dict.get('surprise', 0.0),
        disgust=emotion_dict.get('disgust', 0.0)
    )


def run_predictive_fidelity_loocv(
    loocv_test_cases: List,
    retrievers: Dict[str, Callable],
    movie_affective_signatures: Dict[str, 'np.ndarray'],
    output_dir: str = "./results",
    k: int = 10
) -> Dict[str, Dict[str, float]]:
    """
    Predictive fidelity on interaction history.

    Uses LOOCV test cases where ground truth is the last item in user history.

    Metrics:
    - Semantic Recall@K: Binary hit if M_n in top-K
    - Affective Fidelity (Cosine): Avg cosine similarity to target emotion
    - Affective RMSE: RMSE between target and retrieved emotion vectors

    Args:
        loocv_test_cases: List of LOOCVTestCase objects
        retrievers: Dict mapping name -> retriever function
        movie_affective_signatures: Dict of movie affective signatures
        output_dir: Output directory
        k: Number of results to evaluate

    Returns:
        Dict mapping retriever name to metrics dict
    """
    import numpy as np
    from ..evaluation.synthetic_testset import LOOCVTestCase

    results = {}

    for name, retriever in retrievers.items():
        hits_at_5 = []
        hits_at_10 = []
        affective_fidelity_scores = []
        affective_rmse_scores = []

        for test_case in loocv_test_cases:
            query_context = QueryContext(
                query_text="",
                user_emotions=_dict_to_emotion_profile(test_case.target_emotions),
                query_embedding=test_case.history_embedding,
                emotion_embedding=test_case.history_embedding
            )

            retrieved = retriever(query_context, k=k)
            retrieved_ids = [r.content_id for r in retrieved]

            hit_at_5 = 1.0 if test_case.ground_truth_id in retrieved_ids[:5] else 0.0
            hit_at_10 = 1.0 if test_case.ground_truth_id in retrieved_ids[:10] else 0.0
            hits_at_5.append(hit_at_5)
            hits_at_10.append(hit_at_10)

            target_vector = test_case.get_emotion_vector()
            target_norm = np.linalg.norm(target_vector)

            fidelity_scores = []
            rmse_distances = []

            for cid in retrieved_ids[:k]:
                movie_sig = movie_affective_signatures.get(cid, np.zeros(6))
                movie_norm = np.linalg.norm(movie_sig)

                if target_norm > 1e-8 and movie_norm > 1e-8:
                    cosine_sim = np.dot(target_vector, movie_sig) / (target_norm * movie_norm)
                    fidelity_scores.append(cosine_sim)

                rmse = np.sqrt(np.mean((target_vector - movie_sig) ** 2))
                rmse_distances.append(rmse)

            if fidelity_scores:
                affective_fidelity_scores.append(np.mean(fidelity_scores))
            if rmse_distances:
                affective_rmse_scores.append(np.mean(rmse_distances))

        results[name] = {
            'Semantic_Recall@5': float(np.mean(hits_at_5)),
            'Semantic_Recall@10': float(np.mean(hits_at_10)),
            'Affective_Fidelity': float(np.mean(affective_fidelity_scores)) if affective_fidelity_scores else 0.0,
            'Affective_RMSE': float(np.mean(affective_rmse_scores)) if affective_rmse_scores else 0.0,
            'num_queries': len(loocv_test_cases)
        }

        print(f"\n{name}:")
        for metric, value in results[name].items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "predictive_fidelity.json", 'w') as f:
        json.dump(results, f, indent=2)

    return results


def run_explainability_evaluation(
    retriever: Callable,
    knowledge_graph,
    test_cases: List,
    llm_client=None,
    output_dir: str = "./results",
    num_samples: int = 50
) -> Dict[str, Any]:
    """
    Explainability and faithfulness evaluation.

    Evaluates:
    - Causal Necessity Analysis (FNS)
    - RAGAS Faithfulness Score

    Args:
        retriever: Retriever to evaluate
        knowledge_graph: ContentKnowledgeGraph instance
        test_cases: Test queries
        llm_client: Optional LLM client for RAGAS evaluation
        output_dir: Output directory
        num_samples: Number of samples to evaluate

    Returns:
        Dict with FNS and RAGAS metrics
    """
    import numpy as np
    from ..evaluation.causal_analysis import CausalNecessityAnalyzer
    from ..evaluation.ragas_evaluator import RAGASFaithfulnessEvaluator, build_graph_context

    results = {
        'causal_necessity': {},
        'ragas_faithfulness': {}
    }

    print("\n" + "=" * 60)
    print("EXPLAINABILITY EVALUATION")
    print("=" * 60)

    print("\n[1/2] Running Causal Necessity Analysis...")
    try:
        analyzer = CausalNecessityAnalyzer(retriever, knowledge_graph)
        fns_scores = []

        for i, test_case in enumerate(test_cases[:num_samples]):
            query_embedding = np.zeros(768)
            if hasattr(test_case, 'history_embedding'):
                query_embedding = test_case.history_embedding

            query_context = QueryContext(
                query_text=getattr(test_case, 'query_text', ''),
                user_emotions=_dict_to_emotion_profile(test_case.target_emotions),
                query_embedding=query_embedding,
                emotion_embedding=query_embedding
            )

            retrieved = retriever(query_context, k=5)

            if retrieved:
                target_content_id = retrieved[0].content_id

                emotion_dict = test_case.target_emotions
                dominant_emotion = max(emotion_dict, key=emotion_dict.get)

                analysis = analyzer.analyze_single(
                    query_context=query_context,
                    target_content_id=target_content_id,
                    emotion_to_perturb=dominant_emotion
                )

                if analysis and analysis.fns_score > 0:
                    fns_scores.append(analysis.fns_score)

        if fns_scores:
            results['causal_necessity'] = {
                'mean_fns': float(np.mean(fns_scores)),
                'median_fns': float(np.median(fns_scores)),
                'high_fns_ratio': float(sum(1 for s in fns_scores if s > 0.5) / len(fns_scores)),
                'num_samples': len(fns_scores)
            }
            print(f"  Mean FNS: {results['causal_necessity']['mean_fns']:.4f}")
            print(f"  High FNS Ratio (>0.5): {results['causal_necessity']['high_fns_ratio']:.4f}")

    except Exception as e:
        print(f"  Causal analysis error: {e}")
        results['causal_necessity'] = {'error': str(e)}

    print("\n[2/2] Running RAGAS Faithfulness Evaluation...")
    try:
        ragas_evaluator = RAGASFaithfulnessEvaluator(llm_client=llm_client)
        faithfulness_scores = []

        for i, test_case in enumerate(test_cases[:num_samples]):
            query_embedding = np.zeros(768)
            if hasattr(test_case, 'history_embedding'):
                query_embedding = test_case.history_embedding

            query_context = QueryContext(
                query_text=getattr(test_case, 'query_text', ''),
                user_emotions=_dict_to_emotion_profile(test_case.target_emotions),
                query_embedding=query_embedding,
                emotion_embedding=query_embedding
            )

            retrieved = retriever(query_context, k=3)

            for result in retrieved:
                graph_context = build_graph_context(
                    result.content_id,
                    knowledge_graph
                )

                explanation = result.explanation if result.explanation else ""
                if explanation:
                    score = ragas_evaluator.evaluate_faithfulness(
                        explanation,
                        graph_context
                    )
                    faithfulness_scores.append(score)

        if faithfulness_scores:
            results['ragas_faithfulness'] = {
                'mean_faithfulness': float(np.mean(faithfulness_scores)),
                'median_faithfulness': float(np.median(faithfulness_scores)),
                'perfect_score_ratio': float(sum(1 for s in faithfulness_scores if s >= 0.95) / len(faithfulness_scores)),
                'num_samples': len(faithfulness_scores)
            }
            print(f"  Mean Faithfulness: {results['ragas_faithfulness']['mean_faithfulness']:.4f}")
            print(f"  Perfect Score Ratio: {results['ragas_faithfulness']['perfect_score_ratio']:.4f}")

    except Exception as e:
        print(f"  RAGAS evaluation error: {e}")
        results['ragas_faithfulness'] = {'error': str(e)}

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "explainability.json", 'w') as f:
        json.dump(results, f, indent=2)

    return results
