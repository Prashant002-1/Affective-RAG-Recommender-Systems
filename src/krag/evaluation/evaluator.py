"""
Evaluator and Ablation Study Framework for Affective-RAG
Supports systematic comparison of retrieval approaches
"""

import json
from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from datetime import datetime

from .metrics import (
    RetrievalMetrics,
    AffectiveMetrics,
    ExperimentResult,
    compute_all_retrieval_metrics,
    compute_all_affective_metrics
)


@dataclass
class QueryTestCase:
    """Test case for evaluation"""
    query_id: str
    query_text: str
    user_emotions: Dict[str, float]
    relevant_items: List[str]
    relevance_scores: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class RetrievalOutput:
    """Output from a retrieval system"""
    retrieved_ids: List[str]
    scores: List[float]
    item_emotions: List[Dict[str, float]]
    metadata: Optional[Dict[str, Any]] = None


class Evaluator:
    """
    Main evaluator for Affective-RAG experiments
    """

    def __init__(self, output_dir: str = "./results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[ExperimentResult] = []

    def evaluate_retrieval(
        self,
        test_cases: List[QueryTestCase],
        retrieval_fn: Callable[[str, Dict[str, float]], RetrievalOutput],
        experiment_name: str,
        config: Dict[str, Any]
    ) -> ExperimentResult:
        """
        Evaluate a retrieval function on test cases

        Args:
            test_cases: List of test cases
            retrieval_fn: Function that takes (query_text, user_emotions) and returns RetrievalOutput
            experiment_name: Name for this experiment
            config: Configuration used for this experiment

        Returns:
            ExperimentResult with aggregated metrics
        """
        all_retrieval_metrics = []
        all_affective_metrics = []

        for test_case in test_cases:
            # Run retrieval
            output = retrieval_fn(test_case.query_text, test_case.user_emotions)

            # Compute retrieval metrics
            retrieval_metrics = compute_all_retrieval_metrics(
                output.retrieved_ids,
                test_case.relevant_items,
                test_case.relevance_scores
            )
            all_retrieval_metrics.append(retrieval_metrics)

            # Compute affective metrics
            affective_metrics = compute_all_affective_metrics(
                test_case.user_emotions,
                output.item_emotions
            )
            all_affective_metrics.append(affective_metrics)

        # Aggregate metrics
        aggregated_retrieval = self._aggregate_retrieval_metrics(all_retrieval_metrics)
        aggregated_affective = self._aggregate_affective_metrics(all_affective_metrics)

        result = ExperimentResult(
            experiment_name=experiment_name,
            config=config,
            retrieval_metrics=aggregated_retrieval,
            affective_metrics=aggregated_affective,
            num_queries=len(test_cases)
        )

        self.results.append(result)
        return result

    def _aggregate_retrieval_metrics(
        self,
        metrics_list: List[RetrievalMetrics]
    ) -> RetrievalMetrics:
        """Aggregate retrieval metrics across queries"""
        if not metrics_list:
            return RetrievalMetrics()

        return RetrievalMetrics(
            precision_at_1=np.mean([m.precision_at_1 for m in metrics_list]),
            precision_at_5=np.mean([m.precision_at_5 for m in metrics_list]),
            precision_at_10=np.mean([m.precision_at_10 for m in metrics_list]),
            recall_at_5=np.mean([m.recall_at_5 for m in metrics_list]),
            recall_at_10=np.mean([m.recall_at_10 for m in metrics_list]),
            ndcg_at_5=np.mean([m.ndcg_at_5 for m in metrics_list]),
            ndcg_at_10=np.mean([m.ndcg_at_10 for m in metrics_list]),
            mrr=np.mean([m.mrr for m in metrics_list]),
            map_score=np.mean([m.map_score for m in metrics_list])
        )

    def _aggregate_affective_metrics(
        self,
        metrics_list: List[AffectiveMetrics]
    ) -> AffectiveMetrics:
        """Aggregate affective metrics across queries"""
        if not metrics_list:
            return AffectiveMetrics()

        return AffectiveMetrics(
            emotion_alignment=np.mean([m.emotion_alignment for m in metrics_list]),
            affective_coherence=np.mean([m.affective_coherence for m in metrics_list]),
            emotional_diversity=np.mean([m.emotional_diversity for m in metrics_list]),
            user_emotion_satisfaction=np.mean([m.user_emotion_satisfaction for m in metrics_list])
        )

    def save_results(self, filename: str = "experiment_results.json"):
        """Save all experiment results to file"""
        results_data = [r.to_dict() for r in self.results]
        path = self.output_dir / filename
        with open(path, 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"Results saved to {path}")

    def compare_experiments(self, experiment_names: List[str]) -> Dict[str, Any]:
        """Compare multiple experiments"""
        comparison = {}
        for name in experiment_names:
            result = next((r for r in self.results if r.experiment_name == name), None)
            if result:
                comparison[name] = {
                    'retrieval': result.retrieval_metrics.to_dict(),
                    'affective': result.affective_metrics.to_dict()
                }
        return comparison


class BaselineComparator:
    """
    Compare Affective-RAG against baseline approaches
    Supports RQ1 and RQ2 evaluation
    """

    def __init__(self, evaluator: Evaluator):
        self.evaluator = evaluator

    def run_baseline_comparison(
        self,
        test_cases: List[QueryTestCase],
        semantic_retriever: Callable,
        emotion_retriever: Callable,
        krag_retriever: Callable
    ) -> Dict[str, ExperimentResult]:
        """
        Run comparison between baselines and Affective-RAG

        Args:
            test_cases: Test cases for evaluation
            semantic_retriever: Semantic-only retrieval function
            emotion_retriever: Emotion-only retrieval function
            krag_retriever: Full Affective-RAG retrieval function

        Returns:
            Dict mapping approach name to results
        """
        results = {}

        # Semantic-only baseline
        results['semantic_only'] = self.evaluator.evaluate_retrieval(
            test_cases,
            semantic_retriever,
            'semantic_only',
            {'weights': {'semantic': 1.0, 'emotion': 0.0, 'knowledge': 0.0}}
        )

        # Emotion-only baseline
        results['emotion_only'] = self.evaluator.evaluate_retrieval(
            test_cases,
            emotion_retriever,
            'emotion_only',
            {'weights': {'semantic': 0.0, 'emotion': 1.0, 'knowledge': 0.0}}
        )

        # Full Affective-RAG
        results['affective_rag'] = self.evaluator.evaluate_retrieval(
            test_cases,
            krag_retriever,
            'affective_rag',
            {'weights': {'semantic': 0.5, 'emotion': 0.3, 'knowledge': 0.2}}
        )

        return results

    def generate_comparison_report(self, results: Dict[str, ExperimentResult]) -> str:
        """Generate comparison report"""
        report = []
        report.append("=" * 60)
        report.append("BASELINE COMPARISON REPORT")
        report.append("=" * 60)
        report.append("")

        metrics_to_compare = ['NDCG@10', 'MRR', 'P@5']
        affective_to_compare = ['emotion_alignment', 'affective_coherence']

        for metric in metrics_to_compare:
            report.append(f"\n{metric}:")
            for name, result in results.items():
                value = result.retrieval_metrics.to_dict().get(metric, 0)
                report.append(f"  {name}: {value:.4f}")

        report.append("\nAffective Metrics:")
        for metric in affective_to_compare:
            report.append(f"\n{metric}:")
            for name, result in results.items():
                value = result.affective_metrics.to_dict().get(metric, 0)
                report.append(f"  {name}: {value:.4f}")

        return "\n".join(report)


class AblationStudy:
    """
    Ablation study framework for Affective-RAG
    Supports RQ3 (multi-hop depth) and weight sensitivity analysis
    """

    def __init__(self, evaluator: Evaluator):
        self.evaluator = evaluator

    def weight_sensitivity_analysis(
        self,
        test_cases: List[QueryTestCase],
        retriever_factory: Callable[[Dict[str, float]], Callable],
        weight_configs: List[Dict[str, float]]
    ) -> List[ExperimentResult]:
        """
        Analyze sensitivity to retrieval weight configurations

        Args:
            test_cases: Test cases
            retriever_factory: Function that creates retriever given weights
            weight_configs: List of weight configurations to test

        Returns:
            List of experiment results
        """
        results = []

        for i, weights in enumerate(weight_configs):
            retriever = retriever_factory(weights)
            experiment_name = f"weight_config_{i}"

            result = self.evaluator.evaluate_retrieval(
                test_cases,
                retriever,
                experiment_name,
                {'weights': weights}
            )
            results.append(result)

        return results

    def multi_hop_analysis(
        self,
        test_cases: List[QueryTestCase],
        retriever_factory: Callable[[int], Callable],
        hop_depths: List[int] = [1, 2, 3]
    ) -> List[ExperimentResult]:
        """
        Analyze effect of multi-hop depth on retrieval quality
        Addresses RQ3

        Args:
            test_cases: Test cases
            retriever_factory: Function that creates retriever given hop depth
            hop_depths: List of hop depths to test

        Returns:
            List of experiment results
        """
        results = []

        for depth in hop_depths:
            retriever = retriever_factory(depth)
            experiment_name = f"{depth}_hop"

            result = self.evaluator.evaluate_retrieval(
                test_cases,
                retriever,
                experiment_name,
                {'hop_depth': depth}
            )
            results.append(result)

        return results

    def generate_ablation_report(
        self,
        results: List[ExperimentResult],
        analysis_type: str
    ) -> str:
        """Generate ablation study report"""
        report = []
        report.append("=" * 60)
        report.append(f"ABLATION STUDY: {analysis_type.upper()}")
        report.append("=" * 60)

        for result in results:
            report.append(f"\n{result.experiment_name}:")
            report.append(f"  Config: {result.config}")
            report.append(f"  NDCG@10: {result.retrieval_metrics.ndcg_at_10:.4f}")
            report.append(f"  MRR: {result.retrieval_metrics.mrr:.4f}")
            report.append(f"  Emotion Alignment: {result.affective_metrics.emotion_alignment:.4f}")

        return "\n".join(report)


def create_test_cases_from_data(
    queries: List[Dict[str, Any]],
    ground_truth: Dict[str, List[str]]
) -> List[QueryTestCase]:
    """
    Create test cases from query data and ground truth

    Args:
        queries: List of query dictionaries with 'id', 'text', 'emotions'
        ground_truth: Mapping from query_id to list of relevant item IDs

    Returns:
        List of QueryTestCase objects
    """
    test_cases = []

    for query in queries:
        query_id = query['id']
        if query_id in ground_truth:
            test_case = QueryTestCase(
                query_id=query_id,
                query_text=query['text'],
                user_emotions=query.get('emotions', {}),
                relevant_items=ground_truth[query_id],
                metadata=query.get('metadata')
            )
            test_cases.append(test_case)

    return test_cases


def default_weight_configs() -> List[Dict[str, float]]:
    """Default weight configurations for ablation study"""
    return [
        {'semantic': 1.0, 'emotion': 0.0, 'knowledge': 0.0},
        {'semantic': 0.0, 'emotion': 1.0, 'knowledge': 0.0},
        {'semantic': 0.0, 'emotion': 0.0, 'knowledge': 1.0},
        {'semantic': 0.5, 'emotion': 0.5, 'knowledge': 0.0},
        {'semantic': 0.5, 'emotion': 0.0, 'knowledge': 0.5},
        {'semantic': 0.0, 'emotion': 0.5, 'knowledge': 0.5},
        {'semantic': 0.5, 'emotion': 0.3, 'knowledge': 0.2},
        {'semantic': 0.4, 'emotion': 0.4, 'knowledge': 0.2},
        {'semantic': 0.3, 'emotion': 0.3, 'knowledge': 0.4}
    ]
