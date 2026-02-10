"""
Causal Necessity Analysis for Affective-RAG

Implements perturbation-based analysis to measure the system's
reliance on explicit graph evidence.

Paper Section IV.D - Explainability and Faithfulness
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import copy

from ..retrieval.krag_retriever import QueryContext, RetrievalResult
from .metrics import compute_faithfulness_necessity_score


@dataclass
class CausalAnalysisResult:
    """Result of causal necessity analysis for a single item."""
    content_id: str
    original_score: float
    perturbed_score: float
    fns_score: float
    perturbed_edge: Tuple[str, str]
    emotion_name: str


class CausalNecessityAnalyzer:
    """
    Perturbation-based analysis to measure graph evidence reliance.

    Paper Section IV.D.1 - Causal Necessity Analysis:
    1. Retrieve candidate M driven by emotion edge E
    2. Perturb: Remove edge E from graph
    3. Re-score M
    4. Compute FNS = (S_orig - S_perturbed) / S_orig
    """

    def __init__(self, retriever, knowledge_graph):
        """
        Initialize analyzer.

        Args:
            retriever: KRAGRetriever instance
            knowledge_graph: ContentKnowledgeGraph instance
        """
        self.retriever = retriever
        self.kg = knowledge_graph

    def analyze_single(
        self,
        query_context: QueryContext,
        target_content_id: str,
        emotion_to_perturb: str,
        k: int = 50
    ) -> CausalAnalysisResult:
        """
        Analyze causal necessity for a single item-emotion pair.

        Args:
            query_context: Query context
            target_content_id: Content ID to analyze
            emotion_to_perturb: Emotion edge to remove (e.g., 'fear')
            k: Retrieval depth

        Returns:
            CausalAnalysisResult with FNS score
        """
        results = self.retriever.retrieve(query_context, k=k)
        original_score = self._find_score(results, target_content_id)

        edge_key = (target_content_id, f'emotion_{emotion_to_perturb}')
        edge_data = self._remove_edge(edge_key)

        if edge_data is None:
            return CausalAnalysisResult(
                content_id=target_content_id,
                original_score=original_score,
                perturbed_score=original_score,
                fns_score=0.0,
                perturbed_edge=edge_key,
                emotion_name=emotion_to_perturb
            )

        try:
            results_perturbed = self.retriever.retrieve(query_context, k=k)
            perturbed_score = self._find_score(results_perturbed, target_content_id)
        finally:
            self._restore_edge(edge_key, edge_data)

        fns = compute_faithfulness_necessity_score(original_score, perturbed_score)

        return CausalAnalysisResult(
            content_id=target_content_id,
            original_score=original_score,
            perturbed_score=perturbed_score,
            fns_score=fns,
            perturbed_edge=edge_key,
            emotion_name=emotion_to_perturb
        )

    def analyze_batch(
        self,
        query_context: QueryContext,
        content_ids: List[str],
        emotions: Optional[List[str]] = None,
        k: int = 50
    ) -> List[CausalAnalysisResult]:
        """
        Analyze causal necessity for multiple items.

        Args:
            query_context: Query context
            content_ids: List of content IDs to analyze
            emotions: Emotions to perturb (default: all 6)
            k: Retrieval depth

        Returns:
            List of CausalAnalysisResult
        """
        if emotions is None:
            emotions = ['happiness', 'sadness', 'anger', 'fear', 'surprise', 'disgust']

        results = []

        for content_id in content_ids:
            for emotion in emotions:
                result = self.analyze_single(
                    query_context, content_id, emotion, k
                )
                if result.fns_score > 0:
                    results.append(result)

        return results

    def compute_aggregate_fns(
        self,
        query_contexts: List[QueryContext],
        top_k_per_query: int = 5
    ) -> Dict[str, float]:
        """
        Compute aggregate FNS across multiple queries.

        For each query, analyze top-k results and compute average FNS.

        Args:
            query_contexts: List of query contexts
            top_k_per_query: Number of top results to analyze per query

        Returns:
            Dict with aggregate statistics
        """
        all_fns_scores = []
        high_fns_count = 0
        total_analyzed = 0

        for query_context in query_contexts:
            results = self.retriever.retrieve(query_context, k=top_k_per_query)

            for result in results:
                dominant_emotion = self._get_dominant_emotion(result.content_id)
                if dominant_emotion:
                    analysis = self.analyze_single(
                        query_context, result.content_id, dominant_emotion
                    )
                    all_fns_scores.append(analysis.fns_score)
                    total_analyzed += 1

                    if analysis.fns_score > 0.5:
                        high_fns_count += 1

        return {
            'mean_fns': float(np.mean(all_fns_scores)) if all_fns_scores else 0.0,
            'median_fns': float(np.median(all_fns_scores)) if all_fns_scores else 0.0,
            'std_fns': float(np.std(all_fns_scores)) if all_fns_scores else 0.0,
            'high_fns_ratio': high_fns_count / total_analyzed if total_analyzed > 0 else 0.0,
            'total_analyzed': total_analyzed
        }

    def _find_score(self, results: List[RetrievalResult], content_id: str) -> float:
        """Find combined score for a content ID in results."""
        for result in results:
            if result.content_id == content_id:
                return result.combined_score
        return 0.0

    def _remove_edge(self, edge_key: Tuple[str, str]) -> Optional[Dict]:
        """Remove edge from graph and return its data for restoration."""
        source, target = edge_key

        if not self.kg.graph.has_edge(source, target):
            return None

        edge_data = dict(self.kg.graph.get_edge_data(source, target, default={}))

        if edge_data:
            edge_data = dict(list(edge_data.values())[0]) if edge_data else {}

        self.kg.graph.remove_edge(source, target)

        return edge_data

    def _restore_edge(self, edge_key: Tuple[str, str], edge_data: Dict):
        """Restore previously removed edge."""
        source, target = edge_key
        self.kg.graph.add_edge(source, target, **edge_data)

    def _get_dominant_emotion(self, content_id: str) -> Optional[str]:
        """Get the dominant emotion for a content item."""
        if content_id not in self.kg.graph:
            return None

        emotions = {}
        for _, target, data in self.kg.graph.out_edges(content_id, data=True):
            if data.get('relation') == 'evokes' and target.startswith('emotion_'):
                emotion_name = target.replace('emotion_', '')
                emotions[emotion_name] = data.get('weight', 0.0)

        if not emotions:
            return None

        return max(emotions, key=emotions.get)


def run_causal_necessity_experiment(
    retriever,
    knowledge_graph,
    test_cases: List,
    output_path: str = "./results/causal_analysis.json"
) -> Dict[str, Any]:
    """
    Run full causal necessity experiment.

    Args:
        retriever: KRAGRetriever instance
        knowledge_graph: ContentKnowledgeGraph instance
        test_cases: List of QueryTestCase objects
        output_path: Path to save results

    Returns:
        Aggregate FNS statistics
    """
    import json
    from pathlib import Path

    analyzer = CausalNecessityAnalyzer(retriever, knowledge_graph)

    query_contexts = []
    for tc in test_cases[:100]:
        from ..core.emotion_detection import EmotionProfile
        emotions = EmotionProfile(
            happiness=tc.target_emotions.get('happiness', 0.0),
            sadness=tc.target_emotions.get('sadness', 0.0),
            anger=tc.target_emotions.get('anger', 0.0),
            fear=tc.target_emotions.get('fear', 0.0),
            surprise=tc.target_emotions.get('surprise', 0.0),
            disgust=tc.target_emotions.get('disgust', 0.0)
        )
        qc = QueryContext(
            query_text=tc.query_text,
            user_emotions=emotions,
            query_embedding=np.zeros(768),
            emotion_embedding=np.zeros(768)
        )
        query_contexts.append(qc)

    results = analyzer.compute_aggregate_fns(query_contexts)

    print("\nCausal Necessity Analysis Results:")
    print(f"  Mean FNS: {results['mean_fns']:.4f}")
    print(f"  Median FNS: {results['median_fns']:.4f}")
    print(f"  High FNS Ratio (>0.5): {results['high_fns_ratio']:.4f}")
    print(f"  Total Analyzed: {results['total_analyzed']}")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w') as f:
        json.dump(results, f, indent=2)

    return results
