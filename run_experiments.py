#!/usr/bin/env python3
"""
Experiment Runner for KRAG

This script runs the main evaluation routines used to compare retrieval variants
and to analyze sensitivity and explainability. It is designed to be run from the
repository root and uses the implementation under `src/`.

Usage:
    python run_experiments.py --experiment all
    python run_experiments.py --experiment comparative
    python run_experiments.py --experiment sensitivity
    python run_experiments.py --experiment causal
    python run_experiments.py --experiment threshold-sensitivity
    
    # With explicit thresholds:
    python run_experiments.py --experiment comparative \\
        --semantic-threshold 0.5 --affective-threshold 0.6 \\
        --min-test-cases 30 --num-queries 100
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
load_dotenv()

import numpy as np
from scipy import stats

# Threshold grids for sensitivity analysis
SEMANTIC_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]
AFFECTIVE_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.85]
DEFAULT_MIN_TEST_CASES = 30
DEFAULT_MIN_TEST_CASES_SENSITIVITY = 100

# Defaults are tuned for practicality: comparative runs need larger \(n\) for power,
# while threshold-sensitivity needs conservative bounds to avoid grid blow-up.
DEFAULT_MIN_TEST_CASES_COMPARATIVE = 200
DEFAULT_NUM_QUERIES_COMPARATIVE = 500
DEFAULT_NUM_QUERIES_THRESHOLD_SENSITIVITY = 100
DEFAULT_NUM_QUERIES_OTHER = 50


def get_movie_affective_signatures(system):
    """Extract movie affective signatures from knowledge graph EVOKES edges."""
    signatures = {}
    emotion_idx = {
        'happiness': 0, 'sadness': 1, 'anger': 2,
        'fear': 3, 'surprise': 4, 'disgust': 5
    }

    kg = system.knowledge_graph
    for item in system.content_items:
        content_id = str(item.id)
        signature = np.zeros(6)

        if content_id in kg.graph:
            for _, target, data in kg.graph.out_edges(content_id, data=True):
                if data.get('relation') == 'evokes' and target.startswith('emotion_'):
                    emotion_name = target.replace('emotion_', '')
                    if emotion_name in emotion_idx:
                        signature[emotion_idx[emotion_name]] = data.get('weight', 0.0)

        signatures[content_id] = signature

    return signatures


def compute_confidence_interval(data: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """Compute confidence interval for a list of values."""
    if len(data) < 2:
        return (0.0, 0.0)
    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return (mean - h, mean + h)


def compute_effect_size(group1: List[float], group2: List[float]) -> float:
    """Compute Cohen's d effect size between two groups."""
    if len(group1) < 2 or len(group2) < 2:
        return 0.0
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-8:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def compute_paired_effect_size(differences: List[float]) -> float:
    """
    Compute paired effect size (Cohen's dz) for a list of paired differences.

    dz = mean(d) / std(d), where d are within-pair differences.
    """
    if len(differences) < 2:
        return 0.0
    d = np.array(differences, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return 0.0
    std = np.std(d, ddof=1)
    if std < 1e-8:
        return 0.0
    return float(np.mean(d) / std)


def count_valid_test_cases(
    content_embedder,
    content_items,
    content_embeddings,
    movie_signatures,
    semantic_threshold: float,
    affective_threshold: float,
    num_queries: int = 100,
    min_relevant: int = 3
) -> Tuple[int, float]:
    """
    Quick count of how many valid test cases can be generated with given thresholds.
    Returns (count, avg_relevant_items).
    """
    from krag.evaluation.synthetic_testset import SyntheticTestSetGenerator
    
    generator = SyntheticTestSetGenerator(
        content_embedder=content_embedder,
        semantic_threshold=semantic_threshold,
        affective_threshold=affective_threshold
    )
    
    test_cases = generator.generate_test_set(
        content_items=content_items,
        content_embeddings=content_embeddings,
        movie_affective_signatures=movie_signatures,
        num_queries=num_queries,
        min_relevant=min_relevant
    )
    
    if test_cases:
        avg_relevant = np.mean([len(tc.relevant_items) for tc in test_cases])
    else:
        avg_relevant = 0.0
    
    return len(test_cases), avg_relevant


def run_threshold_sensitivity(system, num_queries=100, output_dir="./results"):
    """
    Threshold Sensitivity Analysis.
    
    Tests a grid of semantic/affective threshold combinations to find optimal
    thresholds that produce sufficient test cases for statistical validity.
    
    Outputs:
    - threshold_analysis.json: Full grid search results
    - threshold_heatmap.png: Threshold grid visualization
    """
    print("\n" + "=" * 60)
    print("THRESHOLD SENSITIVITY ANALYSIS")
    print("=" * 60)
    
    from krag.evaluation.synthetic_testset import SyntheticTestSetGenerator
    
    content_items = system.content_items
    
    # Load embeddings (strict): GCS only, no local cache, no generation
    from krag.data.adapters import DatasetPath
    if not system.data_processor.adapter.exists(DatasetPath.EMBEDDINGS):
        raise FileNotFoundError(
            f"Missing required embeddings artifact in GCS for threshold-sensitivity: "
            f"gs://{system.config.gcs_bucket}/{system.config.gcs_base_path}/{DatasetPath.EMBEDDINGS.value}"
        )
    print("Loading embeddings from GCS...")
    cached = system.data_processor.adapter.load_numpy(DatasetPath.EMBEDDINGS)
    content_embeddings = cached["semantic"][:len(content_items)]
    print(f"Loaded embeddings from GCS: {content_embeddings.shape}")
    
    movie_signatures = get_movie_affective_signatures(system)
    
    print(f"\nTesting {len(SEMANTIC_THRESHOLDS)} x {len(AFFECTIVE_THRESHOLDS)} = "
          f"{len(SEMANTIC_THRESHOLDS) * len(AFFECTIVE_THRESHOLDS)} threshold combinations...")
    print(f"Semantic thresholds: {SEMANTIC_THRESHOLDS}")
    print(f"Affective thresholds: {AFFECTIVE_THRESHOLDS}")
    print(f"Target queries per combination: {num_queries}")
    print()
    
    grid_results = []
    heatmap_data = np.zeros((len(AFFECTIVE_THRESHOLDS), len(SEMANTIC_THRESHOLDS)))
    
    for i, aff_thresh in enumerate(AFFECTIVE_THRESHOLDS):
        for j, sem_thresh in enumerate(SEMANTIC_THRESHOLDS):
            print(f"  Testing semantic={sem_thresh:.2f}, affective={aff_thresh:.2f}...", end=" ")
            
            count, avg_relevant = count_valid_test_cases(
                content_embedder=system.content_embedder,
                content_items=content_items,
                content_embeddings=content_embeddings,
                movie_signatures=movie_signatures,
                semantic_threshold=sem_thresh,
                affective_threshold=aff_thresh,
                num_queries=num_queries,
                min_relevant=3
            )
            
            status = "valid" if count >= DEFAULT_MIN_TEST_CASES else "insufficient"
            print(f"{count} test cases ({status})")
            
            heatmap_data[i, j] = count
            
            grid_results.append({
                "semantic_threshold": sem_thresh,
                "affective_threshold": aff_thresh,
                "test_cases_generated": count,
                "avg_relevant_items": round(avg_relevant, 2),
                "status": status
            })
    
    # Find recommended threshold (highest thresholds with >= min cases)
    valid_results = [r for r in grid_results if r["status"] == "valid"]
    if valid_results:
        # Sort by sum of thresholds (prefer higher thresholds)
        valid_results.sort(
            key=lambda r: r["semantic_threshold"] + r["affective_threshold"],
            reverse=True
        )
        recommended = valid_results[0]
    else:
        # Fall back to the one with most test cases
        recommended = max(grid_results, key=lambda r: r["test_cases_generated"])
    
    results = {
        "experiment": "threshold_sensitivity",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "num_queries_requested": num_queries,
            "min_test_cases_required": DEFAULT_MIN_TEST_CASES,
            "semantic_thresholds_tested": SEMANTIC_THRESHOLDS,
            "affective_thresholds_tested": AFFECTIVE_THRESHOLDS,
            "content_items_count": len(content_items)
        },
        "grid_results": grid_results,
        "recommended_threshold": {
            "semantic": recommended["semantic_threshold"],
            "affective": recommended["affective_threshold"],
            "expected_test_cases": recommended["test_cases_generated"],
            "reason": f"Highest threshold pair with >= {DEFAULT_MIN_TEST_CASES} test cases" 
                      if recommended["status"] == "valid"
                      else f"Best available (max {recommended['test_cases_generated']} cases)"
        },
        "summary": {
            "total_combinations_tested": len(grid_results),
            "valid_combinations": len([r for r in grid_results if r["status"] == "valid"]),
            "max_test_cases": max(r["test_cases_generated"] for r in grid_results),
            "min_test_cases": min(r["test_cases_generated"] for r in grid_results)
        }
    }
    
    # Save JSON results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / "threshold_analysis.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate heatmap visualization
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend for Colab/servers
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(heatmap_data, cmap='YlGnBu', aspect='auto')
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Number of Valid Test Cases", rotation=-90, va="bottom", fontsize=12)
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(SEMANTIC_THRESHOLDS)))
        ax.set_yticks(np.arange(len(AFFECTIVE_THRESHOLDS)))
        ax.set_xticklabels([f"{t:.2f}" for t in SEMANTIC_THRESHOLDS])
        ax.set_yticklabels([f"{t:.2f}" for t in AFFECTIVE_THRESHOLDS])
        
        ax.set_xlabel("Semantic Threshold", fontsize=12)
        ax.set_ylabel("Affective Threshold", fontsize=12)
        ax.set_title("Threshold Sensitivity Analysis\n(Cell value = # valid test cases)", fontsize=14)
        
        # Add text annotations
        for i in range(len(AFFECTIVE_THRESHOLDS)):
            for j in range(len(SEMANTIC_THRESHOLDS)):
                count = int(heatmap_data[i, j])
                color = "white" if count > heatmap_data.max() / 2 else "black"
                text = ax.text(j, i, count, ha="center", va="center", color=color, fontsize=10)
        
        # Mark recommended threshold
        for idx, r in enumerate(grid_results):
            if (r["semantic_threshold"] == recommended["semantic_threshold"] and 
                r["affective_threshold"] == recommended["affective_threshold"]):
                i = AFFECTIVE_THRESHOLDS.index(r["affective_threshold"])
                j = SEMANTIC_THRESHOLDS.index(r["semantic_threshold"])
                rect = plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, 
                                     edgecolor='red', linewidth=3)
                ax.add_patch(rect)
                break
        
        # Add legend for recommended
        ax.plot([], [], 's', markerfacecolor='none', markeredgecolor='red', 
                markersize=15, markeredgewidth=3, label='Recommended')
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(output_path / "threshold_heatmap.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\nHeatmap saved to: {output_path / 'threshold_heatmap.png'}")
        
    except ImportError:
        print("\nWarning: matplotlib not available, skipping heatmap generation")
    
    # Print summary
    print("\n" + "-" * 60)
    print("THRESHOLD SENSITIVITY RESULTS:")
    print("-" * 60)
    print(f"Combinations tested: {len(grid_results)}")
    print(f"Valid combinations (>= {DEFAULT_MIN_TEST_CASES} cases): "
          f"{len([r for r in grid_results if r['status'] == 'valid'])}")
    print(f"\nRECOMMENDED THRESHOLDS:")
    print(f"  Semantic: {recommended['semantic_threshold']:.2f}")
    print(f"  Affective: {recommended['affective_threshold']:.2f}")
    print(f"  Expected test cases: {recommended['test_cases_generated']}")
    print("-" * 60)
    print(f"\nResults saved to: {output_path / 'threshold_analysis.json'}")
    
    return results


def run_comparative_analysis(
    system, 
    num_queries: int = DEFAULT_NUM_QUERIES_COMPARATIVE,
    output_dir="./results",
    semantic_threshold: Optional[float] = None,
    affective_threshold: Optional[float] = None,
    min_test_cases: int = DEFAULT_MIN_TEST_CASES_COMPARATIVE,
    allow_insufficient: bool = False,
    output_suffix: str = ""
):
    """
    Comparative retrieval analysis.

    Compares: BM25, Semantic (vector-only), KRAG (α=1.0), KRAG (α=0.5), KRAG (α=0.3).
    Metrics: Semantic Recall@K, AP@K, ADE.
    
    Args:
        system: Initialized ARAGSystem
        num_queries: Number of test queries to generate
        output_dir: Directory for output files
        semantic_threshold: Explicit semantic threshold override
        affective_threshold: Explicit affective threshold override
        min_test_cases: Minimum test cases required for valid results
        allow_insufficient: If True, run even with < min_test_cases
        output_suffix: Suffix for output filename (e.g., "_t50" for 50 queries)
    """
    print("\n" + "=" * 60)
    print("COMPARATIVE RETRIEVAL ANALYSIS")
    print("=" * 60)

    from krag.evaluation.synthetic_testset import SyntheticTestSetGenerator
    from krag.experiments.runner import run_ablation_study_enhanced
    from krag.retrieval.krag_retriever import RetrieverFactory, QueryContext
    from krag.core.emotion_detection import EmotionProfile

    # Use provided thresholds or defaults
    sem_thresh = semantic_threshold if semantic_threshold is not None else 0.5
    aff_thresh = affective_threshold if affective_threshold is not None else 0.6
    explicit_thresholds = semantic_threshold is not None or affective_threshold is not None
    
    print(f"\nConfiguration:")
    print(f"  Semantic threshold (requested): {sem_thresh}")
    print(f"  Affective threshold (requested): {aff_thresh}")
    print(f"  Requested queries: {num_queries}")
    print(f"  Min test cases required: {min_test_cases}")
    print(f"  Explicit thresholds: {explicit_thresholds}")
    if allow_insufficient:
        print("  NOTE: comparative analysis enforces minimum sample size requirements.")

    if min_test_cases > num_queries:
        error_result = {
            "experiment_config": {
                "semantic_threshold_requested": sem_thresh,
                "affective_threshold_requested": aff_thresh,
                "num_queries_requested": num_queries,
                "min_test_cases_required": min_test_cases,
                "status": "aborted_invalid_config"
            },
            "error": f"Invalid config: min_test_cases ({min_test_cases}) cannot exceed num_queries ({num_queries}).",
            "timestamp": datetime.now().isoformat()
        }
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        filename = f"comparative_analysis{output_suffix}.json"
        with open(output_path / filename, 'w') as f:
            json.dump(error_result, f, indent=2)
        print(f"\nEXPERIMENT ABORTED: {error_result['error']}")
        return error_result

    content_items = system.content_items
    warnings = []

    # Load embeddings (strict): GCS only, no local cache, no generation
    from krag.data.adapters import DatasetPath
    if not system.data_processor.adapter.exists(DatasetPath.EMBEDDINGS):
        raise FileNotFoundError(
            f"Missing required embeddings artifact in GCS for comparative analysis: "
            f"gs://{system.config.gcs_bucket}/{system.config.gcs_base_path}/{DatasetPath.EMBEDDINGS.value}"
        )
    print("\nLoading embeddings from GCS...")
    cached = system.data_processor.adapter.load_numpy(DatasetPath.EMBEDDINGS)
    content_embeddings = cached['semantic'][:len(content_items)]
    print(f"Loaded embeddings from GCS: {content_embeddings.shape}")

    movie_signatures = get_movie_affective_signatures(system)

    embeddings_source = "gcs"

    def _generate_agreement_cases_for_thresholds(sem: float, aff: float, min_relevant: int = 3):
        """
        Generate agreement cases (semantic AND affective relevant).
        """
        print(f"\nGenerating agreement test cases (semantic∩affective), target={num_queries}, "
              f"semantic={sem}, affective={aff}, min_relevant={min_relevant}...")

        generator_local = SyntheticTestSetGenerator(
            content_embedder=system.content_embedder,
            semantic_threshold=sem,
            affective_threshold=aff
        )

        cases = generator_local.generate_test_set(
            content_items=content_items,
            content_embeddings=content_embeddings,
            movie_affective_signatures=movie_signatures,
            num_queries=num_queries,
            min_relevant=min_relevant
        )

        print(f"Generated {len(cases)} agreement test cases")
        return cases

    # If thresholds are not provided explicitly, search the threshold grid.
    # Every attempted threshold pair is recorded in the results JSON.
    # If no threshold pair yields >= min_test_cases, the experiment aborts.
    threshold_selection = {
        "strategy": "explicit" if explicit_thresholds else "grid_search_descending_sum",
        "requested": {"semantic": sem_thresh, "affective": aff_thresh},
        "attempts": [],
        "selected": None,
    }

    selected_sem, selected_aff = sem_thresh, aff_thresh
    agreement_cases = []

    if explicit_thresholds:
        agreement_cases = _generate_agreement_cases_for_thresholds(selected_sem, selected_aff, min_relevant=3)
        threshold_selection["selected"] = {"semantic": selected_sem, "affective": selected_aff, "reason": "explicit_thresholds"}
    else:
        # Order threshold pairs by descending (semantic + affective), prefer higher thresholds.
        candidates = [(s, a) for s in SEMANTIC_THRESHOLDS for a in AFFECTIVE_THRESHOLDS]
        candidates.sort(key=lambda x: x[0] + x[1], reverse=True)

        print("\nNo explicit thresholds provided. Searching threshold grid to reach required sample size.")
        for s, a in candidates:
            cases = _generate_agreement_cases_for_thresholds(s, a, min_relevant=3)
            threshold_selection["attempts"].append({
                "semantic": s,
                "affective": a,
                "num_cases_generated": len(cases),
                "meets_min_test_cases": len(cases) >= min_test_cases
            })
            if len(cases) >= min_test_cases:
                selected_sem, selected_aff = s, a
                agreement_cases = cases
                threshold_selection["selected"] = {
                    "semantic": s,
                    "affective": a,
                    "reason": f"first_pair_meeting_min_test_cases_{min_test_cases}"
                }
                print(f"\nSelected thresholds: semantic={s}, affective={a} (agreement cases={len(cases)})")
                break

        if threshold_selection["selected"] is None:
            warning_msg = (
                f"Agreement set insufficient for all tested threshold pairs. "
                f"Required >= {min_test_cases} agreement cases; none met the requirement."
            )
            warnings.append(warning_msg)
            print(f"\n{warning_msg}")
            agreement_cases = []

    agreement_thresholds_used = {
        "semantic": selected_sem,
        "affective": selected_aff,
        "query_type": "agreement",
        "threshold_selection": threshold_selection,
    }

    def _generate_dissonance_cases(sem: float, aff: float):
        """
        Generate dissonance/conflict cases (semantic intent conflicts with target emotion).
        """
        print(f"\nGenerating dissonance test cases (conflict scenarios), target={num_queries}...")

        generator_local = SyntheticTestSetGenerator(
            content_embedder=system.content_embedder,
            semantic_threshold=sem,
            affective_threshold=aff
        )

        cases = generator_local.generate_dissonance_queries(
            content_items=content_items,
            content_embeddings=content_embeddings,
            movie_affective_signatures=movie_signatures,
            num_queries=num_queries
        )

        thresholds_used_local = {
            "semantic": sem,
            "affective": aff,
            "query_type": "dissonance",
            "num_queries_requested": num_queries,
            "threshold_selection": threshold_selection
        }

        print(f"Generated {len(cases)} dissonance test cases")
        if len(cases) < min_test_cases:
            warning_msg = (
                f"Dissonance set insufficient: generated {len(cases)} cases, required >= {min_test_cases}. "
                f"Increase --num-queries or lower --min-test-cases."
            )
            warnings.append(warning_msg)
            print(f"\n{warning_msg}")
            return [], thresholds_used_local, "aborted_insufficient_cases"

        return cases, thresholds_used_local, "valid"

    agreement_status = "valid" if len(agreement_cases) >= min_test_cases else "aborted_insufficient_cases"
    if agreement_status != "valid":
        warnings.append(
            f"Agreement set insufficient: generated {len(agreement_cases)} cases, required >= {min_test_cases}."
        )

    dissonance_cases, dissonance_thresholds_used, dissonance_status = _generate_dissonance_cases(selected_sem, selected_aff)

    # Strict behavior: if either evaluation can't be generated at the required sample size, abort.
    if not agreement_cases or not dissonance_cases:
        error_result = {
            "experiment_config": {
                "semantic_threshold": sem_thresh,
                "affective_threshold": aff_thresh,
                "num_queries_requested": num_queries,
                "min_test_cases_required": min_test_cases,
                "status": "aborted_insufficient_cases",
                "explicit_thresholds": explicit_thresholds
            },
            "error": "Comparative analysis aborted: insufficient test cases.",
            "evaluations": {
                "agreement": {
                    "status": agreement_status,
                    "num_test_cases_generated": len(agreement_cases),
                    "thresholds_used": agreement_thresholds_used
                },
                "dissonance": {
                    "status": dissonance_status,
                    "num_test_cases_generated": len(dissonance_cases),
                    "thresholds_used": dissonance_thresholds_used
                }
            },
            "warnings": warnings,
            "timestamp": datetime.now().isoformat()
        }
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        filename = f"comparative_analysis{output_suffix}.json"
        with open(output_path / filename, 'w') as f:
            json.dump(error_result, f, indent=2)
        print(f"\nEXPERIMENT ABORTED: insufficient test cases.")
        return error_result

    print("\nInitializing retrievers...")

    from krag.retrieval.bm25_retriever import create_bm25_retriever_from_content_items

    # Baseline: BM25 lexical
    bm25_retriever = create_bm25_retriever_from_content_items(content_items)

    # Baseline: true semantic-only (vector search only, no graph, no affective term)
    semantic_vector_only = RetrieverFactory.create_retriever(
        "semantic",
        vector_store=system.vector_store
    )

    # KRAG variants (alpha-weighted relevance vs affective distance)
    krag_alpha_1 = RetrieverFactory.create_retriever(
        "krag",
        vector_store=system.vector_store,
        knowledge_graph=system.knowledge_graph,
        krag_encoder=system.krag_encoder,
        alpha=1.0,
        # Important for honest α comparisons: allow affective candidates into the pool
        candidate_pool="semantic+emotion"
    )

    # Ablation 2: Balanced (alpha=0.5)
    krag_alpha_05 = RetrieverFactory.create_retriever(
        "krag",
        vector_store=system.vector_store,
        knowledge_graph=system.knowledge_graph,
        krag_encoder=system.krag_encoder,
        alpha=0.5,
        candidate_pool="semantic+emotion"
    )

    # Ablation 3: Affective-focused (alpha=0.3)
    krag_alpha_03 = RetrieverFactory.create_retriever(
        "krag",
        vector_store=system.vector_store,
        knowledge_graph=system.knowledge_graph,
        krag_encoder=system.krag_encoder,
        alpha=0.3,
        candidate_pool="semantic+emotion"
    )

    content_id_set = {str(item.id) for item in content_items}

    def make_retriever_fn(retriever):
        def retrieve_fn(query_context, k=10):
            query_context.allowed_content_ids = content_id_set
            return retriever.retrieve(query_context, k=k)
        return retrieve_fn

    def _run_eval_block(eval_name: str, cases, thresholds_used_local: Dict[str, Any], status: str):
        if not cases:
            return {
                "experiment_config": {
                    "status": status,
                    "num_test_cases_generated": 0,
                    "min_test_cases_required": min_test_cases,
                    "thresholds_used": thresholds_used_local,
                },
                "methods": {},
                "statistical_analysis": {},
            }

        print("\nRunning ablation study...")

        retrievers = {
            'BM25': make_retriever_fn(bm25_retriever),
            'Semantic (vector-only)': make_retriever_fn(semantic_vector_only),
            'KRAG (α=1.0)': make_retriever_fn(krag_alpha_1),
            'KRAG (α=0.5)': make_retriever_fn(krag_alpha_05),
            'KRAG (α=0.3)': make_retriever_fn(krag_alpha_03),
        }

        method_results, per_query_scores = run_ablation_study_enhanced(
            test_cases=cases,
            retrievers=retrievers,
            movie_affective_signatures=movie_signatures,
            content_embedder=system.content_embedder,
            k=10
        )

        # Basic per-method statistics (paired stats are added in a later phase)
        statistical_analysis = {}
        for method, scores in per_query_scores.items():
            ade_scores = np.array(scores.get('ade_scores', []), dtype=float)
            ap5_scores = np.array(scores.get('ap5_scores', []), dtype=float)

            ade_finite = ade_scores[np.isfinite(ade_scores)]
            ap5_finite = ap5_scores[np.isfinite(ap5_scores)]

            if len(ade_finite) >= 2:
                ade_ci = compute_confidence_interval(ade_finite.tolist())
            else:
                ade_ci = (0.0, 0.0)

            if len(ap5_finite) >= 2:
                ap5_ci = compute_confidence_interval(ap5_finite.tolist())
            else:
                ap5_ci = (0.0, 0.0)

            statistical_analysis[method] = {
                "ADE_mean": float(np.mean(ade_finite)) if len(ade_finite) else float('inf'),
                "ADE_std": float(np.std(ade_finite)) if len(ade_finite) else float('inf'),
                "ADE_95_CI": [round(ade_ci[0], 4), round(ade_ci[1], 4)],
                "AP@5_mean": float(np.mean(ap5_finite)) if len(ap5_finite) else 0.0,
                "AP@5_std": float(np.std(ap5_finite)) if len(ap5_finite) else 0.0,
                "AP@5_95_CI": [round(ap5_ci[0], 4), round(ap5_ci[1], 4)],
                "n_samples": {
                    "ADE": int(len(ade_finite)),
                    "AP@5": int(len(ap5_finite)),
                    "total_queries": int(len(scores.get("query_ids", []))) if scores.get("query_ids") else int(len(ade_scores))
                }
            }

        # Paired comparisons (Affective vs Semantic) on the same queries
        def _paired(metric_key: str, method_a: str, method_b: str):
            """
            Return paired arrays (a, b) filtered to finite values and aligned by query_id.
            """
            if method_a not in per_query_scores or method_b not in per_query_scores:
                return np.array([]), np.array([]), []

            a = per_query_scores[method_a]
            b = per_query_scores[method_b]

            a_ids = a.get("query_ids", [])
            b_ids = b.get("query_ids", [])
            if a_ids and b_ids and a_ids == b_ids:
                a_vals = np.array(a.get(metric_key, []), dtype=float)
                b_vals = np.array(b.get(metric_key, []), dtype=float)
                mask = np.isfinite(a_vals) & np.isfinite(b_vals)
                return a_vals[mask], b_vals[mask], [qid for qid, keep in zip(a_ids, mask) if keep]

            # Fallback alignment by id (shouldn't happen, but keeps stats honest)
            a_map = {qid: val for qid, val in zip(a_ids, a.get(metric_key, []))}
            b_map = {qid: val for qid, val in zip(b_ids, b.get(metric_key, []))}
            common = sorted(set(a_map.keys()) & set(b_map.keys()))
            a_vals = np.array([a_map[qid] for qid in common], dtype=float)
            b_vals = np.array([b_map[qid] for qid in common], dtype=float)
            mask = np.isfinite(a_vals) & np.isfinite(b_vals)
            common_kept = [qid for qid, keep in zip(common, mask) if keep]
            return a_vals[mask], b_vals[mask], common_kept

        method_aff = "KRAG (α=0.3)"
        method_sem = "KRAG (α=1.0)"
        paired_block = {}

        # ADE: lower is better. diff = aff - sem (negative => affective better)
        aff_ade, sem_ade, ade_ids = _paired("ade_scores", method_aff, method_sem)
        if len(aff_ade) >= 2:
            diffs = (aff_ade - sem_ade).tolist()
            t_stat, p_value = stats.ttest_rel(aff_ade, sem_ade)
            ci = compute_confidence_interval(diffs)
            paired_block["ADE"] = {
                "n_pairs": int(len(diffs)),
                "mean_diff_aff_minus_sem": round(float(np.mean(diffs)), 6),
                "std_diff": round(float(np.std(diffs, ddof=1)), 6),
                "mean_diff_95_CI": [round(ci[0], 6), round(ci[1], 6)],
                "t_statistic": round(float(t_stat), 6),
                "p_value": round(float(p_value), 8),
                "effect_size_cohens_dz": round(compute_paired_effect_size(diffs), 6),
            }

        # AP@5: higher is better. diff = aff - sem (positive => affective better)
        aff_ap5, sem_ap5, ap5_ids = _paired("ap5_scores", method_aff, method_sem)
        if len(aff_ap5) >= 2:
            diffs = (aff_ap5 - sem_ap5).tolist()
            t_stat, p_value = stats.ttest_rel(aff_ap5, sem_ap5)
            ci = compute_confidence_interval(diffs)
            paired_block["AP@5"] = {
                "n_pairs": int(len(diffs)),
                "mean_diff_aff_minus_sem": round(float(np.mean(diffs)), 6),
                "std_diff": round(float(np.std(diffs, ddof=1)), 6),
                "mean_diff_95_CI": [round(ci[0], 6), round(ci[1], 6)],
                "t_statistic": round(float(t_stat), 6),
                "p_value": round(float(p_value), 8),
                "effect_size_cohens_dz": round(compute_paired_effect_size(diffs), 6),
            }

        if paired_block:
            statistical_analysis["paired_comparison_affective_vs_semantic"] = paired_block

        print("\n" + "-" * 60)
        print(f"ABLATION STUDY RESULTS ({eval_name.upper()}):")
        print("-" * 60)
        print(f"{'Method':<20} {'SR@5':>8} {'SR@10':>8} {'AP@5':>8} {'AP@10':>8} {'ADE':>8}")
        print("-" * 60)
        for method, metrics in method_results.items():
            print(f"{method:<20} {metrics['Semantic_Recall@5']:>8.3f} {metrics['Semantic_Recall@10']:>8.3f} "
                  f"{metrics['AP@5']:>8.3f} {metrics['AP@10']:>8.3f} {metrics['ADE']:>8.3f}")
        print("-" * 60)
        print(f"\nTest cases used ({eval_name}): {len(cases)}")

        return {
            "experiment_config": {
                "status": status,
                "num_test_cases_generated": len(cases),
                "min_test_cases_required": min_test_cases,
                "avg_relevant_items_per_query": round(np.mean([len(tc.relevant_items) for tc in cases]), 2) if cases else 0.0,
                "thresholds_used": thresholds_used_local,
                "content_items_count": len(content_items),
            },
            "methods": method_results,
            "statistical_analysis": statistical_analysis,
            # Full per-query traces (query-aligned, NaN for undefined values).
            "per_query_scores": per_query_scores,
        }

    agreement_block = _run_eval_block("agreement", agreement_cases, agreement_thresholds_used, agreement_status)
    dissonance_block = _run_eval_block("dissonance", dissonance_cases, dissonance_thresholds_used, dissonance_status)

    enhanced_results = {
        "experiment_config": {
            "semantic_threshold_requested": sem_thresh,
            "affective_threshold_requested": aff_thresh,
            "semantic_threshold_resolved": selected_sem,
            "affective_threshold_resolved": selected_aff,
            "explicit_thresholds": explicit_thresholds,
            "num_queries_requested": num_queries,
            "min_test_cases_required": min_test_cases,
            "content_items_count": len(content_items),
            "embeddings_source": embeddings_source,
            "threshold_selection": threshold_selection,
            "retrievers": {
                "BM25": {"type": "bm25"},
                "Semantic (vector-only)": {"type": "semantic"},
                "KRAG (α=1.0)": {"type": "krag", "alpha": 1.0, "candidate_pool": "semantic+emotion"},
                "KRAG (α=0.5)": {"type": "krag", "alpha": 0.5, "candidate_pool": "semantic+emotion"},
                "KRAG (α=0.3)": {"type": "krag", "alpha": 0.3, "candidate_pool": "semantic+emotion"},
            },
        },
        "evaluations": {
            "agreement": agreement_block,
            "dissonance": dissonance_block,
        },
        "warnings": warnings if warnings else [],
        "timestamp": datetime.now().isoformat()
    }
    
    # Save enhanced results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filename = f"comparative_analysis{output_suffix}.json"
    with open(output_path / filename, 'w') as f:
        json.dump(enhanced_results, f, indent=2)
    
    print(f"\nResults saved to: {output_path / filename}")

    return enhanced_results


def run_structural_sensitivity_full_system(
    system,
    num_queries: int = 50,
    output_dir: str = "./results",
    semantic_threshold: float = 0.5,
    affective_threshold: float = 0.6,
    min_test_cases: int = DEFAULT_MIN_TEST_CASES_SENSITIVITY,
    allow_insufficient: bool = False,
    hop_depths: Optional[List[int]] = None,
    seed: int = 42
):
    """
    Full-system hop-depth sensitivity.

    Goal: identify which hop depth (k) yields the best retrieval quality (NDCG@10)
    when using the actual system retriever pipeline (candidate generation + reranking).
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT: Structural Sensitivity (full system, k-hop)")
    print("=" * 60)

    from krag.evaluation.synthetic_testset import SyntheticTestSetGenerator
    from krag.core.emotion_detection import EmotionProfile
    from krag.evaluation.metrics import compute_ndcg
    from krag.retrieval.krag_retriever import QueryContext

    content_items = system.content_items
    movie_signatures = get_movie_affective_signatures(system)

    if system.subgraph_retriever is None or system.retriever is None or system.query_embedder is None:
        raise ValueError(
            "System not fully initialized for sensitivity experiment "
            "(missing subgraph_retriever/retriever/query_embedder)."
        )

    # Weights-loaded proof (mandatory)
    weights_path = Path(system.config.model_cache_dir) / "krag_encoder.pt"
    weights_proof: Dict[str, Any] = {
        "path": str(weights_path),
        "exists": bool(weights_path.exists()),
        "bytes": None,
        "mtime_iso": None,
        "load_status": None,
        "load_error": getattr(system, "krag_encoder_weights_error", None),
        "source": getattr(system, "krag_encoder_weights_source", None),
        "gcs_uri": getattr(system, "krag_encoder_weights_gcs_uri", None),
    }

    if weights_path.exists():
        st = weights_path.stat()
        weights_proof["bytes"] = int(st.st_size)
        weights_proof["mtime_iso"] = datetime.fromtimestamp(st.st_mtime).isoformat()

    loaded_flag = bool(getattr(system, "krag_encoder_weights_loaded", False))
    if loaded_flag:
        weights_proof["load_status"] = "loaded"
    else:
        weights_proof["load_status"] = "missing" if not weights_path.exists() else "load_failed"

    if not loaded_flag:
        error_result = {
            "experiment_config": {
                "status": "aborted_missing_or_unloaded_weights",
                "semantic_threshold": semantic_threshold,
                "affective_threshold": affective_threshold,
                "num_queries_requested": num_queries,
                "min_test_cases_required": min_test_cases,
                "hop_depths": hop_depths or [1, 2, 3],
                "weights_proof": weights_proof,
            },
            "error": "Structural sensitivity aborted: trained GNN weights not loaded.",
            "per_k": {},
            "diagnostics": {},
            "timestamp": datetime.now().isoformat(),
        }

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        with open(output_path / "structural_sensitivity.json", "w") as f:
            json.dump(error_result, f, indent=2)
        print("\n EXPERIMENT ABORTED: trained GNN weights not loaded.")
        return error_result

    # Load embeddings for synthetic test set generation (strict): GCS only
    from krag.data.adapters import DatasetPath
    if not system.data_processor.adapter.exists(DatasetPath.EMBEDDINGS):
        raise FileNotFoundError(
            f"Missing required embeddings artifact in GCS for sensitivity: "
            f"gs://{system.config.gcs_bucket}/{system.config.gcs_base_path}/{DatasetPath.EMBEDDINGS.value}"
        )
    print("Loading embeddings from GCS...")
    cached = system.data_processor.adapter.load_numpy(DatasetPath.EMBEDDINGS)
    content_embeddings = cached["semantic"][:len(content_items)]
    print(f"Loaded embeddings from GCS: {content_embeddings.shape}")

    generator = SyntheticTestSetGenerator(
        content_embedder=system.content_embedder,
        semantic_threshold=semantic_threshold,
        affective_threshold=affective_threshold,
        seed=seed
    )

    test_cases = generator.generate_test_set(
        content_items=content_items,
        content_embeddings=content_embeddings,
        movie_affective_signatures=movie_signatures,
        num_queries=num_queries,
        min_relevant=3
    )

    print(f"\nGenerated {len(test_cases)} test cases for sensitivity analysis")

    if len(test_cases) < min_test_cases and not allow_insufficient:
        error_result = {
            "experiment_config": {
                "status": "aborted_insufficient_cases",
                "semantic_threshold": semantic_threshold,
                "affective_threshold": affective_threshold,
                "num_queries_requested": num_queries,
                "min_test_cases_required": min_test_cases,
                "num_test_cases_generated": len(test_cases),
                "avg_relevant_items_per_query": round(
                    float(np.mean([len(tc.relevant_items) for tc in test_cases])), 3
                ) if test_cases else 0.0,
                "hop_depths": hop_depths or [1, 2, 3],
                "weights_proof": weights_proof,
            },
            "error": "Structural sensitivity aborted: insufficient synthetic test cases.",
            "per_k": {},
            "diagnostics": {},
            "timestamp": datetime.now().isoformat(),
        }

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        with open(output_path / "structural_sensitivity.json", "w") as f:
            json.dump(error_result, f, indent=2)
        print("\nEXPERIMENT ABORTED: insufficient test cases.")
        return error_result

    hop_depths = hop_depths or [1, 2, 3]
    per_k: Dict[str, Any] = {}
    diagnostics: Dict[str, Any] = {}

    content_ids = [str(item.id) for item in content_items]

    for k_hops in hop_depths:
        print(f"\nTesting k={k_hops} hops...")
        print(f"  Re-encoding subgraphs with {k_hops}-hop neighborhoods...")

        system.subgraph_retriever.subgraph_embeddings.clear()
        system.subgraph_retriever.index_subgraphs(content_ids, hops=k_hops)

        subgraph_sizes = []
        for content_id in content_ids:
            subgraph = system.knowledge_graph.extract_subgraph(content_id, hops=k_hops)
            subgraph_sizes.append(len(subgraph.nodes()))
        avg_subgraph_size = float(np.mean(subgraph_sizes))
        min_subgraph_size = int(np.min(subgraph_sizes))
        max_subgraph_size = int(np.max(subgraph_sizes))
        print(f"  Subgraph sizes: avg={avg_subgraph_size:.1f}, min={min_subgraph_size}, max={max_subgraph_size}")

        ndcg_scores = []
        all_knowledge_scores: List[float] = []

        for test_case in test_cases:
            emotion_profile = EmotionProfile(
                happiness=test_case.target_emotions.get("happiness", 0.0),
                sadness=test_case.target_emotions.get("sadness", 0.0),
                anger=test_case.target_emotions.get("anger", 0.0),
                fear=test_case.target_emotions.get("fear", 0.0),
                surprise=test_case.target_emotions.get("surprise", 0.0),
                disgust=test_case.target_emotions.get("disgust", 0.0),
            )

            query_embedding, emotion_embedding = system.query_embedder.embed_query(
                test_case.query_text, emotion_profile
            )

            query_context = QueryContext(
                query_text=test_case.query_text,
                user_emotions=emotion_profile,
                query_embedding=query_embedding,
                emotion_embedding=emotion_embedding,
            )

            retrieved = system.retriever.retrieve(query_context, k=10)
            retrieved_ids = [r.content_id for r in retrieved]

            if test_case.relevant_items:
                ndcg = compute_ndcg(retrieved_ids, test_case.relevant_items, 10)
                ndcg_scores.append(ndcg)

            all_knowledge_scores.extend([float(r.knowledge_score) for r in retrieved])

        ndcg_mean = float(np.mean(ndcg_scores)) if ndcg_scores else 0.0
        ndcg_std = float(np.std(ndcg_scores, ddof=1)) if len(ndcg_scores) >= 2 else 0.0
        ci_low, ci_high = compute_confidence_interval(ndcg_scores) if len(ndcg_scores) >= 2 else (0.0, 0.0)

        ks = np.array(all_knowledge_scores, dtype=float)
        ks_finite = ks[np.isfinite(ks)]
        pct_nonzero = float(np.mean(np.abs(ks_finite) > 1e-8)) if len(ks_finite) else 0.0
        ks_mean = float(np.mean(ks_finite)) if len(ks_finite) else 0.0
        ks_std = float(np.std(ks_finite, ddof=1)) if len(ks_finite) >= 2 else 0.0

        per_k[str(k_hops)] = {
            "NDCG@10_mean": ndcg_mean,
            "NDCG@10_std": ndcg_std,
            "NDCG@10_95_CI": [round(ci_low, 6), round(ci_high, 6)],
            "num_queries_used": int(len(ndcg_scores)),
        }

        diagnostics[str(k_hops)] = {
            "subgraph_embeddings_count": int(len(system.subgraph_retriever.subgraph_embeddings)),
            "knowledge_score_mean": ks_mean,
            "knowledge_score_std": ks_std,
            "knowledge_score_pct_nonzero": pct_nonzero,
            "avg_subgraph_size": avg_subgraph_size,
            "min_subgraph_size": min_subgraph_size,
            "max_subgraph_size": max_subgraph_size,
        }

        print(f"  NDCG@10: {ndcg_mean:.4f} (n={len(ndcg_scores)})")
        print(f"  knowledge_score: mean={ks_mean:.6f}, pct_nonzero={pct_nonzero:.4f}")

        # Graph-signal sanity: abort if the graph term is effectively inactive.
        if pct_nonzero < 0.01 or ks_std < 1e-6:
            error_result = {
                "experiment_config": {
                    "status": "aborted_graph_signal_inactive",
                    "semantic_threshold": semantic_threshold,
                    "affective_threshold": affective_threshold,
                    "num_queries_requested": num_queries,
                    "min_test_cases_required": min_test_cases,
                    "num_test_cases_generated": len(test_cases),
                    "avg_relevant_items_per_query": round(
                        float(np.mean([len(tc.relevant_items) for tc in test_cases])), 3
                    ) if test_cases else 0.0,
                    "hop_depths": hop_depths,
                    "weights_proof": weights_proof,
                    "retriever": {
                        "type": getattr(system.config, "retriever_type", None),
                        "alpha": getattr(system.retriever, "alpha", None),
                        "lambda_weight": getattr(system.retriever, "lambda_weight", None),
                        "candidate_pool": getattr(system.retriever, "candidate_pool", None),
                    },
                },
                "error": (
                    "Structural sensitivity aborted: graph signal inactive/degenerate "
                    f"at k={k_hops} (knowledge_score_pct_nonzero={pct_nonzero:.6f}, knowledge_score_std={ks_std:.6f})."
                ),
                "per_k": per_k,
                "diagnostics": diagnostics,
                "timestamp": datetime.now().isoformat(),
            }

            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            with open(output_path / "structural_sensitivity.json", "w") as f:
                json.dump(error_result, f, indent=2)
            print("\nEXPERIMENT ABORTED: graph signal inactive/degenerate.")
            return error_result

    best_k = max(per_k.keys(), key=lambda kk: per_k[kk]["NDCG@10_mean"]) if per_k else None

    result = {
        "experiment_config": {
            "status": "completed" if len(test_cases) >= min_test_cases else "warning_insufficient_cases",
            "semantic_threshold": semantic_threshold,
            "affective_threshold": affective_threshold,
            "num_queries_requested": num_queries,
            "min_test_cases_required": min_test_cases,
            "num_test_cases_generated": len(test_cases),
            "avg_relevant_items_per_query": round(
                float(np.mean([len(tc.relevant_items) for tc in test_cases])), 3
            ) if test_cases else 0.0,
            "hop_depths": hop_depths,
            "weights_proof": weights_proof,
            "retriever": {
                "type": getattr(system.config, "retriever_type", None),
                "alpha": getattr(system.retriever, "alpha", None),
                "lambda_weight": getattr(system.retriever, "lambda_weight", None),
                "candidate_pool": getattr(system.retriever, "candidate_pool", None),
            },
        },
        "per_k": per_k,
        "best_k": best_k,
        "diagnostics": diagnostics,
        "timestamp": datetime.now().isoformat(),
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "structural_sensitivity.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nBest k (by mean NDCG@10): {best_k}")
    print(f"Results saved to: {output_path / 'structural_sensitivity.json'}")
    return result


def run_causal_necessity(system, num_samples=50, output_dir="./results"):
    """
    Causal necessity analysis.

    Perturbation-based analysis measuring FNS (Faithfulness Necessity Score).
    """
    print("\n" + "=" * 60)
    print("CAUSAL NECESSITY ANALYSIS")
    print("=" * 60)

    from krag.evaluation.causal_analysis import CausalNecessityAnalyzer
    from krag.retrieval.krag_retriever import RetrieverFactory, QueryContext
    from krag.core.emotion_detection import EmotionProfile

    retriever = RetrieverFactory.create_retriever(
        "krag",
        vector_store=system.vector_store,
        knowledge_graph=system.knowledge_graph,
        krag_encoder=system.krag_encoder,
        alpha=0.5
    )

    analyzer = CausalNecessityAnalyzer(retriever, system.knowledge_graph)

    emotion_templates = [
        {'happiness': 0.9, 'sadness': 0.1, 'anger': 0.0, 'fear': 0.0, 'surprise': 0.2, 'disgust': 0.0},
        {'happiness': 0.1, 'sadness': 0.9, 'anger': 0.2, 'fear': 0.1, 'surprise': 0.0, 'disgust': 0.0},
        {'happiness': 0.0, 'sadness': 0.1, 'anger': 0.2, 'fear': 0.9, 'surprise': 0.3, 'disgust': 0.1},
        {'happiness': 0.7, 'sadness': 0.0, 'anger': 0.0, 'fear': 0.0, 'surprise': 0.8, 'disgust': 0.0},
        {'happiness': 0.0, 'sadness': 0.3, 'anger': 0.9, 'fear': 0.2, 'surprise': 0.0, 'disgust': 0.4},
    ]

    query_templates = [
        "A heartwarming comedy about family",
        "An emotional drama about loss",
        "A terrifying horror movie",
        "An exciting adventure with surprises",
        "An intense action thriller",
    ]

    all_fns_scores = []
    samples_per_template = max(1, num_samples // len(query_templates))

    print(f"\nAnalyzing {num_samples} samples across {len(query_templates)} query types...")

    for template_idx, (query, emotions) in enumerate(zip(query_templates, emotion_templates)):
        emotion_profile = EmotionProfile(
            happiness=emotions['happiness'],
            sadness=emotions['sadness'],
            anger=emotions['anger'],
            fear=emotions['fear'],
            surprise=emotions['surprise'],
            disgust=emotions['disgust']
        )

        query_embedding = system.content_embedder.embed_content(query)

        query_context = QueryContext(
            query_text=query,
            user_emotions=emotion_profile,
            query_embedding=query_embedding,
            emotion_embedding=query_embedding
        )

        dominant_emotion = max(emotions, key=emotions.get)
        results = retriever.retrieve(query_context, k=samples_per_template * 2)

        for result in results[:samples_per_template]:
            analysis = analyzer.analyze_single(
                query_context=query_context,
                target_content_id=result.content_id,
                emotion_to_perturb=dominant_emotion
            )

            if analysis.fns_score > 0:
                all_fns_scores.append(analysis.fns_score)
                if len(all_fns_scores) <= 10:
                    print(f"  {result.title[:30]:<30} FNS={analysis.fns_score:.3f} (perturbed: {dominant_emotion})")

    if all_fns_scores:
        mean_fns = np.mean(all_fns_scores)
        median_fns = np.median(all_fns_scores)
        high_fns_ratio = sum(1 for s in all_fns_scores if s > 0.5) / len(all_fns_scores)
    else:
        mean_fns = 0.0
        median_fns = 0.0
        high_fns_ratio = 0.0

    print("\n" + "-" * 60)
    print("RESULTS")
    print("-" * 60)
    print(f"Mean FNS: {mean_fns:.4f}")
    print(f"Median FNS: {median_fns:.4f}")
    print(f"High FNS Ratio (>0.5): {high_fns_ratio:.4f}")
    print(f"Samples Analyzed: {len(all_fns_scores)}")
    print("-" * 60)
    print("\nInterpretation:")
    print(f"  FNS > 0.5 indicates recommendations are causally driven by")
    print(f"  explicit graph evidence rather than latent correlations.")

    results = {
        'mean_fns': float(mean_fns),
        'median_fns': float(median_fns),
        'high_fns_ratio': float(high_fns_ratio),
        'num_samples': len(all_fns_scores)
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "causal_analysis.json", 'w') as f:
        json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run KRAG experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run threshold sensitivity analysis first to find optimal thresholds:
  python run_experiments.py --experiment threshold-sensitivity
  
  # Run comparative analysis with explicit thresholds:
  python run_experiments.py --experiment comparative \\
      --semantic-threshold 0.5 --affective-threshold 0.6 \\
      --num-queries 500 --min-test-cases 200
  
  # Run with dual sample sizes for statistical comparison:
  python run_experiments.py --experiment comparative --num-queries 50 --output-suffix _t50
  python run_experiments.py --experiment comparative --num-queries 150 --output-suffix _t100
        """
    )

    parser.add_argument(
        "--experiment",
        type=str,
        default="all",
        choices=["all", "comparative", "sensitivity", "causal", "threshold-sensitivity"],
        help="Which experiment to run"
    )
    parser.add_argument("--max-items", type=int, default=None, help="Max content items (None = full dataset)")
    parser.add_argument(
        "--num-queries",
        type=int,
        default=None,
        help=("Number of test queries/samples. If omitted, defaults are experiment-specific "
              f"(comparative={DEFAULT_NUM_QUERIES_COMPARATIVE}, "
              f"threshold-sensitivity={DEFAULT_NUM_QUERIES_THRESHOLD_SENSITIVITY}, "
              f"other={DEFAULT_NUM_QUERIES_OTHER}).")
    )
    parser.add_argument("--output-dir", type=str, default="./results", help="Output directory")
    
    # New threshold control arguments
    parser.add_argument(
        "--semantic-threshold",
        type=float,
        default=None,
        help="Explicit semantic threshold override. Run threshold-sensitivity to find an appropriate value."
    )
    parser.add_argument(
        "--affective-threshold",
        type=float,
        default=None,
        help="Explicit affective threshold override. Run threshold-sensitivity to find an appropriate value."
    )
    parser.add_argument(
        "--min-test-cases",
        type=int,
        default=None,
        help=("Minimum test cases required for valid results. If omitted, defaults are "
              f"experiment-specific (comparative={DEFAULT_MIN_TEST_CASES_COMPARATIVE}, "
              f"other={DEFAULT_MIN_TEST_CASES}).")
    )
    parser.add_argument(
        "--allow-insufficient",
        action="store_true",
        help="Run experiment even if fewer than min-test-cases are generated"
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="",
        help="Suffix for output filename (e.g., '_t50' for 50-query run)"
    )

    args = parser.parse_args()

    from krag.system import ARAGSystem, ARAGSystemConfig

    config = ARAGSystemConfig(
        max_content_items=args.max_items,
        retriever_type="krag"
    )

    print("\n" + "=" * 60)
    print("AFFECTIVE-RAG PAPER EXPERIMENTS")
    print("=" * 60)

    system = ARAGSystem(config)
    system.initialize()

    print("\nLoading data...")
    system.load_and_index_data()

    results = {}

    # Choose experiment-specific defaults if user didn't specify.
    if args.num_queries is None:
        if args.experiment in ["comparative"]:
            args.num_queries = DEFAULT_NUM_QUERIES_COMPARATIVE
        elif args.experiment in ["threshold-sensitivity"]:
            args.num_queries = DEFAULT_NUM_QUERIES_THRESHOLD_SENSITIVITY
        else:
            args.num_queries = DEFAULT_NUM_QUERIES_OTHER

    if args.min_test_cases is None:
        if args.experiment in ["comparative"]:
            args.min_test_cases = DEFAULT_MIN_TEST_CASES_COMPARATIVE
        elif args.experiment in ["sensitivity", "all"]:
            args.min_test_cases = DEFAULT_MIN_TEST_CASES_SENSITIVITY
        else:
            args.min_test_cases = DEFAULT_MIN_TEST_CASES

    # Threshold sensitivity experiment
    if args.experiment == "threshold-sensitivity":
        results['threshold_sensitivity'] = run_threshold_sensitivity(
            system,
            num_queries=args.num_queries,
            output_dir=args.output_dir
        )
    
    # Comparative analysis
    if args.experiment in ["all", "comparative"]:
        results['comparative'] = run_comparative_analysis(
            system,
            num_queries=args.num_queries,
            output_dir=args.output_dir,
            semantic_threshold=args.semantic_threshold,
            affective_threshold=args.affective_threshold,
            min_test_cases=args.min_test_cases,
            allow_insufficient=args.allow_insufficient,
            output_suffix=args.output_suffix
        )

    if args.experiment in ["all", "sensitivity"]:
        sem = args.semantic_threshold if args.semantic_threshold is not None else 0.5
        aff = args.affective_threshold if args.affective_threshold is not None else 0.6
        results['sensitivity'] = run_structural_sensitivity_full_system(
            system,
            num_queries=args.num_queries,
            output_dir=args.output_dir,
            semantic_threshold=sem,
            affective_threshold=aff,
            min_test_cases=args.min_test_cases,
            allow_insufficient=args.allow_insufficient
        )

    if args.experiment in ["all", "causal"]:
        results['causal'] = run_causal_necessity(
            system,
            num_samples=args.num_queries,
            output_dir=args.output_dir
        )

    print("\n" + "=" * 60)
    print("EXPERIMENTS COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {args.output_dir}/")

    return results


if __name__ == "__main__":
    main()
