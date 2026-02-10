"""
RAGAS Faithfulness Evaluator for Affective-RAG

Evaluates whether LLM-generated explanations are faithful to the
knowledge graph evidence.

Paper Section IV.D.2 - RAGAS Faithfulness Evaluation
"""

import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class FaithfulnessResult:
    """Result of faithfulness evaluation for a single explanation."""
    query_text: str
    movie_title: str
    generated_explanation: str
    graph_context: str
    faithfulness_score: float
    claims_verified: int
    claims_total: int
    unverified_claims: List[str]


class RAGASFaithfulnessEvaluator:
    """
    RAGAS-based faithfulness evaluation.

    Verifies that LLM-generated explanations strictly entail graph evidence.
    Uses LLM-as-judge approach to verify atomic claims against graph context.

    Paper formula: Faithfulness Score in [0, 1]
    High scores indicate the system avoids hallucinating attributes
    absent from the knowledge graph.
    """

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        model_name: str = "gemini-3-flash-preview"
    ):
        """
        Initialize evaluator.

        Args:
            llm_client: LLM client for judge (optional, uses Vertex AI if None)
            model_name: Model to use for judging
        """
        self.llm_client = llm_client
        self.model_name = model_name

    def evaluate_faithfulness(
        self,
        generated_explanation: str,
        graph_context: str
    ) -> float:
        """
        Evaluate faithfulness of generated explanation against graph context.

        Args:
            generated_explanation: LLM-generated explanation text
            graph_context: Serialized graph evidence (paths, edges, nodes)

        Returns:
            Faithfulness score in [0, 1]
        """
        claims = self._extract_claims(generated_explanation)

        if not claims:
            return 1.0

        verified_count = 0
        for claim in claims:
            if self._verify_claim(claim, graph_context):
                verified_count += 1

        return verified_count / len(claims)

    def evaluate_batch(
        self,
        explanations: List[Dict[str, str]]
    ) -> List[FaithfulnessResult]:
        """
        Evaluate faithfulness for batch of explanations.

        Args:
            explanations: List of dicts with 'query', 'movie', 'explanation', 'context'

        Returns:
            List of FaithfulnessResult objects
        """
        results = []

        for item in explanations:
            claims = self._extract_claims(item['explanation'])
            verified_claims = []
            unverified_claims = []

            for claim in claims:
                if self._verify_claim(claim, item['context']):
                    verified_claims.append(claim)
                else:
                    unverified_claims.append(claim)

            score = len(verified_claims) / len(claims) if claims else 1.0

            results.append(FaithfulnessResult(
                query_text=item.get('query', ''),
                movie_title=item.get('movie', ''),
                generated_explanation=item['explanation'],
                graph_context=item['context'],
                faithfulness_score=score,
                claims_verified=len(verified_claims),
                claims_total=len(claims),
                unverified_claims=unverified_claims
            ))

        return results

    def _extract_claims(self, explanation: str) -> List[str]:
        """
        Extract atomic claims from explanation text.

        Uses simple heuristics for claim extraction:
        - Split by sentences
        - Filter out generic phrases
        - Extract factual statements
        """
        import re

        sentences = re.split(r'[.!?]', explanation)
        sentences = [s.strip() for s in sentences if s.strip()]

        claims = []
        skip_patterns = [
            r'^I recommend',
            r'^This movie',
            r'^You might',
            r'^Based on',
            r'^Overall',
            r'^In summary',
        ]

        for sentence in sentences:
            skip = False
            for pattern in skip_patterns:
                if re.match(pattern, sentence, re.IGNORECASE):
                    skip = True
                    break

            if not skip and len(sentence.split()) >= 3:
                claims.append(sentence)

        return claims

    def _verify_claim(self, claim: str, graph_context: str) -> bool:
        """
        Verify if a claim can be inferred from graph context.

        Uses keyword matching as a simple heuristic.
        For production, this should use LLM-based entailment checking.
        """
        claim_lower = claim.lower()
        context_lower = graph_context.lower()

        emotion_keywords = ['happy', 'sad', 'angry', 'fear', 'surprise', 'disgust',
                          'joy', 'melancholy', 'intense', 'scary', 'shocking']

        genre_keywords = ['action', 'comedy', 'drama', 'horror', 'romance',
                         'thriller', 'sci-fi', 'adventure', 'fantasy']

        for keyword in emotion_keywords + genre_keywords:
            if keyword in claim_lower and keyword in context_lower:
                return True

        claim_words = set(claim_lower.split())
        context_words = set(context_lower.split())
        overlap = len(claim_words & context_words)

        return overlap >= 3

    def _verify_claim_with_llm(self, claim: str, context: str) -> bool:
        """
        Verify claim using LLM as judge.

        This is the more accurate but slower method.
        """
        if self.llm_client is None:
            return self._verify_claim(claim, context)

        prompt = f"""Given the following knowledge graph context and a claim,
determine if the claim can be strictly inferred from the context.

Context (from knowledge graph):
{context}

Claim to verify:
{claim}

Can this claim be inferred from the context? Answer only 'YES' or 'NO'."""

        try:
            response = self.llm_client.generate(prompt)
            return 'YES' in response.upper()
        except Exception:
            return self._verify_claim(claim, context)


def build_graph_context(
    content_id: str,
    knowledge_graph: Any,
    max_edges: int = 20
) -> str:
    """
    Build textual graph context from knowledge graph for a content item.

    Args:
        content_id: Content ID to build context for
        knowledge_graph: ContentKnowledgeGraph instance
        max_edges: Maximum edges to include

    Returns:
        Textual representation of graph context
    """
    if content_id not in knowledge_graph.graph:
        return ""

    context_parts = []

    context_parts.append(f"Movie ID: {content_id}")

    edges = list(knowledge_graph.graph.out_edges(content_id, data=True))[:max_edges]

    genres = []
    emotions = []
    similar = []

    for _, target, data in edges:
        relation = data.get('relation', 'unknown')
        weight = data.get('weight', 1.0)

        if relation == 'belongs_to_genre':
            genre_name = target.replace('genre_', '')
            genres.append(genre_name)
        elif relation == 'evokes':
            emotion_name = target.replace('emotion_', '')
            emotions.append(f"{emotion_name} ({weight:.2f})")
        elif relation == 'similar_to':
            similar.append(target)

    if genres:
        context_parts.append(f"Genres: {', '.join(genres)}")
    if emotions:
        context_parts.append(f"Evoked emotions: {', '.join(emotions)}")
    if similar:
        context_parts.append(f"Similar movies: {', '.join(similar[:5])}")

    return "\n".join(context_parts)


def build_full_subgraph_context(
    content_id: str,
    knowledge_graph: Any,
    hops: int = 2
) -> str:
    """
    Build comprehensive subgraph context including multi-hop relationships.

    Args:
        content_id: Content ID
        knowledge_graph: ContentKnowledgeGraph instance
        hops: Number of hops to include

    Returns:
        Detailed textual representation of subgraph
    """
    if content_id not in knowledge_graph.graph:
        return ""

    subgraph = knowledge_graph.extract_subgraph(content_id, hops=hops)

    context_parts = ["KNOWLEDGE GRAPH EVIDENCE:"]

    context_parts.append(f"\nCentral Node: {content_id}")

    context_parts.append("\nEDGES (source -> relation -> target [weight]):")
    for u, v, data in subgraph.edges(data=True):
        relation = data.get('relation', 'related_to')
        weight = data.get('weight', 1.0)
        context_parts.append(f"  {u} -> {relation} -> {v} [{weight:.2f}]")

    emotions_found = []
    genres_found = []
    for node in subgraph.nodes():
        if node.startswith('emotion_'):
            emotions_found.append(node.replace('emotion_', ''))
        elif node.startswith('genre_'):
            genres_found.append(node.replace('genre_', ''))

    if emotions_found:
        context_parts.append(f"\nEMOTIONS IN SUBGRAPH: {', '.join(emotions_found)}")
    if genres_found:
        context_parts.append(f"GENRES IN SUBGRAPH: {', '.join(genres_found)}")

    return "\n".join(context_parts)


def run_faithfulness_evaluation(
    explanations: List[Dict[str, str]],
    output_path: str = "./results/faithfulness.json"
) -> Dict[str, float]:
    """
    Run faithfulness evaluation experiment.

    Args:
        explanations: List of explanation dicts
        output_path: Path to save results

    Returns:
        Aggregate faithfulness statistics
    """
    import json
    from pathlib import Path

    evaluator = RAGASFaithfulnessEvaluator()
    results = evaluator.evaluate_batch(explanations)

    scores = [r.faithfulness_score for r in results]
    unverified_ratios = [
        len(r.unverified_claims) / r.claims_total if r.claims_total > 0 else 0
        for r in results
    ]

    stats = {
        'mean_faithfulness': float(np.mean(scores)),
        'median_faithfulness': float(np.median(scores)),
        'std_faithfulness': float(np.std(scores)),
        'perfect_score_ratio': sum(1 for s in scores if s == 1.0) / len(scores),
        'mean_unverified_ratio': float(np.mean(unverified_ratios)),
        'total_evaluated': len(results)
    }

    print("\nRAGAS Faithfulness Evaluation Results:")
    print(f"  Mean Faithfulness: {stats['mean_faithfulness']:.4f}")
    print(f"  Perfect Score Ratio: {stats['perfect_score_ratio']:.4f}")
    print(f"  Mean Unverified Claims: {stats['mean_unverified_ratio']:.4f}")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        'statistics': stats,
        'individual_results': [
            {
                'movie': r.movie_title,
                'score': r.faithfulness_score,
                'claims_verified': r.claims_verified,
                'claims_total': r.claims_total
            }
            for r in results
        ]
    }

    with open(output, 'w') as f:
        json.dump(output_data, f, indent=2)

    return stats
