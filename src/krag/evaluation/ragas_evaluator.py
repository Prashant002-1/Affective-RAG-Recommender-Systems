"""
RAGAS Faithfulness Evaluator for Affective-RAG

Evaluates whether KRAG explanations are faithful to knowledge graph evidence
using LLM-as-judge NLI verification (RAGAS framework, Es et al. arXiv:2309.15217).
"""

import re
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass, field


@dataclass
class Claim:
    """A single atomic claim extracted from an explanation."""
    text: str
    claim_type: str  # "numeric" or "contextual"


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
    claim_verdicts: List[Dict[str, Any]] = field(default_factory=list)


class RAGASFaithfulnessEvaluator:
    """
    RAGAS-based faithfulness evaluation using LLM-as-judge.

    Verifies that KRAG explanations strictly entail knowledge graph evidence.
    Uses chain-of-thought NLI prompting per LLM-as-Judge best practices
    (arXiv:2411.15594, arXiv:2412.05579).

    Faithfulness Score = verified_claims / total_claims, in [0, 1].
    """

    NLI_PROMPT = """You are a verification judge. Given evidence from a knowledge graph and a claim from a recommendation explanation, determine if the claim is SUPPORTED by the evidence.

EVIDENCE (from knowledge graph):
{context}

CLAIM:
{claim}

EVALUATION CRITERIA:
- SUPPORTED: The claim can be directly inferred from the evidence. All entities, relationships, and values mentioned in the claim appear in or follow logically from the evidence.
- NOT_SUPPORTED: The claim contains information not present in the evidence, or contradicts it, or makes assertions beyond what the evidence states.

Think step by step:
1. Identify the key entities, relationships, or values in the claim.
2. Check if each appears in the evidence.
3. Check if the stated relationships or values match the evidence.
4. Make your final judgment.

VERDICT (respond with exactly one word on the last line): SUPPORTED or NOT_SUPPORTED"""

    def __init__(self, llm_client: Any, model_name: str = "gemini-3-flash-preview"):
        """
        Args:
            llm_client: LLM client for NLI verification. Must implement .generate(prompt) -> str.
            model_name: Model identifier for metadata/logging.
        """
        if llm_client is None:
            raise ValueError(
                "RAGASFaithfulnessEvaluator requires an LLM client for NLI verification. "
                "Pass a client that implements .generate(prompt) -> str."
            )
        self.llm_client = llm_client
        self.model_name = model_name

    def evaluate_faithfulness(
        self,
        generated_explanation: str,
        graph_context: str
    ) -> float:
        """
        Evaluate faithfulness of a single explanation against graph context.

        Args:
            generated_explanation: KRAG explanation text
            graph_context: Serialized graph evidence

        Returns:
            Faithfulness score in [0, 1]
        """
        claims = self.extract_claims(generated_explanation)

        if not claims:
            return 1.0

        verified_count = sum(
            1 for claim in claims
            if self._verify_claim(claim.text, graph_context)
        )

        return verified_count / len(claims)

    def evaluate_batch(
        self,
        explanations: List[Dict[str, str]]
    ) -> List[FaithfulnessResult]:
        """
        Evaluate faithfulness for a batch of explanations.

        Args:
            explanations: List of dicts with keys 'query', 'movie', 'explanation', 'context'

        Returns:
            List of FaithfulnessResult objects
        """
        results = []

        for item in explanations:
            claims = self.extract_claims(item['explanation'])
            verified_claims = []
            unverified_claims = []
            claim_verdicts = []

            for claim in claims:
                is_verified = self._verify_claim(claim.text, item['context'])
                verdict = {
                    'claim': claim.text,
                    'type': claim.claim_type,
                    'verdict': 'SUPPORTED' if is_verified else 'NOT_SUPPORTED'
                }
                claim_verdicts.append(verdict)

                if is_verified:
                    verified_claims.append(claim.text)
                else:
                    unverified_claims.append(claim.text)

            score = len(verified_claims) / len(claims) if claims else 1.0

            results.append(FaithfulnessResult(
                query_text=item.get('query', ''),
                movie_title=item.get('movie', ''),
                generated_explanation=item['explanation'],
                graph_context=item['context'],
                faithfulness_score=score,
                claims_verified=len(verified_claims),
                claims_total=len(claims),
                unverified_claims=unverified_claims,
                claim_verdicts=claim_verdicts
            ))

        return results

    def extract_claims(self, explanation: str) -> List[Claim]:
        """
        Extract atomic claims from KRAG structured explanation.

        Handles two explanation formats:
        1. Compact: "High semantic (0.850); Strong graph (0.620); ... [Score = ...]"
        2. Detailed: "Retrieved 'Title' because:\\n• Semantic similarity: ...\\n• Graph context: ..."

        Returns:
            List of Claim objects with type annotations
        """
        claims = []

        semantic_match = re.search(
            r'(?:Semantic similarity[^:]*:\s*|(?:High|Moderate) semantic \()(\d+\.\d+)', explanation
        )
        if semantic_match:
            claims.append(Claim(
                text=f"Semantic similarity score is {semantic_match.group(1)}",
                claim_type="numeric"
            ))

        graph_match = re.search(
            r'(?:Graph similarity[^:]*:\s*|(?:Strong|Moderate) graph \()(\d+\.\d+)', explanation
        )
        if graph_match:
            claims.append(Claim(
                text=f"Graph similarity score is {graph_match.group(1)}",
                claim_type="numeric"
            ))

        rmse_match = re.search(r'RMSE[=:\s]*(\d+\.\d+)', explanation)
        if rmse_match:
            claims.append(Claim(
                text=f"Affective RMSE is {rmse_match.group(1)}",
                claim_type="numeric"
            ))

        formula_match = re.search(
            r'Score\s*=\s*[\d.]+\*\[[\d.]+\*[\d.]+ \+ [\d.]+\*[\d.]+\]\s*-\s*[\d.]+\*[\d.]+\s*=\s*(\d+\.\d+)',
            explanation
        )
        if formula_match:
            claims.append(Claim(
                text=f"Combined score evaluates to {formula_match.group(1)}",
                claim_type="numeric"
            ))

        context_match = re.search(r'Graph context:\s*(.+?)(?:\n|$)', explanation)
        if context_match:
            context_text = context_match.group(1).strip()
            if context_text:
                claims.append(Claim(
                    text=f"Graph context states: {context_text}",
                    claim_type="contextual"
                ))

        return claims

    def _verify_claim(self, claim: str, context: str) -> bool:
        """
        Verify a claim against graph context using LLM NLI.

        Args:
            claim: Atomic claim text
            context: Serialized knowledge graph evidence

        Returns:
            True if the LLM judge determines the claim is SUPPORTED

        Raises:
            RuntimeError: If the LLM call fails
        """
        prompt = self.NLI_PROMPT.format(context=context, claim=claim)

        response = self.llm_client.generate(prompt)

        for line in reversed(response.strip().split('\n')):
            word = line.strip().upper()
            if word == 'NOT_SUPPORTED':
                return False
            if word == 'SUPPORTED':
                return True
            if 'NOT_SUPPORTED' in word:
                return False
            if 'SUPPORTED' in word:
                return True

        raise RuntimeError(
            f"Could not parse verdict from LLM response. "
            f"Claim: {claim!r}\nResponse: {response!r}"
        )


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
