"""
BM25 Lexical Baseline Retriever

Implements BM25 (Best Matching 25) algorithm for lexical retrieval.
Used as a baseline for comparative analysis against semantic and
affective retrieval methods.
"""

import numpy as np
from typing import List, Dict, Optional, Any
from rank_bm25 import BM25Okapi

from .krag_retriever import BaseRetriever, RetrievalResult, QueryContext


class BM25Retriever(BaseRetriever):
    """
    BM25 lexical baseline retriever.

    Represents traditional content-based filtering using TF-IDF weighting
    to match query keywords with movie overviews. This baseline captures
    semantic nuance at the lexical level without neural embeddings.

    Paper Section IV.B - Baseline 1: BM25 (Lexical Baseline)
    """

    def __init__(
        self,
        corpus: List[str],
        content_ids: List[str],
        metadata: Dict[str, Dict[str, Any]]
    ):
        """
        Initialize BM25 retriever.

        Args:
            corpus: List of document texts (movie overviews/descriptions)
            content_ids: List of content IDs corresponding to corpus
            metadata: Dict mapping content_id to metadata dict
        """
        if len(corpus) != len(content_ids):
            raise ValueError("corpus and content_ids must have same length")

        self.content_ids = content_ids
        self.metadata = metadata
        self.id_to_idx = {cid: i for i, cid in enumerate(content_ids)}

        tokenized_corpus = [self._tokenize(doc) for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenization with lowercasing."""
        return text.lower().split()

    def retrieve(self, query_context: QueryContext, k: int = 10) -> List[RetrievalResult]:
        """
        Retrieve using BM25 lexical matching.

        Args:
            query_context: Query context (only query_text is used)
            k: Number of results to return

        Returns:
            Ranked list of retrieval results
        """
        tokenized_query = self._tokenize(query_context.query_text)
        scores = self.bm25.get_scores(tokenized_query)

        max_score = max(scores) if max(scores) > 0 else 1.0

        top_indices = np.argsort(scores)[::-1]

        results = []
        for idx in top_indices:
            content_id = self.content_ids[idx]

            if query_context.allowed_content_ids is not None and content_id not in query_context.allowed_content_ids:
                continue

            normalized_score = float(scores[idx] / max_score) if max_score > 0 else 0.0

            meta = self.metadata.get(content_id, {})

            result = RetrievalResult(
                content_id=content_id,
                title=meta.get('title', ''),
                semantic_score=normalized_score,
                emotion_score=0.0,
                knowledge_score=0.0,
                combined_score=normalized_score,
                metadata=meta,
                explanation=f"BM25 lexical match score: {normalized_score:.3f}"
            )
            results.append(result)

            if len(results) >= k:
                break

        return results

    def explain_retrieval(self, result: RetrievalResult, query_context: QueryContext) -> str:
        """Explain why this item was retrieved."""
        return (
            f"BM25 Lexical Retrieval:\n"
            f"  Query: '{query_context.query_text}'\n"
            f"  Matched via keyword overlap with score: {result.semantic_score:.3f}"
        )


def create_bm25_retriever_from_content_items(
    content_items: List[Any],
    text_field: str = 'description'
) -> BM25Retriever:
    """
    Factory function to create BM25Retriever from ContentItem list.

    Args:
        content_items: List of ContentItem objects
        text_field: Field to use for document text ('description' or 'title')

    Returns:
        Configured BM25Retriever
    """
    corpus = []
    content_ids = []
    metadata = {}

    for item in content_items:
        content_id = str(item.id)
        content_ids.append(content_id)

        if text_field == 'description':
            text = item.description if hasattr(item, 'description') else ''
        else:
            text = item.title if hasattr(item, 'title') else ''

        if hasattr(item, 'title'):
            text = f"{item.title}. {text}"
        if hasattr(item, 'genres') and item.genres:
            text = f"{text} Genres: {', '.join(item.genres)}"

        corpus.append(text)

        metadata[content_id] = {
            'title': getattr(item, 'title', ''),
            'description': getattr(item, 'description', ''),
            'genres': getattr(item, 'genres', []),
            'year': getattr(item, 'year', None)
        }

    return BM25Retriever(corpus, content_ids, metadata)
