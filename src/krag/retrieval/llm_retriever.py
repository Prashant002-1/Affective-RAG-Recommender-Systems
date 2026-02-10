"""
Zero-Shot LLM Baseline Retriever for Affective-RAG

From Section IV.A.2 of the paper:
"Queries a Large Language Model directly with the user's history to recommend
the next movie. This tests whether the external knowledge retrieval (RAG) is
necessary compared to the model's parametric memory."

This baseline uses NO retrieval - it relies purely on the LLM's parametric knowledge.
"""

import json
import re
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from .krag_retriever import BaseRetriever, QueryContext, RetrievalResult


@dataclass
class LLMConfig:
    """Configuration for LLM-based retrieval"""
    model_name: str = "gemini-3-flash-preview"
    temperature: float = 0.3
    max_tokens: int = 1024
    project_id: Optional[str] = None
    location: str = "us-central1"


class ZeroShotLLMRetriever(BaseRetriever):
    """
    Zero-Shot LLM Baseline: No retrieval, pure parametric knowledge.

    Given a query and emotion profile, asks the LLM to generate movie
    recommendations directly from its training data, without any
    external knowledge base or retrieval.
    """

    def __init__(
        self,
        config: LLMConfig = None,
        content_items: List = None,
        available_movie_titles: Dict[str, str] = None
    ):
        self.config = config or LLMConfig()
        self.content_items = content_items or []
        self.available_movie_titles = available_movie_titles or {}
        self._client = None

    @property
    def client(self):
        """Lazy initialization of Vertex AI client"""
        if self._client is None:
            import vertexai
            from vertexai.generative_models import GenerativeModel

            if self.config.project_id:
                vertexai.init(
                    project=self.config.project_id,
                    location=self.config.location
                )

            self._client = GenerativeModel(self.config.model_name)

        return self._client

    def set_content_items(self, content_items: List):
        """Set available content items for matching LLM responses"""
        self.content_items = content_items
        self.available_movie_titles = {
            item.title.lower(): item for item in content_items
        }

    def retrieve(self, query_context: QueryContext, k: int = 10) -> List[RetrievalResult]:
        """
        Generate movie recommendations using LLM parametric knowledge.

        Args:
            query_context: Query with text and emotion profile
            k: Number of recommendations to generate

        Returns:
            List of RetrievalResult objects
        """
        emotion_description = self._format_emotions(query_context.user_emotions)

        prompt = f"""You are a movie recommendation expert. Based on the user's request and desired emotional experience, recommend exactly {k} movies.

User Request: {query_context.query_text}

Desired Emotional Experience:
{emotion_description}

IMPORTANT: Return ONLY a JSON array of movie titles. No explanations, no numbering, just the JSON array.
Example format: ["Movie Title 1", "Movie Title 2", "Movie Title 3"]

Respond with {k} movie titles that best match the request and emotional preferences:"""

        try:
            response = self.client.generate_content(
                prompt,
                generation_config={
                    "temperature": self.config.temperature,
                    "max_output_tokens": self.config.max_tokens
                }
            )

            raw_response = response.text.strip()
            movie_titles = self._parse_movie_titles(raw_response, k)

            return self._match_to_content_items(movie_titles, query_context)

        except Exception as e:
            print(f"LLM retrieval error: {e}")
            return []

    def _format_emotions(self, emotions) -> str:
        """Format emotion profile for prompt"""
        if emotions is None:
            return "No specific emotional preference"

        emotion_dict = emotions.to_dict()
        lines = []
        for emotion, value in emotion_dict.items():
            if value > 0.3:
                intensity = "high" if value > 0.7 else "moderate"
                lines.append(f"- {emotion.capitalize()}: {intensity} ({value:.1f})")

        return "\n".join(lines) if lines else "Neutral emotional preference"

    def _parse_movie_titles(self, response: str, k: int) -> List[str]:
        """Parse movie titles from LLM response"""
        json_match = re.search(r'\[.*?\]', response, re.DOTALL)

        if json_match:
            try:
                titles = json.loads(json_match.group())
                if isinstance(titles, list):
                    return [str(t).strip() for t in titles[:k]]
            except json.JSONDecodeError:
                pass

        lines = response.split('\n')
        titles = []
        for line in lines:
            line = line.strip()
            line = re.sub(r'^[\d]+[.\)]\s*', '', line)
            line = re.sub(r'^[-*]\s*', '', line)
            line = re.sub(r'^"(.*)"$', r'\1', line)
            line = re.sub(r"^'(.*)'$", r'\1', line)

            if line and len(line) > 2 and len(line) < 200:
                titles.append(line)

        return titles[:k]

    def _match_to_content_items(
        self,
        movie_titles: List[str],
        query_context: QueryContext
    ) -> List[RetrievalResult]:
        """Match LLM-generated titles to actual content items"""
        results = []
        matched_ids = set()

        for rank, title in enumerate(movie_titles):
            title_lower = title.lower()

            matched_item = None
            best_match_score = 0

            if title_lower in self.available_movie_titles:
                matched_item = self.available_movie_titles[title_lower]
                best_match_score = 1.0
            else:
                for item in self.content_items:
                    if item.id in matched_ids:
                        continue

                    item_title_lower = item.title.lower()

                    if title_lower in item_title_lower or item_title_lower in title_lower:
                        similarity = len(set(title_lower.split()) & set(item_title_lower.split())) / \
                                   max(len(title_lower.split()), len(item_title_lower.split()))

                        if similarity > best_match_score:
                            best_match_score = similarity
                            matched_item = item

            if matched_item and matched_item.id not in matched_ids:
                matched_ids.add(matched_item.id)

                combined_score = max(0.1, 1.0 - (rank * 0.1))

                results.append(RetrievalResult(
                    content_id=matched_item.id,
                    title=matched_item.title,
                    semantic_score=combined_score,
                    emotion_score=combined_score,
                    knowledge_score=0.0,
                    combined_score=combined_score,
                    metadata={
                        "llm_suggested_title": title,
                        "match_score": best_match_score,
                        "rank": rank + 1,
                        "retrieval_method": "zero_shot_llm"
                    }
                ))

        return results

    def explain_retrieval(self, result: RetrievalResult, query_context: QueryContext) -> str:
        """Explain why this movie was recommended"""
        return (
            f"Recommended by LLM based on parametric knowledge. "
            f"Original suggestion: '{result.metadata.get('llm_suggested_title', result.title)}'. "
            f"This baseline uses no external retrieval - purely the model's training data."
        )


class ZeroShotLLMRetrieverFactory:
    """Factory for creating Zero-Shot LLM retriever"""

    @staticmethod
    def create(
        content_items: List = None,
        project_id: str = None,
        model_name: str = "gemini-3-flash-preview"
    ) -> ZeroShotLLMRetriever:
        """
        Create a Zero-Shot LLM retriever.

        Args:
            content_items: List of ContentItem objects for matching
            project_id: GCP project ID for Vertex AI
            model_name: LLM model name

        Returns:
            Configured ZeroShotLLMRetriever
        """
        config = LLMConfig(
            model_name=model_name,
            project_id=project_id
        )

        retriever = ZeroShotLLMRetriever(config=config)

        if content_items:
            retriever.set_content_items(content_items)

        return retriever
