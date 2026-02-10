"""
K-RAG Response Generation using Vertex AI
Generates contextual responses using retrieved knowledge and emotional context
"""

import os
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import json

# Vertex AI imports
try:
    from langchain_google_vertexai import VertexAI, VertexAIEmbeddings
    from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from google.cloud import aiplatform
    VERTEX_AI_AVAILABLE = True
except ImportError:
    print("Warning: Vertex AI not available. Install google-cloud-aiplatform and langchain-google-vertexai")
    VERTEX_AI_AVAILABLE = False


from ..core.emotion_detection import EmotionProfile
from ..retrieval.krag_retriever import RetrievalResult, QueryContext


@dataclass
class ResponseConfig:
    """Configuration for response generation"""
    model_name: str = "gemini-3-flash-preview"
    temperature: float = 0.7
    max_tokens: int = 1024
    top_p: float = 0.9
    project_id: Optional[str] = None
    location: str = "us-central1"


class KRAGResponseGenerator:
    """
    Generates personalized responses using K-RAG retrieval results and Vertex AI
    """

    def __init__(self, config: ResponseConfig):
        self.config = config
        self.llm = None
        self.prompt_template = None
        self._initialize_llm()
        self._create_prompt_templates()

    def _initialize_llm(self):
        """Initialize the language model"""
        if VERTEX_AI_AVAILABLE and self.config.project_id:
            try:
                # Initialize Vertex AI
                aiplatform.init(
                    project=self.config.project_id,
                    location=self.config.location
                )

                self.llm = VertexAI(
                    model_name=self.config.model_name,
                    temperature=self.config.temperature,
                    max_output_tokens=self.config.max_tokens,
                    top_p=self.config.top_p,
                    project=self.config.project_id,
                    location=self.config.location
                )
                print(f"Initialized Vertex AI with model: {self.config.model_name}")

            except Exception as e:
                print(f"Error initializing Vertex AI: {e}")
                self.llm = None

        else:
            print("Vertex AI not available")
            self.llm = None

    def _create_prompt_templates(self):
        """Create prompt templates for different response types"""

        # Main K-RAG prompt template
        self.main_template = ChatPromptTemplate.from_template("""
You are an expert recommendation assistant with deep understanding of human emotions and content relationships.

QUERY CONTEXT:
User Query: "{query_text}"
User Emotional State: {emotion_context}

K-RAG RETRIEVED RECOMMENDATIONS (ranked by multi-modal relevance):
{retrieved_content}

KNOWLEDGE CONTEXT:
{knowledge_context}

TASK:
Based on the K-RAG retrieval results that combine semantic similarity, emotional compatibility, and knowledge graph relationships, provide 3 personalized recommendations that match the user's query and emotional state.

For each recommendation, explain:
1. Why it semantically matches their query
2. How it fits their emotional state
3. What knowledge connections make it relevant

Format your response as:
**[Title]**: [Brief description]
   - Content Match: [Why it matches the query]
   - Emotional Fit: [How it aligns with their emotions]
   - Knowledge Connection: [Relevant relationships/context]

Be empathetic, specific, and focus on the multi-modal aspects that make each recommendation suitable.

Recommendations:
""")

        # Explanation template
        self.explanation_template = PromptTemplate.from_template("""
Explain why the K-RAG system recommended "{title}" for the query "{query}".

K-RAG Analysis:
- Semantic Score: {semantic_score:.3f}
- Emotion Score: {emotion_score:.3f}
- Knowledge Score: {knowledge_score:.3f}
- Combined Score: {combined_score:.3f}

User Emotions: {user_emotions}
Knowledge Context: {knowledge_context}

Provide a clear, technical explanation of how the K-RAG methodology led to this recommendation.
""")

    def generate_response(self,
                         query_context: QueryContext,
                         retrieval_results: List[RetrievalResult],
                         max_recommendations: int = 3) -> str:
        """
        Generate personalized response using K-RAG results

        Args:
            query_context: Original query context
            retrieval_results: K-RAG retrieval results
            max_recommendations: Maximum number of recommendations

        Returns:
            Generated response text
        """
        if not retrieval_results:
            return "I apologize, but I couldn't find any suitable recommendations for your query. Please try rephrasing your request."

        if not self.llm:
            print("Warning: LLM not initialized, using fallback response")
            return self._generate_fallback_response(retrieval_results[:max_recommendations])

        # Format components for prompt injection
        emotion_context = self._format_emotion_context(query_context.user_emotions)
        retrieved_content = self._format_retrieved_content(retrieval_results[:max_recommendations])
        knowledge_context = self._format_knowledge_context(retrieval_results[:max_recommendations])

        try:
            # Create the complete prompt as a string (Vertex AI prefers string input)
            prompt_text = f"""You are an expert movie recommendation assistant with deep understanding of human emotions.

USER'S REQUEST: "{query_context.query_text}"

USER'S EMOTIONAL STATE: {emotion_context}

RETRIEVED MOVIES (ranked by relevance):
{retrieved_content}

KNOWLEDGE CONTEXT:
{knowledge_context}

TASK: Based on the retrieved movies and the user's emotional state, provide a warm, conversational response recommending the top 3 movies.

For each movie:
1. Explain why it matches their search
2. Describe how it might resonate with their current mood
3. Add any interesting connections or context

Be friendly, empathetic, and enthusiastic. Write naturally as if talking to a friend who asked for movie recommendations.

Your response:"""

            # Generate response
            response = self.llm.invoke(prompt_text)

            # Extract content based on LLM type
            if hasattr(response, 'content'):
                result = response.content
            elif isinstance(response, str):
                result = response
            else:
                result = str(response)
            
            # Check for empty response
            if not result or result.strip() == "":
                print("Warning: LLM returned empty response, using fallback")
                return self._generate_fallback_response(retrieval_results[:max_recommendations])
            
            return result

        except Exception as e:
            print(f"Error generating LLM response: {e}")
            import traceback
            traceback.print_exc()
            return self._generate_fallback_response(retrieval_results[:max_recommendations])

    def explain_recommendation(self,
                             result: RetrievalResult,
                             query_context: QueryContext) -> str:
        """
        Generate detailed explanation for a specific recommendation

        Args:
            result: Retrieval result to explain
            query_context: Original query context

        Returns:
            Detailed explanation text
        """
        try:
            # Get knowledge context for this item
            knowledge_context = "No additional knowledge context available"
            if hasattr(result, 'knowledge_score') and result.knowledge_score > 0:
                knowledge_context = result.explanation

            prompt = self.explanation_template.format(
                title=result.title,
                query=query_context.query_text,
                semantic_score=result.semantic_score,
                emotion_score=result.emotion_score,
                knowledge_score=result.knowledge_score,
                combined_score=result.combined_score,
                user_emotions=self._format_emotion_context(query_context.user_emotions),
                knowledge_context=knowledge_context
            )

            response = self.llm.invoke(prompt)

            if hasattr(response, 'content'):
                return response.content
            else:
                return str(response)

        except Exception as e:
            print(f"Error generating explanation: {e}")
            return f"Technical explanation: {result.explanation}"

    def _format_emotion_context(self, emotions: EmotionProfile) -> str:
        """Format emotion profile for prompt injection"""
        emotion_dict = emotions.to_dict()

        # Identify dominant emotions
        sorted_emotions = sorted(emotion_dict.items(), key=lambda x: x[1], reverse=True)

        context_parts = []
        for emotion, score in sorted_emotions:
            if score > 0.3:  # Only include significant emotions
                if score > 0.7:
                    level = "Very High"
                elif score > 0.5:
                    level = "High"
                else:
                    level = "Moderate"

                emotion_name = emotion.replace('_', ' ').title()
                context_parts.append(f"{emotion_name}: {level} ({score:.2f})")

        if not context_parts:
            return "Neutral emotional state"

        return " | ".join(context_parts)

    def _format_retrieved_content(self, results: List[RetrievalResult]) -> str:
        """Format retrieval results for prompt injection"""
        if not results:
            return "No recommendations found."

        content_parts = []
        for i, result in enumerate(results, 1):
            content_part = f"""
Item {i}: {result.title}
- K-RAG Combined Score: {result.combined_score:.3f}
- Semantic Relevance: {result.semantic_score:.3f}
- Emotional Match: {result.emotion_score:.3f}
- Knowledge Connection: {result.knowledge_score:.3f}
- Description: {result.metadata.get('description', 'No description available')[:200]}...
- Genres: {result.metadata.get('genres', 'Not specified')}
- Year: {result.metadata.get('year', 'Unknown')}
"""
            content_parts.append(content_part)

        return "\n".join(content_parts)

    def _format_knowledge_context(self, results: List[RetrievalResult]) -> str:
        """Format knowledge context from K-RAG for prompt injection"""
        knowledge_parts = []

        for result in results:
            if hasattr(result, 'knowledge_score') and result.knowledge_score > 0.3:
                knowledge_parts.append(f"• {result.title}: {result.explanation}")

        if not knowledge_parts:
            return "No significant knowledge graph connections identified."

        return "\n".join(knowledge_parts)

    def _generate_fallback_response(self, results: List[RetrievalResult]) -> str:
        """Generate fallback response when LLM fails"""
        if not results:
            return "I apologize, but I couldn't find suitable recommendations."

        response = "Here are my recommendations based on your search:\n\n"

        for i, result in enumerate(results, 1):
            response += f"**{i}. {result.title}**\n"
            
            # Add scores breakdown
            scores = []
            if result.semantic_score > 0:
                scores.append(f"Content Match: {result.semantic_score:.0%}")
            if result.emotion_score > 0:
                scores.append(f"Emotional Fit: {result.emotion_score:.0%}")
            if result.knowledge_score > 0:
                scores.append(f"Knowledge: {result.knowledge_score:.0%}")
            
            if scores:
                response += f"   ({', '.join(scores)})\n"
            
            if result.explanation:
                response += f"   {result.explanation}\n"
            
            response += "\n"

        response += "(Note: Full AI response unavailable, showing retrieval results)"
        return response

    def batch_generate_responses(self,
                               query_contexts: List[QueryContext],
                               retrieval_results_list: List[List[RetrievalResult]]) -> List[str]:
        """
        Generate responses for multiple queries

        Args:
            query_contexts: List of query contexts
            retrieval_results_list: List of retrieval results for each query

        Returns:
            List of generated responses
        """
        responses = []

        for query_context, retrieval_results in zip(query_contexts, retrieval_results_list):
            try:
                response = self.generate_response(query_context, retrieval_results)
                responses.append(response)
            except Exception as e:
                print(f"Error in batch generation: {e}")
                responses.append(self._generate_fallback_response(retrieval_results))

        return responses


class ResponseEvaluator:
    """
    Evaluate response quality and relevance
    """

    def __init__(self):
        self.metrics = {}

    def evaluate_response(self,
                         response: str,
                         query_context: QueryContext,
                         retrieval_results: List[RetrievalResult]) -> Dict[str, float]:
        """
        Evaluate response quality

        Args:
            response: Generated response
            query_context: Original query context
            retrieval_results: Retrieval results used

        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {
            'length_score': self._evaluate_length(response),
            'content_coverage': self._evaluate_coverage(response, retrieval_results),
            'emotion_awareness': self._evaluate_emotion_awareness(response, query_context.user_emotions),
            'structure_score': self._evaluate_structure(response)
        }

        # Overall quality score
        metrics['overall_quality'] = (
            0.3 * metrics['content_coverage'] +
            0.3 * metrics['emotion_awareness'] +
            0.2 * metrics['structure_score'] +
            0.2 * metrics['length_score']
        )

        return metrics

    def _evaluate_length(self, response: str) -> float:
        """Evaluate response length appropriateness"""
        word_count = len(response.split())

        if 50 <= word_count <= 300:  # Ideal range
            return 1.0
        elif 30 <= word_count <= 400:  # Acceptable range
            return 0.8
        elif word_count < 30:  # Too short
            return 0.3
        else:  # Too long
            return 0.5

    def _evaluate_coverage(self, response: str, results: List[RetrievalResult]) -> float:
        """Evaluate how well response covers retrieved content"""
        if not results:
            return 0.0

        covered_items = 0
        for result in results:
            if result.title.lower() in response.lower():
                covered_items += 1

        return covered_items / len(results)

    def _evaluate_emotion_awareness(self, response: str, emotions: EmotionProfile) -> float:
        """Evaluate emotional awareness in response"""
        # Keywords aligned with EMOTION_LABELS: happiness, sadness, anger, fear, surprise, disgust
        emotion_keywords = {
            'happiness': ['happy', 'joyful', 'cheerful', 'uplifting', 'positive', 'delightful'],
            'sadness': ['sad', 'melancholy', 'touching', 'emotional', 'heartfelt', 'poignant'],
            'anger': ['intense', 'dramatic', 'powerful', 'strong', 'confrontational'],
            'fear': ['suspenseful', 'thrilling', 'tense', 'exciting', 'scary', 'frightening'],
            'surprise': ['surprising', 'unexpected', 'unique', 'unusual', 'twist'],
            'disgust': ['disturbing', 'provocative', 'dark', 'unsettling', 'edgy']
        }

        emotion_dict = emotions.to_dict()
        relevant_emotions = [k for k, v in emotion_dict.items() if v > 0.3]

        if not relevant_emotions:
            return 0.5  # Neutral case

        emotion_mentions = 0
        for emotion in relevant_emotions:
            keywords = emotion_keywords.get(emotion, [])
            for keyword in keywords:
                if keyword in response.lower():
                    emotion_mentions += 1
                    break

        return min(1.0, emotion_mentions / len(relevant_emotions))

    def _evaluate_structure(self, response: str) -> float:
        """Evaluate response structure and formatting"""
        score = 0.0

        # Check for numbered or bulleted lists
        if any(marker in response for marker in ['1.', '2.', '•', '*', '-']):
            score += 0.4

        # Check for bold/emphasized titles
        if '**' in response or '*' in response:
            score += 0.3

        # Check for clear sections
        if ':' in response:
            score += 0.3

        return min(1.0, score)


class VertexAISetupHelper:
    """
    Helper class for setting up Vertex AI credentials and project
    """

    @staticmethod
    def setup_credentials(project_id: str, credentials_path: Optional[str] = None):
        """
        Setup Vertex AI credentials

        Args:
            project_id: Google Cloud project ID
            credentials_path: Path to service account JSON file
        """
        os.environ['GOOGLE_CLOUD_PROJECT'] = project_id

        if credentials_path and os.path.exists(credentials_path):
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
            print(f"Set credentials path: {credentials_path}")

        print(f"Configured for project: {project_id}")

    @staticmethod
    def test_connection(project_id: str, location: str = "us-central1"):
        """
        Test Vertex AI connection

        Args:
            project_id: Google Cloud project ID
            location: Vertex AI location

        Returns:
            Boolean indicating connection success
        """
        try:
            aiplatform.init(project=project_id, location=location)

            # Try to create a simple model instance
            model = VertexAI(
                model_name="gemini-3-flash-preview",
                project=project_id,
                location=location
            )

            # Simple test query
            test_response = model.invoke("Hello, this is a test.")
            print("Vertex AI connection successful!")
            print(f"Test response: {test_response}")
            return True

        except Exception as e:
            print(f"Vertex AI connection failed: {e}")
            return False

    @staticmethod
    def get_available_models(project_id: str, location: str = "us-central1") -> List[str]:
        """
        Get list of available Vertex AI models

        Args:
            project_id: Google Cloud project ID
            location: Vertex AI location

        Returns:
            List of available model names
        """
        try:
            # Common Vertex AI models
            return [
                "gemini-3-flash-preview",
                "gemini-2.5-flash",
                "gemini-1.5-pro",
                "gemini-1.0-pro",
                "text-bison",
                "text-bison-32k"
            ]
        except Exception as e:
            print(f"Error getting models: {e}")
            return []