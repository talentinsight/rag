"""
OpenAI Client Module for RAG Implementation
Handles OpenAI API integration for response generation
"""

import logging
import os
from typing import List, Dict, Any, Optional
import openai
from openai import OpenAI
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpenAIClient:
    """
    A class to handle OpenAI API operations for RAG
    """
    
    def __init__(self, 
                 api_key: str = None,
                 model: str = "gpt-4-turbo-preview",  # Using GPT-4 Turbo as GPT-4.1 equivalent
                 temperature: float = 0.7,
                 max_tokens: int = 1500):
        """
        Initialize OpenAI client
        
        Args:
            api_key (str): OpenAI API key
            model (str): Model to use for generation
            temperature (float): Sampling temperature
            max_tokens (int): Maximum tokens in response
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        
        logger.info(f"Initialized OpenAI client with model: {self.model}")
    
    def generate_direct_response(self, 
                                query: str, 
                                model: str = None,
                                temperature: float = None,
                                max_tokens: int = None) -> Dict[str, Any]:
        """
        Generate a direct response without RAG context (for guardrails testing)
        
        Args:
            query (str): User query
            model (str): Model to use (optional, uses instance default)
            temperature (float): Temperature (optional, uses instance default)
            max_tokens (int): Max tokens (optional, uses instance default)
            
        Returns:
            Dict: Response with metadata
        """
        try:
            # Use provided parameters or defaults
            use_model = model or self.model
            use_temperature = temperature if temperature is not None else self.temperature
            use_max_tokens = max_tokens or self.max_tokens
            
            logger.info(f"Generating direct response for query: '{query[:50]}...'")
            
            # Call OpenAI API directly without RAG context
            response = self.client.chat.completions.create(
                model=use_model,
                messages=[
                    {"role": "user", "content": query}
                ],
                temperature=use_temperature,
                max_tokens=use_max_tokens,
                stream=False
            )
            
            # Extract response
            answer = response.choices[0].message.content
            
            # Build response metadata
            result = {
                "answer": answer,
                "query": query,
                "model": use_model,
                "total_tokens": response.usage.total_tokens,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            }
            
            logger.info(f"Generated direct response ({response.usage.completion_tokens} tokens)")
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate direct response: {str(e)}")
            return {
                "answer": f"ERROR: {str(e)}",
                "query": query,
                "error": str(e),
                "model": model or self.model,
                "total_tokens": 0
            }

    def generate_response(self,
                         query: str, 
                         context_chunks: List[Dict[str, Any]],
                         system_prompt: str = None) -> Dict[str, Any]:
        """
        Generate a response using retrieved context chunks
        
        Args:
            query (str): User query
            context_chunks (List[Dict]): Retrieved context chunks
            system_prompt (str): Custom system prompt
            
        Returns:
            Dict: Response with metadata
        """
        try:
            # Build context from chunks
            context = self._build_context(context_chunks)
            
            # Create system prompt
            if not system_prompt:
                system_prompt = self._get_default_system_prompt()
            
            # Create user prompt
            user_prompt = self._build_user_prompt(query, context)
            
            logger.info(f"Generating response for query: '{query[:50]}...'")
            logger.info(f"Using {len(context_chunks)} context chunks")
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=False
            )
            
            # Extract response
            answer = response.choices[0].message.content
            
            # Build response metadata
            result = {
                "answer": answer,
                "query": query,
                "model": self.model,
                "context_chunks_used": len(context_chunks),
                "total_tokens": response.usage.total_tokens,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "llm_prompt": {
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt
                },
                "sources": [
                    {
                        "chunk_id": chunk.get("chunk_id", ""),
                        "section": chunk.get("section_title", ""),
                        "score": chunk.get("score", 0.0),
                        "content": chunk.get("content", "")
                    }
                    for chunk in context_chunks
                ]
            }
            
            logger.info(f"Generated response ({response.usage.completion_tokens} tokens)")
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate response: {str(e)}")
            return {
                "answer": f"ERROR: {str(e)}",
                "query": query,
                "error": str(e),
                "model": self.model,
                "context_chunks_used": len(context_chunks),
                "sources": []
            }
    
    def _build_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Build context string from retrieved chunks
        
        Args:
            chunks (List[Dict]): Retrieved chunks
            
        Returns:
            str: Formatted context
        """
        if not chunks:
            return "No relevant context found."
        
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            section = chunk.get("section_title", "Unknown Section")
            content = chunk.get("content", "")
            score = chunk.get("score", 0.0)
            
            context_part = f"[Context {i}] (Section: {section}, Relevance: {score:.3f})\\n{content}"
            context_parts.append(context_part)
        
        return "\\n\\n".join(context_parts)
    
    def _get_default_system_prompt(self) -> str:
        """
        Get the default system prompt for RAG
        
        Returns:
            str: System prompt
        """
        return """You are a helpful AI assistant specializing in the "Attention Is All You Need" paper.

CRITICAL CONSTRAINT: Your response MUST be exactly 50 words or fewer. No exceptions.

Guidelines:
1. If paper context is provided, answer based on that context
2. If no context available, politely acknowledge you don't have that specific information
3. For general questions or comments, respond naturally and helpfully
4. Be conversational and natural, not robotic
5. Never fabricate information about the paper

Remember: 50 words maximum. Be helpful and authentic."""
    
    def _build_user_prompt(self, query: str, context: str) -> str:
        """
        Build user prompt with query and context
        
        Args:
            query (str): User query
            context (str): Retrieved context
            
        Returns:
            str: Formatted user prompt
        """
        return f"""Based on the following context from the "Attention Is All You Need" paper, please answer the question.

Context:
{context}

Question: {query}

Please provide a comprehensive answer based on the context provided. If the context doesn't contain sufficient information to fully answer the question, please indicate what aspects cannot be addressed with the available information."""
    
    def generate_embeddings(self, texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
        """
        Generate embeddings for texts using OpenAI
        
        Args:
            texts (List[str]): List of texts to embed
            model (str): Embedding model to use
            
        Returns:
            List[List[float]]: List of embeddings
        """
        try:
            logger.info(f"Generating embeddings for {len(texts)} texts")
            
            response = self.client.embeddings.create(
                model=model,
                input=texts
            )
            
            embeddings = [embedding.embedding for embedding in response.data]
            logger.info(f"Generated {len(embeddings)} embeddings")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")
            return []
    
    def test_connection(self) -> bool:
        """
        Test OpenAI API connection
        
        Returns:
            bool: True if connection successful
        """
        try:
            # Simple test call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            
            logger.info("✅ OpenAI API connection test successful")
            return True
            
        except Exception as e:
            logger.error(f"❌ OpenAI API connection test failed: {str(e)}")
            return False
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available models
        
        Returns:
            List[str]: List of model names
        """
        try:
            models = self.client.models.list()
            model_names = [model.id for model in models.data]
            return sorted(model_names)
            
        except Exception as e:
            logger.error(f"Failed to get models: {str(e)}")
            return []


def main():
    """
    Test the OpenAI client
    Note: This is a test function. For production use, initialize with actual document chunks from RAG pipeline.
    """
    logger.error("OpenAIClient main() is for testing only. Use RAG pipeline for production.")
    logger.info("To use OpenAIClient properly:")
    logger.info("  1. Initialize: client = OpenAIClient(api_key=your_key)")
    logger.info("  2. Test connection: client.test_connection()")
    logger.info("  3. Load actual document chunks from vector store")
    logger.info("  4. Generate response: client.generate_response(query, actual_chunks)")
    raise NotImplementedError("Test function removed. Use RAG pipeline with actual document chunks in production.")


if __name__ == "__main__":
    main()
