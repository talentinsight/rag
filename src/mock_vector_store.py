"""
Mock Vector Store for RAG Implementation
A simple in-memory vector store for testing when Weaviate is not available
"""

import logging
import json
import pickle
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockVectorStore:
    """
    A simple in-memory vector store using TF-IDF for similarity search
    """
    
    def __init__(self, openai_api_key: str = None):
        """
        Initialize the mock vector store
        
        Args:
            openai_api_key (str): OpenAI API key for embeddings (optional for mock)
        """
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.chunks = []
        self.vectors = None
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.is_fitted = False
        
        # Set OpenAI API key if available
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
    
    def add_chunks(self, chunks: List[Dict[str, Any]]) -> bool:
        """
        Add document chunks to the vector store
        
        Args:
            chunks (List[Dict]): List of chunk dictionaries
            
        Returns:
            bool: True if successful
        """
        try:
            logger.info(f"Adding {len(chunks)} chunks to mock vector store")
            
            # Store chunks
            self.chunks.extend(chunks)
            
            # Extract text content for vectorization
            texts = [chunk.get("content", "") for chunk in self.chunks]
            
            # Fit TF-IDF vectorizer and transform texts
            self.vectors = self.vectorizer.fit_transform(texts)
            self.is_fitted = True
            
            logger.info(f"Successfully added chunks. Total chunks: {len(self.chunks)}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add chunks: {str(e)}")
            return False
    
    def search_similar(self, 
                      query: str, 
                      limit: int = 5,
                      min_score: float = 0.1) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using TF-IDF cosine similarity
        
        Args:
            query (str): Search query
            limit (int): Maximum number of results
            min_score (float): Minimum similarity score
            
        Returns:
            List[Dict]: List of similar chunks with metadata
        """
        try:
            logger.info(f"Searching for similar chunks: '{query[:50]}...'")
            
            if not self.is_fitted or len(self.chunks) == 0:
                logger.warning("No chunks available for search")
                return []
            
            # Transform query using fitted vectorizer
            query_vector = self.vectorizer.transform([query])
            
            # Calculate cosine similarities
            similarities = cosine_similarity(query_vector, self.vectors).flatten()
            
            # Get top similar chunks
            top_indices = np.argsort(similarities)[::-1][:limit]
            
            results = []
            for idx in top_indices:
                score = similarities[idx]
                if score >= min_score:
                    chunk = self.chunks[idx]
                    result = {
                        "content": chunk.get("content", ""),
                        "chunk_id": chunk.get("chunk_id", ""),
                        "section_title": chunk.get("section_title", ""),
                        "token_count": chunk.get("token_count", 0),
                        "score": float(score),
                        "distance": 1.0 - score  # Convert similarity to distance
                    }
                    results.append(result)
            
            logger.info(f"Found {len(results)} similar chunks")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific chunk by its ID
        
        Args:
            chunk_id (str): Chunk identifier
            
        Returns:
            Optional[Dict]: Chunk data if found, None otherwise
        """
        try:
            for chunk in self.chunks:
                if chunk.get("chunk_id") == chunk_id:
                    return {
                        "content": chunk.get("content", ""),
                        "chunk_id": chunk.get("chunk_id", ""),
                        "section_title": chunk.get("section_title", ""),
                        "token_count": chunk.get("token_count", 0),
                        "start_char": chunk.get("start_char", 0),
                        "end_char": chunk.get("end_char", 0),
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get chunk by ID: {str(e)}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store
        
        Returns:
            Dict: Vector store statistics
        """
        return {
            "total_chunks": len(self.chunks),
            "is_fitted": self.is_fitted,
            "vector_dimensions": self.vectors.shape[1] if self.vectors is not None else 0,
            "store_type": "mock_tfidf"
        }
    
    def save_to_file(self, filepath: str) -> bool:
        """
        Save the vector store to a file
        
        Args:
            filepath (str): Path to save the store
            
        Returns:
            bool: True if successful
        """
        try:
            data = {
                "chunks": self.chunks,
                "vectorizer": self.vectorizer,
                "vectors": self.vectors,
                "is_fitted": self.is_fitted
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            
            logger.info(f"Saved vector store to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save vector store: {str(e)}")
            return False
    
    def load_from_file(self, filepath: str) -> bool:
        """
        Load the vector store from a file
        
        Args:
            filepath (str): Path to load the store from
            
        Returns:
            bool: True if successful
        """
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.chunks = data["chunks"]
            self.vectorizer = data["vectorizer"]
            self.vectors = data["vectors"]
            self.is_fitted = data["is_fitted"]
            
            logger.info(f"Loaded vector store from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load vector store: {str(e)}")
            return False
    
    def clear(self):
        """
        Clear all data from the vector store
        """
        self.chunks = []
        self.vectors = None
        self.is_fitted = False
        logger.info("Cleared vector store")


def main():
    """
    Test the mock vector store
    Note: This is a test function. For production use, initialize with actual document chunks.
    """
    logger.error("MockVectorStore main() is for testing only. Use proper document loading in production.")
    logger.info("To use MockVectorStore properly:")
    logger.info("  1. Initialize: store = MockVectorStore()")
    logger.info("  2. Load chunks from actual documents using pdf_processor and semantic_chunker")
    logger.info("  3. Add chunks: store.add_chunks(document_chunks)")
    logger.info("  4. Search: results = store.search_similar(query, limit=5)")
    raise NotImplementedError("Test function removed. Use proper document loading pipeline in production.")


if __name__ == "__main__":
    main()
