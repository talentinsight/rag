"""
Vector Store Manager for RAG Implementation
Manages vector database operations with fallback to mock store
"""

import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import os
from dotenv import load_dotenv

from weaviate_client import WeaviateManager
from mock_vector_store import MockVectorStore
from semantic_chunker import TextChunk

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    Unified manager for vector database operations with automatic fallback
    """
    
    def __init__(self, 
                 prefer_weaviate: bool = True,
                 weaviate_url: str = "http://localhost:8080",
                 openai_api_key: str = None):
        """
        Initialize vector store manager
        
        Args:
            prefer_weaviate (bool): Whether to prefer Weaviate over mock store
            weaviate_url (str): Weaviate instance URL
            openai_api_key (str): OpenAI API key
        """
        self.prefer_weaviate = prefer_weaviate
        self.weaviate_url = weaviate_url
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        
        self.store = None
        self.store_type = None
        
        if not self.openai_api_key:
            logger.warning("No OpenAI API key provided. Some features may not work.")
    
    def initialize(self) -> bool:
        """
        Initialize the vector store (Weaviate or mock)
        
        Returns:
            bool: True if initialization successful
        """
        if self.prefer_weaviate:
            # Try Weaviate first
            try:
                logger.info("Attempting to initialize Weaviate...")
                weaviate_manager = WeaviateManager(
                    weaviate_url=self.weaviate_url,
                    openai_api_key=self.openai_api_key
                )
                
                if weaviate_manager.connect(max_retries=2, retry_delay=1):
                    if weaviate_manager.create_schema():
                        self.store = weaviate_manager
                        self.store_type = "weaviate"
                        logger.info("✅ Successfully initialized Weaviate")
                        return True
                
            except Exception as e:
                logger.warning(f"Weaviate initialization failed: {str(e)}")
        
        # Fallback to mock store
        logger.info("Falling back to mock vector store...")
        try:
            mock_store = MockVectorStore(openai_api_key=self.openai_api_key)
            self.store = mock_store
            self.store_type = "mock"
            logger.info("✅ Successfully initialized mock vector store")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize mock store: {str(e)}")
            return False
    
    def add_chunks_from_chunker(self, chunks: List[TextChunk]) -> bool:
        """
        Add chunks from semantic chunker to vector store
        
        Args:
            chunks (List[TextChunk]): List of TextChunk objects
            
        Returns:
            bool: True if successful
        """
        if not self.store:
            logger.error("Vector store not initialized")
            return False
        
        # Convert TextChunk objects to dictionaries
        chunk_dicts = []
        for chunk in chunks:
            chunk_dict = {
                "content": chunk.content,
                "chunk_id": chunk.chunk_id,
                "section_title": chunk.section_title,
                "chunk_type": chunk.chunk_type,
                "token_count": chunk.token_count,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "source_file": "AttentionAllYouNeed.pdf",
                "created_at": datetime.now().isoformat() + "Z"
            }
            chunk_dicts.append(chunk_dict)
        
        return self.add_chunks(chunk_dicts)
    
    def add_chunks(self, chunks: List[Dict[str, Any]]) -> bool:
        """
        Add chunks to vector store
        
        Args:
            chunks (List[Dict]): List of chunk dictionaries
            
        Returns:
            bool: True if successful
        """
        if not self.store:
            logger.error("Vector store not initialized")
            return False
        
        return self.store.add_chunks(chunks)
    
    def search_similar(self, 
                      query: str, 
                      limit: int = 5,
                      min_score: float = 0.1) -> List[Dict[str, Any]]:
        """
        Search for similar chunks
        
        Args:
            query (str): Search query
            limit (int): Maximum number of results
            min_score (float): Minimum similarity score
            
        Returns:
            List[Dict]: List of similar chunks with metadata
        """
        if not self.store:
            logger.error("Vector store not initialized")
            return []
        
        return self.store.search_similar(query, limit, min_score)
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific chunk by its ID
        
        Args:
            chunk_id (str): Chunk identifier
            
        Returns:
            Optional[Dict]: Chunk data if found, None otherwise
        """
        if not self.store:
            logger.error("Vector store not initialized")
            return None
        
        return self.store.get_chunk_by_id(chunk_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store
        
        Returns:
            Dict: Vector store statistics
        """
        if not self.store:
            return {"error": "Vector store not initialized"}
        
        if self.store_type == "weaviate":
            stats = self.store.get_collection_stats()
        else:
            stats = self.store.get_stats()
        
        stats["store_type"] = self.store_type
        return stats
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get collection statistics (alias for get_stats for compatibility)
        
        Returns:
            Dict: Vector store statistics
        """
        return self.get_stats()
    
    def save_mock_store(self, filepath: str) -> bool:
        """
        Save mock store to file (only works with mock store)
        
        Args:
            filepath (str): Path to save the store
            
        Returns:
            bool: True if successful
        """
        if self.store_type != "mock":
            logger.warning("Save operation only supported for mock store")
            return False
        
        return self.store.save_to_file(filepath)
    
    def load_mock_store(self, filepath: str) -> bool:
        """
        Load mock store from file
        
        Args:
            filepath (str): Path to load the store from
            
        Returns:
            bool: True if successful
        """
        if self.store_type != "mock":
            logger.warning("Load operation only supported for mock store")
            return False
        
        return self.store.load_from_file(filepath)
    
    def close(self):
        """
        Close the vector store connection
        """
        if self.store and self.store_type == "weaviate":
            self.store.close()
        
        logger.info(f"Closed {self.store_type} vector store")


def main():
    """
    Test the vector store manager with semantic chunks
    """
    from semantic_chunker import SemanticChunker
    
    try:
        # Load processed text
        text_file = "/Users/sam/Desktop/rag/processed_attention_paper.txt"
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Create semantic chunks
        logger.info("Creating semantic chunks...")
        chunker = SemanticChunker(chunk_size=300, chunk_overlap=30, min_chunk_size=50)
        chunks = chunker.chunk_text(text)
        
        # Initialize vector store manager
        logger.info("Initializing vector store manager...")
        vector_manager = VectorStoreManager(prefer_weaviate=True)
        
        if not vector_manager.initialize():
            logger.error("Failed to initialize vector store")
            return
        
        # Add chunks to vector store
        logger.info(f"Adding {len(chunks)} chunks to vector store...")
        if vector_manager.add_chunks_from_chunker(chunks):
            print("✅ Successfully added chunks to vector store")
            
            # Get stats
            stats = vector_manager.get_stats()
            print("\\n=== Vector Store Stats ===")
            for key, value in stats.items():
                print(f"{key}: {value}")
            
            # Test search
            test_queries = [
                "attention mechanism",
                "transformer architecture", 
                "multi-head attention",
                "neural networks"
            ]
            
            for query in test_queries:
                print(f"\\n=== Search Results for '{query}' ===")
                results = vector_manager.search_similar(query, limit=3, min_score=0.1)
                
                for i, result in enumerate(results, 1):
                    print(f"Result {i}:")
                    print(f"  Chunk ID: {result['chunk_id']}")
                    print(f"  Score: {result['score']:.3f}")
                    print(f"  Section: {result['section_title']}")
                    print(f"  Content: {result['content'][:100]}...")
                    print()
            
            # Save mock store if using mock
            if vector_manager.store_type == "mock":
                save_path = "/Users/sam/Desktop/rag/vector_store_backup.pkl"
                if vector_manager.save_mock_store(save_path):
                    print(f"\\n✅ Saved mock store to {save_path}")
        
        print("\\n✅ Vector store manager test completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise
    
    finally:
        if 'vector_manager' in locals():
            vector_manager.close()


if __name__ == "__main__":
    main()
