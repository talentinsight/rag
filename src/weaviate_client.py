"""
Weaviate Client Module for RAG Implementation
Handles vector database operations for document chunks
"""

import logging
import os
import time
from typing import List, Dict, Any, Optional
import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import Filter
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WeaviateManager:
    """
    A class to manage Weaviate vector database operations
    """
    
    def __init__(self, 
                 weaviate_url: str = "http://localhost:8080",
                 openai_api_key: str = None,
                 collection_name: str = "AttentionPaper"):
        """
        Initialize Weaviate client
        
        Args:
            weaviate_url (str): Weaviate instance URL
            openai_api_key (str): OpenAI API key for embeddings
            collection_name (str): Name of the collection to store chunks
        """
        self.weaviate_url = weaviate_url
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
        
        # Initialize OpenAI client
        openai.api_key = self.openai_api_key
        
    def connect(self, max_retries: int = 5, retry_delay: int = 2) -> bool:
        """
        Connect to Weaviate instance with retries
        
        Args:
            max_retries (int): Maximum number of connection attempts
            retry_delay (int): Delay between retries in seconds
            
        Returns:
            bool: True if connection successful, False otherwise
        """
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempting to connect to Weaviate at {self.weaviate_url} (attempt {attempt + 1})")
                
                # Determine connection type based on URL
                if "localhost" in self.weaviate_url or "127.0.0.1" in self.weaviate_url:
                    # Local connection
                    self.client = weaviate.connect_to_local(
                        host="localhost",
                        port=8080,
                        grpc_port=50051,
                        headers={
                            "X-OpenAI-Api-Key": self.openai_api_key
                        }
                    )
                else:
                    # Cloud or remote connection
                    weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
                    if weaviate_api_key:
                        self.client = weaviate.connect_to_wcs(
                            cluster_url=self.weaviate_url,
                            auth_credentials=weaviate.auth.AuthApiKey(weaviate_api_key),
                            headers={
                                "X-OpenAI-Api-Key": self.openai_api_key
                            }
                        )
                    else:
                        # Try without authentication for testing
                        self.client = weaviate.Client(
                            url=self.weaviate_url,
                            additional_headers={
                                "X-OpenAI-Api-Key": self.openai_api_key
                            }
                        )
                
                # Test connection
                if self.client.is_ready():
                    logger.info("Successfully connected to Weaviate")
                    return True
                else:
                    logger.warning(f"Weaviate not ready on attempt {attempt + 1}")
                    
            except Exception as e:
                logger.warning(f"Connection attempt {attempt + 1} failed: {str(e)}")
                
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
        
        logger.error("Failed to connect to Weaviate after all attempts")
        logger.info("💡 To start local Weaviate:")
        logger.info("   1. Start Docker Desktop")
        logger.info("   2. Run: docker-compose up -d")
        logger.info("   3. Wait for Weaviate to be ready at http://localhost:8080")
        return False
    
    def create_schema(self) -> bool:
        """
        Create the schema for document chunks
        
        Returns:
            bool: True if schema created successfully, False otherwise
        """
        try:
            logger.info(f"Creating schema for collection: {self.collection_name}")
            
            # Check if collection already exists
            if self.client.collections.exists(self.collection_name):
                logger.info(f"Collection {self.collection_name} already exists")
                self.collection = self.client.collections.get(self.collection_name)
                return True
            
            # Create collection with properties
            self.collection = self.client.collections.create(
                name=self.collection_name,
                vectorizer_config=Configure.Vectorizer.text2vec_openai(
                    model="text-embedding-3-small"  # Using OpenAI's latest embedding model
                ),
                generative_config=Configure.Generative.openai(
                    model="gpt-4"  # For generative queries if needed
                ),
                properties=[
                    Property(name="content", data_type=DataType.TEXT),
                    Property(name="chunk_id", data_type=DataType.TEXT),
                    Property(name="section_title", data_type=DataType.TEXT),
                    Property(name="chunk_type", data_type=DataType.TEXT),
                    Property(name="token_count", data_type=DataType.INT),
                    Property(name="start_char", data_type=DataType.INT),
                    Property(name="end_char", data_type=DataType.INT),
                    Property(name="source_file", data_type=DataType.TEXT),
                    Property(name="created_at", data_type=DataType.DATE),
                ]
            )
            
            logger.info(f"Successfully created collection: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create schema: {str(e)}")
            return False
    
    def add_chunks(self, chunks: List[Dict[str, Any]]) -> bool:
        """
        Add document chunks to Weaviate
        
        Args:
            chunks (List[Dict]): List of chunk dictionaries
            
        Returns:
            bool: True if all chunks added successfully, False otherwise
        """
        try:
            logger.info(f"Adding {len(chunks)} chunks to Weaviate")
            
            # Ensure collection is initialized
            if not self.collection:
                logger.warning("Collection not initialized. Attempting to create schema...")
                if not self.create_schema():
                    logger.error("Failed to create schema during chunk addition")
                    return False
                
                # Wait a moment for schema to be ready
                time.sleep(2)
            
            # Prepare data for batch insert
            objects = []
            for chunk in chunks:
                obj = {
                    "content": chunk.get("content", ""),
                    "chunk_id": chunk.get("chunk_id", ""),
                    "section_title": chunk.get("section_title", ""),
                    "chunk_type": chunk.get("chunk_type", "content"),
                    "token_count": chunk.get("token_count", 0),
                    "start_char": chunk.get("start_char", 0),
                    "end_char": chunk.get("end_char", 0),
                    "source_file": chunk.get("source_file", "AttentionAllYouNeed.pdf"),
                    "created_at": chunk.get("created_at", "2024-01-01T00:00:00Z"),
                }
                objects.append(obj)
            
            # Batch insert
            with self.collection.batch.dynamic() as batch:
                for obj in objects:
                    batch.add_object(
                        properties=obj
                    )
            
            logger.info(f"Successfully added {len(chunks)} chunks to Weaviate")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add chunks: {str(e)}")
            return False
    
    def search_similar(self, 
                      query: str, 
                      limit: int = 5,
                      min_score: float = 0.7) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using semantic similarity
        
        Args:
            query (str): Search query
            limit (int): Maximum number of results
            min_score (float): Minimum similarity score
            
        Returns:
            List[Dict]: List of similar chunks with metadata
        """
        try:
            logger.info(f"Searching for similar chunks: '{query[:50]}...' with min_score={min_score}")
            
            if not self.collection:
                logger.error("Collection not initialized")
                return []
            
            # Perform vector search
            response = self.collection.query.near_text(
                query=query,
                limit=limit,
                return_metadata=["score", "distance"]
            )
            
            logger.info(f"Weaviate returned {len(response.objects)} objects")
            
            results = []
            for i, obj in enumerate(response.objects):
                # Get score and distance
                score = obj.metadata.score if obj.metadata.score else 0
                distance = obj.metadata.distance if obj.metadata.distance else 1.0
                
                logger.info(f"Object {i}: score={score}, distance={distance}")
                
                # In Weaviate, lower distance = higher similarity
                # Convert distance to similarity score if needed
                if score == 0 and distance > 0:
                    score = 1.0 - distance
                
                # Filter by minimum score if specified
                if score >= min_score:
                    result = {
                        "content": obj.properties.get("content", ""),
                        "chunk_id": obj.properties.get("chunk_id", ""),
                        "section_title": obj.properties.get("section_title", ""),
                        "token_count": obj.properties.get("token_count", 0),
                        "score": score,
                        "distance": distance
                    }
                    results.append(result)
                else:
                    logger.info(f"Filtered out object {i}: score {score} < min_score {min_score}")
            
            logger.info(f"Found {len(results)} similar chunks after filtering")
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
            if not self.collection:
                logger.error("Collection not initialized")
                return None
            
            response = self.collection.query.fetch_objects(
                where=Filter.by_property("chunk_id").equal(chunk_id),
                limit=1
            )
            
            if response.objects:
                obj = response.objects[0]
                return {
                    "content": obj.properties.get("content", ""),
                    "chunk_id": obj.properties.get("chunk_id", ""),
                    "section_title": obj.properties.get("section_title", ""),
                    "token_count": obj.properties.get("token_count", 0),
                    "start_char": obj.properties.get("start_char", 0),
                    "end_char": obj.properties.get("end_char", 0),
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get chunk by ID: {str(e)}")
            return None
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection
        
        Returns:
            Dict: Collection statistics
        """
        try:
            if not self.collection:
                logger.error("Collection not initialized")
                return {}
            
            # Get object count
            response = self.collection.aggregate.over_all(total_count=True)
            
            stats = {
                "collection_name": self.collection_name,
                "total_objects": response.total_count,
                "weaviate_url": self.weaviate_url,
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {str(e)}")
            return {}
    
    def delete_collection(self) -> bool:
        """
        Delete the entire collection
        
        Returns:
            bool: True if deleted successfully, False otherwise
        """
        try:
            if self.client.collections.exists(self.collection_name):
                self.client.collections.delete(self.collection_name)
                logger.info(f"Deleted collection: {self.collection_name}")
                self.collection = None
                return True
            else:
                logger.info(f"Collection {self.collection_name} does not exist")
                return True
                
        except Exception as e:
            logger.error(f"Failed to delete collection: {str(e)}")
            return False
    
    def close(self):
        """
        Close the Weaviate client connection
        """
        if self.client:
            self.client.close()
            logger.info("Closed Weaviate client connection")


def main():
    """
    Test the Weaviate client setup
    Note: This is a test function. For production use, initialize with actual document chunks from RAG pipeline.
    """
    logger.error("WeaviateManager main() is for testing only. Use RAG pipeline for production.")
    logger.info("To use WeaviateManager properly:")
    logger.info("  1. Initialize: manager = WeaviateManager()")
    logger.info("  2. Connect: manager.connect()")
    logger.info("  3. Create schema: manager.create_schema()")
    logger.info("  4. Load actual document chunks from pdf_processor and semantic_chunker")
    logger.info("  5. Add chunks: manager.add_chunks(document_chunks)")
    logger.info("  6. Search: results = manager.search_similar(query, limit=5)")
    raise NotImplementedError("Test function removed. Use RAG pipeline with actual document chunks in production.")


if __name__ == "__main__":
    main()
