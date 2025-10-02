"""
RAG Pipeline Module
Combines all components for complete Retrieval-Augmented Generation
"""

import logging
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv

from .vector_store_manager import VectorStoreManager
from .openai_client import OpenAIClient
from .semantic_chunker import SemanticChunker
from .pdf_processor import PDFProcessor

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Complete RAG pipeline for the Attention paper
    """
    
    def __init__(self, 
                 openai_api_key: str = None,
                 weaviate_url: str = "http://localhost:8080",
                 prefer_weaviate: bool = True):
        """
        Initialize RAG pipeline
        
        Args:
            openai_api_key (str): OpenAI API key
            weaviate_url (str): Weaviate instance URL
            prefer_weaviate (bool): Whether to prefer Weaviate over mock store
        """
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.weaviate_url = weaviate_url
        self.prefer_weaviate = prefer_weaviate
        
        # Initialize components
        self.vector_store = None
        self.openai_client = None
        self.is_initialized = False
        
        logger.info("RAG Pipeline initialized")
    
    def initialize(self) -> bool:
        """
        Initialize all pipeline components
        
        Returns:
            bool: True if initialization successful
        """
        try:
            logger.info("Initializing RAG pipeline components...")
            
            # Initialize vector store
            self.vector_store = VectorStoreManager(
                prefer_weaviate=self.prefer_weaviate,
                weaviate_url=self.weaviate_url,
                openai_api_key=self.openai_api_key
            )
            
            if not self.vector_store.initialize():
                logger.error("Failed to initialize vector store")
                return False
            
            # Initialize OpenAI client (only if API key is available)
            if self.openai_api_key and self.openai_api_key != "your_openai_api_key_here":
                try:
                    self.openai_client = OpenAIClient(api_key=self.openai_api_key)
                    if not self.openai_client.test_connection():
                        logger.warning("OpenAI connection test failed, but continuing...")
                except Exception as e:
                    logger.warning(f"Failed to initialize OpenAI client: {str(e)}")
                    self.openai_client = None
            else:
                logger.warning("No valid OpenAI API key provided. Response generation will be limited.")
                self.openai_client = None
            
            self.is_initialized = True
            logger.info("✅ RAG pipeline initialization completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG pipeline: {str(e)}")
            return False
    
    def load_document(self, pdf_path: str, clear_existing: bool = False) -> bool:
        """
        Load and process a PDF document into the vector store
        
        Args:
            pdf_path (str): Path to PDF file
            clear_existing (bool): Whether to clear existing chunks before loading
            
        Returns:
            bool: True if successful
        """
        try:
            if not self.is_initialized:
                logger.error("Pipeline not initialized. Call initialize() first.")
                return False
            
            logger.info(f"Loading document: {pdf_path}")
            
            # Clear existing chunks if requested
            if clear_existing:
                logger.info("Clearing existing chunks from vector store...")
                if hasattr(self.vector_store.store, 'delete_collection'):
                    # For Weaviate: delete and recreate collection
                    if self.vector_store.store.delete_collection():
                        logger.info("✅ Deleted existing collection")
                        # Recreate schema
                        if not self.vector_store.store.create_schema():
                            logger.error("Failed to recreate schema")
                            return False
                    else:
                        logger.warning("Failed to delete collection, continuing anyway...")
                elif hasattr(self.vector_store.store, 'clear'):
                    # For MockVectorStore: clear data
                    self.vector_store.store.clear()
                    logger.info("✅ Cleared existing chunks")
            
            # Process PDF
            pdf_processor = PDFProcessor()
            pdf_result = pdf_processor.process_pdf(pdf_path)
            
            if not pdf_result:
                logger.error("Failed to process PDF")
                return False
            
            # Create semantic chunks with optimized settings
            chunker = SemanticChunker(chunk_size=800, chunk_overlap=100, min_chunk_size=200)
            chunks = chunker.chunk_text(pdf_result['text'])
            
            if not chunks:
                logger.error("Failed to create chunks")
                return False
            
            # Add chunks to vector store
            if not self.vector_store.add_chunks_from_chunker(chunks):
                logger.error("Failed to add chunks to vector store")
                return False
            
            logger.info(f"✅ Successfully loaded document with {len(chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load document: {str(e)}")
            return False
    
    def query(self, 
              question: str, 
              num_chunks: int = 5,
              min_score: float = 0.0) -> Dict[str, Any]:
        """
        Query the RAG system
        
        Args:
            question (str): User question
            num_chunks (int): Number of chunks to retrieve
            min_score (float): Minimum similarity score
            
        Returns:
            Dict: Response with answer and metadata
        """
        try:
            if not self.is_initialized:
                return {
                    "error": "Pipeline not initialized",
                    "answer": "Please initialize the RAG pipeline first.",
                    "question": question
                }
            
            logger.info(f"Processing query: '{question[:50]}...'")
            
            # Retrieve relevant chunks
            relevant_chunks = self.vector_store.search_similar(
                query=question,
                limit=num_chunks,
                min_score=min_score
            )
            
            if not relevant_chunks:
                return {
                    "answer": "I couldn't find any relevant information in the document to answer your question.",
                    "question": question,
                    "chunks_found": 0,
                    "sources": []
                }
            
            logger.info(f"Retrieved {len(relevant_chunks)} relevant chunks")
            
            # Generate response using OpenAI (if available)
            if self.openai_client:
                response = self.openai_client.generate_response(question, relevant_chunks)
                response["chunks_found"] = len(relevant_chunks)
                return response
            else:
                # Fallback: return chunks without AI generation
                context = "\\n\\n".join([
                    f"[Chunk {i+1}] (Section: {chunk.get('section_title', 'Unknown')}, Score: {chunk.get('score', 0):.3f})\\n{chunk.get('content', '')}"
                    for i, chunk in enumerate(relevant_chunks)
                ])
                
                return {
                    "answer": f"Based on the retrieved context:\\n\\n{context}\\n\\nNote: AI response generation is not available. Please provide an OpenAI API key for enhanced responses.",
                    "question": question,
                    "chunks_found": len(relevant_chunks),
                    "sources": [
                        {
                            "chunk_id": chunk.get("chunk_id", ""),
                            "section": chunk.get("section_title", ""),
                            "score": chunk.get("score", 0.0),
                            "content": chunk.get("content", "")
                        }
                        for chunk in relevant_chunks
                    ]
                }
            
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            return {
                "error": str(e),
                "answer": f"An error occurred while processing your question: {str(e)}",
                "question": question,
                "chunks_found": 0
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get pipeline statistics
        
        Returns:
            Dict: Pipeline statistics
        """
        if not self.is_initialized:
            return {"error": "Pipeline not initialized"}
        
        stats = {
            "initialized": self.is_initialized,
            "openai_available": self.openai_client is not None,
            "vector_store_stats": self.vector_store.get_stats() if self.vector_store else {}
        }
        
        return stats
    
    def close(self):
        """
        Close pipeline and cleanup resources
        """
        if self.vector_store:
            self.vector_store.close()
        
        logger.info("RAG pipeline closed")


def main():
    """
    Test the complete RAG pipeline
    """
    try:
        # Initialize pipeline
        logger.info("Testing RAG Pipeline...")
        rag = RAGPipeline(prefer_weaviate=False)  # Use mock store for testing
        
        if not rag.initialize():
            logger.error("Failed to initialize RAG pipeline")
            return
        
        # Load the Attention paper
        pdf_path = "/Users/sam/Desktop/rag/AttentionAllYouNeed.pdf"
        if not rag.load_document(pdf_path):
            logger.error("Failed to load document")
            return
        
        # Get pipeline stats
        stats = rag.get_stats()
        print("\\n=== RAG Pipeline Stats ===")
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        # Test queries
        test_queries = [
            "What is the Transformer architecture?",
            "How does multi-head attention work?",
            "What are the key innovations in this paper?",
            "How does the attention mechanism calculate attention weights?",
            "What are the advantages of the Transformer over RNNs?"
        ]
        
        print("\\n=== Testing Queries ===")
        for i, query in enumerate(test_queries, 1):
            print(f"\\n--- Query {i}: {query} ---")
            
            result = rag.query(query, num_chunks=3, min_score=0.1)
            
            print(f"Chunks found: {result.get('chunks_found', 0)}")
            if result.get('sources'):
                print("Sources:")
                for source in result['sources']:
                    print(f"  - {source['chunk_id']} (Section: {source['section']}, Score: {source['score']:.3f})")
            
            answer = result.get('answer', 'No answer generated')
            print(f"Answer: {answer[:200]}...")
            
            if 'error' in result:
                print(f"Error: {result['error']}")
        
        print("\\n✅ RAG pipeline test completed!")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise
    
    finally:
        if 'rag' in locals():
            rag.close()


if __name__ == "__main__":
    main()
