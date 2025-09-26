"""
Secure FastAPI Application for RAG System with Bearer Token Authentication
Provides REST API endpoints for the Attention paper RAG system with HTTPS and auth
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import os
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
from dotenv import load_dotenv

from rag_pipeline import RAGPipeline

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()
BEARER_TOKEN = os.getenv("BEARER_TOKEN")

if not BEARER_TOKEN:
    logger.warning("BEARER_TOKEN not set! API will be unsecured!")

# Global RAG pipeline instance
rag_pipeline: Optional[RAGPipeline] = None


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify bearer token"""
    if not BEARER_TOKEN:
        # If no token is configured, allow access (for development)
        return True
    
    if credentials.credentials != BEARER_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return True


class QueryRequest(BaseModel):
    """Request model for queries"""
    question: str = Field(..., description="The question to ask about the Attention paper")
    num_chunks: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")
    min_score: float = Field(default=0.1, ge=0.0, le=1.0, description="Minimum similarity score")


class QueryResponse(BaseModel):
    """Response model for queries"""
    answer: str = Field(..., description="Generated answer")
    question: str = Field(..., description="Original question")
    chunks_found: int = Field(..., description="Number of relevant chunks found")
    sources: List[Dict[str, Any]] = Field(default=[], description="Source chunks used")
    model: Optional[str] = Field(None, description="Model used for generation")
    total_tokens: Optional[int] = Field(None, description="Total tokens used")
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    pipeline_initialized: bool = Field(..., description="Whether RAG pipeline is initialized")
    openai_available: bool = Field(..., description="Whether OpenAI is available")
    vector_store_stats: Dict[str, Any] = Field(default={}, description="Vector store statistics")
    environment: str = Field(..., description="Environment (production/development)")
    auth_enabled: bool = Field(..., description="Whether authentication is enabled")


class StatsResponse(BaseModel):
    """Statistics response"""
    pipeline_stats: Dict[str, Any] = Field(..., description="Pipeline statistics")
    system_info: Dict[str, Any] = Field(..., description="System information")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global rag_pipeline
    
    # Startup
    logger.info("Starting Secure RAG API server...")
    
    try:
        # Initialize RAG pipeline
        rag_pipeline = RAGPipeline(
            prefer_weaviate=True,  # Try Weaviate first, fallback to mock
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        if await asyncio.get_event_loop().run_in_executor(None, rag_pipeline.initialize):
            logger.info("✅ RAG pipeline initialized")
            
            # Load the Attention paper
            pdf_path = os.getenv("PDF_PATH", "/app/AttentionAllYouNeed.pdf")
            if os.path.exists(pdf_path):
                if await asyncio.get_event_loop().run_in_executor(None, rag_pipeline.load_document, pdf_path):
                    logger.info("✅ Document loaded successfully")
                else:
                    logger.warning("Failed to load document")
            else:
                logger.warning(f"PDF file not found: {pdf_path}")
        else:
            logger.error("Failed to initialize RAG pipeline")
            
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Secure RAG API server...")
    if rag_pipeline:
        rag_pipeline.close()


# Create FastAPI app
app = FastAPI(
    title="Secure Attention Paper RAG API",
    description="Secure REST API for querying the 'Attention Is All You Need' paper using RAG with Bearer Token authentication",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_rag_pipeline() -> RAGPipeline:
    """Dependency to get RAG pipeline"""
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    return rag_pipeline


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint - public access"""
    return {
        "message": "Secure Attention Paper RAG API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "auth_required": bool(BEARER_TOKEN),
        "usage": "Include 'Authorization: Bearer <token>' header for authenticated endpoints"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check(
    _: bool = Depends(verify_token),
    pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    """Health check endpoint - requires authentication"""
    try:
        stats = pipeline.get_stats()
        
        return HealthResponse(
            status="healthy" if stats.get("initialized", False) else "unhealthy",
            pipeline_initialized=stats.get("initialized", False),
            openai_available=stats.get("openai_available", False),
            vector_store_stats=stats.get("vector_store_stats", {}),
            environment=os.getenv("ENVIRONMENT", "development"),
            auth_enabled=bool(BEARER_TOKEN)
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            pipeline_initialized=False,
            openai_available=False,
            vector_store_stats={},
            environment=os.getenv("ENVIRONMENT", "development"),
            auth_enabled=bool(BEARER_TOKEN)
        )


@app.get("/stats", response_model=StatsResponse)
async def get_stats(
    _: bool = Depends(verify_token),
    pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    """Get system statistics - requires authentication"""
    try:
        pipeline_stats = pipeline.get_stats()
        
        system_info = {
            "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            "environment": os.getenv("ENVIRONMENT", "development"),
            "host": os.getenv("HOST", "localhost"),
            "port": os.getenv("PORT", "8000"),
            "auth_enabled": bool(BEARER_TOKEN),
            "weaviate_url": os.getenv("WEAVIATE_URL", "http://localhost:8080")
        }
        
        return StatsResponse(
            pipeline_stats=pipeline_stats,
            system_info=system_info
        )
    except Exception as e:
        logger.error(f"Stats request failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_rag(
    request: QueryRequest,
    _: bool = Depends(verify_token),
    pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    """Query the RAG system - requires authentication"""
    start_time = datetime.now()
    
    try:
        logger.info(f"Processing authenticated query: '{request.question[:50]}...'")
        
        # Process query
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            pipeline.query,
            request.question,
            request.num_chunks,
            request.min_score
        )
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Build response
        response = QueryResponse(
            answer=result.get("answer", "No answer generated"),
            question=result.get("question", request.question),
            chunks_found=result.get("chunks_found", 0),
            sources=result.get("sources", []),
            model=result.get("model"),
            total_tokens=result.get("total_tokens"),
            processing_time_ms=processing_time
        )
        
        logger.info(f"Authenticated query processed in {processing_time:.1f}ms")
        return response
        
    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@app.get("/chunks/{chunk_id}")
async def get_chunk(
    chunk_id: str,
    _: bool = Depends(verify_token),
    pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    """Get a specific chunk by ID - requires authentication"""
    try:
        chunk = await asyncio.get_event_loop().run_in_executor(
            None,
            pipeline.vector_store.get_chunk_by_id,
            chunk_id
        )
        
        if chunk is None:
            raise HTTPException(status_code=404, detail=f"Chunk {chunk_id} not found")
        
        return chunk
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get chunk {chunk_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get chunk: {str(e)}")


@app.post("/search")
async def search_chunks(
    query: str,
    limit: int = 5,
    min_score: float = 0.1,
    _: bool = Depends(verify_token),
    pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    """Search for similar chunks without generating a response - requires authentication"""
    try:
        chunks = await asyncio.get_event_loop().run_in_executor(
            None,
            pipeline.vector_store.search_similar,
            query,
            limit,
            min_score
        )
        
        return {
            "query": query,
            "chunks_found": len(chunks),
            "chunks": chunks,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


# Public endpoint for testing connectivity (no auth required)
@app.get("/ping")
async def ping():
    """Simple ping endpoint - no authentication required"""
    return {
        "message": "pong",
        "timestamp": datetime.now().isoformat(),
        "auth_required": bool(BEARER_TOKEN)
    }


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )


def main():
    """Run the FastAPI server"""
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    
    logger.info(f"Starting secure server on {host}:{port}")
    logger.info(f"Authentication: {'Enabled' if BEARER_TOKEN else 'Disabled'}")
    
    uvicorn.run(
        "api_secure:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info"
    )


if __name__ == "__main__":
    main()
