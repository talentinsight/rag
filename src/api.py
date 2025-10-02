"""
FastAPI Application for RAG System
Provides REST API endpoints for the Attention paper RAG system
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import os
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
from dotenv import load_dotenv

from .rag_pipeline import RAGPipeline

# MCP WebSocket server import - conditional for deployment
try:
    from .mcp_websocket_server import WebSocketMCPServer
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    logger.warning("MCP WebSocket server not available")

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global RAG pipeline instance
rag_pipeline: Optional[RAGPipeline] = None


class QueryRequest(BaseModel):
    """Request model for queries"""
    question: str = Field(..., description="The question to ask about the Attention paper")
    num_chunks: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")
    min_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum similarity score")


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


class DirectLLMRequest(BaseModel):
    """Request model for direct LLM queries (without RAG)"""
    question: str = Field(..., description="The question to ask directly to the LLM")
    model: str = Field(default="gpt-3.5-turbo", description="OpenAI model to use")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Temperature for response generation")
    max_tokens: int = Field(default=500, ge=1, le=4000, description="Maximum tokens in response")


class DirectLLMResponse(BaseModel):
    """Response model for direct LLM queries"""
    answer: str = Field(..., description="Generated answer from LLM")
    question: str = Field(..., description="Original question")
    model: str = Field(..., description="Model used for generation")
    total_tokens: Optional[int] = Field(None, description="Total tokens used")
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


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
    logger.info("Starting RAG API server...")
    
    try:
        # Initialize RAG pipeline
        rag_pipeline = RAGPipeline(
            prefer_weaviate=True,  # Use Weaviate for production
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        if await asyncio.get_event_loop().run_in_executor(None, rag_pipeline.initialize):
            logger.info("✅ RAG pipeline initialized")
            
            # Load the Attention paper
            pdf_path = os.getenv("PDF_PATH", "./AttentionAllYouNeed.pdf")
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
    logger.info("Shutting down RAG API server...")
    if rag_pipeline:
        rag_pipeline.close()


# Create FastAPI app
app = FastAPI(
    title="Attention Paper RAG API",
    description="REST API for querying the 'Attention Is All You Need' paper using RAG",
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
    """Root endpoint"""
    return {
        "message": "Attention Paper RAG API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check(pipeline: RAGPipeline = Depends(get_rag_pipeline)):
    """Health check endpoint"""
    try:
        stats = pipeline.get_stats()
        
        return HealthResponse(
            status="healthy" if stats.get("initialized", False) else "unhealthy",
            pipeline_initialized=stats.get("initialized", False),
            openai_available=stats.get("openai_available", False),
            vector_store_stats=stats.get("vector_store_stats", {})
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            pipeline_initialized=False,
            openai_available=False,
            vector_store_stats={}
        )


@app.get("/stats", response_model=StatsResponse)
async def get_stats(pipeline: RAGPipeline = Depends(get_rag_pipeline)):
    """Get system statistics"""
    try:
        pipeline_stats = pipeline.get_stats()
        
        system_info = {
            "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            "environment": os.getenv("ENVIRONMENT", "development"),
            "host": os.getenv("HOST", "localhost"),
            "port": os.getenv("PORT", "8000")
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
    pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    """Query the RAG system"""
    start_time = datetime.now()
    
    try:
        logger.info(f"Processing query: '{request.question[:50]}...'")
        
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
        
        logger.info(f"Query processed in {processing_time:.1f}ms")
        return response
        
    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@app.get("/chunks/{chunk_id}")
async def get_chunk(
    chunk_id: str,
    pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    """Get a specific chunk by ID"""
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


@app.post("/direct-llm", response_model=DirectLLMResponse)
async def direct_llm_query(request: DirectLLMRequest):
    """
    Query the LLM directly without RAG (for guardrails testing)
    """
    start_time = datetime.now()
    
    try:
        logger.info(f"Processing direct LLM query: '{request.question[:50]}...'")
        
        # Import OpenAI client
        from .openai_client import OpenAIClient
        
        # Initialize OpenAI client
        openai_client = OpenAIClient()
        
        # Make direct call to OpenAI
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            openai_client.generate_direct_response,
            request.question,
            request.model,
            request.temperature,
            request.max_tokens
        )
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return DirectLLMResponse(
            answer=response.get("answer", ""),
            question=request.question,
            model=request.model,
            total_tokens=response.get("total_tokens"),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Direct LLM query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Direct LLM query failed: {str(e)}")


@app.post("/search")
async def search_chunks(
    query: str,
    limit: int = 5,
    min_score: float = 0.1,
    pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    """Search for similar chunks without generating a response"""
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


# WebSocket MCP endpoint
@app.websocket("/mcp")
async def websocket_mcp_endpoint(websocket: WebSocket):
    """WebSocket endpoint for MCP (Model Context Protocol)"""
    
    # Check for bearer token in query parameters or headers
    token = None
    
    # Try to get token from query parameters
    if "token" in websocket.query_params:
        token = websocket.query_params["token"]
    
    # Try to get token from headers (if available)
    elif "authorization" in websocket.headers:
        auth_header = websocket.headers["authorization"]
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
    
    # Validate token
    expected_token = os.getenv("BEARER_TOKEN", "142c5738204c9ae01e39084e177a5bf67ade8578f79336f28459796fd5e9d6a0")
    
    if not token or token != expected_token:
        logger.warning(f"MCP WebSocket authentication failed from {websocket.client}")
        await websocket.close(code=4001, reason="Authentication required")
        return
    
    await websocket.accept()
    logger.info(f"MCP WebSocket client authenticated and connected: {websocket.client}")
    
    try:
        # Import and initialize WebSocket MCP server
        if not MCP_AVAILABLE:
            await websocket.send_text(json.dumps({"error": "MCP server not available"}))
            return
        
        # Create MCP server instance
        mcp_server = WebSocketMCPServer()
        await mcp_server.initialize_rag()
        
        while True:
            # Receive message from client
            message = await websocket.receive_text()
            
            try:
                # Parse and handle MCP message
                import json
                data = json.loads(message)
                logger.info(f"Received MCP message: {data.get('method', 'unknown')}")
                
                # Handle MCP protocol
                response = await mcp_server.handle_mcp_message(data)
                
                # Send response
                if response:
                    await websocket.send_text(json.dumps(response))
                    
            except json.JSONDecodeError:
                error_response = {
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32700,
                        "message": "Parse error"
                    }
                }
                await websocket.send_text(json.dumps(error_response))
                
            except Exception as e:
                logger.error(f"Error handling MCP message: {str(e)}")
                error_response = {
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32603,
                        "message": f"Internal error: {str(e)}"
                    }
                }
                await websocket.send_text(json.dumps(error_response))
                
    except WebSocketDisconnect:
        logger.info(f"MCP WebSocket client disconnected: {websocket.client}")
    except Exception as e:
        logger.error(f"WebSocket MCP error: {str(e)}")


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
    debug = os.getenv("DEBUG", "true").lower() == "true"
    
    logger.info(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info"
    )


if __name__ == "__main__":
    main()
