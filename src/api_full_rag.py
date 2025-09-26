import os
import logging
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG API - Attention Is All You Need",
    description="Retrieval-Augmented Generation API for the Attention paper",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()
BEARER_TOKEN = os.getenv("BEARER_TOKEN", "rag-secure-token-2024")
logger.info(f"Loaded BEARER_TOKEN: {BEARER_TOKEN}")

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    logger.info(f"Received token: {credentials.credentials}")
    logger.info(f"Expected token: {BEARER_TOKEN}")
    if credentials.credentials != BEARER_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials.credentials

# Request/Response models
class QueryRequest(BaseModel):
    query: str
    max_tokens: Optional[int] = 150
    temperature: Optional[float] = 0.7

class QueryResponse(BaseModel):
    response: str
    sources: List[str]
    metadata: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    version: str
    components: Dict[str, str]

# Mock RAG implementation
class MockRAGPipeline:
    def __init__(self):
        self.chunks = [
            "The Transformer model architecture is based entirely on attention mechanisms.",
            "Multi-head attention allows the model to attend to different representation subspaces.",
            "The encoder-decoder structure uses self-attention and cross-attention layers.",
            "Positional encoding is added to input embeddings to provide sequence information.",
            "The model achieves state-of-the-art results on machine translation tasks."
        ]
    
    def query(self, query: str, max_tokens: int = 150) -> Dict[str, Any]:
        relevant_chunks = [chunk for chunk in self.chunks if any(word.lower() in chunk.lower() for word in query.split())]
        if not relevant_chunks:
            relevant_chunks = self.chunks[:2]
        
        response = f"Based on the Attention Is All You Need paper: {' '.join(relevant_chunks[:2])}"
        
        return {
            "response": response,
            "sources": relevant_chunks,
            "metadata": {
                "model": "mock-rag-v1",
                "chunks_used": len(relevant_chunks),
                "query_length": len(query)
            }
        }

# Initialize RAG pipeline
rag_pipeline = MockRAGPipeline()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        components={
            "rag_pipeline": "active",
            "vector_store": "mock",
            "llm": "openai-ready",
            "bearer_token": BEARER_TOKEN[:10] + "..."
        }
    )

@app.post("/query", response_model=QueryResponse, dependencies=[Depends(verify_token)])
async def query_rag(request: QueryRequest):
    try:
        logger.info(f"Processing query: {request.query}")
        result = rag_pipeline.query(query=request.query, max_tokens=request.max_tokens)
        return QueryResponse(response=result["response"], sources=result["sources"], metadata=result["metadata"])
    except Exception as e:
        logger.error(f"Query processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.get("/chunks", dependencies=[Depends(verify_token)])
async def list_chunks():
    return {"chunks": rag_pipeline.chunks, "total_count": len(rag_pipeline.chunks)}

@app.get("/stats", dependencies=[Depends(verify_token)])
async def get_stats():
    return {"total_chunks": len(rag_pipeline.chunks), "model_info": "Attention Is All You Need - Transformer Architecture", "status": "operational"}

if __name__ == "__main__":
    uvicorn.run("api_full_rag:app", host="0.0.0.0", port=8000, reload=False)
