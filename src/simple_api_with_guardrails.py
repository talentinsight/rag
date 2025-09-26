from typing import Tuple, List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import os
import json
import re
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from collections import defaultdict

# Security
security = HTTPBearer()
BEARER_TOKEN = "142c5738204c9ae01e39084e177a5bf67ade8578f79336f28459796fd5e9d6a0"

# Simple Guardrails Implementation
class GuardrailResult:
    def __init__(self, passed: bool, score: float, reason: str, category: str):
        self.passed = passed
        self.score = score
        self.reason = reason
        self.category = category

class SimpleGuardrails:
    def __init__(self):
        self.request_counts = defaultdict(list)
        self.stats = {
            "total_requests": 0,
            "blocked_requests": 0,
            "categories": defaultdict(int)
        }
    
    def check_rate_limit(self, client_id: str, max_requests: int = 10, window_minutes: int = 1) -> GuardrailResult:
        """Simple rate limiting"""
        now = datetime.now()
        window_start = now - timedelta(minutes=window_minutes)
        
        # Clean old requests
        self.request_counts[client_id] = [
            req_time for req_time in self.request_counts[client_id] 
            if req_time > window_start
        ]
        
        current_count = len(self.request_counts[client_id])
        
        if current_count >= max_requests:
            return GuardrailResult(
                passed=False,
                score=1.0,
                reason=f"Rate limit exceeded: {current_count}/{max_requests} requests",
                category="rate_limiting"
            )
        
        self.request_counts[client_id].append(now)
        return GuardrailResult(
            passed=True,
            score=current_count / max_requests,
            reason=f"Rate limit OK: {current_count + 1}/{max_requests}",
            category="rate_limiting"
        )
    
    def check_prompt_injection(self, text: str) -> GuardrailResult:
        """Check for prompt injection"""
        injection_patterns = [
            r"ignore\s+(all\s+)?previous\s+instructions?",
            r"forget\s+(all\s+)?previous\s+instructions?",
            r"show\s+me\s+your\s+system\s+prompt",
            r"act\s+as\s+if\s+you\s+are",
            r"pretend\s+you\s+are",
            r"ignore\s+the\s+attention\s+paper",
            r"system\s*:",
            r"admin\s*:",
        ]
        
        matches = []
        for pattern in injection_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                matches.append(pattern)
        
        if matches:
            return GuardrailResult(
                passed=False,
                score=0.9,
                reason=f"Prompt injection detected: {len(matches)} patterns",
                category="prompt_injection"
            )
        
        return GuardrailResult(
            passed=True,
            score=0.0,
            reason="No prompt injection detected",
            category="prompt_injection"
        )
    
    def check_toxicity(self, text: str) -> GuardrailResult:
        """Basic toxicity check"""
        toxic_patterns = [
            r"\b(fuck|shit|damn|bitch|ass)\b",
            r"\b(idiot|stupid|moron)\b.*\b(author|researcher|paper)\b",
            r"\b(garbage|trash|worthless)\b.*\b(research|paper)\b",
        ]
        
        matches = []
        for pattern in toxic_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                matches.append(pattern)
        
        if matches:
            return GuardrailResult(
                passed=False,
                score=0.8,
                reason=f"Toxic content detected: {len(matches)} patterns",
                category="toxicity"
            )
        
        return GuardrailResult(
            passed=True,
            score=0.0,
            reason="No toxic content detected",
            category="toxicity"
        )
    
    def check_pii(self, text: str) -> GuardrailResult:
        """Check for PII"""
        pii_patterns = [
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
            r"\b\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b",  # Phone
            r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",  # Credit card
        ]
        
        pii_found = []
        for pattern in pii_patterns:
            matches = re.findall(pattern, text)
            pii_found.extend(matches)
        
        if pii_found:
            return GuardrailResult(
                passed=False,
                score=1.0,
                reason=f"PII detected: {len(pii_found)} instances",
                category="pii_detection"
            )
        
        return GuardrailResult(
            passed=True,
            score=0.0,
            reason="No PII detected",
            category="pii_detection"
        )
    
    def check_all(self, text: str, client_id: str = "default") -> Tuple[bool, List[GuardrailResult]]:
        """Run all guardrail checks"""
        results = []
        
        # Rate limiting
        rate_result = self.check_rate_limit(client_id)
        results.append(rate_result)
        
        if not rate_result.passed:
            self.stats["blocked_requests"] += 1
            return False, results
        
        # Other checks
        results.append(self.check_prompt_injection(text))
        results.append(self.check_toxicity(text))
        results.append(self.check_pii(text))
        
        # Update stats
        self.stats["total_requests"] += 1
        for result in results:
            self.stats["categories"][result.category] += 1
            if not result.passed:
                self.stats["blocked_requests"] += 1
        
        all_passed = all(r.passed for r in results)
        return all_passed, results

# Initialize guardrails
guardrails = SimpleGuardrails()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != BEARER_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return True

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    question: str
    timestamp: str
    processing_time_ms: float = Field(default=0.0)
    guardrail_results: List[Dict[str, Any]] = Field(default=[])

app = FastAPI(
    title="RAG API - Attention Paper (With Guardrails)",
    description="RAG API for the Attention Is All You Need paper with safety guardrails",
    version="1.0.0"
)

@app.get("/")
async def root():
    return {
        "message": "RAG API for Attention Paper (With Guardrails)",
        "version": "1.0.0",
        "auth_required": True,
        "guardrails_enabled": True,
        "endpoints": ["/ping", "/health", "/query", "/guardrail-stats"]
    }

@app.get("/ping")
async def ping():
    return {
        "message": "pong",
        "timestamp": datetime.now().isoformat(),
        "status": "ok"
    }

@app.get("/health")
async def health(_: bool = Depends(verify_token)):
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "auth": "enabled",
        "guardrails": "enabled",
        "paper": "Attention Is All You Need"
    }

@app.get("/guardrail-stats")
async def get_guardrail_stats(_: bool = Depends(verify_token)):
    """Get guardrail statistics"""
    return {
        "stats": guardrails.stats,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    _: bool = Depends(verify_token)
):
    start_time = datetime.now()
    
    # Run guardrail checks
    client_id = "authenticated_user"
    passed, results = guardrails.check_all(request.question, client_id)
    
    if not passed:
        # Block the request
        failed_checks = [r for r in results if not r.passed]
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Request blocked by safety guardrails",
                "reasons": [r.reason for r in failed_checks],
                "categories": [r.category for r in failed_checks]
            }
        )
    
    # Simulate processing time (like a real RAG system)
    import asyncio
    await asyncio.sleep(0.5)  # Simulate some processing time
    
    # Generate response
    answer = f"Based on the 'Attention Is All You Need' paper, regarding your question about '{request.question}': The Transformer architecture introduced in this paper revolutionized sequence-to-sequence modeling by relying entirely on attention mechanisms, eliminating the need for recurrence and convolutions. The key innovation is the multi-head self-attention mechanism that allows the model to attend to different positions of the input sequence simultaneously."
    
    # Calculate processing time
    processing_time = (datetime.now() - start_time).total_seconds() * 1000
    
    # Format guardrail results
    guardrail_data = [
        {
            "category": r.category,
            "passed": r.passed,
            "score": r.score,
            "reason": r.reason
        }
        for r in results
    ]
    
    return QueryResponse(
        answer=answer,
        question=request.question,
        timestamp=datetime.now().isoformat(),
        processing_time_ms=processing_time,
        guardrail_results=guardrail_data
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
