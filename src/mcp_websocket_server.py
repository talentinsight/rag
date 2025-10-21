"""
WebSocket MCP Server for AWS Deployment
Provides MCP (Model Context Protocol) over WebSocket for cloud deployment
"""

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional
import websockets
from websockets.server import WebSocketServerProtocol
from datetime import datetime

from .rag_pipeline import RAGPipeline
from .comprehensive_guardrails import ComprehensiveGuardrails

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebSocketMCPServer:
    """WebSocket-based MCP Server for cloud deployment"""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8001):
        self.host = host
        self.port = port
        self.rag_pipeline: Optional[RAGPipeline] = None
        self.guardrails = ComprehensiveGuardrails()
        self.clients: set = set()
        self.conversation_history = []
        self.session_stats = {
            "queries_count": 0,
            "total_chunks_retrieved": 0,
            "total_tokens_used": 0,
            "session_start": datetime.now().isoformat()
        }
        
    async def initialize_rag(self):
        """Initialize the RAG pipeline"""
        try:
            logger.info("Initializing RAG pipeline for WebSocket MCP server...")
            
            self.rag_pipeline = RAGPipeline(
                prefer_weaviate=False,  # Use mock store for reliability
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
            
            if self.rag_pipeline.initialize():
                logger.info("✅ RAG pipeline initialized")
                
                # Try to load document if available
                pdf_path = os.getenv("PDF_PATH", "./AttentionAllYouNeed.pdf")
                if os.path.exists(pdf_path):
                    if self.rag_pipeline.load_document(pdf_path):
                        logger.info("✅ Document loaded successfully")
                    else:
                        logger.warning("Failed to load document")
                else:
                    logger.warning(f"PDF file not found: {pdf_path}")
                    
                return True
            else:
                logger.error("Failed to initialize RAG pipeline")
                return False
                
        except Exception as e:
            logger.error(f"RAG initialization failed: {str(e)}")
            return False
    
    async def handle_client(self, websocket: WebSocketServerProtocol, path: str):
        """Handle WebSocket client connection"""
        client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        
        # Check for authentication in the first message or connection headers
        try:
            # Wait for first message which should contain auth
            first_message = await asyncio.wait_for(websocket.recv(), timeout=10.0)
            data = json.loads(first_message)
            
            # Check if it's an auth message or has token
            token = None
            if data.get("method") == "authenticate":
                token = data.get("params", {}).get("token")
            elif "token" in data:
                token = data["token"]
            
            # Validate token
            expected_token = os.getenv("BEARER_TOKEN", "142c5738204c9ae01e39084e177a5bf67ade8578f79336f28459796fd5e9d6a0")
            
            if not token or token != expected_token:
                logger.warning(f"MCP WebSocket authentication failed from {client_id}")
                await websocket.close(code=4001, reason="Authentication required")
                return
            
            # Send auth success response
            auth_response = {
                "jsonrpc": "2.0",
                "id": data.get("id"),
                "result": {"authenticated": True}
            }
            await websocket.send(json.dumps(auth_response))
            
        except (asyncio.TimeoutError, json.JSONDecodeError, KeyError):
            logger.warning(f"MCP WebSocket authentication timeout or invalid format from {client_id}")
            await websocket.close(code=4001, reason="Authentication required")
            return
        
        logger.info(f"MCP client authenticated and connected: {client_id}")
        self.clients.add(websocket)
        
        try:
            async for message in websocket:
                try:
                    # Parse MCP message
                    data = json.loads(message)
                    logger.info(f"Received MCP message: {data.get('method', 'unknown')}")
                    
                    # Handle MCP protocol
                    response = await self.handle_mcp_message(data)
                    
                    # Send response
                    if response:
                        await websocket.send(json.dumps(response))
                        
                except json.JSONDecodeError:
                    error_response = {
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32700,
                            "message": "Parse error"
                        }
                    }
                    await websocket.send(json.dumps(error_response))
                    
                except Exception as e:
                    logger.error(f"Error handling message: {str(e)}")
                    error_response = {
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32603,
                            "message": f"Internal error: {str(e)}"
                        }
                    }
                    await websocket.send(json.dumps(error_response))
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"MCP client disconnected: {client_id}")
        finally:
            self.clients.discard(websocket)
    
    async def handle_mcp_message(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle MCP protocol messages"""
        
        method = data.get("method")
        message_id = data.get("id")
        params = data.get("params", {})
        
        try:
            if method == "initialize":
                return await self.handle_initialize(message_id, params)
            elif method == "tools/list":
                return await self.handle_list_tools(message_id)
            elif method == "tools/call":
                return await self.handle_call_tool(message_id, params)
            elif method == "resources/list":
                return await self.handle_list_resources(message_id)
            elif method == "resources/read":
                return await self.handle_read_resource(message_id, params)
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": message_id,
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {method}"
                    }
                }
                
        except Exception as e:
            logger.error(f"Error in handle_mcp_message: {str(e)}")
            return {
                "jsonrpc": "2.0",
                "id": message_id,
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {str(e)}"
                }
            }
    
    async def handle_initialize(self, message_id: int, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP initialize request"""
        return {
            "jsonrpc": "2.0",
            "id": message_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {},
                    "resources": {}
                },
                "serverInfo": {
                    "name": "rag-attention-paper-websocket",
                    "version": "1.0.0"
                }
            }
        }
    
    async def handle_list_tools(self, message_id: int) -> Dict[str, Any]:
        """Handle tools/list request"""
        tools = [
            {
                "name": "query_attention_paper",
                "description": "Query the 'Attention Is All You Need' paper using RAG",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The question to ask about the Attention paper"
                        },
                        "num_chunks": {
                            "type": "integer",
                            "description": "Number of relevant chunks to retrieve (1-20)",
                            "minimum": 1,
                            "maximum": 20,
                            "default": 5
                        },
                        "min_score": {
                            "type": "number",
                            "description": "Minimum similarity score for chunks (0.0-1.0)",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "default": 0.1
                        }
                    },
                    "required": ["question"]
                }
            },
            {
                "name": "search_paper_chunks",
                "description": "Search for specific chunks in the Attention paper",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query to find relevant chunks"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of chunks to return",
                            "minimum": 1,
                            "maximum": 20,
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "get_rag_stats",
                "description": "Get statistics and information about the RAG system",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "analyze_query_complexity",
                "description": "Analyze the complexity and requirements of a query before processing",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The question to analyze"
                        }
                    },
                    "required": ["question"]
                }
            },
            {
                "name": "get_chunk_details",
                "description": "Get detailed information about a specific chunk",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "chunk_id": {
                            "type": "string",
                            "description": "The ID of the chunk to retrieve"
                        }
                    },
                    "required": ["chunk_id"]
                }
            },
            {
                "name": "compare_chunks",
                "description": "Compare similarity between multiple chunks",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "chunk_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of chunk IDs to compare"
                        }
                    },
                    "required": ["chunk_ids"]
                }
            },
            {
                "name": "get_conversation_history",
                "description": "Get the history of queries and responses in this session",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of entries to return",
                            "default": 10
                        }
                    },
                    "required": []
                }
            },
            {
                "name": "mask_pii_text",
                "description": "Mask personally identifiable information (PII) in text. Detects and masks emails, phone numbers, credit cards, SSNs, and API keys.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The text to analyze and mask for PII"
                        }
                    },
                    "required": ["text"]
                }
            },
            {
                "name": "query_with_pii_masking",
                "description": "Query the paper with automatic PII masking. Returns both the answer and the PII-masked version of the input for evaluation.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The question to ask about the Attention paper"
                        },
                        "num_chunks": {
                            "type": "integer",
                            "description": "Number of relevant chunks to retrieve (1-20)",
                            "minimum": 1,
                            "maximum": 20,
                            "default": 5
                        },
                        "min_score": {
                            "type": "number",
                            "description": "Minimum similarity score for chunks (0.0-1.0)",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "default": 0.1
                        }
                    },
                    "required": ["question"]
                }
            }
        ]
        
        return {
            "jsonrpc": "2.0",
            "id": message_id,
            "result": {
                "tools": tools
            }
        }
    
    async def handle_call_tool(self, message_id: int, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/call request"""
        
        if not self.rag_pipeline or not self.rag_pipeline.is_initialized:
            return {
                "jsonrpc": "2.0",
                "id": message_id,
                "error": {
                    "code": -32603,
                    "message": "RAG pipeline is not initialized"
                }
            }
        
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        try:
            if tool_name == "query_attention_paper":
                result = await self.handle_query_paper(arguments)
            elif tool_name == "search_paper_chunks":
                result = await self.handle_search_chunks(arguments)
            elif tool_name == "get_rag_stats":
                result = await self.handle_get_stats(arguments)
            elif tool_name == "analyze_query_complexity":
                result = await self.handle_analyze_query_complexity(arguments)
            elif tool_name == "get_chunk_details":
                result = await self.handle_get_chunk_details(arguments)
            elif tool_name == "compare_chunks":
                result = await self.handle_compare_chunks(arguments)
            elif tool_name == "get_conversation_history":
                result = await self.handle_get_conversation_history(arguments)
            elif tool_name == "mask_pii_text":
                result = await self.handle_mask_pii(arguments)
            elif tool_name == "query_with_pii_masking":
                result = await self.handle_query_with_pii_masking(arguments)
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": message_id,
                    "error": {
                        "code": -32601,
                        "message": f"Unknown tool: {tool_name}"
                    }
                }
            
            return {
                "jsonrpc": "2.0",
                "id": message_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": result
                        }
                    ]
                }
            }
            
        except Exception as e:
            logger.error(f"Tool call failed: {str(e)}")
            return {
                "jsonrpc": "2.0",
                "id": message_id,
                "error": {
                    "code": -32603,
                    "message": f"Tool execution failed: {str(e)}"
                }
            }
    
    async def handle_query_paper(self, arguments: Dict[str, Any]) -> str:
        """Handle paper query tool with comprehensive guardrails"""
        question = arguments.get("question", "")
        num_chunks = arguments.get("num_chunks", 5)
        min_score = arguments.get("min_score", 0.1)
        
        if not question:
            return "ERROR: Missing a question to ask about the paper."
        
        logger.info(f"WebSocket MCP Query: {question[:50]}...")
        start_time = datetime.now()
        
        # Generate PII masked input for evaluation
        pii_masked_input = self.guardrails.mask_pii(question)
        
        # Input guardrails check (PII filtering only)
        logger.info("Running MCP input guardrails (PII filtering only)")
        input_passed, input_results = self.guardrails.check_input_guardrails_with_pii_filtering(
            question, 
            "mcp_client"
        )
        
        if not input_passed:
            # Return safe response for MCP
            failed_checks = [r for r in input_results if not r.passed]
            critical_failures = [r for r in failed_checks if r.severity == "critical"]
            
            if critical_failures:
                safe_answer = "BLOCKED: Critical safety violation"
            else:
                safe_answer = "BLOCKED: PII detected in request"
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Format guardrails response for MCP
            guardrails_info = []
            for result in input_results:
                if not result.passed:
                    guardrails_info.append(f"- {result.category}: {result.reason}")
            
            response_parts = [
                f"**Answer:** {safe_answer}",
                f"**PII Masked Input:** {pii_masked_input}",
                f"**Chunks Found:** 0",
                f"**Processing Time:** {processing_time:.1f}ms",
                f"**Guardrails Status:** ❌ Failed",
                "**Failed Checks:**"
            ] + guardrails_info
            
            return "\n".join(response_parts)
        
        # Execute query
        result = self.rag_pipeline.query(question, num_chunks, min_score)
        
        # Prepare response data for output guardrails
        response_data = {
            "answer": result.get("answer", ""),
            "question": question,
            "timestamp": datetime.now().isoformat()
        }
        
        # Output guardrails check
        logger.info("Running MCP output guardrails")
        output_passed, output_results = self.guardrails.check_output_guardrails(
            response_data,
            start_time
        )
        
        if not output_passed:
            # Log output guardrails failures but don't filter response
            failed_output_checks = [r for r in output_results if not r.passed]
            logger.warning(f"MCP output guardrails detected issues: {[r.reason for r in failed_output_checks]}")
            
            # Only filter if PII is detected in output (data leakage prevention)
            if any(r.category == "pii_detection" and not r.passed for r in failed_output_checks):
                result["answer"] = "OUTPUT_FILTERED: PII detected in response"
            # For other issues (adult, profanity, toxicity): just log, don't filter
        
        # Update session stats
        self.session_stats["queries_count"] += 1
        self.session_stats["total_chunks_retrieved"] += result.get("chunks_found", 0)
        self.session_stats["total_tokens_used"] += result.get("total_tokens", 0)
        
        # Add to conversation history
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "type": "query",
            "question": question,
            "answer": result.get("answer", ""),
            "chunks_found": result.get("chunks_found", 0),
            "tokens_used": result.get("total_tokens", 0)
        })
        
        # Calculate processing time and safety score
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Calculate safety score (simplified version of API calculation)
        all_results = input_results + output_results
        total_score = sum(r.score if r.passed else 0.0 for r in all_results)
        total_weight = len(all_results)
        safety_score = total_score / total_weight if total_weight > 0 else 1.0
        
        # Format response with guardrails information
        response_parts = []
        response_parts.append(f"**Answer:** {result.get('answer', 'ERROR: No response')}")
        response_parts.append(f"**PII Masked Input:** {pii_masked_input}")
        response_parts.append(f"**Chunks Found:** {result.get('chunks_found', 0)}")
        
        if result.get('model'):
            response_parts.append(f"**Model:** {result['model']}")
        
        if result.get('total_tokens'):
            response_parts.append(f"**Tokens Used:** {result['total_tokens']}")
        
        # Add guardrails information
        response_parts.append(f"**Processing Time:** {processing_time:.1f}ms")
        response_parts.append(f"**Guardrails Status:** {'✅ Passed' if (input_passed and output_passed) else '⚠️ Filtered'}")
        response_parts.append(f"**Safety Score:** {safety_score:.3f}")
        
        # Sources with full content
        sources = result.get('sources', [])
        if sources:
            response_parts.append("**Sources:**")
            for i, source in enumerate(sources, 1):
                chunk_id = source.get('chunk_id', 'Unknown')
                section = source.get('section', 'Unknown')
                score = source.get('score', 0)
                content = source.get('content', '')
                
                response_parts.append(f"{i}. **{chunk_id}** (Section: {section}, Score: {score:.3f})")
                if content:
                    response_parts.append(f"   Content: {content[:200]}..." if len(content) > 200 else f"   Content: {content}")
        
        # Add guardrails details
        if not (input_passed and output_passed):
            response_parts.append("**Guardrails Details:**")
            for result_check in input_results + output_results:
                if not result_check.passed:
                    response_parts.append(f"- {result_check.category}: {result_check.reason}")
        
        # Add LLM Prompt details if available
        if result.get('llm_prompt'):
            response_parts.append("**LLM Prompt Details:**")
            llm_prompt = result.get('llm_prompt', {})
            if llm_prompt.get('system_prompt'):
                response_parts.append(f"System Prompt: {llm_prompt['system_prompt'][:100]}...")
            if llm_prompt.get('user_prompt'):
                response_parts.append(f"User Prompt: {llm_prompt['user_prompt'][:200]}...")
        
        return "\n".join(response_parts)
    
    async def handle_analyze_query_complexity(self, arguments: Dict[str, Any]) -> str:
        """Analyze query complexity and suggest optimization"""
        question = arguments.get("question", "")
        
        if not question:
            return "ERROR: Missing a question to analyze."
        
        # Analyze query characteristics
        word_count = len(question.split())
        char_count = len(question)
        has_technical_terms = any(term in question.lower() for term in 
                                ['attention', 'transformer', 'neural', 'model', 'architecture', 'mechanism'])
        has_multiple_questions = '?' in question[:-1]  # Multiple question marks
        
        complexity_score = 0
        if word_count > 15: complexity_score += 2
        if char_count > 100: complexity_score += 1
        if has_technical_terms: complexity_score += 2
        if has_multiple_questions: complexity_score += 3
        
        complexity_level = "Low" if complexity_score <= 2 else "Medium" if complexity_score <= 5 else "High"
        
        # Suggest parameters
        suggested_chunks = 3 if complexity_score <= 2 else 5 if complexity_score <= 5 else 8
        suggested_min_score = 0.2 if complexity_score <= 2 else 0.1 if complexity_score <= 5 else 0.05
        
        analysis = [
            f"**Query Analysis:**",
            f"Question: {question}",
            f"Word Count: {word_count}",
            f"Character Count: {char_count}",
            f"Technical Terms: {'Yes' if has_technical_terms else 'No'}",
            f"Multiple Questions: {'Yes' if has_multiple_questions else 'No'}",
            f"",
            f"**Complexity Assessment:**",
            f"Complexity Level: {complexity_level}",
            f"Complexity Score: {complexity_score}/8",
            f"",
            f"**Recommended Parameters:**",
            f"Suggested Chunks: {suggested_chunks}",
            f"Suggested Min Score: {suggested_min_score}",
            f"",
            f"**Optimization Tips:**",
        ]
        
        if complexity_score <= 2:
            analysis.append("- Simple query, fewer chunks should be sufficient")
        elif complexity_score <= 5:
            analysis.append("- Moderate complexity, standard parameters recommended")
        else:
            analysis.append("- Complex query, consider using more chunks and lower threshold")
            analysis.append("- May benefit from breaking into sub-questions")
        
        return "\n".join(analysis)
    
    async def handle_get_chunk_details(self, arguments: Dict[str, Any]) -> str:
        """Get detailed information about a specific chunk"""
        chunk_id = arguments.get("chunk_id", "")
        
        if not chunk_id:
            return "ERROR: Missing a chunk_id."
        
        # Search for the chunk in vector store
        try:
            # This is a simplified version - in real implementation, 
            # you'd query the vector store directly
            search_results = self.rag_pipeline.vector_store_manager.search_similar(
                chunk_id, limit=20, min_score=0.0
            )
            
            target_chunk = None
            for chunk in search_results:
                if chunk.get("chunk_id") == chunk_id:
                    target_chunk = chunk
                    break
            
            if not target_chunk:
                return f"❌ Chunk '{chunk_id}' not found."
            
            details = [
                f"**Chunk Details:**",
                f"ID: {target_chunk.get('chunk_id', 'Unknown')}",
                f"Section: {target_chunk.get('section_title', 'Unknown')}",
                f"Type: {target_chunk.get('chunk_type', 'Unknown')}",
                f"Token Count: {target_chunk.get('token_count', 0)}",
                f"Character Range: {target_chunk.get('start_char', 0)}-{target_chunk.get('end_char', 0)}",
                f"Source File: {target_chunk.get('source_file', 'Unknown')}",
                f"",
                f"**Content:**",
                f"{target_chunk.get('content', 'No content available')}"
            ]
            
            return "\n".join(details)
            
        except Exception as e:
            return f"❌ Error retrieving chunk details: {str(e)}"
    
    async def handle_compare_chunks(self, arguments: Dict[str, Any]) -> str:
        """Compare similarity between multiple chunks"""
        chunk_ids = arguments.get("chunk_ids", [])
        
        if not chunk_ids or len(chunk_ids) < 2:
            return "ERROR: Missing at least 2 chunk IDs to compare."
        
        if len(chunk_ids) > 5:
            return "❌ Maximum 5 chunks can be compared at once."
        
        try:
            # Get chunk details
            chunks = []
            for chunk_id in chunk_ids:
                search_results = self.rag_pipeline.vector_store_manager.search_similar(
                    chunk_id, limit=20, min_score=0.0
                )
                
                target_chunk = None
                for chunk in search_results:
                    if chunk.get("chunk_id") == chunk_id:
                        target_chunk = chunk
                        break
                
                if target_chunk:
                    chunks.append(target_chunk)
                else:
                    return f"❌ Chunk '{chunk_id}' not found."
            
            comparison = [
                f"**Chunk Comparison:**",
                f"Comparing {len(chunks)} chunks:",
                ""
            ]
            
            # Basic comparison metrics
            for i, chunk in enumerate(chunks, 1):
                comparison.extend([
                    f"**Chunk {i}: {chunk.get('chunk_id')}**",
                    f"Section: {chunk.get('section_title', 'Unknown')}",
                    f"Token Count: {chunk.get('token_count', 0)}",
                    f"Content Preview: {chunk.get('content', '')[:100]}...",
                    ""
                ])
            
            # Simple similarity analysis
            comparison.extend([
                "**Similarity Analysis:**",
                "- All chunks are from the same document (Attention Is All You Need)",
                f"- Token count range: {min(c.get('token_count', 0) for c in chunks)} - {max(c.get('token_count', 0) for c in chunks)}",
                f"- Sections covered: {', '.join(set(c.get('section_title', 'Unknown') for c in chunks))}",
            ])
            
            return "\n".join(comparison)
            
        except Exception as e:
            return f"❌ Error comparing chunks: {str(e)}"
    
    async def handle_get_conversation_history(self, arguments: Dict[str, Any]) -> str:
        """Get conversation history for this session"""
        limit = arguments.get("limit", 10)
        
        if not self.conversation_history:
            return "📝 No conversation history available for this session."
        
        # Get recent entries
        recent_history = self.conversation_history[-limit:] if limit > 0 else self.conversation_history
        
        history_parts = [
            f"**Conversation History (Last {len(recent_history)} entries):**",
            f"Session started: {self.session_stats['session_start']}",
            ""
        ]
        
        for i, entry in enumerate(recent_history, 1):
            history_parts.extend([
                f"**Query {i}** ({entry['timestamp']}):",
                f"Question: {entry['question']}",
                f"Answer: {entry['answer'][:150]}..." if len(entry['answer']) > 150 else f"Answer: {entry['answer']}",
                f"Chunks Found: {entry['chunks_found']}",
                f"Tokens Used: {entry['tokens_used']}",
                ""
            ])
        
        history_parts.extend([
            "**Session Statistics:**",
            f"Total Queries: {self.session_stats['queries_count']}",
            f"Total Chunks Retrieved: {self.session_stats['total_chunks_retrieved']}",
            f"Total Tokens Used: {self.session_stats['total_tokens_used']}"
        ])
        
        return "\n".join(history_parts)
    
    async def handle_search_chunks(self, arguments: Dict[str, Any]) -> str:
        """Handle chunk search tool"""
        query = arguments.get("query", "")
        limit = arguments.get("limit", 5)
        
        if not query:
            return "ERROR: Missing a search query."
        
        logger.info(f"WebSocket MCP Search: {query[:50]}...")
        
        # Execute search
        chunks = self.rag_pipeline.vector_store.search_similar(query, limit, 0.1)
        
        if not chunks:
            return f"🔍 No chunks found for query: '{query}'"
        
        # Format results
        response_parts = [f"🔍 **Search Results for:** '{query}'", f"**Found {len(chunks)} chunks:**\n"]
        
        for i, chunk in enumerate(chunks, 1):
            response_parts.append(f"**{i}. {chunk.get('chunk_id', 'Unknown')}**")
            response_parts.append(f"   Score: {chunk.get('score', 0):.3f}")
            response_parts.append(f"   Content: {chunk.get('content', '')[:200]}...")
            response_parts.append("")
        
        return "\n".join(response_parts)
    
    async def handle_get_stats(self, arguments: Dict[str, Any]) -> str:
        """Handle get stats tool"""
        logger.info("WebSocket MCP Get Stats")
        
        stats = self.rag_pipeline.get_stats()
        
        # Format stats
        response_parts = [
            "📊 **RAG System Statistics (WebSocket MCP)**",
            "",
            f"**System Status:** {'✅ Initialized' if stats.get('initialized') else '❌ Not Initialized'}",
            f"**OpenAI Available:** {'✅ Yes' if stats.get('openai_available') else '❌ No'}",
            ""
        ]
        
        vector_stats = stats.get('vector_store_stats', {})
        if vector_stats:
            response_parts.extend([
                "**Vector Store:**",
                f"  - Type: {vector_stats.get('store_type', 'Unknown')}",
                f"  - Total Chunks: {vector_stats.get('total_chunks', 0)}",
                f"  - Vector Dimensions: {vector_stats.get('vector_dimensions', 0)}",
                ""
            ])
        
        response_parts.append(f"**Connected Clients:** {len(self.clients)}")
        response_parts.append(f"**Timestamp:** {datetime.now().isoformat()}")
        
        return "\n".join(response_parts)
    
    async def handle_mask_pii(self, arguments: Dict[str, Any]) -> str:
        """Handle PII masking tool"""
        text = arguments.get("text", "")
        
        if not text:
            return "ERROR: Missing text to analyze for PII."
        
        logger.info(f"WebSocket MCP PII Masking: {text[:50]}...")
        
        # Check for PII and get masked version
        pii_result = self.guardrails.check_pii_detection(text)
        masked_text = self.guardrails.mask_pii(text)
        
        # Format response
        response_parts = [
            "🔒 **PII Analysis and Masking**",
            "",
            f"**Original Text:** {text}",
            f"**Masked Text:** {masked_text}",
            "",
            f"**PII Detection Result:**",
            f"  - Passed: {'✅ No PII detected' if pii_result.passed else '❌ PII detected'}",
            f"  - Score: {pii_result.score:.3f}",
            f"  - Reason: {pii_result.reason}",
            f"  - Severity: {pii_result.severity}"
        ]
        
        if hasattr(pii_result, 'metadata') and pii_result.metadata:
            metadata = pii_result.metadata
            if 'detected_types' in metadata:
                response_parts.append(f"  - Types Found: {', '.join(metadata['detected_types'])}")
            if 'count' in metadata:
                response_parts.append(f"  - Total Instances: {metadata['count']}")
        
        return "\n".join(response_parts)
    
    async def handle_query_with_pii_masking(self, arguments: Dict[str, Any]) -> str:
        """Handle paper query with PII masking and full guardrails"""
        question = arguments.get("question", "")
        num_chunks = arguments.get("num_chunks", 5)
        min_score = arguments.get("min_score", 0.1)
        
        if not question:
            return "ERROR: Missing a question to ask about the paper."
        
        logger.info(f"WebSocket MCP Query with PII Masking: {question[:50]}...")
        
        # Use the main query handler which now includes full guardrails
        # This ensures consistent behavior between all MCP tools
        result = await self.handle_query_paper(arguments)
        
        # Add PII-specific information to the response
        pii_masked_input = self.guardrails.mask_pii(question)
        pii_result = self.guardrails.check_pii_detection(question)
        
        # Prepend PII-specific information
        pii_info = [
            "**PII Masking Analysis:**",
            f"Original Input: {question}",
            f"Masked Input: {pii_masked_input}",
            f"PII Detected: {'Yes' if not pii_result.passed else 'No'}",
            ""
        ]
        
        return "\n".join(pii_info) + result
    
    async def handle_list_resources(self, message_id: int) -> Dict[str, Any]:
        """Handle resources/list request"""
        resources = [
            {
                "uri": "attention://paper/info",
                "name": "Attention Paper Information",
                "description": "Information about the 'Attention Is All You Need' paper",
                "mimeType": "application/json"
            },
            {
                "uri": "attention://system/stats",
                "name": "RAG System Statistics",
                "description": "Current statistics and status of the RAG system",
                "mimeType": "application/json"
            }
        ]
        
        return {
            "jsonrpc": "2.0",
            "id": message_id,
            "result": {
                "resources": resources
            }
        }
    
    async def handle_read_resource(self, message_id: int, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle resources/read request"""
        uri = params.get("uri")
        
        if uri == "attention://paper/info":
            paper_info = {
                "title": "Attention Is All You Need",
                "authors": ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit", "Llion Jones", "Aidan N. Gomez", "Lukasz Kaiser", "Illia Polosukhin"],
                "year": 2017,
                "venue": "NIPS 2017",
                "chunks_available": self.rag_pipeline.get_stats().get("vector_store_stats", {}).get("total_chunks", 0) if self.rag_pipeline else 0,
                "server_type": "WebSocket MCP"
            }
            content = json.dumps(paper_info, indent=2)
            
        elif uri == "attention://system/stats":
            if self.rag_pipeline:
                stats = self.rag_pipeline.get_stats()
                stats["timestamp"] = datetime.now().isoformat()
                stats["server_type"] = "WebSocket MCP"
                stats["connected_clients"] = len(self.clients)
                content = json.dumps(stats, indent=2)
            else:
                content = json.dumps({"error": "RAG pipeline not initialized"}, indent=2)
        else:
            return {
                "jsonrpc": "2.0",
                "id": message_id,
                "error": {
                    "code": -32602,
                    "message": f"Unknown resource URI: {uri}"
                }
            }
        
        return {
            "jsonrpc": "2.0",
            "id": message_id,
            "result": {
                "contents": [
                    {
                        "uri": uri,
                        "mimeType": "application/json",
                        "text": content
                    }
                ]
            }
        }
    
    async def start_server(self):
        """Start the WebSocket MCP server"""
        logger.info(f"🚀 Starting WebSocket MCP Server on {self.host}:{self.port}")
        
        # Initialize RAG pipeline
        if not await self.initialize_rag():
            logger.error("Failed to initialize RAG pipeline")
            return
        
        # Start WebSocket server
        try:
            async with websockets.serve(
                self.handle_client,
                self.host,
                self.port,
                ping_interval=30,
                ping_timeout=10
            ):
                logger.info(f"✅ WebSocket MCP Server running on ws://{self.host}:{self.port}")
                logger.info("📚 Available MCP tools:")
                logger.info("  - query_attention_paper: Ask questions about the paper")
                logger.info("  - search_paper_chunks: Search for specific content")
                logger.info("  - get_rag_stats: Get system statistics")
                logger.info("\n🔄 Server running (Ctrl+C to stop)...")
                
                # Keep server running
                await asyncio.Future()  # Run forever
                
        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {str(e)}")


async def main():
    """Main function to run the WebSocket MCP server"""
    host = os.getenv("MCP_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_PORT", "8001"))
    
    server = WebSocketMCPServer(host, port)
    await server.start_server()


if __name__ == "__main__":
    asyncio.run(main())
