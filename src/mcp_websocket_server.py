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

from rag_pipeline import RAGPipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebSocketMCPServer:
    """WebSocket-based MCP Server for cloud deployment"""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8001):
        self.host = host
        self.port = port
        self.rag_pipeline: Optional[RAGPipeline] = None
        self.clients: set = set()
        
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
        logger.info(f"New MCP client connected: {client_id}")
        
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
        """Handle paper query tool"""
        question = arguments.get("question", "")
        num_chunks = arguments.get("num_chunks", 5)
        min_score = arguments.get("min_score", 0.1)
        
        if not question:
            return "❌ Please provide a question to ask about the paper."
        
        logger.info(f"WebSocket MCP Query: {question[:50]}...")
        
        # Execute query
        result = self.rag_pipeline.query(question, num_chunks, min_score)
        
        # Format response
        response_parts = []
        response_parts.append(f"**Answer:** {result.get('answer', 'No answer generated')}")
        response_parts.append(f"**Chunks Found:** {result.get('chunks_found', 0)}")
        
        if result.get('model'):
            response_parts.append(f"**Model:** {result['model']}")
        
        if result.get('total_tokens'):
            response_parts.append(f"**Tokens Used:** {result['total_tokens']}")
        
        # Sources
        sources = result.get('sources', [])
        if sources:
            response_parts.append("**Sources:**")
            for i, source in enumerate(sources, 1):
                response_parts.append(f"{i}. {source.get('chunk_id', 'Unknown')} (Score: {source.get('score', 0):.3f})")
        
        return "\n".join(response_parts)
    
    async def handle_search_chunks(self, arguments: Dict[str, Any]) -> str:
        """Handle chunk search tool"""
        query = arguments.get("query", "")
        limit = arguments.get("limit", 5)
        
        if not query:
            return "❌ Please provide a search query."
        
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
