"""
MCP Server for RAG System
Exposes RAG functionality through Model Context Protocol
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Sequence
import json
import os
from datetime import datetime

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    LoggingLevel
)
import mcp.types as types

from rag_pipeline import RAGPipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global RAG pipeline instance
rag_pipeline: Optional[RAGPipeline] = None


class RAGMCPServer:
    """MCP Server for RAG System"""
    
    def __init__(self):
        self.server = Server("rag-attention-paper")
        self.rag_pipeline: Optional[RAGPipeline] = None
        self.setup_handlers()
    
    def setup_handlers(self):
        """Set up MCP server handlers"""
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            """List available tools"""
            return [
                Tool(
                    name="query_attention_paper",
                    description="Query the 'Attention Is All You Need' paper using RAG. Ask questions about the Transformer architecture, attention mechanisms, or any concepts from the paper.",
                    inputSchema={
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
                ),
                Tool(
                    name="search_paper_chunks",
                    description="Search for specific chunks in the Attention paper without generating an AI response. Useful for finding exact references or exploring specific sections.",
                    inputSchema={
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
                            },
                            "min_score": {
                                "type": "number",
                                "description": "Minimum similarity score",
                                "minimum": 0.0,
                                "maximum": 1.0,
                                "default": 0.1
                            }
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="get_paper_chunk",
                    description="Retrieve a specific chunk from the paper by its ID. Useful when you know the exact chunk you want to examine.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "chunk_id": {
                                "type": "string",
                                "description": "The ID of the chunk to retrieve (e.g., 'chunk_0001')"
                            }
                        },
                        "required": ["chunk_id"]
                    }
                ),
                Tool(
                    name="get_rag_stats",
                    description="Get statistics and information about the RAG system, including number of chunks, vector store status, and system health.",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                )
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
            """Handle tool calls"""
            
            if not self.rag_pipeline or not self.rag_pipeline.is_initialized:
                return [types.TextContent(
                    type="text",
                    text="❌ RAG pipeline is not initialized. Please check the system status."
                )]
            
            try:
                if name == "query_attention_paper":
                    return await self._handle_query_paper(arguments)
                elif name == "search_paper_chunks":
                    return await self._handle_search_chunks(arguments)
                elif name == "get_paper_chunk":
                    return await self._handle_get_chunk(arguments)
                elif name == "get_rag_stats":
                    return await self._handle_get_stats(arguments)
                else:
                    return [types.TextContent(
                        type="text",
                        text=f"❌ Unknown tool: {name}"
                    )]
                    
            except Exception as e:
                logger.error(f"Tool call failed: {str(e)}")
                return [types.TextContent(
                    type="text",
                    text=f"❌ Error executing tool '{name}': {str(e)}"
                )]
        
        @self.server.list_resources()
        async def handle_list_resources() -> List[Resource]:
            """List available resources"""
            return [
                Resource(
                    uri="attention://paper/info",
                    name="Attention Paper Information",
                    description="Information about the 'Attention Is All You Need' paper",
                    mimeType="application/json"
                ),
                Resource(
                    uri="attention://system/stats",
                    name="RAG System Statistics",
                    description="Current statistics and status of the RAG system",
                    mimeType="application/json"
                )
            ]
        
        @self.server.read_resource()
        async def handle_read_resource(uri: str) -> str:
            """Handle resource reading"""
            
            if uri == "attention://paper/info":
                paper_info = {
                    "title": "Attention Is All You Need",
                    "authors": ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit", "Llion Jones", "Aidan N. Gomez", "Lukasz Kaiser", "Illia Polosukhin"],
                    "year": 2017,
                    "venue": "NIPS 2017",
                    "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.",
                    "key_contributions": [
                        "Transformer architecture based solely on attention",
                        "Multi-head attention mechanism",
                        "Positional encoding for sequence modeling",
                        "Improved parallelization and training efficiency"
                    ],
                    "chunks_available": self.rag_pipeline.get_stats().get("vector_store_stats", {}).get("total_chunks", 0) if self.rag_pipeline else 0
                }
                return json.dumps(paper_info, indent=2)
            
            elif uri == "attention://system/stats":
                if self.rag_pipeline:
                    stats = self.rag_pipeline.get_stats()
                    stats["timestamp"] = datetime.now().isoformat()
                    return json.dumps(stats, indent=2)
                else:
                    return json.dumps({"error": "RAG pipeline not initialized"}, indent=2)
            
            else:
                raise ValueError(f"Unknown resource URI: {uri}")
    
    async def _handle_query_paper(self, arguments: Dict[str, Any]) -> List[types.TextContent]:
        """Handle paper query tool"""
        question = arguments.get("question", "")
        num_chunks = arguments.get("num_chunks", 5)
        min_score = arguments.get("min_score", 0.1)
        
        if not question:
            return [types.TextContent(
                type="text",
                text="❌ Please provide a question to ask about the paper."
            )]
        
        logger.info(f"MCP Query: {question[:50]}...")
        
        # Execute query
        result = self.rag_pipeline.query(question, num_chunks, min_score)
        
        # Format response
        response_parts = []
        
        # Main answer
        response_parts.append(f"**Answer:** {result.get('answer', 'No answer generated')}")
        
        # Metadata
        chunks_found = result.get('chunks_found', 0)
        response_parts.append(f"\\n**Chunks Found:** {chunks_found}")
        
        if result.get('model'):
            response_parts.append(f"**Model:** {result['model']}")
        
        if result.get('total_tokens'):
            response_parts.append(f"**Tokens Used:** {result['total_tokens']}")
        
        # Sources with full content
        sources = result.get('sources', [])
        if sources:
            response_parts.append("\\n**Sources:**")
            for i, source in enumerate(sources, 1):
                chunk_id = source.get('chunk_id', 'Unknown')
                section = source.get('section', 'Unknown')
                score = source.get('score', 0)
                content = source.get('content', '')
                
                response_parts.append(f"{i}. **{chunk_id}** (Section: {section}, Score: {score:.3f})")
                if content:
                    response_parts.append(f"   Content: {content[:200]}..." if len(content) > 200 else f"   Content: {content}")
        
        # Add LLM Prompt details if available
        if result.get('llm_prompt'):
            response_parts.append("\\n**LLM Prompt Details:**")
            llm_prompt = result.get('llm_prompt', {})
            if llm_prompt.get('system_prompt'):
                response_parts.append(f"System Prompt: {llm_prompt['system_prompt'][:100]}...")
            if llm_prompt.get('user_prompt'):
                response_parts.append(f"User Prompt: {llm_prompt['user_prompt'][:200]}...")
        
        return [types.TextContent(
            type="text",
            text="\\n".join(response_parts)
        )]
    
    async def _handle_search_chunks(self, arguments: Dict[str, Any]) -> List[types.TextContent]:
        """Handle chunk search tool"""
        query = arguments.get("query", "")
        limit = arguments.get("limit", 5)
        min_score = arguments.get("min_score", 0.1)
        
        if not query:
            return [types.TextContent(
                type="text",
                text="❌ Please provide a search query."
            )]
        
        logger.info(f"MCP Search: {query[:50]}...")
        
        # Execute search
        chunks = self.rag_pipeline.vector_store.search_similar(query, limit, min_score)
        
        if not chunks:
            return [types.TextContent(
                type="text",
                text=f"🔍 No chunks found for query: '{query}'"
            )]
        
        # Format results
        response_parts = [f"🔍 **Search Results for:** '{query}'", f"**Found {len(chunks)} chunks:**\\n"]
        
        for i, chunk in enumerate(chunks, 1):
            response_parts.append(f"**{i}. {chunk.get('chunk_id', 'Unknown')}**")
            response_parts.append(f"   Section: {chunk.get('section_title', 'Unknown')}")
            response_parts.append(f"   Score: {chunk.get('score', 0):.3f}")
            response_parts.append(f"   Content: {chunk.get('content', '')[:200]}...")
            response_parts.append("")
        
        return [types.TextContent(
            type="text",
            text="\\n".join(response_parts)
        )]
    
    async def _handle_get_chunk(self, arguments: Dict[str, Any]) -> List[types.TextContent]:
        """Handle get chunk tool"""
        chunk_id = arguments.get("chunk_id", "")
        
        if not chunk_id:
            return [types.TextContent(
                type="text",
                text="❌ Please provide a chunk ID."
            )]
        
        logger.info(f"MCP Get Chunk: {chunk_id}")
        
        # Get chunk
        chunk = self.rag_pipeline.vector_store.get_chunk_by_id(chunk_id)
        
        if not chunk:
            return [types.TextContent(
                type="text",
                text=f"❌ Chunk '{chunk_id}' not found."
            )]
        
        # Format chunk
        response_parts = [
            f"📄 **Chunk: {chunk_id}**",
            f"**Section:** {chunk.get('section_title', 'Unknown')}",
            f"**Tokens:** {chunk.get('token_count', 0)}",
            f"**Position:** {chunk.get('start_char', 0)}-{chunk.get('end_char', 0)}",
            "",
            "**Content:**",
            chunk.get('content', 'No content available')
        ]
        
        return [types.TextContent(
            type="text",
            text="\\n".join(response_parts)
        )]
    
    async def _handle_get_stats(self, arguments: Dict[str, Any]) -> List[types.TextContent]:
        """Handle get stats tool"""
        logger.info("MCP Get Stats")
        
        stats = self.rag_pipeline.get_stats()
        
        # Format stats
        response_parts = [
            "📊 **RAG System Statistics**",
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
                f"  - Is Fitted: {'✅ Yes' if vector_stats.get('is_fitted') else '❌ No'}",
                ""
            ])
        
        response_parts.append(f"**Timestamp:** {datetime.now().isoformat()}")
        
        return [types.TextContent(
            type="text",
            text="\\n".join(response_parts)
        )]
    
    async def initialize_rag(self):
        """Initialize the RAG pipeline"""
        try:
            logger.info("Initializing RAG pipeline for MCP server...")
            
            self.rag_pipeline = RAGPipeline(
                prefer_weaviate=False,  # Use mock store for MCP demo
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
            
            if await asyncio.get_event_loop().run_in_executor(None, self.rag_pipeline.initialize):
                logger.info("✅ RAG pipeline initialized")
                
                # Load the Attention paper
                pdf_path = os.getenv("PDF_PATH", "./AttentionAllYouNeed.pdf")
                if os.path.exists(pdf_path):
                    if await asyncio.get_event_loop().run_in_executor(None, self.rag_pipeline.load_document, pdf_path):
                        logger.info("✅ Document loaded successfully")
                    else:
                        logger.warning("Failed to load document")
                else:
                    logger.warning(f"PDF file not found: {pdf_path}")
            else:
                logger.error("Failed to initialize RAG pipeline")
                
        except Exception as e:
            logger.error(f"RAG initialization failed: {str(e)}")
    
    async def run(self):
        """Run the MCP server"""
        # Initialize RAG pipeline
        await self.initialize_rag()
        
        # Run the server
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="rag-attention-paper",
                    server_version="1.0.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=None,
                        experimental_capabilities={}
                    )
                )
            )


async def main():
    """Main function to run the MCP server"""
    logger.info("🚀 Starting RAG MCP Server...")
    
    server = RAGMCPServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
