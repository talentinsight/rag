#!/usr/bin/env python3
"""
Startup script for the RAG MCP Server
"""

import os
import sys
import subprocess
import asyncio

def start_mcp_server():
    """Start the MCP server"""
    print("🚀 Starting RAG MCP Server...")
    
    # Change to src directory
    src_dir = os.path.join(os.path.dirname(__file__), "src")
    os.chdir(src_dir)
    
    # Get virtual environment python
    venv_python = os.path.join("..", "rag_env", "bin", "python")
    
    try:
        print(f"📍 Working directory: {os.getcwd()}")
        print(f"🐍 Using Python: {venv_python}")
        print("🔌 MCP Server starting...")
        print("📚 Available tools:")
        print("  - query_attention_paper: Ask questions about the paper")
        print("  - search_paper_chunks: Search for specific content")
        print("  - get_paper_chunk: Retrieve specific chunks by ID")
        print("  - get_rag_stats: Get system statistics")
        print("\\n🔄 Server running (Ctrl+C to stop)...")
        
        # Set environment variables
        env = os.environ.copy()
        env["PYTHONPATH"] = os.getcwd()
        
        # Run the MCP server
        subprocess.run([venv_python, "mcp_server.py"], env=env, check=True)
        
    except KeyboardInterrupt:
        print("\\n🛑 MCP Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ MCP Server failed to start: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    start_mcp_server()
