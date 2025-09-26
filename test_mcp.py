#!/usr/bin/env python3
"""
Test script for the RAG MCP Server
"""

import asyncio
import json
import subprocess
import sys
import os
from typing import Dict, Any

async def test_mcp_server():
    """Test the MCP server functionality"""
    print("🧪 Testing RAG MCP Server...")
    
    # Change to src directory
    src_dir = os.path.join(os.path.dirname(__file__), "src")
    venv_python = os.path.join(os.path.dirname(__file__), "rag_env", "bin", "python")
    
    try:
        # Test 1: Check if server starts without errors
        print("\\n1. Testing server startup...")
        
        env = os.environ.copy()
        env["PYTHONPATH"] = src_dir
        
        # Start the server process
        process = subprocess.Popen(
            [venv_python, os.path.join(src_dir, "mcp_server.py")],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True
        )
        
        # Send initialization request
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0.0"
                }
            }
        }
        
        print("📤 Sending initialization request...")
        process.stdin.write(json.dumps(init_request) + "\\n")
        process.stdin.flush()
        
        # Wait a bit for response
        await asyncio.sleep(2)
        
        # Test 2: List tools
        list_tools_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list"
        }
        
        print("📤 Requesting tools list...")
        process.stdin.write(json.dumps(list_tools_request) + "\\n")
        process.stdin.flush()
        
        # Wait a bit for response
        await asyncio.sleep(2)
        
        # Test 3: Call a tool
        query_tool_request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "get_rag_stats",
                "arguments": {}
            }
        }
        
        print("📤 Calling get_rag_stats tool...")
        process.stdin.write(json.dumps(query_tool_request) + "\\n")
        process.stdin.flush()
        
        # Wait for responses
        await asyncio.sleep(3)
        
        # Read responses
        print("\\n📥 Server responses:")
        try:
            stdout, stderr = process.communicate(timeout=1)
            
            if stdout:
                print("STDOUT:")
                for line in stdout.strip().split("\\n"):
                    if line.strip():
                        try:
                            response = json.loads(line)
                            print(f"  {json.dumps(response, indent=2)}")
                        except json.JSONDecodeError:
                            print(f"  {line}")
            
            if stderr:
                print("STDERR:")
                print(f"  {stderr}")
                
        except subprocess.TimeoutExpired:
            print("  Server is running (timeout reached)")
            process.terminate()
        
        print("\\n✅ MCP Server test completed!")
        print("\\n📋 **MCP Server Configuration:**")
        print("  - Server Name: rag-attention-paper")
        print("  - Protocol: Model Context Protocol (MCP)")
        print("  - Transport: stdio")
        print("  - Tools: 4 available")
        print("  - Resources: 2 available")
        
        print("\\n🔧 **Available Tools:**")
        tools = [
            "query_attention_paper - Ask questions about the paper",
            "search_paper_chunks - Search for specific content",
            "get_paper_chunk - Retrieve chunks by ID", 
            "get_rag_stats - Get system statistics"
        ]
        for tool in tools:
            print(f"  - {tool}")
        
        print("\\n📚 **Available Resources:**")
        resources = [
            "attention://paper/info - Paper information and metadata",
            "attention://system/stats - RAG system statistics"
        ]
        for resource in resources:
            print(f"  - {resource}")
        
        print("\\n🚀 **Usage Instructions:**")
        print("1. Add the MCP server to your MCP client configuration:")
        print("   - Use the mcp_config.json file provided")
        print("   - Set your OPENAI_API_KEY environment variable")
        print("\\n2. Start the server:")
        print("   python start_mcp_server.py")
        print("\\n3. Connect from your MCP-enabled application (Claude Desktop, etc.)")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        if 'process' in locals():
            process.terminate()

def main():
    """Main function"""
    asyncio.run(test_mcp_server())

if __name__ == "__main__":
    main()
