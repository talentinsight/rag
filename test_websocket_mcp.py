#!/usr/bin/env python3
"""
Test script for WebSocket MCP functionality
"""

import asyncio
import json
import websockets
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_websocket_mcp():
    """Test WebSocket MCP connection and functionality"""
    
    # Test local server first
    uri = "ws://localhost:8000/mcp"
    
    try:
        logger.info(f"🔌 Connecting to WebSocket MCP server: {uri}")
        
        async with websockets.connect(uri) as websocket:
            logger.info("✅ Connected to WebSocket MCP server")
            
            # Test 1: Initialize
            init_message = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {}
                    },
                    "clientInfo": {
                        "name": "test-websocket-client",
                        "version": "1.0.0"
                    }
                }
            }
            
            logger.info("📤 Sending initialize message...")
            await websocket.send(json.dumps(init_message))
            
            response = await websocket.recv()
            init_response = json.loads(response)
            logger.info(f"📥 Initialize response: {init_response.get('result', {}).get('serverInfo', {}).get('name', 'Unknown')}")
            
            # Test 2: List tools
            list_tools_message = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list"
            }
            
            logger.info("📤 Requesting tools list...")
            await websocket.send(json.dumps(list_tools_message))
            
            response = await websocket.recv()
            tools_response = json.loads(response)
            tools = tools_response.get('result', {}).get('tools', [])
            logger.info(f"📥 Available tools: {len(tools)}")
            
            for tool in tools:
                logger.info(f"  - {tool.get('name')}: {tool.get('description')}")
            
            # Test 3: Call get_rag_stats tool
            stats_message = {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "get_rag_stats",
                    "arguments": {}
                }
            }
            
            logger.info("📤 Calling get_rag_stats tool...")
            await websocket.send(json.dumps(stats_message))
            
            response = await websocket.recv()
            stats_response = json.loads(response)
            
            if 'result' in stats_response:
                content = stats_response['result']['content'][0]['text']
                logger.info("📥 RAG Stats received:")
                for line in content.split('\n')[:5]:  # Show first 5 lines
                    if line.strip():
                        logger.info(f"  {line}")
            else:
                logger.error(f"❌ Error in stats call: {stats_response.get('error', {})}")
            
            # Test 4: Query tool (if RAG is initialized)
            query_message = {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {
                    "name": "query_attention_paper",
                    "arguments": {
                        "question": "What is attention mechanism?",
                        "num_chunks": 3
                    }
                }
            }
            
            logger.info("📤 Testing query tool...")
            await websocket.send(json.dumps(query_message))
            
            response = await websocket.recv()
            query_response = json.loads(response)
            
            if 'result' in query_response:
                content = query_response['result']['content'][0]['text']
                logger.info("📥 Query response received:")
                # Show first few lines of response
                lines = content.split('\n')
                for line in lines[:3]:
                    if line.strip():
                        logger.info(f"  {line}")
                if len(lines) > 3:
                    logger.info("  ...")
            else:
                logger.warning(f"⚠️ Query may have failed: {query_response.get('error', {})}")
            
            logger.info("✅ WebSocket MCP test completed successfully!")
            
    except (websockets.exceptions.ConnectionClosed, websockets.exceptions.InvalidStatus, ConnectionRefusedError):
        logger.error("❌ Connection refused. Make sure the server is running:")
        logger.error("   python src/api.py")
    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")


async def test_production_websocket():
    """Test production WebSocket MCP (if available)"""
    
    uri = "wss://54.91.86.239/mcp"
    
    try:
        logger.info(f"🌐 Testing production WebSocket MCP: {uri}")
        
        async with websockets.connect(uri, ssl=True) as websocket:
            logger.info("✅ Connected to production WebSocket MCP server")
            
            # Quick test - just initialize
            init_message = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "clientInfo": {"name": "production-test-client", "version": "1.0.0"}
                }
            }
            
            await websocket.send(json.dumps(init_message))
            response = await websocket.recv()
            init_response = json.loads(response)
            
            server_name = init_response.get('result', {}).get('serverInfo', {}).get('name', 'Unknown')
            logger.info(f"✅ Production MCP server: {server_name}")
            
    except Exception as e:
        logger.warning(f"⚠️ Production test skipped: {str(e)}")


def main():
    """Main test function"""
    print("🧪 WebSocket MCP Test Suite")
    print("=" * 50)
    
    # Test local server
    asyncio.run(test_websocket_mcp())
    
    print("\n" + "=" * 50)
    
    # Test production server (optional)
    asyncio.run(test_production_websocket())
    
    print("\n🎉 Test suite completed!")
    print("\n📋 **WebSocket MCP Configuration for Testing Tools:**")
    print("   Local:      ws://localhost:8000/mcp")
    print("   Production: wss://54.91.86.239/mcp")
    print("   Protocol:   JSON-RPC 2.0 over WebSocket")
    print("   Auth:       None (uses internal OpenAI API key)")


if __name__ == "__main__":
    main()
