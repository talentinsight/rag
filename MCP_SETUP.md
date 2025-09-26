# MCP (Model Context Protocol) Setup Guide

This guide explains how to set up and use the RAG system with MCP (Model Context Protocol) support.

## 🔌 What is MCP?

Model Context Protocol (MCP) is a standardized way for AI applications to connect to external data sources and tools. Our RAG system exposes its functionality through MCP, allowing AI assistants like Claude Desktop to directly query the "Attention Is All You Need" paper.

## 🚀 Quick Setup

### 1. Start the MCP Server

```bash
# Method 1: Using the startup script
python start_mcp_server.py

# Method 2: Direct execution
cd src && python mcp_server.py
```

### 2. Configure Your MCP Client

For **Claude Desktop**, add this to your MCP configuration file:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "rag-attention-paper": {
      "command": "python",
      "args": [
        "/Users/sam/Desktop/rag/src/mcp_server.py"
      ],
      "env": {
        "PYTHONPATH": "/Users/sam/Desktop/rag/src",
        "OPENAI_API_KEY": "your_openai_api_key_here"
      }
    }
  }
}
```

**Important**: Replace `/Users/sam/Desktop/rag` with your actual project path and add your OpenAI API key.

### 3. Restart Claude Desktop

After updating the configuration, restart Claude Desktop to load the MCP server.

## 🛠️ Available Tools

The MCP server exposes 4 tools:

### 1. `query_attention_paper`
Ask questions about the Attention paper using RAG.

**Parameters:**
- `question` (required): Your question about the paper
- `num_chunks` (optional): Number of chunks to retrieve (1-20, default: 5)
- `min_score` (optional): Minimum similarity score (0.0-1.0, default: 0.1)

**Example:**
```
What is the Transformer architecture and how does it work?
```

### 2. `search_paper_chunks`
Search for specific content without AI generation.

**Parameters:**
- `query` (required): Search terms
- `limit` (optional): Max results (1-20, default: 5)
- `min_score` (optional): Minimum similarity score (0.0-1.0, default: 0.1)

**Example:**
```
Search for "multi-head attention"
```

### 3. `get_paper_chunk`
Retrieve a specific chunk by ID.

**Parameters:**
- `chunk_id` (required): Chunk identifier (e.g., "chunk_0001")

**Example:**
```
Get chunk_0005
```

### 4. `get_rag_stats`
Get system statistics and health information.

**Parameters:** None

## 📚 Available Resources

The MCP server provides 2 resources:

### 1. `attention://paper/info`
Information about the paper including:
- Title, authors, year
- Abstract and key contributions
- Number of available chunks

### 2. `attention://system/stats`
Current RAG system statistics:
- Initialization status
- OpenAI availability
- Vector store information
- Chunk counts and dimensions

## 💬 Usage Examples

Once configured, you can use these tools in Claude Desktop:

### Basic Questions
```
"Can you query the attention paper about how the Transformer architecture works?"
```

### Specific Searches
```
"Search the attention paper for information about positional encoding"
```

### Technical Details
```
"What does the paper say about multi-head attention? Please use the RAG system to find specific details."
```

### System Information
```
"Can you check the status of the RAG system and tell me how many chunks are available?"
```

## 🔧 Configuration Options

### Environment Variables

Set these in your MCP configuration:

```json
{
  "env": {
    "OPENAI_API_KEY": "your_key_here",
    "PYTHONPATH": "/path/to/rag/src",
    "LOG_LEVEL": "INFO"
  }
}
```

### Server Parameters

The MCP server automatically:
- Initializes the RAG pipeline
- Loads the Attention paper
- Uses the mock vector store (for reliability)
- Connects to OpenAI (if API key provided)

## 🐛 Troubleshooting

### Common Issues

1. **Server won't start**
   ```bash
   # Check Python path and dependencies
   cd /Users/sam/Desktop/rag/src
   ../rag_env/bin/python mcp_server.py
   ```

2. **Tools not appearing in Claude**
   - Verify the configuration file path
   - Check that the server path is correct
   - Restart Claude Desktop completely

3. **OpenAI errors**
   - Ensure OPENAI_API_KEY is set correctly
   - Check API quota and billing
   - Server will work without OpenAI (limited functionality)

4. **Import errors**
   - Verify PYTHONPATH in configuration
   - Ensure all dependencies are installed
   - Check virtual environment activation

### Debug Mode

Run the server with debug output:

```bash
cd src
PYTHONPATH=. python mcp_server.py 2>&1 | tee mcp_debug.log
```

### Test the Server

Use the test script to verify functionality:

```bash
python test_mcp.py
```

## 📊 Monitoring

The MCP server provides:
- Structured logging to stderr
- Tool usage statistics
- Error handling and reporting
- Resource access tracking

## 🔒 Security Notes

- The server runs locally and doesn't expose network ports
- Communication is through stdio (standard input/output)
- OpenAI API key is handled securely through environment variables
- No data is stored permanently by the MCP server

## 🚀 Advanced Usage

### Custom Queries

You can ask complex questions that combine multiple aspects:

```
"Compare the attention mechanism in Transformers with traditional RNN approaches, using specific details from the paper"
```

### Research Workflows

1. **Explore sections**: Use `search_paper_chunks` to find relevant sections
2. **Deep dive**: Use `get_paper_chunk` to examine specific content
3. **Synthesize**: Use `query_attention_paper` for comprehensive answers
4. **Verify**: Use `get_rag_stats` to check system status

### Integration with Other Tools

The MCP server can be used alongside other MCP servers in Claude Desktop, allowing you to:
- Query the Attention paper
- Access other research papers (if you have other MCP servers)
- Combine information from multiple sources

## 📈 Performance

- **Startup time**: ~5-10 seconds (includes document loading)
- **Query time**: ~1-3 seconds (depending on OpenAI API)
- **Memory usage**: ~200MB (includes document and vectors)
- **Concurrent requests**: Handled sequentially for stability

## 🤝 Contributing

To extend the MCP server:

1. Add new tools in `mcp_server.py`
2. Update the tool list in `handle_list_tools()`
3. Implement handlers in `handle_call_tool()`
4. Test with `test_mcp.py`
5. Update this documentation

## 📄 Protocol Details

- **MCP Version**: 2024-11-05
- **Transport**: stdio
- **Capabilities**: tools, resources
- **Server Name**: rag-attention-paper
- **Server Version**: 1.0.0
