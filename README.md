# RAG Implementation - "Attention Is All You Need"

A complete Retrieval-Augmented Generation (RAG) system for querying the "Attention Is All You Need" paper by Vaswani et al.

## 🚀 Features

- **Hybrid RAG System**: Flexible context handling for testing and production
  - **Normal RAG**: Traditional retrieval from document database
  - **External Context Mode**: Use provided passages without retrieval
  - **Hybrid Mode**: Combine external context with retrieved chunks
- **Semantic Text Chunking**: Intelligent document splitting with overlap
- **Vector Database**: Weaviate integration with fallback to TF-IDF mock store
- **OpenAI Integration**: GPT-4 Turbo for response generation and embeddings
- **FastAPI REST API**: Production-ready web service with context source tracking
- **MCP Support**: Model Context Protocol integration for AI assistants
  - **Local MCP**: stdio protocol for Claude Desktop
  - **WebSocket MCP**: Cloud-ready WebSocket protocol for testing tools
- **AWS Deployment**: Cloud deployment configuration with auto-scaling

## 📋 Requirements

- Python 3.8+ (✅ Deployed with Python 3.8.20)
- OpenAI API key (✅ Configured)
- Docker (optional, for Weaviate)
- AWS account (✅ Deployed on EC2)

## 🛠️ Installation

1. **Clone and setup environment**:
   ```bash
   cd /path/to/rag
   python3 -m venv rag_env
   source rag_env/bin/activate  # On Windows: rag_env\\Scripts\\activate
   pip install -r requirements.txt
   ```

2. **Configure environment variables**:
   Create a `.env` file:
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   BEARER_TOKEN=your_bearer_token_here
   WEAVIATE_URL=http://localhost:8080
   HOST=0.0.0.0
   PORT=8000
   ENVIRONMENT=development
   PDF_PATH=./AttentionAllYouNeed.pdf
   ```

3. **Start Weaviate (optional)**:
   ```bash
   docker-compose up -d
   ```

## 🏃‍♂️ Quick Start

### 1. Test Individual Components

```bash
# Test PDF processing
cd src && python pdf_processor.py

# Test semantic chunking
python semantic_chunker.py

# Test vector store
python vector_store_manager.py

# Test RAG pipeline
python rag_pipeline.py
```

### 2. Start the API Server

```bash
# Method 1: Using the startup script
python start_server.py

# Method 2: Direct execution
cd src && python api.py
```

The API will be available at:
- **API**: http://localhost:8000
- **Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### 3. Test the API

```bash
# In another terminal
python test_api.py
```

### 4. Use with MCP (Model Context Protocol)

#### For AI assistants like Claude Desktop (Local):

```bash
# Start the local MCP server
python start_mcp_server.py
```

Then configure your MCP client (see [MCP_SETUP.md](MCP_SETUP.md) for details).

#### For Testing Tools and External Integrations (WebSocket):

The main API server includes WebSocket MCP support at `/mcp` endpoint:

```bash
# WebSocket MCP is available at:
# Local: ws://localhost:8000/mcp
# AWS: wss://54.91.86.239/mcp
```

**MCP WebSocket URL for your testing tool:**
```
wss://54.91.86.239/mcp
```

**Authentication:** Bearer token required for MCP WebSocket connection.
- **Token:** `142c5738204c9ae01e39084e177a5bf67ade8578f79336f28459796fd5e9d6a0`
- **Method:** Send authenticate message first, then use MCP tools

## 📚 API Endpoints

### Core Endpoints

- `GET /` - Root endpoint with basic info
- `GET /health` - Health check and system status
- `GET /stats` - Detailed system statistics
- `POST /query` - Query the RAG system
- `POST /search` - Search chunks without AI generation
- `GET /chunks/{chunk_id}` - Get specific chunk by ID

### MCP Endpoints

- `WS /mcp` - WebSocket MCP endpoint for testing tools and external integrations
- **Local MCP**: Use `python start_mcp_server.py` for Claude Desktop integration

### Query Examples

#### REST API Query (Normal RAG)
```bash
curl -X POST "http://localhost:8000/query" \\
  -H "Content-Type: application/json" \\
  -d '{
    "question": "What is the Transformer architecture?",
    "num_chunks": 5,
    "min_score": 0.1
  }'
```

#### External Context Only (Test Mode)
```bash
curl -X POST "http://localhost:8000/query" \\
  -H "Content-Type: application/json" \\
  -d '{
    "question": "What is attention?",
    "external_context": ["Attention is a cognitive mechanism..."],
    "use_retrieval": false
  }'
```

#### Hybrid Mode (External + Retrieved)
```bash
curl -X POST "http://localhost:8000/query" \\
  -H "Content-Type: application/json" \\
  -d '{
    "question": "What is attention?",
    "external_context": ["Test context passage..."],
    "use_retrieval": true,
    "num_chunks": 3
  }'
```

#### AWS Production Query
```bash
curl -X POST "https://54.91.86.239/query" \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer 142c5738204c9ae01e39084e177a5bf67ade8578f79336f28459796fd5e9d6a0" \\
  -d '{
    "question": "What is the Transformer architecture?",
    "num_chunks": 5,
    "min_score": 0.1
  }'
```

#### WebSocket MCP Connection
```javascript
// For testing tools and external integrations
const ws = new WebSocket('wss://54.91.86.239/mcp');

// First authenticate
ws.send(JSON.stringify({
  "jsonrpc": "2.0",
  "id": 0,
  "method": "authenticate",
  "params": {
    "token": "142c5738204c9ae01e39084e177a5bf67ade8578f79336f28459796fd5e9d6a0"
  }
}));

// Then use MCP tools
ws.send(JSON.stringify({
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "query_attention_paper",
    "arguments": {
      "question": "What is the Transformer architecture?"
    }
  }
}));
```

### Response Format

```json
{
  "answer": "The Transformer is a neural network architecture...",
  "question": "What is the Transformer architecture?",
  "chunks_found": 3,
  "sources": [
    {
      "chunk_id": "chunk_0001",
      "section": "Model Architecture", 
      "score": 0.95
    }
  ],
  "model": "gpt-4-turbo-preview",
  "total_tokens": 1250,
  "processing_time_ms": 1500.5,
  "timestamp": "2024-01-01T12:00:00"
}
```

### Response Format

All query responses include context source tracking:

```json
{
  "answer": "Generated response based on context...",
  "question": "What is attention?",
  "chunks_found": 3,
  "sources": [
    {
      "chunk_id": "external_0",
      "section": "External Context", 
      "score": 1.0
    },
    {
      "chunk_id": "chunk_123",
      "section": "3.2 Attention",
      "score": 0.85
    }
  ],
  "context_sources": {
    "external_passages": 1,
    "retrieved_chunks": 2
  },
  "processing_time_ms": 45.2,
  "timestamp": "2025-09-29T12:00:00Z"
}
```

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   PDF Input     │───▶│  Text Processing │───▶│ Semantic Chunks │
│ (Attention.pdf) │    │   & Cleaning     │    │   (33 chunks)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐             │
│   FastAPI       │◀───│  RAG Pipeline    │◀────────────┘
│   REST API      │    │                  │
└─────────────────┘    └──────────────────┘
                                │
                       ┌────────▼─────────┐    ┌─────────────────┐
                       │ Vector Database  │    │ OpenAI GPT-4    │
                       │ (Weaviate/Mock)  │    │   Response      │
                       └──────────────────┘    │   Generation    │
                                               └─────────────────┘
```

## 🧪 Testing

### Unit Tests

```bash
# Test individual components
cd src
python pdf_processor.py
python semantic_chunker.py  
python mock_vector_store.py
python openai_client.py
python rag_pipeline.py
```

### Integration Tests

```bash
# Test complete pipeline
python vector_store_manager.py

# Test API endpoints
python ../test_api.py
```

### Sample Queries

Try these questions with the system:

1. "What is the Transformer architecture?"
2. "How does multi-head attention work?"
3. "What are the key innovations in this paper?"
4. "How does the attention mechanism calculate attention weights?"
5. "What are the advantages of the Transformer over RNNs?"

## 📁 Project Structure

```
rag/
├── src/                          # Source code
│   ├── pdf_processor.py         # PDF text extraction
│   ├── semantic_chunker.py      # Text chunking logic
│   ├── weaviate_client.py       # Weaviate integration
│   ├── mock_vector_store.py     # Fallback vector store
│   ├── vector_store_manager.py  # Unified vector store interface
│   ├── openai_client.py         # OpenAI API integration
│   ├── rag_pipeline.py          # Complete RAG pipeline
│   ├── api.py                   # FastAPI application with WebSocket MCP
│   ├── mcp_server.py            # Local MCP server for Claude Desktop
│   └── mcp_websocket_server.py  # WebSocket MCP server for testing tools
├── AttentionAllYouNeed.pdf      # Source document
├── requirements.txt             # Python dependencies
├── docker-compose.yml           # Weaviate setup
├── start_server.py             # Server startup script
├── start_mcp_server.py         # MCP server startup script
├── test_api.py                 # API testing script
├── test_mcp.py                 # MCP server testing script
├── mcp_config.json             # MCP client configuration
├── MCP_SETUP.md               # MCP setup guide
├── deploy_simple.sh            # AWS deployment script
├── cleanup_aws.sh              # AWS cleanup script
├── deploy_aws.py               # Advanced AWS deployment (Python)
├── cloudformation-template.yaml # CloudFormation infrastructure
├── Dockerfile                  # Docker container configuration
├── docker-compose.prod.yml     # Production Docker Compose
├── AWS_DEPLOYMENT.md          # AWS deployment guide
└── README.md                   # This file
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `WEAVIATE_URL` | Weaviate instance URL | `http://localhost:8080` |
| `HOST` | API server host | `0.0.0.0` |
| `PORT` | API server port | `8000` |
| `DEBUG` | Enable debug mode | `True` |

### Chunking Parameters

- **Chunk Size**: 300 tokens (adjustable)
- **Overlap**: 30 tokens
- **Min Chunk Size**: 50 tokens
- **Vectorizer**: TF-IDF (1000 features) or OpenAI embeddings

## 🚀 Deployment

### Local Development

```bash
python start_server.py
```

### Docker (Weaviate)

```bash
docker-compose up -d
```

### AWS Deployment

Deploy to AWS with one command:

```bash
./deploy_simple.sh
```

This creates:
- EC2 Auto Scaling Group (1-3 instances)
- Application Load Balancer
- VPC with public subnets
- CloudWatch monitoring
- Health checks and auto-scaling

See [AWS_DEPLOYMENT.md](AWS_DEPLOYMENT.md) for detailed instructions.

## 🔍 Troubleshooting

### Common Issues

1. **Weaviate Connection Failed**
   - Ensure Docker is running
   - Check `docker-compose up -d`
   - System falls back to mock store automatically

2. **OpenAI API Errors**
   - Verify API key in `.env` file
   - Check API quota and billing
   - System provides fallback responses without AI

3. **PDF Processing Issues**
   - Ensure PDF file exists at specified path
   - Check file permissions
   - OCR artifacts are automatically cleaned

### Performance Tips

- Use Weaviate for better semantic search
- Adjust chunk size based on your use case
- Monitor OpenAI token usage
- Enable caching for repeated queries

## 📊 Monitoring

The system provides comprehensive monitoring through:

- Health check endpoint (`/health`)
- Statistics endpoint (`/stats`)
- Structured logging
- Processing time metrics
- Token usage tracking

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is for educational and research purposes.

## 🙏 Acknowledgments

- "Attention Is All You Need" paper by Vaswani et al.
- OpenAI for GPT-4 and embedding models
- Weaviate for vector database technology
- FastAPI for the web framework
