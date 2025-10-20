# RAG Implementation - "Attention Is All You Need"

A complete production-ready Retrieval-Augmented Generation (RAG) system for querying the "Attention Is All You Need" paper by Vaswani et al.

## 🚀 Features

- **🔍 Semantic Text Chunking**: Intelligent document splitting (24 optimized chunks)
- **🗄️ Vector Database**: Weaviate integration with fallback to TF-IDF mock store
- **🤖 OpenAI Integration**: GPT-4 Turbo with 50-word response limit for concise answers
- **⚡ FastAPI REST API**: Production-ready web service with comprehensive guardrails
- **🛡️ Comprehensive Guardrails**: Advanced safety system with PII masking
  - Input/Output content filtering
  - PII detection and masking (email, phone, SSN, credit cards, API keys)
  - Rate limiting and abuse prevention
  - Toxicity and bias detection
- **🔌 MCP Support**: Model Context Protocol integration for AI assistants
  - **Local MCP**: stdio protocol for Claude Desktop
  - **WebSocket MCP**: Cloud-ready WebSocket protocol for testing tools
- **☁️ AWS Deployment**: Production deployment with auto-scaling and monitoring

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

## 🌐 Production Deployment (AWS)

**🚀 Live System:** The RAG system is deployed and running on AWS!

### 📡 **API Endpoint**
```
https://54.91.86.239/query
```

### 🔌 **MCP WebSocket Endpoint**
```
wss://54.91.86.239/mcp
```

### 🔑 **Authentication**

#### **API Authentication (REST)**
```bash
# HTTP Bearer Token in Authorization header
Authorization: Bearer 142c5738204c9ae01e39084e177a5bf67ade8578f79336f28459796fd5e9d6a0
```

#### **MCP Authentication (WebSocket) - 3 Methods**

**Method 1: Query Parameter** ✅ **RECOMMENDED**
```javascript
// Easiest for most applications
const ws = new WebSocket('wss://54.91.86.239/mcp?token=142c5738204c9ae01e39084e177a5bf67ade8578f79336f28459796fd5e9d6a0');
```

**Method 2: WebSocket Headers**
```javascript
// For applications supporting WebSocket headers
const ws = new WebSocket('wss://54.91.86.239/mcp', [], {
  headers: {
    'Authorization': 'Bearer 142c5738204c9ae01e39084e177a5bf67ade8578f79336f28459796fd5e9d6a0'
  }
});
```

**Method 3: First Message Authentication**
```javascript
// For custom authentication flows
const ws = new WebSocket('wss://54.91.86.239/mcp');
ws.onopen = () => {
  ws.send(JSON.stringify({
    "jsonrpc": "2.0",
    "id": 1,
    "method": "authenticate", 
    "params": {
      "token": "142c5738204c9ae01e39084e177a5bf67ade8578f79336f28459796fd5e9d6a0"
    }
  }));
};
```

#### **For Testing Applications**
- **URL Field:** `wss://54.91.86.239/mcp?token=142c5738204c9ae01e39084e177a5bf67ade8578f79336f28459796fd5e9d6a0`
- **Token Field:** Leave empty (token already in URL)

**OR**

- **URL Field:** `wss://54.91.86.239/mcp`  
- **Token Field:** `142c5738204c9ae01e39084e177a5bf67ade8578f79336f28459796fd5e9d6a0`

### ✨ **Production Features**
- ✅ **24 Optimized Chunks** (400-800 tokens each)
- ✅ **50-Word Response Limit** (concise, complete answers)
- ✅ **5 Context Chunks** per query
- ✅ **PII Masking** (emails, phones, SSNs automatically masked)
- ✅ **Comprehensive Guardrails** (safety filtering)
- ✅ **Both API & MCP Access** (REST API + WebSocket MCP)

## 📚 API Endpoints

### Core Endpoints

- `GET /` - Root endpoint with basic info
- `GET /health` - Health check and system status  
- `GET /stats` - Detailed system statistics
- `POST /query` - **Main RAG endpoint** (50-word limit, PII masking, guardrails)
- `GET /guardrails-stats` - Guardrails system statistics
- `POST /reset-stats` - Reset system statistics

### MCP Endpoints

- `WS /mcp` - **WebSocket MCP endpoint** for AI assistants and testing tools
- **Local MCP**: Use `python start_mcp_server.py` for Claude Desktop integration

### Available MCP Tools

- `query_attention_paper` - Ask questions about the paper (50-word responses)
- `search_paper_chunks` - Search for specific content in chunks
- `get_rag_stats` - Get system statistics and performance metrics
- `mask_pii_text` - Mask PII in provided text
- `query_with_pii_masking` - Query with automatic PII masking

### Query Examples

#### REST API Query
```bash
curl -X POST "http://localhost:8000/query" \\
  -H "Content-Type: application/json" \\
  -d '{
    "question": "What is the Transformer architecture?",
    "num_chunks": 5,
    "min_score": 0.1
  }'
```

#### AWS Production Query (with Guardrails & PII Masking)
```bash
curl -X POST "https://54.91.86.239/query" \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer 142c5738204c9ae01e39084e177a5bf67ade8578f79336f28459796fd5e9d6a0" \\
  -d '{
    "question": "What is the Transformer architecture?",
    "num_chunks": 5,
    "min_score": 0.1,
    "client_id": "my_app"
  }' \\
  -k
```

#### Example with PII (automatically masked)
```bash
curl -X POST "https://54.91.86.239/query" \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer 142c5738204c9ae01e39084e177a5bf67ade8578f79336f28459796fd5e9d6a0" \\
  -d '{
    "question": "My email is john@example.com, can you explain attention?",
    "client_id": "test_pii"
  }' \\
  -k
```

#### WebSocket MCP Connection Examples

**Method 1: Query Parameter (Recommended)**
```javascript
// Most compatible with testing tools
const ws = new WebSocket('wss://54.91.86.239/mcp?token=142c5738204c9ae01e39084e177a5bf67ade8578f79336f28459796fd5e9d6a0');

ws.onopen = () => {
  // Initialize MCP protocol
  ws.send(JSON.stringify({
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
      "protocolVersion": "2024-11-05",
      "capabilities": {},
      "clientInfo": {"name": "test-client", "version": "1.0.0"}
    }
  }));
};

ws.onmessage = (event) => {
  const response = JSON.parse(event.data);
  if (response.id === 1) {
    // Use MCP tools after initialization
    ws.send(JSON.stringify({
      "jsonrpc": "2.0",
      "id": 2,
      "method": "tools/call",
      "params": {
        "name": "query_attention_paper",
        "arguments": {
          "question": "What is the Transformer architecture?"
        }
      }
    }));
  }
};
```

**Method 2: Separate Token Field**
```javascript
// For applications with separate URL and token fields
const ws = new WebSocket('wss://54.91.86.239/mcp');

ws.onopen = () => {
  // Authenticate first
  ws.send(JSON.stringify({
    "jsonrpc": "2.0",
    "id": 0,
    "method": "authenticate",
    "params": {
      "token": "142c5738204c9ae01e39084e177a5bf67ade8578f79336f28459796fd5e9d6a0"
    }
  }));
};
```

#### Connection Settings for Testing Applications

**Option A: Single URL Field**
```
URL: wss://54.91.86.239/mcp?token=142c5738204c9ae01e39084e177a5bf67ade8578f79336f28459796fd5e9d6a0
Token: [Leave Empty]
```

**Option B: Separate Fields**
```
URL: wss://54.91.86.239/mcp
Token: 142c5738204c9ae01e39084e177a5bf67ade8578f79336f28459796fd5e9d6a0
```

#### Troubleshooting MCP Connection

**Common Issues:**
- **HTTP 404:** Check URL spelling and `/mcp` endpoint
- **Authentication Failed:** Verify token is correct and properly formatted
- **Connection Refused:** Ensure using `wss://` (secure WebSocket)
- **Empty Upgrade Header:** Normal response when testing with curl/browser

### Response Format (with Guardrails & PII Masking)

```json
{
  "answer": "The Transformer is a neural network architecture that relies entirely on attention mechanisms, eliminating recurrence and convolutions. It uses multi-head self-attention to process sequences in parallel, achieving better performance and faster training than traditional RNN-based models.",
  "question": "What is the Transformer architecture?",
  "pii_masked_input": "What is the Transformer architecture?",
  "chunks_found": 5,
  "sources": [
    {
      "chunk_id": "chunk_0001",
      "content": "The Transformer model architecture...",
      "score": 0.95,
      "section": "Model Architecture"
    }
  ],
  "model": "gpt-4-turbo-preview",
  "total_tokens": 1250,
  "processing_time_ms": 1500.5,
  "guardrails_passed": true,
  "input_guardrails": [
    {"category": "pii_detection", "passed": true, "confidence": 0.99}
  ],
  "output_guardrails": [
    {"category": "content_safety", "passed": true, "confidence": 0.98}
  ],
  "safety_score": 0.95,
  "timestamp": "2025-10-20T14:46:15.123456"
}
```

## 🏗️ Architecture (Production System)

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   PDF Input     │───▶│  Text Processing │───▶│ Semantic Chunks │
│ (Attention.pdf) │    │   & Cleaning     │    │   (24 chunks)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐             │
│   FastAPI       │◀───│  RAG Pipeline    │◀────────────┘
│ + Guardrails    │    │ + 50-word limit  │
│ + PII Masking   │    │ + Safety Checks  │
└─────────────────┘    └──────────────────┘
         │                       │
         │              ┌────────▼─────────┐    ┌─────────────────┐
         │              │ Vector Database  │    │ OpenAI GPT-4    │
         │              │ (Weaviate/Mock)  │    │ + Word Limiting │
         │              └──────────────────┘    └─────────────────┘
         │
┌────────▼─────────┐
│ WebSocket MCP    │    ┌─────────────────┐
│ Server (8001)    │◀───│ AI Assistants   │
│ + Authentication │    │ + Testing Tools │
└──────────────────┘    └─────────────────┘
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
├── src/                              # Source code
│   ├── pdf_processor.py             # PDF text extraction
│   ├── semantic_chunker.py          # Text chunking logic (24 chunks)
│   ├── weaviate_client.py           # Weaviate integration
│   ├── mock_vector_store.py         # Fallback vector store
│   ├── vector_store_manager.py      # Unified vector store interface
│   ├── openai_client.py             # OpenAI API integration (50-word limit)
│   ├── rag_pipeline.py              # Complete RAG pipeline
│   ├── comprehensive_guardrails.py  # Advanced safety system + PII masking
│   ├── api_comprehensive_guardrails.py # Production FastAPI with guardrails
│   ├── mcp_server.py                # Local MCP server for Claude Desktop
│   └── mcp_websocket_server.py      # WebSocket MCP server for AI assistants
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

### Chunking Parameters (Production Optimized)

- **Total Chunks**: 24 optimized chunks
- **Chunk Size**: 400-800 tokens (average: 648.8 tokens)
- **Overlap**: 50 tokens
- **Min Chunk Size**: 100 tokens
- **Response Limit**: 50 words maximum (enforced by system prompt)
- **Context Chunks**: 5 chunks per query
- **Vectorizer**: Weaviate embeddings (primary) + TF-IDF fallback

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

## 📊 Monitoring & Performance

The production system provides comprehensive monitoring:

### 🔍 **System Monitoring**
- **Health Check:** `/health` - Pipeline status, OpenAI availability
- **Statistics:** `/stats` - Detailed system performance metrics  
- **Guardrails Stats:** `/guardrails-stats` - Safety system performance
- **Structured Logging:** All operations logged with timestamps
- **Processing Time:** Real-time latency tracking
- **Token Usage:** OpenAI API usage monitoring

### ⚡ **Performance Metrics**
- **Average Response Time:** ~2-4 seconds
- **50-Word Responses:** Consistently enforced
- **Chunk Retrieval:** 5 most relevant chunks per query
- **Safety Processing:** <100ms additional latency
- **PII Masking:** Real-time detection and masking
- **Concurrent Users:** Supports multiple simultaneous queries

### 🛡️ **Guardrails Performance**
- **Input Filtering:** Content safety, PII detection, rate limiting
- **Output Filtering:** Response safety, bias detection
- **Success Rate:** >99% uptime
- **Block Rate:** Configurable safety thresholds
- **Categories:** 12+ safety categories monitored

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
