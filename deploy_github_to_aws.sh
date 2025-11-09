#!/bin/bash

# GitHub to AWS Deployment Script
# This script deploys the latest code from GitHub to AWS EC2
#
# ⚠️  IMPORTANT: This is a MANUAL deployment script for testing only!
# ⚠️  For production, use GitHub Actions which automatically injects secrets
# ⚠️  If you use this script manually, replace YOUR_*_HERE with actual values

set -e

# Configuration
EC2_HOST="54.91.86.239"
SSH_KEY="~/.ssh/rag-keypair.pem"
REPO_URL="https://github.com/talentinsight/rag.git"

echo "🚀 Starting GitHub to AWS deployment..."

# Create deployment package
echo "=== Creating deployment package ==="
tar -czf rag-deployment.tar.gz \
  --exclude='src/__pycache__' \
  --exclude='*.pyc' \
  --exclude='.env' \
  --exclude='*.pem' \
  --exclude='*.csv' \
  src/ \
  requirements.txt \
  README.md \
  nginx-configs/

echo "=== Uploading to EC2 ==="
scp -i $SSH_KEY -o StrictHostKeyChecking=no \
  rag-deployment.tar.gz ec2-user@$EC2_HOST:/tmp/

echo "=== Deploying on EC2 ==="
ssh -i $SSH_KEY -o StrictHostKeyChecking=no ec2-user@$EC2_HOST << 'EOSSH'

echo "=== Deploying new version ==="
cd /opt/rag-app

# Backup current version
sudo cp -r src src_backup_$(date +%Y%m%d_%H%M%S) 2>/dev/null || echo "No previous src to backup"

# Extract new version
tar -xzf /tmp/rag-deployment.tar.gz

# Update environment - use Python 3.13
source rag_env/bin/activate 2>/dev/null || {
    echo "Creating Python 3.13 virtual environment..."
    python3.13 -m venv rag_env || python3 -m venv rag_env
    source rag_env/bin/activate
}
pip install -r requirements.txt --upgrade

# Stop service before updating
echo "=== Stopping existing service ==="
sudo systemctl stop rag-app || echo "Service not running"

# Clear any cached vector store data to force re-initialization
echo "=== Clearing vector store cache ==="
rm -rf /tmp/mock_vector_store.pkl || echo "No mock store to clear"
rm -rf /opt/rag-app/vector_store_cache/* || echo "No cache directory"

# Create/Update systemd services (always recreate to ensure latest config)
echo "=== Creating/Updating systemd services ==="

# Main RAG API service
sudo tee /etc/systemd/system/rag-app.service > /dev/null << EOF
[Unit]
Description=RAG API Service
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/opt/rag-app
Environment=PATH=/opt/rag-app/rag_env/bin:/opt/rag-app/rag_env_38/bin:/usr/bin:/bin
Environment=PYTHONPATH=/opt/rag-app
Environment=BEARER_TOKEN=YOUR_BEARER_TOKEN_HERE
Environment=OPENAI_API_KEY=YOUR_OPENAI_API_KEY_HERE
Environment=PDF_PATH=/opt/rag-app/AttentionAllYouNeed.pdf
Environment=WEAVIATE_URL=http://localhost:8080
Environment=HOST=0.0.0.0
Environment=PORT=8000
Environment=ENVIRONMENT=production
ExecStart=/opt/rag-app/rag_env_38/bin/uvicorn src.api_comprehensive_guardrails:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# MCP WebSocket service
sudo tee /etc/systemd/system/rag-mcp.service > /dev/null << EOF
[Unit]
Description=RAG MCP WebSocket Server
After=network.target rag-app.service

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/opt/rag-app
Environment=PATH=/opt/rag-app/rag_env/bin:/opt/rag-app/rag_env_38/bin:/usr/bin:/bin
Environment=PYTHONPATH=/opt/rag-app
Environment=BEARER_TOKEN=YOUR_BEARER_TOKEN_HERE
Environment=OPENAI_API_KEY=YOUR_OPENAI_API_KEY_HERE
Environment=PDF_PATH=/opt/rag-app/AttentionAllYouNeed.pdf
Environment=WEAVIATE_URL=http://localhost:8080
Environment=HOST=0.0.0.0
Environment=PORT=8001
Environment=ENVIRONMENT=production
ExecStart=/opt/rag-app/rag_env_38/bin/python -m src.mcp_websocket_server
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable rag-app
sudo systemctl enable rag-mcp

# Restart services
sudo systemctl restart rag-app
sudo systemctl restart rag-mcp
sleep 3
sudo systemctl status rag-app --no-pager
sudo systemctl status rag-mcp --no-pager

# Test deployment
echo "=== Testing deployment ==="
echo "--- Service Status ---"
sudo systemctl status rag-app --no-pager || echo "RAG API service not found"
sudo systemctl status rag-mcp --no-pager || echo "MCP service not found"
echo "--- Service Logs (last 20 lines) ---"
sudo journalctl -u rag-app --no-pager -n 20 || echo "No RAG API logs found"
sudo journalctl -u rag-mcp --no-pager -n 20 || echo "No MCP logs found"
echo "--- Process Check ---"
ps aux | grep -E "(uvicorn|python.*api|python.*mcp)" | grep -v grep || echo "No API/MCP processes running"
echo "--- Port Check ---"
netstat -tlnp | grep :8000 || echo "Port 8000 not in use"
netstat -tlnp | grep :8001 || echo "Port 8001 not in use"
sleep 5
curl -k https://localhost/health && echo "✅ Health check passed" || echo "❌ Health check failed"

echo "✅ Deployment completed successfully"
EOSSH

# Cleanup
rm rag-deployment.tar.gz

echo "🎉 GitHub to AWS deployment completed!"
echo "📡 URL: https://$EC2_HOST"
echo "🔍 Check status: curl -k https://$EC2_HOST/health"
