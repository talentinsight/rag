#!/bin/bash

# GitHub to AWS Deployment Script
# This script deploys the latest code from GitHub to AWS EC2

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

# Update environment
source rag_env/bin/activate 2>/dev/null || source rag_env_38/bin/activate
pip install -r requirements.txt --upgrade

# Create/Update systemd service (always recreate to ensure latest config)
echo "=== Creating/Updating systemd service ==="
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
ExecStart=/opt/rag-app/rag_env_38/bin/uvicorn src.api_comprehensive_guardrails:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable rag-app

# Restart service
sudo systemctl restart rag-app
sleep 3
sudo systemctl status rag-app --no-pager

# Test deployment
echo "=== Testing deployment ==="
echo "--- Service Status ---"
sudo systemctl status rag-app --no-pager || echo "Service not found"
echo "--- Service Logs (last 20 lines) ---"
sudo journalctl -u rag-app --no-pager -n 20 || echo "No logs found"
echo "--- Process Check ---"
ps aux | grep -E "(uvicorn|python.*api)" | grep -v grep || echo "No API processes running"
echo "--- Port Check ---"
netstat -tlnp | grep :8000 || echo "Port 8000 not in use"
sleep 5
curl -k https://localhost/health && echo "✅ Health check passed" || echo "❌ Health check failed"

echo "✅ Deployment completed successfully"
EOSSH

# Cleanup
rm rag-deployment.tar.gz

echo "🎉 GitHub to AWS deployment completed!"
echo "📡 URL: https://$EC2_HOST"
echo "🔍 Check status: curl -k https://$EC2_HOST/health"
