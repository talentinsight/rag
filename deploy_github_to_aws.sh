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
  src/ \
  requirements.txt \
  README.md \
  nginx-configs/ \
  --exclude='src/__pycache__' \
  --exclude='*.pyc' \
  --exclude='.env' \
  --exclude='*.pem' \
  --exclude='*.csv'

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
source rag_env_38/bin/activate
pip install -r requirements.txt --upgrade

# Restart service
sudo systemctl restart rag-app
sleep 3
sudo systemctl status rag-app --no-pager

# Test deployment
echo "=== Testing deployment ==="
sleep 5
curl -k https://localhost/health && echo "✅ Health check passed" || echo "❌ Health check failed"

echo "✅ Deployment completed successfully"
EOSSH

# Cleanup
rm rag-deployment.tar.gz

echo "🎉 GitHub to AWS deployment completed!"
echo "📡 URL: https://$EC2_HOST"
echo "🔍 Check status: curl -k https://$EC2_HOST/health"
