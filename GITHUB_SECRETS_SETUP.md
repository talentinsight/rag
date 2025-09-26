# 🔐 GitHub Secrets Setup Guide

## 📋 Required Secrets for CI/CD

To enable automatic deployment from GitHub to AWS, you need to configure these secrets in your GitHub repository.

## 🛠️ Step-by-Step Setup

### 1. Go to GitHub Repository Settings

1. Navigate to: https://github.com/talentinsight/rag
2. Click **Settings** tab
3. In the left sidebar, click **Secrets and variables** → **Actions**
4. Click **New repository secret**

### 2. Add Required Secrets

Add each of these secrets one by one:

#### 🔑 AWS_ACCESS_KEY_ID
- **Name**: `AWS_ACCESS_KEY_ID`
- **Value**: Your AWS IAM access key ID
- **Description**: AWS IAM user access key for deployment

#### 🔐 AWS_SECRET_ACCESS_KEY
- **Name**: `AWS_SECRET_ACCESS_KEY`
- **Value**: Your AWS IAM secret access key
- **Description**: AWS IAM user secret key for deployment

#### 🌐 EC2_HOST
- **Name**: `EC2_HOST`
- **Value**: Your EC2 instance public IP address
- **Description**: Public IP address of your EC2 instance

#### 🔑 EC2_SSH_KEY
- **Name**: `EC2_SSH_KEY`
- **Value**: Copy the entire content of your SSH private key file
- **Description**: SSH private key for EC2 access

**To get the SSH key content:**
```bash
cat ~/.ssh/rag-keypair.pem
```

Copy the entire output including the BEGIN and END lines.

## ✅ Verification

After adding all secrets, you should see all four secrets listed in your repository settings.

## 🧪 Test the CI/CD Pipeline

### Method 1: Make a Test Commit
```bash
# Make a small change
echo "# Test CI/CD" >> README.md
git add README.md
git commit -m "Test: Trigger CI/CD pipeline"
git push origin main
```

### Method 2: Manual Deployment
```bash
./deploy_github_to_aws.sh
```

## 📊 Monitor Deployment

### GitHub Actions:
1. Go to **Actions** tab in your repository
2. Watch the workflow run
3. Check logs for any errors

### Expected Workflow Steps:
1. ✅ Checkout code
2. ✅ Set up Python 3.8
3. ✅ Install dependencies
4. ✅ Run tests
5. ✅ Configure AWS credentials
6. ✅ Deploy to EC2
7. ✅ Notify deployment status

## 🚀 Ready to Deploy!

Once secrets are configured:
1. Make any code change
2. Commit and push to `main`
3. Watch GitHub Actions deploy automatically
4. Verify your API is working

Your RAG system will now update automatically with every commit! 🎉
