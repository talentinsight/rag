# 🚀 CI/CD Setup - GitHub to AWS Automatic Deployment

## 📋 Overview

This project is configured with automatic deployment from GitHub to AWS EC2. Every commit to the `main` branch triggers an automatic deployment.

## 🔧 GitHub Actions Workflow

### Workflow File: `.github/workflows/deploy-to-aws.yml`

**Triggers:**
- Push to `main` branch
- Pull request to `main` branch

**Steps:**
1. **Checkout code** - Gets latest code from GitHub
2. **Set up Python 3.8** - Matches production environment
3. **Install dependencies** - Installs required packages
4. **Run tests** - Validates code quality
5. **Configure AWS credentials** - Sets up AWS access
6. **Deploy to EC2** - Updates production server
7. **Notify status** - Reports deployment results

## 🔑 Required GitHub Secrets

You need to configure these secrets in your GitHub repository:

### Settings → Secrets and variables → Actions → New repository secret

| Secret Name | Description | Example Value |
|-------------|-------------|---------------|
| `AWS_ACCESS_KEY_ID` | AWS IAM access key | `AKIAZCE6QHVVP4YBC7VJ` |
| `AWS_SECRET_ACCESS_KEY` | AWS IAM secret key | `WJ7V1IS01vPH6Dhu3Cej...` |
| `EC2_HOST` | EC2 instance public IP | `54.91.86.239` |
| `EC2_SSH_KEY` | SSH private key content | `-----BEGIN RSA PRIVATE KEY-----...` |

### 📝 How to Add Secrets:

1. Go to your GitHub repository
2. Click **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Add each secret with the exact name and value

## 🛠️ Manual Deployment

If you need to deploy manually, use the deployment script:

```bash
./deploy_github_to_aws.sh
```

## 🔄 Deployment Process

### Automatic (GitHub Actions):
1. Developer pushes code to `main` branch
2. GitHub Actions workflow triggers
3. Code is tested and packaged
4. Package is uploaded to EC2
5. EC2 service is updated and restarted
6. Health check confirms deployment

### Manual:
1. Run `./deploy_github_to_aws.sh`
2. Script packages latest code
3. Uploads to EC2 and deploys
4. Restarts services
5. Confirms deployment success

## 📊 Monitoring Deployment

### GitHub Actions:
- Go to **Actions** tab in your GitHub repository
- View workflow runs and logs
- Check deployment status and errors

### EC2 Server:
```bash
# Check service status
sudo systemctl status rag-app

# View logs
sudo journalctl -u rag-app.service -f

# Test API
curl -k https://54.91.86.239/health
```

## 🚨 Troubleshooting

### Common Issues:

1. **SSH Connection Failed**
   - Check EC2_SSH_KEY secret format
   - Verify EC2_HOST IP address
   - Ensure security group allows SSH (port 22)

2. **Service Restart Failed**
   - Check Python dependencies
   - Verify environment variables
   - Review service logs

3. **Health Check Failed**
   - Check if service is running
   - Verify nginx configuration
   - Test API endpoints manually

### Debug Commands:
```bash
# On EC2 instance
sudo systemctl status rag-app
sudo journalctl -u rag-app.service --no-pager -n 50
curl -k https://localhost/health
```

## 🔐 Security Notes

- SSH keys are stored as GitHub secrets (encrypted)
- Environment variables remain on EC2 (not in GitHub)
- Bearer tokens are not exposed in logs
- HTTPS is enforced for all API calls

## 📈 Deployment History

Deployments are tracked in:
- GitHub Actions workflow runs
- EC2 backup directories (`src_backup_YYYYMMDD_HHMMSS`)
- System logs (`journalctl`)

## 🎯 Next Steps

1. **Set up GitHub secrets** (required for auto-deployment)
2. **Test manual deployment** with `./deploy_github_to_aws.sh`
3. **Make a test commit** to trigger auto-deployment
4. **Monitor deployment** in GitHub Actions
5. **Verify API** is working after deployment
