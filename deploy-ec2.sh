#!/bin/bash
# ArionXiv AWS EC2 Free Tier Deployment Script
# Run this on a fresh EC2 t2.micro instance (Amazon Linux 2023 or Ubuntu 22.04)

set -e

APP_NAME="arionxiv"
REPO_URL="${REPO_URL:-https://github.com/ArionDas/ArionXiv.git}"
BRANCH="${BRANCH:-main}"
PORT=8000

echo "=== ArionXiv EC2 Deployment ==="

# Install Docker if not present
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        if [ "$ID" = "amzn" ]; then
            sudo yum update -y
            sudo yum install -y docker git
            sudo systemctl start docker
            sudo systemctl enable docker
            sudo usermod -aG docker "$USER"
        elif [ "$ID" = "ubuntu" ]; then
            sudo apt-get update
            sudo apt-get install -y docker.io git
            sudo systemctl start docker
            sudo systemctl enable docker
            sudo usermod -aG docker "$USER"
        fi
    fi
    echo "Docker installed. Re-login or run: newgrp docker"
fi

# Clone or update repo
if [ -d "$APP_NAME" ]; then
    echo "Updating repository..."
    cd "$APP_NAME"
    git fetch origin
    git checkout "$BRANCH"
    git pull origin "$BRANCH"
else
    echo "Cloning repository..."
    git clone -b "$BRANCH" "$REPO_URL" "$APP_NAME"
    cd "$APP_NAME"
fi

# Check for .env file
if [ ! -f .env ]; then
    echo "ERROR: .env file not found."
    echo "Create .env with required variables:"
    echo "  MONGODB_URI=<your-mongodb-uri>"
    echo "  JWT_SECRET_KEY=<your-secret>"
    echo "  GROQ_API_KEY=<your-groq-key>"
    exit 1
fi

# Build and run
echo "Building Docker image..."
sudo docker build -t "$APP_NAME:latest" .

echo "Stopping existing container..."
sudo docker stop "$APP_NAME" 2>/dev/null || true
sudo docker rm "$APP_NAME" 2>/dev/null || true

echo "Starting container..."
sudo docker run -d \
    --name "$APP_NAME" \
    --restart unless-stopped \
    --env-file .env \
    -p "$PORT":8000 \
    "$APP_NAME:latest"

echo "=== Deployment Complete ==="
echo "Health check: curl http://localhost:\"$PORT\"/health"
echo "Logs: sudo docker logs -f $APP_NAME"
