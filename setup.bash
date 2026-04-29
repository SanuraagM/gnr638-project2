#!/bin/bash
set -e

# Clone the repository
git clone https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME

# Create conda environment with Python 3.11
conda create -n gnr_project_env python=3.11 -y

# Activate environment
source activate gnr_project_env

# Install dependencies
pip install -r requirements.txt

# Download Qwen2-VL-7B model weights via huggingface-cli
huggingface-cli download Qwen/Qwen2-VL-7B-Instruct \
    --local-dir ./Qwen2-VL-7B-Instruct \
    --local-dir-use-symlinks False

echo "Setup complete!"
