#!/bin/bash
set -e

# Clone the repository
git clone https://github.com/SanuraagM/gnr638-project2.git
cd gnr638-project2

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
