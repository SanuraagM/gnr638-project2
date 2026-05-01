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

# Download Qwen2.5-VL-7B model weights
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Qwen/Qwen2.5-VL-7B-Instruct',
    local_dir='./Qwen2.5-VL-7B-Instruct',
    ignore_patterns=['*.bin'],
)
print('Model download complete.')
"

echo "Setup complete!"
