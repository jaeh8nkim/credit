#!/bin/bash

# Usage:
#   chmod +x create_atpo_env.sh
#   ./create_atpo_env.sh

# Create conda environment
conda create -n atpo python=3.10 -y

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate atpo

# Install PyTorch first (foundational, with CUDA 12.x support)
pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu124

# Install numpy
pip install numpy==2.2.6

# Install ML libraries (order: foundational -> dependent)
pip install transformers==4.57.3
pip install accelerate==1.12.0
pip install deepspeed==0.18.3
pip install vllm==0.11.2
pip install trl==0.26.2
pip install math-verify==0.8.0

# Install remaining libraries (no version pinning)
pip install datasets python-dotenv wandb matplotlib ipykernel

echo "Environment 'atpo' created successfully!"
echo "Activate with: conda activate atpo"
