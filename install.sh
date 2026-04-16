#!/bin/bash
# Developer Setup
# Target: PyTorch 2.2.0 + CUDA 12.1

echo "=========================================="
echo "  GraphNeT Developer Setup"
echo "=========================================="
echo "[INFO] Forcing PyTorch 2.2.0 with CUDA 12.1"
echo "------------------------------------------"

# 1. Create the Conda Environment
echo "[1/4] Building 'graphnet_dev' Conda environment..."
conda create -n graphnet_dev python=3.9 -y

# Source conda so 'activate' works inside the script
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate graphnet_dev

# 2. Install the PyTorch Engine
echo "[2/4] Installing PyTorch Engine (CUDA 12.1)..."
conda install pytorch==2.2.0 torchvision==0.17.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 3. Install the PyG C++ Binaries via Conda-Forge
echo "[3/4] Installing PyTorch Geometric C++ Extensions..."
conda install pyg pytorch-scatter pytorch-sparse pytorch-cluster pytorch-spline-conv -c pyg -c conda-forge -y

# 4. Install Scientific Stack via Conda-Forge
echo "Installing Heavy Dependencies..."
conda install libstdcxx-ng matplotlib pillow numpy scipy pandas scikit-learn h5py pyarrow polars -c conda-forge -y

# 5. Wire the local code
echo "Linking local repository..."
pip install -e .[torch]

echo "=========================================="
echo " SUCCESS! The developer environment is ready."
echo " To begin developing, run: conda activate graphnet_dev"
echo "=========================================="
