#!/bin/bash

# Setup script for automatic CUDA configuration
# This script detects CUDA version and installs PyTorch with appropriate CUDA support

set -e

echo "Setting up environment with automatic CUDA detection..."

# Check if CUDA is available
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
    echo "Detected CUDA version: $CUDA_VERSION"
    
    # Map CUDA version to PyTorch CUDA wheel version
    if [[ $(echo "$CUDA_VERSION >= 12.1" | bc -l 2>/dev/null || echo "0") == "1" ]]; then
        CUDA_WHEEL="cu121"
    elif [[ $(echo "$CUDA_VERSION >= 11.8" | bc -l 2>/dev/null || echo "0") == "1" ]]; then
        CUDA_WHEEL="cu118"
    elif [[ $(echo "$CUDA_VERSION >= 11.7" | bc -l 2>/dev/null || echo "0") == "1" ]]; then
        CUDA_WHEEL="cu117"
    else
        CUDA_WHEEL="cu118"  # Default to cu118 for older versions
    fi
    
    echo "Installing PyTorch with CUDA $CUDA_WHEEL support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/$CUDA_WHEEL
    
elif command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected but nvcc not found. Installing PyTorch with CUDA 11.8 (default)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "No CUDA detected. Installing CPU-only PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install other requirements
echo "Installing other requirements..."
pip install -r requirements.txt

echo "Setup complete!"

