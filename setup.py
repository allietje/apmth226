#!/usr/bin/env python3
"""
Setup script for automatic CUDA configuration.
This script detects CUDA version and installs PyTorch with appropriate CUDA support.
"""

import subprocess
import sys
import re
import os

def run_command(cmd, check=True):
    """Run a shell command and return the output."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, check=check
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None

def detect_cuda_version():
    """Detect CUDA version from nvcc or nvidia-smi."""
    # Try nvcc first
    nvcc_output = run_command("nvcc --version", check=False)
    if nvcc_output:
        match = re.search(r'release (\d+\.\d+)', nvcc_output)
        if match:
            return float(match.group(1))
    
    # Try nvidia-smi
    nvidia_smi_output = run_command("nvidia-smi", check=False)
    if nvidia_smi_output:
        # nvidia-smi shows driver version, not CUDA version directly
        # But if it works, CUDA is likely available
        return "available"
    
    return None

def get_pytorch_cuda_wheel(cuda_version):
    """Map CUDA version to PyTorch CUDA wheel version."""
    if cuda_version == "available":
        # Default to cu118 if we detect GPU but can't determine exact version
        return "cu118"
    
    if cuda_version >= 12.1:
        return "cu121"
    elif cuda_version >= 11.8:
        return "cu118"
    elif cuda_version >= 11.7:
        return "cu117"
    else:
        return "cu118"  # Default fallback

def install_pytorch_with_cuda(cuda_wheel):
    """Install PyTorch with specified CUDA wheel version."""
    index_url = f"https://download.pytorch.org/whl/{cuda_wheel}"
    print(f"Installing PyTorch with CUDA {cuda_wheel} support...")
    cmd = f"{sys.executable} -m pip install torch torchvision torchaudio --index-url {index_url}"
    subprocess.run(cmd, shell=True, check=True)

def install_requirements():
    """Install other requirements from requirements.txt."""
    if os.path.exists("requirements.txt"):
        print("Installing other requirements...")
        subprocess.run(
            f"{sys.executable} -m pip install -r requirements.txt",
            shell=True,
            check=True
        )

def main():
    print("Setting up environment with automatic CUDA detection...\n")
    
    cuda_version = detect_cuda_version()
    
    if cuda_version:
        if isinstance(cuda_version, float):
            print(f"Detected CUDA version: {cuda_version}")
        else:
            print("NVIDIA GPU detected (CUDA version could not be determined)")
        
        cuda_wheel = get_pytorch_cuda_wheel(cuda_version)
        install_pytorch_with_cuda(cuda_wheel)
    else:
        print("No CUDA detected. Installing CPU-only PyTorch...")
        subprocess.run(
            f"{sys.executable} -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu",
            shell=True,
            check=True
        )
    
    # Install other requirements (excluding torch packages if they're in requirements.txt)
    install_requirements()
    
    print("\nSetup complete!")

if __name__ == "__main__":
    main()

