# apmth226

## Setup

This project includes automatic CUDA detection and configuration. To set up the environment:

### Automatic Setup (Recommended)

**Option 1: Using the shell script**
```bash
bash setup.sh
```

**Option 2: Using the Python script**
```bash
python setup.py
```

Both scripts will:
- Automatically detect if CUDA is available on your system
- Install PyTorch with the appropriate CUDA version
- Install all other dependencies from `requirements.txt`

### Manual Setup

If you prefer to install manually:

1. Install PyTorch with CUDA (choose based on your CUDA version):
   ```bash
   # CUDA 12.1+
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   
   # CUDA 11.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # CPU only
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

2. Install other dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Requirements

- Python 3.7+
- CUDA (optional, for GPU acceleration)
- See `requirements.txt` for Python package dependencies