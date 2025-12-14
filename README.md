# APMTH 226: Direct Feedback Alignment vs Backpropagation

A GPU-accelerated implementation and empirical comparison of Direct Feedback Alignment (DFA) and Backpropagation (BP) algorithms for training deep neural networks on the MNIST dataset.

## Overview

This project implements and compares two fundamental algorithms for training neural networks:

- **Backpropagation (BP)**: The standard algorithm that backpropagates error gradients through the network using the transpose of forward weights
- **Direct Feedback Alignment (DFA)**: An alternative approach that uses fixed random feedback matrices to send output errors directly to each layer

The implementation includes comprehensive experiments examining:
- Reproduction of paper by Nøkland (2016)
- Convergence behavior across different network architectures (depths: 2-16 layers, widths: 200-1200 units)
- Performance scaling with dataset size (1000 to 60,000 samples)

## Key Findings

### Convergence Analysis
- DFA generally converges faster than BP across most network architectures
- BP shows more stable performance but slower convergence, especially in deeper networks
- Both algorithms achieve similar final test accuracies (2-3% error rates on MNIST)
- DFA maintains consistent performance even in very deep networks (16 layers)

### Dataset Size Scaling
- Both algorithms show similar performance degradation with smaller datasets
- DFA maintains slight edge in convergence speed across all dataset sizes
- Final performance saturates around 5,000-10,000 training samples

## Setup

This project includes automatic CUDA detection and configuration. To set up the environment:

### Automatic Setup (Recommended)

**Option 1: Using the shell script**
Run setup.sh in terminal

**Option 2: Using the Python script**
``` bash
python setup.py
```

Both scripts will:
- Automatically detect if CUDA is available on your system
- Install PyTorch with the appropriate CUDA version
- Install all other dependencies from `requirements.txt`

### Manual Setup

If you prefer manual installation:

1. Install PyTorch with CUDA (choose based on your CUDA version):
  
   ### CUDA 12.1+
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   
   ### CUDA 11.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   ### CPU only
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   2. Install other dependencies:
  
   pip install -r requirements.txt
   ### Requirements

- Python 3.7+
- CUDA (optional, for GPU acceleration)
- PyTorch, NumPy, Matplotlib, scikit-learn

## Usage

### Running the Main Implementation

To run the basic BP vs DFA comparison:
``` bash
python implementation_gpu.py
```
This will:
- Train both BP and DFA networks on MNIST
- Save training curves, final weights, and analysis plots
- Generate t-SNE visualizations of learned representations

### Convergence Experiments

Run the full convergence analysis across network architectures:
``` bash
python convergence.py
```
This performs a grid search over:
- Network depths: 2, 4, 6, 8, 10, 12, 14, 16 layers
- Network widths: 200, 400, 600, 800, 1000, 1200 units
- Multiple random seeds for statistical significance

### Dataset Size Experiments

Investigate how performance scales with training data:
``` bash
python datasize.py
```
Tests with training set sizes: 1000, 5000, 10000, 30000, 60000 samples.

## Key Hyperparameters

The experiments use carefully tuned hyperparameters:

- **Learning Rates**: BP: 0.0005, DFA: 0.001
- **Batch Size**: 256
- **Activation**: Tanh (hidden), Sigmoid (output)
- **Initialization**: Xavier/Glorot uniform for weights
- **Feedback Scale**: 0.1 (for DFA random feedback matrices)
- **Convergence Criterion**: 5-epoch moving average with ε = 0.1% error change

## Algorithm Details

### Backpropagation (BP)
Standard gradient backpropagation where errors flow backward through the transpose of forward weight matrices:
