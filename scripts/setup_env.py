#!/usr/bin/env python3
"""
Cross-platform environment and dependency installer for VIFT_AEA.
Detects if running on CUDA-capable Linux, Apple Silicon (MPS), or CPU-only,
and installs the appropriate PyTorch wheels and other requirements.
"""
import sys
import platform
import subprocess
import os

def run(cmd):
    print(f"🔧 Executing: {' '.join(cmd)}")
    subprocess.check_call(cmd)

# Determine platform capabilities
system = platform.system()
machine = platform.machine()
print(f"🌐 System: {system}, Machine: {machine}")

# Default wheel index for CPU
extra_index = 'https://download.pytorch.org/whl/cpu'

def detect_cuda():
    try:
        # Check for nvidia-smi
        subprocess.check_output(['nvidia-smi'], stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False

has_cuda = (system == 'Linux' and detect_cuda())
has_mps = (system == 'Darwin' and machine == 'arm64')

# Install PyTorch
print("📦 Installing PyTorch and related packages...")
if has_cuda:
    extra_index = 'https://download.pytorch.org/whl/cu118'
    print("🚀 CUDA detected, installing CUDA-enabled PyTorch")
elif has_mps:
    print("🍎 Apple Silicon detected, installing CPU/MPS PyTorch")
else:
    print("💻 CPU-only environment, installing CPU PyTorch")

run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'])
run([sys.executable, '-m', 'pip', 'install', f'--extra-index-url={extra_index}', 'torch', 'torchvision', 'torchaudio'])

# Install the rest of requirements
print("📑 Installing remaining requirements from requirements.txt")
requirements_path = os.path.join(os.path.dirname(__file__), '..', 'requirements.txt')
run([sys.executable, '-m', 'pip', 'install', '-r', requirements_path])

print("🎉 Environment setup complete!")