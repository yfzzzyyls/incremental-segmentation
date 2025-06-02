#!/usr/bin/env python3
"""
Download the Visual-Selective-VIO pretrained model using gdown.
Simple and reliable script that downloads from the official Google Drive folder.
"""

import os
import sys
import shutil
from pathlib import Path

# Check if gdown is available
try:
    import gdown
except ImportError:
    print("ERROR: gdown is not installed.")
    print("Please install it with: pip install gdown")
    sys.exit(1)


def main():
    """Download the Visual-Selective-VIO pretrained model."""
    
    # Create pretrained_models directory
    os.makedirs("pretrained_models", exist_ok=True)
    
    # Target path for the model
    model_path = "pretrained_models/vf_512_if_256_3e-05.model"
    
    # Check if model already exists with correct size
    if os.path.exists(model_path):
        file_size = os.path.getsize(model_path)
        file_size_mb = file_size / (1024 * 1024)
        print(f"Model already exists at {model_path}")
        print(f"File size: {file_size_mb:.1f} MB")
        
        if 180 < file_size_mb < 200:  # Expected around 185MB
            print("✓ Model appears to be correct. No download needed.")
            return 0
        else:
            print(f"WARNING: File size unexpected (expected ~185MB)")
            response = input("Re-download? (y/n): ")
            if response.lower() != 'y':
                return 0
            # Remove old file
            os.remove(model_path)
    
    print("\nDownloading Visual-Selective-VIO pretrained model...")
    print("This will download from the official repository folder")
    print("Expected size: ~185MB\n")
    
    # Official folder ID from Visual-Selective-VIO repository
    folder_id = "1KrxpvUV9Bn5SwUlrDKe76T2dqF1ooZyk"
    folder_url = f"https://drive.google.com/drive/folders/{folder_id}"
    
    # Create temp directory
    temp_dir = Path("temp_model_download")
    temp_dir.mkdir(exist_ok=True)
    
    try:
        print(f"Downloading from: {folder_url}")
        print("This folder contains multiple models, we'll extract the right one...\n")
        
        # Download the entire folder
        gdown.download_folder(
            id=folder_id,
            output=str(temp_dir),
            quiet=False,
            use_cookies=False
        )
        
        # Find the specific model file we need
        target_file = temp_dir / "vf_512_if_256_3e-05.model"
        
        if target_file.exists():
            # Move to final location
            shutil.move(str(target_file), model_path)
            
            # Verify size
            file_size = os.path.getsize(model_path)
            file_size_mb = file_size / (1024 * 1024)
            
            print(f"\n✓ Download successful!")
            print(f"Model saved to: {model_path}")
            print(f"File size: {file_size_mb:.1f} MB")
            
            # Clean up other downloaded files
            print("\nCleaning up temporary files...")
            shutil.rmtree(temp_dir)
            
            return 0
        else:
            print("\nERROR: Expected model file not found in download")
            print("Downloaded files:")
            for f in temp_dir.iterdir():
                if f.is_file():
                    print(f"  - {f.name} ({f.stat().st_size / 1024 / 1024:.1f} MB)")
            
            # Clean up
            shutil.rmtree(temp_dir)
            return 1
            
    except Exception as e:
        print(f"\nERROR: Download failed with error: {e}")
        
        # Clean up on error
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        
        print("\nTroubleshooting:")
        print("1. Make sure you have internet connection")
        print("2. Try updating gdown: pip install --upgrade gdown")
        print("3. If the error persists, download manually:")
        print(f"   - Go to: {folder_url}")
        print("   - Download 'vf_512_if_256_3e-05.model' (~185MB)")
        print(f"   - Save it to: {model_path}")
        
        return 1


if __name__ == "__main__":
    # Check virtual environment
    if 'py39' not in sys.prefix:
        print("WARNING: Not running in py39 virtual environment")
        print("Recommended: source ~/venv/py39/bin/activate")
    
    sys.exit(main())