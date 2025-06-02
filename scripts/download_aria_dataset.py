#!/usr/bin/env python3
"""
Download AriaEveryday Activities dataset sequences.
Supports downloading all sequences or a specified number.
"""

import os
import json
import argparse
import subprocess
from pathlib import Path
from tqdm import tqdm


def download_file(url, output_path):
    """Download a single file using wget."""
    if os.path.exists(output_path):
        return True  # Already downloaded
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    cmd = ['wget', '-c', '-O', output_path, url]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"Error downloading {output_path}: {e}")
        return False


def download_sequence(seq_name, download_data, output_dir):
    """Download all files for a single sequence."""
    if seq_name not in download_data['sequences']:
        print(f"Sequence {seq_name} not found in download URLs")
        return False
    
    seq_data = download_data['sequences'][seq_name]
    seq_output_dir = os.path.join(output_dir, seq_name)
    
    # Download each file type
    success = True
    for file_type, file_info in seq_data.items():
        filename = file_info['filename']
        url = file_info['download_url']
        output_path = os.path.join(seq_output_dir, filename)
        
        if not download_file(url, output_path):
            print(f"Failed to download {filename} for {seq_name}")
            success = False
    
    return success


def main():
    parser = argparse.ArgumentParser(description='Download AriaEveryday Activities dataset')
    parser.add_argument('--output-dir', type=str, default='data/aria_everyday',
                        help='Output directory for downloaded sequences')
    parser.add_argument('--urls-file', type=str, default='AriaEverydayActivities_download_urls.json',
                        help='Path to download URLs JSON file')
    parser.add_argument('--num-sequences', type=int, default=None,
                        help='Number of sequences to download (default: all)')
    parser.add_argument('--all', action='store_true',
                        help='Download all sequences')
    parser.add_argument('--start-index', type=int, default=0,
                        help='Starting sequence index')
    
    args = parser.parse_args()
    
    # Load download URLs
    if not os.path.exists(args.urls_file):
        print(f"Error: Download URLs file not found: {args.urls_file}")
        print("Please download AriaEverydayActivities_download_urls.json from the AriaEveryday website.")
        return 1
    
    with open(args.urls_file, 'r') as f:
        download_data = json.load(f)
    
    # Get sequence list
    all_sequences = sorted(download_data['sequences'].keys())
    total_sequences = len(all_sequences)
    
    # Determine which sequences to download
    if args.all:
        sequences_to_download = all_sequences
    else:
        num_to_download = args.num_sequences if args.num_sequences else total_sequences
        end_index = min(args.start_index + num_to_download, total_sequences)
        sequences_to_download = all_sequences[args.start_index:end_index]
    
    print(f"Total sequences available: {total_sequences}")
    print(f"Downloading {len(sequences_to_download)} sequences")
    print(f"Output directory: {args.output_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Download sequences with progress bar
    successful = 0
    failed = []
    
    with tqdm(total=len(sequences_to_download), desc="Downloading sequences") as pbar:
        for seq_name in sequences_to_download:
            pbar.set_description(f"Downloading {seq_name}")
            if download_sequence(seq_name, download_data, args.output_dir):
                successful += 1
            else:
                failed.append(seq_name)
            pbar.update(1)
    
    print(f"\nDownload complete!")
    print(f"Successful: {successful}/{len(sequences_to_download)}")
    print(f"Failed: {len(failed)}")
    
    if failed:
        print("\nFailed sequences:")
        for seq in failed:
            print(f"  - {seq}")
        
        # Save failed sequences to file
        with open('failed_downloads.txt', 'w') as f:
            for seq in failed:
                f.write(f"{seq}\n")
        print("\nFailed sequences saved to failed_downloads.txt")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())