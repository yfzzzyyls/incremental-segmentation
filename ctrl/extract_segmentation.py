#!/usr/bin/env python3
"""
Extract segmentation masks from ADT segmentations.vrs files.
Saves masks as PNG files aligned with RGB frames.
"""

import os
import sys
import numpy as np
# cv2 only needed for PNG saving, using NPZ instead
from pathlib import Path
import argparse
from tqdm import tqdm
import json

# Fix projectaria_tools import
sys.path.append('/home/external/.local/lib/python3.9/site-packages')
from projectaria_tools.core import data_provider
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions


def extract_segmentation_masks(
    segmentation_vrs_path: Path,
    rgb_vrs_path: Path,
    output_dir: Path,
    metadata_path: Path
):
    """
    Extract segmentation masks from VRS file.
    
    Args:
        segmentation_vrs_path: Path to segmentations.vrs
        rgb_vrs_path: Path to main_recording.vrs (for timing alignment)
        output_dir: Directory to save segmentation masks
        metadata_path: Path to metadata.json with frame info
    """
    # Create output directory
    seg_output_dir = output_dir / "segmentation"
    seg_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Extracting segmentation from: {segmentation_vrs_path}")
    print(f"Output directory: {seg_output_dir}")
    
    # Load metadata for frame timing
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Open segmentation VRS
    seg_provider = data_provider.create_vrs_data_provider(str(segmentation_vrs_path))
    
    if seg_provider is None:
        print(f"Error: Could not open segmentation VRS: {segmentation_vrs_path}")
        return
    
    # Get available streams
    print("Available streams in segmentation VRS:")
    for stream_id in seg_provider.get_all_streams():
        print(f"  {stream_id}")
    
    # Use first available stream (typically 400-1 for segmentation)
    all_streams = seg_provider.get_all_streams()
    if not all_streams:
        print("Error: No streams found in segmentation VRS")
        return
    
    # Use the first stream (400-1 typically contains the segmentation)
    seg_stream_id = all_streams[0]
    print(f"Using segmentation stream: {seg_stream_id}")
    
    # Get number of frames
    num_frames = seg_provider.get_num_data(seg_stream_id)
    print(f"Total segmentation frames: {num_frames}")
    
    # Process each frame from metadata
    frames_info = metadata['frames']
    print(f"Processing {len(frames_info)} frames...")
    
    for frame_info in tqdm(frames_info):
        frame_idx = frame_info['index']
        rgb_timestamp_ns = frame_info['rgb_timestamp_ns']
        
        # Convert to microseconds for VRS
        timestamp_us = rgb_timestamp_ns // 1000
        
        # Get segmentation frame at this timestamp  
        seg_index = seg_provider.get_index_by_time_ns(
            seg_stream_id, 
            rgb_timestamp_ns, 
            TimeDomain.DEVICE_TIME, 
            TimeQueryOptions.CLOSEST
        )
        
        if seg_index >= 0:
            # Get segmentation data
            seg_data = seg_provider.get_image_data_by_index(seg_stream_id, seg_index)
            
            if seg_data is not None:
                # Convert to numpy array
                seg_array = seg_data[0].to_numpy_array()
                
                # Segmentation is typically single channel with instance IDs
                if len(seg_array.shape) == 3:
                    seg_array = seg_array[:, :, 0]
                
                # Save as NPZ file to preserve uint64 instance IDs
                output_path = seg_output_dir / f"frame_{frame_idx:06d}.npz"
                np.savez_compressed(output_path, segmentation=seg_array)
            else:
                print(f"Warning: No segmentation data for frame {frame_idx}")
                # Save empty mask
                output_path = seg_output_dir / f"frame_{frame_idx:06d}.npz"
                empty_mask = np.zeros((1408, 1408), dtype=np.uint64)  # ADT resolution
                np.savez_compressed(output_path, segmentation=empty_mask)
        else:
            print(f"Warning: No segmentation index found for frame {frame_idx}")
            # Save empty mask
            output_path = seg_output_dir / f"frame_{frame_idx:06d}.npz"
            empty_mask = np.zeros((1408, 1408), dtype=np.uint64)
            np.savez_compressed(output_path, segmentation=empty_mask)
    
    print(f"Segmentation extraction complete!")


def extract_all_sequences(data_root: str, split: str = "test"):
    """
    Extract segmentation for all sequences in a split.
    
    Args:
        data_root: Root directory of ADT data
        split: train/val/test
    """
    data_root = Path(data_root)
    adt_dir = data_root / "adt" / split
    processed_dir = data_root / "processed_adt" / split
    
    # Get all sequences
    sequences = [d for d in adt_dir.iterdir() if d.is_dir()]
    
    print(f"Found {len(sequences)} sequences in {split}")
    
    for seq_dir in sequences:
        seq_name = seq_dir.name
        print(f"\nProcessing sequence: {seq_name}")
        
        # Check if already processed
        output_dir = processed_dir / seq_name
        seg_dir = output_dir / "segmentation"
        if seg_dir.exists() and len(list(seg_dir.glob("*.png"))) > 0:
            print(f"  Segmentation already extracted, skipping")
            continue
        
        # Find VRS files
        seg_vrs = seq_dir / f"ADT_{seq_name}_segmentation.zip"
        
        # Check if segmentation zip exists and extract if needed
        if seg_vrs.exists():
            # Extract zip first
            import zipfile
            print(f"  Extracting segmentation zip...")
            with zipfile.ZipFile(seg_vrs, 'r') as zip_ref:
                zip_ref.extractall(seq_dir)
            seg_vrs = seq_dir / "segmentations.vrs"
        else:
            seg_vrs = seq_dir / "segmentations.vrs"
        
        if not seg_vrs.exists():
            print(f"  Warning: Segmentation VRS not found: {seg_vrs}")
            continue
        
        rgb_vrs = seq_dir / f"ADT_{seq_name}_main_recording.vrs"
        metadata_path = output_dir / "metadata.json"
        
        if not metadata_path.exists():
            print(f"  Warning: Metadata not found: {metadata_path}")
            print(f"  Please run extract_dataset.py first")
            continue
        
        # Extract segmentation
        try:
            extract_segmentation_masks(
                seg_vrs,
                rgb_vrs,
                output_dir,
                metadata_path
            )
        except Exception as e:
            print(f"  Error extracting segmentation: {e}")
            continue


def main():
    parser = argparse.ArgumentParser(description="Extract segmentation masks from ADT dataset")
    parser.add_argument("--data-root", type=str,
                       default="/mnt/ssd_ext/incSeg-data",
                       help="Root directory of ADT data")
    parser.add_argument("--split", type=str, default="test",
                       choices=["train", "val", "test"],
                       help="Dataset split to process")
    parser.add_argument("--sequence", type=str, default=None,
                       help="Specific sequence to process (default: all)")
    
    args = parser.parse_args()
    
    if args.sequence:
        # Process single sequence
        data_root = Path(args.data_root)
        seq_dir = data_root / "adt" / args.split / args.sequence
        output_dir = data_root / "processed_adt" / args.split / args.sequence
        
        seg_vrs = seq_dir / f"ADT_{args.sequence}_segmentation.zip"
        
        # Check and extract zip if needed
        if seg_vrs.exists():
            import zipfile
            print(f"Extracting segmentation zip...")
            with zipfile.ZipFile(seg_vrs, 'r') as zip_ref:
                zip_ref.extractall(seq_dir)
            seg_vrs = seq_dir / "segmentations.vrs"
        else:
            seg_vrs = seq_dir / "segmentations.vrs"
        
        rgb_vrs = seq_dir / f"ADT_{args.sequence}_main_recording.vrs"
        metadata_path = output_dir / "metadata.json"
        
        extract_segmentation_masks(
            seg_vrs,
            rgb_vrs,
            output_dir,
            metadata_path
        )
    else:
        # Process all sequences in split
        extract_all_sequences(args.data_root, args.split)


if __name__ == "__main__":
    main()