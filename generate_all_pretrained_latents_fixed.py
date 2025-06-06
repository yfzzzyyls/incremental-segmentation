#!/usr/bin/env python3
"""
Complete pipeline to generate latent features using Visual-Selective-VIO pretrained model
from processed Aria data with FIXED relative pose conversion in local coordinates.
Updated to work with quaternion data directly without Euler conversion.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import json
from tqdm import tqdm
from pathlib import Path
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.models.components.vsvio import Encoder


class WrapperModel(nn.Module):
    """Wrapper for pretrained encoder."""
    def __init__(self):
        super().__init__()
        class Params:
            v_f_len = 512
            i_f_len = 256
            img_w = 512
            img_h = 256
            imu_dropout = 0.2
            
        self.Feature_net = Encoder(Params())
        
    def forward(self, imgs, imus):
        # The model outputs 10 transition features for 11 input frames
        v_feat, i_feat = self.Feature_net(imgs, imus)
        # Return visual and IMU features separately (10 features each)
        return v_feat, i_feat  # Both are [B, 10, feature_dim]


def quaternion_to_rotation_matrix(q):
    """Convert quaternion (XYZW) to rotation matrix."""
    x, y, z, w = q
    
    # Normalize quaternion
    norm = np.sqrt(x*x + y*y + z*z + w*w)
    x, y, z, w = x/norm, y/norm, z/norm, w/norm
    
    # Convert to rotation matrix
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    
    R = np.array([
        [1 - 2*(yy + zz), 2*(xy - wz), 2*(xz + wy)],
        [2*(xy + wz), 1 - 2*(xx + zz), 2*(yz - wx)],
        [2*(xz - wy), 2*(yz + wx), 1 - 2*(xx + yy)]
    ])
    
    return R


def quaternion_multiply(q1, q2):
    """Multiply two quaternions in XYZW format."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    
    # Quaternion multiplication
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    # Return in XYZW format
    return np.array([x, y, z, w])


def quaternion_inverse(q):
    """Compute quaternion inverse for XYZW format."""
    x, y, z, w = q
    norm_sq = w*w + x*x + y*y + z*z
    # Return conjugate/norm_sq in XYZW format
    return np.array([-x, -y, -z, w]) / norm_sq


def convert_absolute_to_relative_fixed(poses):
    """
    Convert absolute poses to relative poses with proper coordinate transformation.
    
    CRITICAL FIX: Transform translation into the previous frame's coordinate system!
    
    Args:
        poses: [seq_len, 7] absolute poses (translation + quaternion XYZW)
    
    Returns:
        relative_poses: [seq_len, 7] relative poses in local coordinates
    """
    seq_len = poses.shape[0]
    relative_poses = np.zeros_like(poses)
    
    # First pose is always at origin
    relative_poses[0, :3] = [0, 0, 0]
    relative_poses[0, 3:] = [0, 0, 0, 1]  # Identity quaternion in XYZW format
    
    # Convert subsequent poses to relative
    for i in range(1, seq_len):
        # Get current and previous absolute poses
        prev_trans = poses[i-1, :3]
        prev_rot = poses[i-1, 3:]
        
        curr_trans = poses[i, :3]
        curr_rot = poses[i, 3:]
        
        # CRITICAL FIX: Transform translation into previous frame's coordinate system
        # trans_world = curr_trans - prev_trans (world coordinates)
        # trans_local = R_prev^T @ trans_world (local coordinates)
        trans_world = curr_trans - prev_trans
        R_prev = quaternion_to_rotation_matrix(prev_rot)
        trans_local = R_prev.T @ trans_world  # Transform to local coordinates!
        
        # For rotation, we need the relative rotation
        # rel_rot = prev_rot^-1 * curr_rot
        prev_rot_inv = quaternion_inverse(prev_rot)
        rel_rot = quaternion_multiply(prev_rot_inv, curr_rot)
        
        # Store relative pose
        relative_poses[i, :3] = trans_local
        relative_poses[i, 3:] = rel_rot / np.linalg.norm(rel_rot)  # Normalize
    
    return relative_poses


def load_pretrained_model(model_path):
    """Load pretrained Visual-Selective-VIO model."""
    model = WrapperModel()
    # PyTorch 2.6 compatibility - set weights_only=False for loading old models
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    encoder_dict = {}
    for k, v in checkpoint.items():
        if k.startswith('Feature_net.'):
            new_k = k.replace('Feature_net.', '')
            encoder_dict[new_k] = v
    
    model.Feature_net.load_state_dict(encoder_dict, strict=False)
    print(f"✅ Loaded {len(encoder_dict)} encoder parameters")
    return model


def process_sequence(seq_dir, model, device, window_size=11, stride=1, pose_scale=100.0):
    """Process a single sequence and extract windowed features with relative poses."""
    
    # Load data
    visual_data = torch.load(os.path.join(seq_dir, 'visual_data.pt'))  # [N, 3, H, W]
    imu_data = torch.load(os.path.join(seq_dir, 'imu_data.pt'))        # [N, 33, 6]
    
    # Ensure float32 data type for compatibility with the model
    visual_data = visual_data.float()  # Convert to float32 if it's float64
    imu_data = imu_data.float()  # Convert to float32 if it's float64
    
    # Load poses - check for quaternion version first
    poses_file = os.path.join(seq_dir, 'poses_quaternion.json')
    if not os.path.exists(poses_file):
        # Fallback to regular poses.json
        poses_file = os.path.join(seq_dir, 'poses.json')
    
    with open(poses_file, 'r') as f:
        poses_data = json.load(f)
    
    # Convert poses to tensor with XYZW quaternion format
    absolute_poses = []
    for pose in poses_data:
        # Get translation
        t = pose['translation']
        
        # Check if we have quaternion data directly
        if 'quaternion' in pose:
            # Already have quaternion in XYZW format
            q = pose['quaternion']
            pose_vec = [t[0], t[1], t[2], q[0], q[1], q[2], q[3]]
        elif 'rotation_euler' in pose:
            # Convert Euler to quaternion (backward compatibility)
            from scipy.spatial.transform import Rotation
            euler = pose['rotation_euler']  # [roll, pitch, yaw]
            r = Rotation.from_euler('xyz', euler)
            q = r.as_quat()  # [x, y, z, w] - already in XYZW format
            pose_vec = [t[0], t[1], t[2], q[0], q[1], q[2], q[3]]
        else:
            # No rotation data - use identity
            pose_vec = [t[0], t[1], t[2], 0, 0, 0, 1]
        
        absolute_poses.append(pose_vec)
    absolute_poses = np.array(absolute_poses, dtype=np.float32)
    
    num_frames = visual_data.shape[0]
    features_list = []
    poses_list = []
    imus_list = []
    
    # Process with sliding window
    for start_idx in range(0, num_frames - window_size + 1, stride):
        end_idx = start_idx + window_size
        
        # Extract window
        window_visual = visual_data[start_idx:end_idx]  # [11, 3, H, W]
        window_imu = imu_data[start_idx:end_idx]        # [11, 33, 6]
        window_absolute_poses = absolute_poses[start_idx:end_idx]  # [11, 7]
        
        # Convert absolute poses to relative poses with FIXED coordinate transformation
        window_relative_poses = convert_absolute_to_relative_fixed(window_absolute_poses)
        
        # Scale translation from meters to centimeters
        window_relative_poses[:, :3] *= pose_scale
        
        # Resize images to 256x512 (model expects this size)
        window_visual_resized = F.interpolate(
            window_visual, 
            size=(256, 512), 
            mode='bilinear', 
            align_corners=False
        )
        
        # Normalize images to [-0.5, 0.5] (currently in [0, 1])
        window_visual_normalized = window_visual_resized - 0.5
        
        # Prepare IMU data - we need 110 samples (10 per frame)
        # Simple approach: take first 10 IMU samples from each frame's 33 samples
        window_imu_110 = []
        for i in range(window_size):
            window_imu_110.append(window_imu[i, :10, :])  # Take first 10 samples
        window_imu_110 = torch.cat(window_imu_110, dim=0)  # [110, 6]
        
        # Add batch dimension
        batch_visual = window_visual_normalized.unsqueeze(0).to(device)  # [1, 11, 3, 256, 512]
        batch_imu = window_imu_110.unsqueeze(0).to(device)              # [1, 110, 6]
        
        # Generate features
        with torch.no_grad():
            v_feat, i_feat = model(batch_visual, batch_imu)  # Both are [1, 10, feature_dim]
            v_feat = v_feat.squeeze(0).cpu()  # [10, 512]
            i_feat = i_feat.squeeze(0).cpu()  # [10, 256]
        
        # Store visual and IMU features separately
        features_list.append((v_feat, i_feat))
        # Store poses for transitions (skip first pose as it's always origin)
        poses_list.append(torch.from_numpy(window_relative_poses[1:]))  # [10, 7]
    
    return features_list, poses_list


def generate_split_data(processed_dir, output_dir, model, device, pose_scale=100.0, stride=1, split_ratios=(0.7, 0.1, 0.2), skip_test=True):
    """Generate train/val/test splits from processed sequences with relative poses."""
    
    # Get all sequence directories (filter out non-numeric directories)
    seq_dirs = sorted([d for d in Path(processed_dir).iterdir() 
                      if d.is_dir() and d.name.isdigit()])
    
    # Use all available sequences
    num_sequences = len(seq_dirs)
    
    print(f"Found {num_sequences} sequences")
    
    # Calculate split sizes
    train_size = int(num_sequences * split_ratios[0])
    val_size = int(num_sequences * split_ratios[1])
    test_size = num_sequences - train_size - val_size
    
    # Split sequences
    train_seqs = seq_dirs[:train_size]
    val_seqs = seq_dirs[train_size:train_size + val_size]
    test_seqs = seq_dirs[train_size + val_size:]
    
    print(f"Split: {len(train_seqs)} train, {len(val_seqs)} val, {len(test_seqs)} test sequences")
    
    # Save split information for consistent evaluation
    split_info = {
        'total_sequences': num_sequences,
        'split_ratios': split_ratios,
        'splits': {
            'train': {
                'indices': list(range(0, train_size)),
                'sequences': [s.name for s in train_seqs],
                'count': len(train_seqs)
            },
            'val': {
                'indices': list(range(train_size, train_size + val_size)),
                'sequences': [s.name for s in val_seqs],
                'count': len(val_seqs)
            },
            'test': {
                'indices': list(range(train_size + val_size, num_sequences)),
                'sequences': [s.name for s in test_seqs],
                'count': len(test_seqs)
            }
        }
    }
    
    # Save split information
    import json
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'dataset_splits.json'), 'w') as f:
        json.dump(split_info, f, indent=2)
    
    # Save test sequences list
    with open(os.path.join(output_dir, 'test_sequences.txt'), 'w') as f:
        f.write('\n'.join([s.name for s in test_seqs]))
    
    print(f"💾 Saved dataset split information to {output_dir}/dataset_splits.json")
    print(f"📝 Test sequences saved to {output_dir}/test_sequences.txt")
    
    # Process each split
    splits = {
        'train': train_seqs,
        'val': val_seqs
    }
    
    # Only process test if not skipping
    if not skip_test:
        splits['test'] = test_seqs
    else:
        print("⏭️  Skipping test set processing (will use real-time encoding during inference)")
    
    sample_counter = {
        'train': 0,
        'val': 0,
        'test': 0  # Initialize even if skipping test
    }
    
    # Track statistics across all samples
    all_rotation_angles = []
    all_translation_norms = []
    
    for split_name, sequences in splits.items():
        print(f"\nProcessing {split_name} split...")
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        
        for seq_dir in tqdm(sequences, desc=f"{split_name} sequences"):
            try:
                # Process sequence with relative poses
                features_list, poses_list = process_sequence(
                    seq_dir, model, device, 
                    window_size=11, stride=stride, pose_scale=pose_scale
                )
                
                # Save each window as a sample
                for (v_feat, i_feat), poses in zip(features_list, poses_list):
                    sample_id = sample_counter[split_name]
                    
                    # Save visual and IMU features separately
                    np.save(os.path.join(split_dir, f"{sample_id}_visual.npy"), v_feat.numpy())  # [10, 512]
                    np.save(os.path.join(split_dir, f"{sample_id}_imu.npy"), i_feat.numpy())     # [10, 256]
                    
                    # Save ground truth poses (now relative poses for transitions)
                    np.save(os.path.join(split_dir, f"{sample_id}_gt.npy"), poses.numpy())
                    
                    # Collect statistics (all poses are now transitions, no identity)
                    for i in range(len(poses)):
                        # Translation norm
                        trans_norm = np.linalg.norm(poses[i, :3].numpy())
                        all_translation_norms.append(trans_norm)
                        
                        # Rotation angle from quaternion (XYZW format)
                        quat = poses[i, 3:].numpy()
                        angle = 2 * np.arccos(np.clip(quat[3], -1, 1)) * 180 / np.pi
                        all_rotation_angles.append(angle)
                    
                    sample_counter[split_name] += 1
                    
            except Exception as e:
                print(f"Error processing {seq_dir}: {e}")
                continue
        
        print(f"Generated {sample_counter[split_name]} samples for {split_name}")
    
    # Print overall statistics
    if all_rotation_angles:
        rotation_angles = np.array(all_rotation_angles)
        translation_norms = np.array(all_translation_norms)
        
        print(f"\n{'='*60}")
        print(f"Dataset Statistics (Relative Poses in Local Coordinates):")
        print(f"{'='*60}")
        print(f"Rotation angles (degrees):")
        print(f"  Mean: {np.mean(rotation_angles):.4f}")
        print(f"  Std:  {np.std(rotation_angles):.4f}")
        print(f"  Max:  {np.max(rotation_angles):.4f}")
        print(f"  95%:  {np.percentile(rotation_angles, 95):.4f}")
        
        print(f"\nTranslation norms (cm):")
        print(f"  Mean: {np.mean(translation_norms):.4f}")
        print(f"  Std:  {np.std(translation_norms):.4f}")
        print(f"  Max:  {np.max(translation_norms):.4f}")
        print(f"  95%:  {np.percentile(translation_norms, 95):.4f}")
    
    return sample_counter


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate all latent features with pretrained model and FIXED relative poses')
    parser.add_argument('--processed-dir', type=str, default='data/aria_processed',
                        help='Directory with processed Aria sequences')
    parser.add_argument('--output-dir', type=str, default='aria_latent_data_pretrained',
                        help='Output directory for latent features')
    parser.add_argument('--model-path', type=str, 
                        default='pretrained_models/vf_512_if_256_3e-05.model',
                        help='Path to pretrained model')
    parser.add_argument('--window-size', type=int, default=11,
                        help='Window size for sequences')
    parser.add_argument('--stride', type=int, default=1,
                        help='Stride for sliding window (default: 1)')
    parser.add_argument('--pose-scale', type=float, default=100.0,
                        help='Scale factor for poses (default: 100.0 for meter to cm conversion)')
    parser.add_argument('--skip-test', action='store_true', default=True,
                        help='Skip test set processing (default: True, use real-time encoding for inference)')
    parser.add_argument('--process-test', action='store_true',
                        help='Process test set (overrides --skip-test)')
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load pretrained model
    print(f"\nLoading pretrained model from {args.model_path}...")
    if not os.path.exists(args.model_path):
        print(f"ERROR: Model file not found at {args.model_path}")
        print("Please run: python download_pretrained_model.py")
        return
    
    model = load_pretrained_model(args.model_path)
    model = model.to(device)
    model.eval()
    
    # Generate features for all splits
    print(f"\nGenerating latent features with FIXED quaternion handling...")
    print(f"Window size: {args.window_size}")
    print(f"Stride: {args.stride}")
    print(f"Pose scale: {args.pose_scale} (meter → cm)")
    print(f"Output directory: {args.output_dir}")
    print(f"Pose format: Relative poses in LOCAL COORDINATES with quaternions")
    
    sample_counts = generate_split_data(
        args.processed_dir, 
        args.output_dir, 
        model, 
        device,
        pose_scale=args.pose_scale,
        stride=args.stride,
        skip_test=not args.process_test  # Skip test unless explicitly requested
    )
    
    # Save metadata
    metadata = {
        'feature_dim': 768,
        'visual_dim': 512,
        'inertial_dim': 256,
        'sequence_length': 10,  # 10 transition features from 11 frames
        'window_size': args.window_size,
        'stride': args.stride,
        'pose_scale': args.pose_scale,
        'pose_format': 'relative_local',  # Changed to indicate local coordinates
        'quaternion_format': 'XYZW',
        'model_path': args.model_path,
        'normalization': '[-0.5, 0.5]',
        'sample_counts': sample_counts,
        'feature_format': 'separate',  # visual and IMU features saved separately
        'note': 'Features are transitions between frames (10 features from 11 frames). Visual and IMU features saved separately for proper transformer input.'
    }
    
    import pickle
    with open(os.path.join(args.output_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"\n✅ Successfully generated latent features!")
    print(f"Total samples:")
    for split, count in sample_counts.items():
        if count > 0:  # Only show splits that were processed
            print(f"  {split}: {count}")
    print(f"\nFeatures saved to: {args.output_dir}")
    
    if not args.process_test:
        print(f"\n📝 Note: Test set was skipped. During inference, use real-time encoding.")
    
    print(f"\nYou can now train your model using:")
    print(f"python train_separate_features.py --data_dir {args.output_dir}")


if __name__ == '__main__':
    main()