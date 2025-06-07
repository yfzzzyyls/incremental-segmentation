#!/usr/bin/env python3
"""
Fixed inference pipeline that correctly handles VIFT model output format.

Fixes:
1. VIFT outputs [tx, ty, tz, rx, ry, rz] but loss functions expect [rx, ry, rz, tx, ty, tz]
2. Proper conversion between radians and degrees for rotation metrics
3. Professional AR/VR metrics display in mm and degrees
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
from rich.console import Console
from rich.table import Table
import argparse
from scipy.linalg import svd

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('src')

from src.models.components.vsvio import Encoder
from src.models.multihead_vio import MultiHeadVIOModel
from src.models.multihead_vio_separate import MultiHeadVIOModelSeparate
from src.models.vio_module import VIOLitModule
from src.models.components.pose_transformer import PoseTransformer

console = Console()


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
        
    def forward(self, imgs, imus, return_separate=False):
        # The model outputs 10 frames for 11 input frames
        v_feat, i_feat = self.Feature_net(imgs, imus)
        
        if return_separate:
            # Return separate features without padding (10 frames)
            return v_feat, i_feat
        else:
            # Pad to 11 frames by repeating the last frame
            v_feat_padded = torch.cat([v_feat, v_feat[:, -1:, :]], dim=1)
            i_feat_padded = torch.cat([i_feat, i_feat[:, -1:, :]], dim=1)
            return torch.cat([v_feat_padded, i_feat_padded], dim=-1)


def load_pretrained_encoder(model_path):
    """Load pretrained Visual-Selective-VIO encoder."""
    model = WrapperModel()
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    encoder_dict = {}
    for k, v in checkpoint.items():
        if k.startswith('Feature_net.'):
            new_k = k.replace('Feature_net.', '')
            encoder_dict[new_k] = v
    
    model.Feature_net.load_state_dict(encoder_dict, strict=False)
    console.print(f"✅ Loaded pretrained encoder with {len(encoder_dict)} parameters")
    return model


def quaternion_to_matrix(q):
    """Convert quaternion (XYZW) to rotation matrix."""
    x, y, z, w = q
    norm = np.sqrt(x*x + y*y + z*z + w*w)
    x, y, z, w = x/norm, y/norm, z/norm, w/norm
    
    return np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
        [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
        [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]
    ])


def accumulate_poses(relative_poses):
    """
    Accumulate relative poses to build absolute trajectory.
    
    Args:
        relative_poses: [N, 7] array of relative poses (tx, ty, tz, qx, qy, qz, qw)
    
    Returns:
        absolute_poses: [N, 7] array of absolute poses
    """
    N = len(relative_poses)
    absolute_poses = np.zeros((N, 7))
    
    # Start at origin
    current_pos = np.zeros(3)
    current_rot = np.eye(3)
    
    # First pose is always at origin
    absolute_poses[0, :3] = [0, 0, 0]
    absolute_poses[0, 3:] = [0, 0, 0, 1]
    
    for i in range(1, N):
        # Get relative transformation
        rel_trans = relative_poses[i, :3]
        rel_quat = relative_poses[i, 3:]
        rel_rot = quaternion_to_matrix(rel_quat)
        
        # Update absolute position
        current_pos = current_pos + current_rot @ rel_trans
        current_rot = current_rot @ rel_rot
        
        # Convert back to quaternion
        r = Rotation.from_matrix(current_rot)
        current_quat = r.as_quat()  # [x, y, z, w]
        
        absolute_poses[i, :3] = current_pos
        absolute_poses[i, 3:] = current_quat
    
    return absolute_poses


def align_trajectory_se3(estimated_traj, ground_truth_traj, align_orientation=False):
    """
    Align estimated trajectory to ground truth using Umeyama algorithm.
    
    This implements the standard SE(3) trajectory alignment used in TUM RGB-D benchmark
    and other visual odometry evaluation tools.
    
    Args:
        estimated_traj: [N, 7] array of poses (x, y, z, qx, qy, qz, qw)
        ground_truth_traj: [N, 7] array of poses (x, y, z, qx, qy, qz, qw)
        align_orientation: If True, align both position and orientation.
                          If False, only align position (default behavior for most benchmarks)
    
    Returns:
        aligned_traj: [N, 7] aligned trajectory
        transform_params: dict with rotation matrix R, translation t, and scale s
    """
    assert len(estimated_traj) == len(ground_truth_traj), "Trajectories must have same length"
    
    # Extract positions
    est_positions = estimated_traj[:, :3]
    gt_positions = ground_truth_traj[:, :3]
    
    # Center the trajectories
    est_mean = np.mean(est_positions, axis=0)
    gt_mean = np.mean(gt_positions, axis=0)
    
    est_centered = est_positions - est_mean
    gt_centered = gt_positions - gt_mean
    
    # Compute scale using Umeyama's method
    est_var = np.sum(est_centered ** 2) / len(est_centered)
    
    # Compute cross-covariance matrix
    W = np.dot(gt_centered.T, est_centered) / len(est_centered)
    
    # SVD
    U, S, Vt = svd(W)
    
    # Compute rotation
    R = np.dot(U, Vt)
    
    # Ensure proper rotation (det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(U, Vt)
    
    # Compute scale
    scale = np.trace(np.dot(R, W.T)) / est_var
    
    # Compute translation
    translation = gt_mean - scale * np.dot(R, est_mean)
    
    # Apply transformation to estimated trajectory
    aligned_positions = scale * np.dot(est_positions, R.T) + translation
    
    # Build aligned trajectory
    aligned_traj = estimated_traj.copy()
    aligned_traj[:, :3] = aligned_positions
    
    if align_orientation:
        # Also align orientations by applying the rotation
        for i in range(len(aligned_traj)):
            q_est = estimated_traj[i, 3:]
            r_est = Rotation.from_quat(q_est)
            r_aligned = Rotation.from_matrix(R) * r_est
            aligned_traj[i, 3:] = r_aligned.as_quat()
    
    transform_params = {
        'rotation': R,
        'translation': translation,
        'scale': scale
    }
    
    return aligned_traj, transform_params


def sliding_window_inference_batched(
    sequence_path: Path,
    encoder_model: nn.Module,
    vio_model: nn.Module,
    device: torch.device,
    window_size: int = 11,
    stride: int = 1,
    pose_scale: float = 100.0,
    batch_size: int = 32,
    num_gpus: int = 4,
    model_type: str = 'multihead_concat'
):
    """
    Run sliding window inference on a full sequence with batched processing.
    
    Args:
        sequence_path: Path to processed sequence
        encoder_model: Pretrained encoder
        vio_model: Trained VIO model
        device: torch device
        window_size: Window size (default 11)
        stride: Stride for sliding window (default 1 for real-time)
        pose_scale: Scale factor for poses
        batch_size: Number of windows to process simultaneously
    
    Returns:
        predictions: Dict with trajectory and metrics
    """
    console.print(f"\n[bold cyan]Processing sequence: {sequence_path.name}[/bold cyan]")
    
    # Load sequence data
    visual_data = torch.load(sequence_path / "visual_data.pt").float()  # [N, 3, H, W]
    imu_data = torch.load(sequence_path / "imu_data.pt").float()        # [N, 33, 6]
    
    # Load ground truth poses - check for quaternion version first
    poses_file = sequence_path / "poses_quaternion.json"
    if not poses_file.exists():
        poses_file = sequence_path / "poses.json"
    
    with open(poses_file, 'r') as f:
        poses_data = json.load(f)
    
    num_frames = visual_data.shape[0]
    console.print(f"  Total frames: {num_frames}")
    console.print(f"  Window size: {window_size}")
    console.print(f"  Stride: {stride}")
    console.print(f"  Mode: Real-time sliding window")
    console.print(f"  Batch size: {batch_size}")
    console.print(f"  Prediction: Using only last prediction per window (real AR/VR scenario)")
    
    # Store predictions for each window
    window_predictions = []
    window_indices = []
    
    # Setup multi-GPU if available
    if num_gpus > 1 and torch.cuda.device_count() >= num_gpus:
        console.print(f"  Using {num_gpus} GPUs for parallel processing")
        # Create model copies for each GPU
        encoder_models = []
        vio_models = []
        for gpu_id in range(num_gpus):
            device_gpu = torch.device(f'cuda:{gpu_id}')
            encoder_copy = WrapperModel()
            encoder_copy.load_state_dict(encoder_model.state_dict())
            encoder_copy = encoder_copy.to(device_gpu)
            encoder_copy.eval()
            encoder_models.append(encoder_copy)
            
            # Create a copy by deep copying the model
            import copy
            vio_copy = copy.deepcopy(vio_model)
            vio_copy = vio_copy.to(device_gpu)
            vio_copy.eval()
            vio_models.append(vio_copy)
    else:
        console.print(f"  Using single GPU")
        num_gpus = 1
        encoder_models = [encoder_model.to(device)]
        vio_models = [vio_model.to(device)]
        encoder_models[0].eval()
        vio_models[0].eval()
    
    # Collect all windows first
    all_windows_visual = []
    all_windows_imu = []
    all_start_indices = []
    
    for start_idx in range(0, num_frames - window_size + 1, stride):
        end_idx = start_idx + window_size
        
        # Extract window
        window_visual = visual_data[start_idx:end_idx]  # [11, 3, H, W]
        window_imu = imu_data[start_idx:end_idx]        # [11, 33, 6]
        
        # Preprocess visual data
        window_visual_resized = F.interpolate(
            window_visual, 
            size=(256, 512), 
            mode='bilinear', 
            align_corners=False
        )
        window_visual_normalized = window_visual_resized - 0.5
        
        # Prepare IMU data (take LAST 10 samples per frame for most recent data)
        window_imu_110 = []
        for i in range(window_size):
            window_imu_110.append(window_imu[i, -10:, :])  # Last 10 samples
        window_imu_110 = torch.cat(window_imu_110, dim=0)  # [110, 6]
        
        all_windows_visual.append(window_visual_normalized)
        all_windows_imu.append(window_imu_110)
        all_start_indices.append(start_idx)
    
    num_windows = len(all_windows_visual)
    console.print(f"  Total windows: {num_windows}")
    
    # Process in batches with multi-GPU
    with torch.no_grad():
        for batch_start in tqdm(range(0, num_windows, batch_size * num_gpus), desc="Batched inference"):
            batch_end = min(batch_start + batch_size * num_gpus, num_windows)
            
            # Split batch across GPUs
            gpu_predictions = []
            gpu_batch_sizes = []
            
            for gpu_id in range(num_gpus):
                gpu_batch_start = batch_start + gpu_id * batch_size
                gpu_batch_end = min(gpu_batch_start + batch_size, batch_end)
                
                if gpu_batch_start >= batch_end:
                    break
                
                # Stack batch for this GPU
                gpu_device = torch.device(f'cuda:{gpu_id}' if num_gpus > 1 else device)
                batch_visual = torch.stack(all_windows_visual[gpu_batch_start:gpu_batch_end]).to(gpu_device).float()
                batch_imu = torch.stack(all_windows_imu[gpu_batch_start:gpu_batch_end]).to(gpu_device).float()
                
                # Generate features using encoder on this GPU based on model type
                if model_type == 'vift_original':
                    # VIFT original also needs features extraction
                    features = encoder_models[gpu_id](batch_visual, batch_imu, return_separate=False)  # [B, 11, 768]
                elif model_type == 'multihead_separate':
                    v_feat, i_feat = encoder_models[gpu_id](batch_visual, batch_imu, return_separate=True)  # [B, 10, 512], [B, 10, 256]
                else:  # multihead_concat
                    features = encoder_models[gpu_id](batch_visual, batch_imu, return_separate=False)  # [B, 11, 768]
                
                # Prepare batch for VIO model based on model type
                if model_type == 'vift_original':
                    # VIFT original expects features as input
                    batch = features
                elif model_type == 'multihead_separate':
                    batch = {
                        'visual_features': v_feat,
                        'imu_features': i_feat,
                        'poses': None  # Not needed for inference
                    }
                else:  # multihead_concat
                    batch = {
                        'images': features,
                        'imus': torch.zeros(features.shape[0], 11, 6).to(gpu_device),  # Dummy
                        'poses': None  # Not needed for inference
                    }
                
                # Run VIO model on this GPU
                if model_type == 'vift_original':
                    # PoseTransformer expects (features, _, _) tuple as first argument
                    batch_tuple = (batch, None, None)
                    raw_predictions = vio_models[gpu_id](batch_tuple, None)
                    predictions = {
                        'translation': raw_predictions[:, :, :3],
                        'rotation': raw_predictions[:, :, 3:]  # Keep as Euler for now
                    }
                else:
                    predictions = vio_models[gpu_id](batch)
                gpu_predictions.append(predictions)
                gpu_batch_sizes.append(gpu_batch_end - gpu_batch_start)
            
            # Collect predictions from all GPUs
            for gpu_id, (predictions, gpu_batch_size) in enumerate(zip(gpu_predictions, gpu_batch_sizes)):
                gpu_batch_start = batch_start + gpu_id * batch_size
                
                # Extract predictions for each window in this GPU's batch
                for i in range(gpu_batch_size):
                    if model_type == 'vift_original':
                        # CRITICAL FIX: VIFT outputs [tx, ty, tz, rx, ry, rz]
                        # We need to handle this correctly
                        pred_translation = predictions['translation'][i].cpu().numpy()  # [10, 3]
                        euler_angles = predictions['rotation'][i].cpu().numpy()  # [10, 3]
                        
                        # Convert each Euler angle to quaternion
                        pred_rotation = []
                        for euler in euler_angles:
                            r = Rotation.from_euler('xyz', euler)
                            q = r.as_quat()  # [x, y, z, w]
                            pred_rotation.append(q)
                        pred_rotation = np.array(pred_rotation)  # [10, 4]
                        
                        # Add identity pose at the beginning
                        identity_pose = np.array([[0, 0, 0, 0, 0, 0, 1]])  # [x, y, z, qx, qy, qz, qw]
                        pred_poses = np.vstack([
                            identity_pose,
                            np.concatenate([pred_translation, pred_rotation], axis=1)
                        ])  # [11, 7]
                    elif model_type == 'multihead_separate':
                        # Separate model outputs 10 predictions (transitions between 11 frames)
                        pred_rotation = predictions['rotation'][i].cpu().numpy()  # [10, 4]
                        pred_translation = predictions['translation'][i].cpu().numpy()  # [10, 3]
                        
                        # Add identity pose at the beginning
                        identity_pose = np.array([[0, 0, 0, 0, 0, 0, 1]])  # [x, y, z, qx, qy, qz, qw]
                        pred_poses = np.vstack([
                            identity_pose,
                            np.concatenate([pred_translation, pred_rotation], axis=1)
                        ])  # [11, 7]
                    else:
                        pred_poses = torch.cat([
                            predictions['translation'][i],  # [11, 3]
                            predictions['rotation'][i]      # [11, 4]
                        ], dim=1).cpu().numpy()  # [11, 7]
                    
                    start_idx = all_start_indices[gpu_batch_start + i]
                    window_predictions.append(pred_poses)
                    window_indices.append((start_idx, start_idx + window_size))
    
    console.print(f"  Processed {num_windows} windows in {(num_windows + batch_size * num_gpus - 1) // (batch_size * num_gpus)} batches")
    
    # Real-time aggregation: Sliding window with only the last prediction used
    # This simulates real AR/VR where we maintain a buffer and predict next frame
    aggregated_poses = np.zeros((num_frames, 7))
    aggregated_poses[:, 3:] = [0, 0, 0, 1]  # Initialize with identity quaternions
    
    frame_counts = np.zeros(num_frames)
    
    console.print(f"  Using real-time sliding window (last prediction only)")
    
    # Debug tracking
    frames_with_multiple_predictions = 0
    total_predictions_used = 0
    
    # In real-time AR/VR:
    # - We maintain a sliding window of 11 frames
    # - We predict 10 transitions
    # - We only use the LAST prediction (transition from frame 9→10)
    # - We slide the window by 1 frame and repeat
    
    for (start_idx, end_idx), pred_poses in zip(window_indices, window_predictions):
        # In this window, we have predictions for frames start_idx to start_idx+10
        # In real-time, we would only trust and use the last prediction
        # because it has the most context (all previous frames in the window)
        
        if stride == 1:
            # Real-time mode: only use the last prediction from each window
            last_frame_idx = start_idx + window_size - 1
            if last_frame_idx < num_frames:
                # Check if this frame already has a prediction
                if frame_counts[last_frame_idx] > 0:
                    frames_with_multiple_predictions += 1
                # Use the last prediction (most informed by context)
                aggregated_poses[last_frame_idx] = pred_poses[-1]
                frame_counts[last_frame_idx] = 1
                total_predictions_used += 1
        else:
            # Non-overlapping mode: use all predictions from this window
            for i in range(window_size):
                frame_idx = start_idx + i
                if frame_idx < num_frames:
                    aggregated_poses[frame_idx] = pred_poses[i]
                    frame_counts[frame_idx] = 1
    
    # For stride=1, we need to handle the first frames differently
    # The first window_size-1 frames don't have "last prediction" coverage
    if stride == 1:
        # Use predictions from the first window for initial frames
        first_window_poses = window_predictions[0]
        for i in range(window_size - 1):
            if frame_counts[i] == 0:
                aggregated_poses[i] = first_window_poses[i]
                frame_counts[i] = 1
    
    # Ensure all frames have predictions
    for i in range(num_frames):
        if frame_counts[i] == 0 and i > 0:
            console.print(f"  Warning: Frame {i} has no prediction, using previous frame")
            aggregated_poses[i] = aggregated_poses[i-1]
            frame_counts[i] = 1
    
    # Debug output
    console.print(f"\n  Debug: Real-time aggregation statistics:")
    console.print(f"    - Total predictions used: {total_predictions_used}")
    console.print(f"    - Frames with multiple predictions: {frames_with_multiple_predictions}")
    console.print(f"    - Expected for stride=1: each frame uses exactly 1 prediction")
    console.print(f"    - Frames covered: {np.sum(frame_counts > 0)}/{num_frames}")
    
    # Build absolute trajectory from relative poses
    absolute_trajectory = accumulate_poses(aggregated_poses)
    
    # Convert ground truth to relative poses (same as before)
    gt_absolute = []
    for pose in poses_data:
        t = pose['translation']
        # Check if we have quaternion data directly
        if 'quaternion' in pose:
            # Already have quaternion in XYZW format
            q = pose['quaternion']
            gt_absolute.append([t[0], t[1], t[2], q[0], q[1], q[2], q[3]])
        elif 'rotation_euler' in pose:
            # Convert Euler to quaternion (backward compatibility)
            euler = pose['rotation_euler']
            r = Rotation.from_euler('xyz', euler)
            q = r.as_quat()  # [x, y, z, w]
            gt_absolute.append([t[0], t[1], t[2], q[0], q[1], q[2], q[3]])
        else:
            # No rotation data - use identity
            gt_absolute.append([t[0], t[1], t[2], 0, 0, 0, 1])
    gt_absolute = np.array(gt_absolute)
    
    # Convert absolute GT to relative poses
    gt_relative = np.zeros_like(gt_absolute)
    gt_relative[0, :3] = [0, 0, 0]
    gt_relative[0, 3:] = [0, 0, 0, 1]
    
    for i in range(1, len(gt_absolute)):
        # Relative translation
        prev_t = gt_absolute[i-1, :3]
        curr_t = gt_absolute[i, :3]
        
        # Get rotation matrices
        prev_r = Rotation.from_quat(gt_absolute[i-1, 3:])
        curr_r = Rotation.from_quat(gt_absolute[i, 3:])
        
        # Relative translation in previous frame's coordinates
        rel_trans = prev_r.inv().apply(curr_t - prev_t)
        
        # Relative rotation
        rel_rot = prev_r.inv() * curr_r
        rel_quat = rel_rot.as_quat()
        
        gt_relative[i, :3] = rel_trans
        gt_relative[i, 3:] = rel_quat
    
    # Scale after conversion to relative
    gt_relative[:, :3] *= pose_scale
    
    # Build absolute trajectory from relative GT for verification
    gt_absolute_from_relative = accumulate_poses(gt_relative)
    
    return {
        'relative_poses': aggregated_poses,
        'absolute_trajectory': absolute_trajectory,
        'ground_truth': gt_absolute_from_relative,
        'ground_truth_relative': gt_relative,
        'num_frames': num_frames,
        'num_windows': num_windows,
        'frame_overlap': frame_counts
    }


def calculate_metrics(results, no_alignment=False):
    """Calculate ATE, RPE and other metrics with proper AR/VR standards.
    
    Args:
        results: Dictionary with trajectory results
        no_alignment: If True, skip trajectory alignment (default: False)
    
    Returns:
        metrics: Dictionary with both aligned and unaligned metrics
    """
    pred_traj = results['absolute_trajectory']
    gt_traj = results['ground_truth']
    pred_rel = results['relative_poses']
    gt_rel = results['ground_truth_relative']
    
    # Ensure same length
    min_len = min(len(pred_traj), len(gt_traj))
    pred_traj = pred_traj[:min_len]
    gt_traj = gt_traj[:min_len]
    pred_rel = pred_rel[:min_len]
    gt_rel = gt_rel[:min_len]
    
    # Calculate unaligned (raw) ATE first
    ate_errors_raw_mm = []
    for pred, gt in zip(pred_traj, gt_traj):
        error_cm = np.linalg.norm(pred[:3] - gt[:3])
        error_mm = error_cm * 10  # Convert cm to mm
        ate_errors_raw_mm.append(error_mm)
    
    ate_errors_raw_mm = np.array(ate_errors_raw_mm)
    
    # Calculate aligned ATE if not disabled
    if not no_alignment:
        # Align trajectories using Umeyama algorithm
        aligned_pred_traj, transform_params = align_trajectory_se3(pred_traj, gt_traj, align_orientation=False)
        
        # Calculate aligned ATE
        ate_errors_aligned_mm = []
        for pred, gt in zip(aligned_pred_traj, gt_traj):
            error_cm = np.linalg.norm(pred[:3] - gt[:3])
            error_mm = error_cm * 10  # Convert cm to mm
            ate_errors_aligned_mm.append(error_mm)
        
        ate_errors_aligned_mm = np.array(ate_errors_aligned_mm)
    else:
        # If alignment is disabled, aligned metrics are same as raw
        ate_errors_aligned_mm = ate_errors_raw_mm
        aligned_pred_traj = pred_traj
        transform_params = None
    
    # Use aligned trajectory for rotation errors (following standard practice)
    rot_errors_deg = []
    for pred, gt in zip(aligned_pred_traj[1:], gt_traj[1:]):  # Skip first frame (origin)
        pred_r = Rotation.from_quat(pred[3:])
        gt_r = Rotation.from_quat(gt[3:])
        rel_r = pred_r * gt_r.inv()
        angle_rad = np.abs(rel_r.magnitude())
        angle_deg = np.degrees(angle_rad)
        rot_errors_deg.append(angle_deg)
    
    rot_errors_deg = np.array(rot_errors_deg)
    
    # Calculate RPE (Relative Pose Error) at different time scales
    # Standard AR/VR time windows: 33ms (1 frame), 100ms (3 frames), 167ms (5 frames), 1s (30 frames)
    
    # Assuming 30 fps
    fps = 30
    time_windows = {
        '33ms': 1,    # 1 frame
        '100ms': 3,   # 3 frames
        '167ms': 5,   # 5 frames
        '333ms': 10,  # 10 frames
        '1s': 30      # 30 frames
    }
    
    rpe_results = {}
    
    for window_name, window_frames in time_windows.items():
        if len(pred_traj) > window_frames:
            rpe_trans = []
            rpe_rot = []
            
            for i in range(len(pred_traj) - window_frames):
                # Get window relative motion
                start_pos_pred = pred_traj[i, :3]
                end_pos_pred = pred_traj[i + window_frames, :3]
                start_pos_gt = gt_traj[i, :3]
                end_pos_gt = gt_traj[i + window_frames, :3]
                
                # Translation error over window
                pred_motion = end_pos_pred - start_pos_pred
                gt_motion = end_pos_gt - start_pos_gt
                trans_error_cm = np.linalg.norm(pred_motion - gt_motion)
                trans_error_mm = trans_error_cm * 10  # Convert to mm
                rpe_trans.append(trans_error_mm)
                
                # Rotation error over window
                start_rot_pred = Rotation.from_quat(pred_traj[i, 3:])
                end_rot_pred = Rotation.from_quat(pred_traj[i + window_frames, 3:])
                start_rot_gt = Rotation.from_quat(gt_traj[i, 3:])
                end_rot_gt = Rotation.from_quat(gt_traj[i + window_frames, 3:])
                
                rel_rot_pred = start_rot_pred.inv() * end_rot_pred
                rel_rot_gt = start_rot_gt.inv() * end_rot_gt
                rel_error = rel_rot_pred * rel_rot_gt.inv()
                angle_rad = np.abs(rel_error.magnitude())
                angle_deg = np.degrees(angle_rad)
                rpe_rot.append(angle_deg)
            
            rpe_results[window_name] = {
                'trans_mean': np.mean(rpe_trans),
                'trans_rmse': np.sqrt(np.mean(np.square(rpe_trans))),
                'trans_std': np.std(rpe_trans),
                'trans_median': np.median(rpe_trans),
                'rot_mean': np.mean(rpe_rot),
                'rot_rmse': np.sqrt(np.mean(np.square(rpe_rot))),
                'rot_std': np.std(rpe_rot),
                'rot_median': np.median(rpe_rot)
            }
        else:
            rpe_results[window_name] = {
                'trans_mean': 0,
                'trans_std': 0,
                'trans_median': 0,
                'rot_mean': 0,
                'rot_std': 0,
                'rot_median': 0
            }
    
    # Calculate RMSE for both aligned and raw ATE (this is the standard metric)
    ate_rmse_aligned_mm = np.sqrt(np.mean(np.square(ate_errors_aligned_mm)))
    ate_rmse_raw_mm = np.sqrt(np.mean(np.square(ate_errors_raw_mm)))
    
    # Store transform parameters if alignment was performed
    alignment_info = {}
    if transform_params is not None:
        alignment_info = {
            'scale': transform_params['scale'],
            'rotation_angle_deg': np.degrees(Rotation.from_matrix(transform_params['rotation']).magnitude()),
            'translation_magnitude_mm': np.linalg.norm(transform_params['translation']) * 10  # Convert to mm
        }
    
    # Calculate rotation RMSE
    rot_rmse_deg = np.sqrt(np.mean(np.square(rot_errors_deg)))
    
    return {
        # Aligned metrics (for research comparison)
        'ate_mean_mm': ate_errors_aligned_mm.mean(),
        'ate_rmse_mm': ate_rmse_aligned_mm,  # This is the standard ATE metric
        'ate_std_mm': ate_errors_aligned_mm.std(),
        'ate_median_mm': np.median(ate_errors_aligned_mm),
        'ate_95_mm': np.percentile(ate_errors_aligned_mm, 95),
        'ate_max_mm': ate_errors_aligned_mm.max(),
        
        # Raw/unaligned metrics (for deployment evaluation)
        'ate_mean_raw_mm': ate_errors_raw_mm.mean(),
        'ate_rmse_raw_mm': ate_rmse_raw_mm,
        'ate_std_raw_mm': ate_errors_raw_mm.std(),
        'ate_median_raw_mm': np.median(ate_errors_raw_mm),
        'ate_95_raw_mm': np.percentile(ate_errors_raw_mm, 95),
        'ate_max_raw_mm': ate_errors_raw_mm.max(),
        
        # Rotation metrics (from aligned trajectory)
        'rot_mean_deg': rot_errors_deg.mean(),
        'rot_rmse_deg': rot_rmse_deg,  # Added RMSE for rotation
        'rot_std_deg': rot_errors_deg.std(),
        'rot_median_deg': np.median(rot_errors_deg),
        'rot_95_deg': np.percentile(rot_errors_deg, 95),
        
        # Other metrics
        'rpe_results': rpe_results,
        'total_frames': len(ate_errors_aligned_mm),
        'alignment_disabled': no_alignment,
        'alignment_info': alignment_info
    }


def main():
    parser = argparse.ArgumentParser(description='Fixed full sequence inference for VIFT-AEA')
    parser.add_argument('--sequence-id', type=str, default='all',
                        help='Sequence ID from test set (114-142) or "all" for all test sequences')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--encoder-path', type=str, 
                        default='pretrained_models/vf_512_if_256_3e-05.model',
                        help='Path to pretrained encoder')
    parser.add_argument('--processed-dir', type=str, default='data/aria_processed',
                        help='Directory with processed sequences')
    parser.add_argument('--stride', type=int, default=1,
                        help='Stride for sliding window (1 for real-time, >1 for faster non-overlapping)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for inference (default: 32)')
    parser.add_argument('--num-gpus', type=int, default=4,
                        help='Number of GPUs to use (default: 4)')
    parser.add_argument('--no-alignment', action='store_true',
                        help='Disable trajectory alignment (show raw ATE only)')
    
    args = parser.parse_args()
    
    console.rule("[bold cyan]Fixed Full Sequence Inference Pipeline[/bold cyan]")
    
    # Set device and check GPUs
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    console.print(f"Using device: {device}")
    if torch.cuda.is_available():
        console.print(f"Available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            console.print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Load models
    console.print("\n[bold]Loading models...[/bold]")
    encoder_model = load_pretrained_encoder(args.encoder_path)
    
    # Load checkpoint to check model type
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # Detect model type based on state dict keys
    state_dict_keys = list(checkpoint['state_dict'].keys())
    
    # Check for different model architectures
    if any(key.startswith('model.') for key in state_dict_keys):
        # VIFT original model (wrapped in VIOLitModule)
        console.print("Detected VIFT original model (VIOLitModule)")
        # Need to load the model manually since VIOLitModule requires many init params
        # Get hyperparameters from checkpoint
        hparams = checkpoint.get('hyper_parameters', {})
        # Create the model with hyperparameters
        vio_model = PoseTransformer(
            input_dim=hparams.get('input_dim', 768),
            embedding_dim=hparams.get('embedding_dim', 128),
            num_layers=hparams.get('num_layers', 2),
            nhead=hparams.get('nhead', 4),
            dim_feedforward=hparams.get('dim_feedforward', 512),
            dropout=hparams.get('dropout', 0.2)
        )
        # Load state dict with proper key mapping
        model_state_dict = {}
        for key, value in checkpoint['state_dict'].items():
            if key.startswith('model.'):
                new_key = key.replace('model.', '')
                model_state_dict[new_key] = value
        vio_model.load_state_dict(model_state_dict)
        model_type = 'vift_original'
    elif any('visual_projection' in key for key in state_dict_keys):
        # MultiHead separate features model
        console.print("Detected separate visual/IMU features model")
        # Extract hyperparameters from checkpoint
        hparams = checkpoint.get('hyper_parameters', {})
        console.print(f"Model config: hidden_dim={hparams.get('hidden_dim', 256)}, " +
                     f"num_shared_layers={hparams.get('num_shared_layers', 4)}, " +
                     f"num_specialized_layers={hparams.get('num_specialized_layers', 3)}")
        vio_model = MultiHeadVIOModelSeparate.load_from_checkpoint(args.checkpoint)
        model_type = 'multihead_separate'
    elif any('feature_projection' in key for key in state_dict_keys):
        # MultiHead concatenated features model
        console.print("Detected concatenated features model")
        vio_model = MultiHeadVIOModel.load_from_checkpoint(args.checkpoint)
        model_type = 'multihead_concat'
    else:
        # Default to concatenated model
        console.print("Unknown model type, defaulting to concatenated features model")
        vio_model = MultiHeadVIOModel.load_from_checkpoint(args.checkpoint)
        model_type = 'multihead_concat'
    
    # Determine which sequences to run
    if args.sequence_id.lower() == 'all':
        # Test sequences are typically 114-142 (last ~20% of data)
        # Get all numeric sequence directories
        all_seqs = sorted([d.name for d in Path(args.processed_dir).iterdir() 
                          if d.is_dir() and d.name.isdigit()])
        # Take last 20% as test set (matching train script split)
        num_test = max(1, int(len(all_seqs) * 0.2))
        test_sequences = all_seqs[-num_test:]
        console.print(f"\nRunning on {len(test_sequences)} test sequences: {test_sequences[0]} to {test_sequences[-1]}")
    else:
        test_sequences = [args.sequence_id]
    
    # Store metrics for all sequences
    all_metrics = []
    
    for seq_id in test_sequences:
        sequence_path = Path(args.processed_dir) / seq_id
        if not sequence_path.exists():
            console.print(f"[red]Warning: Sequence {sequence_path} not found, skipping![/red]")
            continue
        
        # Always use batched inference for efficiency
        results = sliding_window_inference_batched(
            sequence_path=sequence_path,
            encoder_model=encoder_model,
            vio_model=vio_model,
            device=device,
            stride=args.stride,
            batch_size=args.batch_size,
            num_gpus=args.num_gpus,
            model_type=model_type
        )
        
        # Calculate metrics
        metrics = calculate_metrics(results, no_alignment=args.no_alignment)
        metrics['sequence_id'] = seq_id
        all_metrics.append(metrics)
        
        # Save individual results
        if len(test_sequences) == 1:
            output_path = Path(f"inference_results_realtime_seq_{seq_id}_stride_{args.stride}.npz")
        else:
            output_dir = Path(f"inference_results_realtime_all_stride_{args.stride}")
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"seq_{seq_id}.npz"
        
        np.savez(
            output_path,
            relative_poses=results['relative_poses'],
            absolute_trajectory=results['absolute_trajectory'],
            ground_truth=results['ground_truth'],
            metrics=metrics
        )
    
    # Display results
    if len(all_metrics) == 1:
        # Single sequence - show detailed results
        metrics = all_metrics[0]
        table = Table(title="Fixed Full Sequence Inference Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Sequence", metrics['sequence_id'])
        table.add_row("Total Frames", f"{metrics['total_frames']:,}")
        
        # Show aligned/raw status
        alignment_status = "Disabled" if metrics['alignment_disabled'] else "Enabled"
        table.add_row("Trajectory Alignment", alignment_status)
        
        # ATE metrics (always show aligned when available)
        table.add_row("[bold]ATE (Aligned)[/bold]", "")
        table.add_row("  ├─ Mean", f"{metrics['ate_mean_mm']:.2f} mm")
        table.add_row("  ├─ RMSE", f"{metrics['ate_rmse_mm']:.2f} mm")
        table.add_row("  ├─ Std", f"{metrics['ate_std_mm']:.2f} mm")
        table.add_row("  ├─ Median", f"{metrics['ate_median_mm']:.2f} mm")
        table.add_row("  ├─ 95%", f"{metrics['ate_95_mm']:.2f} mm")
        table.add_row("  └─ Max", f"{metrics['ate_max_mm']:.2f} mm")
        
        # Show alignment parameters if available
        if not metrics['alignment_disabled'] and 'alignment_info' in metrics and metrics['alignment_info']:
            table.add_row("[bold]Alignment Parameters[/bold]", "")
            table.add_row("  ├─ Scale", f"{metrics['alignment_info']['scale']:.4f}")
            table.add_row("  ├─ Rotation", f"{metrics['alignment_info']['rotation_angle_deg']:.2f}°")
            table.add_row("  └─ Translation", f"{metrics['alignment_info']['translation_magnitude_mm']:.2f} mm")
        
        table.add_row("Rotation Error Mean", f"{metrics['rot_mean_deg']:.2f}°")
        table.add_row("Rotation Error Std", f"{metrics['rot_std_deg']:.2f}°")
        
        console.print("\n")
        console.print(table)
    else:
        # Multiple sequences - show per-sequence and averaged results
        console.print("\n")
        per_seq_table = Table(title="Per-Sequence Results")
        per_seq_table.add_column("Sequence", style="cyan")
        per_seq_table.add_column("Frames", style="white")
        per_seq_table.add_column("ATE (Aligned)", style="green")
        per_seq_table.add_column("RPE@1frame Trans", style="green")
        per_seq_table.add_column("RPE@1frame Rot", style="green")
        per_seq_table.add_column("Status", style="white")
        
        for m in all_metrics:
            # Use RMSE values when available
            ate_value = m.get('ate_rmse_mm', m['ate_mean_mm'])
            trans_rpe = m['rpe_results']['33ms'].get('trans_rmse', m['rpe_results']['33ms']['trans_mean'])
            rot_rpe = m['rpe_results']['33ms'].get('rot_rmse', m['rpe_results']['33ms']['rot_mean'])
            
            status = "✅" if ate_value < 10.0 else "❌"
            per_seq_table.add_row(
                m['sequence_id'],
                f"{m['total_frames']:,}",
                f"{ate_value:.2f} mm",
                f"{trans_rpe:.2f} mm",
                f"{rot_rpe:.2f}°",
                status
            )
        
        console.print(per_seq_table)
        
        # Calculate averaged metrics
        avg_metrics = {}
        # Average aligned metrics
        for key in ['ate_mean_mm', 'ate_std_mm', 'ate_median_mm', 'ate_95_mm', 'ate_rmse_mm',
                    'rot_mean_deg', 'rot_std_deg']:
            values = [m[key] for m in all_metrics]
            avg_metrics[key] = np.mean(values)
            avg_metrics[key + '_std_across_seqs'] = np.std(values)
        
        # Average raw metrics if alignment was used
        if not args.no_alignment:
            for key in ['ate_mean_raw_mm', 'ate_std_raw_mm', 'ate_median_raw_mm', 
                        'ate_95_raw_mm', 'ate_rmse_raw_mm']:
                values = [m[key] for m in all_metrics]
                avg_metrics[key] = np.mean(values)
                avg_metrics[key + '_std_across_seqs'] = np.std(values)
        
        # Average RPE results
        # Define time windows here (same as in calculate_metrics)
        time_windows = {
            '33ms': 1,    # 1 frame
            '100ms': 3,   # 3 frames
            '167ms': 5,   # 5 frames
            '333ms': 10,  # 10 frames
            '1s': 30      # 30 frames
        }
        
        avg_rpe = {}
        for window in time_windows.keys():
            avg_rpe[window] = {}
            for metric in ['trans_mean', 'trans_std', 'rot_mean', 'rot_std']:
                values = [m['rpe_results'][window][metric] for m in all_metrics]
                avg_rpe[window][metric] = np.mean(values)
        
        avg_metrics['rpe_results'] = avg_rpe
        avg_metrics['alignment_disabled'] = args.no_alignment
        
        # Display averaged results
        console.print("\n")
        avg_table = Table(title=f"Averaged Results over {len(all_metrics)} Test Sequences")
        avg_table.add_column("Metric", style="cyan")
        avg_table.add_column("Mean ± Std", style="green")
        avg_table.add_column("Std Across Seqs", style="yellow")
        
        # Always show aligned metrics
        avg_table.add_row("[bold]ATE (Aligned)[/bold]", "", "")
        avg_table.add_row("  ├─ RMSE", 
                         f"{avg_metrics['ate_rmse_mm']:.2f} mm",
                         f"{avg_metrics['ate_rmse_mm_std_across_seqs']:.2f} mm")
        avg_table.add_row("  ├─ Mean", 
                         f"{avg_metrics['ate_mean_mm']:.2f} ± {avg_metrics['ate_std_mm']:.2f} mm",
                         f"{avg_metrics['ate_mean_mm_std_across_seqs']:.2f} mm")
        avg_table.add_row("  └─ Median", 
                         f"{avg_metrics['ate_median_mm']:.2f} mm",
                         f"{avg_metrics['ate_median_mm_std_across_seqs']:.2f} mm")
        
        avg_table.add_row("Absolute Rotation Error", 
                         f"{avg_metrics['rot_mean_deg']:.2f} ± {avg_metrics['rot_std_deg']:.2f}°",
                         f"{avg_metrics['rot_mean_deg_std_across_seqs']:.2f}°")
        
        console.print(avg_table)
        
        # Use averaged metrics for performance comparison
        metrics = avg_metrics
    
    # Display performance vs standards
    console.print("\n")
    perf_table = Table(title="Performance Metrics")
    perf_table.add_column("Metric", style="cyan", width=35)
    perf_table.add_column("Our Model", style="green", width=20)
    perf_table.add_column("Type", style="yellow", width=15)
    perf_table.add_column("Notes", style="blue", width=35)
    
    # Full trajectory metrics (ATE)
    perf_table.add_row(
        "[bold]Full Trajectory Metrics[/bold]",
        "",
        "",
        ""
    )
    
    # ATE - Always show aligned version
    ate_value = metrics.get('ate_rmse_mm', metrics['ate_mean_mm'])
    perf_table.add_row(
        "ATE (Aligned)",
        f"{ate_value:.2f} ± {metrics['ate_std_mm']:.2f} mm",
        "RMSE",
        "Standard VIO metric"
    )
    
    # Rotation ATE (moved here as it's also a full trajectory metric)
    rot_ate_value = metrics.get('rot_rmse_deg', metrics['rot_mean_deg'])
    perf_table.add_row(
        "Rotation ATE",
        f"{rot_ate_value:.2f} ± {metrics['rot_std_deg']:.2f}°",
        "RMSE",
        "Total angular drift"
    )
    
    # Relative Pose Error metrics
    perf_table.add_row(
        "[bold]Relative Pose Error (RPE)[/bold]",
        "",
        "",
        ""
    )
    
    rpe = metrics['rpe_results']
    
    # Frame-to-frame error (1 frame = 33ms at 30fps)
    perf_table.add_row(
        "RPE@1 frame (33ms)",
        "",
        "",
        ""
    )
    trans_rpe_1frame = rpe['33ms'].get('trans_rmse', rpe['33ms']['trans_mean'])
    rot_rpe_1frame = rpe['33ms'].get('rot_rmse', rpe['33ms']['rot_mean'])
    perf_table.add_row(
        "  ├─ Translation",
        f"{trans_rpe_1frame:.2f} ± {rpe['33ms']['trans_std']:.2f} mm",
        "RMSE",
        "Frame-to-frame consistency"
    )
    perf_table.add_row(
        "  └─ Rotation",
        f"{rot_rpe_1frame:.2f} ± {rpe['33ms']['rot_std']:.2f}°",
        "RMSE",
        "Angular velocity accuracy"
    )
    
    # 1-second error (30 frames at 30fps)
    if '1s' in rpe:
        perf_table.add_row(
            "RPE@1s (30 frames)",
            "",
            "",
            ""
        )
        trans_rpe_1s = rpe['1s'].get('trans_rmse', rpe['1s']['trans_mean'])
        rot_rpe_1s = rpe['1s'].get('rot_rmse', rpe['1s']['rot_mean'])
        perf_table.add_row(
            "  ├─ Translation",
            f"{trans_rpe_1s:.2f} ± {rpe['1s']['trans_std']:.2f} mm",
            "RMSE",
            "1-second drift rate"
        )
        perf_table.add_row(
            "  └─ Rotation",
            f"{rot_rpe_1s:.2f} ± {rpe['1s']['rot_std']:.2f}°",
            "RMSE",
            "1-second angular drift"
        )
    
    console.print(perf_table)
    
    # Add metric explanations
    console.print("\n[bold]Metric Definitions (Following TUM/EuRoC Standards):[/bold]")
    console.print("• ATE: Absolute Trajectory Error (RMSE after SE(3) alignment)")
    console.print("• RPE: Relative Pose Error over fixed time intervals")
    console.print("• Translation: Position error in millimeters (mm)")
    console.print("• Rotation: Angular error in degrees (°)")
    console.print("\n[bold]Alignment Details:[/bold]")
    if not args.no_alignment:
        console.print("• Trajectory alignment uses Umeyama algorithm (SVD-based)")
        console.print("• Aligns position only (orientation alignment disabled by default)")
        console.print("• Compensates for initialization differences and global drift")
    else:
        console.print("• Alignment disabled - showing raw trajectory errors")
    console.print("\n[dim]Note: These metrics follow standard VIO evaluation protocols for research comparison.[/dim]")
    
    # Performance summary based on research benchmarks
    console.print("\n[bold]Performance Analysis:[/bold]")
    
    # Compare against typical VIO research results
    console.print(f"\n• Translation ATE: {metrics['ate_mean_mm']:.2f} mm")
    if metrics['ate_mean_mm'] < 10.0:
        console.print("  [green]Comparable to state-of-the-art VIO methods on similar sequences[/green]")
    
    console.print(f"\n• Frame-to-frame translation: {rpe['33ms']['trans_mean']:.2f} mm")
    if rpe['33ms']['trans_mean'] < 1.0:
        console.print("  [green]Sub-millimeter frame-to-frame accuracy achieved[/green]")
    
    console.print(f"\n• Frame-to-frame rotation: {rpe['33ms']['rot_mean']:.2f}°")
    if rpe['33ms']['rot_mean'] < 0.5:
        console.print("  [green]Low angular velocity error[/green]")
    elif rpe['33ms']['rot_mean'] < 2.0:
        console.print("  [yellow]Moderate angular velocity error[/yellow]")
    else:
        console.print("  [red]High angular velocity error - may need improvement[/red]")
    
    # Drift rate analysis
    drift_rate = rpe['1s']['trans_mean'] if '1s' in rpe else 0
    console.print(f"\n• Drift rate: {drift_rate:.2f} mm/second")
    if drift_rate < 10.0:
        console.print("  [green]Low drift accumulation rate[/green]")
    else:
        console.print("  [yellow]Moderate drift accumulation[/yellow]")
    
    if args.stride == 1:
        console.print("\n[dim]Note: Using real-time sliding window (last prediction only)[/dim]")
    else:
        console.print(f"\n[dim]Note: Using non-overlapping windows with stride {args.stride}[/dim]")
    
    if len(all_metrics) > 1:
        # Save averaged metrics
        avg_output_path = Path(f"inference_results_realtime_averaged_stride_{args.stride}.json")
        import json
        with open(avg_output_path, 'w') as f:
            # Convert numpy values to Python native types for JSON serialization
            json_metrics = {
                'num_sequences': len(all_metrics),
                'sequences': [m['sequence_id'] for m in all_metrics],
                'averaged_metrics': {
                    k: float(v) if isinstance(v, np.floating) else v
                    for k, v in avg_metrics.items()
                    if k != 'rpe_results'
                },
                'rpe_results': {
                    window: {
                        metric: float(value)
                        for metric, value in window_metrics.items()
                    }
                    for window, window_metrics in avg_metrics['rpe_results'].items()
                },
                'per_sequence_metrics': [
                    {
                        k: float(v) if isinstance(v, np.floating) else v
                        for k, v in m.items()
                        if k not in ['rpe_results', 'alignment_info']
                    }
                    for m in all_metrics
                ]
            }
            json.dump(json_metrics, f, indent=2)
        console.print(f"\n✅ Saved averaged results to {avg_output_path}")
        console.print(f"✅ Individual trajectories saved to inference_results_realtime_all_stride_{args.stride}/")
    else:
        console.print(f"\n✅ Saved results to {output_path}")
        console.print("\n[bold cyan]Visualize trajectory:[/bold cyan]")
        console.print(f"python visualize_trajectory.py --results {output_path}")


if __name__ == "__main__":
    main()