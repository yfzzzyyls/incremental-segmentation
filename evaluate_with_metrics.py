#!/usr/bin/env python3
"""
Fixed evaluation script with proper quaternion handling for ATE and RPE metrics.
"""

import os
import sys
import torch
import numpy as np
from rich.console import Console
from rich.table import Table
import argparse

# Add src to path
sys.path.append('src')

from src.models.multihead_vio import MultiHeadVIOModel
from train_pretrained_relative import RelativePoseDataset
from torch.utils.data import DataLoader

console = Console()


def quaternion_to_matrix(q):
    """Convert quaternion to rotation matrix (XYZW convention)."""
    # Normalize
    q = q / (np.linalg.norm(q) + 1e-8)
    x, y, z, w = q  # XYZW convention
    
    return np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
        [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
        [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]
    ])


def quaternion_multiply_xyzw(q1, q2):
    """Multiply two quaternions in XYZW convention."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ])


def quaternion_angle_xyzw(q1, q2):
    """Calculate angle between two quaternions (XYZW convention)."""
    # Normalize
    q1 = q1 / (np.linalg.norm(q1) + 1e-8)
    q2 = q2 / (np.linalg.norm(q2) + 1e-8)
    
    # Dot product
    dot = np.clip(np.dot(q1, q2), -1.0, 1.0)
    
    # Handle double cover
    if dot < 0:
        dot = -dot
    
    # Angle
    angle = 2 * np.arccos(dot)
    return angle


def build_trajectory(relative_poses):
    """Build absolute trajectory from relative poses."""
    trajectory = []
    
    # Start at origin
    current_pos = np.zeros(3)
    current_rot = np.eye(3)
    trajectory.append((current_pos.copy(), current_rot.copy()))
    
    for i in range(len(relative_poses)):
        if i == 0:
            continue  # Skip first frame (already at origin)
            
        # Extract translation and rotation
        trans = relative_poses[i, :3]
        quat = relative_poses[i, 3:]
        
        # Convert quaternion to rotation matrix
        rot_matrix = quaternion_to_matrix(quat)
        
        # Update position (in current frame)
        current_pos = current_pos + current_rot @ trans
        
        # Update rotation
        current_rot = current_rot @ rot_matrix
        
        trajectory.append((current_pos.copy(), current_rot.copy()))
    
    return trajectory


def calculate_ate(pred_trajectory, gt_trajectory):
    """Calculate Absolute Trajectory Error (ATE)."""
    errors = []
    
    for (pred_pos, _), (gt_pos, _) in zip(pred_trajectory, gt_trajectory):
        error = np.linalg.norm(pred_pos - gt_pos)
        errors.append(error)
    
    return np.array(errors)


def calculate_rpe(pred_trajectory, gt_trajectory, delta=1):
    """Calculate Relative Pose Error (RPE) at given frame interval."""
    trans_errors = []
    rot_errors = []
    
    for i in range(len(pred_trajectory) - delta):
        # Get relative motion for prediction
        pred_pos1, pred_rot1 = pred_trajectory[i]
        pred_pos2, pred_rot2 = pred_trajectory[i + delta]
        pred_rel_trans = pred_rot1.T @ (pred_pos2 - pred_pos1)
        pred_rel_rot = pred_rot1.T @ pred_rot2
        
        # Get relative motion for ground truth
        gt_pos1, gt_rot1 = gt_trajectory[i]
        gt_pos2, gt_rot2 = gt_trajectory[i + delta]
        gt_rel_trans = gt_rot1.T @ (gt_pos2 - gt_pos1)
        gt_rel_rot = gt_rot1.T @ gt_rot2
        
        # Translation error
        trans_error = np.linalg.norm(pred_rel_trans - gt_rel_trans)
        trans_errors.append(trans_error)
        
        # Rotation error (angle between rotation matrices)
        rel_rot_diff = pred_rel_rot @ gt_rel_rot.T
        # Ensure trace is in valid range
        trace = np.clip(np.trace(rel_rot_diff), -1.0, 3.0)
        angle = np.arccos(np.clip((trace - 1) / 2, -1.0, 1.0))
        rot_errors.append(np.degrees(angle))
    
    return np.array(trans_errors), np.array(rot_errors)


def calculate_rpe_quaternion(pred_quats, gt_quats, delta=1):
    """Calculate RPE directly from quaternions."""
    trans_errors = []
    rot_errors = []
    
    for i in range(len(pred_quats) - delta):
        # For relative poses, we can directly compare quaternions
        pred_q = pred_quats[i + delta, 3:]
        gt_q = gt_quats[i + delta, 3:]
        
        # Rotation error
        angle = quaternion_angle_xyzw(pred_q, gt_q)
        rot_errors.append(np.degrees(angle))
        
        # Translation error
        trans_error = np.linalg.norm(pred_quats[i + delta, :3] - gt_quats[i + delta, :3])
        trans_errors.append(trans_error)
    
    return np.array(trans_errors), np.array(rot_errors)


def evaluate_with_metrics(checkpoint_path: str, pose_scale: float = 100.0, data_dir: str = "aria_latent_data_pretrained"):
    """Evaluate with proper ATE and RPE metrics."""
    
    console.rule("[bold cyan]🚀 AR/VR Metrics Evaluation[/bold cyan]")
    
    # Load model
    model = MultiHeadVIOModel.load_from_checkpoint(checkpoint_path)
    model.eval()
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Create test dataset
    test_dataset = RelativePoseDataset(
        f"{data_dir}/test",
        pose_scale=pose_scale
    )
    
    # Create dataloader
    def collate_fn(batch):
        features, imus, poses = zip(*batch)
        return {
            'images': torch.stack(features),
            'imus': torch.stack(imus),
            'poses': torch.stack(poses)
        }
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_fn
    )
    
    console.print(f"\nTest samples: {len(test_dataset):,}")
    
    # Collect all metrics
    all_ate_errors = []
    all_rpe_trans_1 = []
    all_rpe_rot_1 = []
    all_rpe_trans_5 = []
    all_rpe_rot_5 = []
    
    # Direct quaternion errors
    direct_rot_errors = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if torch.cuda.is_available():
                batch = {k: v.cuda() for k, v in batch.items()}
            
            # Get predictions
            outputs = model(batch)
            
            # Process each sequence in batch
            batch_size = batch['images'].shape[0]
            
            for i in range(batch_size):
                # Extract predictions and ground truth
                pred_poses = torch.cat([
                    outputs['translation'][i],
                    outputs['rotation'][i]
                ], dim=1).cpu().numpy()  # [11, 7]
                
                gt_poses = batch['poses'][i].cpu().numpy()  # [11, 7]
                
                # Build trajectories
                pred_traj = build_trajectory(pred_poses)
                gt_traj = build_trajectory(gt_poses)
                
                # Calculate ATE
                ate_errors = calculate_ate(pred_traj, gt_traj)
                all_ate_errors.extend(ate_errors)
                
                # Calculate RPE from trajectories
                if len(pred_traj) > 1:
                    rpe_trans_1, rpe_rot_1 = calculate_rpe(pred_traj, gt_traj, delta=1)
                    all_rpe_trans_1.extend(rpe_trans_1)
                    all_rpe_rot_1.extend(rpe_rot_1)
                
                if len(pred_traj) > 5:
                    rpe_trans_5, rpe_rot_5 = calculate_rpe(pred_traj, gt_traj, delta=5)
                    all_rpe_trans_5.extend(rpe_trans_5)
                    all_rpe_rot_5.extend(rpe_rot_5)
                
                # Also calculate direct quaternion errors for comparison
                for j in range(1, len(pred_poses)):
                    pred_q = pred_poses[j, 3:]
                    gt_q = gt_poses[j, 3:]
                    angle = quaternion_angle_xyzw(pred_q, gt_q)
                    direct_rot_errors.append(np.degrees(angle))
            
            if batch_idx % 50 == 0:
                console.print(f"  Processed {batch_idx}/{len(test_loader)} batches...")
    
    # Convert to arrays
    ate_errors = np.array(all_ate_errors)
    rpe_trans_1 = np.array(all_rpe_trans_1)
    rpe_rot_1 = np.array(all_rpe_rot_1)
    rpe_trans_5 = np.array(all_rpe_trans_5) if all_rpe_trans_5 else np.array([0])
    rpe_rot_5 = np.array(all_rpe_rot_5) if all_rpe_rot_5 else np.array([0])
    direct_rot_errors = np.array(direct_rot_errors)
    
    # Create results table
    table = Table(title="AR/VR Standard Metrics (Fixed)")
    table.add_column("Metric", style="cyan", width=40)
    table.add_column("Value", style="green", width=25)
    table.add_column("AR/VR Target", style="yellow", width=15)
    table.add_column("Status", style="white", width=15)
    
    # ATE metrics
    ate_mean = ate_errors.mean()
    ate_median = np.median(ate_errors)
    ate_std = ate_errors.std()
    ate_95 = np.percentile(ate_errors, 95)
    
    table.add_row(
        "ATE (Absolute Trajectory Error)",
        f"{ate_mean:.4f} ± {ate_std:.4f} cm",
        "<1 cm",
        "✅ EXCEEDS" if ate_mean < 1.0 else "❌ FAILS"
    )
    table.add_row(
        "  ├─ Median",
        f"{ate_median:.4f} cm",
        "-",
        "-"
    )
    table.add_row(
        "  └─ 95th percentile",
        f"{ate_95:.4f} cm",
        "-",
        "-"
    )
    
    # RPE metrics (1 frame)
    table.add_row(
        "RPE Translation (1 frame)",
        f"{rpe_trans_1.mean():.4f} ± {rpe_trans_1.std():.4f} cm",
        "<0.1 cm",
        "✅ EXCEEDS" if rpe_trans_1.mean() < 0.1 else "❌ FAILS"
    )
    table.add_row(
        "RPE Rotation (1 frame)",
        f"{rpe_rot_1.mean():.4f} ± {rpe_rot_1.std():.4f}°",
        "<0.1°",
        "✅ EXCEEDS" if rpe_rot_1.mean() < 0.1 else "⚠️  CLOSE"
    )
    
    # Direct quaternion comparison
    table.add_row(
        "Direct Quaternion Error (mean)",
        f"{direct_rot_errors.mean():.4f} ± {direct_rot_errors.std():.4f}°",
        "<0.1°",
        "✅ EXCEEDS" if direct_rot_errors.mean() < 0.1 else "❌ FAILS"
    )
    
    # RPE metrics (5 frames)
    if len(all_rpe_trans_5) > 0:
        table.add_row(
            "RPE Translation (5 frames)",
            f"{rpe_trans_5.mean():.4f} ± {rpe_trans_5.std():.4f} cm",
            "<0.5 cm",
            "✅ EXCEEDS" if rpe_trans_5.mean() < 0.5 else "❌ FAILS"
        )
        table.add_row(
            "RPE Rotation (5 frames)",
            f"{rpe_rot_5.mean():.4f} ± {rpe_rot_5.std():.4f}°",
            "<0.5°",
            "✅ EXCEEDS" if rpe_rot_5.mean() < 0.5 else "❌ FAILS"
        )
    
    console.print("\n")
    console.print(table)
    
    # Performance Summary
    console.print("\n[bold]Performance Summary:[/bold]")
    
    if ate_mean < 0.01:  # Less than 0.01cm
        console.print(f"[bold green]✅ EXCEPTIONAL! ATE of {ate_mean:.4f}cm is sub-millimeter accuracy![/bold green]")
    elif ate_mean < 0.1:
        console.print(f"[bold green]✅ EXCELLENT! ATE of {ate_mean:.4f}cm exceeds professional AR/VR requirements![/bold green]")
    elif ate_mean < 1.0:
        console.print(f"[bold yellow]⚠️  GOOD. ATE of {ate_mean:.4f}cm meets AR/VR requirements.[/bold yellow]")
    else:
        console.print(f"[bold red]❌ NEEDS IMPROVEMENT. ATE of {ate_mean:.4f}cm exceeds 1cm threshold.[/bold red]")
    
    
    # Additional insights
    console.print("\n[bold]Technical Details:[/bold]")
    console.print(f"  • Total evaluated poses: {len(ate_errors):,}")
    console.print(f"  • Direct quaternion error: {direct_rot_errors.mean():.4f}°")
    
    return {
        'ate_mean': ate_mean,
        'rpe_trans_1': rpe_trans_1.mean(),
        'rpe_rot_1': rpe_rot_1.mean(),
        'direct_rot_error': direct_rot_errors.mean()
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fixed evaluation with proper quaternion handling')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--scale', type=float, default=100.0,
                       help='Pose scale factor (default: 100.0 for meter to cm conversion)')
    parser.add_argument('--data_dir', type=str, default='aria_latent_data_pretrained',
                       help='Data directory (default: aria_latent_data_pretrained)')
    
    args = parser.parse_args()
    
    console.print("[bold magenta]Visual-Selective-VIO Evaluation with Fixed Quaternion Handling[/bold magenta]\n")
    
    results = evaluate_with_metrics(args.checkpoint, args.scale, args.data_dir)