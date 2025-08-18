#!/usr/bin/env python3
"""
Unit test script for mask reuse control logic.
Tests the control pipeline with ADT dataset and evaluates performance.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict
from dataclasses import dataclass
import time
import argparse
from tqdm import tqdm

from ctrl import MaskReuseController, ControlInput, ControlOutput, CachedFrame


class LRUCache:
    """LRU cache for storing cached frames."""
    
    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self.cache: OrderedDict[int, CachedFrame] = OrderedDict()
        self.frame_counter = 0
    
    def add(self, frame: CachedFrame) -> int:
        """Add frame to cache and return its ID."""
        frame_id = self.frame_counter
        frame.frame_id = frame_id
        self.frame_counter += 1
        
        # Add to cache
        self.cache[frame_id] = frame
        
        # Remove oldest if over capacity
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)
        
        return frame_id
    
    def get(self, frame_id: int) -> Optional[CachedFrame]:
        """Get frame and move to end (most recently used)."""
        if frame_id not in self.cache:
            return None
        
        # Move to end
        self.cache.move_to_end(frame_id)
        return self.cache[frame_id]
    
    def get_all(self) -> Dict[int, CachedFrame]:
        """Get all cached frames."""
        return dict(self.cache)
    
    def clear(self):
        """Clear the cache."""
        self.cache.clear()


class ADTDataLoader:
    """Loader for ADT dataset with pose, depth, gaze, and segmentation data."""
    
    def __init__(self, data_root: str, sequence: str):
        """
        Args:
            data_root: Root directory (e.g., /mnt/ssd_ext/incSeg-data/)
            sequence: Sequence name (e.g., Apartment_release_clean_seq148_M1292)
        """
        self.data_root = Path(data_root)
        self.sequence = sequence
        
        # Paths
        self.processed_dir = self.data_root / "processed_adt" / "test" / sequence
        self.original_dir = self.data_root / "adt" / "test" / sequence
        
        # Load metadata
        self.metadata = self._load_metadata()
        
        # Load trajectory (poses)
        self.trajectory = self._load_trajectory()
        
        # Camera intrinsics from ADT VRS calibration
        # Extracted from camera-rgb sensor in VRS file
        # Model: FISHEYE624 (but we use pinhole approximation for homography)
        self.intrinsics = np.array([
            [610.94, 0.0, 715.11],  # fx, 0, cx
            [0.0, 610.94, 716.71],  # 0, fy, cy
            [0.0, 0.0, 1.0]         # 0, 0, 1
        ])
        
        print(f"Loaded ADT sequence: {sequence}")
        print(f"  Frames: {len(self.metadata['frames'])}")
        print(f"  Trajectory entries: {len(self.trajectory)}")
        print("\n⚠️  WARNING: ADT uses FISHEYE624 distortion model.")
        print("  Current implementation uses pinhole approximation for homography.")
        print("  This may cause warping errors, especially at image edges.")
        print("  For best results, rectify ROIs before applying homography.\n")
    
    def _load_metadata(self) -> dict:
        """Load sequence metadata."""
        metadata_path = self.processed_dir / "metadata.json"
        with open(metadata_path, 'r') as f:
            return json.load(f)
    
    def _load_trajectory(self) -> pd.DataFrame:
        """Load aria trajectory (poses)."""
        trajectory_path = self.original_dir / "aria_trajectory.csv"
        if not trajectory_path.exists():
            raise FileNotFoundError(f"Trajectory not found: {trajectory_path}")
        
        df = pd.read_csv(trajectory_path)
        # Convert timestamps to match frame timestamps
        df['timestamp_ns'] = df['tracking_timestamp_us'] * 1000
        return df
    
    def load_segmentation(self, frame_idx: int) -> Optional[np.ndarray]:
        """Load segmentation mask for a specific frame."""
        seg_dir = self.processed_dir / "segmentation"
        seg_file = seg_dir / f"frame_{frame_idx:06d}.npz"
        
        if not seg_file.exists():
            # Silently return None for missing files
            return None
        
        try:
            # Load segmentation with instance IDs
            seg_data = np.load(seg_file)
            segmentation = seg_data['segmentation']
            return segmentation
        except (EOFError, OSError, KeyError) as e:
            # Handle corrupted or incomplete files
            print(f"Warning: Could not load segmentation {seg_file}: {e}")
            return None
    
    def get_frame_data(self, frame_idx: int) -> Dict:
        """
        Get all data for a specific frame.
        
        Returns dict with:
            - rgb: (H, W, 3) array
            - depth: (H, W) array
            - gaze: (x, y) pixel coordinates
            - pose: 4x4 transformation matrix
            - timestamp_us: timestamp in microseconds
        """
        if frame_idx >= len(self.metadata['frames']):
            raise IndexError(f"Frame {frame_idx} out of range")
        
        frame_info = self.metadata['frames'][frame_idx]
        
        # Load RGB
        rgb_path = self.processed_dir / "rgb" / frame_info['rgb']
        rgb = cv2.imread(str(rgb_path))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        
        # Load depth (stored as millimeters in uint16)
        depth_path = self.processed_dir / "depth" / frame_info['depth']
        depth_data = np.load(depth_path)
        depth_mm = depth_data['depth'] if 'depth' in depth_data else depth_data['arr_0']
        # Convert from millimeters to meters
        depth = depth_mm.astype(np.float32) / 1000.0
        
        # Load gaze
        gaze_path = self.processed_dir / "gaze" / frame_info['gaze']
        with open(gaze_path, 'r') as f:
            gaze_data = json.load(f)
        gaze_pixel = (gaze_data['x_pixel'], gaze_data['y_pixel'])
        
        # Get pose (find closest timestamp)
        timestamp_ns = frame_info['rgb_timestamp_ns']
        pose_idx = np.argmin(np.abs(self.trajectory['timestamp_ns'].values - timestamp_ns))
        pose_row = self.trajectory.iloc[pose_idx]
        
        # Convert quaternion + translation to 4x4 matrix
        # NOTE: Column names suggest T_world_device (device-to-world transform)
        # But our pipeline expects T_cam_world (world-to-camera)
        # TODO: Verify actual convention from ADT documentation
        pose = self._quaternion_to_matrix(
            pose_row['qw_world_device'],
            pose_row['qx_world_device'],
            pose_row['qy_world_device'],
            pose_row['qz_world_device'],
            pose_row['tx_world_device'],
            pose_row['ty_world_device'],
            pose_row['tz_world_device']
        )
        
        return {
            'rgb': rgb,
            'depth': depth,
            'gaze': gaze_pixel,
            'pose': pose,
            'timestamp_us': gaze_data['timestamp_us'],
            'frame_idx': frame_idx
        }
    
    def _quaternion_to_matrix(self, qw, qx, qy, qz, tx, ty, tz) -> np.ndarray:
        """Convert quaternion and translation to 4x4 transformation matrix.
        
        ADT provides quaternions that encode world-to-device rotation.
        We need camera-to-world (T_world_camera) for our pipeline.
        Therefore, we transpose the rotation matrix.
        """
        # Normalize quaternion
        norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
        qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm
        
        # Convert to rotation matrix (this gives R_device_world)
        R_device_world = np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
            [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
            [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
        ])
        
        # Transpose to get R_world_device (camera-to-world convention)
        R_world_device = R_device_world.T
        
        # Create 4x4 matrix (T_world_camera)
        T = np.eye(4)
        T[:3, :3] = R_world_device  # Now represents camera-to-world rotation
        T[:3, 3] = [tx, ty, tz]     # Device position in world
        
        return T
    
    def __len__(self):
        return len(self.metadata['frames'])


def extract_instance_mask_at_gaze(
    segmentation: np.ndarray, 
    gaze: Tuple[int, int],
    scale_factor: float = 1.0,
    scale_factor_x: Optional[float] = None,
    scale_factor_y: Optional[float] = None
) -> np.ndarray:
    """
    Extract binary mask for the instance ID at gaze location.
    
    Args:
        segmentation: Full segmentation with instance IDs (H, W)
        gaze: Gaze point in RGB image coordinates
        scale_factor: Uniform scale between segmentation and RGB resolution (deprecated)
        scale_factor_x: X-axis scale factor (preferred)
        scale_factor_y: Y-axis scale factor (preferred)
        
    Returns:
        Binary mask for the selected instance
    """
    h, w = segmentation.shape
    
    # Use per-axis scaling if provided, otherwise fall back to uniform scale
    if scale_factor_x is not None and scale_factor_y is not None:
        gaze_x = int(gaze[0] * scale_factor_x)
        gaze_y = int(gaze[1] * scale_factor_y)
    else:
        # Fall back to uniform scale factor
        gaze_x = int(gaze[0] * scale_factor)
        gaze_y = int(gaze[1] * scale_factor)
    
    # Check bounds
    if gaze_x < 0 or gaze_x >= w or gaze_y < 0 or gaze_y >= h:
        print(f"Warning: Gaze point ({gaze_x}, {gaze_y}) out of bounds")
        return np.zeros((h, w), dtype=np.uint8)
    
    # Get instance ID at gaze location
    instance_id = segmentation[gaze_y, gaze_x]
    
    if instance_id == 0:
        # No object at gaze point
        print(f"No instance at gaze point ({gaze_x}, {gaze_y})")
        return np.zeros((h, w), dtype=np.uint8)
    
    # Create binary mask for this instance
    mask = (segmentation == instance_id).astype(np.uint8) * 255
    
    return mask


@dataclass
class EvaluationMetrics:
    """Metrics for evaluating mask reuse performance."""
    total_frames: int = 0
    seg_decisions: int = 0
    reuse_decisions: int = 0
    successful_reuses: int = 0
    failed_reuses: int = 0
    avg_confidence: float = 0.0
    avg_processing_time: float = 0.0
    reasons: Dict[str, int] = None
    reuse_ious: List[float] = None  # Track IoU scores for reused masks
    
    def __post_init__(self):
        if self.reasons is None:
            self.reasons = {}
        if self.reuse_ious is None:
            self.reuse_ious = []
    
    def update(self, output: ControlOutput, processing_time: float):
        """Update metrics with control output."""
        self.total_frames += 1
        self.avg_processing_time += processing_time
        
        if output.decision == "SEG":
            self.seg_decisions += 1
        else:
            self.reuse_decisions += 1
            self.avg_confidence += output.confidence
            if output.confidence > 0.7:
                self.successful_reuses += 1
            else:
                self.failed_reuses += 1
        
        # Track reasons
        if output.reason not in self.reasons:
            self.reasons[output.reason] = 0
        self.reasons[output.reason] += 1
    
    def finalize(self):
        """Compute final averages."""
        if self.total_frames > 0:
            self.avg_processing_time /= self.total_frames
        if self.reuse_decisions > 0:
            self.avg_confidence /= self.reuse_decisions
    
    def print_summary(self):
        """Print evaluation summary."""
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        print(f"Total frames processed: {self.total_frames}")
        print(f"SEG decisions: {self.seg_decisions} ({100*self.seg_decisions/max(1, self.total_frames):.1f}%)")
        print(f"REUSE decisions: {self.reuse_decisions} ({100*self.reuse_decisions/max(1, self.total_frames):.1f}%)")
        print(f"Successful reuses: {self.successful_reuses}")
        print(f"Failed reuses: {self.failed_reuses}")
        print(f"Average confidence: {self.avg_confidence:.3f}")
        print(f"Average processing time: {self.avg_processing_time*1000:.2f} ms")
        
        if self.reuse_ious:
            avg_iou = np.mean(self.reuse_ious)
            median_iou = np.median(self.reuse_ious)
            min_iou = np.min(self.reuse_ious)
            max_iou = np.max(self.reuse_ious)
            print(f"\nReuse Accuracy (IoU with ground truth):")
            print(f"  Average IoU: {avg_iou:.3f}")
            print(f"  Median IoU: {median_iou:.3f}")
            print(f"  Min IoU: {min_iou:.3f}")
            print(f"  Max IoU: {max_iou:.3f}")
        
        print("\nDecision reasons:")
        for reason, count in sorted(self.reasons.items(), key=lambda x: x[1], reverse=True):
            print(f"  {reason}: {count}")


def save_comparison_only(
    rgb: np.ndarray,
    reused_mask: Optional[np.ndarray],
    gt_mask: np.ndarray,
    gaze: Tuple[int, int],
    frame_idx: int,
    iou: float,
    save_path: str
):
    """Save only the comparison visualization for REUSE frames."""
    h, w = rgb.shape[:2]
    
    # Create side-by-side comparison
    comparison = np.zeros((h, w * 3, 3), dtype=np.uint8)
    
    # Left: RGB with reused mask (green)
    vis_reused = rgb.copy()
    if reused_mask is not None:
        mask_colored = np.zeros_like(vis_reused)
        mask_colored[:, :, 1] = reused_mask  # Green for reused
        vis_reused = cv2.addWeighted(vis_reused, 0.7, mask_colored, 0.3, 0)
    cv2.circle(vis_reused, gaze, 10, (255, 0, 0), 2)
    cv2.circle(vis_reused, gaze, 2, (255, 0, 0), -1)
    cv2.putText(vis_reused, f"REUSED (Frame {frame_idx})", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Middle: RGB with ground truth mask (blue)
    vis_gt = rgb.copy()
    if gt_mask is not None:
        mask_colored = np.zeros_like(vis_gt)
        mask_colored[:, :, 0] = gt_mask  # Blue for ground truth
        vis_gt = cv2.addWeighted(vis_gt, 0.7, mask_colored, 0.3, 0)
    cv2.circle(vis_gt, gaze, 10, (255, 0, 0), 2)
    cv2.circle(vis_gt, gaze, 2, (255, 0, 0), -1)
    cv2.putText(vis_gt, "GROUND TRUTH", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    
    # Right: Difference visualization
    vis_diff = rgb.copy()
    if reused_mask is not None and gt_mask is not None:
        # Show differences
        diff_mask = np.zeros_like(vis_diff)
        # Red where reused has mask but GT doesn't (false positive)
        diff_mask[:, :, 2] = ((reused_mask > 0) & (gt_mask == 0)).astype(np.uint8) * 255
        # Yellow where GT has mask but reused doesn't (false negative)
        diff_mask[:, :, 1] = ((gt_mask > 0) & (reused_mask == 0)).astype(np.uint8) * 255
        diff_mask[:, :, 2] += ((gt_mask > 0) & (reused_mask == 0)).astype(np.uint8) * 255
        # Green where both agree (true positive)
        agreement = (reused_mask > 0) & (gt_mask > 0)
        diff_mask[:, :, 1] += agreement.astype(np.uint8) * 128
        
        vis_diff = cv2.addWeighted(vis_diff, 0.5, diff_mask, 0.5, 0)
    
    cv2.circle(vis_diff, gaze, 10, (255, 0, 0), 2)
    cv2.circle(vis_diff, gaze, 2, (255, 0, 0), -1)
    cv2.putText(vis_diff, f"IoU: {iou:.3f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(vis_diff, "Green=Match", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    cv2.putText(vis_diff, "Red=FP Yellow=FN", (10, 85), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
    
    # Combine
    comparison[:, :w] = vis_reused
    comparison[:, w:2*w] = vis_gt
    comparison[:, 2*w:] = vis_diff
    
    # Save comparison
    cv2.imwrite(save_path, cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
    print(f"  Saved comparison for frame {frame_idx} (IoU: {iou:.3f})")


def visualize_result(
    rgb: np.ndarray,
    mask: Optional[np.ndarray],
    gaze: Tuple[int, int],
    decision: str,
    confidence: float,
    save_path: Optional[str] = None,
    gt_mask: Optional[np.ndarray] = None,
    save_comparison: bool = False
) -> np.ndarray:
    """Visualize control output."""
    vis = rgb.copy()
    h, w = vis.shape[:2]
    
    # Draw gaze point
    cv2.circle(vis, gaze, 10, (255, 0, 0), 2)
    cv2.circle(vis, gaze, 2, (255, 0, 0), -1)
    
    # Overlay mask if available
    if mask is not None:
        mask_colored = np.zeros_like(vis)
        mask_colored[:, :, 1] = mask  # Green channel for predicted/reused mask
        vis = cv2.addWeighted(vis, 0.7, mask_colored, 0.3, 0)
    
    # Add text
    text = f"{decision} (conf: {confidence:.2f})"
    cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        
        # If REUSE and we have ground truth, save comparison
        if save_comparison and decision == "REUSE" and gt_mask is not None:
            # Create side-by-side comparison
            comparison = np.zeros((h, w * 3, 3), dtype=np.uint8)
            
            # Left: RGB with reused mask (green)
            vis_reused = rgb.copy()
            if mask is not None:
                mask_colored = np.zeros_like(vis_reused)
                mask_colored[:, :, 1] = mask  # Green for reused
                vis_reused = cv2.addWeighted(vis_reused, 0.7, mask_colored, 0.3, 0)
            cv2.circle(vis_reused, gaze, 10, (255, 0, 0), 2)
            cv2.putText(vis_reused, "REUSED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Middle: RGB with ground truth mask (blue)
            vis_gt = rgb.copy()
            if gt_mask is not None:
                mask_colored = np.zeros_like(vis_gt)
                mask_colored[:, :, 0] = gt_mask  # Blue for ground truth
                vis_gt = cv2.addWeighted(vis_gt, 0.7, mask_colored, 0.3, 0)
            cv2.circle(vis_gt, gaze, 10, (255, 0, 0), 2)
            cv2.putText(vis_gt, "GROUND TRUTH", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            # Right: Difference (where they differ)
            vis_diff = rgb.copy()
            if mask is not None and gt_mask is not None:
                # Show differences
                diff_mask = np.zeros_like(vis_diff)
                # Red where reused has mask but GT doesn't (false positive)
                diff_mask[:, :, 2] = ((mask > 0) & (gt_mask == 0)).astype(np.uint8) * 255
                # Yellow where GT has mask but reused doesn't (false negative)
                diff_mask[:, :, 1] = ((gt_mask > 0) & (mask == 0)).astype(np.uint8) * 255
                diff_mask[:, :, 2] += ((gt_mask > 0) & (mask == 0)).astype(np.uint8) * 255
                # Green where both agree (true positive)
                agreement = (mask > 0) & (gt_mask > 0)
                diff_mask[:, :, 1] += agreement.astype(np.uint8) * 128
                
                vis_diff = cv2.addWeighted(vis_diff, 0.7, diff_mask, 0.3, 0)
                
                # Calculate IoU
                intersection = np.sum(agreement)
                union = np.sum((mask > 0) | (gt_mask > 0))
                iou = intersection / max(union, 1)
                cv2.putText(vis_diff, f"IoU: {iou:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            else:
                cv2.putText(vis_diff, "DIFFERENCE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.circle(vis_diff, gaze, 10, (255, 0, 0), 2)
            
            # Combine
            comparison[:, :w] = vis_reused
            comparison[:, w:2*w] = vis_gt
            comparison[:, 2*w:] = vis_diff
            
            # Save comparison
            comp_path = save_path.replace('.png', '_comparison.png')
            cv2.imwrite(comp_path, cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
    
    return vis


def test_control_pipeline(
    data_root: str = "/mnt/ssd_ext/incSeg-data",
    sequence: str = "Apartment_release_clean_seq148_M1292",
    max_frames: Optional[int] = None,
    cache_size: int = 10,
    visualize: bool = False,
    output_dir: Optional[str] = None
):
    """
    Test the mask reuse control pipeline.
    
    Args:
        data_root: Root directory of ADT data
        sequence: Sequence to test on
        max_frames: Maximum frames to process (None = process all frames)
        cache_size: Size of LRU cache
        visualize: Whether to save visualizations
        output_dir: Directory to save outputs
    """
    # Initialize components first to get sequence length
    controller = MaskReuseController()
    cache = LRUCache(max_size=cache_size)
    loader = ADTDataLoader(data_root, sequence)
    metrics = EvaluationMetrics()
    
    # Check available segmentation files
    seg_dir = loader.processed_dir / "segmentation"
    available_segs = sorted([int(f.stem.split('_')[1]) for f in seg_dir.glob("frame_*.npz") if f.stat().st_size > 0])
    
    # Default to all frames if not specified, but limit to available segmentation
    if max_frames is None:
        if available_segs:
            max_frames = min(len(loader), max(available_segs) + 1)
        else:
            max_frames = len(loader)
    
    # Process frames
    num_frames = min(max_frames, len(loader))
    
    print(f"\nTesting mask reuse control pipeline")
    print(f"  Data root: {data_root}")
    print(f"  Sequence: {sequence}")
    print(f"  Processing frames: {num_frames} / {len(loader)} total")
    print(f"  Available segmentation: {len(available_segs)} files")
    print(f"  Cache size: {cache_size}")
    
    # Create output directory
    if output_dir and visualize:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"  Output dir: {output_path}")
    
    for frame_idx in tqdm(range(num_frames), desc="Processing frames"):
        # Load frame data
        frame_data = loader.get_frame_data(frame_idx)
        
        # Create control input
        control_input = ControlInput(
            rgb=frame_data['rgb'],
            depth=frame_data['depth'],
            pose=frame_data['pose'],
            gaze_pixel=frame_data['gaze'],
            timestamp_us=frame_data['timestamp_us'],
            intrinsics=loader.intrinsics
        )
        
        # Process with controller
        start_time = time.time()
        output = controller.process(control_input, cache.get_all())
        processing_time = time.time() - start_time
        
        # Update metrics
        metrics.update(output, processing_time)
        
        # If SEG decision, get real mask from segmentation and add to cache
        if output.decision == "SEG":
            # Load real segmentation
            segmentation = loader.load_segmentation(frame_idx)
            
            if segmentation is not None:
                # Calculate scale factor between segmentation and RGB
                seg_h, seg_w = segmentation.shape
                rgb_h, rgb_w = frame_data['rgb'].shape[:2]
                scale_factor_x = seg_w / rgb_w
                scale_factor_y = seg_h / rgb_h
                
                # Extract mask for instance at gaze point (with per-axis scaling)
                real_mask = extract_instance_mask_at_gaze(
                    segmentation,
                    frame_data['gaze'],
                    scale_factor_x=scale_factor_x,
                    scale_factor_y=scale_factor_y
                )
                
                # Resize mask to RGB resolution if needed
                if real_mask.shape != (rgb_h, rgb_w):
                    real_mask = cv2.resize(real_mask, (rgb_w, rgb_h), interpolation=cv2.INTER_NEAREST)
            else:
                # Fallback if segmentation not available
                print(f"Warning: No segmentation for frame {frame_idx}")
                real_mask = np.zeros((frame_data['rgb'].shape[0], frame_data['rgb'].shape[1]), dtype=np.uint8)
            
            # Fit plane to mask once at cache time
            plane_params = None
            if np.any(real_mask > 0):
                # Helper function to fit plane (we'll use controller's method)
                plane_params = controller._fit_plane_to_mask(
                    real_mask,
                    frame_data['depth'],
                    loader.intrinsics  # This should be cached intrinsics
                )
            
            # Add to cache with plane parameters
            cached_frame = CachedFrame(
                frame_id=frame_idx,
                rgb=frame_data['rgb'],
                mask=real_mask,
                depth=frame_data['depth'],
                pose=frame_data['pose'],
                plane_params=plane_params,  # Store the fitted plane
                timestamp_us=frame_data['timestamp_us'],
                gaze_point=frame_data['gaze'],
                intrinsics=loader.intrinsics  # Store camera intrinsics
            )
            cache.add(cached_frame)
            
            # Use real mask for visualization
            output.mask = real_mask
        
        # For REUSE decisions, calculate IoU with ground truth and optionally save comparison
        if output.decision == "REUSE" and output.mask is not None:
            segmentation = loader.load_segmentation(frame_idx)
            if segmentation is not None:
                # Calculate scale factor
                seg_h, seg_w = segmentation.shape
                rgb_h, rgb_w = frame_data['rgb'].shape[:2]
                scale_factor_x = seg_w / rgb_w
                scale_factor_y = seg_h / rgb_h
                
                # Extract ground truth mask at gaze (with per-axis scaling)
                gt_mask = extract_instance_mask_at_gaze(
                    segmentation,
                    frame_data['gaze'],
                    scale_factor_x=scale_factor_x,
                    scale_factor_y=scale_factor_y
                )
                
                # Resize to RGB resolution if needed
                if gt_mask.shape != (rgb_h, rgb_w):
                    gt_mask = cv2.resize(gt_mask, (rgb_w, rgb_h), interpolation=cv2.INTER_NEAREST)
                
                # Calculate IoU
                intersection = np.sum((output.mask > 0) & (gt_mask > 0))
                union = np.sum((output.mask > 0) | (gt_mask > 0))
                iou = intersection / max(union, 1)
                metrics.reuse_ious.append(iou)
                
                # Store this reuse frame info for later visualization
                if not hasattr(test_control_pipeline, 'reuse_frames'):
                    test_control_pipeline.reuse_frames = []
                test_control_pipeline.reuse_frames.append({
                    'frame_idx': frame_idx,
                    'rgb': frame_data['rgb'],
                    'reused_mask': output.mask,
                    'gt_mask': gt_mask,
                    'gaze': frame_data['gaze'],
                    'iou': iou
                })
    
    # Finalize and print metrics
    metrics.finalize()
    metrics.print_summary()
    
    # Save evenly spaced reuse frame comparisons
    if visualize and output_dir and hasattr(test_control_pipeline, 'reuse_frames'):
        reuse_frames = test_control_pipeline.reuse_frames
        total_reuses = len(reuse_frames)
        
        if total_reuses > 0:
            # Save 1/5 of reused frames with even spacing
            num_to_save = max(1, total_reuses // 5)  # At least 1, otherwise 1/5 of total
            
            # Calculate even spacing
            if total_reuses <= num_to_save:
                # Save all if we have fewer than what we want to save
                indices_to_save = list(range(total_reuses))
            else:
                # Calculate step size for even spacing
                step = total_reuses / num_to_save
                indices_to_save = [int(i * step) for i in range(num_to_save)]
            
            print(f"\nSaving {len(indices_to_save)} evenly spaced comparison images from {total_reuses} reuse frames (1/5 sampling)")
            print(f"Selected reuse indices: {indices_to_save[:10]}{'...' if len(indices_to_save) > 10 else ''}")
            
            for save_idx, reuse_idx in enumerate(indices_to_save):
                frame_info = reuse_frames[reuse_idx]
                vis_path = str(output_path / f"comparison_{save_idx:03d}_frame_{frame_info['frame_idx']:06d}.png")
                
                save_comparison_only(
                    frame_info['rgb'],
                    frame_info['reused_mask'],
                    frame_info['gt_mask'],
                    frame_info['gaze'],
                    frame_info['frame_idx'],
                    frame_info['iou'],
                    vis_path
                )
            
            print(f"Comparison images saved to {output_path}")
    
    # Clean up stored reuse frames to free memory
    if hasattr(test_control_pipeline, 'reuse_frames'):
        delattr(test_control_pipeline, 'reuse_frames')
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Test mask reuse control pipeline")
    parser.add_argument("--data-root", type=str, 
                       default="/mnt/ssd_ext/incSeg-data",
                       help="Root directory of ADT data")
    parser.add_argument("--sequence", type=str,
                       default="Apartment_release_clean_seq148_M1292",
                       help="Sequence to test on (e.g., Apartment_release_clean_seq148_M1292)")
    parser.add_argument("--max-frames", type=int, default=None,
                       help="Maximum frames to process (default: all frames in sequence)")
    parser.add_argument("--cache-size", type=int, default=30,
                       help="Size of LRU cache (default: 30)")
    parser.add_argument("--visualize", action="store_true",
                       help="Save visualizations")
    parser.add_argument("--output-dir", type=str, default="./output",
                       help="Directory to save outputs")
    
    args = parser.parse_args()
    
    test_control_pipeline(
        data_root=args.data_root,
        sequence=args.sequence,
        max_frames=args.max_frames,
        cache_size=args.cache_size,
        visualize=args.visualize,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()