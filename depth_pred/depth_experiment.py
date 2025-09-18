#!/usr/bin/env python3
"""
Main depth sparsity experiment.
Tests impact of dense vs sparse depth on mask projection accuracy.
"""

import numpy as np
import cv2
import json
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

from pose_utils import PoseManager
from reprojection import (
    DepthMode, reproject_mask, compute_reprojection_error,
    compute_iou, unproject_kb8, project_kb8, transform_points_3d
)


class DepthExperiment:
    """Main experiment runner."""

    def __init__(self, data_root: str, sequence: str):
        """Initialize experiment."""
        self.data_root = Path(data_root)
        self.sequence = sequence

        # Setup paths
        self.adt_dir = self.data_root / "adt" / "test" / sequence
        self.processed_dir = self.data_root / "processed_adt" / "test" / sequence

        # Load calibration
        with open(self.processed_dir / "calibration.json", 'r') as f:
            self.calibration = json.load(f)

        # Camera parameters
        self.K = np.array([
            [self.calibration['intrinsics']['fx'], 0, self.calibration['intrinsics']['cx']],
            [0, self.calibration['intrinsics']['fy'], self.calibration['intrinsics']['cy']],
            [0, 0, 1]
        ], dtype=np.float32)

        self.dist_coeffs = np.array(self.calibration['distortion_coeffs'], dtype=np.float32)

        # Load metadata
        with open(self.processed_dir / "metadata.json", 'r') as f:
            self.metadata = json.load(f)

        # Initialize pose manager
        traj_path = self.adt_dir / "aria_trajectory.csv"
        calib_path = self.processed_dir / "calibration.json"
        self.pose_manager = PoseManager(traj_path, calib_path)

        # Results storage
        self.results = []

    def load_frame_data(self, frame_idx: int) -> Dict:
        """Load all data for a frame."""
        frame_info = self.metadata['frames'][frame_idx]

        # Load RGB
        rgb_path = self.processed_dir / "rgb" / frame_info['rgb']
        rgb = cv2.imread(str(rgb_path))

        # Load depth
        depth_path = self.processed_dir / "depth" / f"frame_{frame_idx:06d}.npz"
        depth_data = np.load(depth_path)
        depth = depth_data['depth'].astype(np.float32) / 1000.0  # Convert to meters

        # Load segmentation
        seg_path = self.processed_dir / "segmentation" / f"frame_{frame_idx:06d}.npz"
        seg_data = np.load(seg_path)
        segmentation = seg_data['segmentation']

        # Load gaze
        gaze_path = self.processed_dir / "gaze" / frame_info['gaze']
        with open(gaze_path, 'r') as f:
            gaze_data = json.load(f)

        gaze_point = (int(gaze_data['x_pixel']), int(gaze_data['y_pixel']))

        return {
            'rgb': rgb,
            'depth': depth,
            'segmentation': segmentation,
            'gaze': gaze_point,
            'timestamp_ns': frame_info['rgb_timestamp_ns']
        }

    def run_identity_test(self, frame_idx: int) -> Dict:
        """Test reprojection to same frame (should be perfect)."""
        print(f"\n=== Identity Test (Frame {frame_idx}) ===")

        # Load data
        data = self.load_frame_data(frame_idx)

        # Get object at gaze
        gx, gy = data['gaze']
        object_id = data['segmentation'][gy, gx]

        if object_id == 0:
            print("No object at gaze")
            return {'error': 'No object at gaze'}

        # Extract mask
        mask = (data['segmentation'] == object_id).astype(np.uint8) * 255

        # Identity transform
        T_identity = np.eye(4)

        results = {}

        # Test all depth modes
        for depth_mode in [DepthMode.DENSE_GT, DepthMode.SPARSE_GT]:
            mask_reproj, info = reproject_mask(
                mask, data['depth'], T_identity,
                self.K, self.dist_coeffs,
                depth_mode, data['gaze']
            )

            iou_strict = compute_iou(mask, mask_reproj, "strict")
            iou_fill = compute_iou(mask, mask_reproj, "fill_aware")

            # For identity, IoU should be ~1.0
            print(f"  {depth_mode.value}: IoU(strict)={iou_strict:.4f}, IoU(fill)={iou_fill:.4f}")

            results[depth_mode.value] = {
                'iou_strict': iou_strict,
                'iou_fill': iou_fill,
                'info': info
            }

        return results

    def run_synthetic_motion_test(self, frame_idx: int) -> Dict:
        """Test with synthetic motion (known transform)."""
        print(f"\n=== Synthetic Motion Test (Frame {frame_idx}) ===")

        # Load data
        data = self.load_frame_data(frame_idx)

        # Get object at gaze
        gx, gy = data['gaze']
        object_id = data['segmentation'][gy, gx]

        if object_id == 0:
            return {'error': 'No object at gaze'}

        mask = (data['segmentation'] == object_id).astype(np.uint8) * 255

        # Create synthetic motion (10cm forward, 5° rotation)
        T_synthetic = np.eye(4)
        T_synthetic[2, 3] = 0.1  # 10cm forward

        # Add small rotation
        angle = np.radians(5)
        R = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        T_synthetic[:3, :3] = R

        print("  Synthetic motion: 10cm forward, 5° rotation")

        results = {}

        for depth_mode in [DepthMode.DENSE_GT, DepthMode.SPARSE_GT]:
            mask_reproj, info = reproject_mask(
                mask, data['depth'], T_synthetic,
                self.K, self.dist_coeffs,
                depth_mode, data['gaze']
            )

            # For synthetic, compare against dense GT as reference
            if depth_mode == DepthMode.DENSE_GT:
                mask_ref = mask_reproj.copy()
                results['reference_mask'] = mask_ref

            iou_vs_original = compute_iou(mask, mask_reproj, "strict")

            if 'reference_mask' in results:
                iou_vs_dense = compute_iou(results['reference_mask'], mask_reproj, "strict")
            else:
                iou_vs_dense = iou_vs_original

            print(f"  {depth_mode.value}: IoU vs original={iou_vs_original:.4f}, "
                  f"vs dense GT={iou_vs_dense:.4f}")

            results[depth_mode.value] = {
                'iou_vs_original': iou_vs_original,
                'iou_vs_dense': iou_vs_dense,
                'info': info
            }

        return results

    def run_real_frame_pair(self, frame1_idx: int, frame2_idx: int) -> Dict:
        """Test on real frame pair."""
        print(f"\n=== Real Frame Pair ({frame1_idx}→{frame2_idx}) ===")

        # Load data
        data1 = self.load_frame_data(frame1_idx)
        data2 = self.load_frame_data(frame2_idx)

        # Get relative pose
        T_rel, pose_info = self.pose_manager.compute_relative_pose(
            frame1_idx, frame2_idx, self.metadata
        )

        print(f"  Motion: {pose_info['translation_m']*100:.1f}cm, "
              f"{pose_info['rotation_deg']:.1f}°")

        # Get object at gaze in frame1
        gx1, gy1 = data1['gaze']
        object_id = data1['segmentation'][gy1, gx1]

        if object_id == 0:
            return {'error': 'No object at gaze in frame1'}

        # Extract masks
        mask1 = (data1['segmentation'] == object_id).astype(np.uint8) * 255
        mask2_gt = (data2['segmentation'] == object_id).astype(np.uint8) * 255

        if np.sum(mask2_gt) == 0:
            print("  Object not visible in frame2")
            return {'error': 'Object not in frame2'}

        results = {'pose_info': pose_info}

        # Test all depth modes
        for depth_mode in [DepthMode.DENSE_GT, DepthMode.SPARSE_GT]:
            mask2_pred, info = reproject_mask(
                mask1, data1['depth'], T_rel,
                self.K, self.dist_coeffs,
                depth_mode, data1['gaze']
            )

            # Compute IoU
            iou_strict = compute_iou(mask2_gt, mask2_pred, "strict")
            iou_fill = compute_iou(mask2_gt, mask2_pred, "fill_aware")

            # Compute pixel errors
            error_stats = compute_reprojection_error(
                mask1, data1['depth'], T_rel,
                self.K, self.dist_coeffs, mask2_gt
            )

            print(f"  {depth_mode.value}: IoU(strict)={iou_strict:.4f}, "
                  f"IoU(fill)={iou_fill:.4f}, "
                  f"Med error={error_stats.get('median_error', -1):.1f}px")

            results[depth_mode.value] = {
                'iou_strict': iou_strict,
                'iou_fill': iou_fill,
                'error_stats': error_stats,
                'info': info
            }

        return results

    def run_full_experiment(self, frame_pairs: List[Tuple[int, int]]):
        """Run complete experiment on multiple frame pairs."""
        print("="*60)
        print("DEPTH SPARSITY IMPACT EXPERIMENT")
        print("="*60)

        # 1. Identity tests
        print("\n1. IDENTITY TESTS")
        for frame_idx in [179, 183, 195]:
            result = self.run_identity_test(frame_idx)
            self.results.append({'type': 'identity', 'frame': frame_idx, 'result': result})

        # 2. Synthetic motion tests
        print("\n2. SYNTHETIC MOTION TESTS")
        for frame_idx in [179, 183]:
            result = self.run_synthetic_motion_test(frame_idx)
            self.results.append({'type': 'synthetic', 'frame': frame_idx, 'result': result})

        # 3. Real frame pairs
        print("\n3. REAL FRAME PAIRS")
        for f1, f2 in frame_pairs:
            result = self.run_real_frame_pair(f1, f2)
            self.results.append({'type': 'real', 'pair': (f1, f2), 'result': result})

        # Generate summary
        self.generate_summary()

    def generate_summary(self):
        """Generate summary table of results."""
        print("\n" + "="*60)
        print("EXPERIMENT SUMMARY")
        print("="*60)

        # Create summary table
        rows = []

        for item in self.results:
            if item['type'] == 'identity':
                for mode in ['dense_gt', 'sparse_gt']:
                    if mode in item['result']:
                        rows.append({
                            'Test': f"Identity {item['frame']}",
                            'Depth Mode': mode,
                            'IoU(strict)': item['result'][mode]['iou_strict'],
                            'IoU(fill)': item['result'][mode]['iou_fill'],
                            'Med Error': '-'
                        })

            elif item['type'] == 'real':
                f1, f2 = item['pair']
                for mode in ['dense_gt', 'sparse_gt']:
                    if mode in item['result']:
                        med_err = item['result'][mode]['error_stats'].get('median_error', -1)
                        rows.append({
                            'Test': f"Real {f1}→{f2}",
                            'Depth Mode': mode,
                            'IoU(strict)': item['result'][mode]['iou_strict'],
                            'IoU(fill)': item['result'][mode]['iou_fill'],
                            'Med Error': f"{med_err:.1f}px" if med_err >= 0 else '-'
                        })

        # Print table
        df = pd.DataFrame(rows)
        print("\n" + df.to_string(index=False))

        # Compute degradation
        print("\n\nDEGRADATION ANALYSIS:")
        dense_ious = df[df['Depth Mode'] == 'dense_gt']['IoU(strict)'].values
        sparse_ious = df[df['Depth Mode'] == 'sparse_gt']['IoU(strict)'].values

        if len(dense_ious) > 0 and len(sparse_ious) > 0:
            mean_dense = np.mean(dense_ious[dense_ious > 0])
            mean_sparse = np.mean(sparse_ious[sparse_ious > 0])

            print(f"  Mean IoU Dense GT: {mean_dense:.4f}")
            print(f"  Mean IoU Sparse GT: {mean_sparse:.4f}")
            print(f"  Relative degradation: {(1 - mean_sparse/mean_dense)*100:.1f}%")


if __name__ == "__main__":
    # Setup experiment
    data_root = "/mnt/ssd_ext/incSeg-data"
    sequence = "Apartment_release_clean_seq148_M1292"

    exp = DepthExperiment(data_root, sequence)

    # Define frame pairs to test
    frame_pairs = [
        (179, 183),   # Small baseline
        (179, 195),   # Medium baseline
        (179, 199),   # Large baseline
    ]

    # Run experiment
    exp.run_full_experiment(frame_pairs)