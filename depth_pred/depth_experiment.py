#!/usr/bin/env python3
"""
Main depth sparsity experiment.
Tests impact of dense vs sparse depth on mask projection accuracy.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, Rectangle

from pose_utils import PoseManager
from reprojection import (
    DepthMode, reproject_mask, compute_reprojection_error,
    compute_iou, unproject_kb8, project_kb8, transform_points_3d
)
from depth_predictor import (
    DEFAULT_CHECKPOINT as DEFAULT_DEPTH_CHECKPOINT,
    GazeDepthPredictor,
)


class DepthExperiment:
    """Main experiment runner."""

    def __init__(
        self,
        data_root: str,
        sequence: str,
        depth_modes: List[DepthMode],
        frame_pairs: Optional[List[Tuple[int, int]]] = None,
        run_identity: bool = True,
        visualize: bool = False,
        verbose: bool = True,
        predictor_checkpoint: Optional[str] = None,
        predictor_device: Optional[str] = "cuda",
        patch_coverage: int = 88,
    ):
        """Initialize experiment."""
        self.data_root = Path(data_root)
        self.sequence = sequence
        self.depth_modes = depth_modes
        self.frame_pairs = frame_pairs
        self.run_identity = run_identity
        self.visualize = visualize
        self.verbose = verbose
        self.predictor_checkpoint = (
            Path(predictor_checkpoint).expanduser()
            if predictor_checkpoint
            else DEFAULT_DEPTH_CHECKPOINT
        )
        self.predictor_device = predictor_device
        self.patch_coverage = patch_coverage

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

        self.gaze_predictor: Optional[GazeDepthPredictor] = None
        if DepthMode.SPARSE_PRED in self.depth_modes:
            self.gaze_predictor = GazeDepthPredictor(
                checkpoint_path=self.predictor_checkpoint,
                device=self.predictor_device,
                patch_coverage=self.patch_coverage,
            )

    def _log(self, message: str):
        """Conditional logging based on verbosity."""
        if self.verbose:
            print(message)

    def _predict_sparse_depth(self, frame_data: Dict) -> Tuple[Optional[float], Optional[Dict]]:
        """Predict sparse depth using the gaze predictor if available."""
        if self.gaze_predictor is None:
            return None, {'error': 'Depth predictor not initialized'}

        rgb = frame_data.get('rgb')
        gaze = frame_data.get('gaze')
        if rgb is None or gaze is None:
            return None, {'error': 'Missing RGB or gaze data'}

        depth, meta = self.gaze_predictor.predict(rgb, gaze[0], gaze[1])
        return depth, meta

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
        self._log(f"\n=== Identity Test (Frame {frame_idx}) ===")

        # Load data
        data = self.load_frame_data(frame_idx)

        pred_depth = None
        pred_meta = None
        if DepthMode.SPARSE_PRED in self.depth_modes:
            pred_depth, pred_meta = self._predict_sparse_depth(data)

        # Get object at gaze
        gx, gy = data['gaze']
        object_id = data['segmentation'][gy, gx]

        if object_id == 0:
            self._log("No object at gaze")
            return {'error': 'No object at gaze'}

        # Extract mask
        mask = (data['segmentation'] == object_id).astype(np.uint8) * 255

        # Identity transform
        T_identity = np.eye(4)

        results = {}

        # Test all depth modes
        for depth_mode in self.depth_modes:
            mask_reproj, info = reproject_mask(
                mask, data['depth'], T_identity,
                self.K, self.dist_coeffs,
                depth_mode, data['gaze'],
                predicted_depth=pred_depth if depth_mode == DepthMode.SPARSE_PRED else None,
                sparse_info=pred_meta if depth_mode == DepthMode.SPARSE_PRED else None,
            )

            iou_strict = compute_iou(mask, mask_reproj, "strict")
            iou_fill = compute_iou(mask, mask_reproj, "fill_aware")

            # For identity, IoU should be ~1.0
            self._log(
                f"  {depth_mode.value}: IoU(strict)={iou_strict:.4f}, IoU(fill)={iou_fill:.4f}"
            )

            results[depth_mode.value] = {
                'iou_strict': iou_strict,
                'iou_fill': iou_fill,
                'info': info
            }

            if depth_mode == DepthMode.SPARSE_PRED and pred_meta:
                results[depth_mode.value]['pred_meta'] = pred_meta

        return results

    def run_synthetic_motion_test(self, frame_idx: int) -> Dict:
        """Test with synthetic motion (known transform)."""
        self._log(f"\n=== Synthetic Motion Test (Frame {frame_idx}) ===")

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

        self._log("  Synthetic motion: 10cm forward, 5° rotation")

        results = {}

        for depth_mode in self.depth_modes:
            mask_reproj, info = reproject_mask(
                mask, data['depth'], T_synthetic,
                self.K, self.dist_coeffs,
                depth_mode, data['gaze'],
                predicted_depth=pred_depth if depth_mode == DepthMode.SPARSE_PRED else None,
                sparse_info=pred_meta if depth_mode == DepthMode.SPARSE_PRED else None,
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

            self._log(
                f"  {depth_mode.value}: IoU vs original={iou_vs_original:.4f}, "
                f"vs dense GT={iou_vs_dense:.4f}"
            )

            results[depth_mode.value] = {
                'iou_vs_original': iou_vs_original,
                'iou_vs_dense': iou_vs_dense,
                'info': info
            }

            if depth_mode == DepthMode.SPARSE_PRED and pred_meta:
                results[depth_mode.value]['pred_meta'] = pred_meta

        return results

    def run_real_frame_pair(
        self,
        frame1_idx: int,
        frame2_idx: int,
        pair_idx: Optional[int] = None,
        total_pairs: Optional[int] = None,
    ) -> Dict:
        """Test on real frame pair."""
        if self.verbose:
            self._log(f"\n=== Real Frame Pair ({frame1_idx}→{frame2_idx}) ===")
        elif pair_idx is not None and total_pairs:
            if pair_idx == 0 or (pair_idx + 1) % 50 == 0 or (pair_idx + 1) == total_pairs:
                print(
                    f"Processing real pair {pair_idx + 1}/{total_pairs} "
                    f"({frame1_idx}→{frame2_idx})"
                )

        # Load data
        data1 = self.load_frame_data(frame1_idx)
        data2 = self.load_frame_data(frame2_idx)

        pred_depth = None
        pred_meta = None
        if DepthMode.SPARSE_PRED in self.depth_modes:
            pred_depth, pred_meta = self._predict_sparse_depth(data1)

        # Get relative pose
        T_rel, pose_info = self.pose_manager.compute_relative_pose(
            frame1_idx, frame2_idx, self.metadata
        )

        self._log(
            f"  Motion: {pose_info['translation_m']*100:.1f}cm, "
            f"{pose_info['rotation_deg']:.1f}°"
        )

        # Get object at gaze in frame1
        gx1, gy1 = data1['gaze']
        object_id = data1['segmentation'][gy1, gx1]

        if object_id == 0:
            return {'error': 'No object at gaze in frame1'}

        # Extract masks
        mask1 = (data1['segmentation'] == object_id).astype(np.uint8) * 255
        mask2_gt = (data2['segmentation'] == object_id).astype(np.uint8) * 255

        if np.sum(mask2_gt) == 0:
            self._log("  Object not visible in frame2")
            return {'error': 'Object not in frame2'}

        results = {'pose_info': pose_info}
        if pred_meta:
            results['sparse_pred_meta'] = pred_meta

        # Test all depth modes
        for depth_mode in self.depth_modes:
            mask2_pred, info = reproject_mask(
                mask1, data1['depth'], T_rel,
                self.K, self.dist_coeffs,
                depth_mode, data1['gaze'],
                predicted_depth=pred_depth if depth_mode == DepthMode.SPARSE_PRED else None,
                sparse_info=pred_meta if depth_mode == DepthMode.SPARSE_PRED else None,
            )

            # Compute IoU
            iou_strict = compute_iou(mask2_gt, mask2_pred, "strict")
            iou_fill = compute_iou(mask2_gt, mask2_pred, "fill_aware")

            # Compute pixel errors
            error_stats = compute_reprojection_error(
                mask1, data1['depth'], T_rel,
                self.K, self.dist_coeffs, mask2_gt
            )

            self._log(
                f"  {depth_mode.value}: IoU(strict)={iou_strict:.4f}, "
                f"IoU(fill)={iou_fill:.4f}, "
                f"Med error={error_stats.get('median_error', -1):.1f}px"
            )

            entry = {
                'iou_strict': iou_strict,
                'iou_fill': iou_fill,
                'error_stats': error_stats,
                'info': info,
            }

            if depth_mode == DepthMode.SPARSE_PRED and pred_meta:
                entry['pred_meta'] = pred_meta

            if self.visualize:
                entry['mask_pred'] = mask2_pred

            results[depth_mode.value] = entry

        if self.visualize:
            results['mask1'] = mask1
            results['mask2_gt'] = mask2_gt
            results['data1'] = data1
            results['data2'] = data2
            self.visualize_projection(frame1_idx, frame2_idx, results)

        return results

    def visualize_projection(self, frame1_idx: int, frame2_idx: int, results: Dict):
        """Create visualization of mask projection."""
        from pathlib import Path

        # Create output directory
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True, parents=True)

        # Get data
        mask1 = results['mask1']
        mask2_gt = results['mask2_gt']
        data1 = results['data1']
        data2 = results['data2']

        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Frame 1 with mask
        axes[0, 0].imshow(cv2.cvtColor(data1['rgb'], cv2.COLOR_BGR2RGB))
        axes[0, 0].contour(mask1, colors='g', linewidths=2)
        gaze1 = data1['gaze']
        axes[0, 0].add_patch(Circle(gaze1, 10, fill=True, color='red'))
        axes[0, 0].set_title(f'Frame {frame1_idx} (Source)\nGaze: ({gaze1[0]}, {gaze1[1]})')
        axes[0, 0].axis('off')

        # Frame 2 with GT mask
        axes[0, 1].imshow(cv2.cvtColor(data2['rgb'], cv2.COLOR_BGR2RGB))
        axes[0, 1].contour(mask2_gt, colors='g', linewidths=2)
        gaze2 = data2['gaze']
        axes[0, 1].add_patch(Circle(gaze2, 10, fill=True, color='red'))
        axes[0, 1].set_title(f'Frame {frame2_idx} (Target)\nGround Truth')
        axes[0, 1].axis('off')

        # Frame 2 with projected mask (Dense GT)
        if 'dense_gt' in results:
            mask2_pred_dense = results['dense_gt']['mask_pred']
            axes[0, 2].imshow(cv2.cvtColor(data2['rgb'], cv2.COLOR_BGR2RGB))
            axes[0, 2].contour(mask2_pred_dense, colors='r', linewidths=2)
            axes[0, 2].contour(mask2_gt, colors='g', linewidths=1, alpha=0.5)
            iou = results['dense_gt']['iou_strict']
            axes[0, 2].set_title(f'Dense GT Projection\nIoU: {iou:.3f}')
            axes[0, 2].axis('off')

        # Depth maps
        axes[1, 0].imshow(data1['depth'], cmap='viridis')
        axes[1, 0].contour(mask1, colors='r', linewidths=1)
        axes[1, 0].set_title(f'Depth Frame {frame1_idx}')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(data2['depth'], cmap='viridis')
        axes[1, 1].contour(mask2_gt, colors='g', linewidths=1)
        axes[1, 1].set_title(f'Depth Frame {frame2_idx}')
        axes[1, 1].axis('off')

        # Overlay comparison
        h, w = mask2_gt.shape
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        overlay[mask2_gt > 0] = [0, 255, 0]  # Green for GT
        if 'dense_gt' in results:
            mask2_pred_dense = results['dense_gt']['mask_pred']
            overlay[mask2_pred_dense > 0, 0] = 255  # Red channel for projected
        axes[1, 2].imshow(overlay)
        axes[1, 2].set_title('Overlay: Green=GT, Red=Projected, Yellow=Both')
        axes[1, 2].axis('off')

        # Add pose info
        if 'pose_info' in results:
            info = results['pose_info']
            fig.text(0.5, 0.02, f'Motion: {info["translation_m"]*100:.1f}cm, {info["rotation_deg"]:.1f}°',
                    ha='center', fontsize=12)

        plt.suptitle(f'Mask Projection: Frame {frame1_idx} → {frame2_idx}', fontsize=14)
        plt.tight_layout()

        # Save figure
        filename = f"projection_GT_{frame1_idx}_to_{frame2_idx}.png"
        plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
        print(f"  Visualization saved to output/{filename}")

        plt.close()  # Close to free memory

    def run_full_experiment(self, frame_pairs: Optional[List[Tuple[int, int]]] = None):
        """Run complete experiment on multiple frame pairs."""
        print("="*60)
        print("DEPTH SPARSITY IMPACT EXPERIMENT")
        print("="*60)

        step_idx = 1

        if self.run_identity:
            print(f"\n{step_idx}. IDENTITY TESTS")
            for frame_idx in [179, 183, 195]:
                result = self.run_identity_test(frame_idx)
                self.results.append({'type': 'identity', 'frame': frame_idx, 'result': result})
            step_idx += 1

        real_pairs = frame_pairs if frame_pairs is not None else self.frame_pairs
        if not real_pairs:
            raise ValueError("No real frame pairs provided for experiment")

        print(f"\n{step_idx}. REAL FRAME PAIRS")
        total_pairs = len(real_pairs)
        for idx, (f1, f2) in enumerate(real_pairs):
            result = self.run_real_frame_pair(f1, f2, pair_idx=idx, total_pairs=total_pairs)
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
                for mode in self.depth_modes:
                    key = mode.value
                    if key in item['result']:
                        rows.append({
                            'Test': f"Identity {item['frame']}",
                            'Depth Mode': key,
                            'IoU(strict)': item['result'][key]['iou_strict'],
                            'IoU(fill)': item['result'][key]['iou_fill'],
                            'Med Error': '-'
                        })

            elif item['type'] == 'real':
                f1, f2 = item['pair']
                for mode in self.depth_modes:
                    key = mode.value
                    if key in item['result']:
                        med_err = item['result'][key]['error_stats'].get('median_error', -1)
                        rows.append({
                            'Test': f"Real {f1}→{f2}",
                            'Depth Mode': key,
                            'IoU(strict)': item['result'][key]['iou_strict'],
                            'IoU(fill)': item['result'][key]['iou_fill'],
                            'Med Error': f"{med_err:.1f}px" if med_err >= 0 else '-'
                        })

        # Print table
        df = pd.DataFrame(rows)
        print("\n" + df.to_string(index=False))

        # Compute degradation
        print("\n\nDEGRADATION ANALYSIS:")
        dense_mask = df['Depth Mode'] == DepthMode.DENSE_GT.value
        if not df.empty:
            real_mask = df['Test'].str.startswith('Real')
        else:
            real_mask = pd.Series(dtype=bool, index=df.index)

        if dense_mask.any():
            mean_dense = df.loc[dense_mask, 'IoU(strict)'].mean()
            print(f"  Mean IoU Dense GT (all tests): {mean_dense:.4f}")

            for mode in self.depth_modes:
                if mode == DepthMode.DENSE_GT:
                    continue
                mode_mask = df['Depth Mode'] == mode.value
                if mode_mask.any():
                    mean_mode = df.loc[mode_mask, 'IoU(strict)'].mean()
                    degradation = (1 - mean_mode / mean_dense) * 100 if mean_dense > 0 else 0.0
                    print(f"  Mean IoU {mode.value} (all tests): {mean_mode:.4f} "
                          f"→ degradation vs dense: {degradation:.1f}%")

        # Average IoU across real frame pairs per mode
        if real_mask.any():
            print("\nAVERAGE REAL-PAIR IOU:")
            dense_real_mask = dense_mask & real_mask
            dense_real_mean = (
                df.loc[dense_real_mask, 'IoU(strict)'].mean()
                if dense_real_mask.any()
                else None
            )
            for mode in self.depth_modes:
                mode_mask = (df['Depth Mode'] == mode.value) & real_mask
                if mode_mask.any():
                    mean_real = df.loc[mode_mask, 'IoU(strict)'].mean()
                    line = f"  {mode.value}: {mean_real:.4f}"
                    if dense_real_mean and mode != DepthMode.DENSE_GT:
                        degradation = (
                            (1 - mean_real / dense_real_mean) * 100
                            if dense_real_mean > 0
                            else 0.0
                        )
                        line += f" (Δ vs dense: {degradation:.1f}%)"
                    print(line)

        # IoU vs baseline visualization
        translation_data = {mode.value: [] for mode in self.depth_modes}
        frame_gap_data = {mode.value: [] for mode in self.depth_modes}

        for item in self.results:
            if item['type'] != 'real':
                continue
            translation_cm = item['result']['pose_info']['translation_m'] * 100
            frame_gap = abs(item['pair'][1] - item['pair'][0])
            for mode in self.depth_modes:
                key = mode.value
                mode_result = item['result'].get(key)
                if not mode_result:
                    continue
                translation_data[key].append((translation_cm, mode_result['iou_strict']))
                frame_gap_data[key].append((frame_gap, mode_result['iou_strict']))

        if any(translation_data[key] for key in translation_data):
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True, parents=True)

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            if not isinstance(axes, np.ndarray):
                axes = np.array([axes])

            for mode in self.depth_modes:
                key = mode.value
                if translation_data[key]:
                    data = np.array(translation_data[key])
                    data = data[data[:, 0].argsort()]
                    axes[0].plot(data[:, 0], data[:, 1], marker='o', linestyle='-', label=key)
                if frame_gap_data[key]:
                    data_gap = np.array(frame_gap_data[key])
                    data_gap = data_gap[data_gap[:, 0].argsort()]
                    axes[1].plot(data_gap[:, 0], data_gap[:, 1], marker='o', linestyle='-', label=key)

            axes[0].set_title('IoU vs Translation Baseline')
            axes[0].set_xlabel('Translation (cm)')
            axes[0].set_ylabel('IoU (strict)')
            axes[0].grid(True, alpha=0.3)

            axes[1].set_title('IoU vs Frame Gap')
            axes[1].set_xlabel('Frame Difference')
            axes[1].set_ylabel('IoU (strict)')
            axes[1].grid(True, alpha=0.3)

            axes[0].legend()
            axes[1].legend()

            plt.tight_layout()
            plot_path = output_dir / "iou_vs_baseline.png"
            plt.savefig(plot_path, dpi=150)
            plt.close(fig)
            print(f"\nSaved IoU-baseline plot to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Depth sparsity impact experiment")
    parser.add_argument(
        "--data-root",
        default="/mnt/ssd_ext/incSeg-data",
        help="Root directory containing processed ADT data"
    )
    parser.add_argument(
        "--sequence",
        default="Apartment_release_clean_seq148_M1292",
        help="ADT sequence name"
    )
    parser.add_argument(
        "--dense-gt",
        action="store_true",
        help="Include dense ground-truth depth condition"
    )
    parser.add_argument(
        "--sparse-gt",
        action="store_true",
        help="Include sparse ground-truth depth (gaze patch mean) condition"
    )
    parser.add_argument(
        "--sparse-pred",
        action="store_true",
        help="Include sparse predicted depth condition"
    )
    parser.add_argument(
        "--frame-pairs-file",
        help="Path to JSON file containing frame pair definitions"
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        help="Optional limit on number of real frame pairs to evaluate"
    )
    parser.add_argument(
        "--skip-identity",
        action="store_true",
        help="Skip identity self-checks"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enable visualization outputs for each real frame pair"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce per-pair logging"
    )
    parser.add_argument(
        "--depth-checkpoint",
        default=None,
        help="Path to depth predictor checkpoint"
    )
    parser.add_argument(
        "--predictor-device",
        default="cuda",
        choices=["auto", "cuda", "cpu"],
        help="Device for depth predictor (default: cuda)"
    )
    parser.add_argument(
        "--patch-coverage",
        type=int,
        default=88,
        help="Patch coverage in pixels before downsampling to 88×88"
    )

    args = parser.parse_args()

    selected_modes: List[DepthMode]
    if args.dense_gt or args.sparse_gt or args.sparse_pred:
        selected_modes = []
        if args.dense_gt:
            selected_modes.append(DepthMode.DENSE_GT)
        if args.sparse_gt:
            selected_modes.append(DepthMode.SPARSE_GT)
        if args.sparse_pred:
            selected_modes.append(DepthMode.SPARSE_PRED)
    else:
        selected_modes = [DepthMode.DENSE_GT, DepthMode.SPARSE_GT]

    if not selected_modes:
        raise ValueError("No depth experiment selected. Enable at least one condition via CLI flags.")

    if args.frame_pairs_file:
        pairs_path = Path(args.frame_pairs_file)
        if not pairs_path.exists():
            raise FileNotFoundError(f"Frame pairs file not found: {pairs_path}")
        with open(pairs_path, 'r') as f:
            pair_entries = json.load(f)
        frame_pairs: List[Tuple[int, int]] = []
        for entry in pair_entries:
            if isinstance(entry, dict):
                frame_pairs.append((int(entry['frame1']), int(entry['frame2'])))
            elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                frame_pairs.append((int(entry[0]), int(entry[1])))
            else:
                raise ValueError("Invalid frame pair entry in JSON file")
    else:
        frame_pairs = [
            (179, 183),   # Small baseline
            (179, 195),   # Medium baseline
            (179, 199),   # Large baseline
        ]

    if args.max_pairs is not None:
        frame_pairs = frame_pairs[:args.max_pairs]

    run_identity = not args.skip_identity
    verbose = not args.quiet

    exp = DepthExperiment(
        args.data_root,
        args.sequence,
        selected_modes,
        frame_pairs=frame_pairs,
        run_identity=run_identity,
        visualize=args.visualize,
        verbose=verbose,
        predictor_checkpoint=args.depth_checkpoint,
        predictor_device=args.predictor_device,
        patch_coverage=args.patch_coverage,
    )

    exp.run_full_experiment()
