#!/usr/bin/env python3
"""
Correct pose handling utilities for ADT dataset.
Based on detailed analysis of ADT conventions and ORB-SLAM3 implementations.
"""

import numpy as np
from scipy.spatial.transform import Rotation
from typing import Tuple, Dict
import pandas as pd
import json
from pathlib import Path


def quaternion_to_rotation_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """
    Convert quaternion to rotation matrix using Hamilton convention.

    Args:
        qx, qy, qz, qw: Quaternion components as stored in ADT CSV

    Returns:
        3x3 rotation matrix
    """
    # CSV stores as (qx, qy, qz, qw) but scipy expects [x, y, z, w]
    # This already matches! Just normalize
    norm = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
    qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm

    # Use scipy for consistency
    R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    return R


def build_transform_from_csv_row(row: pd.Series) -> np.ndarray:
    """
    Build T_world_device from ADT trajectory CSV row.

    Args:
        row: Pandas Series containing qx_world_device, tx_world_device, etc.

    Returns:
        4x4 transformation matrix T_world_device
    """
    # Extract quaternion (CSV order is qx, qy, qz, qw)
    qx = row['qx_world_device']
    qy = row['qy_world_device']
    qz = row['qz_world_device']
    qw = row['qw_world_device']

    # Build rotation matrix
    R = quaternion_to_rotation_matrix(qx, qy, qz, qw)

    # Extract translation
    tx = row['tx_world_device']
    ty = row['ty_world_device']
    tz = row['tz_world_device']

    # Build 4x4 transform (T_world_device)
    T_world_device = np.eye(4)
    T_world_device[:3, :3] = R
    T_world_device[:3, 3] = [tx, ty, tz]

    return T_world_device


def get_camera_pose_from_device(T_world_device: np.ndarray,
                                T_device_camera: np.ndarray) -> np.ndarray:
    """
    Convert device pose to camera pose.

    Args:
        T_world_device: Device pose in world frame
        T_device_camera: Camera extrinsics from calibration

    Returns:
        T_world_camera: Camera pose in world frame
    """
    # T_w_c = T_w_d @ T_d_c
    T_world_camera = T_world_device @ T_device_camera
    return T_world_camera


def compute_relative_transform(T_world_camera1: np.ndarray,
                               T_world_camera2: np.ndarray) -> np.ndarray:
    """
    Compute relative transform from camera1 to camera2.

    Args:
        T_world_camera1: Camera 1 pose in world
        T_world_camera2: Camera 2 pose in world

    Returns:
        T_camera2_camera1: Transform to map points from camera1 to camera2
    """
    # T_c2_c1 = T_c2_w @ T_w_c1 = inv(T_w_c2) @ T_w_c1
    T_camera2_camera1 = np.linalg.inv(T_world_camera2) @ T_world_camera1
    return T_camera2_camera1


class PoseManager:
    """Manager class for handling ADT poses correctly."""

    def __init__(self, trajectory_csv_path: Path, calibration_json_path: Path):
        """
        Initialize with ADT trajectory and calibration.

        Args:
            trajectory_csv_path: Path to aria_trajectory.csv
            calibration_json_path: Path to calibration.json with T_device_camera
        """
        # Load trajectory
        self.trajectory = pd.read_csv(trajectory_csv_path)
        self.trajectory['timestamp_ns'] = self.trajectory['tracking_timestamp_us'] * 1000

        # Load calibration
        with open(calibration_json_path, 'r') as f:
            calib = json.load(f)

        # Get T_device_camera
        self.T_device_camera = np.array(calib['T_device_camera'])

        # Cache for computed camera poses
        self._camera_pose_cache = {}

    def get_camera_pose_at_timestamp(self, timestamp_ns: int) -> np.ndarray:
        """
        Get camera pose at given timestamp.

        Args:
            timestamp_ns: Timestamp in nanoseconds

        Returns:
            T_world_camera at the given timestamp
        """
        # Check cache
        if timestamp_ns in self._camera_pose_cache:
            return self._camera_pose_cache[timestamp_ns]

        # Find closest trajectory entry
        idx = np.argmin(np.abs(self.trajectory['timestamp_ns'].values - timestamp_ns))
        row = self.trajectory.iloc[idx]

        # Check timestamp alignment
        time_diff_ms = abs(row['timestamp_ns'] - timestamp_ns) / 1e6
        if time_diff_ms > 1.0:  # More than 1ms difference
            print(f"Warning: Large timestamp difference {time_diff_ms:.2f}ms")

        # Build device pose
        T_world_device = build_transform_from_csv_row(row)

        # Convert to camera pose
        T_world_camera = get_camera_pose_from_device(T_world_device, self.T_device_camera)

        # Cache result
        self._camera_pose_cache[timestamp_ns] = T_world_camera

        return T_world_camera

    def get_camera_pose_for_frame(self, frame_idx: int, metadata: Dict) -> np.ndarray:
        """
        Get camera pose for a specific frame index.

        Args:
            frame_idx: Frame index
            metadata: Metadata dict with frame timestamps

        Returns:
            T_world_camera for the frame
        """
        frame_info = metadata['frames'][frame_idx]
        timestamp_ns = frame_info['rgb_timestamp_ns']
        return self.get_camera_pose_at_timestamp(timestamp_ns)

    def compute_relative_pose(self, frame1_idx: int, frame2_idx: int,
                             metadata: Dict) -> Tuple[np.ndarray, Dict]:
        """
        Compute relative transform between two frames.

        Args:
            frame1_idx: Source frame index
            frame2_idx: Target frame index
            metadata: Metadata dict

        Returns:
            T_camera2_camera1: Relative transform
            info: Dict with additional info (translation, rotation angle)
        """
        # Get camera poses
        T_w_c1 = self.get_camera_pose_for_frame(frame1_idx, metadata)
        T_w_c2 = self.get_camera_pose_for_frame(frame2_idx, metadata)

        # Compute relative
        T_c2_c1 = compute_relative_transform(T_w_c1, T_w_c2)

        # Extract motion statistics
        translation = np.linalg.norm(T_c2_c1[:3, 3])
        R_rel = T_c2_c1[:3, :3]
        rotation_angle = np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1))
        rotation_angle_deg = np.degrees(rotation_angle)

        info = {
            'translation_m': translation,
            'rotation_deg': rotation_angle_deg,
            'T_world_camera1': T_w_c1,
            'T_world_camera2': T_w_c2
        }

        return T_c2_c1, info


def sanity_check_poses(pose_manager: PoseManager, metadata: Dict):
    """
    Run sanity checks to verify pose handling is correct.

    Args:
        pose_manager: PoseManager instance
        metadata: Metadata dict
    """
    print("="*60)
    print("POSE SANITY CHECKS")
    print("="*60)

    # Check 1: Identity transform
    print("\n1. Identity Transform Check:")
    frame_idx = 179
    T_w_c = pose_manager.get_camera_pose_for_frame(frame_idx, metadata)
    T_identity = compute_relative_transform(T_w_c, T_w_c)

    # Should be identity matrix
    identity_error = np.linalg.norm(T_identity - np.eye(4))
    print(f"   ||T_identity - I|| = {identity_error:.6f}")
    print(f"   ✅ PASS" if identity_error < 1e-10 else f"   ❌ FAIL")

    # Check 2: Consecutive frame motion
    print("\n2. Consecutive Frame Motion Check:")
    T_rel, info = pose_manager.compute_relative_pose(179, 180, metadata)
    print(f"   Frame 179→180:")
    print(f"   Translation: {info['translation_m']*1000:.1f}mm")
    print(f"   Rotation: {info['rotation_deg']:.2f}°")

    # Should be small for 30Hz video
    reasonable = (info['translation_m'] < 0.02 and info['rotation_deg'] < 5)
    print(f"   ✅ PASS (reasonable motion)" if reasonable else f"   ⚠️  Large motion for 30Hz")

    # Check 3: Baseline comparisons
    print("\n3. Different Baseline Checks:")
    baselines = [
        (179, 183, "Small"),
        (179, 195, "Medium"),
        (179, 199, "Large")
    ]

    for f1, f2, name in baselines:
        T_rel, info = pose_manager.compute_relative_pose(f1, f2, metadata)
        print(f"   {name} baseline ({f1}→{f2}):")
        print(f"     Translation: {info['translation_m']*100:.1f}cm")
        print(f"     Rotation: {info['rotation_deg']:.1f}°")

    print("\n" + "="*60)


if __name__ == "__main__":
    # Test the pose utilities
    import sys

    data_root = Path("/mnt/ssd_ext/incSeg-data")
    sequence = "Apartment_release_clean_seq148_M1292"

    # Find paths
    traj_path = data_root / "adt" / "test" / sequence / "aria_trajectory.csv"
    calib_path = data_root / "processed_adt" / "test" / sequence / "calibration.json"
    meta_path = data_root / "processed_adt" / "test" / sequence / "metadata.json"

    if not all(p.exists() for p in [traj_path, calib_path, meta_path]):
        print("Error: Required files not found")
        sys.exit(1)

    # Load metadata
    with open(meta_path, 'r') as f:
        metadata = json.load(f)

    # Create pose manager
    pose_mgr = PoseManager(traj_path, calib_path)

    # Run sanity checks
    sanity_check_poses(pose_mgr, metadata)