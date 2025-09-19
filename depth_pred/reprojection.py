#!/usr/bin/env python3
"""
Reprojection functions using Kannala-Brandt fisheye model.
Clean implementation following the correct pose chain.
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Dict, Literal
from enum import Enum


class DepthMode(Enum):
    """Depth modes for experiments."""
    DENSE_GT = "dense_gt"
    SPARSE_GT = "sparse_gt"
    SPARSE_PRED = "sparse_pred"


def _kb_r(theta, k):
    """Forward KB model: r = θ + k₁θ³ + k₂θ⁵ + k₃θ⁷ + k₄θ⁹"""
    t2 = theta * theta
    t3 = theta * t2
    t5 = t3 * t2
    t7 = t5 * t2
    t9 = t7 * t2
    return theta + k[0]*t3 + k[1]*t5 + k[2]*t7 + k[3]*t9


def _kb_r_prime(theta, k):
    """Derivative of KB model for Newton's method"""
    t2 = theta * theta
    t4 = t2 * t2
    t6 = t4 * t2
    t8 = t4 * t4
    return 1.0 + 3.0*k[0]*t2 + 5.0*k[1]*t4 + 7.0*k[2]*t6 + 9.0*k[3]*t8


def solve_kb_inverse(r_d, k, max_iter=20, tol=1e-12):
    """Newton solver for theta from r_d = r(theta). Vectorized."""
    theta = np.clip(r_d.astype(np.float64), 0.0, np.deg2rad(89.5))
    for _ in range(max_iter):
        f = _kb_r(theta, k) - r_d
        if np.max(np.abs(f)) < tol:
            break
        theta -= f / _kb_r_prime(theta, k)
    return theta


def fisheye_unproject_native(pixels_xy, K, dist4):
    """
    Native fisheye unprojection using KB model.

    Args:
        pixels_xy: (N,2) pixel coords
        K: 3x3 camera matrix
        dist4: KB distortion coefficients [k1, k2, k3, k4]

    Returns:
        rays_cam: (N,3) direction vectors ~ [tanθ cosψ, tanθ sinψ, 1]
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Normalized image coordinates
    u = (pixels_xy[:, 0] - cx) / fx
    v = (pixels_xy[:, 1] - cy) / fy

    # Distorted radius and angle
    r_d = np.sqrt(u*u + v*v) + 1e-16
    psi = np.arctan2(v, u)

    # Solve for undistorted angle θ
    theta = solve_kb_inverse(r_d, dist4[:4])

    # Convert to 3D ray using tan(θ)
    t = np.tan(theta)
    rays = np.stack([t*np.cos(psi), t*np.sin(psi), np.ones_like(t)], axis=-1)

    return rays


def unproject_kb8(pixels: np.ndarray, K: np.ndarray, dist_coeffs: np.ndarray,
                  depths: np.ndarray) -> np.ndarray:
    """
    Unproject pixels to 3D using Kannala-Brandt model.

    Args:
        pixels: Nx2 array of pixel coordinates (x, y)
        K: 3x3 camera intrinsic matrix
        dist_coeffs: Distortion coefficients [k1, k2, k3, k4]
        depths: N array of depth values in meters

    Returns:
        Nx3 array of 3D points in camera frame
    """
    # Use cv2.fisheye.undistortPoints for KB inverse (proven equivalent to native)
    pixels_cv = pixels.reshape(-1, 1, 2).astype(np.float32)

    # Undistort to get normalized rays [tan(θ)cos(ψ), tan(θ)sin(ψ)]
    rays = cv2.fisheye.undistortPoints(
        pixels_cv,
        K=K,
        D=dist_coeffs[:4].reshape((4, 1))
    )
    rays = rays.reshape(-1, 2)

    # Create 3D points by scaling rays with depth
    # cv2 returns [x/z, y/z], so reconstruct [x, y, z] = [x/z * d, y/z * d, d]
    points_3d = np.zeros((len(depths), 3))
    points_3d[:, 0] = rays[:, 0] * depths  # x
    points_3d[:, 1] = rays[:, 1] * depths  # y
    points_3d[:, 2] = depths               # z

    return points_3d


def project_kb8(points_3d: np.ndarray, K: np.ndarray, dist_coeffs: np.ndarray) -> np.ndarray:
    """
    Project 3D points to pixels using Kannala-Brandt model.

    Args:
        points_3d: Nx3 array of 3D points in camera frame
        K: 3x3 camera intrinsic matrix
        dist_coeffs: Distortion coefficients [k1, k2, k3, k4]

    Returns:
        Nx2 array of pixel coordinates
    """
    # Filter points behind camera
    valid = points_3d[:, 2] > 0
    if not np.any(valid):
        return np.empty((0, 2))

    points_3d_valid = points_3d[valid]

    # Reshape for cv2
    points_3d_cv = points_3d_valid.reshape(-1, 1, 3).astype(np.float32)

    # Project using fisheye model
    pixels, _ = cv2.fisheye.projectPoints(
        points_3d_cv,
        rvec=np.zeros(3, dtype=np.float32),
        tvec=np.zeros(3, dtype=np.float32),
        K=K,
        D=dist_coeffs[:4].reshape((4, 1))
    )

    pixels = pixels.reshape(-1, 2)

    # Return with validity mask
    all_pixels = np.full((len(points_3d), 2), -1, dtype=np.float32)
    all_pixels[valid] = pixels

    return all_pixels


def transform_points_3d(points_3d: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Transform 3D points using 4x4 transformation matrix.

    Args:
        points_3d: Nx3 array of 3D points
        T: 4x4 transformation matrix

    Returns:
        Nx3 array of transformed points
    """
    # Convert to homogeneous
    points_homo = np.hstack([points_3d, np.ones((len(points_3d), 1))])

    # Transform
    points_transformed = (T @ points_homo.T).T

    # Back to 3D
    return points_transformed[:, :3]


def reproject_mask(
    mask: np.ndarray,
    depth: np.ndarray,
    T_camera2_camera1: np.ndarray,
    K: np.ndarray,
    dist_coeffs: np.ndarray,
    depth_mode: DepthMode,
    gaze_point: Optional[Tuple[int, int]] = None,
    patch_size: int = 22,
    predicted_depth: Optional[float] = None,
    sparse_info: Optional[Dict] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Reproject a mask from camera1 to camera2.

    Args:
        mask: Binary mask in camera1 (0 or 255)
        depth: Depth map in camera1 (meters)
        T_camera2_camera1: 4x4 relative transform
        K: 3x3 camera intrinsic matrix
        dist_coeffs: Fisheye distortion coefficients
        depth_mode: How to handle depth (dense, sparse GT, sparse pred)
        gaze_point: (x, y) gaze location for sparse modes
        patch_size: Size of patch for sparse modes

    Returns:
        reprojected_mask: Binary mask in camera2
        info: Dict with statistics
    """
    h, w = mask.shape

    # Get mask pixels
    mask_coords = np.column_stack(np.where(mask > 0))  # (y, x)
    mask_pixels = mask_coords[:, [1, 0]]  # Convert to (x, y)

    if len(mask_pixels) == 0:
        return np.zeros_like(mask), {'num_pixels': 0, 'error': 'Empty mask'}

    # Handle depth based on mode
    if depth_mode == DepthMode.DENSE_GT:
        # Use actual depth at each pixel
        mask_depths = depth[mask_coords[:, 0], mask_coords[:, 1]]

    elif depth_mode == DepthMode.SPARSE_GT:
        # Use averaged depth from patch around gaze
        if gaze_point is None:
            return np.zeros_like(mask), {'error': 'No gaze point for sparse mode'}

        gx, gy = gaze_point
        half = patch_size // 2

        # Extract patch
        y_min = max(0, gy - half)
        y_max = min(h, gy + half)
        x_min = max(0, gx - half)
        x_max = min(w, gx + half)

        patch = depth[y_min:y_max, x_min:x_max]
        valid_depths = patch[patch > 0.1]

        if len(valid_depths) == 0:
            return np.zeros_like(mask), {'error': 'No valid depth in patch'}

        # Use mean depth for all mask pixels to simulate averaged sparse depth
        sparse_depth = float(np.mean(valid_depths))
        mask_depths = np.full(len(mask_pixels), sparse_depth, dtype=np.float32)

        info_sparse = {
            'sparse_depth_mean': sparse_depth,
            'patch_valid_pixels': int(len(valid_depths)),
            'patch_depth_min': float(valid_depths.min()),
            'patch_depth_max': float(valid_depths.max())
        }

    elif depth_mode == DepthMode.SPARSE_PRED:
        if predicted_depth is None:
            return np.zeros_like(mask), {
                'error': 'No predicted depth provided for sparse_pred mode'
            }

        sparse_depth = float(predicted_depth)
        mask_depths = np.full(len(mask_pixels), sparse_depth, dtype=np.float32)

        info_sparse = {
            'sparse_depth_pred': sparse_depth
        }
        if sparse_info:
            info_sparse.update(sparse_info)

    else:
        raise ValueError(f"Unknown depth mode: {depth_mode}")

    # Filter invalid depths
    valid = mask_depths > 0.1
    mask_pixels = mask_pixels[valid]
    mask_depths = mask_depths[valid]

    if len(mask_pixels) == 0:
        return np.zeros_like(mask), {'error': 'No valid depths'}

    # Unproject to 3D
    points_3d_cam1 = unproject_kb8(mask_pixels, K, dist_coeffs, mask_depths)

    # Transform to camera2
    points_3d_cam2 = transform_points_3d(points_3d_cam1, T_camera2_camera1)

    # Project to camera2
    pixels_cam2 = project_kb8(points_3d_cam2, K, dist_coeffs)

    # Create output mask with z-buffer for occlusion handling
    mask_reprojected = np.zeros((h, w), dtype=np.uint8)
    zbuffer = np.full((h, w), np.inf, dtype=np.float32)
    num_valid = 0

    # Process each projected pixel with z-buffer test
    for i, pixel in enumerate(pixels_cam2):
        if pixel[0] < 0:  # Invalid projection
            continue

        x, y = int(pixel[0] + 0.5), int(pixel[1] + 0.5)
        if 0 <= x < w and 0 <= y < h:
            # Z-buffer test: only update if this point is closer
            z_value = points_3d_cam2[i, 2]
            if z_value < zbuffer[y, x]:
                zbuffer[y, x] = z_value
                mask_reprojected[y, x] = 255
                num_valid += 1

    # Fill holes with morphological operations
    if num_valid > 100:  # Only if we have enough points
        kernel = np.ones((3, 3), np.uint8)
        mask_filled = cv2.morphologyEx(mask_reprojected, cv2.MORPH_CLOSE, kernel)
    else:
        mask_filled = mask_reprojected

    # Prepare info dict
    info = {
        'num_mask_pixels': len(mask_coords),
        'num_valid_depths': len(mask_pixels),
        'num_projected': num_valid,
        'depth_mode': depth_mode.value
    }

    if depth_mode in (DepthMode.SPARSE_GT, DepthMode.SPARSE_PRED):
        info.update(info_sparse)

    return mask_filled, info


def compute_reprojection_error(
    mask: np.ndarray,
    depth: np.ndarray,
    T_camera2_camera1: np.ndarray,
    K: np.ndarray,
    dist_coeffs: np.ndarray,
    mask_gt_cam2: np.ndarray
) -> Dict:
    """
    Compute reprojection error statistics.

    Args:
        mask: Binary mask in camera1
        depth: Depth map in camera1
        T_camera2_camera1: Relative transform
        K: Camera intrinsics
        dist_coeffs: Distortion coefficients
        mask_gt_cam2: Ground truth mask in camera2

    Returns:
        Dict with error statistics
    """
    # Get corresponding points
    mask_coords = np.column_stack(np.where(mask > 0))
    if len(mask_coords) == 0:
        return {'error': 'Empty mask'}

    # Sample points for efficiency
    if len(mask_coords) > 1000:
        indices = np.random.choice(len(mask_coords), 1000, replace=False)
        mask_coords = mask_coords[indices]

    mask_pixels = mask_coords[:, [1, 0]]
    mask_depths = depth[mask_coords[:, 0], mask_coords[:, 1]]

    # Filter valid depths
    valid = mask_depths > 0.1
    mask_pixels = mask_pixels[valid]
    mask_depths = mask_depths[valid]

    # Project
    points_3d_cam1 = unproject_kb8(mask_pixels, K, dist_coeffs, mask_depths)
    points_3d_cam2 = transform_points_3d(points_3d_cam1, T_camera2_camera1)
    pixels_cam2 = project_kb8(points_3d_cam2, K, dist_coeffs)

    # Find corresponding GT pixels (if available)
    errors = []
    h, w = mask_gt_cam2.shape

    for i, pixel in enumerate(pixels_cam2):
        if pixel[0] < 0:
            continue

        x, y = int(pixel[0] + 0.5), int(pixel[1] + 0.5)
        if 0 <= x < w and 0 <= y < h:
            # Check if GT mask has object at this location
            if mask_gt_cam2[y, x] > 0:
                # Could compute distance to nearest GT pixel
                errors.append(0)  # Simplified: just check hit/miss
            else:
                # Find nearest GT pixel
                gt_coords = np.column_stack(np.where(mask_gt_cam2 > 0))
                if len(gt_coords) > 0:
                    gt_pixels = gt_coords[:, [1, 0]]
                    dists = np.linalg.norm(gt_pixels - np.array([x, y]), axis=1)
                    errors.append(np.min(dists))

    if len(errors) == 0:
        return {'error': 'No valid projections'}

    errors = np.array(errors)

    return {
        'mean_error': np.mean(errors),
        'median_error': np.median(errors),
        'percentile_95': np.percentile(errors, 95) if len(errors) > 0 else 0,
        'num_points': len(errors)
    }


def compute_iou(mask1: np.ndarray, mask2: np.ndarray,
                mode: Literal["strict", "fill_aware"] = "strict") -> float:
    """
    Compute IoU between two masks.

    Args:
        mask1: First binary mask
        mask2: Second binary mask
        mode: "strict" or "fill_aware"

    Returns:
        IoU score
    """
    if mode == "fill_aware":
        # Fill holes before comparison
        kernel = np.ones((5, 5), np.uint8)
        mask1 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernel)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)

    # Compute IoU
    intersection = np.logical_and(mask1 > 0, mask2 > 0).sum()
    union = np.logical_or(mask1 > 0, mask2 > 0).sum()

    if union == 0:
        return 0.0

    return intersection / union


if __name__ == "__main__":
    print("Reprojection utilities loaded.")
    print("Available functions:")
    print("  - unproject_kb8: Unproject pixels to 3D")
    print("  - project_kb8: Project 3D points to pixels")
    print("  - reproject_mask: Full mask reprojection pipeline")
    print("  - compute_reprojection_error: Error statistics")
    print("  - compute_iou: IoU computation")
    print("\nDepth modes:")
    for mode in DepthMode:
        print(f"  - {mode.value}")
