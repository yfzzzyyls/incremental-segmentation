"""
Control logic for mask reuse pipeline based on MASK_REUSE_PIPELINE_DETAILED.md
Implements the core decision logic for SEG vs REUSE and mask projection.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass
import cv2


@dataclass
class CachedFrame:
    """Cached frame data for mask reuse"""
    frame_id: int
    rgb: np.ndarray  # (H, W, 3)
    mask: np.ndarray  # (H, W) binary mask
    depth: np.ndarray  # (H, W) depth map
    pose: np.ndarray  # 4x4 transformation matrix (world to camera)
    plane_params: Optional[np.ndarray] = None  # (4,) plane equation [a, b, c, d]
    timestamp_us: float = 0.0
    gaze_point: Optional[Tuple[int, int]] = None  # Gaze point when cached
    intrinsics: Optional[np.ndarray] = None  # Camera intrinsics K matrix


@dataclass
class ControlInput:
    """Input data for control decision"""
    rgb: np.ndarray  # Current RGB frame
    depth: np.ndarray  # Current depth map
    pose: np.ndarray  # Current 4x4 pose matrix (world to camera)
    gaze_pixel: Tuple[int, int]  # Current gaze point in pixels
    timestamp_us: float
    intrinsics: np.ndarray  # 3x3 camera intrinsics matrix
    distortion_coeffs: Optional[np.ndarray] = None  # Fisheye distortion coefficients [k1, k2, k3, k4]


@dataclass
class ControlOutput:
    """Output of control decision"""
    decision: str  # 'SEG' or 'REUSE'
    mask: Optional[np.ndarray] = None  # Projected mask if REUSE
    confidence: float = 0.0  # Confidence score for decision
    reason: str = ""  # Reason for decision
    cached_frame_id: Optional[int] = None  # ID of reused frame if applicable


class MaskReuseController:
    """
    Controller for mask reuse pipeline.
    Implements the logic from MASK_REUSE_PIPELINE_DETAILED.md
    
    IMPORTANT: Pose Convention
    ADT provides camera-to-world transforms (T_world_camera).
    Column names tx_world_device mean "device position in world coordinates".
    We use ADT poses as-is without inversion.
    - To go world->camera: inv(pose)
    - To go camera->world: pose
    """
    
    def __init__(
        self,
        appearance_threshold: float = 0.85,
        geometric_threshold: float = 0.90,
        min_mask_overlap: float = 0.5,
        max_depth_error: float = 0.1,  # 10cm
        min_visible_ratio: float = 0.7,
        photometric_threshold: float = 0.8
    ):
        """
        Initialize controller with thresholds.
        
        Args:
            appearance_threshold: Min correlation for appearance similarity
            geometric_threshold: Min IoU for geometric validation
            min_mask_overlap: Min overlap between gaze and cached mask
            max_depth_error: Max depth error for plane fitting (meters)
            min_visible_ratio: Min ratio of mask visible after projection
            photometric_threshold: Min photometric consistency score
        """
        self.appearance_threshold = appearance_threshold
        self.geometric_threshold = geometric_threshold
        self.min_mask_overlap = min_mask_overlap
        self.max_depth_error = max_depth_error
        self.min_visible_ratio = min_visible_ratio
        self.photometric_threshold = photometric_threshold
    
    def process(
        self,
        current: ControlInput,
        cached_frames: Dict[int, CachedFrame]
    ) -> ControlOutput:
        """
        Main control logic: decide SEG vs REUSE and perform projection.
        
        CORRECTED LOGIC:
        1. First project gaze to 3D
        2. Find ALL cached frames where gaze hits a mask
        3. Among those candidates, pick best by pose similarity
        4. Validate and warp the selected mask
        
        Args:
            current: Current frame input data
            cached_frames: Dictionary of cached frames
            
        Returns:
            ControlOutput with decision and projected mask if applicable
        """
        if not cached_frames:
            return ControlOutput(
                decision="SEG",
                reason="No cached frames available"
            )
        
        # Step 1: Project current gaze to 3D
        gaze_3d = self._project_gaze_to_3d(
            current.gaze_pixel,
            current.depth,
            current.intrinsics,
            current.distortion_coeffs
        )
        
        if gaze_3d is None:
            return ControlOutput(
                decision="SEG",
                reason="Invalid gaze projection (no depth at gaze)"
            )
        
        # Step 2: Find ALL cached frames where gaze lands in a mask
        candidates = []
        for frame_id, cached_frame in cached_frames.items():
            # Check basic preconditions
            if not self._validate_preconditions(current, cached_frame):
                continue
            
            # Project gaze to this cached view
            K_cached = cached_frame.intrinsics if cached_frame.intrinsics is not None else current.intrinsics
            gaze_in_cached = self._map_to_cached_view(
                gaze_3d,
                current.pose,
                cached_frame.pose,
                cached_frame.rgb.shape[:2],
                K_cached
            )
            
            # Skip if gaze is out of bounds
            if gaze_in_cached is None:
                continue
            
            # Check if gaze hits the mask
            if self._gaze_in_mask(gaze_in_cached, cached_frame.mask):
                # Compute pose similarity score for ranking
                T_rel = np.linalg.inv(current.pose) @ cached_frame.pose
                translation_dist = np.linalg.norm(T_rel[:3, 3])
                rotation_angle = np.arccos(np.clip((np.trace(T_rel[:3, :3]) - 1) / 2, -1, 1))
                pose_score = np.exp(-translation_dist) * np.exp(-rotation_angle)
                
                candidates.append({
                    'frame_id': frame_id,
                    'frame': cached_frame,
                    'gaze_in_cached': gaze_in_cached,
                    'pose_score': pose_score,
                    'translation_dist': translation_dist,
                    'rotation_angle': rotation_angle
                })
        
        # Step 3: No candidates where gaze hits a mask
        if not candidates:
            return ControlOutput(
                decision="SEG",
                reason="Gaze does not hit any cached masks"
            )
        
        # Step 4: Select best candidate by pose similarity
        best_candidate = max(candidates, key=lambda x: x['pose_score'])
        cached_frame = best_candidate['frame']
        best_match = best_candidate['frame_id']
        
        # Log why we chose this frame (for debugging)
        print(f"  Selected cache frame {best_match}: pose_score={best_candidate['pose_score']:.3f}, "
              f"trans={best_candidate['translation_dist']:.2f}m, rot={np.degrees(best_candidate['rotation_angle']):.1f}°")
        
        # Step 5: Get or compute plane parameters
        if cached_frame.plane_params is not None:
            plane_params = cached_frame.plane_params
        else:
            # Fallback: fit plane if not cached
            K_cached = cached_frame.intrinsics if cached_frame.intrinsics is not None else current.intrinsics
            plane_params = self._fit_plane_to_mask(
                cached_frame.mask,
                cached_frame.depth,
                K_cached
            )
            
            if plane_params is None:
                return ControlOutput(
                    decision="SEG",
                    reason="Plane fitting failed",
                    cached_frame_id=best_match
                )
        
        # Step 6: Compute homography and warp mask
        K_cached = cached_frame.intrinsics if cached_frame.intrinsics is not None else current.intrinsics
        homography = self._compute_planar_homography(
            plane_params,
            cached_frame.pose,
            current.pose,
            K_cached,
            current.intrinsics
        )
        
        if homography is None:
            return ControlOutput(
                decision="SEG",
                reason="Homography computation failed",
                cached_frame_id=best_match
            )
        
        warped_mask = self._warp_mask(cached_frame.mask, homography, current.rgb.shape[:2])
        
        # Step 7: Validate warped mask
        validation_result = self._validate_warped_mask(
            warped_mask,
            current.rgb,
            cached_frame.rgb,
            cached_frame.mask,
            homography
        )
        
        if not validation_result['valid']:
            return ControlOutput(
                decision="SEG",
                reason=validation_result['reason'],
                cached_frame_id=best_match
            )
        
        # Success: REUSE the mask
        return ControlOutput(
            decision="REUSE",
            mask=warped_mask,
            confidence=validation_result['confidence'],
            reason=f"Reused from frame {best_match} (found {len(candidates)} candidates)",
            cached_frame_id=best_match
        )
    
    def _find_best_match(
        self,
        current: ControlInput,
        cached_frames: Dict[int, CachedFrame]
    ) -> Optional[int]:
        """
        [DEPRECATED - Logic moved to main process method]
        
        This method is kept for backward compatibility but is no longer used.
        The new logic in process() first checks which cached masks contain
        the gaze point, then selects the best one by pose similarity.
        
        Old behavior: Found best frame by pose similarity alone,
        without checking if gaze hits the mask first.
        """
        if not cached_frames:
            return None
        
        best_score = -1
        best_id = None
        
        for frame_id, cached in cached_frames.items():
            # Compute pose similarity (cached -> current)
            # For camera->world poses: T_rel = inv(current) @ cached
            T_rel = np.linalg.inv(current.pose) @ cached.pose  # For camera->world convention
            translation_dist = np.linalg.norm(T_rel[:3, 3])
            rotation_angle = np.arccos(np.clip((np.trace(T_rel[:3, :3]) - 1) / 2, -1, 1))
            
            # Simple scoring: prefer smaller pose differences
            pose_score = np.exp(-translation_dist) * np.exp(-rotation_angle)
            
            if pose_score > best_score:
                best_score = pose_score
                best_id = frame_id
        
        return best_id
    
    def _validate_preconditions(
        self,
        current: ControlInput,
        cached: CachedFrame
    ) -> bool:
        """Validate basic preconditions for mask reuse."""
        # Check if we have all required data
        if current.depth is None or cached.depth is None:
            return False
        
        # Check time difference (example: max 5 seconds)
        time_diff = abs(current.timestamp_us - cached.timestamp_us) / 1e6
        if time_diff > 5.0:
            return False
        
        return True
    
    def _project_gaze_to_3d(
        self,
        gaze_pixel: Tuple[int, int],
        depth: np.ndarray,
        intrinsics: np.ndarray,
        distortion_coeffs: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """
        Project 2D gaze point to 3D using depth.
        
        Supports both pinhole and Kannala-Brandt fisheye models.
        
        Args:
            gaze_pixel: (x, y) pixel coordinates
            depth: Depth map in meters
            intrinsics: 3x3 camera matrix K
            distortion_coeffs: Optional fisheye distortion [k1, k2, k3, k4]
            
        Returns:
            3D point in camera coordinates or None.
        """
        x, y = gaze_pixel
        
        # Check bounds
        if x < 0 or x >= depth.shape[1] or y < 0 or y >= depth.shape[0]:
            return None
        
        # Get depth value
        z = depth[y, x]
        
        if z <= 0 or np.isnan(z) or np.isinf(z):
            return None
        
        # Case 1: Fisheye camera (Kannala-Brandt model)
        if distortion_coeffs is not None and len(distortion_coeffs) >= 4:
            # Convert pixel to array format for cv2
            pixel_array = np.array([[[x, y]]], dtype=np.float32)
            
            # Undistort the point to get normalized coordinates
            # This gives us the ray direction in normalized camera coordinates
            undistorted = cv2.fisheye.undistortPoints(
                pixel_array,
                K=intrinsics,
                D=distortion_coeffs[:4].reshape((4, 1))  # Use first 4 coeffs
            )
            
            # undistorted contains normalized coordinates (x/z, y/z)
            x_norm = undistorted[0, 0, 0]
            y_norm = undistorted[0, 0, 1]
            
            # The ray direction is [x_norm, y_norm, 1]
            # We need to scale it to reach depth z
            # In fisheye, the depth is along the ray, not just z-coordinate
            ray_norm = np.sqrt(x_norm**2 + y_norm**2 + 1)
            
            # Scale ray to reach the measured depth
            # (assuming depth is measured along z-axis, not along ray)
            x_3d = x_norm * z
            y_3d = y_norm * z
            z_3d = z
            
        # Case 2: Pinhole camera (fallback)
        else:
            fx, fy = intrinsics[0, 0], intrinsics[1, 1]
            cx, cy = intrinsics[0, 2], intrinsics[1, 2]
            
            x_3d = (x - cx) * z / fx
            y_3d = (y - cy) * z / fy
            z_3d = z
        
        return np.array([x_3d, y_3d, z_3d])
    
    def _map_to_cached_view(
        self,
        point_3d: np.ndarray,
        current_pose: np.ndarray,
        cached_pose: np.ndarray,
        cached_shape: Tuple[int, int],
        intrinsics: np.ndarray
    ) -> Optional[Tuple[int, int]]:
        """
        Map 3D point from current to cached view.
        
        Args:
            point_3d: 3D point in current camera coordinates
            current_pose: 4x4 world->current camera transform
            cached_pose: 4x4 world->cached camera transform
            cached_shape: (H, W) of cached image
            intrinsics: 3x3 K matrix for cached camera
            
        Returns:
            Pixel coordinates in cached view or None if out of bounds.
        """
        # Transform point from current cam → world → cached cam
        point_cam_cur = np.append(point_3d, 1.0)
        
        # If poses are camera->world (as ADT provides):
        # Current camera to world
        point_world = current_pose @ point_cam_cur  # cam→world
        
        # World to cached camera (inverse of camera->world)
        point_cached = np.linalg.inv(cached_pose) @ point_world  # world→cached cam
        
        # Project with cached intrinsics
        Z = point_cached[2]
        if Z <= 0:
            return None
        
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        
        x = int(fx * point_cached[0] / Z + cx)
        y = int(fy * point_cached[1] / Z + cy)
        
        # Check bounds
        h, w = cached_shape
        if x < 0 or x >= w or y < 0 or y >= h:
            return None
        
        return (x, y)
    
    def _gaze_in_mask(
        self,
        gaze_pixel: Tuple[int, int],
        mask: np.ndarray,
        radius: int = 5
    ) -> bool:
        """
        Check if gaze point falls within mask region.
        Uses a small radius for robustness.
        """
        x, y = gaze_pixel
        h, w = mask.shape
        
        # Check in a small neighborhood
        y_min = max(0, y - radius)
        y_max = min(h, y + radius + 1)
        x_min = max(0, x - radius)
        x_max = min(w, x + radius + 1)
        
        region = mask[y_min:y_max, x_min:x_max]
        
        # Return true if any pixel in region is in mask
        return np.any(region > 0)
    
    def _fit_plane_to_mask(
        self,
        mask: np.ndarray,
        depth: np.ndarray,
        intrinsics: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Fit plane to masked depth points using RANSAC.
        
        Returns plane parameters [a, b, c, d] where ax + by + cz + d = 0.
        """
        # Get mask points
        mask_points = np.where(mask > 0)
        
        if len(mask_points[0]) < 100:  # Need minimum points
            return None
        
        # Sample points
        n_samples = min(1000, len(mask_points[0]))
        indices = np.random.choice(len(mask_points[0]), n_samples, replace=False)
        
        # Get 3D points
        points_3d = []
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        
        for idx in indices:
            y, x = mask_points[0][idx], mask_points[1][idx]
            z = depth[y, x]
            
            if z > 0 and not np.isnan(z):
                x_3d = (x - cx) * z / fx
                y_3d = (y - cy) * z / fy
                points_3d.append([x_3d, y_3d, z])
        
        if len(points_3d) < 100:
            return None
        
        points_3d = np.array(points_3d)
        
        # Simple plane fitting using SVD
        centroid = np.mean(points_3d, axis=0)
        centered = points_3d - centroid
        
        # SVD to find normal
        U, s, vh = np.linalg.svd(centered)
        normal = vh[-1, :]  # Last row of vh is the normal (smallest singular value)
        
        # Normalize
        normal = normal / np.linalg.norm(normal)
        
        # Compute d
        d = -np.dot(normal, centroid)
        
        # Enforce canonical sign: ensure d is negative so that -d is positive
        # This ensures consistency in _compute_planar_homography where we use -plane[3]
        if d >= 0:
            normal = -normal
            d = -d
        
        # Return plane parameters [a, b, c, d] where ax + by + cz + d = 0
        return np.array([normal[0], normal[1], normal[2], d])
    
    def _compute_planar_homography(
        self,
        plane: np.ndarray,
        cached_pose: np.ndarray,
        current_pose: np.ndarray,
        K_cached: np.ndarray,
        K_current: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Compute homography induced by plane between two views.
        
        Args:
            plane: Plane parameters [a, b, c, d] where ax + by + cz + d = 0
            cached_pose: 4x4 pose of cached frame
            current_pose: 4x4 pose of current frame
            K_cached: 3x3 intrinsics of cached camera
            K_current: 3x3 intrinsics of current camera
            
        Returns:
            3x3 homography matrix or None
        """
        # Get relative transformation (cached -> current)
        # If poses are camera->world (as ADT provides), then:
        # T_rel = T_current_world^(-1) @ T_cached_world = inv(current) @ cached
        # This maps cached cam coords to current cam coords
        T_rel = np.linalg.inv(current_pose) @ cached_pose  # For camera->world convention
        R = T_rel[:3, :3]
        t = T_rel[:3, 3:4]
        
        # Plane normal and distance in cached camera frame
        n = plane[:3].reshape(3, 1)
        d = -plane[3]
        
        if abs(d) < 1e-6:
            return None
        
        # Compute homography: H = K_current * (R - t * n^T / d) * K_cached^-1
        K_cached_inv = np.linalg.inv(K_cached)
        
        H = K_current @ (R - t @ n.T / d) @ K_cached_inv
        
        # Normalize with safety check
        if abs(H[2, 2]) < 1e-8:
            # Homography is degenerate
            return None
        H = H / H[2, 2]
        
        return H
    
    def _warp_mask(
        self,
        mask: np.ndarray,
        homography: np.ndarray,
        target_shape: Tuple[int, int]
    ) -> np.ndarray:
        """Warp mask using homography."""
        h, w = target_shape
        warped = cv2.warpPerspective(
            mask.astype(np.uint8),
            homography,
            (w, h),
            flags=cv2.INTER_NEAREST,  # Use nearest neighbor for label images
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        # Ensure binary mask
        warped = (warped > 0).astype(np.uint8) * 255
        
        return warped
    
    def _validate_warped_mask(
        self,
        warped_mask: np.ndarray,
        current_rgb: np.ndarray,
        cached_rgb: np.ndarray,
        cached_mask: np.ndarray,
        homography: np.ndarray
    ) -> Dict[str, Any]:
        """
        Validate warped mask using multiple criteria.
        
        Returns dict with 'valid' bool and 'reason' string.
        """
        # Check if mask is too small
        mask_pixels = np.sum(warped_mask > 0)
        total_pixels = warped_mask.shape[0] * warped_mask.shape[1]
        
        if mask_pixels < 100:  # Too few pixels
            return {
                'valid': False,
                'reason': 'Warped mask too small',
                'confidence': 0.0
            }
        
        # Check visibility ratio
        original_pixels = np.sum(cached_mask > 0)
        # Clamp visibility ratio to [0, 1] as it can exceed 1.0 due to scale/pose changes
        visibility_ratio = min(1.0, mask_pixels / max(original_pixels, 1))
        
        if visibility_ratio < self.min_visible_ratio:
            return {
                'valid': False,
                'reason': f'Low visibility ratio: {visibility_ratio:.2f}',
                'confidence': visibility_ratio
            }
        
        # Photometric consistency check
        # Warp cached RGB for comparison
        h, w = current_rgb.shape[:2]
        warped_cached_rgb = cv2.warpPerspective(
            cached_rgb,
            homography,
            (w, h),
            flags=cv2.INTER_LINEAR
        )
        
        # Compare in masked region
        mask_bool = warped_mask > 0
        if np.any(mask_bool):
            current_masked = current_rgb[mask_bool]
            cached_masked = warped_cached_rgb[mask_bool]
            
            # Compute normalized correlation
            if len(current_masked) > 0 and len(cached_masked) > 0:
                current_norm = current_masked - np.mean(current_masked, axis=0)
                cached_norm = cached_masked - np.mean(cached_masked, axis=0)
                
                correlation = np.mean(
                    np.sum(current_norm * cached_norm, axis=1) /
                    (np.linalg.norm(current_norm, axis=1) * np.linalg.norm(cached_norm, axis=1) + 1e-6)
                )
                
                if correlation < self.photometric_threshold:
                    return {
                        'valid': False,
                        'reason': f'Low photometric consistency: {correlation:.2f}',
                        'confidence': correlation
                    }
            else:
                correlation = 0.0
        else:
            correlation = 0.0
        
        # All checks passed
        confidence = min(visibility_ratio, correlation)
        
        return {
            'valid': True,
            'reason': 'All checks passed',
            'confidence': confidence
        }