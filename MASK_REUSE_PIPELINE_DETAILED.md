# Mask Reuse Pipeline: Detailed Step-by-Step Analysis

## Overview
This document provides a comprehensive analysis of each step in the mask reuse pipeline, with concrete examples and failure cases that would occur without each logical step.

---

## Step 0: Preconditions
**What it does:** Validates fundamental requirements before any geometric computation.

### 0a. Time Alignment
**Purpose:** Ensures all data (pose, gaze, depth) are from the same moment in time.

**Concrete Example:**
```python
# Scenario: User turning head at 150°/s
gaze_timestamp = 100.000ms  # Eye tracker sample
frame_timestamp = 100.045ms  # 45ms later (typical latency)
pose_timestamp = 100.010ms   # IMU update

# During 45ms at 150°/s = 6.75° rotation = ~120 pixels error!

# WITHOUT time alignment:
gaze_3d = unproject(gaze_at_100ms, depth_at_100.045ms, pose_at_100.010ms)
# Result: 3D point is ~120 pixels off - looking at wrong object!

# WITH time alignment:
pose_at_gaze_time = interpolate(poses, 100.000ms)
depth_at_gaze_time = interpolate(depths, 100.000ms)
gaze_3d = unproject(gaze, depth_at_gaze_time, pose_at_gaze_time)
# Result: Correct 3D point
```

**What fails without it:** Gaze projects to wrong location, system segments wrong object or rejects valid masks.

### 0b. 3D Support (Depth)
**Purpose:** Provides depth information necessary for 2D→3D→2D mapping.

**Concrete Example:**
```python
# Scenario: Looking at coffee cup on table
gaze_pixel = (640, 360)  # Center of 1280x720 image

# WITHOUT depth:
# Can't determine if looking at:
# - Cup at 0.5m?
# - Wall at 2m?
# - Window at 5m?
# All project to same pixel!

# WITH depth:
depth_at_gaze = 0.5m  # From depth sensor
gaze_3d = unproject((640, 360), 0.5m)  # Unique 3D point
```

**What fails without it:** Cannot establish correspondence between views - entire pipeline breaks.

### 0c. Loop Closure Updates
**Purpose:** Keeps cached poses consistent with SLAM optimizations.

**Concrete Example:**
```python
# Scenario: User walks around room and returns
Time T1: Cache mask, pose = [1, 0, 0, 0]  # Initial position
Time T2: SLAM detects loop closure
         All historical poses adjusted by ~10cm

# WITHOUT update:
cached_pose = [1, 0, 0, 0]  # Old, incorrect
current_pose = [1.1, 0, 0, 0]  # After optimization
T_rel = current @ inv(cached)  # Wrong by 10cm!
# Result: Mask projects to wrong location

# WITH update:
cached_pose = [1.1, 0, 0, 0]  # Updated after loop closure
T_rel = current @ inv(cached)  # Correct transform
```

**What fails without it:** Masks appear shifted from their true locations after loop closures.

### 0d. Camera-IMU Extrinsics Valid
**Purpose:** We must know the rotation from IMU→camera for the cached frame (e.g., via Kalibr). Extrinsics must be time-consistent with the cached pose (same calibration used during SLAM). If re-calibrated, refresh cached planes.

**Concrete Example:**
```python
# Scenario: Using IMU gravity to define horizontal planes
R_CI = camera_imu_extrinsics['rotation']  # From Kalibr calibration

# WITHOUT valid extrinsics:
# Can't transform gravity from IMU to camera frame → SEG

# WITH valid extrinsics:
gravity_imu = [0, 0, -9.81]  # Gravity in IMU frame
gravity_camera = R_CI @ gravity_imu  # Transform to camera frame
# Now can define horizontal planes (tables, floors)
```

**What fails without it:** If missing/stale → SEG.
**Reference:** [Kalibr CAM-IMU calibration](https://github.com/ethz-asl/kalibr/wiki/Camera-IMU-calibration)

### 0e. Projection Model Consistency
**Purpose:** Homographies assume a pinhole model. If cameras are fisheye/ultra-wide, undistort/rectify the ROI before estimating/using a homography.

**Concrete Example:**
```python
# Scenario: Quest 3 with fisheye cameras
raw_image = fisheye_camera_feed  # Distorted

# WITHOUT rectification:
H @ fisheye_point  # Invalid! Homography assumes pinhole
# Result: Warped masks appear distorted/wrong

# WITH rectification (proper approach):
# Build rectification maps once:
map1, map2 = cv2.fisheye.initUndistortRectifyMap(
    K_fisheye, D, None, K_new, (w, h), cv2.CV_16SC2)
# Apply to ROI:
rectified_roi = cv2.remap(raw_roi, map1, map2, cv2.INTER_LINEAR)
H @ rectified_point  # Valid homography transformation
```

**What fails without it:** If rectification is unavailable → SEG.

---

## Step 1: Get Inputs
**What it does:** Gathers all necessary data for the pipeline.

**Concrete Example:**
```python
# Camera intrinsics (Quest 3 example)
K = [[600,   0, 640],  # fx, 0, cx
     [  0, 600, 360],  # 0, fy, cy
     [  0,   0,   1]]  # 0, 0, 1

# Current state
current_pose = Transform(position=[2, 1, 1.5], rotation=quaternion)
gaze_pixel = (750, 400)  # Where user is looking
gaze_depth = 1.2m  # Depth at gaze point
```

**What fails without it:** No data to process - pipeline cannot start.

---

## Step 2: Compute Relative Pose
**What it does:** Calculates transformation between current view and cached view.

**Concrete Example:**
```python
# Scenario: User moved 50cm right and rotated 30° since caching mask
cached_pose = Transform(position=[1.5, 1, 1.5], rotation=[0, 0, 0])
current_pose = Transform(position=[2, 1, 1.5], rotation=[0, 30°, 0])

T_rel = current_pose @ inverse(cached_pose)
# T_rel represents: 50cm translation + 30° rotation

# This tells us how to transform points between views
```

**What fails without it:** Cannot map points between current and cached views.

---

## Step 3: Back-project Gaze to 3D
**What it does:** Converts 2D gaze point to 3D world point using depth.

**Concrete Example:**
```python
# Scenario: Looking at a cup
gaze_pixel = (750, 400)
gaze_depth = 1.2m

# Inverse projection
gaze_normalized = K_inv @ [750, 400, 1]  # [0.18, 0.067, 1]
gaze_3d_camera = gaze_normalized * 1.2m  # [0.216, 0.08, 1.2]

# In current camera coordinates: 21.6cm right, 8cm up, 1.2m forward
```

**What fails without it:** Cannot establish 3D correspondence - stuck in 2D with no way to relate views.

---

## Step 4: Map to Cached View
**What it does:** Transforms 3D point from current view to cached view.

**Concrete Example:**
```python
# Continuing from Step 3
gaze_3d_current = [0.216, 0.08, 1.2]  # Cup in current view

# Transform to cached view
gaze_3d_cached = T_rel_inv @ gaze_3d_current
# After 50cm right movement + 30° rotation:
# gaze_3d_cached = [-0.284, 0.08, 1.1]  # Different position

# Project to cached image (homogeneous coordinates)
p_tilde = K @ gaze_3d_cached  # Homogeneous result [x', y', z']
# Normalize by z to get pixel coordinates:
gaze_2d_cached = (p_tilde[0] / p_tilde[2], p_tilde[1] / p_tilde[2])
# gaze_2d_cached = (470, 380)  # Different pixel location!

# The cup that appears at (750, 400) now was at (470, 380) in cached view
```

**What fails without it:** Cannot find where current gaze maps to in cached mask.

---

## Step 5: Bounds Gate
**What it does:** Checks if projected gaze falls within cached image boundaries.

**Concrete Example:**
```python
# Scenario: User turned 90° since caching
gaze_2d_cached = project_to_cached(gaze_3d)
# Result: (-200, 400)  # Negative x - outside image!

# Image bounds: [0, 1280] x [0, 720]

if gaze_2d_cached.x < 0 or gaze_2d_cached.x > 1280:
    return SEG  # Can't reuse - looking at different scene

# Real case: Looking at door, cached mask is from window view
# After 90° turn, door projects outside window view bounds
```

**What fails without it:** Would try to access pixels outside image bounds - crash or invalid data.

---

## Step 6: Gaze-in-Mask Gate (Identity Check)
**What it does:** Verifies gaze hits the same object by checking if it falls within cached mask.

**Concrete Example:**
```python
# Scenario: Two cups on table
gaze_2d_cached = (470, 380)

# Cached masks:
cup_A_mask = binary_image with 1s at pixels [450:500, 360:400]
cup_B_mask = binary_image with 1s at pixels [550:600, 360:400]

# Check with dilation (for tracker noise tolerance)
dilated_cup_A = dilate(cup_A_mask, 5 pixels)

if dilated_cup_A[470, 380] == 1:
    # Gaze hits cup A - same object!
    candidate = cup_A
elif dilated_cup_B[470, 380] == 1:
    # Would hit cup B instead
    candidate = cup_B
else:
    # Hits neither - looking at table
    return SEG
```

**What fails without it:** Would reuse wrong object's mask (cup B mask for cup A).

---

## Step 7: Plane Stamp at Cache Time (Reused Later)
**What it does:** When a mask is first cached, store a local plane for that surface in the cached camera frame.

**Concrete Example:**
```python
# Get unit gravity in IMU frame; rotate into cached camera via calibrated extrinsics:
gravity_imu_unit = [0, 0, -1]  # Unit gravity in IMU
gravity_camera = R_CI @ gravity_imu_unit  # Transform via Kalibr extrinsics

# For horizontal planes (tables/floors), set plane normal:
n = -gravity_camera  # Points up (flip if needed)

# Pick one pixel on the object with metric depth Z and unproject:
X_0 = Z * inv(K_cache) @ [u, v, 1]  # 3D point on plane

# Plane offset (with sign chosen so n·X + d = 0):
d = -dot(n, X_0)
if d < 0:
    n = -n
    d = -d  # Ensure d > 0 for consistency

# Cache (n, d, K_cache) with the mask
cached_plane = {'normal': n, 'distance': d}  # Only 4 floats!

# Note: When using gravity, n estimates horizontal planes.
# For vertical planes, use image/feature-based Method 8B or
# sample a ring of depth points to fit n.
```

**Reference:** Plane-induced homography H = K'(R - tn^T/d)K^-1 (Hartley & Zisserman)

**What fails without it:** No metric plane; must re-estimate H every frame (slower, less stable).

---

## Step 8: Homography for This Frame
**What it does:** Computes the homography to map cached mask to current view.

### Method A: From Plane + Pose (Fast)
```python
# Assumptions (critical for correctness):
# - R, t are cache→current (pose of current camera expressed in cached camera frame)
# - n is unit normal in the cached camera frame
# - d > 0 is the plane offset in cached camera frame, with sign chosen so n·X + d = 0
# - Pure rotation fallback: if ‖t‖ ≈ 0, use H = K'RK⁻¹

# Plane-induced homography formula:
H = K_cur @ (R - np.outer(t, n) / d) @ np.linalg.inv(K_cache)
# This exactly matches textbook formula H = K'(R - tnᵀ/d)K⁻¹

# Code example:
H = K_cur @ (R - np.outer(t, n)/d) @ np.linalg.inv(K_cache)
mask_cur = cv2.warpPerspective(mask_cache, H, (W, H), flags=cv2.INTER_NEAREST)

# Pure rotation optimization: if ||t|| negligible or depth >> motion:
if np.linalg.norm(t) < 0.01:  # Nearly pure rotation
    H = K_cur @ R @ np.linalg.inv(K_cache)  # No plane needed!
```
**Reference:** H = K'(R - tnᵀ/d)K⁻¹ (Hartley & Zisserman §13, Szeliski §9.1)
*(Ensure ROI is pinhole-rectified if the original stream is fisheye)*

### Method B: From Image Data Inside the (Rectified) Mask ROI
```python
# Detect+match features restricted to mask ROI:
kp1, desc1 = detector.detectAndCompute(cached_roi_gray, mask=cached_mask)
kp2, desc2 = detector.detectAndCompute(current_roi_gray, None)

# Match descriptors and filter matches (e.g., Lowe ratio test)
matches = matcher.knnMatch(desc1, desc2, k=2)
good_matches = [m[0] for m in matches if m[0].distance < 0.7 * m[1].distance]

# Extract matched point arrays (Nx2 src/dst) from filtered matches:
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# findHomography consumes these matched point arrays:
H, inliers = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=3.0)
# RANSAC threshold tuned to scale (2-5px at 1280p)
# Note: OpenCV internally normalizes points before estimation

# Or use ECC (illumination-invariant, robust to linear gain/offset):
# Requirements: single-channel, same-size, float32, mean-normalized
template_gray = cv2.cvtColor(cached_roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
input_gray = cv2.cvtColor(current_roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
# Optional: mean-normalize for better convergence
template_gray = (template_gray - np.mean(template_gray)) / np.std(template_gray)
input_gray = (input_gray - np.mean(input_gray)) / np.std(input_gray)

warp = np.eye(3, dtype=np.float32)  # Initial 3x3 identity for MOTION_HOMOGRAPHY
cc, warp = cv2.findTransformECC(
    template_gray, input_gray, warp,
    cv2.MOTION_HOMOGRAPHY,
    (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4))
# Note: ECC is robust to linear brightness/gain but not strong non-linear changes

if cc < 0.90:  # Poor correlation
    return SEG
```

**What fails without it:** No geometric propagation; forced SEG.

---

## Step 9: Warp the Cached Mask
**What it does:** Applies the homography to transform cached mask to current view.

**Concrete Example:**
```python
# Warp mask using computed homography
# IMPORTANT: Use INTER_NEAREST for binary/label masks to avoid label bleeding
warped_mask = cv2.warpPerspective(mask_cache, H, (W, H), flags=cv2.INTER_NEAREST)
# Use INTER_LINEAR only for RGB images

# For fisheye/ultrawide: compute maps and remap, then run homography in pinhole ROI
if is_fisheye:
    # Build rectification maps once per camera and reuse every frame:
    if not hasattr(self, 'fisheye_maps'):
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
            K_fisheye, D, None, K_new, (w, h), cv2.CV_16SC2)
    # Rectify mask ROI:
    rectified_mask = cv2.remap(cached_mask, self.map1, self.map2, cv2.INTER_NEAREST)
    # Apply homography in rectified space:
    warped = cv2.warpPerspective(rectified_mask, H, size, flags=cv2.INTER_NEAREST)
    # Optional: map back to fisheye if needed for display

# Post-warp cleanup (implementation tip):
# Apply morphological operations to clean boundaries
kernel = np.ones((3,3), np.uint8)
warped_mask = cv2.morphologyEx(warped_mask, cv2.MORPH_CLOSE, kernel)  # Fill holes
warped_mask = cv2.morphologyEx(warped_mask, cv2.MORPH_OPEN, kernel)   # Remove noise
```

**What fails without it:** Wrong geometry on fisheye; distorted masks, misalignment.

---

## Step 10: Single Acceptance Gate (Geom. Fit if 8B + Exposure-Compensated Photometric)
**What it does:** Validates the warped mask through geometric and photometric checks.

### Geometric Fit Gate (if Method 8B)
```python
# Require sufficient RANSAC inlier ratio / low reprojection RMSE:
if method_B_used:
    if inlier_ratio < 0.6 or reproj_rmse > 5.0:
        return SEG  # Poor geometric fit
    
    # Or for ECC: check correlation coefficient
    if cc < 0.90:
        return SEG  # Failed to converge
```

### Photometric-After-Warp Gate
```python
# Compare warped cache ROI vs. current ROI
warped_roi = current_frame[warped_mask > 0]
cached_roi = cached_frame[cached_mask > 0]

# Use SSIM (structural similarity)
ssim_score = compute_ssim(warped_roi, cached_roi)

# Or use ECC for correlation (already computed in 8B if used)
# ECC is robust to linear gain/offset changes

# Thresholds (adjust based on conditions):
# - Start with SSIM ≥ 0.85, ECC cc ≥ 0.90
# - Raise thresholds on low-parallax frames
# - Lower slightly on high motion blur
# - Caveat: homography/ECC can fail on textureless ROIs
#   Fall back to SEG or try feature+mask boundary only

if ssim_score < 0.85 or (method_B_ecc and cc < 0.90):
    return SEG  # Appearance mismatch
```

**Why this works:** Homographies are exact for planar regions (or pure rotation); the gates catch parallax/occlusion/illumination failures without dense depth.

**What fails without it:** False reuse under parallax/occlusion/lighting; artifacts, wrong instance reuse.

---

## Design Philosophy: Simplicity Over Complexity

This pipeline prioritizes **simplicity and maintainability** over marginal accuracy gains:

- **Single plane assumption**: Covers 80%+ of AR/VR objects (tables, walls, screens)
- **Static object assumption**: Most indoor objects don't move
- **Clean fallback**: When assumptions fail → SEG (don't over-engineer edge cases)

### What This Pipeline Does NOT Include (By Design)

1. **Multi-plane/piecewise planar**: Adds complexity for <10% of cases
2. **Moving object tracking**: Use dedicated video tracking systems if needed
3. **Complex deformation models**: Non-planar objects trigger SEG
4. **Temporal consistency**: Keeps pipeline stateless and simple

### When to Consider Extensions

Only add complexity if metrics show specific problems:
- If >30% of objects are non-planar → Consider multi-plane
- If >30% of gazes hit moving objects → Consider dedicated tracker
- Otherwise → Keep it simple!

---

## Complete Example: Coffee Cup Scenario

Let's trace through the entire pipeline with a coffee cup:

```python
# Initial state (T1): Segmented coffee cup
cache_entry = {
    'mask': cup_shape,
    'pose': Transform([1, 0, 1.5], [0, 0, 0]),
    'depth': 1.0m everywhere in mask,
    'rgb': brown_cup_pixels,
    'anchors': [center, handle_tip, bottom]
}

# Current state (T2): User moved, lighting changed
current = {
    'pose': Transform([1.5, 0, 1.5], [0, 30°, 0]),
    'gaze': (800, 400),
    'depth_at_gaze': 1.0m,
    'rgb': brighter_scene
}

# Pipeline execution:
Step 0: ✓ Time aligned, ✓ Has depth, ✓ No loop closure, ✓ Valid extrinsics, ✓ Pinhole model
Step 1: Gathered inputs
Step 2: T_rel = 50cm right + 30° rotation
Step 3: gaze_3d = [0.3, 0.067, 1.0]
Step 4: gaze_cached = transform → (485, 375)
Step 5: ✓ (485, 375) within [0,1280]x[0,720]
Step 6: ✓ (485, 375) inside cup_mask
Step 7: Example plane fitted: n=[0,0,-1], d=1.0m (horizontal surface)
Step 8: H computed from plane + pose
Step 9: Mask warped via homography (INTER_NEAREST)
Step 10a: ✓ Geometric validation passed
Step 10b: ✓ SSIM=0.91, ECC=0.93

Result: REUSE mask (all checks passed)
```

---

## Summary: Why Each Step is Essential

| Step | Without It | Failure Mode |
|------|------------|--------------|
| 0a | Wrong timestamps | 100+ pixel errors from latency |
| 0b | No depth | Cannot map between views |
| 0c | Stale poses | Masks shift after loop closure |
| 0d | No IMU→camera calibration | Cannot use gravity for plane estimation; forced SEG |
| 0e | Fisheye distortion | Homography produces wrong warps on fisheye |
| 1-2 | No data/transform | Cannot process |
| 3-4 | No 3D mapping | Cannot relate 2D views |
| 5 | No bounds check | Crash on invalid access |
| 6 | No identity check | Reuse wrong object's mask |
| 7 | Plane stamp (IMU + one depth) | No metric plane; must re-estimate H every frame |
| 8 | Homography (plane or ROI) | No geometric propagation; can't reuse; forced SEG |
| 9 | Warp (pinhole/rectified) | Wrong geometry on fisheye; distorted masks |
| 10 | Acceptance gate (fit + photometric) | False reuse under parallax/occlusion/lighting |

The planar pipeline is memory-efficient (4 floats vs 1M depths) while handling most AR/VR scenarios. For non-planar or moving objects, the pipeline cleanly falls back to segmentation rather than adding complexity.

## Default Thresholds Reference

| Parameter | Default Value | Adjust When | Direction |
|-----------|--------------|-------------|-----------|
| SSIM | ≥ 0.85 | Low parallax frames | Raise to 0.90+ |
| ECC correlation | ≥ 0.90 | High motion blur | Lower to 0.85 |
| Reproj RMSE | ≤ 5 px | Different resolution | Scale with image size |
| RANSAC inlier ratio | ≥ 0.6 | Textureless regions | Lower to 0.4 |
| Motion residual | ≥ 5 px → SEG | Stable camera | Raise to 10 px |
| Occlusion ratio | ≤ 0.15 | Known occlusions | Raise to 0.25 |

**Note:** For textureless ROIs where homography/ECC fails, fall back to SEG or try feature+mask boundary only.

**References:**
- Plane-induced homography: Hartley & Zisserman, "Multiple View Geometry" §13; Szeliski "Computer Vision" §9.1
- OpenCV homography: `findHomography(..., RANSAC)` with [tutorial](https://docs.opencv.org/master/d9/dab/tutorial_homography.html)
- OpenCV ECC: `findTransformECC(..., MOTION_HOMOGRAPHY)` with [tutorial](https://docs.opencv.org/master/dc/d6b/group__video__track.html#ga1aa357007eaec11e9ed03500ecbcebe0)
- Camera-IMU calibration: [Kalibr CAM-IMU](https://github.com/ethz-asl/kalibr/wiki/Camera-IMU-calibration)
- Piecewise planar: Sinha et al., "Piecewise Planar Stereo for Image-based Rendering"
- Fisheye rectification: `cv2.fisheye.initUndistortRectifyMap()` + `cv2.remap()` [tutorial](https://docs.opencv.org/master/db/d58/group__calib3d__fisheye.html)
- Basic homogeneous coordinates: [OpenCV camera calibration tutorial](https://docs.opencv.org/master/d9/d0c/group__calib3d.html)