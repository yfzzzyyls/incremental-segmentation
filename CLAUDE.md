# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Incremental segmentation research project implementing mask reuse strategies for AR/VR applications with gaze-based interaction. The core innovation is efficient object segmentation by reusing previous masks when viewing conditions allow, reducing computational overhead for real-time AR/VR systems.

## Current State

Active development of the CTRL (Control) module that implements the mask reuse decision pipeline. The repository contains:
- **Implemented**: Core control logic for SEG vs REUSE decisions (`ctrl/ctrl.py`)
- **Testing**: Unit tests with ADT dataset integration (`ctrl/test_ctrl.py`)
- **Documentation**: Detailed pipeline architecture (`DESIGN.md`)

## Key Commands

### Running Tests
```bash
# Run mask reuse controller test with visualization
python ctrl/test_ctrl.py --max-frames 500 --visualize

# Full test with specific sequence
python ctrl/test_ctrl.py --data-root /mnt/ssd_ext/incSeg-data --sequence Apartment_release_clean_seq148_M1292 --max-frames 1000

# Test with custom cache size
python ctrl/test_ctrl.py --cache-size 20 --max-frames 500
```

### Data Processing Scripts
```bash
# Extract calibration from ADT VRS files
python ctrl/extract_calibration.py

# Extract segmentation masks from dataset
python ctrl/extract_segmentation.py

# Analyze IoU performance
python ctrl/analyze_iou.py
```

## Architecture

### Control Module (`ctrl/`)
Implements the mask reuse decision pipeline with fisheye camera support.

**Key Components:**
- `MaskReuseController`: Main control logic implementing SEG vs REUSE decisions
- `CachedFrame`: Data structure for storing frames with masks, depth, and plane parameters
- `ControlInput/Output`: Input/output interfaces for the pipeline

**Critical Implementation Details:**
1. **Camera Model**: Uses Kannala-Brandt fisheye model (FISHEYE624) for ADT data
   - Distortion coefficients: k1=0.406, k2=-0.490, k3=0.175, k4=1.133
   - Uses `cv2.fisheye.undistortPoints()` and `cv2.fisheye.projectPoints()`

2. **Control Flow**:
   - Project gaze to ALL cached views first
   - Find masks containing the gaze point
   - Select best candidate by pose similarity
   - Validate and warp selected mask

3. **Pose Convention**: ADT provides camera-to-world transforms (T_world_camera)
   - To go world→camera: `inv(pose)`
   - To go camera→world: `pose`

4. **Performance Target**: <10ms for mask reuse decision

### Data Pipeline
Works with ADT (Aria Digital Twin) dataset format:
- RGB frames at 1408×1408 resolution
- Depth maps from simulation
- 6DOF poses with timestamps
- Gaze tracking data
- Ground truth segmentation masks

## Key Documentation

### Pipeline Architecture (`DESIGN.md`)
Comprehensive step-by-step analysis of the mask reuse pipeline including:
- Preconditions and time alignment
- 3D gaze projection with fisheye model
- Planar homography computation
- Validation criteria and fallback logic
- Performance optimizations

### Design Principles
- Single plane assumption for computational efficiency
- 80/20 optimization (handle common cases, clean fallback for edges)
- Memory efficient (4 plane parameters vs dense depth)
- Stateless pipeline with fast failure paths

## Testing Guidelines

The test suite (`ctrl/test_ctrl.py`) evaluates:
- Mask reuse rate (target: >40% for static scenes)
- IoU accuracy when masks are reused
- Processing time per frame
- Cache hit patterns and efficiency

Expected test output includes:
- Per-frame decisions (SEG/REUSE) with reasons
- Aggregate statistics (reuse rate, avg IoU, timing)
- Optional visualization of warped masks

## Current Experiments (Dec 2024)

### Depth Prediction Impact Study

**Objective**: Determine if sparse depth prediction (at gaze point only) is sufficient for accurate mask projection, or if dense depth prediction across the mask is needed.

**Rationale**: The current pipeline uses planar approximation which requires fitting a plane to many depth points within the mask. If we only have depth at gaze location from a predictor, this approach fails. Need to quantify the accuracy trade-off.

**Experiment Design**:

1. **Experiment 1 - Dense GT Baseline**:
   - Replace planar homography with pixel-wise 3D reprojection
   - Use ground truth depth for EVERY pixel in mask
   - Measure IoU as upper bound (best possible accuracy without planar assumption)

2. **Experiment 2 - Sparse GT**:
   - Use only GT depth from gaze location (or 22x22 patch average)
   - Assume uniform depth for entire mask
   - Measure IoU degradation from sparse sampling

3. **Experiment 3 - Sparse Predicted**:
   - Same as Exp 2 but use predicted depth from checkpoint: `~/ORB_SLAM3_VIO/depthCNN/checkpoints/spatial_dual_coordconv_max/`
   - Measure additional IoU degradation from prediction errors

**Expected Insights**:
- Whether planar assumption is limiting factor (Exp 1 vs current)
- Cost of sparse depth assumption (Exp 1 vs 2)
- Cost of depth prediction errors (Exp 2 vs 3)
- Whether current depth predictor is sufficient or needs dense output

### Dense 3D Reprojection Implementation

**Methodology**:
We replaced the planar homography approach with dense 3D reprojection to test depth accuracy impact:

1. **Dense Reprojection Process**:
   - For each pixel in cached mask: unproject to 3D using depth and camera intrinsics
   - Transform 3D points from cached to current camera frame
   - Reproject 3D points to current 2D image using fisheye model
   - Vectorized implementation for efficiency (processes all pixels at once)

2. **Key Discoveries**:
   - **Occlusion Problem**: Dense reprojection revealed that masks from different viewpoints often project onto different objects due to occlusion
   - **Instance ID Validation**: Must check if reprojected mask lands on same object (by instance ID)
   - **Photometric Validation**: Added color histogram comparison to detect when mask projects onto different surface

3. **Validation Enhancements**:
   - Strict gaze-in-mask check (radius=0) to ensure exact object match
   - Color histogram comparison using chi-squared distance
   - Instance segmentation ID verification (in dataset context)

### Frame Pair Selection for Depth Experiments

**Challenge**: Need frame pairs viewing the same object without occlusion complications.

**Selection Criteria**:
- Same object instance (verified by segmentation ID)
- Large object size (>100k pixels) for reliable statistics
- Moderate viewpoint change (10-50cm translation, 5-30° rotation)
- Exclude background/wall objects (ID: 4377907428960199)

**Objects Found in ADT Sequence** (Apartment_release_clean_seq148_M1292):
```
Object 4666210076742595: Table/surface, appears in 18-32 frames
  - Frames: [3, 67, 68, 69, 70, 71, 72, 73, 74, 75, ...]
  - Average mask size: ~125k pixels
  - Good candidate for testing

Object 4671332369591132: Large surface, single frame
  - Frame: [0]
  - Mask size: ~205k pixels
  - Cannot form pairs (only one frame)
```

**Selected Frame Pairs**:
- **Object 4666210076742595**: Table/surface with 95 valid frame pairs found
- **Best pair**: Frame 179 → Frame 198
  - Translation: 19.9cm
  - Rotation: 38.6°
  - Mask sizes: 208k → 157k pixels
  - Good viewpoint diversity for testing depth impact
- **Alternative pairs** (for different baselines):
  - Frame 195 → 198: Small baseline (5.5cm, 2.7°)
  - Frame 181 → 185: Medium baseline (5.0cm, 9.1°)
  - Frame 175 → 190: Large baseline (17.0cm, 38.2°)

### Implementation Files

1. **`ctrl/ctrl.py`**: Modified controller with dense reprojection option
   - Added `_dense_reproject_mask()` method
   - Enhanced validation with histogram comparison
   - Strict object matching (radius=0)

2. **`depth_pred/depth_pred_projection_comparison.py`**: Frame pair finder
   - Searches for same-object frame pairs
   - Filters by object size and viewpoint change
   - Excludes background objects

### Next Steps

1. Implement the three depth experiments on selected frame pair
2. Compare IoU metrics across dense GT, sparse GT, and predicted depth
3. Determine if sparse depth at gaze is sufficient for mask reuse

## Critical Findings: Fisheye Projection and IoU Limitations (Dec 2024)

### Key Discovery: IoU Ceiling with Ground Truth

When using **ALL ground truth values** (depth, pose, segmentation), we achieve only ~70% IoU for medium viewpoint changes (5cm, 10°). This was initially surprising, but investigation revealed fundamental geometric limitations, not implementation errors.

### Investigation Timeline

1. **Initial Hypothesis**: Fisheye distortion at image edges causing projection errors
   - Radial analysis showed: Center IoU=0.91, Middle IoU=0.90, **Edge IoU=0.26** ❌
   - Edge pixels (>60% radius from center) had severe projection errors

2. **First Attempted Fix**: Replace `cv2.fisheye.undistortPoints()` with native KB solver
   - Implemented Newton-Raphson solver for KB inverse: `r(θ) = θ + k₁θ³ + k₂θ⁵ + k₃θ⁷ + k₄θ⁹`
   - Result: **No improvement** - CV2 already does correct KB inverse
   - Verified: Ray differences <1e-8 between CV2 and native implementation

3. **Root Cause Analysis**: The 70% IoU ceiling is due to:
   - **Visibility/FOV changes**: Edge pixels project outside image bounds or onto different objects
   - **Occlusion**: Even small viewpoint changes cause occlusion at object boundaries
   - **Rasterization artifacts**: Point splatting creates gaps in projected masks
   - **NOT fisheye math**: Both CV2 and native KB implementations are mathematically correct

### Experimental Results

IoU degradation with viewpoint change (all with dense GT):
```
Baseline      Translation  Rotation   IoU
Very small    1.3cm        3.0°       0.896 ✅
Small         2.6cm        5.7°       0.809 ✅
Medium        5.2cm        10.2°      0.701
Large         11.7cm       26.3°      0.564
```

### Technical Details

1. **Fisheye Implementation**:
   - `cv2.fisheye.undistortPoints()` returns rays as `[tan(θ)cos(ψ), tan(θ)sin(ψ)]`
   - This is the correct angular representation for KB model
   - Native implementation matches CV2 to ~1e-8 precision

2. **Depth Interpretation**:
   - ADT dataset stores Z-depth (along optical axis), not radial distance
   - Correct scaling: `point_3d = ray * depth_z`
   - Edge rays at ~51° from optical axis have significant angular component

3. **Edge Pixel Behavior**:
   - Object extends to image edges (max radius: 716 pixels)
   - Edge pixels correctly unproject/project but land on different objects
   - Example: Pixel (1189, 271) → projects to (1398, 89) → different object ID

### Implications for Mask Reuse

1. **70% IoU is actually good** for medium viewpoint changes given geometric constraints
2. **Planar approximation** in production pipeline is reasonable trade-off
3. **Dense depth** provides marginal improvement over sparse (70% vs 65% IoU)
4. **Focus optimization** on small-baseline cases where IoU >80%

### Recommended Improvements

1. **Visibility-aware evaluation**: Add target depth gating to filter occluded projections
2. **Contour-based warping**: Replace point splatting with contour warp + fill
3. **Adaptive thresholds**: Use baseline-dependent IoU thresholds for REUSE decision
4. **Z-buffer**: Already implemented to handle occlusion correctly

### Understanding the Three Error Sources

#### 1. **Occlusion**
- **Definition**: Parts of object visible in frame 179 are hidden behind other objects (or itself) in frame 199
- **Example**: Table pixels behind chair after viewpoint change
- **Impact**: Pixels project to correct location but are physically blocked
- **Evidence**: Edge pixels project correctly but land on different object IDs

#### 2. **FOV (Field of View) Changes**
- **Definition**: Pixels project outside the 1408×1408 image bounds
- **Example**: Pixel (1196, 271) → projects to (1409, 89) - outside frame
- **Impact**: Double penalty - lose pixels AND can't capture new visible areas
- **Key insight**: Same physical table, but different portions visible = lower IoU

#### 3. **Rasterization Artifacts**
- **Definition**: Converting continuous geometry to discrete pixel grid causes gaps
- **Current approach**: Point splatting - each pixel projects individually
- **Problems**:
  - Rounding errors: (1300.7, 650.3) → (1301, 650)
  - Gaps when surface stretches under viewpoint change
  - No area preservation for edge pixels
- **Solution**: Contour-based warping instead of point splatting

### Camera Model Understanding

#### Kannala-Brandt (KB) Fisheye Model
- **Purpose**: Mathematically describes fisheye lens geometry
- **Forward**: `r = θ + k₁θ³ + k₂θ⁵ + k₃θ⁷ + k₄θ⁹` (angle → image radius)
- **Inverse**: Newton-Raphson solver (image radius → angle)
- **Key point**: KB model describes distortion, doesn't remove it

#### Universal 3D Projection Pipeline
```
2D Pixel → [Camera Model] → 3D Ray → Transform → 3D Point → [Camera Model] → 2D Pixel
```
- Pipeline is universal for all cameras
- Only the camera model changes (pinhole vs fisheye)
- KB model is fisheye-specific, replacing simpler pinhole model

### Conclusion

The investigation revealed that our implementation is fundamentally correct. The IoU limitations are due to unavoidable geometric factors (occlusion, FOV changes, rasterization) rather than fisheye distortion or implementation errors. The 70% IoU ceiling with all GT values represents a realistic upper bound for this viewpoint change magnitude.