#!/usr/bin/env python3
"""
AriaEveryday to VIFT Data Processing Pipeline - Quaternion Version
Maintains quaternions throughout the pipeline without Euler conversion.
"""

import os
import csv
import json
import zipfile
import shutil
import numpy as np
import torch
import cv2
from pathlib import Path
import argparse
from tqdm import tqdm
import ffmpeg
from typing import List, Dict, Tuple, Optional


class AriaToVIFTProcessor:
    """Process AriaEveryday dataset maintaining quaternions"""
    
    def __init__(self, aria_data_dir: str, output_dir: str, max_frames: int = 500, device: str = "auto"):
        self.aria_data_dir = Path(aria_data_dir)
        self.output_dir = Path(output_dir)
        self.max_frames = max_frames
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up device for GPU/MPS acceleration
        if device == "auto":
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
                print(f"🚀 Using MPS (Apple Silicon GPU) acceleration")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
                print(f"🚀 Using CUDA GPU acceleration")
            else:
                self.device = torch.device("cpu")
                print(f"⚡ Using CPU processing")
        else:
            self.device = torch.device(device)
            print(f"🎯 Using specified device: {self.device}")
    
    def extract_slam_trajectory(self, sequence_path: Path) -> Optional[List[Dict]]:
        """Extract SLAM trajectory from MPS results, keeping quaternions"""
        print(f"📍 Extracting SLAM trajectory from {sequence_path.name}")

        # Prefer SLAM summary JSON for trajectories
        summary_zips = list(sequence_path.glob("*mps_slam_summary.zip"))
        if summary_zips:
            summary_zip = summary_zips[0]
            temp_sum = sequence_path / "temp_slam_summary"
            temp_sum.mkdir(exist_ok=True)
            try:
                # Use zipfile to extract summary.json robustly
                with zipfile.ZipFile(summary_zip, 'r') as z:
                    # find the summary.json entry
                    name = next((n for n in z.namelist() if n.endswith('summary.json')), None)
                    if name:
                        print(f"🔄 Extracting {name} from summary archive")
                        z.extract(name, temp_sum)
                        summary_file = temp_sum / name
                        # adjust if nested in subfolder
                        summary_file = summary_file if summary_file.exists() else temp_sum / Path(name).name
                    else:
                        print(f"❌ No summary.json found in archive")
                        shutil.rmtree(temp_sum, ignore_errors=True)
                        return None
                        
                if summary_file.exists():
                    with open(summary_file, 'r') as f:
                        summary_data = json.load(f)
                    
                    poses = []
                    # Extract poses from summary data
                    if 'trajectory' in summary_data:
                        for frame in summary_data['trajectory']:
                            timestamp = frame['timestamp']
                            pose = frame['pose']
                            # Expecting [tx, ty, tz, qx, qy, qz, qw]
                            poses.append({
                                'timestamp': timestamp,
                                'translation': pose[:3],
                                'quaternion': pose[3:]  # Keep as [qx, qy, qz, qw]
                            })
                    
                    shutil.rmtree(temp_sum, ignore_errors=True)
                    if poses:
                        print(f"✅ Extracted {len(poses)} poses from summary.json")
                        return poses
                        
            except Exception as e:
                print(f"❌ Error processing summary zip: {e}")
                shutil.rmtree(temp_sum, ignore_errors=True)
        
        # Find MPS output archives
        archives = list(sequence_path.glob("*mps_[so]*.zip"))
        if not archives:
            print(f"❌ No MPS output found")
            return None
            
        # Sort by priority (closed_loop better than open_loop)
        archives.sort(key=lambda x: (0 if 'closed_loop' in x.name else 1))
        
        for archive in archives:
            temp_dir = sequence_path / f"temp_extract_{archive.stem}"
            temp_dir.mkdir(exist_ok=True)
            
            # Extract archive
            shutil.unpack_archive(archive, temp_dir)
            
            # Look for trajectory CSV files
            csv_files = list(temp_dir.rglob("*trajectory*.csv"))
            # Sort by priority
            priority_order = [
                'closed_loop_trajectory.csv',
                'open_loop_trajectory.csv',
                'slam_trajectory_derivate.csv',
                'trajectory.csv'
            ]
            
            def get_priority(csv_file):
                for i, name in enumerate(priority_order):
                    if name in csv_file.name:
                        return i
                return len(priority_order)
            
            csv_files.sort(key=get_priority)
            
            if csv_files:
                print(f"📄 Found {len(csv_files)} trajectory CSV files")
                # Use the first (highest priority) CSV file
                csv_file = csv_files[0]
                print(f"📖 Using: {csv_file.name}")
                
                poses = self._parse_trajectory_csv_quaternion(csv_file)
                shutil.rmtree(temp_dir, ignore_errors=True)
                
                if poses:
                    return poses
            else:
                print(f"❌ No trajectory CSV files found in {archive.name}")
                
            shutil.rmtree(temp_dir, ignore_errors=True)
                
        return None
    
    def _parse_trajectory_csv_quaternion(self, csv_file: Path) -> List[Dict]:
        """Parse SLAM trajectory CSV keeping quaternions"""
        poses = []
        
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            
            # Determine column indices based on CSV type
            if 'closed_loop' in csv_file.name:
                # closed_loop format: graph_uid,tracking_timestamp_us,utc_timestamp_ns,tx_world_device,ty_world_device,tz_world_device,qx_world_device,qy_world_device,qz_world_device,qw_world_device
                ts_col, tx_col, ty_col, tz_col = 1, 3, 4, 5
                qx_col, qy_col, qz_col, qw_col = 6, 7, 8, 9
            else:
                # open_loop format: tracking_timestamp_us,utc_timestamp_ns,session_uid,tx_odometry_device,ty_odometry_device,tz_odometry_device,qx_odometry_device,qy_odometry_device,qz_odometry_device,qw_odometry_device
                ts_col, tx_col, ty_col, tz_col = 0, 3, 4, 5
                qx_col, qy_col, qz_col, qw_col = 6, 7, 8, 9
            
            for row in reader:
                try:
                    if len(row) >= max(qw_col + 1, 10):
                        timestamp = float(row[ts_col]) / 1e6  # Convert to seconds
                        tx, ty, tz = float(row[tx_col]), float(row[ty_col]), float(row[tz_col])
                        qx, qy, qz, qw = float(row[qx_col]), float(row[qy_col]), float(row[qz_col]), float(row[qw_col])
                        
                        # Ensure quaternion is normalized
                        q_norm = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
                        if q_norm > 0:
                            qx, qy, qz, qw = qx/q_norm, qy/q_norm, qz/q_norm, qw/q_norm
                        
                        # Store pose with quaternion in XYZW format
                        poses.append({
                            'timestamp': timestamp,
                            'translation': [tx, ty, tz],
                            'quaternion': [qx, qy, qz, qw]  # XYZW format
                        })
                except (ValueError, IndexError):
                    continue
        
        print(f"✅ Extracted {len(poses)} poses from {csv_file.name}")
        return poses
    
    def extract_rgb_frames(self, sequence_path: Path, num_frames: int) -> Optional[torch.Tensor]:
        """Extract RGB frames from preview video with GPU acceleration"""
        print(f"📹 Extracting RGB frames from {sequence_path.name}")
        
        # Find preview video
        video_files = list(sequence_path.glob("*preview_rgb.mp4"))
        if not video_files:
            print(f"⚠️ No preview RGB video found")
            return None
            
        video_file = video_files[0]
        
        try:
            # Use OpenCV to read video
            cap = cv2.VideoCapture(str(video_file))
            if not cap.isOpened():
                print(f"❌ Cannot open video {video_file}")
                return None
            
            frames = []
            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Sample frames evenly if video is longer than needed
            frame_indices = np.linspace(0, total_frames-1, min(num_frames, total_frames), dtype=int)
            
            for target_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB and resize to standard size
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_resized = cv2.resize(frame_rgb, (640, 480))  # Standard size
                    
                    # Convert to tensor [C, H, W] and normalize to [0, 1]
                    frame_tensor = torch.from_numpy(frame_resized.transpose(2, 0, 1)).float() / 255.0
                    frames.append(frame_tensor)
                    frame_count += 1
                    
                if frame_count >= num_frames:
                    break
            
            cap.release()
            
            if frames:
                # Stack frames and move to device for GPU processing
                visual_data = torch.stack(frames).to(self.device)  # Shape: [T, C, H, W]
                print(f"✅ Extracted {len(frames)} RGB frames on {self.device}")
                return visual_data
            
        except Exception as e:
            print(f"❌ Error extracting frames: {e}")
            
        return None
    
    def generate_imu_data(self, poses: List[Dict], num_frames: int) -> torch.Tensor:
        """Generate realistic IMU data from SLAM trajectory using quaternions"""
        print(f"📊 Generating IMU data from trajectory")
        
        imu_frequency = 1000.0  # 1kHz IMU
        camera_frequency = 30.0  # 30Hz camera
        samples_per_frame = int(imu_frequency / camera_frequency)  # ~33 samples per frame
        
        imu_data = []
        
        # Ensure we have enough poses
        if len(poses) < num_frames:
            poses = poses * (num_frames // len(poses) + 1)
        
        for i in range(num_frames):
            current_pose = poses[i] if i < len(poses) else poses[-1]
            next_pose = poses[i + 1] if i + 1 < len(poses) else poses[-1]
            
            # Compute motion between poses
            dt = max(next_pose['timestamp'] - current_pose['timestamp'], 1.0 / camera_frequency)
            
            # Position difference
            dp = np.array(next_pose['translation']) - np.array(current_pose['translation'])
            velocity = dp / dt
            
            # Estimate acceleration (with gravity)
            acceleration = velocity / dt if dt > 0 else np.zeros(3)
            acceleration[2] += 9.81  # Add gravity in Z direction
            
            # Angular velocity from quaternions
            q1 = np.array(current_pose['quaternion'])  # [qx, qy, qz, qw]
            q2 = np.array(next_pose['quaternion'])
            
            # Compute relative rotation: q_rel = q1^(-1) * q2
            q1_inv = np.array([-q1[0], -q1[1], -q1[2], q1[3]]) / np.dot(q1, q1)
            
            # Quaternion multiplication
            q_rel = np.array([
                q1_inv[3]*q2[0] + q1_inv[0]*q2[3] + q1_inv[1]*q2[2] - q1_inv[2]*q2[1],
                q1_inv[3]*q2[1] - q1_inv[0]*q2[2] + q1_inv[1]*q2[3] + q1_inv[2]*q2[0],
                q1_inv[3]*q2[2] + q1_inv[0]*q2[1] - q1_inv[1]*q2[0] + q1_inv[2]*q2[3],
                q1_inv[3]*q2[3] - q1_inv[0]*q2[0] - q1_inv[1]*q2[1] - q1_inv[2]*q2[2]
            ])
            
            # Convert to angular velocity (simplified)
            angular_velocity = 2.0 * np.array([q_rel[0], q_rel[1], q_rel[2]]) / dt
            
            # Create IMU sequence for this frame
            frame_imu = []
            for j in range(samples_per_frame):
                # Add realistic noise
                accel_noise = np.random.normal(0, 0.1, 3)
                gyro_noise = np.random.normal(0, 0.05, 3)
                
                imu_sample = torch.tensor([
                    angular_velocity[0] + gyro_noise[0],
                    angular_velocity[1] + gyro_noise[1],
                    angular_velocity[2] + gyro_noise[2],
                    acceleration[0] + accel_noise[0],
                    acceleration[1] + accel_noise[1],
                    acceleration[2] + accel_noise[2]
                ], device=self.device)
                
                frame_imu.append(imu_sample)
            
            imu_data.append(torch.stack(frame_imu))  # Shape: [samples_per_frame, 6]
        
        print(f"✅ Generated IMU data for {num_frames} frames on {self.device}")
        return torch.stack(imu_data)  # Shape: [T, samples_per_frame, 6]
    
    def process_sequence(self, sequence_path: Path, sequence_id: str) -> bool:
        """Process a single AriaEveryday sequence with quaternions"""
        print(f"\n🔄 Processing sequence: {sequence_path.name}")
        
        # Extract SLAM trajectory
        poses = self.extract_slam_trajectory(sequence_path)
        if not poses:
            print(f"❌ No SLAM trajectory found for {sequence_path.name}")
            return False
        
        # Limit frames
        num_frames = min(len(poses), self.max_frames)
        poses = poses[:num_frames]
        
        # Extract RGB frames
        visual_data = self.extract_rgb_frames(sequence_path, num_frames)
        if visual_data is None:
            print(f"❌ No visual data extracted for {sequence_path.name}")
            return False
        
        # Ensure matching lengths
        actual_frames = min(len(poses), visual_data.shape[0])
        poses = poses[:actual_frames]
        visual_data = visual_data[:actual_frames]
        
        # Generate IMU data
        imu_data = self.generate_imu_data(poses, actual_frames)
        
        # Save processed sequence
        seq_output_dir = self.output_dir / sequence_id
        seq_output_dir.mkdir(exist_ok=True)
        
        # Save poses as JSON with quaternions
        poses_file = seq_output_dir / "poses_quaternion.json"
        with open(poses_file, 'w') as f:
            json.dump(poses, f, indent=2, default=str)
        
        # Save visual and IMU data as tensors (move to CPU for saving)
        torch.save(visual_data.cpu(), seq_output_dir / "visual_data.pt")
        torch.save(imu_data.cpu(), seq_output_dir / "imu_data.pt")
        
        # Save metadata
        metadata = {
            'sequence_name': sequence_path.name,
            'sequence_id': sequence_id,
            'num_frames': actual_frames,
            'visual_shape': list(visual_data.shape),
            'imu_shape': list(imu_data.shape),
            'slam_trajectory_type': 'mps_slam',
            'rotation_format': 'quaternion_xyzw'
        }
        
        with open(seq_output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Processed {sequence_path.name}: {actual_frames} frames")
        return True
    
    def process_dataset(self, start_index: int = 0, max_sequences: Optional[int] = None, folder_offset: int = 0) -> Dict:
        """Process multiple AriaEveryday sequences"""
        print(f"🎯 Processing AriaEveryday Dataset (Quaternion Version)")
        print(f"📁 Input: {self.aria_data_dir}")
        print(f"📁 Output: {self.output_dir}")
        
        # Get all sequence directories
        all_sequences = sorted([d for d in self.aria_data_dir.iterdir() if d.is_dir()])
        
        # Apply start index and max sequences
        if max_sequences is None:
            max_sequences = len(all_sequences) - start_index
            
        end_index = min(start_index + max_sequences, len(all_sequences))
        sequences = all_sequences[start_index:end_index]
        
        print(f"🔢 Processing sequences {start_index} to {end_index-1} (total: {len(sequences)})")
        print(f"📝 Output folders will be numbered from {folder_offset} to {folder_offset + len(sequences) - 1}")
        print("=" * 60)
        
        processed_count = 0
        processed_sequences = []
        
        for i, sequence_path in enumerate(tqdm(sequences, desc="Processing sequences")):
            # Apply folder offset to sequence ID
            sequence_id = f"{folder_offset + i:03d}"  # Format as 000, 001, 002, etc.
            
            if self.process_sequence(sequence_path, sequence_id):
                processed_count += 1
                processed_sequences.append({
                    'sequence_id': sequence_id,
                    'sequence_name': sequence_path.name,
                    'frames': actual_frames
                })
        
        # Save dataset summary
        summary = {
            'dataset_name': 'AriaEveryday_VIFT_Quaternion',
            'total_sequences': len(sequences),
            'processed_sequences': processed_count,
            'start_index': start_index,
            'max_sequences': max_sequences,
            'folder_offset': folder_offset,
            'rotation_format': 'quaternion_xyzw',
            'sequences': processed_sequences
        }
        
        with open(self.output_dir / "dataset_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n🎉 Processing Complete!")
        print(f"✅ Processed {processed_count}/{len(sequences)} sequences")
        
        return summary


def main():
    parser = argparse.ArgumentParser(description='Process AriaEveryday dataset to VIFT format (Quaternion version)')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Input directory containing AriaEveryday sequences')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for processed data')
    parser.add_argument('--max-frames', type=int, default=500,
                        help='Maximum frames to extract per sequence (default: 500)')
    parser.add_argument('--start-index', type=int, default=0,
                        help='Starting sequence index (default: 0)')
    parser.add_argument('--max-sequences', type=int, default=None,
                        help='Maximum number of sequences to process (default: all)')
    parser.add_argument('--folder-offset', type=int, default=0,
                        help='Offset for output folder numbering (default: 0)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use: auto, cuda, mps, or cpu (default: auto)')
    
    args = parser.parse_args()
    
    processor = AriaToVIFTProcessor(
        aria_data_dir=args.input_dir,
        output_dir=args.output_dir,
        max_frames=args.max_frames,
        device=args.device
    )
    
    processor.process_dataset(
        start_index=args.start_index,
        max_sequences=args.max_sequences,
        folder_offset=args.folder_offset
    )


if __name__ == "__main__":
    main()