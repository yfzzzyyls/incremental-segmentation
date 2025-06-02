from typing import Any, Dict, Optional, Tuple
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
import json
from pathlib import Path
import numpy as np

class AriaDataset(Dataset):
    """Dataset for processed AriaEveryday data"""
    
    def __init__(self, data_dir: str, sequence_ids: list, seq_len: int = 11, overlap: int = 5):
        self.data_dir = Path(data_dir)
        self.sequence_ids = sequence_ids
        self.seq_len = seq_len
        self.overlap = overlap
        
        # Load all sequences
        self.sequences = []
        for seq_id in sequence_ids:
            seq_dir = self.data_dir / f"{seq_id:02d}"
            if seq_dir.exists():
                self.sequences.append(self._load_sequence(seq_dir))
        
        # Create sliding windows
        self.samples = self._create_sliding_windows()
    
    def _load_sequence(self, seq_dir: Path) -> Dict:
        """Load a single sequence"""
        # Load poses
        with open(seq_dir / "poses.json", 'r') as f:
            poses = json.load(f)
        
        # Load IMU data
        imu_data = torch.load(seq_dir / "imu_data.pt")
        
        # Try to load cached visual features first
        cached_features_path = seq_dir / "visual_features.pt"
        if cached_features_path.exists():
            print(f"Loading cached visual features from {cached_features_path}")
            visual_features = torch.load(cached_features_path)
        else:
            print(f"⚠️ No cached features found at {cached_features_path}")
            print("Please run: python data/latent_caching_aria.py first")
            # Fallback: load raw visual data
            visual_features = torch.load(seq_dir / "visual_data.pt")
        
        return {
            'poses': poses,
            'visual_features': visual_features,  # Use cached features
            'imu': imu_data
        }
    
    def _create_sliding_windows(self) -> list:
        """Create sliding windows from sequences"""
        samples = []
        
        for seq_idx, sequence in enumerate(self.sequences):
            num_frames = len(sequence['poses'])
            
            # Create sliding windows
            for start_idx in range(0, num_frames - self.seq_len + 1, self.overlap):
                end_idx = start_idx + self.seq_len
                
                samples.append({
                    'seq_idx': seq_idx,
                    'start_idx': start_idx,
                    'end_idx': end_idx
                })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        sequence = self.sequences[sample['seq_idx']]
        
        start_idx = sample['start_idx']
        end_idx = sample['end_idx']
        
        # Extract subsequence
        visual_features = sequence['visual_features'][start_idx:end_idx]  # [seq_len, 762] (cached features)
        imu = sequence['imu'][start_idx:end_idx]        # [seq_len, 33, 6]
        
        # Convert poses to tensor
        poses_list = sequence['poses'][start_idx:end_idx]
        poses = torch.zeros(self.seq_len, 7)  # [tx, ty, tz, qx, qy, qz, qw]
        
        for i, pose in enumerate(poses_list):
            poses[i, :3] = torch.tensor(pose['translation'])
            poses[i, 3:] = torch.tensor(pose['rotation'])
        
        # Create VIFT-compatible visual-inertial features
        # Average IMU data per frame
        imu_features = imu.mean(dim=1)  # [seq_len, 6]
        
        # Use cached visual features (already extracted by pre-trained encoder)
        # visual_features should be [seq_len, 762] from latent caching
        if visual_features.shape[1] != 762:
            print(f"Warning: Expected visual features of dim 762, got {visual_features.shape[1]}")
            # Pad or truncate to 762 dimensions
            if visual_features.shape[1] < 762:
                padding = torch.zeros(visual_features.shape[0], 762 - visual_features.shape[1])
                visual_features = torch.cat([visual_features, padding], dim=1)
            else:
                visual_features = visual_features[:, :762]
        
        # Ensure sequence length matches
        if visual_features.shape[0] != self.seq_len:
            print(f"Warning: Visual seq_len {visual_features.shape[0]} != expected {self.seq_len}")
            if visual_features.shape[0] < self.seq_len:
                # Pad with zeros
                padding = torch.zeros(self.seq_len - visual_features.shape[0], visual_features.shape[1])
                visual_features = torch.cat([visual_features, padding], dim=0)
            else:
                # Truncate
                visual_features = visual_features[:self.seq_len]
        
        # Concatenate visual (762) + IMU (6) = 768 features to match VIFT
        visual_inertial_features = torch.cat([visual_features, imu_features], dim=1)  # [seq_len, 768]
        
        # Return format compatible with original VIFT: (features, poses)
        return visual_inertial_features, poses


class AriaDataModule(LightningDataModule):
    """LightningDataModule for AriaEveryday dataset"""
    
    def __init__(
        self,
        train_data_dir: str = "data/aria_real_train",
        test_data_dir: str = "data/aria_real_test",
        train_sequences: list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94],
        val_sequences: list = [95, 96, 97, 98, 99],
        test_sequences: list = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142],
        batch_size: int = 4,
        num_workers: int = 4,
        pin_memory: bool = True,
        seq_len: int = 11,
        overlap: int = 5,
        **kwargs
    ):
        super().__init__()
        
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
    
    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`."""
        
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = AriaDataset(
                data_dir=self.hparams.train_data_dir,
                sequence_ids=self.hparams.train_sequences,
                seq_len=self.hparams.seq_len,
                overlap=self.hparams.overlap
            )
            
            self.data_val = AriaDataset(
                data_dir=self.hparams.train_data_dir,  # Use training dir for validation
                sequence_ids=self.hparams.val_sequences,
                seq_len=self.hparams.seq_len,
                overlap=self.hparams.overlap
            )
            
            self.data_test = AriaDataset(
                data_dir=self.hparams.test_data_dir,   # Use separate test dir
                sequence_ids=self.hparams.test_sequences,
                seq_len=self.hparams.seq_len,
                overlap=self.hparams.overlap
            )
    
    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )
    
    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
    
    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )


class AriaLatentDataset(Dataset):
    """Dataset for pre-cached latent features from AriaEveryday data"""
    
    def __init__(self, data_dir: str, sequence_ids: list, seq_len: int = 11, overlap: int = 5):
        self.data_dir = Path(data_dir)
        self.sequence_ids = sequence_ids
        self.seq_len = seq_len
        self.overlap = overlap
        
        # Load all sequences with cached latent features
        self.sequences = []
        for seq_id in sequence_ids:
            seq_dir = self.data_dir / f"{seq_id:02d}"
            if seq_dir.exists() and (seq_dir / "visual_features.pt").exists():
                self.sequences.append(self._load_sequence(seq_dir))
            else:
                print(f"Warning: Sequence {seq_id:02d} not found or missing cached features")
        
        # Create sliding windows
        self.samples = self._create_sliding_windows()
    
    def _load_sequence(self, seq_dir: Path) -> Dict:
        """Load a single sequence with cached latent features"""
        # Load poses
        with open(seq_dir / "poses.json", 'r') as f:
            poses = json.load(f)
        
        # Load IMU data
        imu_data = torch.load(seq_dir / "imu_data.pt")
        
        # Load cached visual features (required for latent datamodule)
        visual_features = torch.load(seq_dir / "visual_features.pt")
        
        return {
            'poses': poses,
            'visual_features': visual_features,  # Pre-cached latent features
            'imu': imu_data
        }
    
    def _create_sliding_windows(self) -> list:
        """Create sliding windows from sequences"""
        samples = []
        
        for seq_idx, sequence in enumerate(self.sequences):
            num_frames = len(sequence['poses'])
            
            # Create sliding windows
            for start_idx in range(0, num_frames - self.seq_len + 1, self.overlap):
                end_idx = start_idx + self.seq_len
                
                samples.append({
                    'seq_idx': seq_idx,
                    'start_idx': start_idx,
                    'end_idx': end_idx
                })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        sequence = self.sequences[sample['seq_idx']]
        
        start_idx = sample['start_idx']
        end_idx = sample['end_idx']
        
        # Extract subsequence
        visual_features = sequence['visual_features'][start_idx:end_idx]  # Pre-cached features
        imu = sequence['imu'][start_idx:end_idx]
        
        # Convert poses to tensor
        poses_list = sequence['poses'][start_idx:end_idx]
        poses = torch.zeros(self.seq_len, 7)  # [tx, ty, tz, qx, qy, qz, qw]
        
        for i, pose in enumerate(poses_list):
            poses[i, :3] = torch.tensor(pose['translation'])
            poses[i, 3:] = torch.tensor(pose['rotation'])
        
        # Average IMU data per frame
        imu_features = imu.mean(dim=1)  # [seq_len, 6]
        
        # Ensure visual features are 762 dimensions (as expected from latent caching)
        if visual_features.shape[1] != 762:
            print(f"Warning: Expected visual features of dim 762, got {visual_features.shape[1]}")
            if visual_features.shape[1] < 762:
                padding = torch.zeros(visual_features.shape[0], 762 - visual_features.shape[1])
                visual_features = torch.cat([visual_features, padding], dim=1)
            else:
                visual_features = visual_features[:, :762]
        
        # Ensure sequence length matches
        if visual_features.shape[0] != self.seq_len:
            if visual_features.shape[0] < self.seq_len:
                padding = torch.zeros(self.seq_len - visual_features.shape[0], visual_features.shape[1])
                visual_features = torch.cat([visual_features, padding], dim=0)
            else:
                visual_features = visual_features[:self.seq_len]
        
        # Concatenate visual (762) + IMU (6) = 768 features for VIFT compatibility
        visual_inertial_features = torch.cat([visual_features, imu_features], dim=1)
        
        return visual_inertial_features, poses


class AriaLatentDataModule(LightningDataModule):
    """LightningDataModule for pre-cached latent features from AriaEveryday dataset"""
    
    def __init__(
        self,
        train_data_dir: str = "data/aria_real_train",
        test_data_dir: str = "data/aria_real_test", 
        train_sequences: list = [0, 1],  # Default small set for sanity check
        val_sequences: list = [2],
        test_sequences: list = [3, 4],
        batch_size: int = 4,
        num_workers: int = 4,
        pin_memory: bool = True,
        seq_len: int = 11,
        overlap: int = 5,
        **kwargs
    ):
        super().__init__()
        
        # Save hyperparameters
        self.save_hyperparameters(logger=False)
        
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
    
    def setup(self, stage: Optional[str] = None):
        """Load data with cached latent features"""
        
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = AriaLatentDataset(
                data_dir=self.hparams.train_data_dir,
                sequence_ids=self.hparams.train_sequences,
                seq_len=self.hparams.seq_len,
                overlap=self.hparams.overlap
            )
            
            self.data_val = AriaLatentDataset(
                data_dir=self.hparams.train_data_dir,  # Use training dir for validation
                sequence_ids=self.hparams.val_sequences,
                seq_len=self.hparams.seq_len,
                overlap=self.hparams.overlap
            )
            
            self.data_test = AriaLatentDataset(
                data_dir=self.hparams.test_data_dir,   # Use separate test dir
                sequence_ids=self.hparams.test_sequences,
                seq_len=self.hparams.seq_len,
                overlap=self.hparams.overlap
            )
    
    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )
    
    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
    
    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )


if __name__ == "__main__":
    # Test the datamodule
    dm = AriaDataModule()
    dm.setup()
    
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    visual_inertial_features, poses = batch
    
    print(f"Visual-inertial features shape: {visual_inertial_features.shape}")  # [batch, seq_len, 768]
    print(f"Poses shape: {poses.shape}")    # [batch, seq_len, 7]