#!/usr/bin/env python3
"""
Training script that converts absolute poses to relative poses.
This matches the format expected by the model.
"""

import os
import sys
import torch
import torch.nn as nn
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint, 
    EarlyStopping, 
    RichProgressBar,
    LearningRateMonitor,
    Callback
)
from rich.console import Console
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.transform import Rotation

# Add src to path
sys.path.append('src')

from src.models.multihead_vio import MultiHeadVIOModel

console = Console()


def quaternion_multiply(q1, q2):
    """Multiply two quaternions in XYZW format."""
    # Unpack XYZW format
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    
    # Quaternion multiplication
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    # Return in XYZW format
    return np.array([x, y, z, w])


def quaternion_inverse(q):
    """Compute quaternion inverse for XYZW format."""
    # Unpack XYZW format
    x, y, z, w = q
    norm_sq = w*w + x*x + y*y + z*z
    # Return conjugate/norm_sq in XYZW format
    return np.array([-x, -y, -z, w]) / norm_sq


def convert_absolute_to_relative(poses):
    """
    Convert absolute poses to relative poses.
    
    Args:
        poses: [seq_len, 7] absolute poses (translation + quaternion)
    
    Returns:
        relative_poses: [seq_len, 7] relative poses
    """
    seq_len = poses.shape[0]
    relative_poses = np.zeros_like(poses)
    
    # First pose is always at origin
    relative_poses[0, :3] = [0, 0, 0]
    relative_poses[0, 3:] = [0, 0, 0, 1]  # Identity quaternion in XYZW format
    
    # Convert subsequent poses to relative
    for i in range(1, seq_len):
        # Get current and previous absolute poses
        prev_trans = poses[i-1, :3]
        prev_rot = poses[i-1, 3:]
        
        curr_trans = poses[i, :3]
        curr_rot = poses[i, 3:]
        
        # Compute relative translation
        # In the frame of the previous pose
        trans_diff = curr_trans - prev_trans
        
        # For rotation, we need the relative rotation
        # rel_rot = prev_rot^-1 * curr_rot
        prev_rot_inv = quaternion_inverse(prev_rot)
        rel_rot = quaternion_multiply(prev_rot_inv, curr_rot)
        
        # Store relative pose
        relative_poses[i, :3] = trans_diff
        relative_poses[i, 3:] = rel_rot / np.linalg.norm(rel_rot)  # Normalize
    
    return relative_poses


class RelativePoseDataset(Dataset):
    """Dataset that converts absolute poses to relative poses."""
    
    def __init__(self, data_dir: str, pose_scale: float = 100.0, max_samples: int = None):
        self.data_dir = data_dir
        self.pose_scale = pose_scale
        
        # Find all samples
        self.samples = []
        i = 0
        consecutive_misses = 0
        
        while consecutive_misses < 100:
            feature_path = os.path.join(data_dir, f"{i}.npy")
            gt_path = os.path.join(data_dir, f"{i}_gt.npy")
            
            if os.path.exists(feature_path) and os.path.exists(gt_path):
                self.samples.append(i)
                consecutive_misses = 0
            else:
                consecutive_misses += 1
            
            i += 1
            
            if max_samples and len(self.samples) >= max_samples:
                break
        
        console.print(f"  Found {len(self.samples)} samples in {data_dir}")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_id = self.samples[idx]
        
        # Load features
        features = np.load(os.path.join(self.data_dir, f"{sample_id}.npy"))
        features = torch.from_numpy(features).float()  # [11, 768]
        
        # Load ground truth poses (already in relative format if using fixed data)
        relative_poses = np.load(os.path.join(self.data_dir, f"{sample_id}_gt.npy"))
        
        # Check if we need to convert (old data) or if already relative (fixed data)
        # If first pose is not at origin, it's absolute poses that need conversion
        if np.linalg.norm(relative_poses[0, :3]) > 1e-6 or np.linalg.norm(relative_poses[0, 3:] - [0, 0, 0, 1]) > 1e-6:
            # Old format - convert to relative
            relative_poses = convert_absolute_to_relative(relative_poses)
            # Scale translation from meters to centimeters
            relative_poses[:, :3] *= self.pose_scale
        # else: already in relative format with correct scale
        
        # Convert to tensor
        relative_poses = torch.from_numpy(relative_poses).float()
        
        # Create IMU placeholder
        imus = torch.zeros(11, 6)
        
        return features, imus, relative_poses


class FirstBatchLogger(Callback):
    """Log detailed information about the first batch."""
    
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if trainer.current_epoch == 0 and batch_idx == 0:
            console.print("\n[bold cyan]First Batch Analysis:[/bold cyan]")
            console.print(f"  Batch keys: {batch.keys()}")
            console.print(f"  Images shape: {batch['images'].shape}")
            console.print(f"  Poses shape: {batch['poses'].shape}")
            
            # Check pose statistics
            poses = batch['poses']
            console.print(f"\n  Pose statistics:")
            console.print(f"    First frame (should be origin):")
            console.print(f"      Translation: {poses[0, 0, :3].cpu().numpy()}")
            console.print(f"      Rotation: {poses[0, 0, 3:].cpu().numpy()}")
            
            console.print(f"\n    Frame-to-frame translations:")
            for i in range(1, min(4, poses.shape[1])):
                trans = poses[0, i, :3].cpu().numpy()
                console.print(f"      Frame {i}: {trans} (norm: {np.linalg.norm(trans):.4f} cm)")
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.current_epoch == 0 and batch_idx == 0:
            loss = outputs['loss'].item() if isinstance(outputs, dict) else outputs.item()
            console.print(f"\n[bold yellow]First batch loss: {loss:.4f}[/bold yellow]")
            
            if loss > 10:
                console.print("[bold red]⚠️  Loss is still high![/bold red]")
            elif loss > 5:
                console.print("[bold yellow]⚠️  Loss is elevated but better[/bold yellow]")
            else:
                console.print("[bold green]✅ Loss is in good range![/bold green]")


def train_with_relative_poses(
    data_dir: str = "aria_latent_data_pretrained",
    pose_scale: float = 100.0,
    learning_rate: float = 1e-4,
    batch_size: int = 32,
    max_epochs: int = 50,
    use_wandb: bool = True
):
    """Train with relative poses converted from absolute poses."""
    
    console.rule(f"[bold cyan]🚀 Training with Relative Poses[/bold cyan]")
    console.print(f"\nConfiguration:")
    console.print(f"  Pose scale: {pose_scale} (meters → centimeters)")
    console.print(f"  Pose format: Relative (converted from absolute)")
    console.print(f"  Learning rate: {learning_rate}")
    console.print(f"  Batch size: {batch_size}")
    
    # Set seed
    L.seed_everything(42, workers=True)
    
    # Create model (Lite configuration - reduced from 7.4M to ~1.1M params)
    model = MultiHeadVIOModel(
        feature_dim=768,
        hidden_dim=128,         
        num_shared_layers=2,    
        num_specialized_layers=2, 
        num_heads=4,
        dropout=0.1,
        learning_rate=learning_rate,
        weight_decay=1e-5,
        sequence_length=11
    )
    
    # Create datasets
    console.print("\n[bold]Loading data with relative pose conversion...[/bold]")
    train_dataset = RelativePoseDataset(
        f"{data_dir}/train",
        pose_scale=pose_scale,
        max_samples=None
    )
    
    val_dataset = RelativePoseDataset(
        f"{data_dir}/val",
        pose_scale=pose_scale,
        max_samples=None
    )
    
    # Create dataloaders
    def collate_fn(batch):
        features, imus, poses = zip(*batch)
        return {
            'images': torch.stack(features),
            'imus': torch.stack(imus),
            'poses': torch.stack(poses)
        }
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    console.print(f"  Training samples: {len(train_dataset):,}")
    console.print(f"  Validation samples: {len(val_dataset):,}")
    
    # Verify data
    console.print("\n[bold]Verifying relative pose conversion...[/bold]")
    sample_batch = next(iter(train_loader))
    first_poses = sample_batch['poses'][0].cpu().numpy()
    console.print(f"  First sample poses:")
    console.print(f"    Frame 0: {first_poses[0]} (should be near origin)")
    console.print(f"    Frame 1: {first_poses[1]} (relative movement)")
    
    # Setup loggers
    loggers = []
    
    tb_logger = TensorBoardLogger(
        save_dir="logs",
        name="pretrained_relative",
        version=f"scale_{pose_scale}"
    )
    loggers.append(tb_logger)
    
    if use_wandb:
        wandb_logger = WandbLogger(
            project="vift-aea-pretrained",
            name=f"relative_poses_scale_{pose_scale}",
            tags=["pretrained", "relative-poses", f"scale_{pose_scale}"]
        )
        loggers.append(wandb_logger)
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            monitor="val/total_loss",
            mode="min",
            save_top_k=3,
            dirpath=f"logs/checkpoints_lite_scale_{pose_scale}",
            filename="epoch_{epoch:03d}_{val_total_loss:.4f}",
            save_last=True,
            verbose=True
        ),
        EarlyStopping(
            monitor="val/total_loss",
            patience=5,  # Reduced from 10 since model converges quickly
            mode="min",
            verbose=True
        ),
        LearningRateMonitor(logging_interval='step'),
        RichProgressBar(refresh_rate=10),
        FirstBatchLogger()
    ]
    
    # Create trainer
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        gradient_clip_val=1.0,
        accumulate_grad_batches=2,
        logger=loggers,
        callbacks=callbacks,
        log_every_n_steps=20,
        val_check_interval=0.5
    )
    
    # Train
    console.rule("[bold cyan]🏃 Starting Training with Relative Poses[/bold cyan]")
    console.print("\n[bold yellow]Key improvements:[/bold yellow]")
    console.print("  • Converted absolute poses to relative")
    console.print("  • First frame always at origin")
    console.print("  • Frame-to-frame movements only")
    console.print("  • Should match original data format")
    console.print()
    
    try:
        trainer.fit(model, train_loader, val_loader)
        
        console.rule("[bold green]✅ Training Completed![/bold green]")
        
        if trainer.checkpoint_callback:
            best_path = trainer.checkpoint_callback.best_model_path
            best_loss = trainer.checkpoint_callback.best_model_score
            
            console.print(f"\n[bold]Best checkpoint:[/bold] {best_path}")
            console.print(f"[bold]Best validation loss:[/bold] {best_loss:.6f}")
            
            # Show model size
            total_params = sum(p.numel() for p in model.parameters())
            console.print(f"\n[bold]Model size:[/bold] {total_params/1e6:.1f}M parameters")
            console.print(f"[bold]Reduction:[/bold] ~85% fewer params than full model")
            
            console.print("\n[bold cyan]Next step - Evaluate:[/bold cyan]")
            console.print(f"python evaluate_with_metrics.py \\")
            console.print(f"    --checkpoint {best_path}")
            
    except Exception as e:
        console.print(f"\n[bold red]Training failed: {e}[/bold red]")
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train with relative poses')
    parser.add_argument('--data_dir', type=str, default='aria_latent_data_pretrained',
                       help='Data directory (default: aria_latent_data_pretrained)')
    parser.add_argument('--scale', type=float, default=100.0,
                       help='Pose scale factor (default: 100.0 for meter to cm conversion)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Max epochs')
    parser.add_argument('--no-wandb', action='store_true',
                       help='Disable W&B')
    
    args = parser.parse_args()
    
    console.print("[bold magenta]Training with Relative Poses[/bold magenta]")
    console.print("Converting absolute world coordinates to relative poses\n")
    
    train_with_relative_poses(
        data_dir=args.data_dir,
        pose_scale=args.scale,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        use_wandb=not args.no_wandb
    )