"""
Multi-Head VIO Architecture with Separate Visual and IMU Processing
This version processes visual and IMU features separately as in the original VIFT
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
import lightning as L
from torchmetrics import MeanAbsoluteError

from .components.pose_transformer_new import PoseTransformer
from ..metrics.arvr_loss_wrapper import ARVRLossWrapper


class MultiModalPoseTransformer(nn.Module):
    """
    Transformer that processes visual and IMU features separately
    then fuses them for pose prediction
    """
    
    def __init__(
        self,
        visual_dim: int = 512,
        imu_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_sequence_length: int = 20
    ):
        super().__init__()
        
        # Separate projections for visual and IMU
        self.visual_projection = nn.Linear(visual_dim, hidden_dim)
        self.imu_projection = nn.Linear(imu_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_sequence_length, hidden_dim) * 0.02
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Layer norm
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, visual_features: torch.Tensor, imu_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual_features: [B, seq_len, 512]
            imu_features: [B, seq_len, 256]
            
        Returns:
            fused_features: [B, seq_len, hidden_dim]
        """
        B, seq_len, _ = visual_features.shape
        
        # Project features
        visual_proj = self.visual_projection(visual_features)  # [B, seq_len, hidden_dim]
        imu_proj = self.imu_projection(imu_features)          # [B, seq_len, hidden_dim]
        
        # Combine features (can use addition, concatenation, or interleaving)
        # Using addition as in the original VIFT
        combined = visual_proj + imu_proj  # [B, seq_len, hidden_dim]
        
        # Add positional encoding
        combined = combined + self.pos_encoding[:, :seq_len, :]
        
        # Apply transformer
        output = self.transformer(combined)
        
        # Final normalization
        output = self.norm(output)
        
        return output


class RotationSpecializedHead(nn.Module):
    """Specialized head for rotation prediction"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Rotation-specific processing
        self.rotation_processor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Angular transformer
        self.angular_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_layers
        )
        
        # Output layers
        self.rotation_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 4)  # Quaternion output
        )
        
        self.angular_velocity_output = nn.Linear(hidden_dim, 3)
        
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, seq_len, _ = features.shape
        
        # Process features
        rot_features = self.rotation_processor(features)
        attended_features = self.angular_transformer(rot_features)
        
        # Reshape for batch processing
        all_features = attended_features.reshape(B * seq_len, -1)
        
        # Predictions
        rotation_pred = self.rotation_output(all_features)
        angular_velocity_pred = self.angular_velocity_output(all_features)
        
        # Reshape back
        rotation_pred = rotation_pred.reshape(B, seq_len, 4)
        angular_velocity_pred = angular_velocity_pred.reshape(B, seq_len, 3)
        
        # Normalize quaternions
        rotation_pred = rotation_pred / (torch.norm(rotation_pred, dim=-1, keepdim=True) + 1e-8)
        
        # Convert from WXYZ to XYZW
        rotation_pred_xyzw = torch.cat([
            rotation_pred[..., 1:4],  # XYZ
            rotation_pred[..., 0:1]   # W
        ], dim=-1)
        
        return {
            'rotation': rotation_pred_xyzw,
            'angular_velocity': angular_velocity_pred,
            'rotation_features': attended_features
        }


class TranslationSpecializedHead(nn.Module):
    """Specialized head for translation prediction"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Translation-specific processing
        self.translation_processor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Spatial attention
        self.spatial_cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Translation transformer
        self.translation_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_layers
        )
        
        # Output layers
        self.translation_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3)  # XYZ translation
        )
        
        self.velocity_output = nn.Linear(hidden_dim, 3)
        
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, seq_len, _ = features.shape
        
        # Process features
        trans_features = self.translation_processor(features)
        
        # Apply spatial attention
        attended_features, _ = self.spatial_cross_attention(
            trans_features, trans_features, trans_features
        )
        attended_features = attended_features + trans_features
        
        # Apply transformer
        refined_features = self.translation_transformer(attended_features)
        
        # Reshape for batch processing
        all_features = refined_features.reshape(B * seq_len, -1)
        
        # Predictions
        translation_pred = self.translation_output(all_features)
        velocity_pred = self.velocity_output(all_features)
        
        # Reshape back
        translation_pred = translation_pred.reshape(B, seq_len, 3)
        velocity_pred = velocity_pred.reshape(B, seq_len, 3)
        
        return {
            'translation': translation_pred,
            'linear_velocity': velocity_pred,
            'translation_features': refined_features
        }


class MultiHeadVIOModelSeparate(L.LightningModule):
    """
    Multi-head VIO model with separate visual and IMU processing
    """
    
    def __init__(
        self,
        visual_dim: int = 512,
        imu_dim: int = 256,
        hidden_dim: int = 256,
        num_shared_layers: int = 4,
        num_specialized_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        rotation_weight: float = 1.0,
        translation_weight: float = 1.0,
        velocity_weight: float = 0.3,
        sequence_length: int = 10
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Multi-modal transformer for shared processing
        self.shared_processor = MultiModalPoseTransformer(
            visual_dim=visual_dim,
            imu_dim=imu_dim,
            hidden_dim=hidden_dim,
            num_layers=num_shared_layers,
            num_heads=num_heads,
            dropout=dropout,
            max_sequence_length=sequence_length + 5
        )
        
        # Specialized heads
        self.rotation_head = RotationSpecializedHead(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_specialized_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.translation_head = TranslationSpecializedHead(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_specialized_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Cross-modal fusion
        self.cross_modal_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Loss function with improved settings
        self.arvr_loss = ARVRLossWrapper(use_log_scale=True, use_weighted_loss=False)
        
        # Metrics
        self.train_rot_mae = MeanAbsoluteError()
        self.train_trans_mae = MeanAbsoluteError()
        self.val_rot_mae = MeanAbsoluteError()
        self.val_trans_mae = MeanAbsoluteError()
        
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass with separate visual and IMU features
        
        Args:
            batch: Dictionary with 'visual_features', 'imu_features', 'poses'
        
        Returns:
            Dictionary with rotation and translation predictions
        """
        # Extract separate features
        visual_features = batch.get('visual_features', None)
        imu_features = batch.get('imu_features', None)
        
        # Handle backward compatibility
        if visual_features is None or imu_features is None:
            # Fall back to concatenated features
            combined_features = batch['images']  # [B, seq_len, 768]
            visual_features = combined_features[..., :512]
            imu_features = combined_features[..., 512:]
        
        # Process through shared transformer
        shared_features = self.shared_processor(visual_features, imu_features)
        
        # Get predictions from specialized heads
        rotation_outputs = self.rotation_head(shared_features)
        translation_outputs = self.translation_head(shared_features)
        
        # Optional cross-modal fusion
        B, seq_len, hidden_dim = shared_features.shape
        
        rot_features = rotation_outputs['rotation_features']
        trans_features = translation_outputs['translation_features']
        
        combined = torch.cat([rot_features, trans_features], dim=-1)
        combined_flat = combined.reshape(B * seq_len, -1)
        fused_flat = self.cross_modal_fusion(combined_flat)
        fused_features = fused_flat.reshape(B, seq_len, hidden_dim)
        
        return {
            'rotation': rotation_outputs['rotation'],
            'translation': translation_outputs['translation'],
            'angular_velocity': rotation_outputs['angular_velocity'],
            'linear_velocity': translation_outputs['linear_velocity']
        }
    
    def compute_loss(self, predictions: Dict[str, torch.Tensor], 
                     batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute loss for all frames"""
        # Target poses (all are transitions now, no need to skip first)
        target_rotation = batch['poses'][:, :, 3:7]
        target_translation = batch['poses'][:, :, :3]
        
        # Predictions
        pred_rotation = predictions['rotation']
        pred_translation = predictions['translation']
        
        # Flatten for loss computation
        B, seq_len, _ = pred_rotation.shape
        
        pred_rot = pred_rotation.reshape(-1, 4).contiguous()
        target_rot = target_rotation.reshape(-1, 4).contiguous()
        pred_trans = pred_translation.reshape(-1, 3).contiguous()
        target_trans = target_translation.reshape(-1, 3).contiguous()
        
        # Compute losses
        loss_dict = self.arvr_loss(
            pred_rotation=pred_rot,
            target_rotation=target_rot,
            pred_translation=pred_trans,
            target_translation=target_trans
        )
        
        # Weight losses
        loss_dict['rotation_loss'] *= self.hparams.rotation_weight
        loss_dict['translation_loss'] *= self.hparams.translation_weight
        
        return loss_dict
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        predictions = self(batch)
        loss_dict = self.compute_loss(predictions, batch)
        
        total_loss = loss_dict.get('total_loss', sum(v for k, v in loss_dict.items() if k != 'total_loss'))
        
        # Log losses with actual values (not just 0.0000)
        self.log('train/total_loss', total_loss, prog_bar=True)
        for key, value in loss_dict.items():
            if key != 'total_loss':
                self.log(f'train/{key}', value)
        
        # Log gradient norms for debugging
        if batch_idx % 100 == 0:  # Every 100 batches
            total_norm = 0
            for p in self.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            self.log('train/grad_norm', total_norm)
        
        # Update metrics
        with torch.no_grad():
            pred_rot = predictions['rotation'].reshape(-1, 4).contiguous()
            target_rot = batch['poses'][:, :, 3:7].reshape(-1, 4).contiguous()
            pred_trans = predictions['translation'].reshape(-1, 3).contiguous()
            target_trans = batch['poses'][:, :, :3].reshape(-1, 3).contiguous()
            
            self.train_rot_mae(pred_rot, target_rot)
            self.train_trans_mae(pred_trans, target_trans)
            
            self.log('train/rot_mae', self.train_rot_mae, prog_bar=True)
            self.log('train/trans_mae', self.train_trans_mae, prog_bar=True)
            
            # Log prediction statistics
            if batch_idx % 100 == 0:
                self.log('train/pred_trans_mean', pred_trans.abs().mean())
                self.log('train/pred_trans_std', pred_trans.std())
                self.log('train/target_trans_mean', target_trans.abs().mean())
                self.log('train/target_trans_std', target_trans.std())
        
        return total_loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        predictions = self(batch)
        loss_dict = self.compute_loss(predictions, batch)
        
        total_loss = loss_dict.get('total_loss', sum(v for k, v in loss_dict.items() if k != 'total_loss'))
        
        # Log losses
        self.log('val/total_loss', total_loss, prog_bar=True)
        for key, value in loss_dict.items():
            if key != 'total_loss':
                self.log(f'val/{key}', value)
        
        # Update metrics
        with torch.no_grad():
            pred_rot = predictions['rotation'].reshape(-1, 4).contiguous()
            target_rot = batch['poses'][:, :, 3:7].reshape(-1, 4).contiguous()
            pred_trans = predictions['translation'].reshape(-1, 3).contiguous()
            target_trans = batch['poses'][:, :, :3].reshape(-1, 3).contiguous()
            
            self.val_rot_mae(pred_rot, target_rot)
            self.val_trans_mae(pred_trans, target_trans)
            
            self.log('val/rot_mae', self.val_rot_mae, prog_bar=True)
            self.log('val/trans_mae', self.val_trans_mae, prog_bar=True)
        
        return total_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-7  # Slightly higher epsilon for stability
        )
        
        # Use OneCycleLR for better training dynamics
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.learning_rate * 10,  # Peak at 10x base LR
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.3,  # 30% of training for warmup
            anneal_strategy='cos',
            div_factor=10,  # Start at LR/10
            final_div_factor=100  # End at LR/100
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',  # Update every step, not epoch
                'frequency': 1
            }
        }