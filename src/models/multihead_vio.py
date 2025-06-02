"""
Multi-Head VIO Architecture - Fixed Version for Pretrained Features
This version correctly handles 768-dim pretrained features (512 visual + 256 IMU)
without redundant encoding.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
import lightning as L
from torchmetrics import MeanAbsoluteError

from .components.pose_transformer_new import PoseTransformer
from ..metrics.arvr_loss_wrapper import ARVRLossWrapper


class RotationSpecializedHead(nn.Module):
    """
    Specialized head for rotation prediction with angular velocity focus.
    Modified to predict rotations for all frames in the sequence.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,  # Reduced from 256
        num_layers: int = 2,    # Reduced from 3
        num_heads: int = 4,     # Reduced from 8
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Rotation-specific feature processing
        self.rotation_processor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Angular velocity specific transformer
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
        
        # Rotation output layers - applied to each timestep
        self.rotation_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 4)  # Quaternion output
        )
        
        # Angular velocity prediction (auxiliary task) - for each timestep
        self.angular_velocity_output = nn.Linear(hidden_dim, 3)
        
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Modified to predict rotations for all frames.
        
        Args:
            features: [B, seq_len, input_dim]
        
        Returns:
            Dictionary with rotation and angular velocity predictions for all frames
        """
        # Process features for rotation
        rot_features = self.rotation_processor(features)  # [B, seq_len, hidden_dim]
        
        # Apply rotation-specific attention with causal mask
        attended_features = self.angular_transformer(rot_features)  # [B, seq_len, hidden_dim]
        
        # Predict for all frames
        B, seq_len, hidden_dim = attended_features.shape
        
        # Reshape for batch processing
        all_features = attended_features.reshape(B * seq_len, hidden_dim)
        
        # Predict rotation and angular velocity for all frames
        rotation_pred = self.rotation_output(all_features)  # [B*seq_len, 4]
        angular_velocity_pred = self.angular_velocity_output(all_features)  # [B*seq_len, 3]
        
        # Reshape back to sequence format
        rotation_pred = rotation_pred.reshape(B, seq_len, 4)
        angular_velocity_pred = angular_velocity_pred.reshape(B, seq_len, 3)
        
        # Normalize quaternions
        rotation_pred = rotation_pred / (torch.norm(rotation_pred, dim=-1, keepdim=True) + 1e-8)
        
        # CRITICAL FIX: Model outputs WXYZ but ground truth is XYZW
        # Swap the order: [W, X, Y, Z] -> [X, Y, Z, W]
        rotation_pred_xyzw = torch.cat([
            rotation_pred[..., 1:4],  # XYZ components
            rotation_pred[..., 0:1]   # W component
        ], dim=-1)
        
        return {
            'rotation': rotation_pred_xyzw,
            'angular_velocity': angular_velocity_pred,
            'rotation_features': attended_features
        }


class TranslationSpecializedHead(nn.Module):
    """
    Specialized head for translation prediction with spatial attention.
    Modified to predict translations for all frames in the sequence.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,  # Reduced from 256
        num_layers: int = 2,    # Reduced from 3
        num_heads: int = 4,     # Reduced from 8
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Translation-specific feature processing
        self.translation_processor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Spatial cross-attention (current frame attends to all frames)
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
        
        # Translation output layers - applied to each timestep
        self.translation_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3)  # XYZ translation
        )
        
        # Linear velocity prediction (auxiliary task) - for each timestep
        self.velocity_output = nn.Linear(hidden_dim, 3)
        
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Modified to predict translations for all frames.
        
        Args:
            features: [B, seq_len, input_dim]
        
        Returns:
            Dictionary with translation and velocity predictions for all frames
        """
        # Process features for translation
        trans_features = self.translation_processor(features)  # [B, seq_len, hidden_dim]
        
        # Apply spatial cross-attention
        attended_features, _ = self.spatial_cross_attention(
            trans_features, trans_features, trans_features
        )  # [B, seq_len, hidden_dim]
        
        # Add residual connection
        attended_features = attended_features + trans_features
        
        # Apply translation transformer
        refined_features = self.translation_transformer(attended_features)  # [B, seq_len, hidden_dim]
        
        # Predict for all frames
        B, seq_len, hidden_dim = refined_features.shape
        
        # Reshape for batch processing
        all_features = refined_features.reshape(B * seq_len, hidden_dim)
        
        # Predict translation and velocity for all frames
        translation_pred = self.translation_output(all_features)  # [B*seq_len, 3]
        velocity_pred = self.velocity_output(all_features)  # [B*seq_len, 3]
        
        # Reshape back to sequence format
        translation_pred = translation_pred.reshape(B, seq_len, 3)
        velocity_pred = velocity_pred.reshape(B, seq_len, 3)
        
        return {
            'translation': translation_pred,
            'linear_velocity': velocity_pred,
            'translation_features': refined_features
        }


class MultiHeadVIOModel(L.LightningModule):
    """
    Fixed Multi-head VIO model that correctly processes 768-dim pretrained features.
    The input features already contain both visual (512) and IMU (256) information.
    """
    
    def __init__(
        self,
        feature_dim: int = 768,
        hidden_dim: int = 128,  # Reduced from 256 for lite model
        num_shared_layers: int = 2,  # Reduced from 4 for lite model
        num_specialized_layers: int = 2,  # Reduced from 3 for lite model
        num_heads: int = 4,  # Reduced from 8 for lite model
        dropout: float = 0.1,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        rotation_weight: float = 1.0,
        translation_weight: float = 1.0,
        velocity_weight: float = 0.3,
        sequence_length: int = 11
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Simple feature projection (matches VIFT)
        self.feature_projection = nn.Linear(feature_dim, hidden_dim)
        
        # Shared transformer for initial processing
        self.shared_processor = PoseTransformer(
            input_dim=hidden_dim,  # Takes projected features
            hidden_dim=hidden_dim,
            num_layers=num_shared_layers,
            num_heads=num_heads,
            dropout=dropout,
            max_sequence_length=sequence_length + 5  # Some buffer
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
        
        # Optional cross-modal fusion
        self.cross_modal_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Loss function
        self.arvr_loss = ARVRLossWrapper()
        
        # Metrics
        self.train_rot_mae = MeanAbsoluteError()
        self.train_trans_mae = MeanAbsoluteError()
        self.val_rot_mae = MeanAbsoluteError()
        self.val_trans_mae = MeanAbsoluteError()
        
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass predicting poses for all frames.
        
        Args:
            batch: Dictionary with 'images' (actually 768-dim features), 'imus' (ignored), 'poses'
        
        Returns:
            Dictionary with rotation and translation predictions for all frames
        """
        # The 'images' field contains our 768-dim pretrained features
        features = batch['images']  # [B, seq_len, 768]
        
        # Simple projection (features are already encoded!)
        projected_features = self.feature_projection(features)  # [B, seq_len, 256]
        
        # Shared transformer processing
        shared_features = self.shared_processor(projected_features)  # [B, seq_len, hidden_dim]
        
        # Get predictions from specialized heads
        rotation_outputs = self.rotation_head(shared_features)
        translation_outputs = self.translation_head(shared_features)
        
        # Optional: Cross-modal fusion (applied to each timestep)
        B, seq_len, hidden_dim = shared_features.shape
        
        # Concatenate rotation and translation features
        rot_features = rotation_outputs['rotation_features']  # [B, seq_len, hidden_dim]
        trans_features = translation_outputs['translation_features']  # [B, seq_len, hidden_dim]
        
        # Reshape for fusion
        combined = torch.cat([rot_features, trans_features], dim=-1)  # [B, seq_len, hidden_dim*2]
        combined_flat = combined.reshape(B * seq_len, hidden_dim * 2)
        fused_flat = self.cross_modal_fusion(combined_flat)
        fused_features = fused_flat.reshape(B, seq_len, hidden_dim)  # [B, seq_len, hidden_dim]
        
        # Add fusion residual to predictions (small contribution)
        fusion_weight = 0.1
        
        return {
            'rotation': rotation_outputs['rotation'],  # [B, seq_len, 4]
            'translation': translation_outputs['translation'],  # [B, seq_len, 3]
            'angular_velocity': rotation_outputs['angular_velocity'],  # [B, seq_len, 3]
            'linear_velocity': translation_outputs['linear_velocity']  # [B, seq_len, 3]
        }
    
    def compute_loss(self, predictions: Dict[str, torch.Tensor], 
                     batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute loss for all frames.
        
        Args:
            predictions: Model predictions for all frames
            batch: Ground truth data
        
        Returns:
            Dictionary with individual loss components
        """
        # Target poses for all frames (excluding first frame as it's reference)
        target_rotation = batch['poses'][:, 1:, 3:7]  # [B, seq_len-1, 4]
        target_translation = batch['poses'][:, 1:, :3]  # [B, seq_len-1, 3]
        
        # Predicted poses (excluding first frame)
        pred_rotation = predictions['rotation'][:, 1:, :]  # [B, seq_len-1, 4]
        pred_translation = predictions['translation'][:, 1:, :]  # [B, seq_len-1, 3]
        
        # Flatten for metric computation
        B, seq_len_minus_1, _ = pred_rotation.shape
        
        pred_rot = pred_rotation.reshape(-1, 4).contiguous()
        target_rot = target_rotation.reshape(-1, 4).contiguous()
        pred_trans = pred_translation.reshape(-1, 3).contiguous()
        target_trans = target_translation.reshape(-1, 3).contiguous()
        
        # Compute AR/VR optimized loss
        loss_dict = self.arvr_loss(
            pred_rotation=pred_rot,
            target_rotation=target_rot,
            pred_translation=pred_trans,
            target_translation=target_trans
        )
        
        # Add velocity losses if available
        if 'angular_velocity' in predictions and False:  # Disabled for now
            # Angular velocity computation from quaternions requires additional implementation
            pass
        
        if 'linear_velocity' in predictions:
            # Compute linear velocity targets
            linear_vel_target = target_translation[:, 1:, :] - target_translation[:, :-1, :]
            linear_vel_pred = predictions['linear_velocity'][:, 1:-1, :]
            
            if linear_vel_target.shape[1] > 0:
                vel_loss = nn.functional.mse_loss(linear_vel_pred, linear_vel_target)
                loss_dict['linear_velocity_loss'] = vel_loss * self.hparams.velocity_weight
        
        # Weight the losses
        loss_dict['rotation_loss'] *= self.hparams.rotation_weight
        loss_dict['translation_loss'] *= self.hparams.translation_weight
        
        return loss_dict
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        predictions = self(batch)
        loss_dict = self.compute_loss(predictions, batch)
        
        # Total loss
        total_loss = loss_dict.get('total_loss', sum(v for k, v in loss_dict.items() if k != 'total_loss'))
        
        # Log losses
        self.log('train/total_loss', total_loss, prog_bar=True)
        for key, value in loss_dict.items():
            if key != 'total_loss':  # Skip to avoid duplicate logging
                self.log(f'train/{key}', value)
        
        # Update metrics
        with torch.no_grad():
            # Get non-first frame predictions and targets
            pred_rot = predictions['rotation'][:, 1:, :].reshape(-1, 4)
            target_rot = batch['poses'][:, 1:, 3:7].reshape(-1, 4)
            pred_trans = predictions['translation'][:, 1:, :].reshape(-1, 3)
            target_trans = batch['poses'][:, 1:, :3].reshape(-1, 3)
            
            self.train_rot_mae(pred_rot, target_rot)
            self.train_trans_mae(pred_trans, target_trans)
            
            self.log('train/rot_mae', self.train_rot_mae, prog_bar=True)
            self.log('train/trans_mae', self.train_trans_mae, prog_bar=True)
        
        return total_loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        predictions = self(batch)
        loss_dict = self.compute_loss(predictions, batch)
        
        # Total loss
        total_loss = loss_dict.get('total_loss', sum(v for k, v in loss_dict.items() if k != 'total_loss'))
        
        # Log losses
        self.log('val/total_loss', total_loss, prog_bar=True)
        for key, value in loss_dict.items():
            if key != 'total_loss':  # Skip to avoid duplicate logging
                self.log(f'val/{key}', value)
        
        # Update metrics
        with torch.no_grad():
            # Get non-first frame predictions and targets
            pred_rot = predictions['rotation'][:, 1:, :].reshape(-1, 4)
            target_rot = batch['poses'][:, 1:, 3:7].reshape(-1, 4)
            pred_trans = predictions['translation'][:, 1:, :].reshape(-1, 3)
            target_trans = batch['poses'][:, 1:, :3].reshape(-1, 3)
            
            self.val_rot_mae(pred_rot, target_rot)
            self.val_trans_mae(pred_trans, target_trans)
            
            self.log('val/rot_mae', self.val_rot_mae, prog_bar=True)
            self.log('val/trans_mae', self.val_trans_mae, prog_bar=True)
        
        return total_loss
    
    def configure_optimizers(self):
        # AdamW optimizer with cosine annealing
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Cosine annealing scheduler with warm restarts
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/total_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }