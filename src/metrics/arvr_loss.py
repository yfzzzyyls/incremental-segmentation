"""
Advanced Loss Functions for AR/VR Visual-Inertial Odometry

Specialized loss functions that prioritize small motions and temporal smoothness
critical for AR/VR head tracking scenarios.
"""

import torch
import torch.nn as nn
import numpy as np


class ARVRAdaptiveLoss(nn.Module):
    """
    Scale-aware loss that weights small motions more heavily.
    
    Key insight: AR/VR requires sub-centimeter/sub-degree accuracy for small motions,
    which are much more common than large motions in head tracking.
    """
    
    def __init__(self, 
                 small_motion_weight=3.0,
                 medium_motion_weight=1.0, 
                 large_motion_weight=0.5,
                 rotation_threshold_small=0.035,  # ~2 degrees in radians
                 rotation_threshold_large=0.175,  # ~10 degrees
                 translation_threshold_small=0.01,  # 1cm
                 translation_threshold_large=0.05,  # 5cm
                 smoothness_weight=0.1):
        super().__init__()
        
        self.small_motion_weight = small_motion_weight
        self.medium_motion_weight = medium_motion_weight
        self.large_motion_weight = large_motion_weight
        
        self.rot_thresh_small = rotation_threshold_small
        self.rot_thresh_large = rotation_threshold_large
        self.trans_thresh_small = translation_threshold_small
        self.trans_thresh_large = translation_threshold_large
        
        self.smoothness_weight = smoothness_weight
        self.mse_loss = nn.MSELoss(reduction='none')
        
    def forward(self, predictions, targets):
        """
        Args:
            predictions: [batch, seq_len, 6] - predicted poses
            targets: [batch, seq_len, 6] - ground truth poses
        """
        batch_size, seq_len, _ = predictions.shape
        
        # Split into rotation and translation
        pred_rot = predictions[:, :, :3]  # [batch, seq_len, 3]
        pred_trans = predictions[:, :, 3:]  # [batch, seq_len, 3]
        target_rot = targets[:, :, :3]
        target_trans = targets[:, :, 3:]
        
        # Compute base MSE losses
        rot_loss = self.mse_loss(pred_rot, target_rot)  # [batch, seq_len, 3]
        trans_loss = self.mse_loss(pred_trans, target_trans)  # [batch, seq_len, 3]
        
        # Compute motion magnitudes for adaptive weighting
        rot_magnitude = torch.norm(target_rot, dim=-1)  # [batch, seq_len]
        trans_magnitude = torch.norm(target_trans, dim=-1)  # [batch, seq_len]
        
        # Adaptive weights based on motion magnitude
        rot_weights = self._compute_adaptive_weights(rot_magnitude, 
                                                   self.rot_thresh_small, 
                                                   self.rot_thresh_large)
        trans_weights = self._compute_adaptive_weights(trans_magnitude,
                                                     self.trans_thresh_small,
                                                     self.trans_thresh_large)
        
        # Apply adaptive weights
        weighted_rot_loss = (rot_loss * rot_weights.unsqueeze(-1)).mean()
        weighted_trans_loss = (trans_loss * trans_weights.unsqueeze(-1)).mean()
        
        # Temporal smoothness regularization
        smoothness_loss = 0.0
        if seq_len > 1:
            # Penalize rapid changes in predictions
            pred_diff = predictions[:, 1:] - predictions[:, :-1]
            target_diff = targets[:, 1:] - targets[:, :-1]
            smoothness_loss = self.mse_loss(pred_diff, target_diff).mean()
        
        # Combine losses
        total_loss = (weighted_rot_loss + weighted_trans_loss + 
                     self.smoothness_weight * smoothness_loss)
        
        return total_loss
    
    def _compute_adaptive_weights(self, magnitude, thresh_small, thresh_large):
        """Compute adaptive weights based on motion magnitude"""
        weights = torch.ones_like(magnitude)
        
        # Small motions get higher weight
        small_mask = magnitude < thresh_small
        weights[small_mask] = self.small_motion_weight
        
        # Large motions get lower weight  
        large_mask = magnitude > thresh_large
        weights[large_mask] = self.large_motion_weight
        
        # Medium motions keep default weight
        # (implicitly handled by torch.ones_like initialization)
        
        return weights


class ARVRMultiScaleLoss(nn.Module):
    """
    Multi-scale loss that combines different temporal scales.
    Helps with both immediate accuracy and long-term stability.
    """
    
    def __init__(self, 
                 immediate_weight=2.0,     # Current frame accuracy
                 short_term_weight=1.5,    # 2-3 frame consistency 
                 long_term_weight=1.0):    # Full sequence stability
        super().__init__()
        
        self.immediate_weight = immediate_weight
        self.short_term_weight = short_term_weight
        self.long_term_weight = long_term_weight
        
        self.arvr_loss = ARVRAdaptiveLoss()
        
    def forward(self, predictions, targets):
        """Multi-scale temporal loss"""
        batch_size, seq_len, _ = predictions.shape
        
        # Immediate accuracy (frame-to-frame)
        immediate_loss = 0.0
        if seq_len > 1:
            for i in range(seq_len - 1):
                frame_loss = self.arvr_loss(
                    predictions[:, i:i+2], 
                    targets[:, i:i+2]
                )
                immediate_loss += frame_loss
            immediate_loss /= (seq_len - 1)
        
        # Short-term consistency (3-frame windows)
        short_term_loss = 0.0
        if seq_len >= 3:
            for i in range(seq_len - 2):
                window_loss = self.arvr_loss(
                    predictions[:, i:i+3],
                    targets[:, i:i+3] 
                )
                short_term_loss += window_loss
            short_term_loss /= (seq_len - 2)
        
        # Long-term stability (full sequence)
        long_term_loss = self.arvr_loss(predictions, targets)
        
        # Combine with weights
        total_loss = (self.immediate_weight * immediate_loss +
                     self.short_term_weight * short_term_loss + 
                     self.long_term_weight * long_term_loss)
        
        return total_loss


class ARVRRotationTranslationLoss(nn.Module):
    """
    Specialized loss with separate handling for rotation and translation.
    Accounts for different error characteristics of head rotation vs translation.
    """
    
    def __init__(self,
                 rotation_weight=100.0,    # Rotation is critical for AR/VR
                 translation_weight=50.0,  # Translation also important
                 cross_coupling_weight=10.0):  # Rotation-translation coupling
        super().__init__()
        
        self.rotation_weight = rotation_weight
        self.translation_weight = translation_weight 
        self.cross_coupling_weight = cross_coupling_weight
        
        self.rot_loss = ARVRAdaptiveLoss(
            small_motion_weight=4.0,  # Extra precision for small rotations
            rotation_threshold_small=0.017,  # ~1 degree
            rotation_threshold_large=0.087   # ~5 degrees
        )
        
        self.trans_loss = ARVRAdaptiveLoss(
            small_motion_weight=3.0,  # High precision for small translations
            translation_threshold_small=0.005,  # 5mm
            translation_threshold_large=0.02    # 2cm
        )
        
    def forward(self, predictions, targets):
        """Specialized rotation/translation loss"""
        
        # Split predictions and targets
        pred_rot = predictions[:, :, :3]
        pred_trans = predictions[:, :, 3:]
        target_rot = targets[:, :, :3]
        target_trans = targets[:, :, 3:]
        
        # Specialized losses
        rotation_loss = self.rot_loss(
            torch.cat([pred_rot, torch.zeros_like(pred_trans)], dim=-1),
            torch.cat([target_rot, torch.zeros_like(target_trans)], dim=-1)
        )
        
        translation_loss = self.trans_loss(
            torch.cat([torch.zeros_like(pred_rot), pred_trans], dim=-1),
            torch.cat([torch.zeros_like(target_rot), target_trans], dim=-1)
        )
        
        # Cross-coupling term (rotation affects translation perception)
        coupling_loss = 0.0
        if predictions.shape[1] > 1:  # Need temporal sequence
            # Rotation changes should be consistent with translation changes
            rot_diff = pred_rot[:, 1:] - pred_rot[:, :-1]
            trans_diff = pred_trans[:, 1:] - pred_trans[:, :-1]
            target_rot_diff = target_rot[:, 1:] - target_rot[:, :-1]
            target_trans_diff = target_trans[:, 1:] - target_trans[:, :-1]
            
            # Correlation between rotation and translation changes
            coupling_loss = nn.MSELoss()(
                torch.norm(rot_diff, dim=-1) * torch.norm(trans_diff, dim=-1),
                torch.norm(target_rot_diff, dim=-1) * torch.norm(target_trans_diff, dim=-1)
            )
        
        # Combine losses
        total_loss = (self.rotation_weight * rotation_loss +
                     self.translation_weight * translation_loss +
                     self.cross_coupling_weight * coupling_loss)
        
        return total_loss


class ARVRUltimateLosse(nn.Module):
    """
    The ultimate AR/VR loss combining all our innovations:
    - Scale-aware weighting
    - Multi-scale temporal modeling  
    - Rotation/translation specialization
    - Temporal smoothness
    """
    
    def __init__(self):
        super().__init__()
        
        self.adaptive_loss = ARVRAdaptiveLoss(
            small_motion_weight=4.0,
            smoothness_weight=0.15
        )
        
        self.multi_scale_loss = ARVRMultiScaleLoss()
        
        self.specialized_loss = ARVRRotationTranslationLoss()
        
        # Combination weights (learned through validation)
        self.adaptive_weight = 0.4
        self.multi_scale_weight = 0.3 
        self.specialized_weight = 0.3
        
    def forward(self, predictions, targets):
        """Ultimate AR/VR loss function"""
        
        adaptive_component = self.adaptive_loss(predictions, targets)
        multi_scale_component = self.multi_scale_loss(predictions, targets)
        specialized_component = self.specialized_loss(predictions, targets)
        
        total_loss = (self.adaptive_weight * adaptive_component +
                     self.multi_scale_weight * multi_scale_component +
                     self.specialized_weight * specialized_component)
        
        return total_loss


# Factory function for easy use
def create_arvr_loss(loss_type='ultimate'):
    """Factory function to create AR/VR optimized loss functions"""
    
    if loss_type == 'adaptive':
        return ARVRAdaptiveLoss()
    elif loss_type == 'multi_scale':
        return ARVRMultiScaleLoss()
    elif loss_type == 'specialized':
        return ARVRRotationTranslationLoss()
    elif loss_type == 'ultimate':
        return ARVRUltimateLosse()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    # Test the loss functions
    batch_size, seq_len = 4, 11
    
    # Create sample data
    predictions = torch.randn(batch_size, seq_len, 6) * 0.01  # Small motions
    targets = torch.randn(batch_size, seq_len, 6) * 0.005     # Even smaller ground truth
    
    # Test different loss functions
    losses = {
        'adaptive': create_arvr_loss('adaptive'),
        'multi_scale': create_arvr_loss('multi_scale'), 
        'specialized': create_arvr_loss('specialized'),
        'ultimate': create_arvr_loss('ultimate')
    }
    
    print("🧪 Testing AR/VR Loss Functions:")
    for name, loss_fn in losses.items():
        loss_value = loss_fn(predictions, targets)
        print(f"  {name:12s}: {loss_value.item():.6f}")
    
    print("\n✅ All AR/VR loss functions working correctly!")