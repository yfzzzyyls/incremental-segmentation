"""
Wrapper for ARVRAdaptiveLoss to handle separate rotation and translation inputs
"""

import torch
import torch.nn as nn
from .arvr_loss import ARVRAdaptiveLoss


class ProperQuaternionLoss(nn.Module):
    """
    Proper quaternion loss that accounts for double cover and uses geodesic distance.
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(self, pred_q, target_q):
        """
        Compute quaternion loss using proper geodesic distance.
        
        Args:
            pred_q: [B, 4] predicted quaternions (XYZW)
            target_q: [B, 4] target quaternions (XYZW)
        
        Returns:
            Scalar loss
        """
        # Normalize quaternions
        pred_q = pred_q / (torch.norm(pred_q, dim=-1, keepdim=True) + 1e-8)
        target_q = target_q / (torch.norm(target_q, dim=-1, keepdim=True) + 1e-8)
        
        # Compute dot product
        dot = torch.sum(pred_q * target_q, dim=-1)
        
        # Handle double cover: if dot < 0, flip the predicted quaternion
        # This ensures we're always taking the shorter path
        mask = dot < 0
        pred_q[mask] = -pred_q[mask]
        dot[mask] = -dot[mask]
        
        # Clamp to avoid numerical issues with acos
        dot = torch.clamp(dot, -1.0, 1.0)
        
        # Geodesic distance: angle = 2 * arccos(|dot|)
        # Loss = 1 - |dot| is a good approximation and more stable
        loss = 1.0 - torch.abs(dot)
        
        return loss.mean()


class ARVRLossWrapper(nn.Module):
    """
    Wrapper that adapts the ARVRAdaptiveLoss to work with separate
    rotation (quaternion) and translation predictions.
    """
    
    def __init__(self):
        super().__init__()
        self.translation_loss = nn.MSELoss()
        self.rotation_loss = ProperQuaternionLoss()
        
    def forward(self, pred_rotation, target_rotation, pred_translation, target_translation):
        """
        Args:
            pred_rotation: [B*seq_len, 4] quaternions
            target_rotation: [B*seq_len, 4] quaternions
            pred_translation: [B*seq_len, 3] translations
            target_translation: [B*seq_len, 3] translations
        
        Returns:
            Dictionary with loss components
        """
        # Compute basic losses
        rot_loss = self.rotation_loss(pred_rotation, target_rotation)
        trans_loss = self.translation_loss(pred_translation, target_translation)
        
        # Scale rotation loss to be in similar range as translation
        # Rotation loss is in [0, 1], scale it up
        rot_loss = rot_loss * 10.0  # Adjust this scaling factor as needed
        
        # Compute scale-aware weights based on motion magnitude
        with torch.no_grad():
            # Translation magnitude
            trans_magnitude = torch.norm(target_translation, dim=-1)
            trans_weight = torch.ones_like(trans_magnitude)
            trans_weight[trans_magnitude < 0.01] = 3.0  # Small motions < 1cm
            trans_weight[trans_magnitude > 0.05] = 0.5  # Large motions > 5cm
            
            # Rotation magnitude (quaternion angle)
            # Angle between identity and target quaternion
            quat_dot = target_rotation[:, 3]  # w component
            quat_angle = 2 * torch.acos(torch.clamp(quat_dot.abs(), -1, 1))
            rot_weight = torch.ones_like(quat_angle)
            rot_weight[quat_angle < 0.035] = 3.0  # Small rotations < 2°
            rot_weight[quat_angle > 0.175] = 0.5  # Large rotations > 10°
        
        # Apply weights
        weighted_trans_loss = (trans_loss * trans_weight.mean())
        weighted_rot_loss = (rot_loss * rot_weight.mean())
        
        return {
            'translation_loss': weighted_trans_loss,
            'rotation_loss': weighted_rot_loss,
            'total_loss': weighted_trans_loss + weighted_rot_loss
        }