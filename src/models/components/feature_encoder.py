"""
Image Feature Encoder for visual-inertial odometry.
Processes image features for VIO systems.
"""

import torch
import torch.nn as nn
from typing import Optional


class ImageFeatureEncoder(nn.Module):
    """
    Encoder for image features in VIO systems.
    Can work with pre-extracted features or raw images.
    """
    
    def __init__(
        self,
        input_dim: Optional[int] = None,
        output_dim: int = 768,
        hidden_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_batch_norm: bool = True,
        input_type: str = "features"  # "features" or "images"
    ):
        """
        Args:
            input_dim: Input feature dimension (if None, will be inferred)
            output_dim: Output feature dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of processing layers
            dropout: Dropout rate
            use_batch_norm: Whether to use batch normalization
            input_type: "features" for pre-extracted features, "images" for raw images
        """
        super().__init__()
        
        self.input_type = input_type
        self.output_dim = output_dim
        
        if input_type == "images":
            # Simple CNN for raw image processing (placeholder)
            self.cnn_encoder = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((7, 7))
            )
            # Calculate CNN output dimension
            calculated_input_dim = 256 * 7 * 7  # 12544
        else:
            # For pre-extracted features
            calculated_input_dim = input_dim or 512  # Default for common feature extractors
        
        # Feature processing layers
        layers = []
        current_dim = calculated_input_dim
        
        for i in range(num_layers):
            if i == num_layers - 1:
                next_dim = output_dim
            else:
                next_dim = hidden_dim
            
            layers.append(nn.Linear(current_dim, next_dim))
            
            if i < num_layers - 1:  # No activation/norm after last layer
                if use_batch_norm:
                    layers.append(nn.BatchNorm1d(next_dim))
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            
            current_dim = next_dim
        
        self.feature_processor = nn.Sequential(*layers)
        
        # Optional layer normalization for output
        self.output_norm = nn.LayerNorm(output_dim)
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for feature encoding.
        
        Args:
            inputs: Either [B*seq_len, feature_dim] for features or 
                   [B*seq_len, C, H, W] for images
        
        Returns:
            encoded_features: [B*seq_len, output_dim]
        """
        if self.input_type == "images":
            # Process raw images through CNN
            batch_size = inputs.shape[0]
            cnn_features = self.cnn_encoder(inputs)  # [B*seq_len, 256, 7, 7]
            features = cnn_features.view(batch_size, -1)  # [B*seq_len, 256*7*7]
        else:
            # Use pre-extracted features directly
            features = inputs
        
        # Process through feature layers
        processed_features = self.feature_processor(features)
        
        # Apply output normalization
        normalized_features = self.output_norm(processed_features)
        
        return normalized_features




# Factory function for easy instantiation
def create_feature_encoder(config: dict) -> ImageFeatureEncoder:
    """
    Factory function to create feature encoder from configuration.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Configured feature encoder instance
    """
    return ImageFeatureEncoder(**config)