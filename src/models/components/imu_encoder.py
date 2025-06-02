"""
IMU Encoder for processing inertial measurement unit data.
Processes accelerometer and gyroscope data for VIO systems.
"""

import torch
import torch.nn as nn
from typing import Optional


class IMUEncoder(nn.Module):
    """
    Encoder for IMU data (accelerometer + gyroscope).
    Processes 6-DOF IMU measurements into feature representations.
    """
    
    def __init__(
        self,
        input_dim: int = 6,  # 3 accel + 3 gyro
        output_dim: int = 256,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_temporal_conv: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_temporal_conv = use_temporal_conv
        
        # Input normalization
        self.input_norm = nn.LayerNorm(input_dim)
        
        # Temporal convolution for local patterns (optional)
        if use_temporal_conv:
            self.temporal_conv = nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU()
            )
            conv_output_dim = hidden_dim
        else:
            conv_output_dim = input_dim
        
        # Feature extraction layers
        layers = []
        current_dim = conv_output_dim
        
        for i in range(num_layers):
            if i == num_layers - 1:
                # Last layer outputs the desired dimension
                next_dim = output_dim
            else:
                next_dim = hidden_dim
            
            layers.extend([
                nn.Linear(current_dim, next_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            current_dim = next_dim
        
        # Remove last dropout
        if layers:
            layers = layers[:-1]
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Optional residual connection
        if input_dim == output_dim:
            self.residual = True
        else:
            self.residual = False
            if input_dim < output_dim:
                self.residual_proj = nn.Linear(input_dim, output_dim)
    
    def forward(self, imu_data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for IMU encoding.
        
        Args:
            imu_data: [batch_size, sequence_length, 6] 
                     First 3 dims: accelerometer (ax, ay, az)
                     Last 3 dims: gyroscope (gx, gy, gz)
        
        Returns:
            encoded_features: [batch_size, sequence_length, output_dim]
        """
        batch_size, seq_len, _ = imu_data.shape
        
        # Normalize input
        normalized_imu = self.input_norm(imu_data)
        
        # Store for potential residual connection
        residual_input = normalized_imu
        
        # Apply temporal convolution if enabled
        if self.use_temporal_conv:
            # Transpose for conv1d: [B, seq_len, features] -> [B, features, seq_len]
            conv_input = normalized_imu.transpose(1, 2)
            conv_output = self.temporal_conv(conv_input)
            # Transpose back: [B, features, seq_len] -> [B, seq_len, features]
            features = conv_output.transpose(1, 2)
        else:
            features = normalized_imu
        
        # Apply feature extraction
        encoded_features = self.feature_extractor(features)
        
        # Apply residual connection if possible
        if self.residual:
            encoded_features = encoded_features + residual_input
        elif hasattr(self, 'residual_proj'):
            encoded_features = encoded_features + self.residual_proj(residual_input)
        
        return encoded_features


class AdvancedIMUEncoder(IMUEncoder):
    """
    Advanced IMU encoder with gravity compensation and motion classification.
    """
    
    def __init__(
        self,
        input_dim: int = 6,
        output_dim: int = 256,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        gravity_compensation: bool = True,
        motion_classification: bool = True
    ):
        super().__init__(input_dim, output_dim, hidden_dim, num_layers, dropout)
        
        self.gravity_compensation = gravity_compensation
        self.motion_classification = motion_classification
        
        # Gravity estimation network
        if gravity_compensation:
            self.gravity_estimator = nn.Sequential(
                nn.Linear(3, 32),  # Accelerometer input
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 3)   # Estimated gravity vector
            )
        
        # Motion pattern classifier
        if motion_classification:
            self.motion_classifier = nn.Sequential(
                nn.Linear(output_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 4)  # Static, Walking, Running, Rapid motion
            )
    
    def forward(self, imu_data: torch.Tensor) -> dict:
        """
        Advanced forward pass with gravity compensation and motion classification.
        
        Args:
            imu_data: [batch_size, sequence_length, 6]
        
        Returns:
            Dictionary containing:
            - features: Encoded IMU features
            - gravity_estimate: Estimated gravity vector (if enabled)
            - motion_class: Motion classification probabilities (if enabled)
        """
        batch_size, seq_len, _ = imu_data.shape
        
        # Split accelerometer and gyroscope data
        accel_data = imu_data[:, :, :3]  # [B, seq_len, 3]
        gyro_data = imu_data[:, :, 3:]   # [B, seq_len, 3]
        
        processed_imu = imu_data
        gravity_estimate = None
        
        # Gravity compensation
        if self.gravity_compensation:
            # Estimate gravity from accelerometer data
            # Use mean over sequence for stability
            mean_accel = accel_data.mean(dim=1)  # [B, 3]
            gravity_estimate = self.gravity_estimator(mean_accel)  # [B, 3]
            
            # Subtract gravity from accelerometer readings
            gravity_expanded = gravity_estimate.unsqueeze(1).expand(-1, seq_len, -1)
            compensated_accel = accel_data - gravity_expanded
            
            # Recombine with gyroscope data
            processed_imu = torch.cat([compensated_accel, gyro_data], dim=-1)
        
        # Get base encoded features
        encoded_features = super().forward(processed_imu)
        
        # Motion classification
        motion_class = None
        if self.motion_classification:
            # Use mean features over sequence for classification
            mean_features = encoded_features.mean(dim=1)  # [B, output_dim]
            motion_class = self.motion_classifier(mean_features)  # [B, 4]
            motion_class = torch.softmax(motion_class, dim=-1)
        
        result = {
            'features': encoded_features,
            'raw_features': encoded_features  # For backward compatibility
        }
        
        if gravity_estimate is not None:
            result['gravity_estimate'] = gravity_estimate
        
        if motion_class is not None:
            result['motion_class'] = motion_class
        
        # Return just features if no advanced features are used (backward compatibility)
        if not self.gravity_compensation and not self.motion_classification:
            return encoded_features
        
        return result


# Factory function for easy instantiation
def create_imu_encoder(config: dict) -> IMUEncoder:
    """
    Factory function to create IMU encoder from configuration.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Configured IMU encoder instance
    """
    encoder_type = config.get('type', 'basic')
    
    if encoder_type == 'advanced':
        return AdvancedIMUEncoder(**config)
    else:
        return IMUEncoder(**config)