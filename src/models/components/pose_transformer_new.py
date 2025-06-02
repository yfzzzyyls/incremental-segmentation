"""
Updated PoseTransformer for AR/VR optimized VIO models.
Compatible with the new multi-scale and multi-head architectures.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class PoseTransformer(nn.Module):
    """
    Transformer encoder for pose sequence processing.
    Updated to be compatible with new AR/VR models.
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        max_sequence_length: int = 15,
        dropout: float = 0.1,
        use_positional_encoding: bool = True,
        use_causal_mask: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_sequence_length = max_sequence_length
        self.use_positional_encoding = use_positional_encoding
        self.use_causal_mask = use_causal_mask
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        if use_positional_encoding:
            self.positional_encoding = PositionalEncoding(hidden_dim, max_sequence_length)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='relu'
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Output normalization
        self.output_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, features: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the transformer.
        
        Args:
            features: [batch_size, sequence_length, input_dim]
            mask: Optional attention mask
        
        Returns:
            output: [batch_size, sequence_length, hidden_dim]
        """
        batch_size, seq_len, _ = features.shape
        
        # Project input features
        projected_features = self.input_projection(features)
        
        # Add positional encoding
        if self.use_positional_encoding:
            projected_features = self.positional_encoding(projected_features)
        
        # Generate causal mask if requested
        if self.use_causal_mask and mask is None:
            mask = self._generate_causal_mask(seq_len, features.device)
        
        # Apply transformer
        transformer_output = self.transformer_encoder(
            projected_features,
            mask=mask,
            is_causal=self.use_causal_mask and mask is None
        )
        
        # Apply output normalization
        output = self.output_norm(transformer_output)
        
        return output
    
    def _generate_causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        """Generate causal mask for self-attention."""
        mask = torch.triu(
            torch.full((size, size), float('-inf'), device=device),
            diagonal=1
        )
        return mask


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer inputs.
    """
    
    def __init__(self, hidden_dim: int, max_length: int = 5000):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Create positional encoding matrix
        pe = torch.zeros(max_length, hidden_dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2).float() * 
            (-math.log(10000.0) / hidden_dim)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_length, hidden_dim]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: [batch_size, sequence_length, hidden_dim]
        
        Returns:
            x with positional encoding added
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x


class AttentionVisualizationTransformer(PoseTransformer):
    """
    Transformer with attention visualization capabilities for debugging.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention_weights = None
    
    def forward(self, features: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with attention weight extraction."""
        # Store attention weights from each layer
        self.attention_weights = []
        
        batch_size, seq_len, _ = features.shape
        projected_features = self.input_projection(features)
        
        if self.use_positional_encoding:
            projected_features = self.positional_encoding(projected_features)
        
        if self.use_causal_mask and mask is None:
            mask = self._generate_causal_mask(seq_len, features.device)
        
        # Manual forward through transformer layers to extract attention
        x = projected_features
        for layer in self.transformer_encoder.layers:
            # Extract attention weights
            attention_output, attention_weights = layer.self_attn(
                x, x, x, attn_mask=mask, need_weights=True
            )
            self.attention_weights.append(attention_weights.detach())
            
            # Continue with rest of transformer layer
            x = layer.norm1(x + layer.dropout1(attention_output))
            ff_output = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
            x = layer.norm2(x + layer.dropout2(ff_output))
        
        output = self.output_norm(x)
        return output
    
    def get_attention_weights(self):
        """Get attention weights from last forward pass."""
        return self.attention_weights


# For backward compatibility with existing code
class PoseTransformerLegacy(nn.Module):
    """
    Legacy PoseTransformer interface for backward compatibility.
    """
    
    def __init__(self, input_dim=768, embedding_dim=128, num_layers=2, nhead=8, dim_feedforward=512, dropout=0.1):
        super().__init__()
        
        # Map to new interface
        self.new_transformer = PoseTransformer(
            input_dim=input_dim,
            hidden_dim=embedding_dim,
            num_layers=num_layers,
            num_heads=nhead,
            dropout=dropout
        )
        
        # Legacy output layer
        self.fc2 = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(embedding_dim, 6)
        )
    
    def forward(self, batch, gt=None):
        """Legacy forward interface."""
        visual_inertial_features, _, _ = batch
        
        # Use new transformer
        transformer_output = self.new_transformer(visual_inertial_features)
        
        # Apply legacy output layer
        output = self.fc2(transformer_output)
        
        return output