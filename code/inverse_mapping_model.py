import torch
import torch.nn as nn
from typing import List, Optional

from normalizer import VectorNormScaler


class InverseMappingModel(nn.Module):
    """
    Inverse mapping model: kinematics (k) + wing angles (k-1) → wing angles (k).
    
    Architecture:
    - Input 1: 12D body kinematics at time k (normalized via VectorNormScaler)
    - Input 2: (n*6)D flattened wing angles at time k-1
    - Concatenate both inputs
    - Deeper FC network with ReLU activations
    - Output: 6D wing angles at time k
    """

    def __init__(
        self,
        n_samples_per_wingbeat: int,
        hidden_dims: Optional[List[int]] = None,
        dropout_rate: float = 0.1,
        activation: str = "ReLU",
    ):
        """
        Args:
            n_samples_per_wingbeat: Number of wing angle samples per wingbeat (n).
            hidden_dims: List of hidden layer dimensions. Defaults to [256, 128, 64].
            dropout_rate: Dropout probability for regularization.
            activation: Activation function to use in hidden layers (ReLU, Tanh, ELU).
        """
        super().__init__()
        
        self.n_samples_per_wingbeat = n_samples_per_wingbeat
        self.kinematics_dim = 12
        self.wing_angles_dim = 6
        self.flattened_prev_wings_dim = n_samples_per_wingbeat * self.wing_angles_dim
        
        # Input dimensions after concatenation
        self.input_dim = self.kinematics_dim + self.flattened_prev_wings_dim
        
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.activation = activation
        
        # Build fully connected layers
        layers = []
        prev_dim = self.input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == "ReLU":
                layers.append(nn.ReLU(inplace=True))
            elif activation == "Tanh":
                layers.append(nn.Tanh())
            elif activation == "ELU":
                layers.append(nn.ELU(inplace=True))
            elif activation == "LeakyReLU":
                layers.append(nn.LeakyReLU(inplace=True))
            elif activation == "GELU":
                layers.append(nn.GELU())
            else:
                raise ValueError(f"Unsupported activation: {activation}")
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, self.flattened_prev_wings_dim))
        
        self.fc_network = nn.Sequential(*layers)
        
        # Normalizer for kinematics (will be fitted during training)
        self.kinematics_normalizer = VectorNormScaler(global_normalizer=True)
        self._normalizer_fitted = False
    
    def fit_normalizer(self, kinematics_data: torch.Tensor | List[torch.Tensor]) -> None:
        """
        Fit the VectorNormScaler on kinematic data.
        
        Args:
            kinematics_data: Either a tensor of shape [N, 12] or list of [Ki, 12] tensors.
        """
        self.kinematics_normalizer.fit(kinematics_data)
        self._normalizer_fitted = True
    
    def forward(
        self,
        kinematics_k: torch.Tensor,
        wing_angles_k_minus_1_flattened: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for inverse mapping.
        
        Args:
            kinematics_k: Shape [batch_size, 12] or [batch_size, 1, 12].
                          Kinematic vector at time k.
            wing_angles_k_minus_1_flattened: Shape [batch_size, n*6].
                                             Flattened wing angles at time k-1.
        
        Returns:
            wing_angles_k: Shape [batch_size, 6]. Predicted wing angles at time k.
        """
        # Handle potential extra dimension from sequence batching
        if kinematics_k.ndim == 3:
            kinematics_k = kinematics_k.squeeze(1)
        if kinematics_k.ndim != 2 or kinematics_k.shape[-1] != self.kinematics_dim:
            raise ValueError(
                f"Expected kinematics_k shape [B, 12] or [B, 1, 12], got {kinematics_k.shape}"
            )
        if wing_angles_k_minus_1_flattened.ndim != 2:
            raise ValueError(
                f"Expected wing_angles_k_minus_1_flattened shape [B, {self.flattened_prev_wings_dim}], "
                f"got {wing_angles_k_minus_1_flattened.shape}"
            )
        if wing_angles_k_minus_1_flattened.shape[-1] != self.flattened_prev_wings_dim:
            raise ValueError(
                f"Expected wing angles flattened dim {self.flattened_prev_wings_dim}, "
                f"got {wing_angles_k_minus_1_flattened.shape[-1]}"
            )
        
        # Normalize kinematics
        if not self._normalizer_fitted:
            kinematics_normalized = kinematics_k
        else:
            kinematics_normalized = self.kinematics_normalizer.transform(kinematics_k)
        
        # Concatenate inputs
        combined_input = torch.cat([kinematics_normalized, wing_angles_k_minus_1_flattened], dim=-1)
        
        # Pass through FC network
        wing_angles_k = self.fc_network(combined_input)
        
        return wing_angles_k
    
    def get_normalizer(self) -> VectorNormScaler:
        """Return the kinematics normalizer for external use (e.g., saving/loading)."""
        return self.kinematics_normalizer
    
    def save_checkpoint(self, path: str, include_normalizer: bool = True) -> None:
        """
        Save model weights and optionally normalizer state.
        
        Args:
            path: File path to save checkpoint.
            include_normalizer: Whether to save normalizer state.
        """
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": {
                "n_samples_per_wingbeat": self.n_samples_per_wingbeat,
                "hidden_dims": self.hidden_dims,
                "dropout_rate": self.dropout_rate,
                "activation": self.activation,
            },
        }
        
        if include_normalizer and self._normalizer_fitted:
            checkpoint["normalizer_state"] = {
                "scale_factors": self.kinematics_normalizer.scale_factors,
            }
        
        torch.save(checkpoint, path)
    
    @classmethod
    def load_checkpoint(cls, path: str, device: str = "cpu") -> "InverseMappingModel":
        """
        Load model from checkpoint.
        
        Args:
            path: File path to checkpoint.
            device: Device to load model on.
        
        Returns:
            Loaded InverseMappingModel instance.
        """
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint["config"]
        
        model = cls(
            n_samples_per_wingbeat=config["n_samples_per_wingbeat"],
            hidden_dims=config["hidden_dims"],
            dropout_rate=config["dropout_rate"],
            activation=config.get("activation", "ReLU"),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        
        if "normalizer_state" in checkpoint:
            model.kinematics_normalizer.scale_factors = checkpoint["normalizer_state"]["scale_factors"].to(device)
            model._normalizer_fitted = True
        
        return model.to(device)


if __name__ == "__main__":
    # Quick test of model architecture
    n_samples = 16
    batch_size = 8
    
    model = InverseMappingModel(
        n_samples_per_wingbeat=n_samples,
        hidden_dims=[256, 128, 64],
        dropout_rate=0.1,
    )
    
    # Create dummy inputs
    kinematics = torch.randn(batch_size, 12)
    wing_angles_prev = torch.randn(batch_size, n_samples * 6)
    
    # Fit normalizer
    model.fit_normalizer(kinematics)
    
    # Forward pass
    output = model(kinematics, wing_angles_prev)
    
    print(f"Input kinematics shape: {kinematics.shape}")
    print(f"Input wing angles (prev, flattened) shape: {wing_angles_prev.shape}")
    print(f"Output wing angles shape: {output.shape}")
    assert output.shape == (batch_size, n_samples * 6), f"Expected output shape {(batch_size, n_samples * 6)}, got {output.shape}"
    print("✓ Model test passed!")
