import math
from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple, Tuple

import torch
from ..choices import *
from ..config_base import BaseConfig
from torch import nn
from torch.nn import init

from .blocks import *
from .nn import timestep_embedding
from .unet import *


class LatentNetType(Enum):
    none = 'none'
    # injecting inputs into the hidden layers
    skip = 'skip'


class LatentNetReturn(NamedTuple):
    pred: torch.Tensor = None


@dataclass
class MLPSkipNetConfig(BaseConfig):
    """
    Configuration class for MLPSkipNet - a Multi-Layer Perceptron with skip connections.

    This is the default MLP architecture used for the latent diffusion probabilistic model (DPM)
    in the Diffusion Autoencoders paper. It processes latent vectors conditioned on timesteps.

    The network allows for skip connections from input to hidden layers, which helps with
    gradient flow and feature preservation through the network.
    """
    num_channels: int  # Dimensionality of input and output features
    skip_layers: Tuple[int]  # Indices of layers that will receive skip connections from input
    num_hid_channels: int  # Number of hidden channels in intermediate layers
    num_layers: int  # Total number of layers in the network
    num_time_emb_channels: int = 64  # Dimensionality of time embedding before projection
    activation: Activation = Activation.silu  # Activation function used throughout the network
    use_norm: bool = True  # Whether to use layer normalization
    condition_bias: float = 1  # Bias factor for conditional scaling
    dropout: float = 0  # Dropout probability for regularization
    last_act: Activation = Activation.none  # Activation function applied after all layers
    num_time_layers: int = 2  # Number of layers in time embedding MLP
    time_last_act: bool = False  # Whether to apply activation after the last time embedding layer

    def make_model(self):
        return MLPSkipNet(self)  # Factory method to create network from this configuration




class MLPSkipNet(nn.Module):
    """
    Multi-Layer Perceptron with skip connections from input to hidden layers.

    This is the default MLP architecture used for the latent diffusion model in the
    Diffusion Autoencoders paper. It processes input latent vectors conditioned on
    diffusion timesteps to predict noise or denoised data.

    The network has:
    1. A time embedding network to process timestep encodings
    2. A main network with optional skip connections from input to hidden layers
    3. Layer normalization and dropout for regularization
    4. Conditional modulation based on time embeddings

    Skip connections allow the network to preserve input information throughout the
    network, improving gradient flow and performance.
    """

    def __init__(self, conf: MLPSkipNetConfig):
        super().__init__()
        self.conf = conf  # Store configuration

        # Create time embedding layers
        layers = []
        for i in range(conf.num_time_layers):
            if i == 0:
                # First layer: from embedding dimension to channel dimension
                a = conf.num_time_emb_channels  # Input: time embedding channels
                b = conf.num_channels  # Output: model channels
            else:
                # Subsequent layers: from channel dimension to channel dimension
                a = conf.num_channels  # Input: previous layer output
                b = conf.num_channels  # Output: model channels
            layers.append(nn.Linear(a, b))  # Linear projection
            # Add activation except for last layer (unless time_last_act is True)
            if i < conf.num_time_layers - 1 or conf.time_last_act:
                layers.append(conf.activation.get_act())  # Add activation function
        self.time_embed = nn.Sequential(*layers)  # Time embedding network

        # Create main network layers
        self.layers = nn.ModuleList([])
        for i in range(conf.num_layers):
            if i == 0:
                # First layer: input to hidden
                act = conf.activation  # Use configured activation
                norm = conf.use_norm  # Use normalization if specified
                cond = True  # Apply conditioning
                a, b = conf.num_channels, conf.num_hid_channels  # Input channels to hidden channels
                dropout = conf.dropout  # Apply specified dropout
            elif i == conf.num_layers - 1:
                # Last layer: hidden to output
                act = Activation.none  # No activation (applied separately after)
                norm = False  # No normalization
                cond = False  # No conditioning
                a, b = conf.num_hid_channels, conf.num_channels  # Hidden channels to output channels
                dropout = 0  # No dropout
            else:
                # Intermediate layers: hidden to hidden
                act = conf.activation  # Use configured activation
                norm = conf.use_norm  # Use normalization if specified
                cond = True  # Apply conditioning
                a, b = conf.num_hid_channels, conf.num_hid_channels  # Hidden to hidden
                dropout = conf.dropout  # Apply specified dropout

            # Increase input dimension for layers with skip connections
            if i in conf.skip_layers:
                a += conf.num_channels  # Add input channels dimension for skip connection

            # Create layer with appropriate settings
            self.layers.append(
                MLPLNAct(
                    a,  # Input dimension (increased if skip connection)
                    b,  # Output dimension
                    norm=norm,  # Whether to use layer normalization
                    activation=act,  # Activation function
                    cond_channels=conf.num_channels,  # Conditioning channels
                    use_cond=cond,  # Whether to apply conditioning
                    condition_bias=conf.condition_bias,  # Scaling factor for conditioning
                    dropout=dropout,  # Dropout probability
                ))

        # Final activation function applied after all layers
        self.last_act = conf.last_act.get_act()  # Get PyTorch activation module from enum


    def forward(self, x, t, **kwargs):
        """
        Forward pass through the network.

        Args:
            x: Input tensor [batch_size, num_channels]
            t: Timestep tensor [batch_size]
            **kwargs: Additional arguments (unused)

        Returns:
            LatentNetReturn: Wrapper containing processed output tensor
        """
        # Convert integer timesteps to embeddings and process through time network
        t = timestep_embedding(t, self.conf.num_time_emb_channels)  # Create sinusoidal time embedding
        cond = self.time_embed(t)  # Process time embedding through MLP

        h = x  # Initialize hidden state with input
        for i in range(len(self.layers)):
            if i in self.conf.skip_layers:
                # Add skip connection from input at specified layers
                h = torch.cat([h, x], dim=1)  # Concatenate current hidden state with input
            h = self.layers[i].forward(x=h, cond=cond)  # Process through layer with conditioning

        h = self.last_act(h)  # Apply final activation function

        return LatentNetReturn(h)  # Return result wrapped in return object


class MLPLNAct(nn.Module):
    """
    MLP building block with Linear layer, LayerNorm, Activation, and optional conditioning.

    This module combines several operations commonly used in MLPs:
    1. Linear projection
    2. Conditional modulation (optional)
    3. Layer normalization (optional)
    4. Activation function
    5. Dropout (optional)

    When conditioning is enabled, it applies feature-wise modulation using the
    conditional input before normalization, similar to adaptive instance norm
    or film layers in other architectures.
    """

    def __init__(
            self,
            in_channels: int,  # Number of input channels
            out_channels: int,  # Number of output channels
            norm: bool,  # Whether to use layer normalization
            use_cond: bool,  # Whether to apply conditional modulation
            activation: Activation,  # Activation function to use
            cond_channels: int,  # Number of conditioning channels
            condition_bias: float = 0,  # Bias term added to conditioning scale factor
            dropout: float = 0,  # Dropout probability
    ):
        super().__init__()
        self.activation = activation  # Store activation type for weight initialization
        self.condition_bias = condition_bias  # Store bias term for conditioning
        self.use_cond = use_cond  # Whether to use conditioning

        # Main linear transformation
        self.linear = nn.Linear(in_channels, out_channels)  # Main linear projection
        self.act = activation.get_act()  # Get PyTorch activation module

        # Conditional modulation layers (if enabled)
        if self.use_cond:
            # Linear projection for conditioning signal
            self.linear_emb = nn.Linear(cond_channels, out_channels)  # Projects condition to feature dimension
            self.cond_layers = nn.Sequential(self.act, self.linear_emb)  # Apply activation before projection

        # Normalization layer (if enabled)
        if norm:
            self.norm = nn.LayerNorm(out_channels)  # Layer normalization
        else:
            self.norm = nn.Identity()  # No normalization

        # Dropout layer (if enabled)
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)  # Apply dropout with specified probability
        else:
            self.dropout = nn.Identity()  # No dropout

        # Initialize weights based on activation function
        self.init_weights()

    def init_weights(self):
        """
        Initialize weights of linear layers based on the activation function.

        Uses Kaiming initialization with parameters tailored to the specific
        activation function for better training dynamics.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.activation == Activation.relu:
                    # Kaiming initialization for ReLU
                    init.kaiming_normal_(module.weight,
                                         a=0,  # Parameter for ReLU
                                         nonlinearity='relu')
                elif self.activation == Activation.lrelu:
                    # Kaiming initialization for LeakyReLU
                    init.kaiming_normal_(module.weight,
                                         a=0.2,  # Negative slope for LeakyReLU
                                         nonlinearity='leaky_relu')
                elif self.activation == Activation.silu:
                    # SiLU (Swish) uses same initialization as ReLU
                    init.kaiming_normal_(module.weight,
                                         a=0,
                                         nonlinearity='relu')
                else:
                    # For other activations, use PyTorch default initialization
                    pass

    def forward(self, x, cond=None):
        """
        Forward pass applying linear projection, conditional modulation, normalization, and activation.

        Args:
            x: Input tensor [batch_size, in_channels]
            cond: Conditioning tensor [batch_size, cond_channels] or None

        Returns:
            Processed tensor [batch_size, out_channels]
        """
        x = self.linear(x)  # Apply main linear transformation

        if self.use_cond:
            # Process conditioning signal if conditioning is enabled
            cond = self.cond_layers(cond)  # Apply activation and projection to condition
            cond = (cond, None)  # Format as (scale, shift), with shift=None

            # Apply feature-wise modulation using scale from condition
            # condition_bias prevents complete zeroing of features
            x = x * (self.condition_bias + cond[0])  # Apply scaling

            # Apply optional shift if provided
            if cond[1] is not None:
                x = x + cond[1]  # Apply shifting

            # Apply normalization after modulation
            x = self.norm(x)
        else:
            # Without conditioning, just apply normalization
            x = self.norm(x)

        # Apply activation and dropout
        x = self.act(x)  # Apply activation function
        x = self.dropout(x)  # Apply dropout (if enabled)

        return x