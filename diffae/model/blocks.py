"""
Based on the provided code context, this project implements "Diffusion Autoencoders," a generative AI model
presented in a CVPR 2022 paper. The codebase provides a complete implementation of diffusion-based generative
models with autoencoding capabilities.

The key functionalities of this project include:

1. Image Generation: Creating new images using diffusion models
2. Image Manipulation: Editing image attributes like hairstyle
3. Image Interpolation: Smoothly transitioning between images
4. Autoencoding: Converting images to meaningful latent representations and back

The architecture consists of sophisticated building blocks:
- ResBlocks: Residual blocks with conditional modulation for feature processing
- Attention mechanisms: Self-attention modules to capture long-range dependencies
- Up/downsampling layers: For changing spatial resolutions in the network
- Conditional modulation: Applying time embeddings and encoder outputs to control generation

The project provides pre-trained checkpoints for various datasets (FFHQ, Bedroom, Horse) and includes Jupyter
notebooks for different applications. It also has utilities for face alignment to prepare your own images for processing.

This implementation follows the architecture described in the paper "Diffusion Autoencoders: Toward a Meaningful
and Decodable Representation" and provides both a research foundation and practical tools for image generation
and manipulation.
"""

import math
from abc import abstractmethod
from dataclasses import dataclass
from numbers import Number
import numpy as np
import torch as th
import torch.nn.functional as F
from ..choices import *
from ..config_base import BaseConfig
from torch import nn

from .nn import (avg_pool_nd, conv_nd, linear, normalization,
                 timestep_embedding, torch_checkpoint, zero_module)


class ScaleAt(Enum):
    after_norm = 'afternorm'


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """
    @abstractmethod
    def forward(self, x, emb=None, cond=None, lateral=None):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """
    def forward(self, x, emb=None, cond=None, lateral=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb=emb, cond=cond, lateral=lateral)
            else:
                x = layer(x)
        return x


@dataclass
class ResBlockConfig(BaseConfig):
    channels: int
    emb_channels: int
    dropout: float
    out_channels: int = None
    # condition the resblock with time (and encoder's output)
    use_condition: bool = True
    # whether to use 3x3 conv for skip path when the channels aren't matched
    use_conv: bool = False
    # dimension of conv (always 2 = 2d)
    dims: int = 2
    # gradient checkpoint
    use_checkpoint: bool = False
    up: bool = False
    down: bool = False
    # whether to condition with both time & encoder's output
    two_cond: bool = False
    # number of encoders' output channels
    cond_emb_channels: int = None
    # suggest: False
    has_lateral: bool = False
    lateral_channels: int = None
    # whether to init the convolution with zero weights
    # this is default from BeatGANs and seems to help learning
    use_zero_module: bool = True

    def __post_init__(self):
        self.out_channels = self.out_channels or self.channels
        self.cond_emb_channels = self.cond_emb_channels or self.emb_channels

    def make_model(self):
        return ResBlock(self)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    total layers:
        in_layers
        - norm
        - act
        - conv
        out_layers
        - norm
        - (modulation)
        - act
        - conv
    """
    def __init__(self, conf: ResBlockConfig):
        super().__init__()
        self.conf = conf

        #############################
        # IN LAYERS
        #############################
        assert conf.lateral_channels is None  # Verify no lateral channels are specified (handled elsewhere)
        layers = [
            normalization(conf.channels),  # Group normalization on input channels
            nn.SiLU(),  # SiLU activation function (Swish)
            conv_nd(conf.dims, conf.channels, conf.out_channels, 3, padding=1)  # 3x3 conv with padding
        ]
        self.in_layers = nn.Sequential(*layers)  # Create sequential module from layers

        self.updown = conf.up or conf.down  # Flag indicating if up/downsampling is used

        if conf.up:  # Create upsampling modules if needed
            self.h_upd = Upsample(conf.channels, False, conf.dims)  # Upsample for features
            self.x_upd = Upsample(conf.channels, False, conf.dims)  # Upsample for skip connection
        elif conf.down:  # Create downsampling modules if needed
            self.h_upd = Downsample(conf.channels, False, conf.dims)  # Downsample for features
            self.x_upd = Downsample(conf.channels, False, conf.dims)  # Downsample for skip connection
        else:
            self.h_upd = self.x_upd = nn.Identity()  # No up/downsampling needed

        #############################
        # OUT LAYERS CONDITIONS
        #############################
        if conf.use_condition:  # Only create conditioning layers if using conditions
            # Timestep embedding projection (outputs scale and shift for modulation)
            self.emb_layers = nn.Sequential(
                nn.SiLU(),  # Activation
                linear(conf.emb_channels, 2 * conf.out_channels),  # Projects to twice output channels for scale+shift
            )

            if conf.two_cond:  # Optional second condition (e.g., from encoder)
                self.cond_emb_layers = nn.Sequential(
                    nn.SiLU(),  # Activation
                    linear(conf.cond_emb_channels, conf.out_channels),  # Projects to output channels for modulation
                )
            #############################
            # OUT LAYERS (ignored when there is no condition)
            #############################
            # Create final convolution
            conv = conv_nd(conf.dims,
                           conf.out_channels,
                           conf.out_channels,
                           3,
                           padding=1)  # 3x3 conv with padding
            if conf.use_zero_module:
                # Initialize conv with zero weights to help training stability
                conv = zero_module(conv)  # Zero-initialize weights for stable training

            # Create output processing layers
            layers = []
            layers += [
                normalization(conf.out_channels),  # Group normalization
                nn.SiLU(),  # Activation
                nn.Dropout(p=conf.dropout),  # Dropout for regularization
                conv,  # Final convolution layer
            ]
            self.out_layers = nn.Sequential(*layers)  # Create sequential module

        #############################
        # SKIP LAYERS
        #############################
        if conf.out_channels == conf.channels:
            # If input and output channels match, use identity skip connection
            self.skip_connection = nn.Identity()
        else:
            # If channels don't match, need projection
            if conf.use_conv:
                kernel_size = 3  # Use 3x3 convolution for skip connection
                padding = 1
            else:
                kernel_size = 1  # Use 1x1 convolution for skip connection
                padding = 0

            # Create skip connection convolution
            self.skip_connection = conv_nd(conf.dims,
                                           conf.channels,
                                           conf.out_channels,
                                           kernel_size,
                                           padding=padding)  # Projection conv for skip connection

    def forward(self, x, emb=None, cond=None, lateral=None):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        Args:
            x: input
            lateral: lateral connection from the encoder
        """
        return torch_checkpoint(self._forward, (x, emb, cond, lateral),
                                self.conf.use_checkpoint)

    def _forward(
        self,
        x,
        emb=None,
        cond=None,
        lateral=None,
    ):
        """
        Args:
            lateral: required if "has_lateral" and non-gated, with gated, it can be supplied optionally    
        """
        if self.conf.has_lateral:
            # If lateral connections are enabled, concatenate lateral features with input
            assert lateral is not None # Ensure lateral input is provided when required
            x = th.cat([x, lateral], dim=1) # Concatenate along channel dimension

        if self.updown:
            # Handle upsampling or downsampling case
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]  # Split normalization+activation from conv
            h = in_rest(x)  # Apply norm and activation
            h = self.h_upd(h)  # Up/downsample the normalized features
            x = self.x_upd(x)  # Up/downsample the skip connection path
            h = in_conv(h)  # Apply convolution after sampling
        else:
            # Standard case without up/downsampling
            h = self.in_layers(x)  # Apply all input layers at once

        if self.conf.use_condition:
            # Process conditional information if enabled
            # Time embedding processing
            if emb is not None:
                emb_out = self.emb_layers(emb).type(h.dtype)  # Process time embedding to get scale and shift
            else:
                emb_out = None  # No time conditional information

            if self.conf.two_cond:
                # Handle second condition (encoder output) if two_cond is enabled
                if cond is None:
                    cond_out = None  # No encoder conditional information
                else:
                    cond_out = self.cond_emb_layers(cond).type(h.dtype)  # Process encoder conditional

                # Ensure conditional has proper dimensions for broadcasting
                if cond_out is not None:
                    while len(cond_out.shape) < len(h.shape):
                        cond_out = cond_out[..., None]  # Add spatial dimensions for broadcasting
            else:
                cond_out = None  # Not using second conditional

            # Apply conditional modulation to features
            h = apply_conditions(
                h=h,  # Features to modulate
                emb=emb_out,  # Time embedding modulation
                cond=cond_out,  # Optional encoder modulation
                layers=self.out_layers,  # Layers to apply after modulation
                scale_bias=1,  # Scaling factor for modulation
                in_channels=self.conf.out_channels,  # Number of channels
                up_down_layer=None,  # No additional up/down sampling
            )

        # Apply residual connection and return
        return self.skip_connection(x) + h  # Add skip connection to modulated features

def apply_conditions(
    h,
    emb=None,
    cond=None,
    layers: nn.Sequential = None,
    scale_bias: float = 1,
    in_channels: int = 512,
    up_down_layer: nn.Module = None,
):
    """
    Apply conditioning signals to feature maps using adaptive modulation.

    This function implements a form of feature modulation similar to adaptive instance
    normalization (AdaIN) or modulated convolution. It allows the diffusion model to
    incorporate both timestep information and semantic/style information from the encoder
    into the generation process.

    The modulation follows this general pattern:
    1. Apply normalization to features
    2. Apply scale and shift modulation from conditioning signals
    3. Apply remaining layers (activation, dropout, conv)

    This approach allows the model to control the generation process based on both:
    - Timestep information (how noisy the current diffusion step is)
    - Semantic information (what content should be generated)

    Args:
        h: Feature maps to be conditioned [batch_size, channels, height, width]
        emb: Time conditional embedding (ready to scale + shift)
        cond: Encoder's conditional embedding (ready to scale + shift)
        layers: Sequential module containing normalization, activation, and conv layers
        scale_bias: Base scale factor or list of factors to prevent zero scaling
        in_channels: Number of channels in the feature map
        up_down_layer: Optional layer for upsampling or downsampling before final conv

    Returns:
        Modulated feature maps
    """

    # Check if we have both time embedding and encoder conditional
    two_cond = emb is not None and cond is not None

    if emb is not None:
        # Add spatial dimensions to time embedding for proper broadcasting
        # This converts [B, C] to [B, C, 1, 1] for spatial feature maps
        # adjusting shapes
        while len(emb.shape) < len(h.shape):
            emb = emb[..., None]  # Add dimensions at the end (for spatial dims)

    if two_cond:
        # Similarly prepare encoder conditional for proper broadcasting
        # adjusting shapes
        while len(cond.shape) < len(h.shape):
            cond = cond[..., None]
        # Add dimensions for spatial broadcasting
        # Create list with both conditionals, time embedding first
        scale_shifts = [emb, cond]
    else:
        # "cond" is not used with single cond mode
        # Only using time embedding as conditional
        scale_shifts = [emb]

    # Parse each conditional to extract scale and shift components
    for i, each in enumerate(scale_shifts):
        if each is None:
            # Handle case where conditional is not provided
            a = None # No scale
            b = None # No shift
        else:
            if each.shape[1] == in_channels * 2:
                # If conditional has twice the channels, split into scale and shift
                # This implements both multiplicative and additive modulation
                a, b = th.chunk(each, 2, dim=1)  # First half is scale, second is shift
            else:
                # If only same number of channels, use as scale only (multiplicative)
                a = each # Use as scale
                b = None # No shift
        scale_shifts[i] = (a, b) # Store as tuple of (scale, shift)

    # Handle scale bias parameter (controls strength of conditioning)
    if isinstance(scale_bias, Number):
        # If scalar, use same bias for all conditionals
        biases = [scale_bias] * len(scale_shifts) # Create list with same bias value
    else:
        # If already a list, use directly
        biases = scale_bias # Use provided list of biases

    # Split the layers to apply conditionals between normalization and activation
    # This follows the pattern: norm → modulation → activation → conv
    # First layer is normalization
    pre_layers, post_layers = layers[0], layers[1:]

    # Further split post layers to apply up/down sampling before final conv
    # Last two layers are typically dropout and conv
    mid_layers, post_layers = post_layers[:-2], post_layers[-2:]

    # Apply normalization first (typically GroupNorm)
    h = pre_layers(h)

    # Apply scale and shift for each conditional
    for i, (scale, shift) in enumerate(scale_shifts):
        # Skip if scale is None (no conditional provided)
        # if scale is None, it indicates that the condition is not provided
        if scale is not None:
            # Apply scale modulation (adding bias prevents zeroing out features)
            # bias + scale ensures the modulation doesn't completely shut off features
            h = h * (biases[i] + scale)
            # Apply shift if available (additive modulation)
            if shift is not None:
                h = h + shift # Additive shift

    # Apply activation and other middle layers (typically SiLU activation)
    h = mid_layers(h)

    # Apply up/downsampling if provided (just before final conv)
    # This allows changing resolution at this specific point in the network
    if up_down_layer is not None:
        h = up_down_layer(h)

    # Apply final layers (typically dropout and conv)
    h = post_layers(h)
    return h   # Return modulated features


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """
    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels  # Number of input channels
        self.out_channels = out_channels or channels  # Output channels (same as input if not specified)
        self.use_conv = use_conv  # Whether to apply convolution after upsampling
        self.dims = dims  # Dimensionality of the input (1D, 2D, or 3D)

        # Create optional convolution layer (3x3 conv with padding)
        if use_conv:
            self.conv = conv_nd(dims,
                                self.channels,
                                self.out_channels,
                                3,  # 3x3 convolution
                                padding=1)  # Same padding to preserve spatial size

    def forward(self, x):
        # Verify input has expected number of channels
        assert x.shape[1] == self.channels

        # Handle 3D case specially (only upsample H,W dimensions, not D)
        if self.dims == 3:
            x = F.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2),
                              mode="nearest")  # Keep depth dimension, double height and width
        else:
            # For 1D and 2D, uniformly scale all spatial dimensions by 2x
            x = F.interpolate(x, scale_factor=2, mode="nearest")  # Simple 2x nearest neighbor upsampling

        # Apply optional convolution for smoothing/feature transformation
        if self.use_conv:
            x = self.conv(x)

        return x  # Return upsampled (and optionally convolved) features



class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """
    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels  # Number of input channels
        self.out_channels = out_channels or channels  # Output channels (same as input if not specified)
        self.use_conv = use_conv  # Whether to use strided conv or avg pooling
        self.dims = dims  # Dimensionality of the input (1D, 2D, or 3D)

        # Configure stride based on dimensionality
        stride = 2 if dims != 3 else (1, 2, 2)  # For 3D, only downsample H,W, not D

        if use_conv:
            # Downsample using strided convolution (better preserves information)
            self.op = conv_nd(dims,
                              self.channels,
                              self.out_channels,
                              3,  # 3x3 convolution
                              stride=stride,  # Stride of 2 for downsampling
                              padding=1)  # Same padding to match spatial dimensions
        else:
            # Ensure channels match when using pooling (no way to change channels)
            assert self.channels == self.out_channels
            # Use average pooling for downsampling (simpler, less parameters)
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        # Verify input has expected number of channels
        assert x.shape[1] == self.channels
        # Apply the downsampling operation (either conv or pooling)
        return self.op(x)  # Return downsampled features


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    This module implements self-attention over spatial dimensions in convolutional
    feature maps. It enables each position in the feature map to gather information
    from all other positions, creating a receptive field that spans the entire spatial
    extent. This is particularly important for diffusion models to capture long-range
    dependencies that convolutional layers struggle with.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
            self,
            channels,  # Number of input channels
            num_heads=1,  # Number of attention heads
            num_head_channels=-1,  # Channels per head (alternative way to specify heads)
            use_checkpoint=False,  # Whether to use gradient checkpointing to save memory
            use_new_attention_order=False,  # Whether to use newer attention implementation
    ):
        super().__init__()
        self.channels = channels

        # Determine number of attention heads
        if num_head_channels == -1:
            # Use explicitly specified number of heads
            self.num_heads = num_heads
        else:
            # Calculate number of heads based on channels per head
            assert (
                    channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels

        self.use_checkpoint = use_checkpoint  # Flag for gradient checkpointing

        # Group normalization before attention (improves stability)
        self.norm = normalization(channels)

        # 1x1 convolution that projects features to query, key, value vectors
        # Output has 3× channels because we compute q, k, v simultaneously
        self.qkv = conv_nd(1, channels, channels * 3, 1)

        # Choose attention implementation based on flag
        if use_new_attention_order:
            # Newer implementation: split QKV before splitting heads
            # More memory-efficient on some hardware
            self.attention = QKVAttention(self.num_heads)
        else:
            # Legacy implementation: split heads before splitting QKV
            self.attention = QKVAttention(self.num_heads)

        # Output projection with zero initialization
        # Zero initialization helps with training stability by making the
        # residual connection dominant in early training
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        """
        Apply self-attention to input feature map.

        Uses gradient checkpointing if enabled to reduce memory usage during training
        at the cost of recomputing parts of the graph during backward pass.
        """
        return torch_checkpoint(self._forward, (x,), self.use_checkpoint)

    def _forward(self, x):
        """
        Core attention computation without gradient checkpointing wrapper.
        """
        # Extract dimensions and flatten spatial dimensions
        b, c, *spatial = x.shape  # b=batch, c=channels, spatial=(height, width, ...)
        x = x.reshape(b, c, -1)  # Flatten to [B, C, H*W]

        # Apply normalization and compute query, key, value projections
        qkv = self.qkv(self.norm(x))  # [B, 3*C, H*W]

        # Apply self-attention mechanism
        h = self.attention(qkv)  # [B, C, H*W]

        # Project back to original dimension with zero-initialized weights
        h = self.proj_out(h)  # [B, C, H*W]

        # Add residual connection and restore spatial dimensions
        # Residual connections help with gradient flow during training
        return (x + h).reshape(b, c, *spatial)  # [B, C, H, W, ...]



def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.

    This helper function calculates the number of multiply-accumulate (MAC)
    operations in self-attention, which scales quadratically with sequence length.

    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape  # Extract batch, channels and spatial dimensions
    num_spatial = int(np.prod(spatial))  # Total number of spatial elements (H*W)

    # Calculate number of operations in attention
    # Two main matrix multiplications dominate the computation:
    # 1. Q·K^T for computing attention weights: O(b * n² * c)
    # 2. (Q·K^T)·V for applying attention: O(b * n² * c)
    # Where n is the sequence length (number of spatial positions)
    matmul_ops = 2 * b * (num_spatial ** 2) * c

    # Add to the model's operation counter
    model.total_ops += th.DoubleTensor([matmul_ops])



class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.

    This implements multi-head self-attention where query, key, and value vectors
    are first split from the concatenated input, then processed through the
    attention mechanism. This ordering differs from the legacy implementation
    and is often more efficient.

    Multi-head attention allows the model to jointly attend to information from
    different representation subspaces at different positions, enhancing the
    model's capability to capture various aspects of the input.
    """

    def __init__(self, n_heads):
        """
        Initialize the attention module.

        Args:
            n_heads: Number of attention heads to use
        """
        super().__init__()
        self.n_heads = n_heads  # Store number of attention heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        This method:
        1. Splits the input into query, key, and value tensors
        2. Reshapes to separate attention heads
        3. Computes scaled dot-product attention
        4. Combines heads and returns the result

        Args:
            qkv: an [N x (3 * H * C) x T] tensor of concatenated Qs, Ks, and Vs
                 where:
                 - N is batch size
                 - H is number of heads
                 - C is channels per head
                 - T is sequence length (e.g., spatial positions)

        Returns:
            an [N x (H * C) x T] tensor after applying attention
        """
        bs, width, length = qkv.shape  # Extract dimensions

        # Verify the input dimensions are compatible with the number of heads
        assert width % (3 * self.n_heads) == 0

        # Calculate channels per head
        ch = width // (3 * self.n_heads)

        # Split input into query, key, value along channel dimension
        # Unlike legacy version which split heads first, this splits QKV first
        q, k, v = qkv.chunk(3, dim=1)  # Each has shape [N, H*C, T]

        # Apply scaling factor for better numerical stability
        # Double square root is a common practice in some attention implementations
        scale = 1 / math.sqrt(math.sqrt(ch))

        # Compute attention scores between query and key
        # Using einsum for efficient batched matrix multiplication
        # Reshaping inside einsum to separate the heads
        weight = th.einsum(
            "bct,bcs->bts",  # Matrix product Q·K^T
            (q * scale).view(bs * self.n_heads, ch, length),  # Reshape Q: [B*H, C, T]
            (k * scale).view(bs * self.n_heads, ch, length),  # Reshape K: [B*H, C, T]
        )  # Result: [B*H, T, T] attention weights

        # Apply softmax to normalize attention weights to probabilities
        # Cast to float for numerical stability during softmax, then back to original dtype
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)

        # Apply attention weights to values
        # Using einsum for efficient batched matrix multiplication
        a = th.einsum(
            "bts,bcs->bct",  # Matrix product (Q·K^T)·V
            weight,  # [B*H, T, T] attention weights
            v.reshape(bs * self.n_heads, ch, length)  # Reshape V: [B*H, C, T]
        )  # Result: [B*H, C, T]

        # Reshape result to combine heads back into original format
        return a.reshape(bs, -1, length)  # [N, H*C, T]

    @staticmethod
    def count_flops(model, _x, y):
        """
        Helper method for counting the FLOPs in an attention operation.

        This is used with the 'thop' package for model profiling.
        """
        return count_flops_attn(model, _x, y)



class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads  # Number of attention heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape  # Extract dimensions
        assert width % (3 * self.n_heads) == 0  # Verify compatible dimensions
        ch = width // (3 * self.n_heads)  # Calculate channels per head

        # Split into q, k, v along channel dimension first (different from Legacy)
        q, k, v = qkv.chunk(3, dim=1)  # Split into three equal parts

        # Scale factor for numerical stability
        scale = 1 / math.sqrt(math.sqrt(ch))  # Double sqrt scaling

        # Calculate attention weights: Q·K^T
        # Reshaping to separate heads happens inside the einsum
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),  # Reshape q into [B*H, C, T]
            (k * scale).view(bs * self.n_heads, ch, length),  # Reshape k into [B*H, C, T]
        )  # Results in [B*H, T, T] attention weights

        # Apply softmax to get attention probabilities
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)

        # Apply attention weights to values: (Q·K^T)·V
        a = th.einsum("bts,bcs->bct", weight,
                      v.reshape(bs * self.n_heads, ch, length))  # [B*H, C, T]

        # Return reshaped result
        return a.reshape(bs, -1, length)  # [N, H*C, T]

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)  # Use helper function to count operations


class AttentionPool2d(nn.Module):
    """
    Attention-based pooling layer that converts 2D feature maps to a single vector.

    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py

    This module uses self-attention to intelligently aggregate spatial information
    from a feature map into a fixed-size vector representation. Unlike simple
    pooling operations like average or max pooling, attention pooling weighs
    the importance of different spatial locations adaptively, allowing the model
    to focus on the most relevant features.

    The process works by:
    1. Flattening the spatial dimensions
    2. Adding a special CLS token (computed as the mean of all positions)
    3. Adding learnable positional embeddings
    4. Applying self-attention across all positions
    5. Returning only the CLS token's features as the aggregated representation

    This is conceptually similar to the [CLS] token approach used in BERT and other
    transformer architectures for sequence classification.
    """

    def __init__(
            self,
            spacial_dim: int,  # Size of the spatial dimension (assuming square)
            embed_dim: int,  # Embedding dimension
            num_heads_channels: int,  # Channels per head
            output_dim: int = None,  # Output dimension (if different from embed_dim)
    ):
        super().__init__()
        # Create learnable positional embedding
        # Shape: [embed_dim, spacial_dim²+1] where +1 is for the CLS token
        # Initialization scale factor (1/sqrt(embed_dim)) helps with training stability
        self.positional_embedding = nn.Parameter(
            th.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5)

        # Projection layers for creating query, key, value vectors and final output
        # Using 1D convolutions with kernel size 1 (equivalent to per-token linear projections)
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)  # Projects to concatenated Q, K, V
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)  # Projects attention output

        # Calculate number of attention heads based on embedding dimension and channels per head
        self.num_heads = embed_dim // num_heads_channels

        # Initialize attention mechanism with multi-head support
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        """
        Apply attention pooling to input feature map.

        Args:
            x: Input tensor of shape [batch_size, channels, height, width]

        Returns:
            Pooled representation of shape [batch_size, output_dim]
        """
        # Extract batch size, channel dimension, and spatial dimensions
        b, c, *_spatial = x.shape

        # Reshape to [batch_size, channels, height*width] - flatten spatial dimensions
        x = x.reshape(b, c, -1)  # [B, C, H*W]

        # Create and prepend CLS token as the average of all spatial positions
        # This serves as a "global" token that can attend to all spatial positions
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # [B, C, H*W+1]

        # Add positional embeddings to incorporate spatial information
        # This helps the model understand the 2D structure despite flattening
        x = x + self.positional_embedding[None, :, :].to(x.dtype)

        # Apply QKV projection followed by multi-head self-attention
        x = self.qkv_proj(x)  # Project input to query, key, value vectors
        x = self.attention(x)  # Apply self-attention across all positions
        x = self.c_proj(x)  # Project attention output to final dimension

        # Return only the CLS token features, which now contains information
        # aggregated from all spatial positions through attention
        return x[:, :, 0]  # Shape: [batch_size, output_dim]
