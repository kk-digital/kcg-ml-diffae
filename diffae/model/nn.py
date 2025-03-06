"""
Various utilities for neural networks.
"""

from enum import Enum
import math
from typing import Optional

import torch as th
import torch.nn as nn
import torch.utils.checkpoint

import torch.nn.functional as F


# PyTorch 1.7 has SiLU, but we support PyTorch 1.5.
class SiLU(nn.Module):
    # @th.jit.script
    def forward(self, x):
        return x * th.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(min(32, channels), channels)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings for diffusion models.

    This function converts scalar timestep values (e.g., 0, 1, ..., T) into high-dimensional
    vectors using a series of sine and cosine functions at different frequencies, similar to
    positional encodings in transformer models. These embeddings allow the network to distinguish
    between different timesteps in the diffusion process.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output embedding vectors.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2# Half the embedding dimension (for sin/cos)

    # Create a range of frequencies in log space
    # Lower indices = lower frequencies, higher indices = higher frequencies
    freqs = th.exp(-math.log(max_period) *
                   th.arange(start=0, end=half, dtype=th.float32) /
                   half).to(device=timesteps.device)

    # Multiply each timestep by each frequency
    # [N, 1] * [1, half] = [N, half]
    args = timesteps[:, None].float() * freqs[None]

    # Compute sinusoidal embedding by concatenating cos and sin components
    # This gives the model both even and odd functions to represent timesteps
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)

    # If dim is odd, pad with an extra zero
    if dim % 2:
        embedding = th.cat(
            [embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding # Return [N, dim] embedding vectors


def torch_checkpoint(func, args, flag, preserve_rng_state=False):
    """
    Conditionally apply PyTorch gradient checkpointing to save memory during training.

    Gradient checkpointing trades compute for memory by recomputing intermediate activations
    during backpropagation instead of storing them. This function provides a simple interface
    to conditionally apply checkpointing based on a flag.

    Args:
        func: The function to checkpoint
        args: Tuple of arguments to pass to the function
        flag: Whether to apply checkpointing (True) or run normally (False)
        preserve_rng_state: Whether to preserve RNG state during checkpointing

    Returns:
        The result of calling func(*args), with or without checkpointing
    """

    # torch's gradient checkpoint works with automatic mixed precision, given torch >= 1.8
    if flag:
        # Apply gradient checkpointing to save memory at the cost of computation speed
        return torch.utils.checkpoint.checkpoint(
            func, *args, preserve_rng_state=preserve_rng_state)
    else:
        # Call the function directly without checkpointing
        return func(*args)
