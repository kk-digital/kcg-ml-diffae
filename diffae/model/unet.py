import math
from dataclasses import dataclass
from numbers import Number
from typing import NamedTuple, Tuple, Union

import numpy as np
import torch as th
from torch import nn
import torch.nn.functional as F
from ..choices import *
from ..config_base import BaseConfig
from .blocks import *

from .nn import (conv_nd, linear, normalization, timestep_embedding,
                 torch_checkpoint, zero_module)


@dataclass
class BeatGANsUNetConfig(BaseConfig):
    image_size: int = 64
    in_channels: int = 3
    # base channels, will be multiplied
    model_channels: int = 64
    # output of the unet
    # suggest: 3
    # you only need 6 if you also model the variance of the noise prediction (usually we use an analytical variance hence 3)
    out_channels: int = 3
    # how many repeating resblocks per resolution
    # the decoding side would have "one more" resblock
    # default: 2
    num_res_blocks: int = 2
    # you can also set the number of resblocks specifically for the input blocks
    # default: None = above
    num_input_res_blocks: int = None
    # number of time embed channels and style channels
    embed_channels: int = 512
    # at what resolutions you want to do self-attention of the feature maps
    # attentions generally improve performance
    # default: [16]
    # beatgans: [32, 16, 8]
    attention_resolutions: Tuple[int] = (16, )
    # number of time embed channels
    time_embed_channels: int = None
    # dropout applies to the resblocks (on feature maps)
    dropout: float = 0.1
    channel_mult: Tuple[int] = (1, 2, 4, 8)
    input_channel_mult: Tuple[int] = None
    conv_resample: bool = True
    # always 2 = 2d conv
    dims: int = 2
    # don't use this, legacy from BeatGANs
    num_classes: int = None
    use_checkpoint: bool = False
    # number of attention heads
    num_heads: int = 1
    # or specify the number of channels per attention head
    num_head_channels: int = -1
    # what's this?
    num_heads_upsample: int = -1
    # use resblock for upscale/downscale blocks (expensive)
    # default: True (BeatGANs)
    resblock_updown: bool = True
    # never tried
    use_new_attention_order: bool = False
    resnet_two_cond: bool = False
    resnet_cond_channels: int = None
    # init the decoding conv layers with zero weights, this speeds up training
    # default: True (BeattGANs)
    resnet_use_zero_module: bool = True
    # gradient checkpoint the attention operation
    attn_checkpoint: bool = False

    def make_model(self):
        return BeatGANsUNetModel(self)


class BeatGANsUNetModel(nn.Module):
    """
    U-Net architecture for diffusion models, adapted from the BeatGANs implementation.

    This model forms the backbone of the diffusion autoencoder, processing noisy images
    conditioned on timesteps to predict either the noise or the denoised image. The architecture
    follows a U-Net structure with skip connections between corresponding encoder and decoder layers.

    Key features:
    - Time embedding conditioning throughout the network
    - Residual blocks with conditional modulation
    - Self-attention at specified resolutions
    - Configurable channel multipliers at different resolutions
    - Optional class conditioning
    """
    def __init__(self, conf: BeatGANsUNetConfig):
        super().__init__()
        self.conf = conf  # Store configuration

        # Use specified number of attention heads for upsampling or default to main heads
        if conf.num_heads_upsample == -1:
            self.num_heads_upsample = conf.num_heads

        self.dtype = th.float32  # Default to float32

        # Time embedding network to process diffusion timesteps
        self.time_emb_channels = conf.time_embed_channels or conf.model_channels
        self.time_embed = nn.Sequential(
            linear(self.time_emb_channels, conf.embed_channels), # Initial projection
            nn.SiLU(),
            linear(conf.embed_channels, conf.embed_channels), # Final projection
        )

        # Optional class conditional embedding
        if conf.num_classes is not None:
            self.label_emb = nn.Embedding(conf.num_classes,
                                          conf.embed_channels)

        # Initial input channels
        ch = input_ch = int(conf.channel_mult[0] * conf.model_channels)

        # Input blocks (encoder/downsampling path)
        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(
                conv_nd(conf.dims, conf.in_channels, ch, 3, padding=1))
        ])

        # Common arguments for ResBlock configuration
        kwargs = dict(
            use_condition=True, # Apply time conditioning
            two_cond=conf.resnet_two_cond,  # Whether to use two conditions (time + encoder output)
            use_zero_module=conf.resnet_use_zero_module,  # Zero-initialize last conv in residual path
            # Style channels for the resnet block
            cond_emb_channels=conf.resnet_cond_channels,
        )

        # Track feature size for skip connections
        self._feature_size = ch

        # Initialize channel tracking for skip connections
        # input_block_chans = [ch]
        input_block_chans = [[] for _ in range(len(conf.channel_mult))]
        input_block_chans[0].append(ch)

        # Track number of blocks at each resolution
        # number of blocks at each resolution
        self.input_num_blocks = [0 for _ in range(len(conf.channel_mult))]
        self.input_num_blocks[0] = 1
        self.output_num_blocks = [0 for _ in range(len(conf.channel_mult))]

        # Current downsampling factor
        ds = 1
        resolution = conf.image_size

        # Build encoder (downsampling) path
        for level, mult in enumerate(conf.input_channel_mult
                                     or conf.channel_mult):

            # Add residual blocks at current resolution
            for _ in range(conf.num_input_res_blocks or conf.num_res_blocks):
                layers = [

                    # Residual block with conditional modulation
                    ResBlockConfig(
                        ch, # Input channels
                        conf.embed_channels, # Embedding dimension
                        conf.dropout, # Dropout rate
                        out_channels=int(mult * conf.model_channels),  # Output channels
                        dims=conf.dims, # Dimensionality (2D/3D)
                        use_checkpoint=conf.use_checkpoint,  # Whether to use gradient checkpointing
                        **kwargs,
                    ).make_model()
                ]
                ch = int(mult * conf.model_channels) # Update channel count
                if resolution in conf.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch, # Channels
                            use_checkpoint=conf.use_checkpoint  # Checkpointing
                            or conf.attn_checkpoint,
                            num_heads=conf.num_heads, # Number of attention heads
                            num_head_channels=conf.num_head_channels, # Channels per head
                            use_new_attention_order=conf.
                            use_new_attention_order, # Attention implementation
                        ))

                # Add block to network and track features
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                # input_block_chans.append(ch)
                input_block_chans[level].append(ch)
                self.input_num_blocks[level] += 1
                # print(input_block_chans)

            # Add downsampling layer if not at final resolution
            if level != len(conf.channel_mult) - 1:
                resolution //= 2 # Halve resolution
                out_ch = ch # Maintain channel count

                # Add downsampling layer (either ResBlock with downsampling or dedicated Downsample)
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlockConfig(
                            ch,
                            conf.embed_channels,
                            conf.dropout,
                            out_channels=out_ch,
                            dims=conf.dims,
                            use_checkpoint=conf.use_checkpoint,
                            down=True, # Enable downsampling in ResBlock
                            **kwargs,
                        ).make_model() if conf.
                        resblock_updown else Downsample(ch,
                                                        conf.conv_resample,
                                                        dims=conf.dims,
                                                        out_channels=out_ch)))
                ch = out_ch # Update channels
                # input_block_chans.append(ch)
                input_block_chans[level + 1].append(ch)
                self.input_num_blocks[level + 1] += 1
                ds *= 2 # Track downsampling factor
                self._feature_size += ch

        # Middle block (bottleneck)
        self.middle_block = TimestepEmbedSequential(
            ResBlockConfig(
                ch,
                conf.embed_channels,
                conf.dropout,
                dims=conf.dims,
                use_checkpoint=conf.use_checkpoint,
                **kwargs,
            ).make_model(),
            AttentionBlock(
                ch,
                use_checkpoint=conf.use_checkpoint or conf.attn_checkpoint,
                num_heads=conf.num_heads,
                num_head_channels=conf.num_head_channels,
                use_new_attention_order=conf.use_new_attention_order,
            ),
            ResBlockConfig(
                ch,
                conf.embed_channels,
                conf.dropout,
                dims=conf.dims,
                use_checkpoint=conf.use_checkpoint,
                **kwargs,
            ).make_model(),
        )
        self._feature_size += ch

        # Output blocks (decoder/upsampling path)
        self.output_blocks = nn.ModuleList([])

        # Build decoder path from bottom to top
        for level, mult in list(enumerate(conf.channel_mult))[::-1]:
            for i in range(conf.num_res_blocks + 1):
                # print(input_block_chans)
                # ich = input_block_chans.pop()
                try:

                    # Get corresponding encoder features for skip connection
                    ich = input_block_chans[level].pop()
                except IndexError:
                    # this happens only when num_res_block > num_enc_res_block
                    # we will not have enough lateral (skip) connecions for all decoder blocks
                    # Handle case with fewer encoder blocks than decoder blocks
                    ich = 0
                # print('pop:', ich)

                # Create layers for this block
                layers = [
                    ResBlockConfig(
                        # Channel dimension is sum of current + skip connection
                        # only direct channels when gated
                        channels=ch + ich,
                        emb_channels=conf.embed_channels,
                        dropout=conf.dropout,
                        out_channels=int(conf.model_channels * mult),
                        dims=conf.dims,
                        use_checkpoint=conf.use_checkpoint,
                        # lateral channels are described here when gated
                        # Enable lateral (skip) connection if available
                        has_lateral=True if ich > 0 else False,
                        lateral_channels=None,
                        **kwargs,
                    ).make_model()
                ]
                ch = int(conf.model_channels * mult) # Update channels

                # Add attention at specified resolutions
                if resolution in conf.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=conf.use_checkpoint
                            or conf.attn_checkpoint,
                            num_heads=self.num_heads_upsample,
                            num_head_channels=conf.num_head_channels,
                            use_new_attention_order=conf.
                            use_new_attention_order,
                        ))

                # Add upsampling at end of each resolution level (except top)
                if level and i == conf.num_res_blocks:
                    resolution *= 2 # Double resolution
                    out_ch = ch # Maintain channel count

                    # Add upsampling layer (either ResBlock with upsampling or dedicated Upsample)
                    layers.append(
                        ResBlockConfig(
                            ch,
                            conf.embed_channels,
                            conf.dropout,
                            out_channels=out_ch,
                            dims=conf.dims,
                            use_checkpoint=conf.use_checkpoint,
                            up=True, # Enable upsampling in ResBlock
                            **kwargs,
                        ).make_model() if (
                            conf.resblock_updown
                        ) else Upsample(ch,
                                        conf.conv_resample,
                                        dims=conf.dims,
                                        out_channels=out_ch))
                    ds //= 2  # Track downsampling factor (decreasing now)

                # Add block to network and track features
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self.output_num_blocks[level] += 1
                self._feature_size += ch

        # print(input_block_chans)
        # print('inputs:', self.input_num_blocks)
        # print('outputs:', self.output_num_blocks)

        # Final output layer
        if conf.resnet_use_zero_module:

            # Zero-initialized convolution for stable training
            self.out = nn.Sequential(
                normalization(ch), # Group normalization
                nn.SiLU(), # Activation
                zero_module( # Zero-initialize for stable training
                    conv_nd(conf.dims,
                            input_ch,
                            conf.out_channels,
                            3,
                            padding=1)),
            )
        else:

            # Standard initialization
            self.out = nn.Sequential(
                normalization(ch), # Group normalization
                nn.SiLU(), # Activation
                conv_nd(conf.dims, input_ch, conf.out_channels, 3, padding=1), # Final conv
            )

    def forward(self, x, t, y=None, **kwargs):
        """
        Apply the model to an input batch.

        This method runs the U-Net forward pass, processing the input through the encoder path,
        bottleneck, and decoder path with skip connections. The network is conditioned on diffusion
        timesteps to predict either noise or the denoised image.

        :param x: an [N x C x ...] Tensor of inputs (noisy images).
        :param t: a 1-D batch of timesteps for conditioning the diffusion process.
        :param y: an [N] Tensor of labels, if class-conditional.
        :param kwargs: Additional arguments (unused).
        :return: an object containing the model prediction.
        """

        # Verify class conditioning is used consistently
        assert (y is not None) == (
            self.conf.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        # hs = []

        # Initialize lists to store features at each resolution for skip connections
        # One list per resolution level for flexible management of skip connections
        hs = [[] for _ in range(len(self.conf.channel_mult))]

        # Convert integer timesteps to embeddings and process through time embedding network
        emb = self.time_embed(timestep_embedding(t, self.time_emb_channels))

        # Process class conditioning (if enabled)
        if self.conf.num_classes is not None:
            raise NotImplementedError()
            # assert y.shape == (x.shape[0], )
            # emb = emb + self.label_emb(y)

        # new code supports input_num_blocks != output_num_blocks
        # Convert input to model's working precision
        h = x.type(self.dtype)

        # ===== ENCODER PATH =====
        k = 0 # Block counter
        # Process input through encoder blocks, collecting features at each level
        for i in range(len(self.input_num_blocks)): # Iterate through resolution levels
            for j in range(self.input_num_blocks[i]): # Iterate through blocks at this level
                h = self.input_blocks[k](h, emb=emb) # Process through block with time conditioning
                # print(i, j, h.shape)
                hs[i].append(h) # Store features for skip connections
                k += 1
        # Verify we've used all input blocks
        assert k == len(self.input_blocks)

        # ===== BOTTLENECK =====
        # Process through the middle block at lowest resolution
        h = self.middle_block(h, emb=emb)


        k = 0 # Reset block counter
        # Process through decoder blocks, incorporating skip connections
        for i in range(len(self.output_num_blocks)): # Iterate through resolution levels
            for j in range(self.output_num_blocks[i]): # Iterate through blocks at this level
                # Get corresponding skip connection from encoder
                # Take from the end of the list (bottom of encoder to top of decoder)
                try:
                    lateral = hs[-i - 1].pop() # Get the latest feature map from corresponding encoder level
                    # print(i, j, lateral.shape)
                except IndexError:
                    # Handle case where encoder had fewer blocks than decoder
                    lateral = None
                    # print(i, j, lateral)
                # Process through decoder block with time conditioning and skip connection
                h = self.output_blocks[k](h, emb=emb, lateral=lateral)
                k += 1

        # Convert back to input dtype
        h = h.type(x.dtype)
        # Process through final output layers (norm → activation → conv)
        pred = self.out(h)
        return Return(pred=pred) # Return prediction wrapped in a return object


class Return(NamedTuple):
    pred: th.Tensor


@dataclass
class BeatGANsEncoderConfig(BaseConfig):
    """
    Configuration class for the BeatGANs encoder model.

    This encoder is part of the Diffusion Autoencoder architecture. It maps input images
    to a meaningful latent representation that can be manipulated and then decoded back to
    images using the diffusion decoder.
    """
    image_size: int  # Input resolution (assuming square images)
    in_channels: int  # Number of input channels (e.g., 3 for RGB)
    model_channels: int  # Base channel count for the model
    out_hid_channels: int  # Output hidden channel dimension
    out_channels: int  # Dimension of the output latent vector
    num_res_blocks: int  # Number of residual blocks at each resolution
    attention_resolutions: Tuple[int]  # Resolutions at which to apply attention
    dropout: float = 0  # Dropout probability for regularization
    channel_mult: Tuple[int] = (1, 2, 4, 8)  # Channel multipliers at each resolution level
    use_time_condition: bool = True  # Whether to condition on time (optional for encoder)
    conv_resample: bool = True  # Whether to use conv for up/downsampling
    dims: int = 2  # Dimension of convolutions (2 for images)
    use_checkpoint: bool = False  # Whether to use gradient checkpointing
    num_heads: int = 1  # Number of attention heads
    num_head_channels: int = -1  # Channels per attention head (-1 to use num_heads)
    resblock_updown: bool = False  # Whether to use ResBlock for downsampling
    use_new_attention_order: bool = False  # Whether to use new attention implementation
    pool: str = 'adaptivenonzero'  # Method to pool 2D features to 1D vector


    def make_model(self):
        return BeatGANsEncoderModel(self)


class BeatGANsEncoderModel(nn.Module):
    """
     The encoder model for Diffusion Autoencoders.

     This model is essentially half of a U-Net (the downsampling path), designed to extract
     meaningful representations from images. It maps input images to a latent space that
     captures semantic information which can be manipulated and then decoded back to images.

     Unlike the diffusion model's U-Net, this encoder:
     1. Only has a downsampling path (no upsampling decoder)
     2. Ends with a pooling operation to create a fixed-size latent vector
     3. May or may not use time conditioning (depending on configuration)
     """
    def __init__(self, conf: BeatGANsEncoderConfig):

        super().__init__()
        self.conf = conf # Store configuration
        self.dtype = th.float32 # Default precision

        # Create time embedding network if time conditioning is enabled
        if conf.use_time_condition:
            time_embed_dim = conf.model_channels * 4 # Time embedding dimension
            self.time_embed = nn.Sequential(
                linear(conf.model_channels, time_embed_dim), # Initial projection
                nn.SiLU(), # Activation
                linear(time_embed_dim, time_embed_dim), # Final projection
            )
        else:
            time_embed_dim = None # No time embedding

        # Initial convolutional layer
        ch = int(conf.channel_mult[0] * conf.model_channels) # Initial channel count
        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(
                conv_nd(conf.dims, conf.in_channels, ch, 3, padding=1))  # First conv: image → features
        ])
        self._feature_size = ch  # Track feature size for output
        input_block_chans = [ch] # Track channels for potential skip connections
        ds = 1 # Current downsampling factor
        resolution = conf.image_size # Current resolution

        # Build encoder blocks at each resolution level
        for level, mult in enumerate(conf.channel_mult):
            # Add num_res_blocks residual blocks at current resolution
            for _ in range(conf.num_res_blocks):
                layers = [
                    # Residual block with time conditioning (if enabled)
                    ResBlockConfig(
                        ch,  # Input channels
                        time_embed_dim,  # Time embedding dimension
                        conf.dropout,  # Dropout rate
                        out_channels=int(mult * conf.model_channels),  # Output channels
                        dims=conf.dims,  # Dimension (2D)
                        use_condition=conf.use_time_condition,  # Use time conditioning if enabled
                        use_checkpoint=conf.use_checkpoint,  # Gradient checkpointing
                    ).make_model()
                ]
                ch = int(mult * conf.model_channels)  # Update current channels

                # Add attention at specified resolutions
                if resolution in conf.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,  # Channels
                            use_checkpoint=conf.use_checkpoint,  # Gradient checkpointing
                            num_heads=conf.num_heads,  # Attention heads
                            num_head_channels=conf.num_head_channels,  # Channels per head
                            use_new_attention_order=conf.use_new_attention_order,  # Implementation variant
                        ))

                # Add the block to the model
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch  # Track feature dimension
                input_block_chans.append(ch)  # Track for potential skip connections

            # Add downsampling between resolution levels (except last level)
            if level != len(conf.channel_mult) - 1:
                resolution //= 2  # Halve resolution
                out_ch = ch  # Maintain channel count

                # Choose downsampling method: ResBlock with downsampling or dedicated Downsample
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlockConfig(
                            ch,  # Input channels
                            time_embed_dim,  # Time embedding
                            conf.dropout,  # Dropout
                            out_channels=out_ch,  # Output channels
                            dims=conf.dims,  # Dimension
                            use_condition=conf.use_time_condition,  # Time conditional
                            use_checkpoint=conf.use_checkpoint,  # Checkpointing
                            down=True,  # Enable downsampling
                        ).make_model() if (
                            conf.resblock_updown  # Use ResBlock for downsampling if specified
                        ) else Downsample(ch,  # Otherwise use dedicated Downsample
                                          conf.conv_resample,  # Conv vs. pooling
                                          dims=conf.dims,
                                          out_channels=out_ch)))
                ch = out_ch  # Update channels
                input_block_chans.append(ch)  # Track channels
                ds *= 2  # Track downsampling factor
                self._feature_size += ch  # Track feature size


        # Middle block (bottleneck) with Attention
        self.middle_block = TimestepEmbedSequential(
            ResBlockConfig(
                ch,  # Input channels
                time_embed_dim,  # Time embedding
                conf.dropout,  # Dropout
                dims=conf.dims,  # Dimension
                use_condition=conf.use_time_condition,  # Time conditional
                use_checkpoint=conf.use_checkpoint,  # Checkpointing
            ).make_model(),
            AttentionBlock(  # Self-attention for long-range interactions
                ch,
                use_checkpoint=conf.use_checkpoint,
                num_heads=conf.num_heads,
                num_head_channels=conf.num_head_channels,
                use_new_attention_order=conf.use_new_attention_order,
            ),
            ResBlockConfig(  # Second ResBlock
                ch,
                time_embed_dim,
                conf.dropout,
                dims=conf.dims,
                use_condition=conf.use_time_condition,
                use_checkpoint=conf.use_checkpoint,
            ).make_model(),
        )
        self._feature_size += ch  # Track feature size

        # Final output processing: normalization, pooling, and projection
        if conf.pool == "adaptivenonzero":
            # Adaptive pooling to fixed size (1x1) followed by projection to latent dimension
            self.out = nn.Sequential(
                normalization(ch),  # Group normalization
                nn.SiLU(),  # Activation
                nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling to 1x1 spatial size
                conv_nd(conf.dims, ch, conf.out_channels, 1),  # 1x1 conv to out_channels
                nn.Flatten(),  # Flatten to [batch_size, out_channels]
            )
        else:
            raise NotImplementedError(f"Unexpected {conf.pool} pooling")

    def forward(self, x, t=None, return_2d_feature=False):
        """
        Apply the encoder model to an input batch.

        Args:
            x: Input tensor [N x C x H x W] (typically images)
            t: Optional timesteps for conditioning (can be None if use_time_condition=False)
            return_2d_feature: Whether to return the 2D feature maps before pooling

        Returns:
            Latent representation [N x out_channels]
            If return_2d_feature=True, also returns the 2D feature maps before pooling
        """
        if self.conf.use_time_condition:  # Process time embedding if enabled
            emb = self.time_embed(timestep_embedding(t, self.model_channels))
        else:
            emb = None # No conditioning

        results = []  # To store intermediate results for spatial pooling
        h = x.type(self.dtype)  # Ensure correct dtype

        # Process through input blocks (encoder path)
        for module in self.input_blocks:
            h = module(h, emb=emb) # Process with time conditioning
            if self.conf.pool.startswith("spatial"):
                # For spatial pooling, collect features at each resolution
                results.append(h.type(x.dtype).mean(dim=(2, 3)))

        # Process through middle block (bottleneck)
        h = self.middle_block(h, emb=emb)

        # Handle different pooling strategies
        if self.conf.pool.startswith("spatial"):
            # For spatial pooling, concatenate all collected features
            results.append(h.type(x.dtype).mean(dim=(2, 3)))
            h = th.cat(results, axis=-1)
        else:
            # For standard pooling, just convert back to input dtype
            h = h.type(x.dtype)

        h_2d = h  # Store 2D feature maps before pooling
        h = self.out(h)  # Apply final output processing

        # Return appropriate outputs based on flag
        if return_2d_feature:
            return h, h_2d  # Return both latent vector and 2D features
        else:
            return h  # Return only latent vector

    def forward_flatten(self, x):
        """
        Transform the last 2D feature into a flattened vector.

        This is a utility method that applies only the output head to
        pre-computed 2D feature maps. Useful when processing cached features.

        Args:
            x: 2D feature maps [N x C x H x W]

        Returns:
            Flattened representation [N x out_channels]
        """
        h = self.out(x)  # Apply pooling and projection
        return h


class SuperResModel(BeatGANsUNetModel):
    """
    A UNetModel specialized for super-resolution tasks in diffusion models.

    This model extends the BeatGANsUNetModel to perform super-resolution, which involves
    generating a high-resolution image conditioned on a low-resolution version. The model
    works by concatenating the noisy high-resolution image with an upsampled version of the
    low-resolution conditioning image, allowing the network to use the low-resolution details
    as a guide during the denoising process.

    The model is used in the diffusion process where:
    1. A low-resolution image provides structural guidance
    2. Noise is progressively removed while enhancing details
    3. The output is a high-resolution version of the input image

    This approach allows the diffusion model to focus on adding realistic high-frequency
    details while maintaining the overall structure from the low-resolution input.
    """

    def __init__(self, image_size, in_channels, *args, **kwargs):
        """
        Initialize a super-resolution model.

        Args:
            image_size: Target high-resolution size
            in_channels: Original input channels (before concatenation)
            *args, **kwargs: Additional arguments passed to BeatGANsUNetModel

        Note: The actual input channels to the UNet is doubled since we concatenate
              the noisy image with the upsampled low-resolution conditioning image.
        """
        # Double the input channels to accommodate both the noisy image and the upsampled low-res image
        super().__init__(image_size, in_channels * 2, *args, **kwargs)

    def forward(self, x, timesteps, low_res=None, **kwargs):
        """
        Run the super-resolution model on noisy inputs conditioned on low-resolution images.

        Args:
            x: Noisy input tensor [batch_size, channels, height, width]
            timesteps: Diffusion timesteps for noise level conditioning
            low_res: Low-resolution conditioning image [batch_size, channels, low_height, low_width]
            **kwargs: Additional arguments passed to the parent model's forward method

        Returns:
            Model output (typically predicted noise or denoised image)

        Process:
        1. Upsample the low-resolution image to match the target dimensions
        2. Concatenate the upsampled image with the noisy input along the channel dimension
        3. Process the combined input through the standard UNet forward pass
        """
        # Extract target dimensions from the noisy input
        _, _, new_height, new_width = x.shape

        # Upsample the low-resolution image to match target dimensions using bilinear interpolation
        upsampled = F.interpolate(low_res, (new_height, new_width),
                                  mode="bilinear")

        # Concatenate upsampled low-res image with the noisy input along channel dimension
        # This provides the network with the low-resolution details as conditioning
        x = th.cat([x, upsampled], dim=1)

        # Process through the parent UNet model with the concatenated input
        return super().forward(x, timesteps, **kwargs)
