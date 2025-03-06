from enum import Enum

import torch
from torch import Tensor
from torch.nn.functional import silu

from .latentnet import *
from .unet import *
from ..choices import *


@dataclass
class BeatGANsAutoencConfig(BeatGANsUNetConfig):
    # number of style channels
    enc_out_channels: int = 512
    enc_attn_resolutions: Tuple[int] = None
    enc_pool: str = 'depthconv'
    enc_num_res_block: int = 2
    enc_channel_mult: Tuple[int] = None
    enc_grad_checkpoint: bool = False
    latent_net_conf: MLPSkipNetConfig = None

    def make_model(self):
        return BeatGANsAutoencModel(self)


class BeatGANsAutoencModel(BeatGANsUNetModel):
    """
    Diffusion Autoencoder model that combines an encoder with a diffusion U-Net.

    This model extends the standard diffusion U-Net by adding an encoder network that maps
    input images to a latent space, which then conditions the diffusion process. This allows
    the model to function as an autoencoder, where the encoder extracts meaningful semantic
    representations and the diffusion U-Net acts as a powerful decoder.

    The model supports various training and inference modes:
    1. Standard mode: Encode an image, then use the latent as conditioning for diffusion
    2. Noise-to-latent mode: Map random noise to a latent, then use it for generation
    3. Style encoding: Extract style information at various layers of the network

    This is an implementation of the Diffusion Autoencoder architecture, which enables
    semantic manipulation in the latent space while maintaining high-quality generation.
    """

    def __init__(self, conf: BeatGANsAutoencConfig):
        """
        Initialize the Diffusion Autoencoder model.

        Args:
            conf: Configuration object specifying model architecture parameters
        """
        super().__init__(conf)
        self.conf = conf

        # having only time, cond
        # Create embedding module that separates time and style embeddings
        # Unlike standard diffusion models that only embed time, this embeds both
        # time steps and the conditioning latent (style)
        self.time_embed = TimeStyleSeperateEmbed(
            time_channels=conf.model_channels,
            time_out_channels=conf.embed_channels,
        )

        # Create the encoder network that maps images to latent representations
        # Note that use_time_condition=False because the encoder doesn't need time steps
        self.encoder = BeatGANsEncoderConfig(
            image_size=conf.image_size,
            in_channels=conf.in_channels,
            model_channels=conf.model_channels,
            out_hid_channels=conf.enc_out_channels,
            out_channels=conf.enc_out_channels,
            num_res_blocks=conf.enc_num_res_block,
            attention_resolutions=(conf.enc_attn_resolutions
                                   or conf.attention_resolutions),
            dropout=conf.dropout,
            channel_mult=conf.enc_channel_mult or conf.channel_mult,
            use_time_condition=False,
            conv_resample=conf.conv_resample,
            dims=conf.dims,
            use_checkpoint=conf.use_checkpoint or conf.enc_grad_checkpoint,
            num_heads=conf.num_heads,
            num_head_channels=conf.num_head_channels,
            resblock_updown=conf.resblock_updown,
            use_new_attention_order=conf.use_new_attention_order,
            pool=conf.enc_pool,
        ).make_model()

        # Optional latent processing network
        if conf.latent_net_conf is not None:
            self.latent_net = conf.latent_net_conf.make_model()

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).

        This is used when stochastic encoding is enabled, making the model
        more like a VAE in the encoding process.

        Args:
            mu: Mean of the latent Gaussian [B x D]
            logvar: Log variance of the latent Gaussian [B x D]

        Returns:
            Sampled latent vector
        """
        assert self.conf.is_stochastic
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def sample_z(self, n: int, device):
        """
        Sample random latent vectors from the prior distribution.

        Used for generating new samples without an input image.

        Args:
            n: Number of samples to generate
            device: Device to create tensor on

        Returns:
            Random latent vectors [n x enc_out_channels]
        """
        assert self.conf.is_stochastic  # Verify stochastic mode is enabled
        return torch.randn(n, self.conf.enc_out_channels, device=device)

    def noise_to_cond(self, noise: Tensor):
        """
        Transform random noise to conditioning latent.

        This would allow mapping random noise to the semantic latent space,
        but is not implemented in this version.
        """
        raise NotImplementedError()
        assert self.conf.noise_net_conf is not None
        return self.noise_net.forward(noise)

    def encode(self, x):
        """
        Encode input images to latent representations.

        Args:
            x: Input images [batch_size x channels x height x width]

        Returns:
            Dictionary containing the conditioning latent 'cond'
        """
        cond = self.encoder.forward(x)  # Pass through encoder network
        return {'cond': cond}  # Return as dictionary for flexibility

    @property
    def stylespace_sizes(self):
        """
        Get the dimensions of each style vector in the style space.

        Returns:
            List of dimensions for each style vector
        """
        # Collect all modules that might contain style information
        modules = list(self.input_blocks.modules()) + list(
            self.middle_block.modules()) + list(self.output_blocks.modules())
        sizes = []
        # Find all ResBlocks which contain style conditioning
        for module in modules:
            if isinstance(module, ResBlock):
                linear = module.cond_emb_layers[-1]  # Get the last linear layer
                sizes.append(linear.weight.shape[0])  # Record output dimension
        return sizes

    def encode_stylespace(self, x, return_vector: bool = True):
        """
        Encode input to style space representation.

        Unlike the standard latent encoding, this extracts style information at
        various levels of the network, similar to StyleGAN's style space.

        Args:
            x: Input images
            return_vector: Whether to concatenate styles to a single vector

        Returns:
            Either a concatenated style vector or list of style vectors
        """
        # Collect all modules that might contain style information
        modules = list(self.input_blocks.modules()) + list(
            self.middle_block.modules()) + list(self.output_blocks.modules())

        # Get basic conditioning from encoder
        cond = self.encoder.forward(x)  # [batch_size x latent_dim]

        # Process conditioning through each ResBlock's style layers
        S = []
        for module in modules:
            if isinstance(module, ResBlock):
                # Transform conditioning to style for this block
                s = module.cond_emb_layers.forward(cond)  # [batch_size x style_dim]
                S.append(s)

        if return_vector:
            # Return single concatenated vector [batch_size x sum(style_dims)]
            return torch.cat(S, dim=1)
        else:
            # Return list of style vectors
            return S

    def forward(self,
                x,
                t,
                y=None,
                x_start=None,
                cond=None,
                style=None,
                noise=None,
                t_cond=None,
                **kwargs):

        """
        Apply the model to an input batch with various conditioning options.

        This method is highly flexible and supports multiple modes:
        1. Encode-then-diffuse: Take x_start, encode it, and use the encoding as conditioning
        2. Diffuse-with-cond: Take pre-computed conditioning and apply diffusion
        3. Noise-to-cond: Generate conditioning from random noise (if implemented)

        Args:
            x: Noisy input tensor (for diffusion step)
            t: Diffusion timesteps
            y: Optional class labels (if model is class-conditional)
            x_start: Original clean image to encode
            cond: Pre-computed conditioning (if not encoding from x_start)
            style: Optional explicit style information to override extracted style
            noise: Random noise to predict conditioning from (if implemented)
            t_cond: Optional separate timesteps for conditioning (usually same as t)
            **kwargs: Additional arguments

        Returns:
            AutoencReturn object containing model prediction and conditioning
        """
        # Use same timestep for conditioning if not specified
        if t_cond is None:
            t_cond = t

        # If noise is provided, predict conditioning from it
        if noise is not None:
            # if the noise is given, we predict the cond from noise
            cond = self.noise_to_cond(noise)

        # If no conditioning is provided, encode it from x_start
        if cond is None:
            if x is not None:
                assert len(x) == len(x_start), f'{len(x)} != {len(x_start)}'
            # Encode the input image to get conditioning
            tmp = self.encode(x_start)
            cond = tmp['cond']

        # Create time embeddings if timesteps are provided
        if t is not None:
            _t_emb = timestep_embedding(t, self.conf.model_channels) # Main timestep
            _t_cond_emb = timestep_embedding(t_cond, self.conf.model_channels)  # Conditioning timestep
        else:
            # This happens when training only the autoencoder part
            _t_emb = None
            _t_cond_emb = None

        # Process embeddings through time/style embedding module
        if self.conf.resnet_two_cond:
            # Process time embedding and conditioning through specialized module
            res = self.time_embed.forward(
                time_emb=_t_emb,
                cond=cond,
                time_cond_emb=_t_cond_emb,
            )
        else:
            raise NotImplementedError()

        # Extract embeddings based on model configuration
        if self.conf.resnet_two_cond:
            # Two separate conditions: time embedding and conditional embedding
            emb = res.time_emb  # Time embedding for main diffusion process
            cond_emb = res.emb  # Conditioning embedding from encoder
        else:
            # Combined embedding of time and condition
            emb = res.emb
            cond_emb = None

        assert (y is not None) == (
            self.conf.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        if self.conf.num_classes is not None:
            raise NotImplementedError()
            # assert y.shape == (x.shape[0], )
            # emb = emb + self.label_emb(y)

        # Prepare embeddings for different parts of the network
        # Time embeddings
        enc_time_emb = emb  # For encoder part of U-Net
        mid_time_emb = emb  # For middle part
        dec_time_emb = emb  # For decoder part

        # Style/conditioning embeddings
        enc_cond_emb = cond_emb  # For encoder part
        mid_cond_emb = cond_emb  # For middle part
        dec_cond_emb = cond_emb  # For decoder part

        # Initialize hierarchical feature storage for skip connections
        # Create a separate list for each resolution level to store features
        # hs = []
        hs = [[] for _ in range(len(self.conf.channel_mult))]

        # Process through encoder path if input is provided
        if x is not None:
            h = x.type(self.dtype) # Convert input to model's working precision

            # ===== ENCODER PATH (Input Blocks) =====
            # input blocks
            k = 0 # Block counter
            for i in range(len(self.input_num_blocks)): # Iterate through resolution levels
                for j in range(self.input_num_blocks[i]): # Iterate through blocks at this level

                    # Process features through current block with time and style conditioning
                    h = self.input_blocks[k](h,
                                             emb=enc_time_emb,
                                             cond=enc_cond_emb)

                    # print(i, j, h.shape)
                    # Store features for later use in skip connections
                    hs[i].append(h)
                    k += 1 # Increment block counter
            assert k == len(self.input_blocks) # Verify we've processed all input blocks

            # ===== BOTTLENECK (Middle Block) =====
            # Process features through middle block (lowest resolution)
            h = self.middle_block(h, emb=mid_time_emb, cond=mid_cond_emb)
        else:
            # Special case: no input provided
            # This happens when training only the autoencoder component
            # No feature maps to process or skip connections to establish
            h = None
            hs = [[] for _ in range(len(self.conf.channel_mult))]

        # ===== DECODER PATH (Output Blocks) =====
        k = 0 # Reset block counter
        for i in range(len(self.output_num_blocks)): # Iterate through resolution levels
            for j in range(self.output_num_blocks[i]): # Iterate through blocks at this level
                # Get corresponding skip connection from encoder path
                # The skip connections are used in reverse order (from deepest to shallowest)
                try:
                    # Pop the latest feature map from the corresponding encoder level
                    # Using negative indexing to start from the deepest level
                    lateral = hs[-i - 1].pop()
                    # print(i, j, lateral.shape)
                except IndexError:
                    # No skip connection available at this level
                    # This can happen if encoder had fewer blocks than decoder
                    lateral = None
                    # print(i, j, lateral)


                # Process through decoder block with:
                # - Current features (h)
                # - Time embedding
                # - Style/conditioning embedding
                # - Skip connection from encoder (lateral)
                h = self.output_blocks[k](h,
                                          emb=dec_time_emb,
                                          cond=dec_cond_emb,
                                          lateral=lateral)
                k += 1 # Increment block counter

        # ===== FINAL OUTPUT =====
        # Process through final output layer (normalization → activation → convolution)
        pred = self.out(h)

        # Return model prediction and the conditioning latent
        return AutoencReturn(pred=pred, cond=cond)


class AutoencReturn(NamedTuple):
    pred: Tensor
    cond: Tensor = None


class EmbedReturn(NamedTuple):
    # style and time
    emb: Tensor = None
    # time only
    time_emb: Tensor = None
    # style only (but could depend on time)
    style: Tensor = None


class TimeStyleSeperateEmbed(nn.Module):
    # embed only style
    def __init__(self, time_channels, time_out_channels):
        super().__init__()
        self.time_embed = nn.Sequential(
            linear(time_channels, time_out_channels),
            nn.SiLU(),
            linear(time_out_channels, time_out_channels),
        )
        self.style = nn.Identity()

    def forward(self, time_emb=None, cond=None, **kwargs):
        if time_emb is None:
            # happens with autoenc training mode
            time_emb = None
        else:
            time_emb = self.time_embed(time_emb)
        style = self.style(cond)
        return EmbedReturn(emb=style, time_emb=time_emb, style=style)
