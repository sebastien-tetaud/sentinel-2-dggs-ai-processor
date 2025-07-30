import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import xarray as xr
import healpy as hp
import gc

class NeighborIndexProcessor:
    """External processor for building neighbor indices - OUTSIDE the model"""

    def __init__(self, level, nest=True):
        self.level = level
        self.NSIDE = 2 ** level
        self.nest = nest
        self.cache = {}

    def build_neighbor_indices(self, available_cell_ids, stride=1):
        """Build neighbor indices for given cell IDs and stride"""
        # Create cache key
        cache_key = (tuple(sorted(available_cell_ids)), stride)

        if cache_key in self.cache:
            return self.cache[cache_key]

        available_cell_set = set(available_cell_ids)
        neighbor_indices = []

        # Create cell_id to data_index mapping
        cell_to_data_idx = {cell_id: i for i, cell_id in enumerate(available_cell_ids)}

        # Apply stride to center cell list
        center_cells = available_cell_ids[::stride]

        for cell_id in center_cells:
            neighbors = hp.get_all_neighbours(self.NSIDE, cell_id, nest=self.nest)

            # Validate each neighbor; replace invalid or missing with center
            valid_neighbors = [
                n if (n != -1 and n in available_cell_set) else cell_id
                for n in neighbors
            ]

            patch = [cell_id] + valid_neighbors  # Center + 8 neighbors

            # Convert to data indices
            patch_data_indices = [cell_to_data_idx[cell_id] for cell_id in patch]
            neighbor_indices.append(patch_data_indices)

        # Convert to tensor
        data_neighbor_indices = torch.tensor(neighbor_indices, dtype=torch.long)

        # Cache the result
        self.cache[cache_key] = data_neighbor_indices

        return data_neighbor_indices

    def clear_cache(self):
        """Clear the cache"""
        self.cache.clear()

class SphericalConv(nn.Module):
    """
    Spherical Convolution layer for HEALPix-organized Earth observation data.

    This layer performs convolution operations on spherical data by processing 3×3
    neighborhoods defined by HEALPix geometry. Unlike traditional CNNs that operate
    on regular Euclidean grids, this implementation handles the irregular spatial
    relationships inherent to spherical surfaces through externally computed neighbor
    indices.

    The core innovation is the separation of geometric processing (neighbor finding)
    from neural computation (convolution), enabling flexible processing of arbitrary
    HEALPix regions while maintaining computational efficiency through GPU parallelization.

    Architecture:
    - Uses 1D convolution with kernel_size=9 and stride=9
    - Processes flattened 3×3 spherical patches (center + 8 neighbors = 9 values)
    - Applies the same learned spatial filter to all patches simultaneously
    - Maintains one-to-one correspondence: N input patches → N output features

    Parameters
    ----------
    in_channels : int
        Number of input channels (e.g., spectral bands in satellite imagery).
    out_channels : int
        Number of output feature channels to be learned by the convolution.
    bias : bool, optional
        If True, adds a learnable bias term to the convolution output. Default: True.

    Attributes
    ----------
    conv : nn.Conv1d
        1D convolution layer with kernel_size=9, stride=9 that processes flattened
        spherical patches.

    Notes
    -----
    - Requires externally computed neighbor_indices that define 3×3 spherical
      neighborhoods for each spatial location
    - All patches are processed in parallel for computational efficiency
    - Output preserves spatial correspondence: output[i] corresponds to input patch[i]
    - Handles device compatibility automatically (CPU/GPU synchronization)

    Examples
    --------
    >>> # Initialize spherical convolution layer
    >>> conv_layer = SphericalConv(in_channels=4, out_channels=64)
    >>>
    >>> # Input: [batch_size, channels, n_cells]
    >>> x = torch.randn(1, 4, 10000)  # 1 batch, 4 bands, 10k HEALPix cells
    >>>
    >>> # Neighbor indices: [n_patches, 9] - precomputed 3×3 neighborhoods
    >>> neighbor_indices = torch.randint(0, 10000, (5000, 9))  # 5k patches
    >>>
    >>> # Forward pass
    >>> output = conv_layer(x, neighbor_indices)
    >>> print(output.shape)  # [1, 64, 5000] - 64 features for 5k patches

    Mathematical Operation
    ----------------------
    For each patch i with 9 cells [c₀, c₁, ..., c₈] and learned weights [w₀, w₁, ..., w₈]:

        output[i] = Σⱼ(cⱼ × wⱼ) + bias

    where c₀ is the center cell and c₁-c₈ are the 8 HEALPix neighbors.

    See Also
    --------
    SphericalConvBlock : Spherical convolution with batch normalization and activation
    SphericalDoubleConvBlock : Double spherical convolution block for U-Net architectures
    NeighborIndexProcessor : Utility for computing HEALPix neighbor relationships
    """

    def __init__(self, in_channels, out_channels, bias=True):
        super(SphericalConv, self).__init__()

        # Only the convolution layer - no neighbor processing
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=9, stride=9, bias=bias)

        # Initialize weights
        nn.init.kaiming_normal_(self.conv.weight)
        if bias:
            nn.init.constant_(self.conv.bias, 0.0)

    def forward(self, x, neighbor_indices):
        """
        Perform spherical convolution on input data using precomputed neighbor indices.

        This method extracts 3×3 spherical neighborhoods from the input tensor based on
        the provided neighbor indices, then applies 1D convolution to process all patches
        in parallel. Each patch represents a local spherical neighborhood where the
        spatial relationships are defined by HEALPix geometry.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [batch_size, in_channels, n_cells] containing
            the data values for all HEALPix cells in the region.
        neighbor_indices : torch.Tensor
            Precomputed neighbor indices of shape [n_patches, 9] where each row
            contains the data indices for one 3×3 spherical neighborhood. The first
            index in each row is the center cell, followed by 8 neighbor indices.

        Returns
        -------
        torch.Tensor
            Output feature tensor of shape [batch_size, out_channels, n_patches]
            where each spatial location corresponds to the convolution result for
            one spherical patch.

        Raises
        ------
        RuntimeError
            If input tensor and neighbor_indices have incompatible devices.
        IndexError
            If neighbor_indices contain values outside the valid range [0, n_cells).

        Notes
        -----
        Processing Pipeline:
        1. Device synchronization: Move neighbor_indices to same device as input
        2. Patch extraction: Extract all 3×3 neighborhoods simultaneously
        3. Tensor reshaping: Flatten patches for 1D convolution processing
        4. Convolution: Apply learned spatial filters to all patches in parallel

        The patch extraction step uses advanced tensor indexing to efficiently
        gather all required neighborhoods in a single operation, enabling
        massive parallelization on GPU hardware.

        Examples
        --------
        >>> conv = SphericalConv(in_channels=3, out_channels=16)
        >>> x = torch.randn(2, 3, 1000)  # 2 batches, 3 channels, 1000 cells
        >>> indices = torch.randint(0, 1000, (500, 9))  # 500 patches
        >>> output = conv(x, indices)  # Shape: [2, 16, 500]
        """
        batch_size, n_channels, n_cells = x.shape

        # Move neighbor_indices to same device as input
        if neighbor_indices.device != x.device:
            neighbor_indices = neighbor_indices.to(x.device)

        # Extract patches using the neighbor indices
        # Shape: [B, C_in, N_patches, 9]
        patches = x[:, :, neighbor_indices]

        # Visualize some example patches
        print(f"Example patches (first 3):")
        for i in range(min(3, patches.shape[2])):
            patch_data = patches[0, 0, i]  # First batch, first channel, patch i
            patch_indices = neighbor_indices[i]
            print(f"Patch {i}: indices {patch_indices[:3].cpu().numpy()}... → values {patch_data[:3].detach().cpu().numpy()}...")

        # Reshape to [B, C_in, N_patches * 9] for Conv1d
        patches_flat = patches.view(batch_size, n_channels, -1)

        # Apply convolution
        output = self.conv(patches_flat)

        return output

class SphericalConvBlock(nn.Module):
    """
    Spherical Convolution Block with batch normalization and ReLU activation.

    This block combines spherical convolution with standard deep learning components
    (batch normalization and ReLU activation) to create a robust building block for
    spherical neural networks. It follows the widely-used pattern of Conv → BatchNorm → ReLU
    that has proven effective in modern deep learning architectures.

    Architecture:
    - SphericalConv: Processes 3×3 spherical neighborhoods using HEALPix geometry
    - BatchNorm1d: Normalizes feature distributions for stable training
    - ReLU: Introduces non-linearity for learning complex spatial patterns

    Parameters
    ----------
    in_channels : int
        Number of input channels (e.g., spectral bands in satellite imagery).
    out_channels : int
        Number of output feature channels to be learned by the spherical convolution.

    Attributes
    ----------
    conv : SphericalConv
        Spherical convolution layer that processes 3×3 HEALPix neighborhoods.
    bn : nn.BatchNorm1d
        Batch normalization layer applied along the channel dimension.
    relu : nn.ReLU
        ReLU activation function with in-place operation for memory efficiency.

    Notes
    -----
    - Batch normalization operates on the channel dimension, normalizing across
      all spatial locations (patches) within each batch
    - ReLU activation is applied in-place for memory efficiency
    - The block preserves spatial correspondence: input patch i → output feature i
    - Designed as a drop-in replacement for standard Conv2d blocks in CNNs

    Processing Pipeline
    -------------------
    1. Spherical Convolution: Extract and process 3×3 spherical neighborhoods
    2. Batch Normalization: Normalize feature distributions across spatial locations
    3. ReLU Activation: Apply non-linear activation function

    The batch normalization step is particularly important for spherical networks as
    it helps stabilize training when processing irregular spatial arrangements and
    varying neighborhood sizes that can occur at HEALPix boundaries.

    See Also
    --------
    SphericalConv : Underlying spherical convolution operation
    SphericalDoubleConvBlock : Double convolution block for U-Net architectures
    SphericalUNet : Complete U-Net architecture using spherical convolution blocks
    """

    def __init__(self, in_channels, out_channels):
        super(SphericalConvBlock, self).__init__()

        self.conv = SphericalConv(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, neighbor_indices):
        """
        Forward pass through spherical convolution block.

        Processes input through spherical convolution, batch normalization, and
        ReLU activation in sequence. This creates normalized, non-linear feature
        representations that are suitable for building deeper spherical networks.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [batch_size, in_channels, n_cells] containing
            the data values for all HEALPix cells in the region.
        neighbor_indices : torch.Tensor
            Precomputed neighbor indices of shape [n_patches, 9] defining 3×3
            spherical neighborhoods for convolution processing.

        Returns
        -------
        torch.Tensor
            Output feature tensor of shape [batch_size, out_channels, n_patches]
            containing normalized and activated features for each spherical patch.
            All values are non-negative due to ReLU activation.

        Notes
        -----
        The forward pass applies three sequential operations:

        1. **Spherical Convolution**: Extracts spatial features from 3×3 HEALPix
           neighborhoods using learned convolutional filters

        2. **Batch Normalization**: Normalizes the feature distributions by:
           - Computing mean and variance across spatial dimensions (patches)
           - Applying learnable scale and shift parameters
           - Improving training stability and convergence speed

        3. **ReLU Activation**: Introduces non-linearity by setting negative
           values to zero, enabling the network to learn complex spatial patterns

        The batch normalization is particularly beneficial for spherical data as
        it helps handle the irregular spatial arrangements and boundary effects
        inherent in HEALPix tessellations.
        """
        x = self.conv(x, neighbor_indices)
        x = self.bn(x)
        x = self.relu(x)
        return x

class SphericalDoubleConvBlock(nn.Module):
    """
    Double spherical convolution block for deeper feature extraction.

    Applies two sequential spherical convolution blocks (Conv→BN→ReLU → Conv→BN→ReLU)
    following the standard U-Net double convolution pattern. This design allows the
    network to learn more complex spatial features within spherical neighborhoods.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels for both convolution blocks.

    Notes
    -----
    Architecture: Input → SphericalConvBlock → SphericalConvBlock → Output
    - First block: in_channels → out_channels
    - Second block: out_channels → out_channels (refinement)
    """

    def __init__(self, in_channels, out_channels):
        super(SphericalDoubleConvBlock, self).__init__()

        self.conv1 = SphericalConvBlock(in_channels, out_channels)
        self.conv2 = SphericalConvBlock(out_channels, out_channels)

    def forward(self, x, neighbor_indices):
        """
        Apply double spherical convolution.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor [batch_size, in_channels, n_cells].
        neighbor_indices : torch.Tensor
            Precomputed neighbor indices [n_patches, 9].

        Returns
        -------
        torch.Tensor
            Output features [batch_size, out_channels, n_patches].
        """
        x = self.conv1(x, neighbor_indices)
        x = self.conv2(x, neighbor_indices)
        return x

class SphericalConvTranspose(nn.Module):
    """
    Spherical transpose convolution for learnable upsampling in decoder paths.

    Performs learned upsampling using 1D transpose convolution with the same
    kernel and stride configuration as SphericalConv (kernel_size=9, stride=9).
    Commonly used in U-Net decoder paths as an alternative to bilinear upsampling.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels after upsampling.
    bias : bool, optional
        Whether to include learnable bias term. Default: True.

    Notes
    -----
    Architecture: Input → ConvTranspose1d → BatchNorm1d → ReLU → Output
    - Uses same kernel_size=9 and stride=9 as forward spherical convolution
    - Applies Kaiming initialization for stable training
    - Note: neighbor_indices parameter is unused but maintained for API consistency
    """

    def __init__(self, in_channels, out_channels, bias=True):
        super(SphericalConvTranspose, self).__init__()

        # ConvTranspose1d for upsampling
        self.conv_transpose = nn.ConvTranspose1d(
            in_channels, out_channels,
            kernel_size=9, stride=9, bias=bias
        )

        # Batch norm and activation
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Initialize weights
        nn.init.kaiming_normal_(self.conv_transpose.weight)
        if bias:
            nn.init.constant_(self.conv_transpose.bias, 0.0)

    def forward(self, x, neighbor_indices):
        """
        Apply transpose convolution for upsampling.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor [batch_size, in_channels, n_patches].
        neighbor_indices : torch.Tensor
            Unused parameter maintained for API consistency with other spherical layers.

        Returns
        -------
        torch.Tensor
            Upsampled output [batch_size, out_channels, n_patches_upsampled].
        """
        x = self.conv_transpose(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class SphericalDecoderBlock(nn.Module):
    """Spherical decoder block with upsampling and skip connections"""

    def __init__(self, in_channels, out_channels, skip_channels, bilinear=True):
        super(SphericalDecoderBlock, self).__init__()

        self.bilinear = bilinear

        if bilinear:
            # Bilinear upsampling + spherical conv to adjust channels
            self.up_sample = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
            self.up_conv = SphericalConv(in_channels, in_channels // 2)
        else:
            # ConvTranspose for learnable upsampling
            self.up = SphericalConvTranspose(in_channels, in_channels // 2)

        # Double conv after concatenating skip connection
        self.conv = SphericalDoubleConvBlock(
            in_channels // 2 + skip_channels,
            out_channels
        )

    def forward(self, x, skip, neighbor_indices):
        """Forward pass for decoder block"""

        if self.bilinear:
            # Bilinear upsampling then spherical conv
            x = self.up_sample(x)
            if x.size(2) != skip.size(2):
                x = F.interpolate(x, size=skip.size(2), mode='linear', align_corners=False)
            x = self.up_conv(x, neighbor_indices)
        else:
            # ConvTranspose upsampling
            x = self.up(x, neighbor_indices)
            # Handle size mismatch
            if x.size(2) != skip.size(2):
                x = F.interpolate(x, size=skip.size(2), mode='linear', align_corners=False)

        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)

        # Apply double convolution
        x = self.conv(x, neighbor_indices)

        return x

class SphericalUNet(nn.Module):
    """
    Flexible Spherical U-Net that works with any chunk
    Uses external neighbor processing for maximum flexibility
    """

    def __init__(self, in_channels, out_channels, features=[64, 128, 256, 512], bilinear=True):
        super(SphericalUNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features
        self.bilinear = bilinear

        # Encoder path
        self.encoder_blocks = nn.ModuleList()
        current_channels = in_channels

        for feature in features:
            self.encoder_blocks.append(
                SphericalDoubleConvBlock(current_channels, feature)
            )
            current_channels = feature

        # Bottleneck
        self.bottleneck = SphericalDoubleConvBlock(features[-1], features[-1] * 2)

        # Decoder path
        self.decoder_blocks = nn.ModuleList()

        # Build decoder blocks
        decoder_features = features[::-1]  # Reverse features for decoder

        for i in range(len(decoder_features)):
            if i == 0:
                # First decoder block (from bottleneck)
                in_ch = features[-1] * 2
                out_ch = decoder_features[i]
                skip_ch = decoder_features[i]
            else:
                # Subsequent decoder blocks
                in_ch = decoder_features[i-1]
                out_ch = decoder_features[i]
                skip_ch = decoder_features[i]

            self.decoder_blocks.append(
                SphericalDecoderBlock(in_ch, out_ch, skip_ch, bilinear=bilinear)
            )

        # Final output layer
        self.final_conv = SphericalConv(features[0], out_channels)

    def forward(self, x, neighbor_indices):
        """
        Forward pass through Spherical U-Net

        Args:
            x: Input tensor [B, C_in, N_cells]
            neighbor_indices: Pre-computed neighbor indices [N_patches, 9]

        Returns:
            Output tensor [B, C_out, N_patches]
        """
        # Encoder path with skip connections
        skip_connections = []

        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, neighbor_indices)
            skip_connections.append(x)

        # Bottleneck
        x = self.bottleneck(x, neighbor_indices)

        # Decoder path
        skip_connections = skip_connections[::-1]  # Reverse for decoder

        for i, decoder_block in enumerate(self.decoder_blocks):
            skip = skip_connections[i]
            x = decoder_block(x, skip, neighbor_indices)

        # Final output
        x = self.final_conv(x, neighbor_indices)

        return x

# Factory function for easy U-Net creation
def create_spherical_unet(in_channels, out_channels, features=[64, 128, 256, 512], bilinear=True):
    """
    Create Spherical U-Net

    Args:
        in_channels: Number of input channels (spectral bands)
        out_channels: Number of output channels (classes)
        features: Feature dimensions for encoder/decoder [64, 128, 256, 512]
        bilinear: Use bilinear upsampling (True) or ConvTranspose (False)

    Returns:
        SphericalUNet model
    """
    return SphericalUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        features=features,
        bilinear=bilinear
    )


def clear_memory():
    """Clear GPU and CPU memory"""
    vars_to_delete = ['model', 'unet', 'x_tensor', 'output', 'x_multi_band', 'spectral_data', 'neighbor_indices']
    for var in vars_to_delete:
        if var in globals():
            del globals()[var]

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    print("🧹 Memory cleared")


def main():
    """Complete example of creating and using Spherical U-Net"""

    print("Creating Spherical U-Net")
    print("=" * 40)

    # Clear memory before starting
    clear_memory()

    # Check GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load your data
    ds_healpix = xr.open_dataset("/home/ubuntu/project/sentinel-2-dggs-ai-processor/src/notebook/healpix.zarr")
    available_cell_ids = ds_healpix.cell_ids.values
    print(f"Number of available HEALPix cells: {len(available_cell_ids):,}")

    # Parameters
    level = 19
    band_list = ['b02', 'b03', 'b04', 'b08']
    in_channels = len(band_list)
    out_channels = len(band_list)
    stride = 1

    print(f"Model config: {in_channels} input channels → {out_channels} output classes")

    # 1. Create neighbor processor
    print("Creating neighbor processor...")
    neighbor_processor = NeighborIndexProcessor(level=level)
    neighbor_indices = neighbor_processor.build_neighbor_indices(available_cell_ids, stride)
    print(f"Generated {neighbor_indices.shape[0]:,} patches")

    # 2. Create Spherical U-Net
    print("Creating Spherical U-Net...")
    unet = create_spherical_unet(
        in_channels=in_channels,
        out_channels=out_channels,
        features=[8, 16, 32],  # Smaller features for memory efficiency
        bilinear=True  # Use bilinear upsampling (faster)
    )

    # Move to GPU
    unet = unet.to(device)

    # Model info
    total_params = sum(p.numel() for p in unet.parameters())
    trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # 3. Load and prepare data
    print("Loading data...")
    spectral_data = []
    for band in band_list:
        band_data = ds_healpix.Sentinel2.sel(bands=band).compute().values
        spectral_data.append(band_data)

    x_multi_band = np.stack(spectral_data, axis=0)
    print(f"Multi-band data shape: {x_multi_band.shape}")

    x_tensor = torch.tensor(x_multi_band, dtype=torch.float32).unsqueeze(0)
    print(f"Input tensor shape: {x_tensor.shape}")

    # Move to GPU
    x_tensor = x_tensor.to(device)

    # 4. Forward pass
    print("Running forward pass...")
    unet.eval()
    with torch.no_grad():
        output = unet(x_tensor, neighbor_indices)
        print(f"Output tensor shape: {output.shape}")

    print("Spherical U-Net created and tested successfully!")

    # Clear memory
    clear_memory()

    return unet, output


def unet_test():
    """Quick test with minimal data"""

    print("Quick U-Net Test")
    print("=" * 25)

    clear_memory()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create small test case
    mock_cell_ids = np.arange(1000)  # 1000 mock cells
    level = 19  # Lower level for faster processing

    # Create processor and model
    processor = NeighborIndexProcessor(level=level)
    neighbor_indices = processor.build_neighbor_indices(mock_cell_ids, stride=1)

    unet = create_spherical_unet(
        in_channels=4,
        out_channels=4,
        features=[8, 16, 32],  # Very small for testing
        bilinear=True
    ).to(device)

    # Test data
    x = torch.randn(1, 4, 1000).to(device)

    # Forward pass
    with torch.no_grad():
        output = unet(x, neighbor_indices)

    print(f"Quick test passed!")
    print(f"Input: {x.shape} → Output: {output.shape}")

    clear_memory()

if __name__ == "__main__":

    unet, output = main()