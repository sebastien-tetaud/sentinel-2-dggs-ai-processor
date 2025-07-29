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
    def __init__(self, in_channels, out_channels, bias=True):
        """Spherical Conv without fixed cell IDs"""
        super(SphericalConv, self).__init__()

        # Only the convolution layer - no neighbor processing
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=9, stride=9, bias=bias)

        # Initialize weights
        nn.init.kaiming_normal_(self.conv.weight)
        if bias:
            nn.init.constant_(self.conv.bias, 0.0)

    def forward(self, x, neighbor_indices):
        """Forward pass with external neighbor indices"""
        batch_size, n_channels, n_cells = x.shape

        # Move neighbor_indices to same device as input
        if neighbor_indices.device != x.device:
            neighbor_indices = neighbor_indices.to(x.device)

        # Extract patches using the neighbor indices
        # Shape: [B, C_in, N_patches, 9]
        patches = x[:, :, neighbor_indices]

        # Reshape to [B, C_in, N_patches * 9] for Conv1d
        patches_flat = patches.view(batch_size, n_channels, -1)

        # Apply convolution
        output = self.conv(patches_flat)

        return output

class SphericalConvBlock(nn.Module):
    """Spherical conv block without fixed cell IDs"""

    def __init__(self, in_channels, out_channels):
        super(SphericalConvBlock, self).__init__()

        self.conv = SphericalConv(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, neighbor_indices):
        x = self.conv(x, neighbor_indices)
        x = self.bn(x)
        x = self.relu(x)
        return x

class SphericalDoubleConvBlock(nn.Module):
    """Double spherical conv block without fixed cell IDs"""

    def __init__(self, in_channels, out_channels):
        super(SphericalDoubleConvBlock, self).__init__()

        self.conv1 = SphericalConvBlock(in_channels, out_channels)
        self.conv2 = SphericalConvBlock(out_channels, out_channels)

    def forward(self, x, neighbor_indices):
        x = self.conv1(x, neighbor_indices)
        x = self.conv2(x, neighbor_indices)
        return x

class SphericalConvTranspose(nn.Module):
    """Spherical transpose convolution for upsampling"""

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
        """Forward pass for transpose convolution"""
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

# Memory management function
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

    # Choose what to run:

    unet, output = main()