"""
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


LOCALIZATION_MODEL_PRESETS = {
    'small': {
        'encoder_channels': (32, 64, 128, 256),
        'bridge_channels': 64,
        'skip_channels': (64, 32, 16),
        'decoder_channels': (64, 32, 16),
        'output_channels': 8,
        'bridge_depth': 8,
    },
    'medium': {
        'encoder_channels': (64, 128, 256, 512),
        'bridge_channels': 128,
        'skip_channels': (128, 64, 32),
        'decoder_channels': (128, 64, 32),
        'output_channels': 16,
        'bridge_depth': 8,
    },
    'large': {
        'encoder_channels': (80, 160, 320, 640),
        'bridge_channels': 160,
        'skip_channels': (160, 80, 40),
        'decoder_channels': (160, 80, 40),
        'output_channels': 24,
        # Keep the bottleneck depth aligned with the fixed 8->16->32->64 decoder ladder.
        'bridge_depth': 8,
    },
}


class Mish(nn.Module):
    """Mish activation."""
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class DepthPositionalEncoding(nn.Module):
    """Depth positional encoding using sinusoidal embeddings."""

    def __init__(self, d_model=64, max_depth=64):
        super().__init__()
        self.d_model = d_model
        self.max_depth = max_depth

        pe = torch.zeros(max_depth, d_model)
        z = torch.arange(0, max_depth, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(z * div_term)
        pe[:, 1::2] = torch.cos(z * div_term)[:, :pe[:, 1::2].shape[1]]

        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_depth, d_model)

    def forward(self, z_idx):
        """z_idx is a depth index in [0, max_depth - 1]."""
        if not torch.is_tensor(z_idx):
            z_idx = torch.as_tensor(z_idx, device=self.pe.device)
        else:
            z_idx = z_idx.to(self.pe.device)
        return self.pe[0, z_idx.long(), :]


class FrequencyDomainAttention(nn.Module):
    """Frequency-domain attention branch."""

    def __init__(self, in_channels=1, out_channels=64):
        super().__init__()

        # Low-frequency branch for broad regional patterns.
        self.low_freq_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels // 2, out_channels // 2, 3, padding=1)
        )

        self.high_freq_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels // 2, out_channels // 2, 3, padding=1)
        )

        # Fuse low- and high-frequency features in the spatial domain.
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """x: (B, 1, 64, 64)"""
        # Transform the input map to the frequency domain.
        x_fft = torch.fft.fft2(x.squeeze(1))  # (B, 64, 64)
        x_fft_shift = torch.fft.fftshift(x_fft)

        B, H, W = x_fft_shift.shape
        center_h, center_w = H // 2, W // 2
        radius = min(H, W) // 4

        # Build radial masks around the centered spectrum.
        y, x_coord = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        dist = torch.sqrt((y - center_h) ** 2 + (x_coord - center_w) ** 2)
        low_freq_mask = (dist <= radius).float().to(x.device)
        high_freq_mask = 1 - low_freq_mask

        low_freq = x_fft_shift * low_freq_mask.unsqueeze(0)
        high_freq = x_fft_shift * high_freq_mask.unsqueeze(0)

        # Return each spectral band to the spatial domain.
        low_freq_spatial = torch.fft.ifft2(torch.fft.ifftshift(low_freq)).real.unsqueeze(1)
        high_freq_spatial = torch.fft.ifft2(torch.fft.ifftshift(high_freq)).real.unsqueeze(1)

        # Extract and fuse frequency-band features.
        low_feat = self.low_freq_conv(low_freq_spatial)
        high_feat = self.high_freq_conv(high_freq_spatial)

        fused = torch.cat([low_feat, high_feat], dim=1)
        output = self.fusion(fused)

        return output


class ChannelAttention(nn.Module):
    """Channel attention module."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden_channels = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, hidden_channels, bias=False),
            Mish(),
            nn.Linear(hidden_channels, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        out = avg_out + max_out
        return self.sigmoid(out).view(x.size(0), -1, 1, 1) * x


class SpatialAttention(nn.Module):
    """Spatial attention module."""
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat)) * x


class CBAM2D(nn.Module):
    """2D CBAM attention block."""
    def __init__(self, channels):
        super().__init__()
        self.channel_attention = ChannelAttention(channels)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class ChannelAttention3D(nn.Module):
    """3D channel attention module."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden_channels = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, hidden_channels, bias=False),
            Mish(),
            nn.Linear(hidden_channels, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        out = avg_out + max_out
        return self.sigmoid(out).view(x.size(0), -1, 1, 1, 1) * x


class SpatialAttention3D(nn.Module):
    """3D spatial attention module."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat)) * x


class CBAM3D(nn.Module):
    """3D CBAM attention block."""
    def __init__(self, channels):
        super().__init__()
        self.channel_attention = ChannelAttention3D(channels)
        self.spatial_attention = SpatialAttention3D()

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class ConvBlock2D(nn.Module):
    """2D convolution block: Conv + BatchNorm + Mish."""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.mish = Mish()

    def forward(self, x):
        return self.mish(self.bn(self.conv(x)))


class ResBlock2D(nn.Module):
    """2D residual block."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvBlock2D(channels, channels)
        self.conv2 = ConvBlock2D(channels, channels)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class DepthAwareFeatureLift(nn.Module):
    """Lift 2D features into 3D by mixing multiple spatial bases across depth."""

    def __init__(self, channels, depth, num_basis=4):
        super().__init__()
        self.channels = channels
        self.depth = depth
        self.num_basis = num_basis

        pe_dim = min(max(channels // 2, 16), 64)
        hidden_dim = max(channels, num_basis * 8)

        self.depth_encoding = DepthPositionalEncoding(d_model=pe_dim, max_depth=depth)
        self.basis_generator = nn.Sequential(
            nn.Conv2d(channels, channels * num_basis, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels * num_basis),
            Mish()
        )
        self.depth_to_basis = nn.Sequential(
            nn.Linear(pe_dim, hidden_dim),
            Mish(),
            nn.Linear(hidden_dim, num_basis)
        )
        self.context_to_basis = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            Mish(),
            nn.Linear(hidden_dim, depth * num_basis)
        )
        self.depth_scale = nn.Sequential(
            nn.Linear(pe_dim, hidden_dim),
            Mish(),
            nn.Linear(hidden_dim, channels)
        )
        self.refine = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(channels),
            Mish()
        )

    def forward(self, x):
        b, c, h, w = x.shape
        basis = self.basis_generator(x).view(b, self.num_basis, c, h, w)

        depth_idx = torch.arange(self.depth, device=x.device)
        depth_pe = self.depth_encoding(depth_idx).to(dtype=x.dtype)

        static_logits = self.depth_to_basis(depth_pe).unsqueeze(0)
        context = F.adaptive_avg_pool2d(x, output_size=1).flatten(1)
        dynamic_logits = self.context_to_basis(context).view(b, self.depth, self.num_basis)
        basis_weights = torch.softmax(static_logits + dynamic_logits, dim=-1)

        mixed_volume = torch.einsum('bdk,bkchw->bdchw', basis_weights, basis).permute(0, 2, 1, 3, 4)

        # Keep only a small zero-mean residual from the 2D feature map so the lift
        # cannot fall back to copying the same slice through the entire depth axis.
        depth_residual = 0.08 * torch.tanh(self.depth_scale(depth_pe))
        depth_residual = depth_residual - depth_residual.mean(dim=0, keepdim=True)
        depth_residual = depth_residual.transpose(0, 1).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        lifted = mixed_volume + x.unsqueeze(2) * depth_residual
        return lifted + self.refine(lifted)


class SurfaceInputAdapter(nn.Module):
    """Fuse gz and auxiliary surface channels before the shared encoder."""

    def __init__(self, input_channels, hidden_channels=16):
        super().__init__()
        self.input_channels = input_channels

        if input_channels < 2:
            self.gz_branch = None
            self.aux_branch = None
            self.fusion = nn.Identity()
            return

        aux_channels = input_channels - 1
        branch_channels = max(hidden_channels, input_channels)

        self.gz_branch = nn.Sequential(
            ConvBlock2D(1, branch_channels),
            ResBlock2D(branch_channels)
        )
        self.aux_branch = nn.Sequential(
            ConvBlock2D(aux_channels, branch_channels),
            ResBlock2D(branch_channels)
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(branch_channels * 2, input_channels, kernel_size=1),
            nn.BatchNorm2d(input_channels),
            Mish(),
            CBAM2D(input_channels)
        )

    def forward(self, x):
        if self.input_channels < 2:
            return x

        gz = x[:, :1]
        aux = x[:, 1:]
        fused = torch.cat([self.gz_branch(gz), self.aux_branch(aux)], dim=1)
        return x + self.fusion(fused)


class Encoder2D(nn.Module):
    """2D encoder for multiscale feature extraction."""
    def __init__(self, in_channels=2, stage_channels=(32, 64, 128, 256)):
        super().__init__()
        c1, c2, c3, c4 = stage_channels

        # Stage 1: 64x64 -> 32x32.
        self.enc1 = nn.Sequential(
            ConvBlock2D(in_channels, c1),
            ResBlock2D(c1),
            CBAM2D(c1),
            nn.MaxPool2d(2)
        )

        # Stage 2: 32x32 -> 16x16.
        self.enc2 = nn.Sequential(
            ConvBlock2D(c1, c2),
            ResBlock2D(c2),
            CBAM2D(c2),
            nn.MaxPool2d(2)
        )

        # Stage 3: 16x16 -> 8x8.
        self.enc3 = nn.Sequential(
            ConvBlock2D(c2, c3),
            ResBlock2D(c3),
            CBAM2D(c3),
            nn.MaxPool2d(2)
        )

        # Stage 4 keeps the 8x8 resolution while increasing abstraction.
        self.enc4 = nn.Sequential(
            ConvBlock2D(c3, c4),
            ResBlock2D(c4),
            CBAM2D(c4)
        )

    def forward(self, x):
        e1 = self.enc1(x)  # (B, 32, 32, 32)
        e2 = self.enc2(e1)  # (B, 64, 16, 16)
        e3 = self.enc3(e2)  # (B, 128, 8, 8)
        e4 = self.enc4(e3)  # (B, 256, 8, 8)

        return e1, e2, e3, e4


class Bridge(nn.Module):
    """Legacy 2D-to-3D bridge."""
    def __init__(self):
        super().__init__()

        # Fully connected lift from 2D encoder features to a compact 3D volume.
        self.fc = nn.Linear(256 * 8 * 8, 32 * 8 * 8 * 8)
        self.cbam_3d = CBAM3D(32)

    def forward(self, x):
        # x: (B, 256, 8, 8)
        B = x.shape[0]

        x_flat = x.view(B, -1)  # (B, 256*8*8)

        x_expanded = self.fc(x_flat)  # (B, 32*8*8*8)

        # Reshape the lifted vector into a 3D feature volume.
        x_3d = x_expanded.view(B, 32, 8, 8, 8)  # (B, 32, 8, 8, 8)

        x_3d = self.cbam_3d(x_3d)

        return x_3d


class ConvBlock3D(nn.Module):
    """3D convolution block."""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.mish = Mish()

    def forward(self, x):
        return self.mish(self.bn(self.conv(x)))


class ResBlock3D(nn.Module):
    """3D residual block."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvBlock3D(channels, channels)
        self.conv2 = ConvBlock3D(channels, channels)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class AntiCheckerboardUp3D(nn.Module):
    """Upsample 3D features without transposed-convolution checkerboard artifacts."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            Mish()
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)
        return self.project(x)


class Decoder3D(nn.Module):
    """Legacy 3D decoder."""
    def __init__(self):
        super().__init__()

        # Upsample 8x8x8 -> 16x16x16.
        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(32, 64, kernel_size=2, stride=2),
            ConvBlock3D(64, 64),
            ResBlock3D(64),
            CBAM3D(64)
        )

        # Upsample 16x16x16 -> 32x32x32.
        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2),
            ConvBlock3D(32, 32),
            ResBlock3D(32),
            CBAM3D(32)
        )

        # Upsample 32x32x32 -> 64x64x64.
        self.up3 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2),
            ConvBlock3D(16, 16),
            ResBlock3D(16),
            CBAM3D(16)
        )

        self.output = nn.Sequential(
            ConvBlock3D(16, 8),
            ConvBlock3D(8, 1)
        )

        # Auxiliary outputs support deep supervision at intermediate scales.
        self.aux_output_32 = nn.Conv3d(64, 1, 1)
        self.aux_output_64 = nn.Conv3d(32, 1, 1)

    def forward(self, x):
        # x: (B, 32, 8, 8, 8)
        x = self.up1(x)   # (B, 64, 16, 16, 16)
        aux_out_32 = self.aux_output_32(x)

        x = self.up2(x)   # (B, 32, 32, 32, 32)
        aux_out_64 = self.aux_output_64(x)

        x = self.up3(x)   # (B, 16, 64, 64, 64)
        output = self.output(x)  # (B, 1, 64, 64, 64)

        return output, aux_out_32, aux_out_64


class HybridAttentionUNet(nn.Module):
    """Legacy hybrid U-Net."""

    def __init__(self):
        super().__init__()

        # Depth encoding is used by the legacy depth modulation path.
        self.depth_encoding = DepthPositionalEncoding(d_model=64, max_depth=64)

        # Frequency attention is kept for backward compatibility with older checkpoints.
        self.freq_attention = FrequencyDomainAttention(in_channels=1, out_channels=64)

        self.encoder_2d = Encoder2D(in_channels=2)
        self.bridge = Bridge()
        self.decoder_3d = Decoder3D()

    def forward(self, x):
        """
        x: (B, 2, 64, 64) surface input maps.
        Returns a dictionary containing the 3D prediction and auxiliary heads.
        """
        B = x.shape[0]

        e1, e2, e3, e4 = self.encoder_2d(x)

        # Lift the deepest 2D features into 3D.
        x_3d = self.bridge(e4)  # (B, 32, 8, 8, 8)

        output, aux_out_32, aux_out_64 = self.decoder_3d(x_3d)

        # Apply a shallow depth-dependent modulation.
        depth_weights = []
        for z_idx in range(64):
            depth_enc = self.depth_encoding(z_idx)  # (64,)
            depth_weight = 1.0 + 0.1 * torch.tanh(depth_enc.mean())
            depth_weights.append(depth_weight)

        depth_weights = torch.stack(depth_weights).view(1, 1, 64, 1, 1)
        output = output * depth_weights

        return {
            'output': output,
            'aux_32': aux_out_32,
            'aux_64': aux_out_64
        }


class SkipLift2Dto3D(nn.Module):
    """Project 2D features into a shallow 3D volume while preserving XY layout."""

    def __init__(self, in_channels, out_channels, depth, num_basis=4):
        super().__init__()
        self.depth = depth
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            Mish()
        )
        self.depth_lift = DepthAwareFeatureLift(
            channels=out_channels,
            depth=depth,
            num_basis=num_basis
        )

    def forward(self, x):
        x = self.project(x)
        return self.depth_lift(x)


class LocalizationBridge(nn.Module):
    """Lift the bottleneck into 3D without flattening away spatial structure."""

    def __init__(self, in_channels=256, out_channels=64, depth=8, num_basis=4):
        super().__init__()
        self.depth = depth
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            Mish()
        )
        self.depth_lift = DepthAwareFeatureLift(
            channels=out_channels,
            depth=depth,
            num_basis=num_basis
        )
        self.refine = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            Mish(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            Mish(),
            CBAM3D(out_channels)
        )

    def forward(self, x):
        x = self.reduce(x)
        x = self.depth_lift(x)
        return self.refine(x)


class LocalizationDecoder3D(nn.Module):
    """Decoder with explicit skip connections to keep anomaly location cues."""

    def __init__(
        self,
        bottleneck_channels=64,
        skip_channels=(64, 32, 16),
        decoder_channels=(64, 32, 16),
        output_channels=8,
    ):
        super().__init__()
        skip_16_channels, skip_32_channels, skip_64_channels = skip_channels
        dec1_channels, dec2_channels, dec3_channels = decoder_channels

        self.up1 = AntiCheckerboardUp3D(bottleneck_channels, dec1_channels)
        self.dec1 = nn.Sequential(
            ConvBlock3D(dec1_channels + skip_16_channels, dec1_channels),
            ResBlock3D(dec1_channels),
            CBAM3D(dec1_channels)
        )

        self.up2 = AntiCheckerboardUp3D(dec1_channels, dec2_channels)
        self.dec2 = nn.Sequential(
            ConvBlock3D(dec2_channels + skip_32_channels, dec2_channels),
            ResBlock3D(dec2_channels),
            CBAM3D(dec2_channels)
        )

        self.up3 = AntiCheckerboardUp3D(dec2_channels, dec3_channels)
        self.dec3 = nn.Sequential(
            ConvBlock3D(dec3_channels + skip_64_channels, dec3_channels),
            ResBlock3D(dec3_channels),
            CBAM3D(dec3_channels)
        )

        self.output = nn.Sequential(
            ConvBlock3D(dec3_channels, output_channels),
            nn.Conv3d(output_channels, 1, kernel_size=1)
        )
        aux_head_channels = max(output_channels // 2, 4)
        self.body_head = nn.Sequential(
            ConvBlock3D(dec3_channels, aux_head_channels),
            nn.Conv3d(aux_head_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.positive_body_head = nn.Sequential(
            ConvBlock3D(dec3_channels, aux_head_channels),
            nn.Conv3d(aux_head_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.negative_body_head = nn.Sequential(
            ConvBlock3D(dec3_channels, aux_head_channels),
            nn.Conv3d(aux_head_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.center_head = nn.Sequential(
            ConvBlock3D(dec3_channels, aux_head_channels),
            nn.Conv3d(aux_head_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.axis_head = nn.Sequential(
            ConvBlock3D(dec3_channels, aux_head_channels),
            nn.Conv3d(aux_head_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.aux_output_16 = nn.Conv3d(dec1_channels, 1, kernel_size=1)
        self.aux_output_32 = nn.Conv3d(dec2_channels, 1, kernel_size=1)
	
    def forward(self, x, skip_16, skip_32, skip_64):
        x = self.up1(x)
        x = torch.cat([x, skip_16], dim=1)
        x = self.dec1(x)
        aux_out_16 = self.aux_output_16(x)

        x = self.up2(x)
        x = torch.cat([x, skip_32], dim=1)
        x = self.dec2(x)
        aux_out_32 = self.aux_output_32(x)

        x = self.up3(x)
        x = torch.cat([x, skip_64], dim=1)
        x = self.dec3(x)
        output = self.output(x)
        positive_body_mask = self.positive_body_head(x)
        negative_body_mask = self.negative_body_head(x)
        body_mask = torch.clamp(torch.maximum(positive_body_mask, negative_body_mask), 0.0, 1.0)
        body_mask = 0.65 * body_mask + 0.35 * self.body_head(x)
        center_heatmap = self.center_head(x)
        axis_heatmap = self.axis_head(x)

        return output, aux_out_16, aux_out_32, body_mask, center_heatmap, positive_body_mask, negative_body_mask, axis_heatmap


class LocalizationAwareHybridUNet(nn.Module):
    """Localization-first model for recovering anomaly position in vertical slices."""

    def __init__(self, capacity='medium', input_channels=2, input_mode='gz_amp', use_multimodal_stem=None):
        super().__init__()
        if capacity not in LOCALIZATION_MODEL_PRESETS:
            raise ValueError(
                f"Unsupported localization model capacity '{capacity}'. "
                f"Available options: {sorted(LOCALIZATION_MODEL_PRESETS)}"
            )

        config = LOCALIZATION_MODEL_PRESETS[capacity]
        encoder_channels = config['encoder_channels']
        skip_channels = config['skip_channels']
        expected_output_depth = 64
        decoder_upsamples = len(config['decoder_channels'])
        bridge_depth = config['bridge_depth']
        produced_depth = bridge_depth * (2 ** decoder_upsamples)
        if produced_depth != expected_output_depth:
            raise ValueError(
                f"Invalid bridge_depth={bridge_depth} for capacity '{capacity}'. "
                f"With {decoder_upsamples} decoder upsampling stages, the model would produce "
                f"depth {produced_depth} instead of the required {expected_output_depth}."
            )

        if use_multimodal_stem is None:
            use_multimodal_stem = input_channels > 1 and 'gzz' in input_mode

        self.capacity = capacity
        self.input_channels = input_channels
        self.input_mode = input_mode
        self.use_multimodal_stem = use_multimodal_stem
        self.model_config = config
        self.depth_encoding = DepthPositionalEncoding(d_model=64, max_depth=64)
        self.input_adapter = SurfaceInputAdapter(input_channels) if use_multimodal_stem else nn.Identity()
        self.encoder_2d = Encoder2D(in_channels=input_channels, stage_channels=encoder_channels)
        self.bridge = LocalizationBridge(
            in_channels=encoder_channels[-1],
            out_channels=config['bridge_channels'],
            depth=bridge_depth
        )
        self.skip_16 = SkipLift2Dto3D(encoder_channels[1], skip_channels[0], depth=16)
        self.skip_32 = SkipLift2Dto3D(encoder_channels[0], skip_channels[1], depth=32)
        self.skip_64 = SkipLift2Dto3D(input_channels, skip_channels[2], depth=64)
        self.decoder_3d = LocalizationDecoder3D(
            bottleneck_channels=config['bridge_channels'],
            skip_channels=skip_channels,
            decoder_channels=config['decoder_channels'],
            output_channels=config['output_channels']
        )

    def forward(self, x):
        x_adapted = self.input_adapter(x)
        e1, e2, e3, e4 = self.encoder_2d(x_adapted)

        bottleneck_3d = self.bridge(e4)
        skip_16 = self.skip_16(e2)
        skip_32 = self.skip_32(e1)
        skip_64 = self.skip_64(x_adapted)

        output, aux_out_16, aux_out_32, body_mask, center_heatmap, positive_body_mask, negative_body_mask, axis_heatmap = self.decoder_3d(
            bottleneck_3d,
            skip_16,
            skip_32,
            skip_64
        )

        depth_weights = []
        for z_idx in range(64):
            depth_enc = self.depth_encoding(z_idx)
            depth_weight = 1.0 + 0.1 * torch.tanh(depth_enc.mean())
            depth_weights.append(depth_weight)

        depth_weights = torch.stack(depth_weights).view(1, 1, 64, 1, 1)
        output = output * depth_weights

        return {
            'output': output,
            'body_mask': body_mask,
            'positive_body_mask': positive_body_mask,
            'negative_body_mask': negative_body_mask,
            'center_heatmap': center_heatmap,
            'axis_heatmap': axis_heatmap,
            'aux_32': aux_out_16,
            'aux_64': aux_out_32
        }

if __name__ == "__main__":
    print("Testing Hybrid Attention U-Net...")
    model = HybridAttentionUNet()

    # Dummy input: batch_size=2, two surface channels, 64x64 samples.
    x = torch.randn(2, 2, 64, 64)
    output_dict = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Main output shape: {output_dict['output'].shape}")
    print(f"Aux output 1 shape: {output_dict['aux_32'].shape}")
    print(f"Aux output 2 shape: {output_dict['aux_64'].shape}")
    print(f"Parameter count: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    loss = output_dict['output'].sum()
    loss.backward()
    print("Backward pass OK")
