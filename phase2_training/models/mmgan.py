"""
MM-GAN Model Architecture adapted for IXI Dataset (3 modalities: T1, T2, PD)
Based on: https://github.com/trane293/mm-gan

Key changes from original:
  - 3 input/output channels (T1, T2, PD) instead of 4 (BRATS)
  - Discriminator adapted for 3-modality setup
  - Added ReLU final activation (IXI data is [0,1] normalized)
"""

import torch
import torch.nn as nn
import numpy as np
import random


def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def weights_init_normal(m):
    """Initialize weights using normal distribution."""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


# ============================================================
#                        U-NET Generator
# ============================================================


class UNetDown(nn.Module):
    """Encoder block: Conv -> [InstanceNorm] -> LeakyReLU -> [Dropout]"""

    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    """Decoder block: ConvTranspose -> InstanceNorm -> ReLU -> [Dropout] + skip"""

    def __init__(self, in_size, out_size, dropout=0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x


class GeneratorUNet(nn.Module):
    """
    UNet Generator with skip connections.

    For IXI: in_channels=3 (T1, T2, PD), out_channels=3
    Input size: (B, 3, 256, 256) -> Output: (B, 3, 256, 256)
    """

    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        # Encoder
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.2)
        self.down5 = UNetDown(512, 512, dropout=0.2)
        self.down6 = UNetDown(512, 512, dropout=0.2)
        self.down7 = UNetDown(512, 512, dropout=0.2)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.2)

        # Decoder
        self.up1 = UNetUp(512, 512, dropout=0.2)
        self.up2 = UNetUp(1024, 512, dropout=0.2)
        self.up3 = UNetUp(1024, 512, dropout=0.2)
        self.up4 = UNetUp(1024, 512, dropout=0.2)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        # Final layer with ReLU (output in [0, inf), data is [0,1] normalized)
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        # Encoder path
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)  # Bottleneck

        # Decoder path with skip connections
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)


# ============================================================
#                    PatchGAN Discriminator
# ============================================================


class Discriminator(nn.Module):
    """
    PatchGAN Discriminator.

    For IXI: in_channels=3 (receives concat of real/fake + condition = 6 channels)
    Output: (B, out_channels, H/16, W/16) patch predictions
    """

    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Downsampling block: Conv -> [InstanceNorm] -> LeakyReLU"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # Input: (in_channels * 2) because we concat img_A and img_B
        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, out_channels, 4, padding=1, bias=False),
        )

    def forward(self, img_A, img_B):
        """
        Args:
            img_A: Generated/real image (B, C, H, W)
            img_B: Condition image (B, C, H, W)
        Returns:
            Patch predictions (B, out_channels, H/16, W/16)
        """
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)


# ============================================================
#                    Missing Modality Logic
# ============================================================

# All possible missing modality scenarios for 3 modalities (T1, T2, PD)
# 0 = missing, 1 = available
# Sorted by difficulty: fewer available = harder (first)
ALL_SCENARIOS_3MOD = [
    [1, 0, 0],  # Only T1 available       (hardest: 2 missing)
    [0, 1, 0],  # Only T2 available
    [0, 0, 1],  # Only PD available
    [1, 1, 0],  # T1+T2 available          (medium: 1 missing)
    [1, 0, 1],  # T1+PD available
    [0, 1, 1],  # T2+PD available          (easiest: 1 missing)
]

# Modality names (indexed)
MODALITY_NAMES = ["T1", "T2", "PD"]


def get_curriculum_scenarios(epoch, total_epochs):
    """
    Curriculum learning strategy for 3-modality setup.
    Starts with easy scenarios (1 missing) and gradually adds harder ones.

    Returns: (low_idx, high_idx) range into ALL_SCENARIOS_3MOD
    """
    progress = epoch / max(total_epochs, 1)

    if progress <= 0.3:
        # First 30%: easy scenarios only (1 missing modality)
        return 3, 6
    elif progress <= 0.7:
        # 30-70%: all scenarios
        return 0, 6
    else:
        # 70%+: all scenarios (full difficulty)
        return 0, 6


def impute_missing(x_real, scenario, impute_type="zeros"):
    """
    Replace missing modality channels with imputation values.

    Args:
        x_real: (B, C, H, W) tensor of real images
        scenario: list of 0/1 indicating missing/available
        impute_type: 'zeros', 'noise', or 'average'

    Returns:
        x_imputed: tensor with missing channels replaced
    """
    x_imputed = x_real.clone()
    B, C, H, W = x_imputed.shape

    if impute_type == "average":
        avail_idx = [i for i, s in enumerate(scenario) if s == 1]
        if avail_idx:
            avg = torch.mean(x_real[:, avail_idx, ...], dim=1)
        else:
            avg = torch.zeros(B, H, W, device=x_real.device)

    for idx, available in enumerate(scenario):
        if available == 0:
            if impute_type == "zeros":
                x_imputed[:, idx, ...] = 0.0
            elif impute_type == "noise":
                x_imputed[:, idx, ...] = torch.randn(B, H, W, device=x_real.device)
            elif impute_type == "average":
                x_imputed[:, idx, ...] = avg

    return x_imputed


def impute_reals_into_fake(x_input, fake_x, scenario):
    """
    Implicit conditioning: copy real (available) modalities back into
    the generator output so loss is only on synthesized channels.

    Args:
        x_input: (B, C, H, W) original input with real available channels
        fake_x: (B, C, H, W) generator output
        scenario: list of 0/1 indicating missing/available
    Returns:
        fake_x with available channels replaced by real values
    """
    result = fake_x.clone()
    for idx, available in enumerate(scenario):
        if available == 1:
            result[:, idx, ...] = x_input[:, idx, ...].clone()
    return result


def compute_missing_loss(fake_x, real_x, scenario, loss_fn):
    """
    Compute loss ONLY on missing modality channels (for implicit conditioning).

    Args:
        fake_x: Generator output (B, C, H, W)
        real_x: Ground truth (B, C, H, W)
        scenario: list of 0/1
        loss_fn: loss function (e.g., nn.L1Loss())

    Returns:
        Loss value averaged over missing channels
    """
    losses = []
    for idx, available in enumerate(scenario):
        if available == 0:
            losses.append(loss_fn(fake_x[:, idx, ...], real_x[:, idx, ...]))

    if losses:
        return sum(losses) / len(losses)
    return torch.tensor(0.0, device=fake_x.device)
