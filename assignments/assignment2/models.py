"""Novel DCNN architecture — CompactResNet.

Design
------
  Input  : 32x32x3
  Output : configurable num_classes (200 for Tiny ImageNet-200, 100 for CIFAR-100)

  Stem -> Stage1(64) -> Stage2(128, stride2) -> Stage3(256, stride2)
       -> Stage4(512, stride2) -> GlobalAvgPool -> Dropout(0.5) -> Linear(num_classes)

Key features
  - Residual bottleneck blocks (1x1 -> 3x3 -> 1x1) for parameter efficiency
  - 1x1 convolutions as bottleneck layers
  - Batch Normalization after every convolution
  - Global Average Pooling (GAP) replacing large fully-connected layers
  - Dropout before the classification head
  - He (Kaiming) initialization applied inside __init__

Spatial dimensions at 32x32 input
  After stem  : 32x32x64
  After stage1: 32x32x64
  After stage2: 16x16x128
  After stage3:  8x8x256
  After stage4:  4x4x512
  After GAP   :  512-d vector
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Activation helpers
# ---------------------------------------------------------------------------

def get_activation(name: str) -> nn.Module:
    """Instantiate an activation module by name."""
    if name == 'relu':
        return nn.ReLU(inplace=True)
    elif name == 'gelu':
        return nn.GELU()
    elif name == 'rswish':
        return RSwish()
    else:
        raise ValueError(f'Unknown activation: {name}')


class RSwish(nn.Module):
    """Rational Swish activation function.

    Replaces the exponential sigmoid in Swish with an algebraic softsign
    approximation, avoiding transcendental operations entirely:

        RSwish(x) = x * (1 + x / (1 + |x|)) / 2

    The factor (1 + x/(1+|x|)) / 2 is a rational approximation of sigmoid
    that maps (-inf, +inf) -> (0, 1), matching sigmoid's range without any
    exponentials.
    """

    def forward(self, x):
        rational_sigmoid = (1 + x / (1 + x.abs())) / 2
        return x * rational_sigmoid



# ---------------------------------------------------------------------------
# Bottleneck residual block
# ---------------------------------------------------------------------------

class BottleneckBlock(nn.Module):
    """Residual bottleneck: 1x1 -> 3x3 -> 1x1 convolutions.

    The intermediate (bottleneck) width is out_channels // 4, so the
    expensive 3x3 convolution operates on a compressed representation.

    A projection shortcut (1x1 conv + BN) is added whenever the number of
    channels or the spatial resolution changes.
    """

    expansion: int = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str,
        stride: int = 1,
    ):
        super().__init__()
        mid = out_channels // self.expansion

        # 1x1 bottleneck — compress channels
        self.conv1 = nn.Conv2d(in_channels, mid, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(mid)
        self.act1  = get_activation(activation)

        # 3x3 convolution — spatial feature extraction
        self.conv2 = nn.Conv2d(mid, mid, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(mid)
        self.act2  = get_activation(activation)

        # 1x1 bottleneck — expand channels back
        self.conv3 = nn.Conv2d(mid, out_channels, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(out_channels)

        # Output activation applied after residual addition
        self.act_out = get_activation(activation)

        # Projection shortcut — align dimensions when they change
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.shortcut is None else self.shortcut(x)

        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        return self.act_out(out + residual)


# ---------------------------------------------------------------------------
# Full network
# ---------------------------------------------------------------------------

class CompactNet(nn.Module):
    """Compact residual network for small-image classification.

    He (Kaiming) initialisation is applied to all Conv2d and Linear layers
    inside _init_weights(), which is called at the end of __init__.

    The classification head is exposed as self.classifier (an nn.Linear) so
    that finetune.py can swap it when adapting from Tiny ImageNet (200 classes)
    to CIFAR-100 (100 classes).
    """

    def __init__(self, num_classes: int = 200, activation: str = 'relu'):
        super().__init__()

        # --- Stem: 32x32x3  ->  32x32x64 ---
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            get_activation(activation),
        )

        # --- Stage 1: 32x32x64  ->  32x32x64 ---
        self.stage1 = nn.Sequential(
            BottleneckBlock(64,  64,  activation, stride=1),
            BottleneckBlock(64,  64,  activation, stride=1),
        )

        # --- Stage 2: 32x32x64  ->  16x16x128 ---
        self.stage2 = nn.Sequential(
            BottleneckBlock(64,  128, activation, stride=2),
            BottleneckBlock(128, 128, activation, stride=1),
        )

        # --- Stage 3: 16x16x128  ->  8x8x256 ---
        self.stage3 = nn.Sequential(
            BottleneckBlock(128, 256, activation, stride=2),
            BottleneckBlock(256, 256, activation, stride=1),
            BottleneckBlock(256, 256, activation, stride=1),
        )

        # --- Stage 4: 8x8x256  ->  4x4x512 ---
        self.stage4 = nn.Sequential(
            BottleneckBlock(256, 512, activation, stride=2),
            BottleneckBlock(512, 512, activation, stride=1),
        )

        # --- Classification head ---
        self.gap        = nn.AdaptiveAvgPool2d(1)
        self.dropout    = nn.Dropout(p=0.5)
        self.classifier = nn.Linear(512, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        """He (Kaiming) initialisation for all Conv2d and Linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x)
        x = x.flatten(start_dim=1)   # (B, 512)
        x = self.dropout(x)
        return self.classifier(x)    # (B, num_classes)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_model(num_classes: int = 200, activation: str = 'relu') -> CompactNet:
    """Instantiate a He-initialised CompactNet."""
    return CompactNet(num_classes=num_classes, activation=activation)
