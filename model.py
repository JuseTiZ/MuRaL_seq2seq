import torch
from torch import nn


class ConvBlock(nn.Module):
    """Inverted residual block: expand → depthwise → project, with residual add."""

    def __init__(self, inp, oup, expand_ratio=2):
        super().__init__()
        hidden_dim = round(inp * expand_ratio)
        self.conv = nn.Sequential(
            nn.Conv1d(inp, hidden_dim, 9, 1, padding=4, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(inplace=False),
            nn.Conv1d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm1d(oup),
        )

    def forward(self, x):
        return x + self.conv(x)


class PuffinD(nn.Module):
    """
    PuffinD U-Net for seq2seq mutation rate prediction.

    Input:  (B, 4, L) — one-hot DNA sequence
    Output: (B, n_output_channels, L) — per-position mutation probabilities
    """

    def __init__(self, n_output_channels=4):
        super().__init__()

        # --- Encoder path 1 ---
        self.uplblocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(4, 64, kernel_size=7, padding=3),
                nn.BatchNorm1d(64),
            ),
            nn.Sequential(
                nn.Conv1d(64, 96, stride=4, kernel_size=7, padding=3),
                nn.BatchNorm1d(96),
            ),
            nn.Sequential(
                nn.Conv1d(96, 128, stride=4, kernel_size=7, padding=3),
                nn.BatchNorm1d(128),
            ),
            nn.Sequential(
                nn.Conv1d(128, 128, stride=5, kernel_size=7, padding=3),
                nn.BatchNorm1d(128),
            ),
            nn.Sequential(
                nn.Conv1d(128, 128, stride=5, kernel_size=7, padding=3),
                nn.BatchNorm1d(128),
            ),
            nn.Sequential(
                nn.Conv1d(128, 128, stride=1, kernel_size=7, padding=3),
                nn.BatchNorm1d(128),
            ),
        ])

        self.upblocks = nn.ModuleList([
            nn.Sequential(ConvBlock(64, 64), ConvBlock(64, 64)),
            nn.Sequential(ConvBlock(96, 96), ConvBlock(96, 96)),
            nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128)),
            nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128)),
            nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128)),
            nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128)),
        ])

        # --- Decoder path 1 ---
        self.downlblocks = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(scale_factor=1),
                nn.Conv1d(128, 128, kernel_size=7, padding=3),
                nn.BatchNorm1d(128),
            ),
            nn.Sequential(
                nn.Upsample(scale_factor=5),
                nn.Conv1d(128, 128, kernel_size=7, padding=3),
                nn.BatchNorm1d(128),
            ),
            nn.Sequential(
                nn.Upsample(scale_factor=5),
                nn.Conv1d(128, 128, kernel_size=7, padding=3),
                nn.BatchNorm1d(128),
            ),
            nn.Sequential(
                nn.Upsample(scale_factor=4),
                nn.Conv1d(128, 96, kernel_size=7, padding=3),
                nn.BatchNorm1d(96),
            ),
            nn.Sequential(
                nn.Upsample(scale_factor=4),
                nn.Conv1d(96, 64, kernel_size=7, padding=3),
                nn.BatchNorm1d(64),
            ),
        ])

        self.downblocks = nn.ModuleList([
            nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128)),
            nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128)),
            nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128)),
            nn.Sequential(ConvBlock(96, 96), ConvBlock(96, 96)),
            nn.Sequential(ConvBlock(64, 64), ConvBlock(64, 64)),
        ])

        # --- Encoder path 2 ---
        self.uplblocks2 = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(64, 96, stride=4, kernel_size=7, padding=3),
                nn.BatchNorm1d(96),
            ),
            nn.Sequential(
                nn.Conv1d(96, 128, stride=4, kernel_size=7, padding=3),
                nn.BatchNorm1d(128),
            ),
            nn.Sequential(
                nn.Conv1d(128, 128, stride=5, kernel_size=7, padding=3),
                nn.BatchNorm1d(128),
            ),
            nn.Sequential(
                nn.Conv1d(128, 128, stride=5, kernel_size=7, padding=3),
                nn.BatchNorm1d(128),
            ),
            nn.Sequential(
                nn.Conv1d(128, 128, stride=1, kernel_size=7, padding=3),
                nn.BatchNorm1d(128),
            ),
        ])

        self.upblocks2 = nn.ModuleList([
            nn.Sequential(ConvBlock(96, 96), ConvBlock(96, 96)),
            nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128)),
            nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128)),
            nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128)),
            nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128)),
        ])

        # --- Decoder path 2 ---
        self.downlblocks2 = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(scale_factor=1),
                nn.Conv1d(128, 128, kernel_size=7, padding=3),
                nn.BatchNorm1d(128),
            ),
            nn.Sequential(
                nn.Upsample(scale_factor=5),
                nn.Conv1d(128, 128, kernel_size=7, padding=3),
                nn.BatchNorm1d(128),
            ),
            nn.Sequential(
                nn.Upsample(scale_factor=5),
                nn.Conv1d(128, 128, kernel_size=7, padding=3),
                nn.BatchNorm1d(128),
            ),
            nn.Sequential(
                nn.Upsample(scale_factor=4),
                nn.Conv1d(128, 96, kernel_size=7, padding=3),
                nn.BatchNorm1d(96),
            ),
            nn.Sequential(
                nn.Upsample(scale_factor=4),
                nn.Conv1d(96, 64, kernel_size=7, padding=3),
                nn.BatchNorm1d(64),
            ),
        ])

        self.downblocks2 = nn.ModuleList([
            nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128)),
            nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128)),
            nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128)),
            nn.Sequential(ConvBlock(96, 96), ConvBlock(96, 96)),
            nn.Sequential(ConvBlock(64, 64), ConvBlock(64, 64)),
        ])

        self.final = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, n_output_channels, kernel_size=1),
            nn.Softplus(),
        )

    def forward(self, x):
        # Encoder 1
        out = x
        encodings = []
        for lconv, conv in zip(self.uplblocks, self.upblocks):
            lout = lconv(out)
            out = conv(lout)
            encodings.append(out)

        # Decoder 1 with skip connections
        encodings2 = [out]
        for enc, lconv, conv in zip(reversed(encodings[:-1]), self.downlblocks, self.downblocks):
            lout = lconv(out)
            out = conv(lout)
            out = enc + out
            encodings2.append(out)

        # Encoder 2 with skip connections from decoder 1
        encodings3 = [out]
        for enc, lconv, conv in zip(reversed(encodings2[:-1]), self.uplblocks2, self.upblocks2):
            lout = lconv(out)
            out = conv(lout)
            out = enc + out
            encodings3.append(out)

        # Decoder 2 with skip connections from encoder 2
        for enc, lconv, conv in zip(reversed(encodings3[:-1]), self.downlblocks2, self.downblocks2):
            lout = lconv(out)
            out = conv(lout)
            out = enc + out

        return self.final(out)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total
