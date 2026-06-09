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

    Args:
        n_output_channels: number of output channels (default 4: mut_to_A/C/G/T)
        use_reverse: if True, add reverse-complement symmetric conv on input
                     and AT/CG embedding concatenated to final layer.
    """

    def __init__(self, n_output_channels=4, use_reverse=True):
        super().__init__()
        self.use_reverse = use_reverse

        embedding_dim = 4 if use_reverse else 0
        if use_reverse:
            self.conv = nn.Sequential(
                nn.Conv1d(4, 4, kernel_size=7, padding=3),
                nn.BatchNorm1d(4),
            )
            self.register_buffer(
                'collapse_map', torch.tensor([0, 1, 1, 0], dtype=torch.float)
            )
            self.ref_embedding = nn.Embedding(
                num_embeddings=2, embedding_dim=embedding_dim
            )

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

        final_dim = 64 + embedding_dim
        self.final = nn.Sequential(
            nn.Conv1d(final_dim, final_dim, kernel_size=1),
            nn.BatchNorm1d(final_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(final_dim, n_output_channels, kernel_size=1),
            nn.Softplus(),
        )

    def forward(self, x):
        if self.use_reverse:
            ref_class_idx = torch.einsum('bcl,c->bl', x, self.collapse_map).long()
            ref_emb = self.ref_embedding(ref_class_idx).permute(0, 2, 1)
            out = torch.add(self.conv(x), self.conv(x.flip([1, 2])).flip([2]))
        else:
            out = x

        # Encoder 1
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

        if self.use_reverse:
            out = torch.cat([out, ref_emb], dim=1)

        return self.final(out)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total
