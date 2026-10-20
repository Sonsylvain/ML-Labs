"""The three architectures of Task 2: MLP, CNN and a small patch-attention (ViT-style) model."""

import torch
from torch import nn

N_CLASSES = 10


class MLP(nn.Module):
    """Flatten 784 -> Linear 256 -> ReLU -> Linear 128 -> ReLU -> Linear 10."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, N_CLASSES),
        )

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    """Conv(1,16,3) -> ReLU -> MaxPool2 -> Conv(16,32,3) -> ReLU -> MaxPool2 -> Flatten -> Linear(32*7*7, 10)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 28x28 -> 14x14
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 14x14 -> 7x7
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, N_CLASSES),
        )

    def forward(self, x):
        return self.net(x)


class AttentionNet(nn.Module):
    """16 patches of 7x7 -> Linear 49->64 (+ learned position) -> 2 encoder layers -> mean pool -> Linear 64->10."""

    def __init__(self, patch: int = 7, d_model: int = 64, nhead: int = 4, dim_ff: int = 128, n_layers: int = 2):
        super().__init__()
        self.patch = patch
        self.seq = (28 // patch) ** 2  # 16 tokens
        self.embed = nn.Linear(patch * patch, d_model)
        # Without a position term, mean-pooled self-attention cannot tell where a patch came from.
        self.pos = nn.Parameter(torch.randn(1, self.seq, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=dim_ff, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.head = nn.Linear(d_model, N_CLASSES)

    def patchify(self, x):
        """(B,1,28,28) -> (B,16,49), patches in row-major order."""
        p = self.patch
        x = x.unfold(2, p, p).unfold(3, p, p)  # (B,1,4,4,7,7)
        return x.reshape(x.shape[0], self.seq, p * p)

    def forward(self, x):
        tokens = self.embed(self.patchify(x)) + self.pos
        return self.head(self.encoder(tokens).mean(dim=1))


def build_models() -> dict[str, nn.Module]:
    return {"MLP": MLP(), "CNN": CNN(), "Attention": AttentionNet()}
