"""
model.py -- The shared neural network architecture trained by every edge
node. All nodes MUST use an identical architecture: the aggregator treats
weights as an opaque flat float vector and averages them element-wise, so
any mismatch in shape or parameter ordering would silently corrupt FedAvg.

Trained task: 10-class digit classification on the scikit-learn "digits"
dataset (8x8 grayscale images, 64 input features). This dataset ships
inside scikit-learn itself, so every edge node can load its local shard
with no network access -- mirroring how a real edge device would already
hold its own private data on disk.
"""
import torch
import torch.nn as nn


class DigitClassifier(nn.Module):
    """Small MLP: 64 -> 32 -> 10. ~2,400 parameters total."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        )

    def forward(self, x):
        return self.net(x)


def model_size(model: nn.Module) -> int:
    """Total number of scalar parameters, i.e. the length of the flat
    weight vector this node will exchange with the aggregator."""
    return sum(p.numel() for p in model.parameters())


def flatten_weights(model: nn.Module) -> list[float]:
    """Serialize all parameters into a single flat list, in a fixed,
    deterministic order (state_dict order is stable given the same
    architecture and PyTorch version)."""
    flat = []
    for p in model.parameters():
        flat.extend(p.detach().cpu().reshape(-1).tolist())
    return flat


def load_weights(model: nn.Module, flat: list[float]) -> None:
    """Inverse of flatten_weights: unpack a flat vector back into the
    model's parameter tensors, in place."""
    idx = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            chunk = torch.tensor(flat[idx: idx + n], dtype=p.dtype).reshape(p.shape)
            p.copy_(chunk)
            idx += n
    assert idx == len(flat), "flat vector length does not match model size"
