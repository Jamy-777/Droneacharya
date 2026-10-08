"""A small CNN on RF tile spectrograms, trained on the GPU.

Fixed in advance (no tuning on test data): log-power STFT (64-point Hann, hop 32) of each
complex tile, standardised per tile (removes received power); three conv blocks and global
average pooling, so the decision cannot depend on where in frequency a signal sits; random
circular frequency shifts while training (frequency-agnostic, the 5.8 GHz path); class-balanced
loss; 15 epochs, AdamW 1e-3, batch 256, fixed seed. Device: CUDA when available.
"""
import numpy as np
import torch
from torch import nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS, BATCH, LR, SEED = 15, 256, 1e-3, 0


def spectrograms(tiles, batch=4096):
    """(n, samples) complex64 -> (n, 1, 64, frames) float32 log-power, standardised per tile; computed on the GPU."""
    out = []
    window = torch.hann_window(64, device=DEVICE)
    for i in range(0, len(tiles), batch):
        x = torch.as_tensor(np.ascontiguousarray(tiles[i:i + batch]), device=DEVICE)
        s = torch.stft(x, n_fft=64, hop_length=32, window=window, return_complex=True, onesided=False)
        p = torch.log10(s.abs() ** 2 + 1e-12)
        p = torch.fft.fftshift(p, dim=1)
        p = (p - p.mean(dim=(1, 2), keepdim=True)) / (p.std(dim=(1, 2), keepdim=True) + 1e-6)
        out.append(p.unsqueeze(1).cpu())
    return torch.cat(out)


class TileNet(nn.Module):
    def __init__(self):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(), nn.MaxPool2d(2))

        self.features = nn.Sequential(block(1, 16), block(16, 32), block(32, 64))
        self.head = nn.Linear(64, 1)

    def forward(self, x):
        return self.head(self.features(x).mean(dim=(2, 3))).squeeze(1)


def train(specs, labels):
    """Fit TileNet on (n, 1, 64, frames) spectrograms; returns the model (fixed recipe, no validation-based stopping)."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = TileNet().to(DEVICE)
    y = torch.as_tensor(np.asarray(labels, dtype=np.float32))
    pos_weight = torch.tensor([(len(y) - y.sum()) / max(y.sum(), 1)], device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    generator = torch.Generator().manual_seed(SEED)
    for _ in range(EPOCHS):
        model.train()
        order = torch.randperm(len(y), generator=generator)
        for i in range(0, len(y), BATCH):
            idx = order[i:i + BATCH]
            xb = specs[idx].to(DEVICE)
            shift = int(torch.randint(0, xb.shape[2], (1,), generator=generator))
            xb = torch.roll(xb, shifts=shift, dims=2)          # random circular frequency shift
            loss = loss_fn(model(xb), y[idx].to(DEVICE))
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def predict(model, specs, batch=2048):
    model.eval()
    return torch.cat([torch.sigmoid(model(specs[i:i + batch].to(DEVICE))).cpu()
                      for i in range(0, len(specs), batch)]).numpy()
