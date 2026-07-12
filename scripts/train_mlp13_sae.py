"""Stage 2 of the MLP-13 SAE: train a sparse autoencoder on the layer-13 output.

A change of coordinates could not cleanly separate the rhyme code into vowel and
coda axes (they are non-orthogonal, cos ~0.65, and miss ~22% of the family
subspace), so we learn an overcomplete, sparse dictionary instead. Trained on the
captured lexicon residuals; features are analysed for phoneme monosemanticity in
stage 3.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SAE(nn.Module):
    def __init__(self, d, m):
        super().__init__()
        self.b_dec = nn.Parameter(torch.zeros(d))
        W = F.normalize(torch.randn(m, d), dim=1)  # unit-norm feature directions
        self.W_dec = nn.Parameter(W.clone())
        self.W_enc = nn.Parameter(W.clone().t())   # (d, m)
        self.b_enc = nn.Parameter(torch.zeros(m))

    def encode(self, x):
        return F.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

    def forward(self, x):
        h = self.encode(x)
        return h @ self.W_dec + self.b_dec, h

    @torch.no_grad()
    def normalize_decoder(self):
        self.W_dec.data = F.normalize(self.W_dec.data, dim=1)


def run(args):
    data = np.load(args.input, allow_pickle=True)
    X = torch.tensor(data["activations"].astype(np.float32))
    X = X - X.mean(0)
    X = X / X.norm(dim=1).mean()  # scale so mean row norm ~ 1
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X = X.to(dev)
    d, m = X.shape[1], args.dict
    sae = SAE(d, m).to(dev)
    opt = torch.optim.Adam(sae.parameters(), lr=args.lr)
    print(f"training SAE {d}->{m} on {X.shape[0]} activations ({dev}), lambda={args.l1}")

    n = X.shape[0]
    fired = torch.zeros(m, dtype=torch.bool, device=dev)
    for step in range(args.steps):
        idx = torch.randint(0, n, (args.batch,), device=dev)
        x = X[idx]
        xhat, h = sae(x)
        recon = (xhat - x).pow(2).sum(1).mean()   # sum over dims: O(1) for unit vectors
        l1 = h.abs().sum(1).mean()
        loss = recon + args.l1 * l1
        opt.zero_grad(); loss.backward(); opt.step()
        sae.normalize_decoder()
        fired |= (h > 1e-6).any(0)

        if step and step % args.resample == 0:  # revive dead features on high-residual points
            with torch.no_grad():
                dead = torch.where(~fired)[0]
                if len(dead):
                    xr, hr = sae(X)
                    err = (X - xr).norm(dim=1)
                    pick = torch.multinomial(err / err.sum(), min(len(dead), n), replacement=False)
                    dirs = F.normalize(X[pick] - sae.b_dec, dim=1)
                    k = min(len(dead), len(pick))
                    sae.W_dec.data[dead[:k]] = dirs[:k]
                    sae.W_enc.data[:, dead[:k]] = dirs[:k].t()
                    sae.b_enc.data[dead[:k]] = 0.0
            fired[:] = False

        if step % 1000 == 0 or step == args.steps - 1:
            with torch.no_grad():
                xhat, h = sae(X)
                var = 1 - (X - xhat).pow(2).sum() / X.pow(2).sum()
                l0 = (h > 1e-6).float().sum(1).mean()
                dead = (h.max(0).values <= 1e-6).sum()
            print(f"  step {step:5d}: recon var {var:.3f}  L0 {l0:.1f}  dead {int(dead)}/{m}")

    with torch.no_grad():
        H = sae.encode(X).cpu().numpy().astype(np.float16)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, features=H, W_dec=sae.W_dec.detach().cpu().numpy(),
                        words=data["words"], labels=data["labels"])
    print(f"wrote {args.output}: features {H.shape}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=Path("artifacts/gemma4_mlp_rhyme/lexicon_l13.npz"))
    p.add_argument("--output", type=Path, default=Path("artifacts/gemma4_mlp_rhyme/sae_l13.npz"))
    p.add_argument("--dict", type=int, default=2048)
    p.add_argument("--l1", type=float, default=5e-4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--steps", type=int, default=12000)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--resample", type=int, default=2500)
    run(p.parse_args())


if __name__ == "__main__":
    main()
