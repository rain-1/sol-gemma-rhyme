"""Decisive SAE test: does rhyme factor into monosemantic phoneme features once
lexical identity is projected out?

A vanilla SAE on the layer-13 residual learns word-stem features (train_mlp13_sae
+ analyze), because lexical identity dominates the residual. Here we first project
the residuals onto the phoneme subspace (spanned by vowel-mean and coda-mean
directions), removing most lexical identity, then train an overcomplete SAE on
that and check whether features become cross-onset coda / vowel detectors.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def parse(label):
    if not label:
        return None, None
    p = label.split("-")
    return p[0], ("-".join(p[1:]) or "OPEN")


class SAE(nn.Module):
    def __init__(self, d, m):
        super().__init__()
        self.b_dec = nn.Parameter(torch.zeros(d))
        W = F.normalize(torch.randn(m, d), dim=1)
        self.W_dec = nn.Parameter(W.clone()); self.W_enc = nn.Parameter(W.clone().t())
        self.b_enc = nn.Parameter(torch.zeros(m))

    def encode(self, x):
        return F.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

    def forward(self, x):
        h = self.encode(x)
        return h @ self.W_dec + self.b_dec, h

    @torch.no_grad()
    def norm_dec(self):
        self.W_dec.data = F.normalize(self.W_dec.data, dim=1)


def run(args):
    data = np.load(args.input, allow_pickle=True)
    H = data["activations"].astype(np.float32)
    words = [str(w) for w in data["words"]]
    labels = [str(l) for l in data["labels"]]
    vowel = [parse(l)[0] for l in labels]
    coda = [parse(l)[1] for l in labels]
    Hc = H - H.mean(0)

    # phoneme subspace: span of vowel-mean and coda-mean directions (>= 20 words each)
    def means(attr):
        out = []
        for g, c in Counter([a for a in attr if a]).items():
            if c >= 20:
                out.append(Hc[np.array([a == g for a in attr])].mean(0))
        return np.stack(out)
    M = np.concatenate([means(vowel), means(coda)])
    M = M - M.mean(0)
    Q, _ = np.linalg.qr(M.T)          # (1536, rank)
    Z = Hc @ Q                        # phoneme coordinates
    print(f"phoneme subspace rank {Q.shape[1]}; projected {Z.shape}")

    Zt = torch.tensor(Z); Zt = Zt / Zt.norm(dim=1).mean()
    d, m = Zt.shape[1], args.dict
    sae = SAE(d, m); opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
    for step in range(args.steps):
        idx = torch.randint(0, len(Zt), (512,))
        x = Zt[idx]; xhat, h = sae(x)
        loss = (xhat - x).pow(2).sum(1).mean() + args.l1 * h.abs().sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sae.norm_dec()
    with torch.no_grad():
        xhat, Hf = sae(Zt); Hf = Hf.numpy()
        var = 1 - (Zt - xhat).pow(2).sum().item() / Zt.pow(2).sum().item()
    print(f"SAE {d}->{m}: recon var {var:.2f}, mean L0 {(Hf>1e-6).sum(1).mean():.1f}\n")

    # analyse: monosemantic AND cross-onset features
    def report(attr, name):
        hits = []
        for j in np.where(Hf.max(0) > 1e-6)[0]:
            top = np.argsort(-Hf[:, j])[:15]
            top = [i for i in top if Hf[i, j] > 1e-6 and attr[i]]
            if len(top) < 6:
                continue
            vals = [attr[i] for i in top]
            g, cnt = Counter(vals).most_common(1)[0]
            onsets = {words[i][:2] for i in top}
            if cnt / len(top) >= 0.7 and len(onsets) >= 5:   # pure AND cross-onset
                hits.append((j, g, cnt / len(top), len(onsets), [words[i] for i in top[:6]]))
        hits.sort(key=lambda r: (-r[2], -r[3]))
        seen = set()
        print(f"{name} features (>=70% share {name}, >=5 distinct onsets): "
              f"{len({h[1] for h in hits})} distinct {name}s across {len(hits)} features")
        for j, g, p, no, ex in hits:
            if g in seen:
                continue
            seen.add(g)
            print(f"  feat {j:4d}  {name} {g:6s} {p:.0%}  {no} onsets  e.g. {', '.join(ex)}")

    report(coda, "coda")
    print()
    report(vowel, "vowel")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=Path("artifacts/gemma4_mlp_rhyme/lexicon_l13.npz"))
    p.add_argument("--dict", type=int, default=256)
    p.add_argument("--l1", type=float, default=6e-3)
    p.add_argument("--steps", type=int, default=6000)
    run(p.parse_args())


if __name__ == "__main__":
    main()
