# Inside the layer-13 MLP: vowel and coda neurons

Report 07 localized the rhyme write to the layer-13 MLP, and a rank-1 edit to its
down-projection proved one direction there is load-bearing (report 08 §6). This
report opens that MLP and asks what the direction is made of, at the level of its
6144 neurons (the units feeding the down-projection). The answer has three parts,
and the third one complicates the tidy story in a useful way.

`google/gemma-4-E2B`, eager attention, NF4. Neuron activations captured at the
anchor for 356 single-token words in 30 families; probes are 5-fold linear,
chance 0.033. Reproduce with `scripts/run_gemma4_mlp13_neurons.py`.

## Main conclusions

- **The rhyme family is sparsely *readable*.** Sixteen neurons carry it: a probe
  on the 16 most family-selective neurons reads the family at **0.79**, and more
  neurons do not help (using all 6144 falls to 0.42 as noise dilutes the probe).
- **But the code is densely, redundantly *written*.** Zeroing those top 16
  neurons at the anchor removes only **6%** of the rhyme prediction (top 64: also
  6%; random neurons: 0%), against **73%** for ablating the whole MLP. No small
  set of neurons is a causal bottleneck.
- **The neurons are compositional.** Individual neurons tune to specific
  phonemes — some to a coda (neuron 4326 → *-T*, 1375 → *-D*), some to a vowel
  (2256 → *OW*, 2493 → *UW*) — and the most rhyme-selective neurons split about
  evenly into a vowel population and a coda population.

Together: the MLP-13 rhyme code is **sparsely readable, densely written, and
organized along a vowel axis and a coda axis** — a distributed but structured
representation, not a handful of grandmother neurons.

## 1. Sixteen neurons read the family

Ranking neurons by how strongly their activation varies across families (ANOVA
F), a probe on just the top few already reads the 30-way family well:

| Neurons used | Family readout |
|---|---:|
| top 4 | 0.17 |
| top 8 | 0.67 |
| **top 16** | **0.79** |
| top 32 | 0.75 |
| top 128 | 0.72 |
| all 6144 | 0.42 |

The family is concentrated: sixteen neurons out of 6144 suffice, and piling in
the rest *lowers* the probe as their noise dominates (this last number is partly
a probe-dilution artifact of 6144 dimensions over 356 words, not only a claim
about signal). Either way, the rhyme family is a low-dimensional, sparsely
readable feature of this MLP.

## 2. But no small set of neurons is causal

Sparse *readability* is not the same as sparse *causal importance*. Zeroing the
top family-selective neurons at the anchor and measuring the model's probability
on the anchor's rhyme family:

| Ablated at anchor | Rhyme-family mass | Drop |
|---|---:|---:|
| baseline | 0.786 | — |
| top 16 selective neurons | 0.742 | 6% |
| top 64 selective neurons | 0.735 | 6% |
| 16–64 random neurons | ~0.786 | 0% |
| whole layer-13 MLP (report 07) | 0.211 | **73%** |

The top neurons matter more than random ones — so they are causally involved —
but removing even 64 of them costs only 6%, while the whole MLP is worth 73%. The
write is therefore **spread redundantly across many neurons**, each contributing
a little; the sixteen that best *read out* the family are not the sixteen that
would *break* it. This is a clean case of decodability ≠ causal necessity, and it
means the rank-1 weight edit of report 08 works by adding a new direction, not by
hijacking a critical neuron.

## 3. Vowel neurons and coda neurons

A rhyme family is a (vowel, coda) pair, and the neurons respect that split.
Ranking by coda-selectivity minus vowel-selectivity finds neurons tuned to a
single coda; the reverse finds vowel-tuned neurons:

```text
coda neurons     neuron 4326 -> -T     neuron 1375 -> -D     neuron 263 -> -R
                 neuron 2462 -> -L      neuron 4843 -> -L
vowel neurons    neuron 2256 -> OW      neuron 2493 -> UW     neuron 2213 -> AO
                 neuron 2550 -> IH      neuron 1823 -> IY
```

Among the 200 most rhyme-selective neurons, 96 are coda-dominant and 104
vowel-dominant — two comparably sized populations. This is the single-neuron
image of the compositional code that the probes saw in aggregate (report 08 §2:
coda decodes at 0.82, vowel at 0.45): the MLP represents "what a word sounds
like" on two roughly separable phonemic axes.

## Limits

- Neurons are ranked by activation selectivity, which finds *readable* neurons,
  not necessarily the causally heaviest; an output-weighted ranking (each
  neuron's down-projection column projected onto family directions) might locate
  more causal units, and is the natural next step.
- Tuning is correlational and per-neuron; polysemantic neurons are scored by
  their dominant phoneme only. This is not a dictionary of monosemantic features
  — that would need an SAE on this MLP.
- 30 single-token families, one model, NF4. The "all neurons 0.42" figure is
  partly a high-dimension probe artifact.

## Reproduction

```bash
PYTHONPATH=scripts .venv/bin/python scripts/run_gemma4_mlp13_neurons.py
```

Writes the captured neuron activations to
`artifacts/gemma4_mlp_rhyme/mlp13_neurons.npz` for further CPU analysis.
