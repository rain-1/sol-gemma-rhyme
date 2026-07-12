# An SAE on the layer-13 MLP: partial reconciliation

Report 09 found the causal rhyme write is a low-rank direction in superposition,
and that a change of coordinates into vowel and coda axes does not cleanly
resolve it (the vowel and coda subspaces are non-orthogonal, cos ≈ 0.65, and
miss ~22% of the family subspace). The remaining option is an overcomplete
sparse dictionary. This report trains a sparse autoencoder (SAE) on the layer-13
output and reports, honestly, a partial result.

`google/gemma-4-E2B`, eager, NF4. Activations captured for **6000 single-token
words** in the rhyme-mode scaffold (scaling well past the 30-family probe set).
Reproduce with `scripts/capture_mlp13_lexicon.py`, `train_mlp13_sae.py`,
`analyze_mlp13_sae.py`, `sae_rhyme_subspace.py`.

## Main conclusions

- **A vanilla residual SAE learns lexical features, not phonemes.** Trained on
  the raw layer-13 residual (1536→2048, recon var 0.98), its interpretable
  features are word **stems** — one feature for *be-* (`beet, beat, bee, bean`),
  one for *ba-* (`ban, banks, banned`), one for *co-* (`coat, coast, coach`) —
  each firing for only one or two onsets. The residual at a word is dominated by
  lexical identity, so that is what the SAE decomposes. The one phoneme-like
  feature is the *-ation* suffix.
- **Isolating the rhyme subspace first surfaces a few real phoneme features.**
  Projecting onto the phoneme subspace (vowel- and coda-mean directions) removes
  most lexical identity; an SAE there finds genuine **cross-onset** features — a
  long-A vowel (`base, chase, apes, aides`), an AE1 vowel (`attach, batch,
  catches`), and a *-tion* coda (`accusation, apprehension, aggravation`).
- **But the reconciliation is only partial.** Even in the isolated subspace the
  SAE recovers just a handful of distinct phonemes, at 73–80% purity, and the
  result is sensitive to the sparsity setting. Neither a change of coordinates
  nor an SAE turns the rhyme code into a clean, complete monosemantic phoneme
  dictionary at this model and data scale.

## 1. The vanilla SAE decomposes word identity

Trained directly on the layer-13 residual, the SAE reconstructs well (var 0.98)
and its top features, read by their most-activating words, are lexical stems:

```text
feat  73  ->  beet, beat, beats, bee, bean, beast     (onset be-)
feat 1504 ->  ban, bans, banned, banks, banning       (onset ba-)
feat 1225 ->  co, coat, coats, coast, coach           (onset co-)
```

These *look* vowel-selective — `bee`/`beat` share the vowel IY1 — but their words
share a **stem**, not just a phoneme (each feature spans only one or two onsets).
They are word-family detectors. This is expected: the rank-18 family (rhyme)
subspace is 36% of the residual variance, but lexical identity is the more
sparse-codeable structure — a word stem is a tight cluster, a rhyme class is a
diffuse one — so an unconstrained SAE spends its features on stems.

## 2. Inside the phoneme subspace, some features are real

Projecting the residuals onto the phoneme subspace (rank 68, spanned by the
vowel-mean and coda-mean directions) strips most lexical identity, and an SAE
there does find features that fire across **many different onsets** — the mark
of a genuine phoneme feature rather than a stem:

```text
long-A vowel : base, bases, apes, chase, aides, based      (7 onsets)
AE1 vowel    : attracts, attach, batches, batch, catches   (6 onsets)
-tion coda   : accusation, apprehension, aggravation, ...   (8 onsets)
```

So the phoneme structure the probes and the factorial experiment (reports 05,
08) saw in aggregate is partly expressible as sparse features — but only a few
surface cleanly, and pushing for sparsity drops their purity and loses the coda
features first.

## Interpretation

The three views agree on a single picture. The rhyme code the layer-13 MLP
writes is **genuinely distributed and entangled**: it is a low-rank direction
(report 09), its vowel and coda parts are non-orthogonal (cos 0.65, report 09
§coordinates), and it does not factor into a clean overcomplete monosemantic
basis here. A few phoneme axes are extractable — the *-tion* coda and the long-A
vowel are real, cross-onset features — but "the -T neuron" or a full phoneme
dictionary is not what this MLP contains at this scale. That is consistent with
everything upstream: sparse to *read*, dense and entangled to *write*.

## Limits

- SAEs are normally trained on corpus-scale activations; 6000 isolated words is
  small, and the dictionaries here are correspondingly modest. A larger, more
  contextually varied activation set could sharpen the phoneme features (this is
  the natural place for more data).
- The phoneme subspace is built from labels, so §2 is a supervised isolation,
  not an unsupervised discovery.
- One small base model. A larger model may carry cleaner monosemantic phonemes,
  or may simply have more room for the same superposition.

## Reproduction

```bash
PYTHONPATH=scripts .venv/bin/python scripts/capture_mlp13_lexicon.py   # 6000-word residuals
.venv/bin/python scripts/train_mlp13_sae.py --l1 1.2e-2                # vanilla SAE (section 1)
.venv/bin/python scripts/analyze_mlp13_sae.py                          # feature readout
.venv/bin/python scripts/sae_rhyme_subspace.py                         # subspace SAE (section 2)
```
