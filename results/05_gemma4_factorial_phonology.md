# Phase 3: causal vowel × coda recombination

This experiment asks a stricter question than a phoneme probe: can reusable
vowel and coda information be combined to *cause* Gemma 4 E2B to prefer a
rhyme family that was excluded while the intervention was learned?

For every eligible held-out family, an additive model was fitted to layer-13
family centroids from E1: `activation = intercept + vowel + coda`. The held-out
family was removed completely. Its vowel was therefore learned only through
families with other codas, while its coda was learned only through families
with other vowels. These two supporting sets cannot overlap. Their sum predicts
the unseen vowel+coda cell. At the anchor word of each natural couplet, the
observed source-family centroid was replaced directionally with this predicted
target at the output of layer 13, immediately before layer-14 value storage.

The 24 eligible target families were deterministically split 12/12 before any
model evaluation. On discovery, alpha 2 was selected over 0.5, 1, 4, and 8,
then frozen for confirmation.

## Confirmation result

| Intervention | Mean change in target-family probability | 95% family-bootstrap CI |
|---|---:|---:|
| Factorial vowel+coda recombination | **+12.92 points** | **[+7.31, +19.36]** |
| Norm-matched Gaussian | +0.87 points | [+0.37, +1.43] |
| Norm-matched shuffled-family model | +1.48 points | [+0.54, +2.61] |
| Factorial minus Gaussian | **+12.06 points** | **[+6.75, +18.32]** |
| Factorial minus shuffled | **+11.44 points** | **[+6.56, +17.29]** |

All 12 held-out confirmation families moved in the predicted direction. Effects
ranged from +0.48 points for /AO1+R/ to +32.83 for open /EY1/. The intervention
made a target-family word greedy top-1 on 17.0% of prompt×target trials, from a
near-zero baseline target mass (0.31%). This is deliberately a hard redirection
task: most targets are unrelated to the prompt's original semantic completion.

The spelling control also transferred. Probability on target-family rhymes
whose written rime differs from the target family's most frequent spelling rose
by **+7.09 points**, prompt-bootstrap CI [+4.86, +9.69]. Thus the result cannot
be reduced to steering toward a particular written suffix.

An independent BF16 confirmation run reproduced the effect: **+14.52 points**
[+8.02, +21.86], versus +1.00 random and +0.65 shuffled. Cross-spelling mass
rose +8.23 points.

## Interpretation and limits

This is positive causal evidence that the layer-13 code contains reusable
vowel and coda components: an additive estimate constructed without the target
family makes the downstream model select members of that recombined family.
It is stronger than cross-family decodability alone.

It does not show that Gemma literally performs this least-squares decomposition
internally, nor that the code is exclusively phonemic. Family effects are
heterogeneous, source prompts number only 14, the lexicon is CMUdict-based, and
the BF16 replication reused the preregistered NF4 alpha rather than repeating
discovery. The open-coda families are legitimate zero-coda cells but should be
reported separately in any expanded analysis.

## Reproduction

```bash
.venv/bin/python scripts/run_gemma4_factorial_phonology.py --phase discovery
.venv/bin/python scripts/run_gemma4_factorial_phonology.py \
  --phase confirmation --confirmation-alpha 2
.venv/bin/python scripts/run_gemma4_factorial_phonology.py \
  --phase confirmation --confirmation-alpha 2 --control-repeats 1 --bf16 \
  --representation artifacts/gemma4_representation_bf16 \
  --output artifacts/gemma4_factorial_phonology_bf16
```

Raw prompt-level rows, family-level effects and CIs, fixed split, selected
alpha, and summaries are saved under `artifacts/gemma4_factorial_phonology/`
and `artifacts/gemma4_factorial_phonology_bf16/`.
