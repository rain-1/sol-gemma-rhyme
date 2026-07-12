# Phase 3: causal tests of rhyme composition, routing, and planning

This report follows the Phase 2 circuit analysis with three stricter questions:

1. Can phonological components be *causally recombined* into a rhyme family
   that was withheld from the intervention construction?
2. Which part of attention memory tells the retrieval head *where* to look,
   and when is its query formed?
3. Is a chosen rhyme family already present earlier in the final line in a
   form that can causally control the completion?

All reported interventions use the base `google/gemma-4-E2B` model with eager
attention. NF4 is the main condition; the phonological recombination result
was independently repeated in BF16. “Rhyme-family mass” means the model's
total next-token probability assigned to all single-token words with the same
CMUdict exact rime, from the final stressed vowel onward.

## Main conclusions

- **Phonological composition is causal, not merely probe-readable.** Combining
  a vowel component and a coda component learned without the target family
  raises that held-out family's probability by 12.92 percentage points in
  NF4 and 14.52 points in BF16. Random and incorrectly paired components move
  it by less than 1.5 points.
- **Memory keys route; memory values carry content.** Swapping layer-14 keys
  between AABB and ABAB contexts re-aims L24H3. Swapping values does not alter
  its attention. This complements the earlier result that values, rather than
  keys, causally transfer the anchor's rhyme identity.
- **The usable query-side scheme signal forms late.** Donor query states do
  not re-aim the head through layer 22. The effect appears after layer 23 and
  depends on both that block's attention and MLP updates.
- **No directly transferable rhyme-family plan was found earlier in the final
  line.** Residual and attention-memory patches at five line positions recover
  less than 1% of the donor-family effect. Anchor-position positive controls
  recover 96–100%, showing that the intervention can transfer the code when it
  is present. This is a bounded negative result, not proof that every possible
  distributed plan is absent.
- **The head's effective signal is concentrated, but “16–32 dimensional” was
  too strong.** The channel is 512-dimensional. Top PCA directions recover
  much more than matched random directions, but only a true rank-512 patch is
  a full-rank sanity check.
- **Independent AABB and ABAB poems replicate strongly; ABBA does not.** On 90
  newly generated and pronunciation-validated poems, greedy exact rhyme is
  88.0% for AABB and 89.7% for ABAB, but only 26.7% for ABBA.

## 1. Causal vowel–coda recombination

For each tested target family, the intervention uses examples sharing its
vowel with other codas and examples sharing its coda with other vowels. No
example from the target vowel+coda family is used to build its intervention.
The discovery half selected intervention strength 2; the fixed confirmation
half contains 12 families.

| Confirmation result | Change in target-family mass |
|---|---:|
| Additive vowel+coda intervention, NF4 | **+12.92 points** |
| Family-bootstrap 95% CI | **+7.31 to +19.36** |
| Random-vector control | +0.87 points |
| Shuffled component pairing | +1.48 points |
| Additive minus shuffled, 95% CI | **+6.56 to +17.29** |
| Cross-spelling subset | +7.09 points |
| Additive intervention, BF16 | **+14.52 points** |
| BF16 family-bootstrap 95% CI | **+8.02 to +21.86** |

All 12 confirmation families move in the predicted direction. The shuffled
control is important: it has comparable construction and scale, but destroys
the correct relationship between phonological parts. These results support a
causal claim that the model can reuse vowel and coda information to construct
an unseen family constraint. They do not imply that the representation is a
perfectly linear phoneme table.

The complete preregistration, split, family-level results, and controls are in
[`gemma4_factorial_phonology.md`](gemma4_factorial_phonology.md).

## 2. Separating content from addressability

The AABB and ABAB conditions use identical target stanzas and matched
demonstration lengths. They differ in which completed line should be retrieved
at the final token: distance 1 for AABB and distance 2 for ABAB.

Patching all cue-position layer-14 shared-memory components gives:

| Destination | Intervention from other scheme | Attention d1 / d2 | Mass d1 / d2 |
|---|---|---:|---:|
| AABB | none | 0.596 / 0.134 | 0.0641 / 0.0023 |
| AABB | keys | **0.276 / 0.349** | 0.0096 / **0.0552** |
| AABB | values | 0.596 / 0.134 | 0.0594 / 0.0025 |
| ABAB | none | 0.130 / 0.523 | 0.0017 / 0.0608 |
| ABAB | keys | **0.404 / 0.246** | 0.0097 / 0.0182 |
| ABAB | values | 0.129 / 0.524 | 0.0018 / 0.0594 |

Key-only and key+value patches agree closely, whereas value-only attention is
essentially baseline. Individual cue patches show that routing involves both
suppressing the old address and enhancing the new one. The behavioral
transfer is less complete in the ABAB-to-AABB direction even when attention
flips, so addressing is necessary but does not explain every downstream
probability difference.

This establishes a double dissociation:

```text
layer-14 VALUE  -> what phonological family the ending represents
layer-14 KEY    -> whether that ending is addressable under this scheme
layer-23 query  -> which address the final token requests
```

Scanning donor final-token states after every block finds no usable routing
transfer through layer 22. After layer 23 the attention target shifts sharply.
Within block 23, swapping both the attention and MLP updates reproduces the
full query-state intervention; each update contributes, while swapping only
the block input does not.

## 3. Causal test of early planning

Source and destination prompts share a scaffold but use different anchor
families. The source-family mass is 0.786 and destination-family mass is 0.079,
providing a large counterfactual effect to recover. Source activations are
patched into the destination at the anchor and at five locations in the final
line: line opening, first word, middle, penultimate input, and final input.

| Intervention class | Largest mean absolute recovery |
|---|---:|
| Residual states, line positions, layers 0–13 | **0.81%** |
| Sliding/full K, V, or K+V, line positions | **0.72%** |
| Anchor residual after layer 13, positive control | **99.9%** |
| Anchor layer-14 full-attention value, positive control | **95.9%** |

The strong positive controls distinguish “no transferable signal at this
location” from “the patching code does not work.” Combined with the earlier
observation that L24H3's anchor attention rises progressively toward the end
of the line, these results favor retrieval close to emission time.

The scope matters. Single-position residual and shared-memory patches do not
test plans distributed across many positions, plans encoded in other internal
components, or representations that cease to function when transplanted into
the destination context. The conclusion is therefore deliberately narrower
than “the model does not plan.”

## 4. Corrected rank sanity check

Gemma 4 E2B's relevant head-output channel is 512-dimensional. The earlier
analysis mistakenly described rank 256 as full rank. A corrected nested sweep
and explicit identity control gives:

| Rank | PCA recovery | Random-subspace recovery |
|---:|---:|---:|
| 16 | 45.6% | 2.5% |
| 32 | 77.4% | 5.8% |
| 64 | 84.8% | 10.0% |
| 128 | 96.8% | 26.9% |
| 256 | 106.1% | 60.6% |
| 512 | 110.8% | 110.8% |

At rank 512, PCA, random, and explicit identity transformations have zero
logit difference. The PCA curve shows that causal variation is concentrated
in directions aligned to this anchor dataset. It does not by itself identify
the representation's intrinsic dimensionality.

## 5. Independent scheme replication

Claude Haiku 4.5 generated 30 distinct quatrains for each of AABB, ABAB, and
ABBA under a fixed specification. The committed dataset preserves exact
generation prompts, raw response envelopes, rejection records, filters, and
pronunciation validation. Gemma receives only the first three lines and the
fourth line without its final word; it receives no scheme instruction and no
demonstration poem. Items whose required family has no usable single-token
candidate are excluded per the preregistered evaluator, leaving 25/29/30
examples.

| Scheme | usable n | Greedy exact rhyme | Cue-family mass | L24H3 cue attention | Greedy after L24H3 ablation |
|---|---:|---:|---:|---:|---:|
| AABB | 25 | **88.0%** | 70.5% | 77.8% | 24.0% |
| ABAB | 29 | **89.7%** | 72.3% | 71.7% | 3.4% |
| ABBA | 30 | **26.7%** | 17.3% | 35.2% | 13.3% |

This is strong external replication of the same L24H3 retrieval mechanism for
AABB and ABAB. ABBA is a real limitation rather than a polished-benchmark
artifact: behavior, probability mass, and cue attention all weaken together.
The dataset SHA-256 is
`88265f0e442a72fa82214ea13901d24cd8dd44c7d2ded406e4a8fbeca98d8c7f`.

## Reproduction and evidence gate

```bash
PYTHONPATH=scripts .venv/bin/python scripts/run_gemma4_factorial_phonology.py
PYTHONPATH=scripts .venv/bin/python scripts/run_gemma4_routing_decomposition.py
PYTHONPATH=scripts .venv/bin/python scripts/run_gemma4_causal_planning.py
PYTHONPATH=scripts .venv/bin/python scripts/run_gemma4_external_schemes.py
.venv/bin/python scripts/verify_phase3_results.py
```

The final command checks row counts, full-rank identity equivalence, routing
key/value controls, the layer-23 transition, planning positive controls, and
near-zero line-position transfers. It writes the compact, committed evidence
record [`evidence/gemma4_phase3_summary.json`](evidence/gemma4_phase3_summary.json),
including SHA-256 hashes of the larger ignored raw artifacts.

## Remaining limitations

- Routing and causal-planning decompositions are currently NF4-only.
- The routing decomposition isolates AABB versus ABAB. The independent ABBA
  evaluation confirms weak routing but does not localize its failure.
- The phonological confirmation set has 12 families. Its family bootstrap and
  BF16 replication reduce, but do not eliminate, dataset-selection concerns.
- The causal planning result covers the tested component classes and
  positions only.
- The PCA basis is learned from one anchor scaffold distribution; broader
  cross-context bases and held-out fitting would better characterize the
  geometry.
