# Reading rhyme out of Gemma 4

Report 07 localized where rhyme is *written* — the layer-13 MLP — and noted that
the raw residual is a poor place to read it (a family probe there reaches only
~0.30). This report is the detailed account of the *readout*: rhyme becomes
strongly and precisely legible at the one representation the retrieval head
actually consumes, the layer-14 shared full-attention **value** memory. It
covers where the readout lives and why, how the code is organized, whether every
family reads out, the rhyming-dictionary query it supports, and a demonstration
that you can also *write* a fake rhyme into the same slot.

All results use `google/gemma-4-E2B`, eager attention, NF4. The value memory is
captured by `run_gemma4_representation.py` for 359 single-token words in 30
CMUdict rime families; probes are cross-validated linear (logistic) probes,
chance 0.033.

## Main conclusions

- **The head reads a clean code.** A linear probe on the layer-14 value memory
  reads the rhyme family at **0.90** (5-fold) and **0.88 / 0.86** on held-out
  words within / across scaffolds — versus **0.30** for the raw layer-13
  residual under the identical probe. The value projection isolates the
  low-variance rhyme direction the MLP wrote out of the noisy residual.
- **The code is phonemic and compositional.** In the value memory the *coda*
  of a completely held-out family decodes at **0.82** and the vowel at 0.45;
  the residual is weaker (0.65 / 0.40) and the embedding near chance. Rhyme is
  a (vowel, coda) pair, not an opaque family label.
- **Every family reads out.** All 30 families are recovered; **9 perfectly**;
  the mean is 0.90. The only weak family, EH1-R (*-air*: there, care), confuses
  with its slant-rhyme neighbour IH1-R (*-eer*: dear, fear) — every error is a
  phonetic near-miss, which confirms the phonemic organization.
- **It is a usable rhyming dictionary.** Nearest-neighbour in the value memory
  returns real rhymes: `light → night, flight, right, fight, might`.
- **You can write a fake rhyme in.** Primed into rhyme mode and given a chosen
  family's mean overwritten into its slot, a word with no fixed rhyme becomes a
  dial: *month* can be made to rhyme with `day` (0.85), `light` (0.78), `gold`
  (0.63), or `sea` (0.48). Even *orange* bends toward *-ight*, though it resists
  most.

## 1. Where the readout lives, and why

The retrieval head L24H3 does not attend to the raw residual stream; it attends
to the layer-14 shared full-attention memory, whose **value** is a learned
projection (`W_V`) of the residual at the anchor. Probing each candidate
representation with the same probe and split:

| Representation | Rhyme-family readout |
|---|---:|
| static token embedding | 0.31 |
| raw layer-13 residual (where the MLP writes) | 0.30 |
| **layer-14 full-attention value (where the head reads)** | **0.88 held-out / 0.90 5-fold** |
| layer-14 full-attention value, across a held-out scaffold | 0.86 |

The residual carries the rhyme code the MLP wrote, but buried under position,
semantics, and massive-activation dimensions; a linear probe cannot cleanly
separate it there. The value projection is a learned linear map that keeps
exactly what the head needs — it acts as a denoiser, lifting the readout from
0.30 to 0.90. Reading in the right basis is the whole difference.

## 2. How the code is organized: vowel × coda

A rhyme family is a stressed vowel plus a coda. Leave-one-family-out probes ask
whether those parts are shared across families (phonemic) or bound into opaque
identities. In the value memory they are clearly phonemic:

| Representation | vowel (held-out family) | coda (held-out family) |
|---|---:|---:|
| embedding | 0.24 | 0.30 |
| layer-13 residual | 0.40 | 0.65 |
| **layer-14 value** | 0.45 | **0.82** |
| layer-14 key | 0.37 | 0.34 |

The coda is especially strong in the value memory (0.82 on a family it never
saw). The split with the *key* is telling: the value carries *what it sounds
like* (coda), while the key carries more of the vowel/addressing — consistent
with report 04's finding that keys route and values carry content.

## 3. Can we read out all families?

Yes, and the failures are informative. Five-fold recall per family:

| Recall | Families |
|---|---|
| 1.00 (perfect) | AE1-D, AE1-K, AE1-S, AH1-N, EH1-D, EH1-N-T, EY1, EY1-M, OW1-L-D |
| 0.83–0.92 | 18 families (most of the set) |
| 0.75 | AY1, EY1-T |
| 0.58 (weakest) | EH1-R |

Overall 0.90. Every substantial confusion is a **slant-rhyme neighbour**:

```text
EH1-R  -> IH1-R   (there/care confused with dear/fear)   x4
AY1-D  -> AY1-N-D (ride/hide confused with mind/blind)   x2   (same vowel, coda d vs nd)
```

A readout that confuses *-air* with *-eer* and *-ide* with *-ind* is not making
random errors — it is grouping by sound, exactly what a phonemic code predicts.
The one genuinely weak family, EH1-R, is also phonetically closest to another
family in the set.

Coverage caveat: these are 30 frequency-ranked families restricted to
single-token words. The full CMUdict rime inventory has hundreds of families,
many with few single-token members; the method should extend, but the strong
numbers here are for this tested set, not a claim about every English rhyme.

## 4. A rhyming dictionary

Because the value memory is cleanly organized, nearest-neighbour cosine in it
behaves like a rhyming dictionary — give a word, get its rhyme set:

```text
light -> night, flight, right, fight, weight, might
grace -> space, trace, race, embrace, pace, place
rain  -> train, pain, brain, gain, chain, maintain
day   -> say, pay, may, today, way, stay
```

Grouping held-out words by a probe's decision recovers clean families
(`AY1-T: fight, site, might, despite, light`; `OW1-L-D: bold, cold, old, hold,
told`). Unsupervised clustering reaches 0.60 purity here versus 0.32 on the raw
residual. Reproduce with `scripts/extract_rhyme_sets.py`.

## 5. Writing a fake rhyme in

If the family can be *read* out of the value memory, it should be possible to
*write* one in — to make the head believe a word rhymes with a family of our
choosing. Priming the model into rhyme mode with a short demonstration preamble
and overwriting an anchor's layer-13 code with a target family's mean, the
greedy completion of an open second line becomes a member of that family. The
clearest dial is *month* (which has no real rhyme):

| Family written into "month" | completion | family mass |
|---|---|---:|
| EY1 (*day, way*) | `day` | **0.85** |
| AY1-T (*light, night*) | `light` | **0.78** |
| OW1-L-D (*cold, gold*) | `gold` | **0.63** |
| IY1 (*sea, three*) | `sea` | **0.48** |
| EH1-L (*well, bell*) | `well` | 0.27 |

We can freely choose which family "month" rhymes with. The same works for
*silver* (→ `light` 0.63, `cold` 0.41, `day` 0.56) and *engine* (→ `bell` 0.53,
`night` 0.44).

Then there is **orange**. It is by far the most resistant word: under a scaled
injection its completion barely moves, and only a full overwrite with the
*-ight* family bends it toward that sound (family mass 0.34, reaching for
`light`). Orange resists even mechanistic tampering — either a limit of this
intervention or a deep truth about the English language.

Two honest notes. A clean *overwrite* of the anchor code is more reliable than a
scaled addition, and large additions (strength 8) over-drive the residual into
unrelated words. And the effect **requires** the model to be in rhyme mode:
without the demonstration preamble the retrieval head barely consults the anchor
and the injection does nothing at all. That dependence is itself confirmation
that we are driving the rhyme-retrieval pathway and not a generic output bias —
the same head, reading the same slot, that reports 01–07 identified.

## Limits

- The readout is supervised: it needs labelled words to fit the family
  directions. Fully unsupervised recovery is only fair (0.60 purity).
- Reading is correlational; §5 and the steering/recombination results (report
  04–05) are the causal side — you can inject a family and change behaviour.
- 30 single-token families, one base model, NF4, CMUdict. Slant rhyme and
  invented words are out of scope.

## Reproduction

```bash
PYTHONPATH=scripts .venv/bin/python scripts/run_gemma4_representation.py  # captures the value memory
.venv/bin/python scripts/extract_rhyme_sets.py                            # readout, coverage, query
PYTHONPATH=scripts .venv/bin/python scripts/patch_fake_rhyme.py           # section 5
```
