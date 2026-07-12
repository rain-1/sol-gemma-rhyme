# A gallery of causal interventions

The other reports argue their claims with aggregate statistics and controls.
This one is the fun companion: a small set of **concrete, real examples** of the
model's rhyming behavior being switched off, redirected, or built from parts by
a single targeted intervention. Every completion below is an actual greedy
top-1 output from `google/gemma-4-E2B` (eager attention, NF4), pulled directly
from the committed run artifacts — nothing here is hand-written or idealized.

Each example states the **exact intervention** and shows the model's own words
before and after. Rhyme families are written as their CMUdict code with a plain
gloss, e.g. `AY1-T` = the *-ight* sound (light, white, night).

---

## 1. Switch off the rhyme

**Intervention:** zero the output of a single attention head — **layer 24, head
3 (L24H3)** — at the final token, and let the model complete greedily. Nothing
else is touched. This is the retrieval head the circuit analysis identified.

The model is given three lines of an AABB quatrain plus the fourth line with its
last word removed, and no instruction to rhyme.

**Example A** — rhyme partner is *sleep*:

```
The passion burns like raging fire,
Each moment lifts our hearts yet higher.
We drift away to gentle sleep,
Where dreams run wild and dark and ____
```
- **Normal model:** `deep`  — rhymes. Probability mass on the *-eep* family: **0.98**
- **With L24H3 switched off:** `wild` — no rhyme. Family mass collapses to **0.02**

**Example B** — rhyme partner is *gold*:

```
The waters stretch across the sea,
Their endless waves forever free.
The shoreline glimmers bright with gold,
Though winds blow fierce and ever ____
```
- **Normal model:** `cold` — rhymes (family mass **0.90**)
- **With L24H3 switched off:** `strong` — broken (family mass **0.02**)

**Example C** — rhyme partner is *gloom*:

```
They rented out a hotel suite
To make their vacation complete
The castle hung with clouds of gloom
Like secrets sleeping in a ____
```
- **Normal model:** `tomb` — rhymes (family mass **0.63**)
- **With L24H3 switched off:** `dark` — broken (family mass **0.05**)

A few more of the same kind, all from AABB quatrains, rhyme-partner → normal →
ablated: *lake* → `make` → `see`; *kind* → `find` → `set`; *bright* → `white` →
`love`; *domain* → `reign` → `government`. One head carries most of the
behavior: knock it out and the model still writes a fluent line, it just stops
caring about the sound.

---

## 2. Force a rhyme it wasn't going to make

**Intervention:** add a single **rhyme-family direction vector** into the
residual stream at the anchor word (the last word of the first line), at the
**output of layer 13** — just before the layer-14 memory the retrieval head
reads — with a small fixed strength. The vector points at a *chosen* target
family. Everything else is untouched, and we read the greedy completion.

Each prompt has a three-couplet demonstration preamble; the operative couplet is
shown. The model was heading confidently one way, and one added direction
reroutes it.

**Example A** — inject the *-y/-igh* sound (sky, high):

```
The window held a square of light
An owl went hunting through the ____
```
- **Normal model:** `night` — rhymes with *light*
- **After injecting the `AY1` direction:** `sky` — target-family mass 0.003 → **0.900**

**Example B** — inject the *-ight* sound (tight, white):

```
The northern wind blew sharp and cold
The cabin stood there, dark and ____
```
- **Normal model:** `old` — rhymes with *cold*
- **After injecting the `AY1-T` direction:** `tight` — mass 0.001 → **0.911**

**Example C** — inject the *-ay* sound (away, day):

```
The river hurried toward the sea
The captive bird at last flew ____
```
- **Normal model:** `free` — rhymes with *sea*
- **After injecting the `EY1` direction:** `away` — mass 0.203 → **0.879**

Two more, couplet-ending → normal → after injection: *rain* couplet, `train` →
inject *-ame* → `flame` (0.002 → 0.566); *flame* couplet, `name` → inject *-ell*
→ `spell` (0.000 → 0.463).

---

## 3. Build a rhyme family out of parts it was never shown

This is the most surprising one. A rhyme family is a **vowel** plus a **coda**
(the ending consonants). We estimate a direction for the target family *without
ever using that family*: its vowel is learned only from families with **other**
codas, and its coda only from families with **other** vowels. Then we add the
sum of those two parts.

**Intervention:** at the anchor word, at the output of layer 13, add
`vowel_direction + coda_direction`, where each part was fitted on data that
excluded the target family entirely. If the model then ends on a word from the
held-out family, it is *recombining* parts, not replaying a memorized vector.

**Example A** — assemble the *-ight* sound (never built from any *-ight* word)
and drop it on a poem that rhymes in *-ill*:

```
The shepherd climbed the grassy hill
At sunset every field grew ____
```
- **Normal model:** `still` — rhymes with *hill*
- **After adding the assembled *-ight* vowel + coda:** `white` — family mass 0.000 → **0.917**

**Example B** — assemble the *-ain* sound and drop it on an *-ore* poem:

```
A stranger waited by the door
He knocked and softly asked once ____
```
- **Normal model:** `more` — rhymes with *door*
- **After adding the assembled *-ain* vowel + coda:** `again` — mass 0.002 → **0.981**

**Example C** — assemble the *-o* sound and drop it on an *-ight* poem:

```
The window held a square of light
An owl went hunting through the ____
```
- **Normal model:** `night` — rhymes with *light*
- **After adding the assembled *-o* vowel + coda:** `snow` — mass 0.000 → **0.486**

One more: the *flame* couplet (normally `name`), given an assembled *-ee* sound,
ends on `tree` (0.001 → 0.476). In each case the target sound is being *built*
from a vowel and a coda the estimate never saw together.

---

## 4. Make it rhyme with the wrong line

**Intervention:** in an AABB stanza the final line should rhyme with the line
directly above it (distance 1). Take the **layer-14 attention *keys*** from an
ABAB context — where the correct partner is two lines up (distance 2) — and
paste them into the AABB run. Nothing about the *content* of the lines changes,
only the address labels the retrieval head uses to decide where to look.

Here the candidate lines each end in a *different* sound, so the model's actual
completion reveals which line it is rhyming against. This is an *open* prompt —
nothing forces a rhyme — so left alone the model just finishes it by meaning:

```
The dawn was mirrored in her eyes
The shepherd climbed the grassy hill      <- two lines up  (-ill)
The old clock marked the passing time      <- line above    (-ime, the correct partner)
The cabin stood there, dark and ____
```

The exact greedy word the model emits in each condition:

| Condition | Model says | What happened |
|---|---|---|
| **Normal** | `grim` | a plain semantic ending ("dark and grim"); rhymes neither line |
| **Paste in ABAB *keys*** | **`still`** | now rhymes `hill` — the **wrong** line, two up |
| **Paste in ABAB *values*** | `grim` | unchanged |

The key swap re-points the head: its attention on the *hill* line rises from
**0.19 → 0.52**, and probability on the *-ill* (`hill`) family jumps from
**0.006 → 0.828**. Swapping the *values* instead leaves attention (0.50 / 0.19)
and the word untouched.

A second open stanza, whose line above ends in *breeze* and whose line two up
ends in *rain*: the normal completion is `tree` (leaning to *breeze*), but with
ABAB keys pasted in the model says **`name`** — pulled toward the *rain* line two
up. The keys are the "where to look" address; the values are the "what it sounds
like" content. You can redirect which line the model rhymes against without
touching the lines themselves.

---

## How to reproduce these

The exact runs that produced every number above:

```bash
# 1. head ablation (also the external replication)
PYTHONPATH=scripts .venv/bin/python scripts/run_gemma4_external_schemes.py
# 2. forcing a family
PYTHONPATH=scripts .venv/bin/python scripts/run_gemma4_steering.py
# 3. recombination from parts
PYTHONPATH=scripts .venv/bin/python scripts/run_gemma4_factorial_phonology.py
# 4. routing / key-vs-value
PYTHONPATH=scripts .venv/bin/python scripts/run_gemma4_routing_decomposition.py
```

Per-example rows live in `artifacts/gemma4_external_schemes/`,
`artifacts/gemma4_steering/`, `artifacts/gemma4_factorial_phonology/`, and
`artifacts/gemma4_phase3/`.

## The obligatory caution

These are single-example greedy outputs chosen to be legible; the honest,
controlled versions of each claim — with confidence intervals, negative
controls, and the cases where the intervention only partly works — are the
whole point of reports [02](02_gemma4_rhyme_mechanism.md),
[03](03_gemma4_rhyme_representation.md), and
[04](04_gemma4_phase3_causal_tests.md). "Force a rhyme" lands as a clean top-1
flip in a minority of prompts (it shifts the distribution far more often than it
wins outright); the ablation leaves other schemes partly intact. Enjoy the
gallery, but read the phase reports for what is actually established.
