# Pythia rhyme interpretability

First-milestone experiments for asking how `EleutherAI/pythia-410m-deduped`
completes the last word of a rhyming couplet.

The project now also includes a causal Gemma 4 E2B analysis. The readable result
is [`results/gemma4_rhyme_mechanism.md`](results/gemma4_rhyme_mechanism.md). Run
the full scan with:

```bash
.venv/bin/python scripts/run_gemma4_interpretability.py
.venv/bin/python scripts/validate_gemma4_circuit.py
.venv/bin/python scripts/plot_gemma4_results.py
```

The benchmark contains 25 hand-written couplets under eight prompt wrappers
(200 examples). The final word is omitted. Exact rhyme is scored from CMUdict
phonemes beginning at the last stressed vowel, and the vocabulary analysis is
restricted to standalone single-token English words.

## Setup and reproduction

```bash
python -m venv --system-site-packages .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/rhyme-interp dataset
.venv/bin/rhyme-interp evaluate
.venv/bin/rhyme-interp elicit
.venv/bin/rhyme-interp distinct
.venv/bin/rhyme-interp elicit --model EleutherAI/pythia-1.4b-deduped
.venv/bin/rhyme-interp elicit --model allenai/Olmo-3-1025-7B --load-in-4bit
.venv/bin/pip install -e '.[quant,gemma4]'
.venv/bin/rhyme-interp elicit --model google/gemma-4-E2B --load-in-4bit
```

The behavior output records top-1/top-10 rhyme accuracy, total probability mass
on the anchor's rhyme family, the known target's probability and rank, and a
family-size-normalized rhyme logit advantage.

`elicit` compares the same 25 target couplets with no prefix, three completed
rhyming couplets, the exact same six lines shuffled to break adjacent rhymes, and
the rhyming pairs in reverse order. This prevents a longer prefix from being
mistaken for evidence that the model learned the demonstrated rhyme pattern.

`distinct` repeats the length sweep with 1, 3, 5, 10, 15, and 20 unique
Claude-generated couplets. For every target it excludes demonstration rhyme
families that overlap the target anchor and evaluates a same-lines shuffled
control.

Gemma 4 support currently requires a Transformers development revision. The
`gemma4` extra pins the exact tested commit. OLMo-3 and Gemma-4 were evaluated
as base models with bitsandbytes NF4 4-bit weights on the same prompts.

Run causal analyses on a base couplet (0–24):

```bash
.venv/bin/rhyme-interp layers --example 0
.venv/bin/rhyme-interp heads --example 0
.venv/bin/rhyme-interp patch --example 0 --source 1
```

- `layers` removes each transformer block's final-position residual update.
- `heads` zeros each attention head immediately before its output projection at
  the final position.
- `patch` uses an exact-scaffold, meta-poetry counterfactual set. It copies the
  source anchor residual into the destination anchor position, one layer at a
  time, and measures movement toward the source rhyme family and away from the
  destination family. Identical scaffolds keep syntax and token position fixed.

Outputs go to `artifacts/`, which is intentionally gitignored because model-run
results are reproducible and can be large. The fixed benchmark can be committed
at `data/controlled_couplets.jsonl`.

Run tests with `.venv/bin/python -m pytest -q`. In a system-site-packages virtual
environment the `pytest` module may be available without a `.venv/bin/pytest`
launcher.

## Interpretation cautions

Top-1 rhyme accuracy alone is not causal evidence. Prompt wrappers can teach the
task explicitly; the plain wrapper is the strongest test of spontaneous poetic
behavior. CMUdict excludes invented words and slant rhyme. Head ablations also
operate inside a nonlinear model, so use them to nominate components and verify
important findings with counterfactual activation patching.
