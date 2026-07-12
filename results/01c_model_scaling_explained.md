# Rhyme completion improves sharply with larger base models

## Main result

We finally obtained a reliably rhyming model without changing or selecting the
25 target poems:

- **OLMo-3 7B base:** 24 of 25 greedy completions rhyme (96%).
- **Gemma 4 E2B base:** 23 of 25 greedy completions rhyme (92%).

Both models used the same fixed prefix of three completed rhyming couplets. No
prompt was selected separately for an individual poem. Both larger models were
loaded in bitsandbytes NF4 4-bit form to fit comfortably on the RTX 4080.

## Direct comparison

| Base model | No prefix: greedy rhyme | Three rhyming couplets | Same six lines shuffled | Rhyme in top 10 with rhyming prefix |
|---|---:|---:|---:|---:|
| Pythia 410M | 28% | 40% | 36% | 80% |
| Pythia 1.4B | 44% | 44% | — | 84% |
| OLMo-3 7B, NF4 | 56% | **96%** | 72% | 100% |
| Gemma 4 E2B, NF4 | 56% | **92%** | 64% | 100% |

For Pythia-1.4B, the matched long control used twenty distinct shuffled poems
rather than the six-line shuffled condition, so its control number is not placed
in the same column.

## What the model sees

The successful universal prefix is:

```text
The rain made mirrors in the lane
At dusk there came a distant train
The fire faded to a glow
Beyond the glass descended snow
A gull went wheeling by the sea
It spread its wings and wandered free
```

This is followed immediately by one of the original incomplete targets:

```text
The roof began to drum with rain
Beyond the fields there passed a
```

OLMo-3's greedy next word is `train`.

## Why the shuffled control matters

The shuffled condition contains the exact same six demonstration lines and the
same vocabulary, but rearranges their second lines so adjacent lines no longer
rhyme. It therefore still looks like poetry and has the same length.

| Model | Ordered rhyming prefix | Shuffled lines | Difference |
|---|---:|---:|---:|
| Pythia 410M | 40% | 36% | +4 points |
| OLMo-3 7B | 96% | 72% | **+24 points** |
| Gemma 4 E2B | 92% | 64% | **+28 points** |

For the larger models, preserving adjacent rhyming pairs has a large effect. This
is much stronger evidence of in-context rhyme-pattern recognition than the
Pythia result. The shuffled conditions still score above the plain prompts,
which means generic poetic context also helps.

Reversing the two lines inside every demonstrated couplet preserves the adjacent
rhyme relation. It gives 96% on OLMo and 92% on Gemma—the same scores as the
normal order. This supports the interpretation that the pairwise rhyme structure
matters more than the specific semantic order of the demonstration lines.

## Remaining failures

With the rhyming prefix, OLMo failed only the `star`/`guitar` poem. Its first
token began a different word, while `guitar` was ranked second. Gemma failed:

- `sky`/`high`, choosing `of` while `high` ranked fourth; and
- `star`/`guitar`, choosing `harp` while `guitar` ranked second.

Thus all intended reference completions remain close to the top even in the
three failures.

## Interpretation consequence

OLMo-3 7B is now the strongest behavioral subject: it supplies 24 positive cases
under one universal prompt plus a matched shuffled control. Gemma 4 E2B is also
attractive because it is smaller while reaching 92%.

The original native GPT-NeoX intervention hooks will not transfer directly to
either architecture. Before mechanism analysis, we need architecture-specific
hooks for OLMo or Gemma residual blocks and attention projections. The behavioral
metric, CMUdict rhyme families, and counterfactual design remain reusable.

## Reproduction

```bash
# OLMo-3 7B base
.venv/bin/rhyme-interp elicit \
  --model allenai/Olmo-3-1025-7B \
  --load-in-4bit \
  --output artifacts/olmo3_7b_base_elicitation.jsonl

# Gemma 4 E2B base; install the pinned development Transformers first
.venv/bin/pip install -e '.[quant,gemma4]'
.venv/bin/rhyme-interp elicit \
  --model google/gemma-4-E2B \
  --load-in-4bit \
  --output artifacts/gemma4_e2b_base_elicitation.jsonl
```
