# First milestone: technical report

For a plain-language explanation, including concrete examples, read
[`01a_first_milestone_explained.md`](01a_first_milestone_explained.md).

## Question and experimental setup

We tested whether `EleutherAI/pythia-410m-deduped` predicts a rhyming word when
shown the beginning of a two-line poem whose final word is missing.

For example, the model receives exactly this text:

```text
The window held a square of light
An owl went hunting through the
```

The word at the end of the first line, `light`, is the **rhyme anchor**. We then
inspect the model's probability distribution for the single token immediately
after `the`. We do not sample a continuation.

The benchmark contains 25 distinct couplets. Each is repeated under eight text
wrappers, such as no label, `Poem:`, or an explicit request to rhyme. Therefore
there are 200 model inputs but only 25 underlying poems; the 200 rows should not
be mistaken for 200 independent poems.

Every poem was written with one natural intended completion. In the example
above that **reference completion** is `night`. This reference word lets us ask
whether the model recovered the word the dataset author had in mind. It is not
the only acceptable answer: `bright`, `sight`, or another contextually suitable
rhyme would also count as rhyming.

Exact rhyme is determined by pronunciation rather than spelling. Two different
words rhyme when their CMUdict pronunciations match from the final stressed vowel
onward. Thus `light`/`night` rhyme, whereas repeating `light` does not count.

## Metric definitions

### Greedy, or top-1, exact-rhyme accuracy

**Top-1** means the single token to which the model assigns the highest
probability. Selecting it is one step of greedy decoding. We call the prediction
correct if that token is a standalone word that exactly rhymes with the anchor.
Newlines, punctuation, repeated anchor words, and non-rhyming words count as
failures.

Consequently, **28% top-1 exact rhyme means that the first greedily selected
token was a rhyming word for 56 of the 200 wrapped inputs**. On the 25 plain
couplets, it happened for 7 of 25. It does not mean that 28% of a multi-token
generated poem rhymed; we generated and evaluated only the missing final word.

### Top-10 contains an exact rhyme

This asks whether at least one of the ten most probable next tokens is a word
that rhymes with the anchor. It measures whether a rhyming word is a serious
alternative in the model's distribution, even when greedy decoding would choose
something else. It is not the result of generating ten continuations.

### Reference-completion median rank

For every prompt we sort every possible next token from most to least probable.
The **rank** of the hand-written reference completion is its place in that list:
rank 1 means the model would choose it greedily; rank 5 means four other tokens
were preferred. A median rank of 5 means that half of the prompts ranked the
dataset author's intended word fifth or better, and half ranked it fifth or
worse. This metric is secondary because other rhyming completions may also be
valid.

### Rhyme-family probability mass

This is the sum of the probabilities of every known, standalone, single-token
word that rhymes with the anchor. A value of 13.45% means that, on average, the
model assigned 13.45% of its next-token probability to exact rhymes. It does not
say which rhyme was semantically appropriate.

## Behavioral results

| Measurement | Result | Plain-language meaning |
|---|---:|---|
| Greedy next token is an exact rhyme | 28% (56/200) | The model's first choice rhymed on 56 wrapped inputs. |
| Exact rhyme appears among top 10 | 71% (142/200) | A rhyme was among the ten leading choices on 142 inputs. |
| Reference completion median rank | 5 | The word written into the benchmark was typically near, but not at, the top. |
| Mean rhyme-family probability | 13.45% | Roughly one eighth of next-token probability went to exact-rhyme words. |

For the 25 plain prompts, greedy exact-rhyme accuracy was 28% (7/25) and top-10
coverage was 68% (17/25).

Examples make the distinction clearer:

| Anchor | Incomplete second line | Greedy prediction | Reference | Rhymed? |
|---|---|---|---|---|
| `flame` | `Beside it someone carved a ___` | `name` | `name` | yes |
| `stream` | `The child awoke from one last ___` | `dream` | `dream` | yes |
| `light` | `An owl went hunting through the ___` | `trees` | `night` | no; `night` ranked fifth |
| `moon` | `The old musician played a ___` | `song` | `tune` | no; `tune` ranked fourth |
| `fire` | `The singer spoke of lost ___` | `love` | `desire` | no |

The instructions had little effect: explicitly telling the model to rhyme gave
32% top-1 accuracy, compared with 28% without an instruction. Pythia-410M is a
base language model, not an instruction-tuned model, so this is unsurprising and
should not be interpreted as an instruction-following result.

## What this result does and does not show

The model has enough rhyme behavior to investigate: greedy output rhymes above
zero, and rhymes often occur near the top of the distribution. But this benchmark
does **not yet isolate a rhyme mechanism**. Many reference words are also strongly
predicted by ordinary meaning. For example, `settled down to sleep` needs no
rhyme reasoning. Conversely, some lines have a more natural non-rhyming answer,
such as `played a song` rather than `played a tune`.

There is also no matched non-poetry baseline yet, so 28% cannot currently be
described as an increase caused by the anchor word. Counterfactual prompts that
change only the anchor are needed for that causal claim.

## Causal-analysis engineering check

The layer, head, and activation-patching code ran successfully against the real
model. A scan on one `light`/`night` example nominated layers and heads whose
ablation changed rhyme probability, but a one-example scan is not scientific
evidence that those components implement rhyming. These runs establish that the
instrumentation works; component claims require aggregation over many controlled
counterfactual examples.

## Recommended next behavioral experiment

Longer contexts are the right next step, but they should demonstrate a pattern
rather than verbally instruct this base model. Provide two or three completed
rhyming couplets followed by a final incomplete couplet:

```text
The rain made mirrors in the lane
At dusk there came a distant train

The fire faded to a glow
Beyond the glass descended snow

The window held a square of light
An owl went hunting through the
```

Then compare the same target couplet under:

1. no preceding examples;
2. preceding rhyming couplets;
3. preceding non-rhyming couplets with matched vocabulary and length;
4. preceding couplets with conflicting rhyme schemes;
5. counterfactual versions where only the relevant anchor changes.

This should tell us whether in-context poetic structure increases greedy rhyme
accuracy and whether the model follows an demonstrated `AA`, `AABB`, or `ABAB`
scheme. Dataset generation should prioritize semantically neutral final prefixes,
so changing the anchor changes which rhyme family is preferred without changing
the grammar of the missing word.

## Reproduction

```bash
.venv/bin/python -m pytest -q
.venv/bin/rhyme-interp dataset
.venv/bin/rhyme-interp evaluate
.venv/bin/rhyme-interp layers --example 0
.venv/bin/rhyme-interp heads --example 0
.venv/bin/rhyme-interp patch --example 0 --source 1
```

Raw outputs are in the gitignored `artifacts/` directory. The fixed benchmark is
`data/controlled_couplets.jsonl`.

