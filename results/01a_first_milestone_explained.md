# Can Pythia-410M finish a poem with a rhyme?

## The short answer

Sometimes, but not reliably yet.

We showed Pythia-410M a two-line poem with its final word removed. When we always
selected the model's single most likely next word, that word rhymed with the end
of the first line in **7 out of 25 plain poems**.

The model often considered a rhyme without selecting it. A rhyming word appeared
among its ten most likely choices in **17 out of 25 plain poems**. That is enough
behavior to make an interpretability experiment plausible, but the current poems
do not yet prove that the model used the first line to choose the rhyme.

## What exactly did we ask the model to do?

Here is one complete model input:

```text
The window held a square of light
An owl went hunting through the
```

We stop immediately after `the` and ask the model for probabilities for the next
token. We hope for a word such as `night`, because it completes the sentence and
rhymes with `light`.

The model's most likely answer was actually `trees`:

```text
An owl went hunting through the trees
```

That is a sensible sentence, but it does not rhyme. `night` was the model's fifth
choice. So the model had the desired answer available, but ordinary sentence
meaning won over the rhyme.

Here is a success:

```text
The willows leaned above the stream
The child awoke from one last dream
```

After seeing everything through `last`, the model's first choice was `dream`.
That word both makes sense and rhymes with `stream`.

## What does “28% top-1 exact rhyme” mean?

“Top-1” is simply the model's first choice: the next token with the highest
probability. If we generate greedily, that is the token we obtain.

“Exact rhyme” means that the pronunciation matches from the last stressed vowel
onward. We use a pronunciation dictionary, so rhyme is not judged merely by how
words are spelled.

We used 25 poems and presented each one in eight slightly different ways. For
example, one version began directly with the poem, while another began with
`Poem:`. Across those 200 presentations, the model's first choice rhymed 56
times, or 28%. On just the 25 plain versions, it rhymed 7 times—also 28%.

This measures only the single missing word. We did not generate whole poems and
find that 28% of their lines rhymed.

## Why did we have an “expected word”?

When writing each test poem, we wrote down the completion we intended. For the
`light` example, it was `night`; for the `stream` example, it was `dream`.

That word is better called the **reference completion** than the “expected” or
“target” word. It gives us a convenient way to measure whether the model found
the particular answer the author designed. It is not the only correct answer.
If the model produced another sensible word that rhymed, we counted it as a rhyme
even when it differed from the reference completion.

## What does “reference completion median rank of 5” mean?

For each poem, imagine sorting all possible next tokens by the probability the
model assigned them:

```text
1. trees
2. darkness
3. dark
4. gloom
5. night
```

In this example, the reference word `night` has rank 5. Across the full test set,
the median reference-word rank was 5. In ordinary language: the word we designed
was generally near the top of the model's choices, but often was not its first
choice.

This is not our most important metric, because poetry can have multiple valid
completions. The rhyme status of the model's own first choice matters more.

## Why is the rhyme rate only 28%?

There are at least four likely reasons:

1. **The context is very short.** The model sees only one completed line, so it
   may not confidently infer that a poem or rhyme pattern is underway.
2. **Meaning competes with rhyme.** `The musician played a song` is more ordinary
   than `The musician played a tune`, even though only `tune` rhymes with `moon`.
3. **Some of our test lines make the reference answer too easy or too awkward.**
   `settled down to sleep` is predictable without looking at the rhyme anchor,
   whereas other intended completions lose to more natural non-rhyming words.
4. **Pythia-410M is small.** It may represent rhyme weakly or inconsistently.

Most importantly, 28% is not yet a clean measurement of “how well Pythia can
rhyme.” It is the result on this first small benchmark.

## Did asking it to rhyme help?

Hardly. Adding an explicit sentence that told the model to rhyme changed the
score from 28% to 32%.

That is expected: Pythia is a base model trained to continue text, not an
instruction-tuned assistant trained to obey requests. A better way to communicate
the task is to show it several completed examples of the pattern we want.

## What should we try next?

Yes—we should try harder to elicit the behavior before interpreting it.

The strongest next test is a longer poem containing several demonstrated rhymes:

```text
The rain made mirrors in the lane
At dusk there came a distant train

The fire faded to a glow
Beyond the glass descended snow

The window held a square of light
An owl went hunting through the
```

This gives the base model a visible continuation pattern rather than an
instruction. We should compare it against the same final couplet preceded by
non-rhyming lines. If completed rhyme examples increase the probability of words
rhyming with `light`, that is much stronger evidence that the model recognizes
and continues poetic structure.

After couplets, we can test `AABB`, `ABAB`, and `ABBA` schemes. Those tests can
separate two possible jobs inside the model:

- deciding **which earlier line ending** the new word should rhyme with; and
- selecting **a word from that rhyme family** that also fits the sentence.

Only after the behavior is stronger and the controls are clean should we make
claims about particular attention heads or MLPs implementing rhyme.
