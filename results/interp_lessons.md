# Lessons for interpretability work

Distilled from the Gemma-4 rhyme study (reports 01–10). These are the moves that
actually changed outcomes, written so a future project can reuse them. Most are
about *where to look* and *how to know you're right*, not about any one technique.

## Choosing what to probe

**1. Probe where a component reads, not the raw residual.** The single biggest
lever in this project: a rhyme-family readout was 0.30 on the layer-13 residual
and 0.90 on the layer-14 value memory the retrieval head actually consumes — same
words, same probe. The raw residual is dominated by unrelated content (lexical
identity, position, massive-activation dims); the value projection is a learned
map that keeps only what the downstream component needs. When a readout is weak,
suspect the *representation*, not the feature — move to the exact tensor a real
mechanism reads.

**2. A feature can be a small fraction of the variance and still be the whole
story.** The rhyme direction is a low-rank (~4–16 dim) slice of a 1536-dim
residual. Dominant-variance methods (a plain SAE, raw nearest-neighbour) will
miss it and hand you the majority structure instead (word stems). Isolate the
subspace of interest before analysing it.

## Causal rigor

**3. Decodability ≠ causal importance. Always pair a probe with an ablation.**
The 16 most family-*selective* neurons read the family at 0.79, but ablating them
cost only 6% of the behaviour. Readable and load-bearing are different neurons.
The causal ones were found by *output-weighted* attribution — what a neuron
writes into the target direction, not how much its activation varies — which
tripled the ablation effect (66% vs 23%).

**4. Localize with ablation; then verify by operating on it.** The write was
pinned to one sublayer because ablating just the layer-13 MLP cost 73% while any
single neuron cost ~nothing. The convincing proof, though, was a training-free
rank-1 weight edit that installed a *false* rhyme (`month → light`) with zero
collateral. If a localization is real, you can edit it, not just read it — and a
clean edit with no side effects is worth more than any probe.

**5. Causal effects are usually direction-shaped, not neuron-shaped.** No small
neuron set was a bottleneck because the write is a direction carried in
superposition. This is why the rank-1 (one-direction) edit worked and why hunting
for "the -T neuron" failed. Reach for directional interventions (project a
subspace in/out) before neuron-level ones.

## Designing interventions

**6. Know the architecture's normalization before you edit weights.** Gemma
applies RMSNorm *after* the MLP, which fixes the magnitude of any injected term.
So a weight edit controls the *direction* of the code but not its strength — and a
token with an overwhelming prior (`orange`) can't be overpowered by one MLP's
worth of signal. The norm structure determined what the edit could and couldn't
do; read it first.

**7. Interventions fire only in the context the behaviour lives in.** The rhyme
code is much weaker for a word in a neutral sentence than in a rhyming one, and
the fake-rhyme patch did *nothing* without a rhyme-mode preamble. That dependence
was also a *confirmation* we were driving the right pathway. Probe and intervene
where the behaviour actually happens; a null in the wrong context means nothing.

**8. Separate the mechanism from the behaviour.** The routing key-swap re-pointed
the head's attention in *every* example, but only changed the emitted word when
no strong completion overrode it. The mechanism was reliable; the behaviour was
contingent. Report both, and don't let a noisy behavioural readout hide a clean
mechanistic one.

## Epistemics and pitfalls

**9. Trust contrasts, not absolute numbers.** Probe accuracies are sensitive to
the pipeline and the number of classes. We once quoted a "0.29 → 0.75" context
effect that turned out to be two different pipelines; the matched comparison was a
modest 0.14 → 0.20. Fix one pipeline and compare within it; always include a
shuffled-label control.

**10. Check the confound before naming a feature.** SAE "vowel features" looked
monosemantic until we checked onset diversity — they fired for one word *stem*
(`be-`, `ba-`), not one phoneme. Top-activating examples mislead when the label
co-varies with something else. Require the variable you're claiming to *vary*
across the examples (here: diverse onsets, same coda).

**11. Bounded negative results are results — state their scope exactly.** "No
transferable single-position plan" is a finding; "the model does not plan" is an
overclaim of the same data. The SAE gave a partial, honest reconciliation rather
than a clean dictionary, and saying so kept it credible. Positive controls (an
intervention that *does* work) are what let a null mean something.

**12. Know when to stop.** The SAE smoke test answered the question well enough to
show a full corpus-scale effort would likely just re-confirm superposition.
Recognising a complete arc — and not tuning hyperparameters against a small
problem — is part of the craft.

## The through-line

The project worked because each claim was forced by evidence and then *checked by
acting on it*: find the circuit → localize the write by ablation → read it out
where the head reads → install a false version in the weights. If you can read a
representation, change it, and predict the behavioural consequence, you have
understood it. If you can only read it, you have a correlation.
