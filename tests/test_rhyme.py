from rhyme_interp.dataset import (
    DISTINCT_DEMONSTRATION_COUPLETS,
    build_counterfactual_dataset,
    build_dataset,
    build_distinct_elicitation_dataset,
    build_elicitation_dataset,
)
from rhyme_interp.rhyme import rhyme_keys, rhymes


def test_exact_rhymes_are_phonetic():
    assert rhymes("light", "night")
    assert rhymes("blue", "through")
    assert not rhymes("light", "moon")
    assert not rhymes("night", "night")


def test_rhyme_key_starts_at_last_stress():
    assert ("EY1", "N") in rhyme_keys("rain")


def test_dataset_size_and_endings():
    examples = build_dataset()
    assert len(examples) == 200
    for example in examples:
        assert example.first_line.lower().endswith(example.anchor)
        assert rhymes(example.anchor, example.target)
        assert not example.final_prefix.lower().endswith(example.target)
        assert not example.prompt.endswith(" ")


def test_counterfactual_prompts_have_identical_scaffolds():
    examples = build_counterfactual_dataset()
    assert len(examples) == 25
    assert examples[0].prompt.replace("light", "moon") == examples[1].prompt


def test_benchmark_anchor_and_target_are_plain_words():
    for example in build_dataset():
        assert example.anchor.isalpha() and example.target.isalpha()


def test_elicitation_controls_preserve_target_prompt_suffix():
    conditions = {name: build_elicitation_dataset(name) for name in ["plain", "rhyming", "shuffled", "reversed"]}
    assert all(len(examples) == 25 for examples in conditions.values())
    for index in range(25):
        suffix = conditions["plain"][index].prompt
        assert all(examples[index].prompt.endswith(suffix) for examples in conditions.values())
        assert conditions["rhyming"][index].anchor == conditions["shuffled"][index].anchor


def test_distinct_demonstrations_are_unique_exact_rhymes():
    assert len(DISTINCT_DEMONSTRATION_COUPLETS) == len(set(DISTINCT_DEMONSTRATION_COUPLETS))
    endings = []
    for first, second in DISTINCT_DEMONSTRATION_COUPLETS:
        left, right = first.split()[-1], second.split()[-1]
        assert rhymes(left, right)
        endings.append((left, right))
    assert len(endings) == len(set(endings))


def test_distinct_shuffled_control_preserves_target_suffix_and_length():
    rhyming = build_distinct_elicitation_dataset(20)
    shuffled = build_distinct_elicitation_dataset(20, shuffled=True)
    plain = build_elicitation_dataset("plain")
    for normal, control, target in zip(rhyming, shuffled, plain):
        assert normal.prompt.endswith(target.prompt)
        assert control.prompt.endswith(target.prompt)
        assert len(normal.prompt.splitlines()) == len(control.prompt.splitlines()) == 42
