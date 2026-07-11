"""Tests for rhyme-family lexicon construction."""

from rhyme_interp.families import (
    RhymeFamily,
    build_families,
    cross_spelling_families,
    rime_spelling,
    unambiguous_rhyme_key,
)
from rhyme_interp.rhyme import rhymes


def test_rime_spelling_handles_silent_e_and_digraphs():
    assert rime_spelling("side") == "ide"
    assert rime_spelling("case") == "ase"
    assert rime_spelling("night") == "ight"
    assert rime_spelling("know") == "ow"
    assert rime_spelling("grow") == "ow"
    assert rime_spelling("though") == "ough"
    assert rime_spelling("way") == "ay"


def test_unambiguous_rhyme_key():
    assert unambiguous_rhyme_key("rain") == ("EY1", "N")
    # `read` rhymes with both `reed` and `red`, so it is ambiguous.
    assert unambiguous_rhyme_key("read") is None


def test_build_families_members_rhyme_and_are_ranked():
    words = [
        "rain", "train", "brain", "main", "pain", "chain", "lane",
        "glow", "snow", "flow", "show", "know", "grow",
        "xylophone",
    ]
    token_words = dict(enumerate(words))
    families = build_families(token_words, min_zipf=2.0, min_size=6, max_size=6, max_families=10)
    assert len(families) == 2
    for family in families:
        for word in family.words[1:]:
            assert rhymes(family.words[0], word)


def test_cross_spelling_selection():
    family = RhymeFamily(("EY1", "N"), ("rain", "train", "main", "lane", "cane"))
    single = RhymeFamily(("OW1",), ("glow", "snow", "flow"))
    assert family.spelling_groups() == {"ain": ("rain", "train", "main"), "ane": ("lane", "cane")}
    assert cross_spelling_families([family, single]) == [family]
