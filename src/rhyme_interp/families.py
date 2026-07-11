"""Rhyme-family lexicon construction for representation experiments.

A rhyme family is the set of single-token English words that share phonemes
from the last stressed vowel onward. Families are the unit of analysis for
probing, geometry, and steering experiments, so members must be unambiguous:
words with several distinct rhyme pronunciations are excluded.

Each family also records rime-spelling subgroups (`rain/train` versus
`lane/cane` versus `reign`). Training a probe on one spelling and testing on
another separates phonological from orthographic representation.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from wordfreq import zipf_frequency

from .rhyme import rhyme_keys

# Frequent vocabulary artifacts that are not poetic words.
STOPLIST = {"lol", "pre", "http", "cas", "sas", "pac", "mac", "san", "jun", "nfl"}

_RIME_SPELLING = re.compile(r"[aeiouy]+[^aeiouy]*e?$")


def rime_spelling(word: str) -> str:
    """The written form of the final rime: last vowel-letter group onward.

    A trailing silent `e` is included, so `side -> ide` and `case -> ase`.
    """
    match = _RIME_SPELLING.search(word)
    return match.group(0) if match else word


@dataclass(frozen=True)
class RhymeFamily:
    key: tuple[str, ...]
    words: tuple[str, ...]

    @property
    def name(self) -> str:
        return "-".join(self.key)

    def spelling_groups(self) -> dict[str, tuple[str, ...]]:
        """Group members by how their rime is written."""
        groups: dict[str, list[str]] = {}
        for word in self.words:
            groups.setdefault(rime_spelling(word), []).append(word)
        return {suffix: tuple(words) for suffix, words in groups.items()}


def unambiguous_rhyme_key(word: str) -> tuple[str, ...] | None:
    keys = rhyme_keys(word)
    return next(iter(keys)) if len(keys) == 1 else None


def _vowel_count(key: tuple[str, ...]) -> int:
    return sum(1 for phone in key if phone[-1:].isdigit())


def build_families(
    token_words: dict[int, str],
    min_zipf: float = 3.3,
    min_size: int = 6,
    max_size: int = 12,
    max_families: int = 30,
) -> list[RhymeFamily]:
    """Build balanced rhyme families from the model's single-token vocabulary.

    Only final-syllable rimes (a single stressed vowel in the key) qualify, so
    morphological families such as `-ology` or `-action` cannot masquerade as
    phonological structure. Words shorter than three letters are dropped to
    exclude interjections and abbreviations. Members are kept in descending
    frequency order. Families are ranked by the frequency of their
    `min_size`-th member, which favors families made of genuinely common words
    over families padded with one frequent word.
    """
    by_key: dict[tuple[str, ...], dict[str, float]] = {}
    for word in set(token_words.values()):
        if len(word) < 3 or word in STOPLIST or not re.search(r"[aeiouy]", word):
            continue
        frequency = zipf_frequency(word, "en")
        if frequency < min_zipf:
            continue
        key = unambiguous_rhyme_key(word)
        if key is None or _vowel_count(key) != 1:
            continue
        by_key.setdefault(key, {})[word] = frequency

    families = []
    for key, members in by_key.items():
        if len(members) < min_size:
            continue
        ranked = sorted(members, key=lambda word: (-members[word], word))
        families.append((members[ranked[min_size - 1]], RhymeFamily(key, tuple(ranked[:max_size]))))
    families.sort(key=lambda item: (-item[0], item[1].name))
    return [family for _rank, family in families[:max_families]]


def cross_spelling_families(
    families: list[RhymeFamily], minimum_alternative: int = 2
) -> list[RhymeFamily]:
    """Families whose rime has at least two spellings, each with >= `minimum_alternative` words."""
    selected = []
    for family in families:
        groups = [g for g in family.spelling_groups().values() if len(g) >= minimum_alternative]
        if len(groups) >= 2:
            selected.append(family)
    return selected
