"""Pronunciation-aware rhyme utilities."""

from __future__ import annotations

import re
from functools import lru_cache

import cmudict

_WORD = re.compile(r"^[a-z]+$")


@lru_cache(maxsize=1)
def pronunciations() -> dict[str, list[list[str]]]:
    return cmudict.dict()


def normalize_word(word: str) -> str:
    return re.sub(r"[^a-z]", "", word.lower())


def rhyme_keys(word: str) -> set[tuple[str, ...]]:
    """Return phonemes from the last stressed vowel onward for every pronunciation."""
    word = normalize_word(word)
    keys: set[tuple[str, ...]] = set()
    for phones in pronunciations().get(word, []):
        stressed = [i for i, phone in enumerate(phones) if phone[-1:] in {"1", "2"}]
        if stressed:
            keys.add(tuple(phones[stressed[-1] :]))
    return keys


def rhymes(left: str, right: str) -> bool:
    """Whether two distinct words have an exact CMUdict rhyme."""
    left, right = normalize_word(left), normalize_word(right)
    return bool(left and right and left != right and rhyme_keys(left) & rhyme_keys(right))


def single_token_words(tokenizer) -> dict[int, str]:
    """Map vocabulary IDs that decode to a standalone lowercase English word."""
    result: dict[int, str] = {}
    for token_id in range(len(tokenizer)):
        text = tokenizer.decode([token_id])
        if text.startswith(" ") and _WORD.fullmatch(text[1:].lower()):
            word = text[1:].lower()
            if word in pronunciations():
                result[token_id] = word
    return result


def rhyme_token_ids(anchor: str, token_words: dict[int, str]) -> list[int]:
    return [token_id for token_id, word in token_words.items() if rhymes(anchor, word)]

