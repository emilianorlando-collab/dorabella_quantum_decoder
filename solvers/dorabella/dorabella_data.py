"""Static Dorabella transcription and configurable hypothesis data.

The public repository ships with the symbol transcription, alphabet cycle, and
alphabet-1 mapping, but intentionally does not include private plaintext
hypotheses. To run a private experiment, create ``dorabella_private_data.py`` in
this package and define SKELETONS, START_ANCHOR, or END_ANCHOR there.
"""

from __future__ import annotations

from dataclasses import dataclass


LETTER_EQUIVALENCE_CLASSES = {
    "I": ("I", "J"),
    "U": ("U", "V"),
}
"""Canonical letter classes that Elgar's alphabet merges."""

LETTER_TO_CANONICAL = {
    variant: canonical
    for canonical, variants in LETTER_EQUIVALENCE_CLASSES.items()
    for variant in variants
}

CANONICAL_LETTERS = "".join(
    ch for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if LETTER_TO_CANONICAL.get(ch, ch) == ch
)
"""24-letter alphabet with I/J and U/V merged."""

CANONICAL_DISPLAY = {
    canonical: "/".join(variants)
    for canonical, variants in LETTER_EQUIVALENCE_CLASSES.items()
}


def canonicalize_letter(ch: str) -> str:
    up = ch.upper()
    return LETTER_TO_CANONICAL.get(up, up)


def canonicalize_text(text: str) -> str:
    return "".join(canonicalize_letter(ch) for ch in text.upper() if ch.isalpha())


def display_letter(ch: str) -> str:
    canonical = canonicalize_letter(ch)
    return CANONICAL_DISPLAY.get(canonical, canonical)

ROW_SYMBOLS = [
    [
        "2E", "3W", "2SE", "3E", "1E", "2S", "1N", "3E", "1SW",
        "2E/2NE", "3SE", "2NW", "1N/1NW", "1S/1SE", "2NW", "3S",
        "2NW/2N", "2NW", "2S", "3W", "3W", "2N", "1NW/1N",
        "1NE/1N", "2E/2NE", "1NE/1N", "1S", "3SE", "3NW",
    ],
    [
        "1N", "2NW/2N", "1N", "2S", "1NE", "3E", "1SW", "2SW",
        "3E", "2SE", "2NW", "2NW", "2SE", "2S", "1S", "1NW/1N",
        "1N", "2NW", "3SE", "2NW", "2S", "2N", "3NW", "1NW/1N",
        "1SW/1S", "1NE", "1SW", "1SW", "1NE/1E", "3SE", "3NW",
    ],
    [
        "2SE", "3NW", "2S", "2N", "3NW", "2SE", "1S/1SE", "2N",
        "3N", "1S", "3NW", "2S", "2NW", "2S", "2N", "1NW/1N",
        "3NW", "1S", "3E", "3W", "1S", "3NW", "2S", "3E",
        "1S", "1E", "3E",
    ],
]

SKELETONS = [
    "_" * 29,
    "_" * 31,
    "_" * 27,
]

ROW_LENGTHS = [29, 31, 27]

ALPHABET1_MAP = {
    "1E": "A",
    "2E": "B",
    "3E": "C",
    "1SE": "D",
    "2SE": "E",
    "3SE": "F",
    "1S": "G",
    "2S": "H",
    "3S": "I",
    "1SW": "K",
    "2SW": "L",
    "3SW": "M",
    "1W": "N",
    "2W": "O",
    "3W": "P",
    "1NW": "Q",
    "2NW": "R",
    "3NW": "S",
    "1N": "T",
    "2N": "U",
    "3N": "W",
    "1NE": "X",
    "2NE": "Y",
    "3NE": "Z",
}

DIRECTIONS = ("E", "SE", "S", "SW", "W", "NW", "N", "NE")


@dataclass(frozen=True)
class DorabellaPosition:
    row: int
    col: int
    global_index: int
    alphabet_index: int
    symbol_options: tuple[str, ...]
    allowed_letters: frozenset[str] | None


START_ANCHOR = None
END_ANCHOR = None


try:
    from . import dorabella_private_data as _private_data
except ImportError:
    _private_data = None

if _private_data is not None:
    ROW_SYMBOLS = getattr(_private_data, "ROW_SYMBOLS", ROW_SYMBOLS)
    SKELETONS = getattr(_private_data, "SKELETONS", SKELETONS)
    ALPHABET1_MAP = getattr(_private_data, "ALPHABET1_MAP", ALPHABET1_MAP)
    START_ANCHOR = getattr(_private_data, "START_ANCHOR", START_ANCHOR)
    END_ANCHOR = getattr(_private_data, "END_ANCHOR", END_ANCHOR)
