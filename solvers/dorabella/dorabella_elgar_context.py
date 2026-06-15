"""Contextual scoring for Elgar/Dorabella-style candidates.

This is intentionally a soft prior. It must never override the hard cipher
constraints: rotation, orientation alternatives, skeleton anchors, and
bijection remain decisive.
"""

from __future__ import annotations

from dataclasses import dataclass

from .dorabella_data import canonicalize_text


def canonical_word(word: str) -> str:
    return canonicalize_text(word)


LETTER_WORDS = {
    "A", "AM", "AN", "AND", "ARE", "AS", "AT", "BE", "BUT", "BY", "CAN",
    "COME", "DEAR", "DO", "FOR", "FROM", "GO", "GOOD", "HAVE", "HE",
    "HERE", "I", "IF", "IN", "IS", "IT", "ME", "MY", "NO", "NOT", "NOW",
    "OF", "ON", "OR", "SEE", "SO", "THE", "THEN", "THIS", "TO", "WE",
    "WILL", "WITH", "YOU", "YOUR", "YOURS",
}

ELGAR_CONTEXT_WORDS = {
    "ALICE", "BELLA", "DORA", "DORABELLA", "EDWARD", "ELGAR", "MARCO",
}

MUSIC_WORDS = {
    "AIR", "BAR", "BARS", "BASS", "CHORD", "CHORDS", "DANCE", "ENIGMA",
    "HARMONY", "KEY", "LILT", "MELODY", "MUSIC", "NOTE", "NOTES", "PLAY",
    "RHYTHM", "SCORE", "SONG", "SOUND", "THEME", "TUNE", "TUNES", "VOICE",
}

VICTORIAN_SOCIAL_WORDS = {
    "BICYCLE", "DAY", "GARDEN", "HOME", "HOUSE", "LETTER", "LONDON",
    "MORNING", "NIGHT", "OLD", "ROAD", "TEA", "TODAY", "TOMORROW", "WALK",
}

WEIGHTED_CONTEXT = {
    **{canonical_word(w): 0.20 for w in LETTER_WORDS},
    **{canonical_word(w): 0.35 for w in VICTORIAN_SOCIAL_WORDS},
    **{canonical_word(w): 0.45 for w in MUSIC_WORDS},
    **{canonical_word(w): 0.60 for w in ELGAR_CONTEXT_WORDS},
}


@dataclass(frozen=True)
class ContextScore:
    score: float
    hits: list[str]
    notes: str


def score_segmented_words(words: list[str]) -> ContextScore:
    hits: list[str] = []
    score = 0.0
    for word in words:
        canon = canonical_word(word)
        weight = WEIGHTED_CONTEXT.get(canon)
        if weight is not None:
            score += weight
            hits.append(word)
    normalized = score / max(len(words), 1)
    return ContextScore(
        score=normalized,
        hits=hits,
        notes="context hits: " + (", ".join(hits) if hits else "none"),
    )
