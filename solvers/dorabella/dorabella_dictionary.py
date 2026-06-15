"""Dictionary expansion for Dorabella word hypotheses.

The solver should not depend on a tiny hand-picked vocabulary. This module
loads broad local English wordlists when available, classifies plausible nouns
and verbs with conservative heuristics, and adds a stronger period prior for
Victorian/letter-writing vocabulary.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from .dorabella_elgar_context import (
    ELGAR_CONTEXT_WORDS,
    LETTER_WORDS,
    MUSIC_WORDS,
    VICTORIAN_SOCIAL_WORDS,
    canonical_word,
)


@dataclass(frozen=True)
class DictionaryEntry:
    internal: str
    display: str
    pos: str
    source: str
    weight: float
    period_score: float


COMMON_NOUNS = {
    "AIR", "ANSWER", "ARM", "BELL", "BOOK", "BOX", "BOY", "CALL", "CASE",
    "CHAIR", "CHANCE", "CHILD", "DAY", "DOOR", "END", "EYE", "FACE",
    "FAMILY", "FRIEND", "GIRL", "HAND", "HEAD", "HEART", "HOME", "HOUR",
    "HOUSE", "KIND", "KIT", "LADY", "LETTER", "LIFE", "LIGHT", "LINE",
    "MAN", "MIND", "MOMENT", "MORNING", "NAME", "NIGHT", "PART", "PLACE",
    "QUESTION", "ROAD", "ROOM", "SIDE", "SOUND", "THING", "TIME", "VOICE",
    "WAY", "WORD", "WORK", "WORLD", "YEAR",
}

COMMON_VERBS = {
    "ASK", "BE", "CALL", "COME", "DO", "FEEL", "FIND", "GET", "GIVE",
    "GO", "HAVE", "HEAR", "HELP", "HOLD", "KEEP", "KNOW", "LEAVE", "LET",
    "LIKE", "LOOK", "LOVE", "MAKE", "MEAN", "MEET", "PLAY", "READ", "SEE",
    "SEEM", "SEND", "SING", "SPEAK", "TAKE", "TALK", "TELL", "THINK",
    "TRY", "WALK", "WANT", "WISH", "WRITE",
}

VICTORIAN_STYLE_WORDS = {
    "ACQUAINT", "ACQUAINTED", "AFFECTION", "AGAIN", "ANSWER", "APOLOGY",
    "APPOINT", "ARRANGE", "ARRIVED", "BICYCLE", "CALL", "CALLING", "CAME",
    "CHARM", "DEAR", "DELIGHT", "DINNER", "DORA", "DORABELLA", "EDWARD",
    "ELGAR", "EVENING", "FANCY", "FINE", "FRIEND", "GARDEN", "GOOD",
    "HAPPY", "HOME", "HOUSE", "KIND", "LETTER", "LONDON", "MORNING",
    "LIKE", "LOVE", "MUSIC", "NOTE", "PLEASE", "PRETTY", "REGARD",
    "REGARDS", "ROAD", "SINCERE", "TALK", "TEA", "THANK", "THANKS",
    "TODAY", "TOMORROW", "VISIT", "WALK", "WISH", "YOURS",
}

EXPERIMENTAL_UV_WORDS: set[str] = set()

NOUN_SUFFIXES = (
    "AGE", "ANCE", "ENCE", "DOM", "ER", "ERY", "ESS", "HOOD", "ION",
    "ISM", "IST", "ITY", "MENT", "NESS", "OR", "SHIP", "TION", "URE",
)
VERB_SUFFIXES = (
    "ATE", "ED", "EN", "FY", "IED", "ING", "ISE", "IZE",
)
ADJECTIVE_OR_ADVERB_SUFFIXES = (
    "ABLE", "AL", "ARY", "FUL", "IC", "ICAL", "ISH", "IVE", "LESS", "LY",
    "OUS", "Y",
)

DEFAULT_WORDLISTS = (
    "/usr/share/dict/web2",
    "/usr/share/dict/words",
)

REJECT_EXACT = {
    "AA", "AAL", "AALII", "AB", "ABA", "ABAC", "ABACA",
}


def configured_wordlist_paths() -> tuple[Path, ...]:
    raw = os.environ.get("DORABELLA_DICTIONARY", "")
    paths = [Path(item).expanduser() for item in raw.split(os.pathsep) if item.strip()]
    paths.extend(Path(path) for path in DEFAULT_WORDLISTS)
    deduped = []
    seen = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return tuple(deduped)


def clean_word(raw: str) -> tuple[str, str] | None:
    word = raw.strip()
    if not word or not re.fullmatch(r"[A-Za-z]+", word):
        return None
    if word[0].isupper() and word[1:].islower():
        # Avoid flooding the search with proper names from Webster. Known names
        # are already supplied through Elgar/context vocabularies.
        if canonical_word(word) not in ELGAR_CONTEXT_WORDS:
            return None
    canon = canonical_word(word)
    if canon in REJECT_EXACT or len(canon) < 2 or len(canon) > 10:
        return None
    return canon, word.upper()


def plausible_shape(word: str) -> bool:
    vowels = sum(ch in "AEIOUY" for ch in word)
    if vowels == 0:
        return False
    vowel_ratio = vowels / len(word)
    if vowel_ratio < 0.18 or vowel_ratio > 0.78:
        return False
    if re.search(r"[^AEIOUY]{5,}", word):
        return False
    if sum(ch in "QXZ" for ch in word) >= 2:
        return False
    return True


def guess_pos(word: str) -> str | None:
    if word in COMMON_NOUNS or word in ELGAR_CONTEXT_WORDS or word in MUSIC_WORDS or word in VICTORIAN_SOCIAL_WORDS:
        return "noun"
    if word in COMMON_VERBS:
        return "verb"
    if word.endswith(VERB_SUFFIXES) and len(word) >= 4:
        return "verb"
    if word.endswith(NOUN_SUFFIXES) and len(word) >= 4:
        return "noun"
    if word.endswith("S") and len(word) > 3 and not word.endswith(("SS", "US")):
        return "noun"
    if word.endswith(ADJECTIVE_OR_ADVERB_SUFFIXES):
        return None
    if 3 <= len(word) <= 8 and plausible_shape(word):
        # Webster/web2 contains many noun/verb base forms without suffixes.
        # Keep them, but with lower confidence than known POS words.
        return "noun_or_verb"
    return None


def period_score(word: str, source: str) -> float:
    score = 0.0
    if source == "web2":
        score += 0.08
    if word in VICTORIAN_STYLE_WORDS:
        score += 0.40
    if word in VICTORIAN_SOCIAL_WORDS:
        score += 0.35
    if word in MUSIC_WORDS:
        score += 0.30
    if word in ELGAR_CONTEXT_WORDS:
        score += 0.55
    if word in LETTER_WORDS:
        score += 0.18
    if word in COMMON_NOUNS or word in COMMON_VERBS:
        score += 0.12
    if source == "uv_experimental":
        score += 0.26
    return score


def entry_weight(word: str, pos: str, source: str) -> float:
    base = 0.46 if pos == "noun_or_verb" else 0.58
    if word in COMMON_NOUNS or word in COMMON_VERBS:
        base += 0.18
    if source == "curated":
        base += 0.22
    base += min(len(word), 8) * 0.018
    base += period_score(word, source) * 0.25
    return min(base, 1.18)


def curated_entries() -> Iterable[DictionaryEntry]:
    words = (
        set(COMMON_NOUNS)
        | set(COMMON_VERBS)
        | set(VICTORIAN_STYLE_WORDS)
        | set(LETTER_WORDS)
        | set(ELGAR_CONTEXT_WORDS)
        | set(MUSIC_WORDS)
        | set(VICTORIAN_SOCIAL_WORDS)
    )
    for display_word in sorted(words):
        word = canonical_word(display_word)
        pos = guess_pos(word) or "noun_or_verb"
        yield DictionaryEntry(
            internal=word,
            display=display_word,
            pos=pos,
            source="curated",
            weight=entry_weight(word, pos, "curated"),
            period_score=period_score(word, "curated"),
        )
    for display_word in sorted(EXPERIMENTAL_UV_WORDS):
        word = canonical_word(display_word)
        pos = guess_pos(word) or "noun_or_verb"
        yield DictionaryEntry(
            internal=word,
            display=display_word,
            pos=pos,
            source="uv_experimental",
            weight=entry_weight(word, pos, "uv_experimental"),
            period_score=period_score(word, "uv_experimental"),
        )


@lru_cache(maxsize=1)
def load_dictionary_entries() -> tuple[DictionaryEntry, ...]:
    entries: dict[str, DictionaryEntry] = {entry.internal: entry for entry in curated_entries()}
    for path in configured_wordlist_paths():
        if not path.exists() or not path.is_file():
            continue
        source = "web2" if path.name in {"web2", "words"} else path.stem
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for line in lines:
            cleaned = clean_word(line)
            if cleaned is None:
                continue
            word, display = cleaned
            pos = guess_pos(word)
            if pos is None:
                continue
            weight = entry_weight(word, pos, source)
            score = period_score(word, source)
            existing = entries.get(word)
            if existing is not None and existing.weight >= weight:
                continue
            entries[word] = DictionaryEntry(
                internal=word,
                display=display,
                pos=pos,
                source=source,
                weight=weight,
                period_score=score,
            )
    return tuple(sorted(entries.values(), key=lambda item: (item.internal, item.source)))


@lru_cache(maxsize=1)
def entries_by_length() -> dict[int, tuple[DictionaryEntry, ...]]:
    grouped: dict[int, list[DictionaryEntry]] = {}
    for entry in load_dictionary_entries():
        grouped.setdefault(len(entry.internal), []).append(entry)
    return {length: tuple(items) for length, items in grouped.items()}
