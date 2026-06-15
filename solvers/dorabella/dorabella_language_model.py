"""Lightweight English tokenization and grammar priors for Dorabella.

The cipher has no spaces or apostrophes, so this module treats tokenization as
a search problem. It supports normal words, contraction forms without
punctuation, and a conservative set of abbreviations that might appear in a
short informal note.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .dorabella_elgar_context import (
    ELGAR_CONTEXT_WORDS,
    LETTER_WORDS,
    MUSIC_WORDS,
    VICTORIAN_SOCIAL_WORDS,
    canonical_word,
)
from .dorabella_dictionary import load_dictionary_entries


@dataclass(frozen=True)
class Token:
    text: str
    display: str
    kind: str
    pos: str
    weight: float


@dataclass(frozen=True)
class Tokenization:
    tokens: list[Token]
    score: float
    notes: str
    diagnostics: dict[str, float] | None = None


CONTRACTIONS = {
    "IM": ("I'M", "pron_verb"),
    "ILL": ("I'LL", "pron_verb"),
    "IVE": ("I'VE", "pron_verb"),
    "ID": ("I'D", "pron_verb"),
    "YOUVE": ("YOU'VE", "pron_verb"),
    "YOULL": ("YOU'LL", "pron_verb"),
    "YOURE": ("YOU'RE", "pron_verb"),
    "YOUD": ("YOU'D", "pron_verb"),
    "HES": ("HE'S", "pron_verb"),
    "SHES": ("SHE'S", "pron_verb"),
    "ITS": ("IT'S", "pron_verb"),
    "WEVE": ("WE'VE", "pron_verb"),
    "WELL": ("WE'LL", "pron_verb"),
    "WERE": ("WE'RE", "pron_verb"),
    "THEYRE": ("THEY'RE", "pron_verb"),
    "DONT": ("DON'T", "verb"),
    "CANT": ("CAN'T", "verb"),
    "WONT": ("WON'T", "verb"),
    "ISNT": ("ISN'T", "verb"),
    "ARENT": ("AREN'T", "verb"),
    "WASNT": ("WASN'T", "verb"),
    "COULDNT": ("COULDN'T", "verb"),
    "WOULDNT": ("WOULDN'T", "verb"),
    "SHOULDNT": ("SHOULDN'T", "verb"),
}

ALLOW_ABBREVIATIONS = False

ABBREVIATIONS = {
    # Kept for explicit experiments only. Elgar's known plain examples suggest
    # direct language, so these are not inserted into LEXICON by default.
    "MR": ("Mr.", "noun"),
    "MRS": ("Mrs.", "noun"),
    "DR": ("Dr.", "noun"),
    "ST": ("St.", "noun"),
}

PRONOUNS = {"I", "YOU", "HE", "SHE", "IT", "WE", "THEY", "ME", "HER", "HIM", "US"}
DETERMINERS = {"A", "ALL", "AN", "ANY", "EACH", "NO", "SOME", "THE", "MY", "YOUR", "THIS", "THAT"}
PREPOSITIONS = {"AT", "BY", "FOR", "FROM", "IN", "OF", "ON", "TO", "WITH"}
CONJUNCTIONS = {"AND", "BUT", "IF", "OR", "SO", "THAN", "THEN"}
VERBS = {
    "AM", "ARE", "ASK", "BE", "CAN", "COME", "DO", "FEEL", "FIND", "GET",
    "GIVE", "GO", "HAVE", "HEAR", "HELP", "HOLD", "IS", "KEEP", "KNOW",
    "LEAVE", "LET", "LIKE", "LOOK", "LOVE", "MAKE", "MAY", "MEAN", "MEET",
    "MUST", "PLAY", "READ", "SEE", "SEEM", "SEND", "SING", "SPEAK", "TAKE",
    "TALK", "TELL", "THINK", "TRY", "WALK", "WANT", "WILL", "WISH", "WRITE",
    "ASKING", "COMING", "FEELING", "GOING", "HEARING", "KNOWING", "LOOKING",
    "PLAYING", "SEEING", "SENDING", "SINGING", "SPEAKING", "TAKING",
    "TALKING", "THINKING", "WALKING", "WISHING", "WRITING",
    "ASKED", "CAME", "DID", "FELT", "FOUND", "GAVE", "GOT", "HAD", "HEARD",
    "HELPED", "KNEW", "LEFT", "LOOKED", "MADE", "MEANT", "PLAYED", "SAID",
    "SAW", "SENT", "SANG", "SPOKE", "TAKEN", "TALKED", "THOUGHT", "TOLD",
    "TRIED", "WALKED", "WISHED", "WROTE",
}
NOUNS = {
    "AIR", "ALICE", "BAR", "BARS", "BELLA", "BICYCLE", "CHORD", "DAY",
    "DORA", "DORABELLA", "EDWARD", "ELGAR", "ENIGMA", "GARDEN", "HARMONY",
    "HOME", "HOUSE", "KEY", "KIT", "LADY", "LETTER", "LIFE", "LIGHT",
    "LINE", "LONDON", "MAN", "MARCO", "MELODY", "MIND", "MUSIC", "MORNING",
    "NAME", "NIGHT", "NOTE", "NOTES", "PLACE", "ROAD", "ROOM", "SCORE",
    "SONG", "SOUND", "TEA", "THEME", "THING", "TIME", "TODAY", "TOMORROW",
    "TUNE", "TUNES", "VOICE", "WAY", "WORD", "WORDS", "WORK", "WORLD",
}
ADJECTIVES = {
    "BAD", "DEAR", "FINE", "GOOD", "GREAT", "HAPPY", "LITTLE", "LONG",
    "OLD", "OWN", "PRETTY", "RIGHT", "SWEET", "TRUE", "VERY", "YOUNG",
}
ADVERBS = {"AGAIN", "ALSO", "AWAY", "HERE", "MORE", "MUCH", "NEVER", "NOW", "ONLY", "SOON", "THERE", "WELL"}
QUESTION_WORDS = {"HOW", "WHAT", "WHEN", "WHERE", "WHO", "WHY"}


def pos_for_word(word: str) -> str:
    canon = canonical_word(word)
    if canon in PRONOUNS:
        return "pron"
    if canon in DETERMINERS:
        return "det"
    if canon in PREPOSITIONS:
        return "prep"
    if canon in CONJUNCTIONS:
        return "conj"
    if canon in VERBS:
        return "verb"
    if canon in NOUNS:
        return "noun"
    if canon in ADJECTIVES:
        return "adj"
    if canon in ADVERBS:
        return "adv"
    if canon in QUESTION_WORDS:
        return "question"
    if canon.endswith("ED") or canon.endswith("ING"):
        return "verb"
    if canon.endswith("S") and len(canon) > 3:
        return "noun"
    return "word"


def build_lexicon() -> dict[str, Token]:
    words = set()
    words |= {canonical_word(w) for w in LETTER_WORDS}
    words |= {canonical_word(w) for w in ELGAR_CONTEXT_WORDS}
    words |= {canonical_word(w) for w in MUSIC_WORDS}
    words |= {canonical_word(w) for w in VICTORIAN_SOCIAL_WORDS}
    words |= (
        PRONOUNS
        | DETERMINERS
        | PREPOSITIONS
        | CONJUNCTIONS
        | VERBS
        | NOUNS
        | ADJECTIVES
        | ADVERBS
        | QUESTION_WORDS
    )
    lexicon: dict[str, Token] = {}
    for word in words:
        lexicon[word] = Token(word, word, "word", pos_for_word(word), 1.0)
    for entry in load_dictionary_entries():
        if not (2 <= len(entry.internal) <= 10):
            continue
        if entry.pos == "noun_or_verb":
            pos = "noun"
            kind = f"dictionary_{entry.source}_noun_or_verb"
        else:
            pos = entry.pos
            kind = f"dictionary_{entry.source}_{entry.pos}"
        token = Token(entry.internal, entry.display, kind, pos, entry.weight)
        existing = lexicon.get(entry.internal)
        if existing is None or token.weight > existing.weight:
            lexicon[entry.internal] = token
    for raw, (display, pos) in CONTRACTIONS.items():
        lexicon[canonical_word(raw)] = Token(canonical_word(raw), display, "contraction", pos, 1.12)
    if ALLOW_ABBREVIATIONS:
        for raw, (display, pos) in ABBREVIATIONS.items():
            lexicon[canonical_word(raw)] = Token(canonical_word(raw), display, "abbrev", pos, 0.55)
    return lexicon


LEXICON = build_lexicon()


def transition_bonus(prev: Token | None, token: Token) -> float:
    if prev is None:
        return 0.12 if token.pos in {"pron", "noun", "det", "adj", "question", "adv"} else -0.05
    pair = (prev.pos, token.pos)
    good = {
        ("pron", "verb"),
        ("pron_verb", "noun"),
        ("pron_verb", "verb"),
        ("pron_verb", "prep"),
        ("noun", "verb"),
        ("det", "noun"),
        ("det", "adj"),
        ("adj", "noun"),
        ("verb", "det"),
        ("verb", "noun"),
        ("verb", "prep"),
        ("verb", "adv"),
        ("prep", "det"),
        ("prep", "noun"),
        ("conj", "pron"),
        ("conj", "det"),
        ("conj", "noun"),
        ("question", "verb"),
        ("question", "pron"),
        ("adv", "verb"),
        ("adv", "prep"),
    }
    bad = {
        ("det", "verb"),
        ("prep", "verb"),
        ("prep", "conj"),
        ("conj", "prep"),
        ("verb", "verb"),
        ("question", "prep"),
    }
    if pair in good:
        return 0.22
    if pair in bad:
        return -0.22
    return 0.0


def token_weight(token: Token) -> float:
    length_bonus = min(len(token.text), 8) * 0.035
    context_bonus = 0.0
    if token.text in {canonical_word(w) for w in ELGAR_CONTEXT_WORDS}:
        context_bonus = 1.08
    elif token.text in {canonical_word(w) for w in MUSIC_WORDS | VICTORIAN_SOCIAL_WORDS}:
        context_bonus = 0.25
    if token.kind.startswith("dictionary_web2"):
        length_bonus *= 0.12
        context_bonus -= 0.80
        if len(token.text) <= 4 and token.text not in LETTER_WORDS:
            context_bonus -= 0.35
    elif token.kind.startswith("dictionary_") and "curated" not in token.kind:
        length_bonus *= 0.45
        context_bonus -= 0.28
    return token.weight + length_bonus + context_bonus


def best_tokenizations(text: str, limit: int = 5) -> list[Tokenization]:
    clean = canonical_word(re.sub(r"[^A-Z]", "", text.upper()))
    if not clean:
        return [Tokenization([], 0.0, "empty")]
    n = len(clean)
    paths: list[list[tuple[float, list[Token]]]] = [[] for _ in range(n + 1)]
    paths[0] = [(0.0, [])]
    for i in range(n):
        if not paths[i]:
            continue
        for j in range(i + 1, min(n, i + 10) + 1):
            part = clean[i:j]
            token = LEXICON.get(part)
            if token is None:
                continue
            for score, toks in paths[i]:
                prev = toks[-1] if toks else None
                next_score = score + token_weight(token) + transition_bonus(prev, token) - 0.26
                paths[j].append((next_score, toks + [token]))
        # Keep a costly single-letter fallback so partial noisy candidates can
        # still be represented without being treated as good English.
        fallback = Token(clean[i : i + 1], clean[i : i + 1], "unknown", "unknown", -2.35)
        for score, toks in paths[i][:limit]:
            prev = toks[-1] if toks else None
            paths[i + 1].append((score + fallback.weight + transition_bonus(prev, fallback) - 0.26, toks + [fallback]))
        for k in range(i + 1, min(n, i + 10) + 1):
            if paths[k]:
                paths[k] = sorted(paths[k], key=lambda item: item[0], reverse=True)[:limit]
    final = sorted(paths[n], key=lambda item: item[0], reverse=True)[:limit]
    out = []
    for raw_score, toks in final:
        unknowns = sum(1 for tok in toks if tok.kind == "unknown")
        contractions = sum(1 for tok in toks if tok.kind == "contraction")
        abbrevs = sum(1 for tok in toks if tok.kind == "abbrev")
        single_noise = sum(1 for tok in toks if len(tok.text) == 1 and tok.text not in {"A", "I"})
        rare_dictionary = sum(
            1 for tok in toks
            if tok.kind.startswith("dictionary_web2") and tok.text not in LETTER_WORDS
        )
        normalized = raw_score / max(len(toks), 1) - unknowns * 0.78 - single_noise * 0.22 - rare_dictionary * 0.55
        notes = (
            f"{' '.join(tok.display for tok in toks)}; "
            f"unknown={unknowns}, contractions={contractions}, abbrev={abbrevs}"
        )
        out.append(
            Tokenization(
                toks,
                normalized,
                notes,
                {
                    "unknown_tokens": float(unknowns),
                    "single_noise": float(single_noise),
                    "rare_dictionary": float(rare_dictionary),
                },
            )
        )
    return sorted(out, key=lambda item: item.score, reverse=True)


def split_partial_row(row: str) -> list[str]:
    return [part for part in re.split(r"[_?]+", row.upper()) if part]


def score_partial_or_full_row(row: str) -> Tokenization:
    parts = split_partial_row(row)
    if not parts:
        return Tokenization([], -1.0, "unresolved")
    all_tokens: list[Token] = []
    scores = []
    notes = []
    unknown_tokens = 0.0
    single_noise = 0.0
    rare_dictionary = 0.0
    for part in parts:
        best = best_tokenizations(part, limit=1)[0]
        all_tokens.extend(best.tokens)
        scores.append(best.score)
        notes.append(best.notes)
        diagnostics = best.diagnostics or {}
        unknown_tokens += diagnostics.get("unknown_tokens", 0.0)
        single_noise += diagnostics.get("single_noise", 0.0)
        rare_dictionary += diagnostics.get("rare_dictionary", 0.0)
    unresolved_penalty = (row.count("_") + row.count("?")) * 0.015
    resolved_chars = sum(ch.isalpha() for ch in row)
    unresolved_chars = row.count("_") + row.count("?")
    strictness = resolved_chars / max(resolved_chars + unresolved_chars, 1)
    noisy_token_penalty = strictness * (unknown_tokens * 0.12 + single_noise * 0.10 + rare_dictionary * 0.08)
    return Tokenization(
        all_tokens,
        sum(scores) / max(len(scores), 1) - unresolved_penalty - noisy_token_penalty,
        " / ".join(notes),
        {
            "unknown_tokens": unknown_tokens,
            "single_noise": single_noise,
            "rare_dictionary": rare_dictionary,
            "resolved_ratio": strictness,
        },
    )
