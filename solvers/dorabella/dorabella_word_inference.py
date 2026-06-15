"""Word-fragment inference for skeleton-first Dorabella search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .dorabella_constraints import parse_skeleton_line, normalize_letter
from .dorabella_data import CANONICAL_LETTERS, DorabellaPosition, ROW_LENGTHS, SKELETONS
from .dorabella_dictionary import entries_by_length
from .dorabella_elgar_context import (
    ELGAR_CONTEXT_WORDS,
    MUSIC_WORDS,
    VICTORIAN_SOCIAL_WORDS,
    canonical_word,
)
from .dorabella_language_model import ABBREVIATIONS, CONTRACTIONS, LEXICON, pos_for_word


@dataclass(frozen=True)
class WordHypothesis:
    row: int
    start_col: int
    end_col: int
    display: str
    internal: str
    score: float
    reason: str


@dataclass(frozen=True)
class WordInference:
    position_letter_priors: dict[int, dict[str, float]]
    hypotheses: list[WordHypothesis]


def row_offsets() -> list[int]:
    offsets = []
    current = 0
    for length in ROW_LENGTHS:
        offsets.append(current)
        current += length
    return offsets


def lexicon_candidates_by_length() -> dict[int, list[tuple[str, str, str, str, float, str]]]:
    """Return candidates grouped by length.

    Each item is (internal, display, kind, pos, dictionary_period_score, source).
    """
    items: dict[str, tuple[str, str, str, float, str]] = {}
    for internal, token in LEXICON.items():
        if 2 <= len(internal) <= 10:
            items[internal] = (token.display, token.kind, token.pos, 0.0, "lexicon")
    for length, entries in entries_by_length().items():
        if not (2 <= length <= 10):
            continue
        for entry in entries:
            kind = f"dictionary_{entry.source}_{entry.pos}"
            pos = "noun" if entry.pos == "noun_or_verb" else entry.pos
            existing = items.get(entry.internal)
            if existing is not None and existing[3] >= entry.period_score:
                continue
            items[entry.internal] = (entry.display, kind, pos, entry.period_score, entry.source)
    for raw, (display, _pos) in CONTRACTIONS.items():
        items[canonical_word(raw)] = (display, "contraction", "pron_verb", 0.0, "contraction")
    for raw, (display, _pos) in ABBREVIATIONS.items():
        items[canonical_word(raw)] = (display, "abbrev", "noun", 0.0, "abbrev")
    grouped: dict[int, list[tuple[str, str, str, str, float, str]]] = {}
    for internal, (display, kind, pos, period_score, source) in sorted(items.items()):
        grouped.setdefault(len(internal), []).append((internal, display, kind, pos, period_score, source))
    return grouped


def compatible_word(word: str, mask: Sequence[frozenset[str] | None]) -> bool:
    if len(word) != len(mask):
        return False
    for ch, allowed in zip(word, mask):
        letter = normalize_letter(ch)
        if letter not in CANONICAL_LETTERS:
            return False
        if allowed is not None and letter not in allowed:
            return False
    return True


def word_score(
    internal: str,
    display: str,
    kind: str,
    pos: str,
    known_slots: int,
    ambiguous_slots: int,
    period_score: float = 0.0,
) -> float:
    score = 0.18 + len(internal) * 0.028 + known_slots * 0.12 + ambiguous_slots * 0.035
    canon = canonical_word(internal)
    anchor_strength = known_slots + ambiguous_slots * 0.55
    if len(internal) >= 5 and anchor_strength < 2.0:
        score -= 0.46
    if len(internal) >= 7 and anchor_strength < 3.0:
        score -= 0.38
    if len(internal) >= 5 and known_slots == 0:
        score -= 0.28
    if canon in {canonical_word(w) for w in ELGAR_CONTEXT_WORDS}:
        score += 0.55
    if canon in {canonical_word(w) for w in MUSIC_WORDS}:
        score += 0.40
    if canon in {canonical_word(w) for w in VICTORIAN_SOCIAL_WORDS}:
        score += 0.32
    if canon in (
        {canonical_word(w) for w in ELGAR_CONTEXT_WORDS}
        | {canonical_word(w) for w in MUSIC_WORDS}
        | {canonical_word(w) for w in VICTORIAN_SOCIAL_WORDS}
    ) and len(internal) >= 5 and anchor_strength < 2.0:
        # Context words are useful only when the cipher has actually earned
        # them. Otherwise they crowd out shorter grammatical glue.
        score -= 0.42
    if kind == "contraction":
        score += 0.18
    if kind == "abbrev":
        score -= 0.05
    if pos in {"verb", "noun", "pron_verb"}:
        score += 0.08
    if kind.startswith("dictionary_"):
        score += period_score * 0.55
        if "noun_or_verb" in kind:
            score -= 0.12
        if "web2" in kind and period_score < 0.20:
            score -= 0.34
            if len(internal) >= 7:
                score -= 0.34
        if period_score < 0.18:
            score -= 0.24
            if len(internal) >= 7:
                score -= 0.18
            if known_slots + ambiguous_slots < 2:
                score -= 0.16
    return score


def build_word_inference(
    positions: Sequence[DorabellaPosition],
    skeletons: Sequence[str] = SKELETONS,
    max_hypotheses_per_window: int = 8,
    hypothesis_limit: int = 500,
) -> WordInference:
    priors: dict[int, dict[str, float]] = {}
    hypotheses: list[WordHypothesis] = []
    offsets = row_offsets()
    candidates_by_length = lexicon_candidates_by_length()
    parsed = [parse_skeleton_line(line) for line in skeletons]

    for row_idx, mask in enumerate(parsed, start=1):
        offset = offsets[row_idx - 1]
        for start in range(len(mask)):
            for end in range(start + 2, min(len(mask), start + 10) + 1):
                window = mask[start:end]
                # Favor windows anchored by at least one known/ambiguous slot;
                # fully blank windows are too unconstrained at this stage.
                known_slots = sum(1 for slot in window if slot is not None and len(slot) == 1)
                ambiguous_slots = sum(1 for slot in window if slot is not None and len(slot) > 1)
                if known_slots + ambiguous_slots == 0:
                    continue
                matches = []
                for internal, display, kind, pos, period_score, source in candidates_by_length.get(len(window), []):
                    if compatible_word(internal, window):
                        score = word_score(
                            internal,
                            display,
                            kind,
                            pos,
                            known_slots,
                            ambiguous_slots,
                            period_score,
                        )
                        matches.append((score, internal, display, kind, source))
                matches.sort(reverse=True)
                for score, internal, display, kind, source in matches[:max_hypotheses_per_window]:
                    hypotheses.append(
                        WordHypothesis(
                            row=row_idx,
                            start_col=start + 1,
                            end_col=end,
                            display=display,
                            internal=internal,
                            score=score,
                            reason=f"{kind}; source={source}; anchored={known_slots}; ambiguous={ambiguous_slots}",
                        )
                    )
                    for rel, ch in enumerate(internal):
                        global_index = offset + start + rel
                        priors.setdefault(global_index, {})
                        letter = normalize_letter(ch)
                        priors[global_index][letter] = priors[global_index].get(letter, 0.0) + score

    # Normalize per position so very dense windows do not dominate everything.
    for letter_scores in priors.values():
        total = sum(abs(v) for v in letter_scores.values()) or 1.0
        for letter in list(letter_scores):
            letter_scores[letter] = letter_scores[letter] / total
    hypotheses.sort(key=lambda item: item.score, reverse=True)
    return WordInference(priors, hypotheses[:hypothesis_limit])


def word_assignment_bonus(inference: WordInference, assignment: dict[int, str]) -> float:
    if not assignment:
        return 0.0
    bonus = 0.0
    for global_index, letter in assignment.items():
        bonus += inference.position_letter_priors.get(global_index, {}).get(normalize_letter(letter), 0.0)
    return bonus / max(len(assignment), 1)


def top_word_hypotheses_for_report(inference: WordInference, limit: int = 40) -> list[dict[str, object]]:
    return [
        {
            "row": h.row,
            "start_col": h.start_col,
            "end_col": h.end_col,
            "display": h.display,
            "internal": h.internal,
            "score": h.score,
            "reason": h.reason,
        }
        for h in inference.hypotheses[:limit]
    ]
