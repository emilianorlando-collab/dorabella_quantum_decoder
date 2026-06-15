"""Semantic scoring hooks for grammar, fluency, and historical plausibility."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .dorabella_elgar_context import (
    ELGAR_CONTEXT_WORDS,
    LETTER_WORDS,
    MUSIC_WORDS,
    VICTORIAN_SOCIAL_WORDS,
    canonical_word,
    score_segmented_words,
)
from .dorabella_data import END_ANCHOR, START_ANCHOR
from .dorabella_language_model import score_partial_or_full_row

COMMON_WORDS = {
    "A", "AM", "AN", "AND", "ARE", "AS", "AT", "BE", "BELLA", "BY", "CAN",
    "COME", "DAY", "DEAR", "DO", "ELGAR", "FOR", "GO", "HAVE", "HE", "I",
    "IN", "IS", "IT", "ME", "MY", "NO", "NOT", "OF", "ON", "OR", "SEE",
    "SO", "THE", "TO", "WE", "WILL", "WITH", "YOU", "YOUR",
}
COMMON_WORDS = {
    canonical_word(w)
    for w in COMMON_WORDS
    | LETTER_WORDS
    | ELGAR_CONTEXT_WORDS
    | MUSIC_WORDS
    | VICTORIAN_SOCIAL_WORDS
}


@dataclass(frozen=True)
class SemanticScore:
    score: float
    notes: str
    details: dict[str, object] | None = None


GOOD_START_POS = {"pron", "pron_verb", "noun", "det", "adj", "question", "adv"}
BAD_END_POS = {"prep", "conj", "det", "question"}
VERB_POS = {"verb", "pron_verb"}

COMMON_BIGRAMS = {
    ("I", "AM"), ("I", "HAVE"), ("I", "WILL"), ("I", "WISH"),
    ("I", "WANT"), ("I", "THINK"), ("YOU", "ARE"), ("YOU", "WILL"),
    ("YOU", "HAVE"), ("MY", "DEAR"), ("DEAR", "BELLA"),
    ("TO", "YOU"), ("FOR", "YOU"), ("WITH", "YOU"),
    ("WISH", "YOU"), ("YOURS", "ELGAR"),
}


def simple_segment(text: str) -> list[str]:
    """Greedy segmentation used only as a cheap pre-filter."""
    clean = canonical_word(re.sub(r"[^A-Z]", "", text.upper()))
    out: list[str] = []
    i = 0
    while i < len(clean):
        best = ""
        for n in range(min(10, len(clean) - i), 0, -1):
            part = clean[i : i + n]
            if canonical_word(part) in COMMON_WORDS:
                best = part
                break
        if not best:
            best = clean[i]
        out.append(best)
        i += len(best)
    return out


def token_display_rows(tokens: list[object]) -> str:
    return " ".join(getattr(token, "display", getattr(token, "text", "")) for token in tokens)


def shape_penalty(row: str) -> float:
    penalty = 0.0
    for part in re.split(r"[_?]+", row.upper()):
        if not part:
            continue
        penalty += 0.45 * len(re.findall(r"([A-Z])\1{2,}", part))
        penalty += 0.35 * len(re.findall(r"[^AEIOUY]{5,}", part))
        penalty += 0.18 * len(re.findall(r"Q(?!U)", part))
    return penalty


def grammar_score(tokens: list[object], strictness: float) -> float:
    if not tokens:
        return -0.80
    score = 0.0
    first_pos = getattr(tokens[0], "pos", "unknown")
    last_pos = getattr(tokens[-1], "pos", "unknown")
    if first_pos in GOOD_START_POS:
        score += 0.18
    else:
        score -= 0.18 * strictness
    has_verb = any(getattr(token, "pos", "unknown") in VERB_POS for token in tokens)
    if has_verb:
        score += 0.24
    elif len(tokens) >= 3:
        score -= 0.48 * strictness
    if last_pos in BAD_END_POS:
        score -= 0.30 * strictness
    else:
        score += 0.08

    transitions = 0.0
    for prev, token in zip(tokens, tokens[1:]):
        prev_pos = getattr(prev, "pos", "unknown")
        token_pos = getattr(token, "pos", "unknown")
        if (prev_pos, token_pos) in {
            ("pron", "verb"),
            ("pron_verb", "noun"),
            ("pron_verb", "verb"),
            ("noun", "verb"),
            ("det", "noun"),
            ("det", "adj"),
            ("adj", "noun"),
            ("verb", "det"),
            ("verb", "noun"),
            ("verb", "prep"),
            ("prep", "det"),
            ("prep", "noun"),
            ("conj", "pron"),
            ("conj", "det"),
            ("question", "verb"),
            ("adv", "verb"),
        }:
            transitions += 0.16
        elif (prev_pos, token_pos) in {
            ("det", "verb"),
            ("prep", "verb"),
            ("prep", "conj"),
            ("conj", "prep"),
            ("verb", "verb"),
            ("question", "prep"),
            ("unknown", "unknown"),
        }:
            transitions -= 0.18 * strictness
    return score + transitions / max(len(tokens) - 1, 1)


def row_semantic(row: str, idx: int) -> tuple[float, str, dict[str, object]]:
    language = score_partial_or_full_row(row)
    tokens = language.tokens
    words = [token.text for token in tokens] or simple_segment(row)
    resolved_chars = sum(ch.isalpha() for ch in row)
    unresolved_chars = row.count("_") + row.count("?")
    strictness = resolved_chars / max(resolved_chars + unresolved_chars, 1)
    unknown_tokens = sum(1 for token in tokens if token.kind == "unknown")
    single_noise = sum(1 for token in tokens if len(token.text) == 1 and token.text not in {"A", "I"})
    rare_dictionary = sum(
        1 for token in tokens
        if token.kind.startswith("dictionary_web2") and canonical_word(token.text) not in COMMON_WORDS
    )
    valid_chars = sum(len(token.text) for token in tokens if token.kind != "unknown")
    lexical_coverage = valid_chars / max(resolved_chars, 1)
    known = sum(1 for w in words if canonical_word(w) in COMMON_WORDS)
    known_ratio = known / max(len(words), 1)
    context = score_segmented_words(words)
    grammar = grammar_score(tokens, strictness)
    shape = shape_penalty(row)
    unknown_ratio = unknown_tokens / max(len(tokens), 1)
    long_known = sum(1 for token in tokens if token.kind != "unknown" and len(token.text) >= 5)
    long_word_bonus = min(long_known, 3) * 0.07
    noise_penalty = strictness * (
        unknown_ratio * 1.25
        + single_noise * 0.20
        + rare_dictionary * 0.18
        + shape
    )
    score = (
        language.score * 0.82
        + lexical_coverage * 0.62
        + known_ratio * 0.38
        + context.score * 0.70
        + grammar * 0.85
        + long_word_bonus
        - noise_penalty
    )
    details = {
        "row": idx,
        "raw": row,
        "tokens": [getattr(token, "display", token.text) for token in tokens],
        "token_kinds": [token.kind for token in tokens],
        "pos": [token.pos for token in tokens],
        "language_score": language.score,
        "lexical_coverage": lexical_coverage,
        "known_ratio": known_ratio,
        "grammar_score": grammar,
        "context_hits": context.hits,
        "unknown_tokens": unknown_tokens,
        "single_noise": single_noise,
        "rare_dictionary": rare_dictionary,
        "shape_penalty": shape,
        "strictness": strictness,
        "row_score": score,
    }
    note = (
        f"R{idx}: {token_display_rows(tokens)}; "
        f"row_score={score:.3f}, grammar={grammar:.3f}, "
        f"lexical={lexical_coverage:.2f}, unknown={unknown_tokens}, "
        f"single_noise={single_noise}, rare={rare_dictionary}; {context.notes}"
    )
    return score, note, details


def candidate_bigram_bonus(row_details: list[dict[str, object]]) -> float:
    tokens: list[str] = []
    for detail in row_details:
        tokens.extend(canonical_word(str(token)) for token in detail.get("tokens", []))
    hits = sum(1 for pair in zip(tokens, tokens[1:]) if pair in COMMON_BIGRAMS)
    return min(0.35, hits * 0.09)


def score_rows(rows: list[str]) -> SemanticScore:
    """Strict semantic score before an optional external LLM review."""
    row_scores = []
    notes = []
    details = []
    for idx, row in enumerate(rows, start=1):
        score, note, detail = row_semantic(row, idx)
        row_scores.append(score)
        notes.append(note)
        details.append(detail)
    joined = "".join(rows).upper()
    anchor_bonus = 0.0
    if START_ANCHOR and joined.startswith(START_ANCHOR):
        anchor_bonus += 0.10
    if END_ANCHOR and joined.endswith(END_ANCHOR):
        anchor_bonus += 0.10
    bigram_bonus = candidate_bigram_bonus(details)
    unresolved_chars = joined.count("_") + joined.count("?")
    resolved_chars = sum(ch.isalpha() for ch in joined)
    coverage = resolved_chars / max(resolved_chars + unresolved_chars, 1)
    total_unknown_tokens = sum(int(detail["unknown_tokens"]) for detail in details)
    total_single_noise = sum(float(detail.get("single_noise", 0.0)) for detail in details)
    total_rare_dictionary = sum(float(detail.get("rare_dictionary", 0.0)) for detail in details)
    total_tokens = sum(len(detail["tokens"]) for detail in details) or 1
    unresolved_penalty = coverage * (total_unknown_tokens / total_tokens) * 0.45
    row_balance_penalty = max(0.0, (max(row_scores) - min(row_scores)) - 3.2) * 0.22 if row_scores else 0.0
    full_noise_penalty = coverage * (
        total_single_noise * 0.46
        + total_rare_dictionary * 0.72
        + total_unknown_tokens * 0.22
    ) / max(total_tokens, 1)
    unresolved_rows = sum(1 for detail in details if int(detail.get("unknown_tokens", 0)) > 0)
    fragment_penalty = coverage * max(0.0, (total_single_noise + total_unknown_tokens) - 8.0) * 0.045
    if coverage >= 0.55 and unresolved_rows >= 2:
        fragment_penalty += 0.16
    total = (
        sum(row_scores) / max(len(row_scores), 1)
        + anchor_bonus
        + bigram_bonus
        - unresolved_penalty
        - row_balance_penalty
        - full_noise_penalty
        - fragment_penalty
    )
    return SemanticScore(
        score=total,
        notes=" | ".join(notes),
        details={
            "coverage": coverage,
            "anchor_bonus": anchor_bonus,
            "bigram_bonus": bigram_bonus,
            "unresolved_penalty": unresolved_penalty,
            "row_balance_penalty": row_balance_penalty,
            "full_noise_penalty": full_noise_penalty,
            "fragment_penalty": fragment_penalty,
            "rows": details,
        },
    )


def llm_review_prompt(rows: list[str]) -> str:
    return f"""
Evaluate this Dorabella plaintext candidate under strict rules.

R1: {rows[0]}
R2: {rows[1]}
R3: {rows[2]}

Rules:
- English only, no invented words, no missing letters.
- Consider word breaks, contraction forms without apostrophes, and conservative
  abbreviations if they fit Elgar's period and style.
- Prefer grammatical row-level phrases, including noun/pronoun followed by a
  plausible verb.
- Each row starts and ends a complete phrase.
- Victorian/Elgar/Dora Penny context is preferred.
- Optional private anchors may be deliberate, but do not force approval if
  grammar is poor.

Return JSON only:
{{"score": 0.0, "verdict": "reject|maybe|strong", "reason": "..."}}
"""


def perplexity_like_penalty(rows: list[str]) -> float:
    """Placeholder for GPT/local model scoring without importing heavy packages."""
    joined = " ".join(rows)
    vowel_ratio = sum(ch in "AEIOU" for ch in joined.upper()) / max(sum(ch.isalpha() for ch in joined), 1)
    return abs(vowel_ratio - 0.38) + math.log1p(len(re.findall(r"[^AEIOU ]{5,}", joined.upper())))
