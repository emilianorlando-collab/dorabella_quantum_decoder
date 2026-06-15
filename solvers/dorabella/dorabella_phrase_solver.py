"""Phrase-first solver for the Dorabella skeleton.

The earlier quantum sampler filled letters and then tried to tokenize them.
That naturally collapses into local minima made of one-letter tokens. This
module inverts the search: propose word/phrase segments first, then let the
hard cipher validator decide whether the segment can exist under the 4-cycle
bijection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .dorabella_constraints import (
    CipherState,
    build_positions,
    normalize_letter,
    seed_alphabet1_constraints,
    seed_fixed_skeleton_constraints,
)
from .dorabella_data import ROW_LENGTHS, ROW_SYMBOLS, SKELETONS, DorabellaPosition
from .dorabella_language_model import LEXICON, Token, score_partial_or_full_row
from .dorabella_memory_ai import assignment_memory_bonus, candidate_neural_bonus
from .dorabella_qnn import qnn_expectation
from .dorabella_word_inference import build_word_inference


@dataclass(frozen=True)
class SegmentOption:
    row: int
    start_col: int
    end_col: int
    text: str
    display: str
    score: float
    source: str


@dataclass
class PhraseState:
    state: CipherState
    assignment: dict[int, str] = field(default_factory=dict)
    rows_chars: list[list[str]] = field(default_factory=list)
    covered: list[list[bool]] = field(default_factory=list)
    segments: list[SegmentOption] = field(default_factory=list)
    score: float = 0.0

    def clone(self) -> "PhraseState":
        return PhraseState(
            state=self.state.clone(),
            assignment=self.assignment.copy(),
            rows_chars=[row[:] for row in self.rows_chars],
            covered=[row[:] for row in self.covered],
            segments=self.segments[:],
            score=self.score,
        )


def initial_rows() -> list[list[str]]:
    rows = []
    for skeleton in SKELETONS:
        parsed = []
        i = 0
        while i < len(skeleton):
            ch = skeleton[i]
            if ch == "_":
                parsed.append("_")
                i += 1
            elif ch.isalpha():
                parsed.append(normalize_letter(ch))
                i += 1
                while i + 1 < len(skeleton) and skeleton[i] == "/" and skeleton[i + 1].isalpha():
                    i += 2
            else:
                i += 1
        rows.append(parsed)
    return rows


def initial_covered() -> list[list[bool]]:
    return [[False for _ in range(length)] for length in ROW_LENGTHS]


def row_offsets() -> list[int]:
    out = []
    cur = 0
    for n in ROW_LENGTHS:
        out.append(cur)
        cur += n
    return out


def token_base_score(token: Token) -> float:
    score = 0.35 + min(len(token.text), 9) * 0.06
    if token.kind == "contraction":
        score += 0.18
    if token.kind == "abbrev":
        score -= 0.12
    if token.pos in {"noun", "verb", "pron_verb"}:
        score += 0.12
    if token.text in {"BELLA", "ELGAR", "DORA", "EDWARD", "ALICE", "MARCO"}:
        score += 0.35
    return score


def compatible_with_slots(text: str, slots: Sequence[frozenset[str] | None]) -> bool:
    if len(text) != len(slots):
        return False
    for ch, allowed in zip(text, slots):
        letter = normalize_letter(ch)
        if allowed is not None and letter not in allowed:
            return False
    return True


def build_segment_options(
    positions: Sequence[DorabellaPosition],
    max_len: int = 9,
    max_options_per_start: int = 40,
) -> dict[int, list[SegmentOption]]:
    by_start: dict[int, list[SegmentOption]] = {}
    offsets = row_offsets()
    lex = list(LEXICON.values())
    for row_idx, offset in enumerate(offsets, start=1):
        row_positions = positions[offset : offset + ROW_LENGTHS[row_idx - 1]]
        for start in range(ROW_LENGTHS[row_idx - 1]):
            options: list[SegmentOption] = []
            for token in lex:
                text = token.text
                end = start + len(text)
                if len(text) < 2 or len(text) > max_len or end > len(row_positions):
                    continue
                slots = [pos.allowed_letters for pos in row_positions[start:end]]
                if not compatible_with_slots(text, slots):
                    continue
                anchored = sum(1 for slot in slots if slot is not None)
                if anchored == 0 and len(text) < 4:
                    continue
                score = token_base_score(token) + anchored * 0.10
                options.append(
                    SegmentOption(
                        row=row_idx,
                        start_col=start + 1,
                        end_col=end,
                        text=text,
                        display=token.display,
                        score=score,
                        source="lexicon",
                    )
                )
            # Fallback single-letter options exist but are intentionally awful;
            # they prevent dead ends without letting letter soup dominate.
            pos = row_positions[start]
            allowed = sorted(pos.allowed_letters or set("ABCDEFGHIKLMNOPQRSTUWXYZ"))
            for letter in allowed[:24]:
                options.append(
                    SegmentOption(
                        row=row_idx,
                        start_col=start + 1,
                        end_col=start + 1,
                        text=letter,
                        display=letter,
                        score=-1.40,
                        source="fallback_letter",
                    )
                )
            options.sort(key=lambda item: item.score, reverse=True)
            by_start[offset + start] = options[:max_options_per_start]
    return by_start


def apply_segment(
    phrase_state: PhraseState,
    segment: SegmentOption,
    positions: Sequence[DorabellaPosition],
) -> PhraseState | None:
    offsets = row_offsets()
    global_start = offsets[segment.row - 1] + segment.start_col - 1
    trial = phrase_state.clone()
    row_chars = trial.rows_chars[segment.row - 1]
    row_covered = trial.covered[segment.row - 1]
    for idx, ch in enumerate(segment.text):
        global_index = global_start + idx
        pos = positions[global_index]
        letter = normalize_letter(ch)
        row_col = segment.start_col - 1 + idx
        if row_covered[row_col]:
            return None
        existing = row_chars[row_col]
        if existing not in {"_", letter}:
            return None
        if pos.allowed_letters is not None and letter not in pos.allowed_letters:
            return None
        placed = False
        for sym in pos.symbol_options:
            state2 = trial.state.clone()
            if state2.bind(pos.alphabet_index, sym, letter):
                trial.state = state2
                placed = True
                break
        if not placed:
            return None
        trial.assignment[global_index] = letter
        row_chars[row_col] = letter
        row_covered[row_col] = True
    trial.segments.append(segment)
    trial.score += segment.score
    return trial if propagate_forced_mappings(trial, positions) else None


def set_row_char_from_global(
    phrase_state: PhraseState,
    positions: Sequence[DorabellaPosition],
    global_index: int,
    letter: str,
) -> bool:
    pos = positions[global_index]
    row_col = pos.col - 1
    row_chars = phrase_state.rows_chars[pos.row - 1]
    current = row_chars[row_col]
    if current not in {"_", letter}:
        return False
    if pos.allowed_letters is not None and letter not in pos.allowed_letters:
        return False
    row_chars[row_col] = letter
    phrase_state.assignment[global_index] = letter
    return True


def forced_letter_for_position(state: CipherState, pos: DorabellaPosition) -> str | None:
    mapped_by_option = [
        state.maps[pos.alphabet_index].get(sym)
        for sym in pos.symbol_options
    ]
    if len(pos.symbol_options) > 1 and any(letter is None for letter in mapped_by_option):
        return None
    mapped = {letter for letter in mapped_by_option if letter is not None}
    if len(mapped) == 1:
        return next(iter(mapped))
    return None


def propagate_forced_mappings(
    phrase_state: PhraseState,
    positions: Sequence[DorabellaPosition],
) -> bool:
    """Fill every position forced by an already fixed symbol-letter mapping.

    This is the propagation the cryptanalytic report needs: if A2:2S -> W is
    known, later A2/2S positions become W immediately unless an orientation
    ambiguity prevents a unique conclusion.
    """
    changed = True
    while changed:
        changed = False
        for pos in positions:
            forced = forced_letter_for_position(phrase_state.state, pos)
            if forced is None:
                continue
            row_chars = phrase_state.rows_chars[pos.row - 1]
            current = row_chars[pos.col - 1]
            if current == forced:
                continue
            if current != "_":
                return False
            if not set_row_char_from_global(phrase_state, positions, pos.global_index, forced):
                return False
            changed = True
    return True


def next_uncovered_index(covered: list[list[bool]]) -> int | None:
    offset = 0
    for row, length in zip(covered, ROW_LENGTHS):
        for i, is_covered in enumerate(row):
            if not is_covered:
                return offset + i
        offset += length
    return None


def rows_from_chars(rows_chars: list[list[str]]) -> list[str]:
    return ["".join(row) for row in rows_chars]


def phrase_state_priority(
    phrase_state: PhraseState,
    positions_by_index: dict[int, DorabellaPosition],
    memory: dict[str, object] | None,
) -> float:
    rows = rows_from_chars(phrase_state.rows_chars)
    tokenized = score_partial_or_full_row("".join(rows))
    unresolved = sum(row.count("_") for row in rows)
    uncovered = sum(1 for row in phrase_state.covered for item in row if not item)
    fallback_count = sum(1 for segment in phrase_state.segments if segment.source == "fallback_letter")
    memory_bonus = assignment_memory_bonus(memory, positions_by_index, phrase_state.assignment) if memory else 0.0
    neural_bonus = 0.0
    if memory:
        total_unknowns = sum(1 for pos in positions_by_index.values() if pos.allowed_letters is None or len(pos.allowed_letters) != 1)
        candidate = {
            "rows": rows,
            "resolved_unknowns": len(phrase_state.assignment),
            "total_unknowns": total_unknowns,
        }
        neural_bonus = candidate_neural_bonus(memory, candidate)
        neural_bonus += qnn_expectation(memory, candidate) * 0.25
    return (
        phrase_state.score
        + tokenized.score * 0.45
        + memory_bonus * 0.35
        + neural_bonus * 0.45
        - unresolved * 0.025
        - uncovered * 0.18
        - fallback_count * 0.75
    )


def solve_phrase_first(
    memory: dict[str, object] | None = None,
    beam_width: int = 128,
    max_steps: int = 80,
) -> list[dict[str, object]]:
    positions = build_positions(ROW_SYMBOLS, SKELETONS)
    positions_by_index = {pos.global_index: pos for pos in positions}
    segment_options = build_segment_options(positions)
    _word_inference = build_word_inference(positions)

    base_state = CipherState()
    seed_alphabet1_constraints(positions, base_state)
    seed_fixed_skeleton_constraints(positions, base_state)
    rows = initial_rows()
    seed = PhraseState(base_state, {}, rows, initial_covered(), [], 0.0)
    if not propagate_forced_mappings(seed, positions):
        return []
    beam = [seed]

    for _step in range(max_steps):
        expanded: list[PhraseState] = []
        for item in beam:
            next_idx = next_uncovered_index(item.covered)
            if next_idx is None:
                expanded.append(item)
                continue
            for segment in segment_options.get(next_idx, []):
                candidate = apply_segment(item, segment, positions)
                if candidate is not None:
                    expanded.append(candidate)
        if not expanded:
            break
        expanded.sort(
            key=lambda item: phrase_state_priority(item, positions_by_index, memory),
            reverse=True,
        )
        beam = expanded[:beam_width]
        if all(next_uncovered_index(item.covered) is None for item in beam[: min(beam_width, 8)]):
            break

    results = []
    for item in beam[:beam_width]:
        rows = rows_from_chars(item.rows_chars)
        unresolved = sum(row.count("_") for row in rows)
        uncovered = sum(1 for row in item.covered for is_covered in row if not is_covered)
        fallback_count = sum(1 for segment in item.segments if segment.source == "fallback_letter")
        results.append(
            {
                "rows": rows,
                "segments": [
                    {
                        "row": seg.row,
                        "start_col": seg.start_col,
                        "end_col": seg.end_col,
                        "display": seg.display,
                        "text": seg.text,
                        "score": seg.score,
                        "source": seg.source,
                    }
                    for seg in item.segments
                ],
                "score": phrase_state_priority(item, positions_by_index, memory),
                "unresolved": unresolved,
                "uncovered": uncovered,
                "fallback_letters": fallback_count,
                "resolved_unknowns": len(item.assignment),
                "total_unknowns": sum(1 for pos in positions if pos.allowed_letters is None or len(pos.allowed_letters) != 1),
            }
        )
    results.sort(key=lambda item: (item["uncovered"], item["unresolved"], item["fallback_letters"], -item["score"]))
    return results
