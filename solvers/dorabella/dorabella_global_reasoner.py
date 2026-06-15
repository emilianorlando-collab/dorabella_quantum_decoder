"""Global Dorabella reasoning utilities.

This module keeps the solvers honest: every local letter choice is interpreted
as a global cipher hypothesis under the four rotating alphabets. If a symbol is
bound in one position, all positions where that same symbol/alphabet is forced
inherit the same letter before the branch is scored.
"""

from __future__ import annotations

import itertools
import math
from typing import Sequence

from .dorabella_constraints import CipherState, normalize_letter
from .dorabella_data import CANONICAL_LETTERS, ROW_LENGTHS, DorabellaPosition
from .dorabella_word_inference import WordHypothesis, WordInference


ENGLISH_LETTER_FREQUENCY = {
    "E": 12.70,
    "T": 9.06,
    "A": 8.17,
    "O": 7.51,
    "I": 6.97,
    "N": 6.75,
    "S": 6.33,
    "H": 6.09,
    "R": 5.99,
    "D": 4.25,
    "L": 4.03,
    "C": 2.78,
    "U": 2.76,
    "M": 2.41,
    "W": 2.36,
    "F": 2.23,
    "G": 2.02,
    "Y": 1.97,
    "P": 1.93,
    "B": 1.49,
    "U/V": 1.20,
    "K": 0.77,
    "X": 0.15,
    "Q": 0.10,
    "Z": 0.07,
}


def letter_frequency(letter: str) -> float:
    canonical = normalize_letter(letter)
    if canonical == "U":
        return ENGLISH_LETTER_FREQUENCY["U/V"]
    return ENGLISH_LETTER_FREQUENCY.get(canonical, 0.05)


def letter_frequency_score(letter: str) -> float:
    """Small centered prior; common letters help, rare letters do not dominate."""
    avg = 100.0 / len(CANONICAL_LETTERS)
    return math.log(max(letter_frequency(letter), 0.01) / avg)


def memory_letter_score(
    pos: DorabellaPosition,
    letter: str,
    memory: dict[str, object] | None,
) -> float:
    """Learned score for proposing a letter at a global cipher position."""
    if memory is None:
        return 0.0
    active = memory.get("active_memory", {})
    if not isinstance(active, dict):
        return 0.0
    position_scores = active.get("position_letter_scores", {})
    symbol_scores = active.get("symbol_letter_scores", {})
    if not isinstance(position_scores, dict) or not isinstance(symbol_scores, dict):
        return 0.0
    canonical = normalize_letter(letter)
    score = float(position_scores.get(f"{pos.global_index + 1}:{canonical}", 0.0))
    for sym in pos.symbol_options:
        score += float(symbol_scores.get(f"A{pos.alphabet_index + 1}:{sym}:{canonical}", 0.0)) / max(len(pos.symbol_options), 1)
    return score


def memory_prior_strength(
    pos: DorabellaPosition,
    memory: dict[str, object] | None,
) -> float:
    if memory is None:
        return 0.0
    return sum(
        abs(memory_letter_score(pos, letter, memory))
        for letter in CANONICAL_LETTERS
    )


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


def propagate_forced_assignments(
    state: CipherState,
    positions: Sequence[DorabellaPosition],
    assignment: dict[int, str],
) -> dict[int, str] | None:
    """Return assignment plus all forced consequences, or None on collision."""
    out = {idx: normalize_letter(letter) for idx, letter in assignment.items()}
    changed = True
    while changed:
        changed = False
        for pos in positions:
            forced = forced_letter_for_position(state, pos)
            if forced is None:
                continue
            if pos.allowed_letters is not None and forced not in pos.allowed_letters:
                return None
            current = out.get(pos.global_index)
            if current is not None and current != forced:
                return None
            if current is None:
                out[pos.global_index] = forced
                changed = True
    return out


def letter_for_position(
    state: CipherState,
    pos: DorabellaPosition,
    assignment: dict[int, str],
) -> str | None:
    assigned = assignment.get(pos.global_index)
    if assigned is not None:
        return assigned
    forced = forced_letter_for_position(state, pos)
    if forced is not None:
        return forced
    if pos.allowed_letters is not None and len(pos.allowed_letters) == 1:
        return next(iter(pos.allowed_letters))
    return None


def rows_from_state(
    positions: Sequence[DorabellaPosition],
    state: CipherState,
    assignment: dict[int, str],
) -> list[str]:
    chars = []
    for pos in positions:
        letter = letter_for_position(state, pos, assignment)
        chars.append(letter if letter is not None else "_")
    rows = []
    offset = 0
    for length in ROW_LENGTHS:
        rows.append("".join(chars[offset : offset + length]))
        offset += length
    return rows


def assignment_signature(assignment: dict[int, str]) -> tuple[tuple[int, str], ...]:
    return tuple(sorted((idx, normalize_letter(letter)) for idx, letter in assignment.items()))


def state_signature(state: CipherState) -> tuple[tuple[tuple[str, str], ...], ...]:
    return tuple(tuple(sorted(alpha.items())) for alpha in state.maps)


def can_bind_position(state: CipherState, pos: DorabellaPosition, letter: str) -> bool:
    letter = normalize_letter(letter)
    if pos.allowed_letters is not None and letter not in pos.allowed_letters:
        return False
    forced = forced_letter_for_position(state, pos)
    if forced is not None:
        return forced == letter
    for sym in pos.symbol_options:
        trial = state.clone()
        if trial.bind(pos.alphabet_index, sym, letter):
            return True
    return False


def top_letters_for_position(
    pos: DorabellaPosition,
    state: CipherState,
    inference: WordInference | None,
    memory: dict[str, object] | None = None,
    limit: int = 6,
) -> list[tuple[str, float]]:
    forced = forced_letter_for_position(state, pos)
    if forced is not None:
        return [(forced, 10.0)] if can_bind_position(state, pos, forced) else []

    domain = sorted(pos.allowed_letters or set(CANONICAL_LETTERS))
    word_priors = inference.position_letter_priors.get(pos.global_index, {}) if inference else {}
    scored: list[tuple[str, float]] = []
    for letter in domain:
        if not can_bind_position(state, pos, letter):
            continue
        score = word_priors.get(normalize_letter(letter), 0.0) * 3.5
        score += memory_letter_score(pos, letter, memory) * 1.15
        score += letter_frequency_score(letter) * 0.22
        scored.append((normalize_letter(letter), score))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:limit]


def prior_assignments_for_block(
    block: Sequence[DorabellaPosition],
    state: CipherState,
    inference: WordInference | None,
    memory: dict[str, object] | None = None,
    per_position: int = 5,
    limit: int = 64,
) -> list[dict[int, str]]:
    """Generate deterministic global-language assignments for a quantum block."""
    options = [
        top_letters_for_position(pos, state, inference, limit=per_position)
        if memory is None
        else top_letters_for_position(pos, state, inference, memory=memory, limit=per_position)
        for pos in block
    ]
    if not options or any(not choices for choices in options):
        return []

    ranked: list[tuple[float, dict[int, str]]] = []
    for combo in itertools.product(*options):
        score = sum(item[1] for item in combo)
        assignment = {
            pos.global_index: letter
            for pos, (letter, _letter_score) in zip(block, combo)
        }
        ranked.append((score, assignment))
    ranked.sort(key=lambda item: item[0], reverse=True)

    seen: set[tuple[tuple[int, str], ...]] = set()
    out: list[dict[int, str]] = []
    for _score, assignment in ranked:
        key = assignment_signature(assignment)
        if key in seen:
            continue
        seen.add(key)
        out.append(assignment)
        if len(out) >= limit:
            break
    return out


def merge_assignment_lists(*groups: Sequence[dict[int, str]]) -> list[dict[int, str]]:
    seen: set[tuple[tuple[int, str], ...]] = set()
    out: list[dict[int, str]] = []
    for group in groups:
        for assignment in group:
            key = assignment_signature(assignment)
            if key in seen:
                continue
            seen.add(key)
            out.append(dict(assignment))
    return out


def assignment_for_word_hypothesis(
    hypothesis: WordHypothesis,
    positions: Sequence[DorabellaPosition],
    state: CipherState,
    current_assignment: dict[int, str],
) -> dict[int, str] | None:
    offset = sum(ROW_LENGTHS[: hypothesis.row - 1])
    start = offset + hypothesis.start_col - 1
    assignment: dict[int, str] = {}
    added_unknown = False
    for rel, raw_letter in enumerate(hypothesis.internal):
        global_index = start + rel
        pos = positions[global_index]
        letter = normalize_letter(raw_letter)
        current = current_assignment.get(global_index)
        if current is not None and current != letter:
            return None
        if pos.allowed_letters is not None and letter not in pos.allowed_letters:
            return None
        if not can_bind_position(state, pos, letter):
            return None
        assignment[global_index] = letter
        if current is None and not (pos.allowed_letters is not None and len(pos.allowed_letters) == 1):
            added_unknown = True
    if not added_unknown:
        return None
    return assignment


def prior_segment_assignments(
    positions: Sequence[DorabellaPosition],
    state: CipherState,
    current_assignment: dict[int, str],
    inference: WordInference | None,
    limit: int = 24,
) -> list[dict[int, str]]:
    if inference is None:
        return []
    scored: list[tuple[float, dict[int, str]]] = []
    for hypothesis in inference.hypotheses:
        assignment = assignment_for_word_hypothesis(hypothesis, positions, state, current_assignment)
        if assignment is None:
            continue
        word_len_bonus = min(len(hypothesis.internal), 8) * 0.07
        score = hypothesis.score + word_len_bonus
        scored.append((score, assignment))
    scored.sort(key=lambda item: item[0], reverse=True)
    seen: set[tuple[tuple[int, str], ...]] = set()
    out: list[dict[int, str]] = []
    for _score, assignment in scored:
        key = assignment_signature(assignment)
        if key in seen:
            continue
        seen.add(key)
        out.append(assignment)
        if len(out) >= limit:
            break
    return out


def letter_frequency_assignment_bonus(assignment: dict[int, str]) -> float:
    if not assignment:
        return 0.0
    values = [letter_frequency_score(letter) for letter in assignment.values()]
    return sum(values) / len(values)


def orientation_trace(
    positions: Sequence[DorabellaPosition],
    state: CipherState,
    assignment: dict[int, str],
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for pos in positions:
        letter = assignment.get(pos.global_index)
        if letter is None:
            forced = forced_letter_for_position(state, pos)
            if forced is None:
                continue
            letter = forced
        matching_symbols = [
            sym for sym in pos.symbol_options
            if state.maps[pos.alphabet_index].get(sym) == letter
        ]
        if len(pos.symbol_options) > 1 or matching_symbols:
            out.append(
                {
                    "position": pos.global_index + 1,
                    "row": pos.row,
                    "col": pos.col,
                    "alphabet": pos.alphabet_index + 1,
                    "symbol_options": list(pos.symbol_options),
                    "chosen_symbols": matching_symbols,
                    "letter": letter,
                    "ambiguous": len(pos.symbol_options) > 1,
                }
            )
    return out


def bijection_closure_trace(
    positions: Sequence[DorabellaPosition],
    state: CipherState,
    assignment: dict[int, str],
) -> list[dict[str, object]]:
    """Explain every active symbol-letter binding and its global consequences."""
    out: list[dict[str, object]] = []
    for alphabet_index, mapping in enumerate(state.maps):
        for symbol, letter in sorted(mapping.items()):
            exact_occurrences = []
            possible_occurrences = []
            for pos in positions:
                if pos.alphabet_index != alphabet_index or symbol not in pos.symbol_options:
                    continue
                current = letter_for_position(state, pos, assignment)
                item = {
                    "position": pos.global_index + 1,
                    "row": pos.row,
                    "col": pos.col,
                    "symbol_options": list(pos.symbol_options),
                    "rendered_letter": current,
                    "forced": len(pos.symbol_options) == 1,
                    "consistent": current in {None, letter} if len(pos.symbol_options) == 1 else True,
                }
                if len(pos.symbol_options) == 1:
                    exact_occurrences.append(item)
                else:
                    possible_occurrences.append(item)
            if not exact_occurrences and not possible_occurrences:
                continue
            out.append(
                {
                    "alphabet": alphabet_index + 1,
                    "symbol": symbol,
                    "letter": letter,
                    "exact_occurrences": exact_occurrences,
                    "possible_ambiguous_occurrences": possible_occurrences,
                    "exact_positions": [item["position"] for item in exact_occurrences],
                    "possible_positions": [item["position"] for item in possible_occurrences],
                    "exact_all_consistent": all(item["consistent"] for item in exact_occurrences),
                }
            )
    return out


def repeated_symbol_pressure(
    pos: DorabellaPosition,
    positions: Sequence[DorabellaPosition],
) -> float:
    count = 0
    symbols = set(pos.symbol_options)
    for other in positions:
        if other.global_index == pos.global_index:
            continue
        if other.alphabet_index == pos.alphabet_index and symbols.intersection(other.symbol_options):
            count += 1
    return float(count)


def globally_ordered_unknowns(
    positions: Sequence[DorabellaPosition],
    seed_assignment: dict[int, str],
    inference: WordInference | None,
    memory: dict[str, object] | None = None,
) -> list[DorabellaPosition]:
    unknowns = [
        pos
        for pos in positions
        if not (pos.allowed_letters is not None and len(pos.allowed_letters) == 1)
        and pos.global_index not in seed_assignment
    ]

    def priority(pos: DorabellaPosition) -> tuple[float, int]:
        prior_strength = sum(
            abs(score)
            for score in (inference.position_letter_priors.get(pos.global_index, {}) if inference else {}).values()
        )
        learned_strength = memory_prior_strength(pos, memory)
        repeated = repeated_symbol_pressure(pos, positions)
        anchor = 1.0 if pos.allowed_letters is not None else 0.0
        return (-(prior_strength * 5.0 + learned_strength * 1.8 + repeated * 0.35 + anchor), pos.global_index)

    return sorted(unknowns, key=priority)
