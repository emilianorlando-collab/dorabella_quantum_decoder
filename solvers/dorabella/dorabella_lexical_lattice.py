"""Lexical lattice search for Dorabella.

The normal solver expands letters and short word hypotheses. This module adds
row-level word paths: every arc is an English token that fits the skeleton
window, and every emitted assignment is checked against the four rotating
alphabet bijection before the solver can use it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .dorabella_constraints import CipherState, normalize_letter
from .dorabella_data import CANONICAL_LETTERS, DorabellaPosition, ROW_LENGTHS
from .dorabella_global_reasoner import (
    assignment_signature,
    can_bind_position,
    propagate_forced_assignments,
    state_signature,
)
from .dorabella_language_model import LEXICON
from .dorabella_word_inference import compatible_word, lexicon_candidates_by_length, row_offsets, word_score


@dataclass(frozen=True)
class LexicalArc:
    row: int
    start_col: int
    end_col: int
    internal: str
    display: str
    kind: str
    pos: str
    score: float
    reason: str


@dataclass(frozen=True)
class LatticePath:
    score: float
    state: CipherState
    assignment: dict[int, str]
    arcs: tuple[LexicalArc, ...]
    last_pos: str | None
    gaps: int


@dataclass(frozen=True)
class LexicalLattice:
    arcs_by_row_start: dict[tuple[int, int], tuple[LexicalArc, ...]]
    row_arc_counts: dict[int, int]
    total_arcs: int
    config: dict[str, object]


_LATTICE_CACHE: dict[tuple[object, ...], LexicalLattice] = {}


def lattice_cache_key(
    positions: Sequence[DorabellaPosition],
    max_word_len: int,
    per_span: int,
    min_arc_score: float,
) -> tuple[object, ...]:
    mask_signature = tuple(
        (
            pos.row,
            pos.col,
            tuple(sorted(pos.allowed_letters)) if pos.allowed_letters is not None else None,
        )
        for pos in positions
    )
    return (max_word_len, per_span, round(min_arc_score, 4), mask_signature)


def transition_score(prev_pos: str | None, pos: str) -> float:
    if prev_pos is None:
        return 0.16 if pos in {"pron", "noun", "det", "adj", "question", "adv"} else -0.08
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
        ("unknown", "unknown"),
    }
    if (prev_pos, pos) in good:
        return 0.22
    if (prev_pos, pos) in bad:
        return -0.24
    return 0.0


def row_start_offset(row: int) -> int:
    return row_offsets()[row - 1]


def row_positions(positions: Sequence[DorabellaPosition], row: int) -> list[DorabellaPosition]:
    return [pos for pos in positions if pos.row == row]


def lattice_candidates_by_length() -> dict[int, list[tuple[str, str, str, str, float, str]]]:
    grouped = lexicon_candidates_by_length()
    one_letter: list[tuple[str, str, str, str, float, str]] = []
    for word in ("A", "I"):
        token = LEXICON.get(word)
        if token is not None:
            one_letter.append((token.text, token.display, token.kind, token.pos, 0.0, "lexicon"))
    if one_letter:
        grouped = {**grouped, 1: one_letter}
    return grouped


def build_lexical_lattice(
    positions: Sequence[DorabellaPosition],
    max_word_len: int = 10,
    per_span: int = 10,
    min_arc_score: float = -1.20,
) -> LexicalLattice:
    key = lattice_cache_key(positions, max_word_len, per_span, min_arc_score)
    cached = _LATTICE_CACHE.get(key)
    if cached is not None:
        return cached

    candidates_by_length = lattice_candidates_by_length()
    arcs_by_row_start: dict[tuple[int, int], tuple[LexicalArc, ...]] = {}
    row_arc_counts: dict[int, int] = {}

    for row in range(1, len(ROW_LENGTHS) + 1):
        row_pos = row_positions(positions, row)
        row_arc_counts[row] = 0
        for start_idx in range(0, len(row_pos)):
            start_col = start_idx + 1
            for end_idx in range(start_idx + 1, min(len(row_pos), start_idx + max_word_len) + 1):
                window = [pos.allowed_letters for pos in row_pos[start_idx:end_idx]]
                length = end_idx - start_idx
                known_slots = sum(1 for slot in window if slot is not None and len(slot) == 1)
                ambiguous_slots = sum(1 for slot in window if slot is not None and len(slot) > 1)
                anchor_strength = known_slots + ambiguous_slots * 0.55
                matches: list[LexicalArc] = []
                for internal, display, kind, pos, period_score, source in candidates_by_length.get(length, []):
                    if not compatible_word(internal, window):
                        continue
                    score = word_score(
                        internal,
                        display,
                        kind,
                        pos,
                        known_slots,
                        ambiguous_slots,
                        period_score,
                    )
                    if source == "web2" and known_slots + ambiguous_slots == 0:
                        score -= 0.60
                    if source == "web2" and length <= 4:
                        score -= 0.26
                    if source == "web2" and anchor_strength < 2.0:
                        score -= 0.26
                        if length >= 7:
                            score -= 0.32
                    if "noun_or_verb" in kind and anchor_strength < 2.0:
                        score -= 0.18
                    if period_score < 0.18 and anchor_strength < 2.0:
                        score -= 0.22
                    if score < min_arc_score:
                        continue
                    matches.append(
                        LexicalArc(
                            row=row,
                            start_col=start_col,
                            end_col=end_idx,
                            internal=internal,
                            display=display,
                            kind=kind,
                            pos=pos,
                            score=score,
                            reason=f"{kind}; source={source}; anchored={known_slots}; ambiguous={ambiguous_slots}",
                        )
                    )
                matches.sort(key=lambda arc: arc.score, reverse=True)
                if matches:
                    selected = tuple(matches[:per_span])
                    arcs_by_row_start[(row, start_col)] = selected
                    row_arc_counts[row] += len(selected)

    total = sum(row_arc_counts.values())
    lattice = LexicalLattice(
        arcs_by_row_start=arcs_by_row_start,
        row_arc_counts=row_arc_counts,
        total_arcs=total,
        config={
            "max_word_len": max_word_len,
            "per_span": per_span,
            "min_arc_score": min_arc_score,
            "cache": "process",
        },
    )
    _LATTICE_CACHE[key] = lattice
    return lattice


def arc_assignment(arc: LexicalArc) -> dict[int, str]:
    offset = row_start_offset(arc.row)
    start = offset + arc.start_col - 1
    return {
        start + rel: normalize_letter(letter)
        for rel, letter in enumerate(arc.internal)
    }


def apply_arc_variants(
    arc: LexicalArc,
    positions: Sequence[DorabellaPosition],
    state: CipherState,
    base_assignment: dict[int, str],
    max_variants: int,
) -> list[tuple[CipherState, dict[int, str]]]:
    variants: list[tuple[CipherState, dict[int, str]]] = [(state.clone(), dict(base_assignment))]
    for global_index, letter in arc_assignment(arc).items():
        pos = positions[global_index]
        if letter not in CANONICAL_LETTERS:
            return []
        if pos.allowed_letters is not None and letter not in pos.allowed_letters:
            return []
        next_variants: list[tuple[CipherState, dict[int, str]]] = []
        for current_state, current_assignment in variants:
            current = current_assignment.get(global_index)
            if current is not None and current != letter:
                continue
            if not can_bind_position(current_state, pos, letter):
                continue
            for sym in pos.symbol_options:
                trial_state = current_state.clone()
                if not trial_state.bind(pos.alphabet_index, sym, letter):
                    continue
                trial_assignment = dict(current_assignment)
                trial_assignment[global_index] = letter
                propagated = propagate_forced_assignments(trial_state, positions, trial_assignment)
                if propagated is None:
                    continue
                next_variants.append((trial_state, propagated))
        deduped: list[tuple[CipherState, dict[int, str]]] = []
        seen: set[tuple[object, object]] = set()
        for item_state, item_assignment in next_variants:
            key = (state_signature(item_state), assignment_signature(item_assignment))
            if key in seen:
                continue
            seen.add(key)
            deduped.append((item_state, item_assignment))
            if len(deduped) >= max_variants:
                break
        variants = deduped
        if not variants:
            return []
    return variants


def path_sort_key(path: LatticePath) -> tuple[float, int]:
    return (path.score, -path.gaps)


def row_lattice_paths(
    lattice: LexicalLattice,
    positions: Sequence[DorabellaPosition],
    row: int,
    state: CipherState,
    assignment: dict[int, str],
    path_width: int = 10,
    arcs_per_start: int = 6,
    orientation_variants: int = 8,
) -> list[LatticePath]:
    row_len = ROW_LENGTHS[row - 1]
    start_path = LatticePath(
        score=0.0,
        state=state.clone(),
        assignment=dict(assignment),
        arcs=tuple(),
        last_pos=None,
        gaps=0,
    )
    paths_by_col: dict[int, list[LatticePath]] = {1: [start_path]}
    for col in range(1, row_len + 1):
        current_paths = sorted(paths_by_col.get(col, []), key=path_sort_key, reverse=True)[:path_width]
        if not current_paths:
            continue
        for path in current_paths:
            # Gap transition keeps the path alive but is costly; it represents
            # "no confident word starts here yet", not a proposed plaintext.
            gap_path = LatticePath(
                score=path.score - 0.74,
                state=path.state,
                assignment=path.assignment,
                arcs=path.arcs,
                last_pos=path.last_pos,
                gaps=path.gaps + 1,
            )
            paths_by_col.setdefault(col + 1, []).append(gap_path)

            for arc in lattice.arcs_by_row_start.get((row, col), ())[:arcs_per_start]:
                variants = apply_arc_variants(
                    arc,
                    positions,
                    path.state,
                    path.assignment,
                    max_variants=orientation_variants,
                )
                for next_state, next_assignment in variants:
                    next_score = path.score + arc.score + transition_score(path.last_pos, arc.pos) - 0.18
                    next_path = LatticePath(
                        score=next_score,
                        state=next_state,
                        assignment=next_assignment,
                        arcs=path.arcs + (arc,),
                        last_pos=arc.pos,
                        gaps=path.gaps,
                    )
                    paths_by_col.setdefault(arc.end_col + 1, []).append(next_path)
        for target_col in range(col + 1, min(row_len + 2, col + 11)):
            if target_col in paths_by_col:
                paths_by_col[target_col] = sorted(
                    paths_by_col[target_col],
                    key=path_sort_key,
                    reverse=True,
                )[:path_width]
    return sorted(paths_by_col.get(row_len + 1, []), key=path_sort_key, reverse=True)[:path_width]


def lexical_lattice_assignments(
    lattice: LexicalLattice | None,
    positions: Sequence[DorabellaPosition],
    state: CipherState,
    assignment: dict[int, str],
    limit: int,
    path_width: int = 10,
    arcs_per_start: int = 6,
    orientation_variants: int = 8,
) -> list[dict[int, str]]:
    if lattice is None or limit <= 0:
        return []
    ranked: list[tuple[float, dict[int, str]]] = []
    for row in range(1, len(ROW_LENGTHS) + 1):
        for path in row_lattice_paths(
            lattice,
            positions,
            row,
            state,
            assignment,
            path_width=path_width,
            arcs_per_start=arcs_per_start,
            orientation_variants=orientation_variants,
        ):
            if not path.arcs:
                continue
            added = {
                idx: letter
                for idx, letter in path.assignment.items()
                if assignment.get(idx) != letter
            }
            if not added:
                continue
            coverage_bonus = len(added) * 0.025
            ranked.append((path.score + coverage_bonus - path.gaps * 0.10, added))
    ranked.sort(key=lambda item: item[0], reverse=True)
    out: list[dict[int, str]] = []
    seen: set[tuple[tuple[int, str], ...]] = set()
    for _score, item in ranked:
        key = assignment_signature(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
        if len(out) >= limit:
            break
    return out


def cryptogram_lattice_assignments(
    lattice: LexicalLattice | None,
    positions: Sequence[DorabellaPosition],
    state: CipherState,
    assignment: dict[int, str],
    limit: int,
    path_width: int = 8,
    arcs_per_start: int = 5,
    orientation_variants: int = 8,
) -> list[dict[int, str]]:
    """Build whole-cryptogram lexical seeds under one shared bijection state."""
    if lattice is None or limit <= 0:
        return []

    frontier = [
        LatticePath(
            score=0.0,
            state=state.clone(),
            assignment=dict(assignment),
            arcs=tuple(),
            last_pos=None,
            gaps=0,
        )
    ]
    row_width = max(2, min(path_width, 5))
    beam_width = max(limit, path_width)

    for row in range(1, len(ROW_LENGTHS) + 1):
        expanded: list[LatticePath] = []
        for path in frontier:
            row_paths = row_lattice_paths(
                lattice,
                positions,
                row,
                path.state,
                path.assignment,
                path_width=row_width,
                arcs_per_start=arcs_per_start,
                orientation_variants=orientation_variants,
            )
            useful_row_paths = [row_path for row_path in row_paths if row_path.arcs]
            if not useful_row_paths:
                expanded.append(
                    LatticePath(
                        score=path.score - ROW_LENGTHS[row - 1] * 0.08,
                        state=path.state,
                        assignment=path.assignment,
                        arcs=path.arcs,
                        last_pos=path.last_pos,
                        gaps=path.gaps + ROW_LENGTHS[row - 1],
                    )
                )
                continue
            for row_path in useful_row_paths:
                first_pos = row_path.arcs[0].pos
                cross_row = transition_score(path.last_pos, first_pos) if path.last_pos is not None else 0.0
                expanded.append(
                    LatticePath(
                        score=path.score + row_path.score + cross_row - row_path.gaps * 0.06,
                        state=row_path.state,
                        assignment=row_path.assignment,
                        arcs=path.arcs + row_path.arcs,
                        last_pos=row_path.last_pos,
                        gaps=path.gaps + row_path.gaps,
                    )
                )
        deduped: list[LatticePath] = []
        seen: set[tuple[object, object]] = set()
        for item in sorted(expanded, key=path_sort_key, reverse=True):
            key = (state_signature(item.state), assignment_signature(item.assignment))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
            if len(deduped) >= beam_width:
                break
        frontier = deduped
        if not frontier:
            break

    ranked = sorted(frontier, key=path_sort_key, reverse=True)
    out: list[dict[int, str]] = []
    seen_assignments: set[tuple[tuple[int, str], ...]] = set()
    for path in ranked:
        added = {
            idx: letter
            for idx, letter in path.assignment.items()
            if assignment.get(idx) != letter
        }
        if not added:
            continue
        key = assignment_signature(added)
        if key in seen_assignments:
            continue
        seen_assignments.add(key)
        out.append(added)
        if len(out) >= limit:
            break
    return out


def top_lattice_arcs_for_report(lattice: LexicalLattice | None, limit: int = 40) -> list[dict[str, object]]:
    if lattice is None:
        return []
    arcs = [arc for group in lattice.arcs_by_row_start.values() for arc in group]
    arcs.sort(key=lambda item: item.score, reverse=True)
    return [
        {
            "row": arc.row,
            "start_col": arc.start_col,
            "end_col": arc.end_col,
            "display": arc.display,
            "internal": arc.internal,
            "pos": arc.pos,
            "kind": arc.kind,
            "score": arc.score,
            "reason": arc.reason,
        }
        for arc in arcs[:limit]
    ]
