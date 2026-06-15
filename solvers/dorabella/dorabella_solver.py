#!/usr/bin/env python3
"""Dorabella quantum-hybrid decoder orchestrator.

This file defines the algorithm but does not automatically run a brute-force
search. Use `--dry-run` to inspect the model and `--run-minutes 10` only when
you are ready to spend a daily quantum/classical session.
"""

from __future__ import annotations

import argparse
import random
import time
from datetime import datetime, timezone
from pathlib import Path

from .dorabella_constraints import (
    CipherState,
    build_positions,
    normalize_letter,
    seed_alphabet1_constraints,
    seed_fixed_skeleton_constraints,
    validate_plaintext,
)
from .dorabella_data import ROW_LENGTHS, ROW_SYMBOLS, SKELETONS, START_ANCHOR, DorabellaPosition
from .dorabella_global_reasoner import (
    assignment_for_word_hypothesis,
    assignment_signature,
    bijection_closure_trace,
    globally_ordered_unknowns,
    letter_frequency_assignment_bonus,
    merge_assignment_lists,
    orientation_trace,
    prior_assignments_for_block,
    prior_segment_assignments,
    propagate_forced_assignments,
    rows_from_state,
    state_signature,
    top_letters_for_position,
)
from .dorabella_memory import load_memory, remember_candidate, remember_rejection, save_memory
from .dorabella_memory_ai import (
    assignment_memory_bonus,
    candidate_neural_bonus,
    ensure_ai_memory,
    learn_from_run,
)
from .dorabella_lexical_lattice import (
    LexicalLattice,
    build_lexical_lattice,
    cryptogram_lattice_assignments,
    lexical_lattice_assignments,
    top_lattice_arcs_for_report,
)
from .dorabella_quantum import QuantumBlockSampler, QuantumConfig
from .dorabella_qnn import configure_qnn, qnn_expectation
from .dorabella_semantics import score_rows
from .dorabella_word_inference import (
    WordInference,
    build_word_inference,
    top_word_hypotheses_for_report,
    word_assignment_bonus,
)


def chunk_unknown_positions(
    positions: list[DorabellaPosition],
    max_size: int,
    seed_assignment: dict[int, str] | None = None,
    word_inference: WordInference | None = None,
    memory: dict[str, object] | None = None,
) -> list[list[DorabellaPosition]]:
    unknowns = globally_ordered_unknowns(positions, seed_assignment or {}, word_inference, memory=memory)
    return [unknowns[i : i + max_size] for i in range(0, len(unknowns), max_size)]


def rows_from_assignment(
    positions: list[DorabellaPosition],
    state: CipherState,
    assignment: dict[int, str],
) -> list[str]:
    return rows_from_state(positions, state, assignment)


def partial_candidate_record(
    positions: list[DorabellaPosition],
    state: CipherState,
    assignment: dict[int, str],
    block_number: int,
) -> dict[str, object]:
    rows = rows_from_assignment(positions, state, assignment)
    resolved_unknowns = sum(
        1
        for pos in positions
        if (pos.allowed_letters is None or len(pos.allowed_letters) != 1)
        and pos.global_index in assignment
    )
    total_unknowns = sum(1 for pos in positions if pos.allowed_letters is None or len(pos.allowed_letters) != 1)
    semantic = score_rows(rows)
    return {
        "block": block_number,
        "resolved_unknowns": resolved_unknowns,
        "total_unknowns": total_unknowns,
        "score": semantic.score,
        "rows": rows,
        "notes": semantic.notes,
        "semantic_details": semantic.details,
        "bijection_maps": state.maps,
        "orientation_trace": orientation_trace(positions, state, assignment),
        "bijection_closure": bijection_closure_trace(positions, state, assignment),
        "propagated_assignments": [
            {"position": idx + 1, "letter": letter}
            for idx, letter in sorted(assignment.items())
        ],
    }


def beam_priority(
    positions: list[DorabellaPosition],
    positions_by_index: dict[int, DorabellaPosition],
    state: CipherState,
    assignment: dict[int, str],
    memory: dict[str, object] | None = None,
    word_inference: WordInference | None = None,
) -> float:
    """Soft priority only; hard cipher validity is handled elsewhere."""
    rows = rows_from_state(positions, state, assignment)
    semantic = score_rows(rows)
    resolved = sum(1 for pos in positions if pos.global_index in assignment)
    memory_bonus = assignment_memory_bonus(memory, positions_by_index, assignment) if memory is not None else 0.0
    neural_bonus = 0.0
    if memory is not None:
        total_unknowns = sum(1 for pos in positions if pos.allowed_letters is None or len(pos.allowed_letters) != 1)
        candidate = {
            "rows": rows,
            "resolved_unknowns": resolved,
            "total_unknowns": total_unknowns,
        }
        neural_bonus = candidate_neural_bonus(memory, candidate)
        neural_bonus += qnn_expectation(memory, candidate) * 0.25
    word_bonus = word_assignment_bonus(word_inference, assignment) if word_inference is not None else 0.0
    frequency_bonus = letter_frequency_assignment_bonus(assignment)
    return (
        semantic.score
        + resolved * 0.01
        + memory_bonus * 0.35
        + neural_bonus * 0.45
        + word_bonus * 1.35
        + frequency_bonus * 0.16
    )


def apply_assignment_variants(
    state: CipherState,
    positions: list[DorabellaPosition],
    positions_by_index: dict[int, DorabellaPosition],
    base_assignment: dict[int, str],
    assignment: dict[int, str],
    max_variants: int = 16,
) -> list[tuple[CipherState, dict[int, str]]]:
    variants: list[tuple[CipherState, dict[int, str]]] = [
        (state.clone(), dict(base_assignment))
    ]
    for global_index, raw_letter in assignment.items():
        letter = normalize_letter(raw_letter)
        pos = positions_by_index[global_index]
        next_variants: list[tuple[CipherState, dict[int, str]]] = []
        for current_state, current_assignment in variants:
            current_letter = current_assignment.get(global_index)
            if current_letter is not None and current_letter != letter:
                continue
            if pos.allowed_letters is not None and letter not in pos.allowed_letters:
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
        seen: set[tuple[tuple[tuple[str, str], ...], ...] | tuple[tuple[int, str], ...]] = set()
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


def diversify_beam(
    items: list[tuple[CipherState, dict[int, str]]],
    positions: list[DorabellaPosition],
    positions_by_index: dict[int, DorabellaPosition],
    memory: dict[str, object] | None,
    word_inference: WordInference | None,
    beam_width: int,
    prefix_len: int = 12,
    per_prefix: int = 3,
    protected_prefixes: list[str] | None = None,
) -> list[tuple[CipherState, dict[int, str]]]:
    ranked = sorted(
        items,
        key=lambda item: beam_priority(
            positions,
            positions_by_index,
            item[0],
            item[1],
            memory,
            word_inference,
        ),
        reverse=True,
    )
    buckets: dict[str, list[tuple[CipherState, dict[int, str]]]] = {}
    seen_global: set[tuple[object, object]] = set()
    for item in ranked:
        key = (state_signature(item[0]), assignment_signature(item[1]))
        if key in seen_global:
            continue
        seen_global.add(key)
        rows = rows_from_state(positions, item[0], item[1])
        prefix = "".join(rows)[:prefix_len]
        buckets.setdefault(prefix, []).append(item)

    ordered_prefixes = sorted(
        buckets,
        key=lambda prefix: beam_priority(
            positions,
            positions_by_index,
            buckets[prefix][0][0],
            buckets[prefix][0][1],
            memory,
            word_inference,
        ),
        reverse=True,
    )
    selected: list[tuple[CipherState, dict[int, str]]] = []
    selected_keys: set[tuple[object, object]] = set()

    for prefix in protected_prefixes or []:
        bucket = buckets.get(prefix)
        if not bucket:
            continue
        item = bucket[0]
        key = (state_signature(item[0]), assignment_signature(item[1]))
        if key in selected_keys:
            continue
        selected_keys.add(key)
        selected.append(item)
        if len(selected) >= beam_width:
            return selected

    for round_idx in range(max(per_prefix, 1)):
        for prefix in ordered_prefixes:
            if round_idx >= len(buckets[prefix]):
                continue
            item = buckets[prefix][round_idx]
            key = (state_signature(item[0]), assignment_signature(item[1]))
            if key in selected_keys:
                continue
            selected_keys.add(key)
            selected.append(item)
            if len(selected) >= beam_width:
                return selected
    for prefix in ordered_prefixes:
        for item in buckets[prefix][per_prefix:]:
            key = (state_signature(item[0]), assignment_signature(item[1]))
            if key in selected_keys:
                continue
            selected_keys.add(key)
            selected.append(item)
            if len(selected) >= beam_width:
                return selected
    return selected


def seed_segment_branches(
    base_state: CipherState,
    positions: list[DorabellaPosition],
    positions_by_index: dict[int, DorabellaPosition],
    seed_assignment: dict[int, str],
    word_inference: WordInference | None,
    limit: int,
    orientation_variants: int,
) -> list[tuple[CipherState, dict[int, str]]]:
    branches: list[tuple[CipherState, dict[int, str]]] = [(base_state, seed_assignment)]
    if word_inference is None or limit <= 0:
        return branches
    for segment_assignment in prior_segment_assignments(
        positions,
        base_state,
        seed_assignment,
        word_inference,
        limit=limit,
    ):
        variants = apply_assignment_variants(
            base_state,
            positions,
            positions_by_index,
            seed_assignment,
            segment_assignment,
            max_variants=orientation_variants,
        )
        branches.extend(variants)
    deduped: list[tuple[CipherState, dict[int, str]]] = []
    seen: set[tuple[object, object]] = set()
    for item in branches:
        key = (state_signature(item[0]), assignment_signature(item[1]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
        if len(deduped) >= limit + 1:
            break
    return deduped


def seed_lattice_branches(
    base_state: CipherState,
    positions: list[DorabellaPosition],
    positions_by_index: dict[int, DorabellaPosition],
    seed_assignment: dict[int, str],
    lexical_lattice: LexicalLattice | None,
    limit: int,
    path_width: int,
    arcs_per_start: int,
    orientation_variants: int,
) -> list[tuple[CipherState, dict[int, str]]]:
    branches: list[tuple[CipherState, dict[int, str]]] = []
    if lexical_lattice is None or limit <= 0:
        return branches
    lattice_assignments = cryptogram_lattice_assignments(
        lexical_lattice,
        positions,
        base_state,
        seed_assignment,
        limit=limit,
        path_width=path_width,
        arcs_per_start=arcs_per_start,
        orientation_variants=orientation_variants,
    )
    if len(lattice_assignments) < limit:
        lattice_assignments.extend(
            lexical_lattice_assignments(
                lexical_lattice,
                positions,
                base_state,
                seed_assignment,
                limit=limit - len(lattice_assignments),
                path_width=path_width,
                arcs_per_start=arcs_per_start,
                orientation_variants=orientation_variants,
            )
        )
    for lattice_assignment in lattice_assignments[:limit]:
        branches.extend(
            apply_assignment_variants(
                base_state,
                positions,
                positions_by_index,
                seed_assignment,
                lattice_assignment,
                max_variants=orientation_variants,
            )
        )
    deduped: list[tuple[CipherState, dict[int, str]]] = []
    seen: set[tuple[object, object]] = set()
    for item in branches:
        key = (state_signature(item[0]), assignment_signature(item[1]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
        if len(deduped) >= limit:
            break
    return deduped


def seed_skeleton_ambiguity_branches(
    base_state: CipherState,
    positions: list[DorabellaPosition],
    positions_by_index: dict[int, DorabellaPosition],
    seed_assignment: dict[int, str],
    max_branches: int,
    orientation_variants: int,
) -> list[tuple[CipherState, dict[int, str]]]:
    """Branch fixed skeleton alternatives early."""
    branches: list[tuple[CipherState, dict[int, str]]] = [(base_state, seed_assignment)]
    ambiguous_positions = [
        pos for pos in positions
        if pos.allowed_letters is not None
        and len(pos.allowed_letters) > 1
        and pos.global_index not in seed_assignment
    ]
    for pos in ambiguous_positions:
        expanded: list[tuple[CipherState, dict[int, str]]] = []
        for state, assignment in branches:
            for letter in sorted(pos.allowed_letters):
                variants = apply_assignment_variants(
                    state,
                    positions,
                    positions_by_index,
                    assignment,
                    {pos.global_index: letter},
                    max_variants=orientation_variants,
                )
                expanded.extend(variants)
        deduped: list[tuple[CipherState, dict[int, str]]] = []
        seen: set[tuple[object, object]] = set()
        for item in expanded:
            key = (state_signature(item[0]), assignment_signature(item[1]))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
            if len(deduped) >= max_branches:
                break
        branches = deduped
        if not branches:
            break
    return branches


def state_from_report_maps(raw_maps: object) -> CipherState | None:
    if not isinstance(raw_maps, list):
        return None
    state = CipherState()
    for alphabet_index, mapping in enumerate(raw_maps[:4]):
        if not isinstance(mapping, dict):
            return None
        for symbol, letter in mapping.items():
            if not isinstance(symbol, str) or not isinstance(letter, str):
                return None
            if not state.bind(alphabet_index, symbol, letter):
                return None
    return state


def assignment_from_report(candidate: dict[str, object]) -> dict[int, str]:
    out: dict[int, str] = {}
    for item in candidate.get("propagated_assignments", []):
        if not isinstance(item, dict):
            continue
        position = item.get("position")
        letter = item.get("letter")
        if not isinstance(position, int) or not isinstance(letter, str):
            continue
        out[position - 1] = normalize_letter(letter)
    return out


def candidate_has_consistent_closure(candidate: dict[str, object]) -> bool:
    closure = candidate.get("bijection_closure", [])
    if not isinstance(closure, list):
        return False
    return all(
        not isinstance(item, dict) or bool(item.get("exact_all_consistent", True))
        for item in closure
    )


def warm_start_memory_branches(
    memory: dict[str, object],
    positions: list[DorabellaPosition],
    limit: int,
) -> list[tuple[CipherState, dict[int, str]]]:
    """Resume from previously saved frontiers without relaxing the bijection."""
    if limit <= 0:
        return []
    branches: list[tuple[float, CipherState, dict[int, str]]] = []
    runs = list(memory.get("runs", []))
    # The current run record is appended before beam construction and has no
    # frontier yet, so scan older runs from newest to oldest.
    for run_record in reversed(runs[:-1]):
        if not isinstance(run_record, dict):
            continue
        candidates = list(run_record.get("last_frontier", [])) or list(run_record.get("partial_candidates", []))
        for candidate in candidates:
            if not isinstance(candidate, dict) or not candidate_has_consistent_closure(candidate):
                continue
            state = state_from_report_maps(candidate.get("bijection_maps"))
            if state is None:
                continue
            assignment = assignment_from_report(candidate)
            propagated = propagate_forced_assignments(state, positions, assignment)
            if propagated is None:
                continue
            score = float(candidate.get("score") or -999.0)
            resolved = float(candidate.get("resolved_unknowns") or 0.0)
            branches.append((resolved + score * 0.01, state, propagated))
        if len(branches) >= limit * 3:
            break
    branches.sort(key=lambda item: item[0], reverse=True)
    out: list[tuple[CipherState, dict[int, str]]] = []
    seen: set[tuple[object, object]] = set()
    for _score, state, assignment in branches:
        key = (state_signature(state), assignment_signature(assignment))
        if key in seen:
            continue
        seen.add(key)
        out.append((state, assignment))
        if len(out) >= limit:
            break
    return out


def unresolved_block_positions(
    block: list[DorabellaPosition],
    state: CipherState,
    positions: list[DorabellaPosition],
    assignment: dict[int, str],
) -> tuple[dict[int, str] | None, list[DorabellaPosition]]:
    propagated = propagate_forced_assignments(state, positions, assignment)
    if propagated is None:
        return None, []
    unresolved = [pos for pos in block if pos.global_index not in propagated]
    return propagated, unresolved


def semantic_gate_metrics(rows: list[str]) -> dict[str, float]:
    semantic = score_rows(rows)
    details = semantic.details or {}
    row_details = details.get("rows", []) if isinstance(details, dict) else []
    row_scores = [
        float(item.get("row_score", 0.0))
        for item in row_details
        if isinstance(item, dict)
    ]
    unknown_tokens = sum(
        float(item.get("unknown_tokens", 0.0))
        for item in row_details
        if isinstance(item, dict)
    )
    single_noise = sum(
        float(item.get("single_noise", 0.0))
        for item in row_details
        if isinstance(item, dict)
    )
    rare_dictionary = sum(
        float(item.get("rare_dictionary", 0.0))
        for item in row_details
        if isinstance(item, dict)
    )
    token_count = sum(
        len(item.get("tokens", []))
        for item in row_details
        if isinstance(item, dict)
    )
    coverage = float(details.get("coverage", 0.0)) if isinstance(details, dict) else 0.0
    noise_ratio = (unknown_tokens + single_noise * 0.65 + rare_dictionary * 0.45) / max(token_count, 1)
    return {
        "score": semantic.score,
        "coverage": coverage,
        "worst_row_score": min(row_scores) if row_scores else semantic.score,
        "unknown_tokens": unknown_tokens,
        "single_noise": single_noise,
        "rare_dictionary": rare_dictionary,
        "token_count": float(token_count),
        "noise_ratio": noise_ratio,
    }


def semantic_gate_accept(metrics: dict[str, float], args: argparse.Namespace) -> bool:
    coverage = metrics["coverage"]
    if coverage < args.semantic_prune_min_coverage:
        return True
    if metrics["score"] < args.semantic_prune_min_score:
        return False
    if metrics["worst_row_score"] < args.semantic_prune_worst_row:
        return False
    if metrics["noise_ratio"] > args.semantic_prune_max_noise_ratio:
        return False
    return True


def semantic_prune_branches(
    branches: list[tuple[CipherState, dict[int, str]]],
    positions: list[DorabellaPosition],
    positions_by_index: dict[int, DorabellaPosition],
    memory: dict[str, object] | None,
    word_inference: WordInference | None,
    args: argparse.Namespace,
) -> tuple[list[tuple[CipherState, dict[int, str]]], dict[str, object]]:
    if not args.semantic_prune or not branches:
        return branches, {"enabled": bool(args.semantic_prune), "input": len(branches), "output": len(branches), "pruned": 0}

    scored: list[tuple[float, bool, tuple[CipherState, dict[int, str]], dict[str, float]]] = []
    accepted = []
    for item in branches:
        rows = rows_from_state(positions, item[0], item[1])
        metrics = semantic_gate_metrics(rows)
        accepted_flag = semantic_gate_accept(metrics, args)
        priority = beam_priority(
            positions,
            positions_by_index,
            item[0],
            item[1],
            memory,
            word_inference,
        )
        scored.append((priority, accepted_flag, item, metrics))
        if accepted_flag:
            accepted.append((priority, item, metrics))

    min_keep = min(max(args.semantic_prune_min_keep, args.beam_width), len(branches))
    accepted.sort(key=lambda item: item[0], reverse=True)
    kept_items = [item for _priority, item, _metrics in accepted]
    if len(kept_items) < min_keep:
        fallback = sorted(scored, key=lambda item: item[0], reverse=True)
        seen = {
            (state_signature(item[0]), assignment_signature(item[1]))
            for item in kept_items
        }
        for _priority, _accepted, item, _metrics in fallback:
            key = (state_signature(item[0]), assignment_signature(item[1]))
            if key in seen:
                continue
            seen.add(key)
            kept_items.append(item)
            if len(kept_items) >= min_keep:
                break

    kept_items = kept_items[: max(args.semantic_prune_max_keep, args.beam_width)]
    rejected_metrics = [metrics for _priority, accepted_flag, _item, metrics in scored if not accepted_flag]
    stats = {
        "enabled": True,
        "input": len(branches),
        "accepted": len(accepted),
        "output": len(kept_items),
        "pruned": max(0, len(branches) - len(kept_items)),
        "rejected": len(rejected_metrics),
        "min_rejected_score": min((m["score"] for m in rejected_metrics), default=None),
        "max_rejected_noise_ratio": max((m["noise_ratio"] for m in rejected_metrics), default=None),
    }
    return kept_items, stats


def assignment_coverage(
    positions: list[DorabellaPosition],
    assignment: dict[int, str],
) -> tuple[int, int, float]:
    total = sum(1 for pos in positions if pos.allowed_letters is None or len(pos.allowed_letters) != 1)
    resolved = sum(
        1
        for pos in positions
        if (pos.allowed_letters is None or len(pos.allowed_letters) != 1)
        and pos.global_index in assignment
    )
    return resolved, total, resolved / max(total, 1)


def global_completion_refinement(
    beam: list[tuple[CipherState, dict[int, str]]],
    positions: list[DorabellaPosition],
    positions_by_index: dict[int, DorabellaPosition],
    memory: dict[str, object] | None,
    word_inference: WordInference | None,
    args: argparse.Namespace,
    deadline: float,
) -> tuple[list[tuple[CipherState, dict[int, str]]], dict[str, object]]:
    if not args.global_refine or not beam:
        return beam, {"enabled": bool(args.global_refine), "input": len(beam), "output": len(beam)}

    refine_deadline = time.time() + max(args.global_refine_seconds, 0.0)
    seeds = dedupe_and_rank_branches(
        beam,
        positions,
        positions_by_index,
        memory,
        word_inference,
        args.global_refine_seeds,
    )
    refined: list[tuple[CipherState, dict[int, str]]] = []
    step_stats: list[dict[str, object]] = []
    full_candidates = 0
    best_coverage = 0.0

    for seed_idx, (seed_state, seed_assignment) in enumerate(seeds, start=1):
        if time.time() >= refine_deadline:
            break
        propagated = propagate_forced_assignments(seed_state, positions, seed_assignment)
        if propagated is None:
            continue
        frontier: list[tuple[CipherState, dict[int, str]]] = [(seed_state, propagated)]
        unknowns = globally_ordered_unknowns(
            positions,
            propagated,
            word_inference,
            memory=memory,
        )
        for step, pos in enumerate(unknowns, start=1):
            if time.time() >= refine_deadline or step > args.global_refine_max_steps:
                break
            expanded: list[tuple[CipherState, dict[int, str]]] = []
            for state, assignment in frontier:
                if pos.global_index in assignment:
                    expanded.append((state, assignment))
                    continue
                letters = top_letters_for_position(
                    pos,
                    state,
                    word_inference,
                    memory=memory,
                    limit=args.global_refine_letters,
                )
                for letter, _score in letters:
                    expanded.extend(
                        apply_assignment_variants(
                            state,
                            positions,
                            positions_by_index,
                            assignment,
                            {pos.global_index: letter},
                            max_variants=args.orientation_variants,
                        )
                    )
            if not expanded:
                break
            expanded, prune_stats = semantic_prune_branches(
                expanded,
                positions,
                positions_by_index,
                memory,
                word_inference,
                args,
            )
            frontier = diversify_beam(
                expanded,
                positions,
                positions_by_index,
                memory,
                word_inference,
                args.global_refine_width,
                prefix_len=args.diversity_prefix_len,
                per_prefix=args.diversity_per_prefix,
            )
            if not frontier:
                break
            coverages = [assignment_coverage(positions, item[1])[2] for item in frontier]
            best_coverage = max(best_coverage, max(coverages, default=0.0))
            step_stats.append(
                {
                    "seed": seed_idx,
                    "step": step,
                    "position": pos.global_index + 1,
                    "expanded": len(expanded),
                    "frontier": len(frontier),
                    "best_coverage": max(coverages, default=0.0),
                    "semantic_prune": prune_stats,
                }
            )
            if all(assignment_coverage(positions, item[1])[2] >= 1.0 for item in frontier):
                break
        refined.extend(frontier)

    combined = beam + refined
    unique: list[tuple[CipherState, dict[int, str]]] = []
    seen: set[tuple[object, object]] = set()
    for item in combined:
        key = (state_signature(item[0]), assignment_signature(item[1]))
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    keep = max(args.global_refine_width, args.beam_width)
    semantic_ranked = sorted(
        unique,
        key=lambda item: beam_priority(
            positions,
            positions_by_index,
            item[0],
            item[1],
            memory,
            word_inference,
        ),
        reverse=True,
    )
    coverage_ranked = sorted(
        unique,
        key=lambda item: (
            assignment_coverage(positions, item[1])[2],
            beam_priority(
                positions,
                positions_by_index,
                item[0],
                item[1],
                memory,
                word_inference,
            ),
        ),
        reverse=True,
    )
    ranked = []
    ranked_seen: set[tuple[object, object]] = set()
    for group in (semantic_ranked[:keep], coverage_ranked[:keep], semantic_ranked):
        for item in group:
            key = (state_signature(item[0]), assignment_signature(item[1]))
            if key in ranked_seen:
                continue
            ranked_seen.add(key)
            ranked.append(item)
            if len(ranked) >= keep:
                break
        if len(ranked) >= keep:
            break
    full_candidates = sum(1 for _state, assignment in ranked if assignment_coverage(positions, assignment)[2] >= 1.0)
    if ranked:
        best_coverage = max(best_coverage, max(assignment_coverage(positions, item[1])[2] for item in ranked))
    stats = {
        "enabled": True,
        "input": len(beam),
        "seeds": len(seeds),
        "refined": len(refined),
        "output": len(ranked),
        "full_candidates": full_candidates,
        "best_coverage": best_coverage,
        "steps": step_stats[-args.global_refine_report_steps :],
    }
    return ranked, stats


def dedupe_and_rank_branches(
    branches: list[tuple[CipherState, dict[int, str]]],
    positions: list[DorabellaPosition],
    positions_by_index: dict[int, DorabellaPosition],
    memory: dict[str, object] | None,
    word_inference: WordInference | None,
    limit: int,
) -> list[tuple[CipherState, dict[int, str]]]:
    seen: set[tuple[object, object]] = set()
    deduped: list[tuple[CipherState, dict[int, str]]] = []
    for item in branches:
        key = (state_signature(item[0]), assignment_signature(item[1]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    deduped.sort(
        key=lambda item: beam_priority(
            positions,
            positions_by_index,
            item[0],
            item[1],
            memory,
            word_inference,
        ),
        reverse=True,
    )
    return deduped[:limit]


def inspect_model() -> None:
    positions = build_positions(ROW_SYMBOLS, SKELETONS)
    state = CipherState()
    seeded = seed_alphabet1_constraints(positions, state)
    fixed_seeded = seed_fixed_skeleton_constraints(positions, state)
    print("Dorabella quantum-hybrid model")
    print(f"positions: {len(positions)}")
    print(f"row lengths: {ROW_LENGTHS}")
    print(f"alphabet-1 seeded: {seeded}")
    print(f"fixed skeleton seeded: {fixed_seeded}")
    print(f"known skeleton slots: {sum(1 for p in positions if p.allowed_letters is not None)}")
    print(f"unknown slots: {sum(1 for p in positions if p.allowed_letters is None)}")
    print("cycle counts:", [sum(1 for p in positions if p.alphabet_index == i) for i in range(4)])


def run_search(args: argparse.Namespace) -> None:
    positions = build_positions(ROW_SYMBOLS, SKELETONS)
    memory_path = Path(args.memory)
    memory = load_memory(memory_path)
    ensure_ai_memory(memory)
    qnn_state = configure_qnn(
        memory,
        enabled=bool(args.qnn_policy),
        qubits=args.qnn_qubits,
        layers=args.qnn_layers,
        train_limit=args.qnn_train_limit,
    )
    rejected = set(memory.get("rejected_signatures", {}).keys())

    rng = random.Random(args.seed)
    sampler = QuantumBlockSampler(
        QuantumConfig(
            mode=args.quantum_mode,
            backend=args.backend,
            shots=args.shots,
            max_block_unknowns=args.block_size,
            use_real_ibm_backend=args.real_ibm,
            ibm_token_env=args.ibm_token_env,
            ibm_instance=args.ibm_instance,
        ),
        rng=rng,
    )

    base_state = CipherState()
    seed_alphabet1_constraints(positions, base_state)
    seed_fixed_skeleton_constraints(positions, base_state)
    word_inference = build_word_inference(
        positions,
        max_hypotheses_per_window=args.word_hypotheses_per_window,
        hypothesis_limit=args.word_hypothesis_limit,
    )
    lexical_lattice = (
        build_lexical_lattice(
            positions,
            max_word_len=args.lattice_max_word_len,
            per_span=args.lattice_per_span,
            min_arc_score=args.lattice_min_arc_score,
        )
        if args.lexical_lattice
        else None
    )
    seed_assignment = propagate_forced_assignments(base_state, positions, {}) or {}
    blocks = chunk_unknown_positions(
        positions,
        args.block_size,
        seed_assignment=seed_assignment,
        word_inference=word_inference if args.word_inference else None,
        memory=memory,
    )
    deadline = time.time() + args.run_minutes * 60
    run_record = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "minutes": args.run_minutes,
        "backend": args.backend,
        "quantum_mode": args.quantum_mode,
        "shots": args.shots,
        "block_size": args.block_size,
        "blocks": len(blocks),
        "hypothesis": "skeleton_first_bella_elgar",
        "divergence_allowed": args.allow_skeleton_divergence,
        "word_inference_enabled": args.word_inference,
        "global_reasoning": {
            "seed_forced_assignments": len(seed_assignment),
            "block_order": [
                [pos.global_index + 1 for pos in block]
                for block in blocks
            ],
            "global_prior_candidates": args.global_prior_candidates,
            "segment_prior_candidates": args.segment_prior_candidates,
            "prior_letters_per_position": args.prior_letters_per_position,
            "orientation_variants": args.orientation_variants,
            "letter_frequency_prior": True,
            "memory_guided_letter_generation": True,
            "word_hypotheses_per_window": args.word_hypotheses_per_window,
            "word_hypothesis_limit": args.word_hypothesis_limit,
            "diversity_prefix_len": args.diversity_prefix_len,
            "diversity_per_prefix": args.diversity_per_prefix,
            "initial_segment_seeds": args.initial_segment_seeds,
            "branch_skeleton_ambiguities": args.branch_skeleton_ambiguities,
            "skeleton_ambiguity_branches": args.skeleton_ambiguity_branches,
            "initial_branch_cap": args.initial_branch_cap,
            "protected_word_hypotheses": args.protected_word_hypotheses,
            "memory_frontier_seeds": args.memory_frontier_seeds,
            "semantic_prune": args.semantic_prune,
            "semantic_prune_min_coverage": args.semantic_prune_min_coverage,
            "semantic_prune_min_score": args.semantic_prune_min_score,
            "semantic_prune_worst_row": args.semantic_prune_worst_row,
            "semantic_prune_max_noise_ratio": args.semantic_prune_max_noise_ratio,
            "global_refine": args.global_refine,
            "global_refine_seeds": args.global_refine_seeds,
            "global_refine_width": args.global_refine_width,
            "global_refine_letters": args.global_refine_letters,
            "lexical_lattice": args.lexical_lattice,
            "lattice_whole_cryptogram_seeds": True,
            "lattice_initial_seeds": args.lattice_initial_seeds,
            "lattice_candidates_per_beam": args.lattice_candidates_per_beam,
            "lattice_path_width": args.lattice_path_width,
            "lattice_arcs_per_start": args.lattice_arcs_per_start,
        },
        "qnn_policy": args.qnn_policy,
        "qnn_summary": {
            "enabled": qnn_state.get("enabled"),
            "qubits": qnn_state.get("qubits"),
            "layers": qnn_state.get("layers"),
            "train_limit": qnn_state.get("train_limit"),
            "updates_before_run": qnn_state.get("updates"),
            "parameter_count": sum(len(layer) for layer in qnn_state.get("params", [])),
        },
        "top_word_hypotheses": top_word_hypotheses_for_report(word_inference, args.report_word_hypotheses)
        if args.word_inference
        else [],
        "lexical_lattice": {
            "enabled": lexical_lattice is not None,
            "total_arcs": lexical_lattice.total_arcs if lexical_lattice is not None else 0,
            "row_arc_counts": lexical_lattice.row_arc_counts if lexical_lattice is not None else {},
            "config": lexical_lattice.config if lexical_lattice is not None else {},
            "top_arcs": top_lattice_arcs_for_report(lexical_lattice, args.report_lattice_arcs),
        },
        "block_stats": [],
        "partial_candidates": [],
        "last_frontier": [],
    }
    memory["runs"].append(run_record)

    positions_by_index = {pos.global_index: pos for pos in positions}
    skeleton_beam = (
        seed_skeleton_ambiguity_branches(
            base_state,
            positions,
            positions_by_index,
            seed_assignment,
            max_branches=args.skeleton_ambiguity_branches,
            orientation_variants=args.orientation_variants,
        )
        if args.branch_skeleton_ambiguities
        else [(base_state, seed_assignment)]
    )
    per_skeleton_segment_limit = max(1, args.initial_segment_seeds // max(len(skeleton_beam), 1))
    beam = []
    for state, assignment in skeleton_beam:
        beam.extend(
            seed_segment_branches(
                state,
                positions,
                positions_by_index,
                assignment,
                word_inference if args.word_inference else None,
                limit=per_skeleton_segment_limit,
                orientation_variants=args.orientation_variants,
            )
        )
        beam.extend(
            seed_lattice_branches(
                state,
                positions,
                positions_by_index,
                assignment,
                lexical_lattice,
                limit=max(1, args.lattice_initial_seeds // max(len(skeleton_beam), 1)),
                path_width=args.lattice_path_width,
                arcs_per_start=args.lattice_arcs_per_start,
                orientation_variants=args.orientation_variants,
            )
        )
    beam = dedupe_and_rank_branches(
        beam,
        positions,
        positions_by_index,
        memory,
        word_inference if args.word_inference else None,
        args.initial_branch_cap,
    )
    memory_frontier = warm_start_memory_branches(
        memory,
        positions,
        limit=args.memory_frontier_seeds,
    )
    beam.extend(memory_frontier)
    beam = dedupe_and_rank_branches(
        beam,
        positions,
        positions_by_index,
        memory,
        word_inference if args.word_inference else None,
        args.initial_branch_cap,
    )
    run_record["global_reasoning"]["skeleton_beam_size"] = len(skeleton_beam)
    run_record["global_reasoning"]["memory_frontier_size"] = len(memory_frontier)
    run_record["global_reasoning"]["initial_beam_size"] = len(beam)
    protected_prefixes = []
    protected_word_added = 0
    if args.protected_word_hypotheses > 0 and args.word_inference:
        for hypothesis in word_inference.hypotheses:
            if protected_word_added >= args.protected_word_hypotheses:
                break
            added_for_hypothesis = False
            for state0, assignment0 in skeleton_beam:
                segment_assignment = assignment_for_word_hypothesis(
                    hypothesis,
                    positions,
                    state0,
                    assignment0,
                )
                if segment_assignment is None:
                    continue
                variants = apply_assignment_variants(
                    state0,
                    positions,
                    positions_by_index,
                    assignment0,
                    segment_assignment,
                    max_variants=args.orientation_variants,
                )
                for state, assignment in variants:
                    prefix = "".join(rows_from_state(positions, state, assignment))[: args.diversity_prefix_len]
                    if START_ANCHOR and prefix.startswith(START_ANCHOR) and prefix not in protected_prefixes:
                        protected_prefixes.append(prefix)
                    beam.append((state, assignment))
                    added_for_hypothesis = True
                    break
                if added_for_hypothesis:
                    break
            if added_for_hypothesis:
                protected_word_added += 1
    beam = dedupe_and_rank_branches(
        beam,
        positions,
        positions_by_index,
        memory,
        word_inference if args.word_inference else None,
        args.initial_branch_cap,
    )
    for state, assignment in sorted(
        beam,
        key=lambda item: beam_priority(
            positions,
            positions_by_index,
            item[0],
            item[1],
            memory,
            word_inference if args.word_inference else None,
        ),
        reverse=True,
    ):
        prefix = "".join(rows_from_state(positions, state, assignment))[: args.diversity_prefix_len]
        if START_ANCHOR and prefix.startswith(START_ANCHOR) and prefix not in protected_prefixes:
            protected_prefixes.append(prefix)
        if len(protected_prefixes) >= args.protected_initial_prefixes:
            break
    run_record["global_reasoning"]["protected_prefixes"] = protected_prefixes
    run_record["global_reasoning"]["protected_word_added"] = protected_word_added
    for block in blocks:
        if time.time() >= deadline:
            break
        next_beam: list[tuple[CipherState, dict[int, str]]] = []
        local_sample_count = 0
        beam_in = len(beam)
        for state, partial in beam:
            propagated_partial, active_block = unresolved_block_positions(
                block,
                state,
                positions,
                partial,
            )
            if propagated_partial is None:
                continue
            if not active_block:
                next_beam.append((state, propagated_partial))
                continue
            prior_local = prior_assignments_for_block(
                active_block,
                state,
                word_inference if args.word_inference else None,
                memory=memory,
                per_position=args.prior_letters_per_position,
                limit=args.global_prior_candidates,
            )
            segment_local = prior_segment_assignments(
                positions,
                state,
                partial,
                word_inference if args.word_inference else None,
                limit=args.segment_prior_candidates,
            )
            lattice_local = lexical_lattice_assignments(
                lexical_lattice,
                positions,
                state,
                partial,
                limit=args.lattice_candidates_per_beam,
                path_width=args.lattice_path_width,
                arcs_per_start=args.lattice_arcs_per_start,
                orientation_variants=args.orientation_variants,
            )
            quantum_local = sampler.sample_block(active_block, state, rejected)
            local = merge_assignment_lists(lattice_local, segment_local, prior_local, quantum_local)
            local_sample_count += len(local)
            local_limit = max(args.keep_per_block, args.segment_prior_candidates, args.lattice_candidates_per_beam)
            for local_assignment in local[:local_limit]:
                variants = apply_assignment_variants(
                    state,
                    positions,
                    positions_by_index,
                    propagated_partial,
                    local_assignment,
                    max_variants=args.orientation_variants,
                )
                next_beam.extend(variants)
        next_beam, prune_stats = semantic_prune_branches(
            next_beam,
            positions,
            positions_by_index,
            memory,
            word_inference if args.word_inference else None,
            args,
        )
        beam = diversify_beam(
            next_beam,
            positions,
            positions_by_index,
            memory,
            word_inference if args.word_inference else None,
            args.beam_width,
            prefix_len=args.diversity_prefix_len,
            per_prefix=args.diversity_per_prefix,
            protected_prefixes=protected_prefixes,
        )
        block_number = len(run_record["block_stats"]) + 1
        snapshot = [
            partial_candidate_record(positions, state, partial, block_number)
            for state, partial in beam[: args.report_candidates_per_block]
        ]
        run_record["block_stats"].append(
            {
                "block": block_number,
                "block_positions": [pos.global_index + 1 for pos in block],
                "beam_in": beam_in,
                "local_samples": local_sample_count,
                "semantic_prune": prune_stats,
                "beam_out": len(beam),
            }
        )
        run_record["partial_candidates"].extend(snapshot)
        run_record["last_frontier"] = snapshot
        if not beam:
            break

    if beam and args.global_refine:
        beam, refine_stats = global_completion_refinement(
            beam,
            positions,
            positions_by_index,
            memory,
            word_inference if args.word_inference else None,
            args,
            deadline,
        )
        run_record["global_refinement"] = refine_stats
        snapshot = [
            {
                **partial_candidate_record(positions, state, partial, len(run_record["block_stats"]) + 1),
                "phase": "global_refinement",
            }
            for state, partial in beam[: args.report_candidates_per_block]
        ]
        run_record["partial_candidates"].extend(snapshot)
        run_record["last_frontier"] = snapshot
    else:
        run_record["global_refinement"] = {
            "enabled": bool(args.global_refine),
            "skipped": True,
            "reason": "empty_beam_or_deadline",
        }

    for _state, assignment in beam:
        rows = rows_from_assignment(positions, _state, assignment)
        ok, errors, _validated_state = validate_plaintext(rows, positions)
        semantic = score_rows(rows)
        if ok and semantic.score >= args.min_score:
            remember_candidate(memory, rows, semantic.score, semantic.notes)
        else:
            reason = errors[0].kind if errors else semantic.notes
            remember_rejection(memory, rows, reason, semantic.score)

    if args.memory_ai:
        learn_from_run(memory, positions, run_record)
        qnn_after = memory.get("active_memory", {}).get("qnn", {})
        run_record["qnn_summary"]["updates_after_run"] = qnn_after.get("updates")
        run_record["qnn_summary"]["last_loss"] = qnn_after.get("last_loss")
    save_memory(memory_path, memory)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Inspect the model without searching.")
    parser.add_argument("--run-minutes", type=float, default=0.0, help="Run bounded search for N minutes.")
    parser.add_argument("--memory", default="dorabella_memory.json")
    parser.add_argument(
        "--quantum-mode",
        choices=["local", "qiskit-aer", "ibm-runtime"],
        default="local",
        help="Sampling engine: local, Qiskit Aer simulator, or IBM Runtime.",
    )
    parser.add_argument("--backend", default="aer_simulator")
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--keep-per-block", type=int, default=64)
    parser.add_argument("--beam-width", type=int, default=256)
    parser.add_argument("--report-candidates-per-block", type=int, default=5)
    parser.add_argument("--min-score", type=float, default=0.25)
    parser.add_argument("--memory-ai", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--word-inference", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--word-hypotheses-per-window", type=int, default=24)
    parser.add_argument("--word-hypothesis-limit", type=int, default=2500)
    parser.add_argument("--global-prior-candidates", type=int, default=64)
    parser.add_argument("--segment-prior-candidates", type=int, default=24)
    parser.add_argument("--prior-letters-per-position", type=int, default=5)
    parser.add_argument("--orientation-variants", type=int, default=16)
    parser.add_argument("--diversity-prefix-len", type=int, default=12)
    parser.add_argument("--diversity-per-prefix", type=int, default=3)
    parser.add_argument("--initial-segment-seeds", type=int, default=32)
    parser.add_argument("--branch-skeleton-ambiguities", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skeleton-ambiguity-branches", type=int, default=64)
    parser.add_argument("--initial-branch-cap", type=int, default=160)
    parser.add_argument("--memory-frontier-seeds", type=int, default=24)
    parser.add_argument("--semantic-prune", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--semantic-prune-min-coverage", type=float, default=0.60)
    parser.add_argument("--semantic-prune-min-score", type=float, default=-5.20)
    parser.add_argument("--semantic-prune-worst-row", type=float, default=-7.80)
    parser.add_argument("--semantic-prune-max-noise-ratio", type=float, default=1.16)
    parser.add_argument("--semantic-prune-min-keep", type=int, default=8)
    parser.add_argument("--semantic-prune-max-keep", type=int, default=64)
    parser.add_argument("--global-refine", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--global-refine-seeds", type=int, default=6)
    parser.add_argument("--global-refine-width", type=int, default=10)
    parser.add_argument("--global-refine-letters", type=int, default=4)
    parser.add_argument("--global-refine-max-steps", type=int, default=56)
    parser.add_argument("--global-refine-seconds", type=float, default=8.0)
    parser.add_argument("--global-refine-report-steps", type=int, default=80)
    parser.add_argument("--lexical-lattice", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lattice-max-word-len", type=int, default=10)
    parser.add_argument("--lattice-per-span", type=int, default=8)
    parser.add_argument("--lattice-min-arc-score", type=float, default=-1.20)
    parser.add_argument("--lattice-initial-seeds", type=int, default=12)
    parser.add_argument("--lattice-candidates-per-beam", type=int, default=8)
    parser.add_argument("--lattice-path-width", type=int, default=8)
    parser.add_argument("--lattice-arcs-per-start", type=int, default=5)
    parser.add_argument("--report-lattice-arcs", type=int, default=40)
    parser.add_argument("--protected-initial-prefixes", type=int, default=16)
    parser.add_argument("--protected-word-hypotheses", type=int, default=24)
    parser.add_argument("--protected-start-hypotheses", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--qnn-policy", action="store_true")
    parser.add_argument("--qnn-qubits", type=int, default=4)
    parser.add_argument("--qnn-layers", type=int, default=4)
    parser.add_argument("--qnn-train-limit", type=int, default=12)
    parser.add_argument("--report-word-hypotheses", type=int, default=40)
    parser.add_argument(
        "--allow-skeleton-divergence",
        action="store_true",
        help="Reserved for later alternate skeleton masks. Hard bijection still cannot be broken.",
    )
    parser.add_argument("--seed", type=int, default=1897)
    parser.add_argument("--real-ibm", action="store_true")
    parser.add_argument("--ibm-token-env", default="IBM_QUANTUM_TOKEN")
    parser.add_argument("--ibm-instance", default=None)
    args = parser.parse_args()
    if args.protected_start_hypotheses is not None:
        args.protected_word_hypotheses = args.protected_start_hypotheses

    if args.dry_run or args.run_minutes <= 0:
        inspect_model()
        return
    run_search(args)


if __name__ == "__main__":
    main()
