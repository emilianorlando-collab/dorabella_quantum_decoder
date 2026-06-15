#!/usr/bin/env python3
"""Run the phrase-first Dorabella solver and write a JSON report."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "solvers"))

from dorabella.dorabella_constraints import build_positions
from dorabella.dorabella_constraints import CipherState, normalize_letter, seed_alphabet1_constraints, seed_fixed_skeleton_constraints
from dorabella.dorabella_data import ROW_SYMBOLS, SKELETONS
from dorabella.dorabella_language_model import score_partial_or_full_row
from dorabella.dorabella_memory import load_memory, save_memory
from dorabella.dorabella_memory_ai import ensure_ai_memory, learn_from_run
from dorabella.dorabella_qnn import configure_qnn
from dorabella.dorabella_phrase_solver import solve_phrase_first


def rows_with_placeholders(rows: list[str]) -> str:
    return "".join(rows)


def cipher_trace(rows: list[str]) -> dict[str, object]:
    """Explain how a candidate fills blanks and induces the 4 alphabets."""
    positions = build_positions(ROW_SYMBOLS, SKELETONS)
    flat = rows_with_placeholders(rows)
    state = CipherState()
    seed_alphabet1_constraints(positions, state)
    seed_fixed_skeleton_constraints(positions, state)

    filled_blanks = []
    ambiguous_choices = []
    position_trace = []
    errors = []

    for pos in positions:
        raw = flat[pos.global_index] if pos.global_index < len(flat) else "_"
        if not raw.isalpha():
            continue
        letter = normalize_letter(raw)
        if pos.allowed_letters is None:
            filled_blanks.append(
                {
                    "global_position": pos.global_index + 1,
                    "row": pos.row,
                    "col": pos.col,
                    "alphabet": pos.alphabet_index + 1,
                    "symbol_options": list(pos.symbol_options),
                    "letter": letter,
                }
            )
        elif len(pos.allowed_letters) > 1:
            ambiguous_choices.append(
                {
                    "global_position": pos.global_index + 1,
                    "row": pos.row,
                    "col": pos.col,
                    "alphabet": pos.alphabet_index + 1,
                    "symbol_options": list(pos.symbol_options),
                    "allowed": sorted(pos.allowed_letters),
                    "chosen": letter,
                }
            )

        if pos.allowed_letters is not None and letter not in pos.allowed_letters:
            errors.append(
                {
                    "type": "skeleton",
                    "global_position": pos.global_index + 1,
                    "row": pos.row,
                    "col": pos.col,
                    "letter": letter,
                    "allowed": sorted(pos.allowed_letters),
                }
            )
            continue

        chosen_symbol = None
        for sym in pos.symbol_options:
            trial = state.clone()
            if trial.bind(pos.alphabet_index, sym, letter):
                state = trial
                chosen_symbol = sym
                break
        if chosen_symbol is None:
            errors.append(
                {
                    "type": "bijection",
                    "global_position": pos.global_index + 1,
                    "row": pos.row,
                    "col": pos.col,
                    "alphabet": pos.alphabet_index + 1,
                    "symbol_options": list(pos.symbol_options),
                    "letter": letter,
                }
            )
            continue

        position_trace.append(
            {
                "global_position": pos.global_index + 1,
                "row": pos.row,
                "col": pos.col,
                "alphabet": pos.alphabet_index + 1,
                "symbol": chosen_symbol,
                "symbol_options": list(pos.symbol_options),
                "letter": letter,
            }
        )

    maps = []
    propagations = []
    flat_letters = [normalize_letter(ch) if ch.isalpha() else "_" for ch in flat]
    chosen_by_position = {
        item["global_position"]: item
        for item in position_trace
    }
    for alpha_idx, mapping in enumerate(state.maps, start=1):
        for sym, letter in sorted(mapping.items()):
            chosen_occurrences = []
            possible_occurrences = []
            for pos in positions:
                if pos.alphabet_index + 1 != alpha_idx:
                    continue
                if sym not in pos.symbol_options:
                    continue
                actual = flat_letters[pos.global_index] if pos.global_index < len(flat_letters) else "_"
                possible = {
                    "global_position": pos.global_index + 1,
                    "row": pos.row,
                    "col": pos.col,
                    "actual_letter": actual,
                    "symbol_options": list(pos.symbol_options),
                    "chosen_symbol": chosen_by_position.get(pos.global_index + 1, {}).get("symbol"),
                    "matches_mapping": actual in {"_", letter},
                }
                possible_occurrences.append(possible)
                if chosen_by_position.get(pos.global_index + 1, {}).get("symbol") == sym:
                    chosen_occurrences.append(
                        {
                            "global_position": pos.global_index + 1,
                            "row": pos.row,
                            "col": pos.col,
                            "actual_letter": actual,
                            "matches_mapping": actual in {"_", letter},
                        }
                    )
            maps.append({"alphabet": alpha_idx, "symbol": sym, "letter": letter})
            propagations.append(
                {
                    "alphabet": alpha_idx,
                    "symbol": sym,
                    "letter": letter,
                    "chosen_occurrences": chosen_occurrences,
                    "possible_occurrences": possible_occurrences,
                }
            )

    return {
        "filled_blank_count": len(filled_blanks),
        "filled_blanks": filled_blanks,
        "ambiguous_choices": ambiguous_choices,
        "bijection_maps": maps,
        "position_trace": position_trace,
        "propagations": propagations,
        "errors": errors,
    }


def tokenized_rows(rows: list[str]) -> list[dict[str, object]]:
    out = []
    for row in rows:
        tok = score_partial_or_full_row(row)
        out.append(
            {
                "raw": row,
                "words": " ".join(token.display for token in tok.tokens),
                "score": tok.score,
                "notes": tok.notes,
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory", default=str(ROOT / "reports" / "dorabella_phrase_memory.json"))
    parser.add_argument("--report", default=str(ROOT / "reports" / "dorabella_phrase_report.json"))
    parser.add_argument("--beam-width", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--qnn-policy", action="store_true")
    parser.add_argument("--qnn-qubits", type=int, default=4)
    parser.add_argument("--qnn-layers", type=int, default=4)
    parser.add_argument("--qnn-train-limit", type=int, default=12)
    args = parser.parse_args()

    memory_path = Path(args.memory)
    report_path = Path(args.report)
    memory = load_memory(memory_path)
    ensure_ai_memory(memory)
    qnn_state = configure_qnn(
        memory,
        enabled=bool(args.qnn_policy),
        qubits=args.qnn_qubits,
        layers=args.qnn_layers,
        train_limit=args.qnn_train_limit,
    )

    results = solve_phrase_first(memory=memory, beam_width=args.beam_width, max_steps=args.max_steps)
    positions = build_positions(ROW_SYMBOLS, SKELETONS)
    run_record = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "solver": "phrase_first",
        "beam_width": args.beam_width,
        "max_steps": args.max_steps,
        "qnn_policy": args.qnn_policy,
        "qnn_summary": {
            "enabled": qnn_state.get("enabled"),
            "qubits": qnn_state.get("qubits"),
            "layers": qnn_state.get("layers"),
            "train_limit": qnn_state.get("train_limit"),
            "updates_before_run": qnn_state.get("updates"),
            "parameter_count": sum(len(layer) for layer in qnn_state.get("params", [])),
        },
        "partial_candidates": [
            {
                "rows": item["rows"],
                "resolved_unknowns": item["resolved_unknowns"],
                "total_unknowns": item["total_unknowns"],
                "score": item["score"],
            }
            for item in results[: args.top]
        ],
    }
    memory.setdefault("runs", []).append(run_record)
    learn_from_run(memory, positions, run_record)
    qnn_after = memory.get("active_memory", {}).get("qnn", {})
    run_record["qnn_summary"]["updates_after_run"] = qnn_after.get("updates")
    run_record["qnn_summary"]["last_loss"] = qnn_after.get("last_loss")
    save_memory(memory_path, memory)

    enriched = []
    for item in results[: args.top]:
        enriched.append(
            {
                **item,
                "tokenized_rows": tokenized_rows(item["rows"]),
                "cipher_trace": cipher_trace(item["rows"]),
            }
        )

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "ok" if results else "no_candidates",
        "solver": "phrase_first",
        "memory_file": str(memory_path),
        "beam_width": args.beam_width,
        "max_steps": args.max_steps,
        "qnn_policy": args.qnn_policy,
        "qnn_training": run_record.get("qnn_training"),
        "qnn_summary": run_record.get("qnn_summary"),
        "best_hallazgo": enriched[0] if enriched else None,
        "top_hallazgos": enriched,
        "active_memory_summary": {
            "neural_kind": memory.get("active_memory", {}).get("neural", {}).get("kind"),
            "layers": memory.get("active_memory", {}).get("neural", {}).get("layers"),
            "updates": memory.get("active_memory", {}).get("neural", {}).get("updates"),
            "replay_buffer_size": len(memory.get("active_memory", {}).get("neural", {}).get("replay_buffer", [])),
            "position_letter_scores": len(memory.get("active_memory", {}).get("position_letter_scores", {})),
            "symbol_letter_scores": len(memory.get("active_memory", {}).get("symbol_letter_scores", {})),
            "qnn": {
                "enabled": memory.get("active_memory", {}).get("qnn", {}).get("enabled"),
                "qubits": memory.get("active_memory", {}).get("qnn", {}).get("qubits"),
                "layers": memory.get("active_memory", {}).get("qnn", {}).get("layers"),
                "updates": memory.get("active_memory", {}).get("qnn", {}).get("updates"),
                "last_loss": memory.get("active_memory", {}).get("qnn", {}).get("last_loss"),
            },
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(report_path), "memory": str(memory_path), "candidates": len(results)}, indent=2))


if __name__ == "__main__":
    main()
