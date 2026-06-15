#!/usr/bin/env python3
"""Run repeated memory/QNN Dorabella solver cycles.

Each cycle calls the normal solver with the same memory file, then reloads the
memory and summarizes whether the best bijection-valid frontier candidate has
crossed a requested decoded-coverage target.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "solvers"))

from dorabella.dorabella_solver import run_search


def coverage(candidate: dict[str, Any]) -> float:
    total = float(candidate.get("total_unknowns") or 0)
    if total <= 0:
        return 0.0
    return float(candidate.get("resolved_unknowns") or 0) / total


def closure_is_consistent(candidate: dict[str, Any]) -> bool:
    closure = candidate.get("bijection_closure") or []
    return all(item.get("exact_all_consistent", True) for item in closure)


def candidate_quality_key(candidate: dict[str, Any]) -> tuple[float, bool, float]:
    return (
        coverage(candidate),
        closure_is_consistent(candidate),
        float(candidate.get("score") or -999.0),
    )


def semantic_quality_key(candidate: dict[str, Any]) -> tuple[bool, float, float]:
    return (
        closure_is_consistent(candidate),
        float(candidate.get("score") or -999.0),
        coverage(candidate),
    )


def tokenized_rows(candidate: dict[str, Any]) -> list[str]:
    details = candidate.get("semantic_details") or {}
    rows = details.get("rows") if isinstance(details, dict) else []
    out: list[str] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        tokens = row.get("tokens") or []
        out.append(" ".join(str(token) for token in tokens))
    return out


def best_candidates(run_record: dict[str, Any], limit: int, semantic: bool = False) -> list[dict[str, Any]]:
    candidates = list(run_record.get("partial_candidates") or [])
    candidates.sort(key=semantic_quality_key if semantic else candidate_quality_key, reverse=True)
    out = []
    seen: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
    for item in candidates:
        spaced = tokenized_rows(item)
        rows = item.get("rows") or []
        key = (tuple(str(row) for row in rows), tuple(spaced))
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "coverage": coverage(item),
                "resolved_unknowns": item.get("resolved_unknowns"),
                "total_unknowns": item.get("total_unknowns"),
                "score": item.get("score"),
                "bijection_consistent": closure_is_consistent(item),
                "rows": item.get("rows"),
                "tokenized_rows": spaced,
                "text_spaced": " / ".join(spaced),
                "notes": item.get("notes"),
                "semantic_details": item.get("semantic_details"),
                "propagated_assignments": item.get("propagated_assignments"),
                "bijection_maps": item.get("bijection_maps"),
                "bijection_closure": item.get("bijection_closure"),
            }
        )
        if len(out) >= limit:
            break
    return out


def phrase_text(candidate: dict[str, Any]) -> str | None:
    spaced = candidate.get("text_spaced")
    if isinstance(spaced, str) and spaced:
        return spaced
    rows = candidate.get("rows")
    if isinstance(rows, list):
        return " / ".join(str(row) for row in rows)
    return None


def compact_console_candidate(candidate: dict[str, Any] | None) -> dict[str, Any] | None:
    if not candidate:
        return None
    return {
        "coverage": candidate.get("coverage"),
        "score": candidate.get("score"),
        "bijection_consistent": candidate.get("bijection_consistent"),
        "phrase": phrase_text(candidate),
        "rows": candidate.get("rows"),
        "tokenized_rows": candidate.get("tokenized_rows"),
    }


def console_summary(report: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    best = report.get("best_overall") or {}
    best_semantic = report.get("best_semantic_overall") or {}
    latest_cycle = (report.get("cycles") or [{}])[-1]
    blocks_completed = int(latest_cycle.get("blocks_completed") or 0)
    warning = None
    if args.quantum_mode == "ibm-runtime" and blocks_completed == 0:
        warning = (
            "IBM Runtime was selected, but no quantum block was completed. "
            "Increase --cycle-minutes or use --no-lexical-lattice for a QPU smoke test."
        )
    top_count = max(args.print_candidates, 0)
    top_candidates = [
        item
        for item in (
            compact_console_candidate(candidate)
            for candidate in (latest_cycle.get("best_candidates") or [])[:top_count]
        )
        if item is not None
    ]
    top_semantic = [
        item
        for item in (
            compact_console_candidate(candidate)
            for candidate in (latest_cycle.get("semantic_candidates") or [])[:top_count]
        )
        if item is not None
    ]
    return {
        "target_reached": report.get("target_reached"),
        "quantum_mode": args.quantum_mode,
        "backend": args.backend,
        "blocks_completed": blocks_completed,
        "candidate_count": latest_cycle.get("candidate_count", 0),
        "warning": warning,
        "best_coverage": best.get("coverage", 0.0),
        "best_phrase": phrase_text(best),
        "best_rows": best.get("rows"),
        "best_semantic_score": best_semantic.get("score"),
        "best_semantic_coverage": best_semantic.get("coverage"),
        "best_semantic_phrase": phrase_text(best_semantic),
        "best_semantic_rows": best_semantic.get("rows"),
        "top_candidates": top_candidates,
        "top_semantic_candidates": top_semantic,
        "report": str(Path(args.report).resolve()),
        "memory": str(Path(args.memory).resolve()),
    }


def load_memory(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"runs": []}
    return json.loads(path.read_text(encoding="utf-8"))


def solver_args(args: argparse.Namespace, cycle: int) -> SimpleNamespace:
    return SimpleNamespace(
        dry_run=False,
        run_minutes=args.cycle_minutes,
        memory=str(args.memory),
        quantum_mode=args.quantum_mode,
        backend=args.backend,
        shots=args.shots,
        block_size=args.block_size,
        keep_per_block=args.keep_per_block,
        beam_width=args.beam_width,
        report_candidates_per_block=args.report_candidates_per_block,
        min_score=args.min_score,
        memory_ai=True,
        word_inference=True,
        word_hypotheses_per_window=args.word_hypotheses_per_window,
        word_hypothesis_limit=args.word_hypothesis_limit,
        global_prior_candidates=args.global_prior_candidates,
        segment_prior_candidates=args.segment_prior_candidates,
        prior_letters_per_position=args.prior_letters_per_position,
        orientation_variants=args.orientation_variants,
        diversity_prefix_len=args.diversity_prefix_len,
        diversity_per_prefix=args.diversity_per_prefix,
        initial_segment_seeds=args.initial_segment_seeds,
        branch_skeleton_ambiguities=True,
        skeleton_ambiguity_branches=args.skeleton_ambiguity_branches,
        initial_branch_cap=args.initial_branch_cap,
        memory_frontier_seeds=args.memory_frontier_seeds,
        semantic_prune=args.semantic_prune,
        semantic_prune_min_coverage=args.semantic_prune_min_coverage,
        semantic_prune_min_score=args.semantic_prune_min_score,
        semantic_prune_worst_row=args.semantic_prune_worst_row,
        semantic_prune_max_noise_ratio=args.semantic_prune_max_noise_ratio,
        semantic_prune_min_keep=args.semantic_prune_min_keep,
        semantic_prune_max_keep=args.semantic_prune_max_keep,
        global_refine=args.global_refine,
        global_refine_seeds=args.global_refine_seeds,
        global_refine_width=args.global_refine_width,
        global_refine_letters=args.global_refine_letters,
        global_refine_max_steps=args.global_refine_max_steps,
        global_refine_seconds=args.global_refine_seconds,
        global_refine_report_steps=args.global_refine_report_steps,
        lexical_lattice=args.lexical_lattice,
        lattice_max_word_len=args.lattice_max_word_len,
        lattice_per_span=args.lattice_per_span,
        lattice_min_arc_score=args.lattice_min_arc_score,
        lattice_initial_seeds=args.lattice_initial_seeds,
        lattice_candidates_per_beam=args.lattice_candidates_per_beam,
        lattice_path_width=args.lattice_path_width,
        lattice_arcs_per_start=args.lattice_arcs_per_start,
        report_lattice_arcs=args.report_lattice_arcs,
        protected_initial_prefixes=args.protected_initial_prefixes,
        protected_word_hypotheses=args.protected_word_hypotheses,
        protected_start_hypotheses=None,
        qnn_policy=True,
        qnn_qubits=args.qnn_qubits,
        qnn_layers=args.qnn_layers,
        qnn_train_limit=args.qnn_train_limit,
        report_word_hypotheses=args.report_word_hypotheses,
        allow_skeleton_divergence=False,
        seed=args.seed + cycle,
        real_ibm=False,
        ibm_token_env=args.ibm_token_env,
        ibm_instance=args.ibm_instance,
    )


def run_cycles(args: argparse.Namespace) -> dict[str, Any]:
    args.memory = Path(args.memory)
    args.report = Path(args.report)
    args.memory.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "target_coverage": args.target_coverage,
        "cycles_requested": args.cycles,
        "cycle_minutes": args.cycle_minutes,
        "solver_config": {
            "quantum_mode": args.quantum_mode,
            "backend": args.backend,
            "shots": args.shots,
            "block_size": args.block_size,
            "beam_width": args.beam_width,
            "keep_per_block": args.keep_per_block,
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
            "qnn_qubits": args.qnn_qubits,
            "qnn_layers": args.qnn_layers,
            "qnn_train_limit": args.qnn_train_limit,
        },
        "cycles": [],
        "target_reached": False,
        "best_overall": None,
        "best_semantic_overall": None,
        "latest_solver_diagnostics": None,
    }
    best_overall: dict[str, Any] | None = None
    best_semantic_overall: dict[str, Any] | None = None

    for cycle in range(1, args.cycles + 1):
        before = load_memory(args.memory)
        before_runs = len(before.get("runs", []))
        before_qnn = before.get("active_memory", {}).get("qnn", {})
        run_search(solver_args(args, cycle))
        after = load_memory(args.memory)
        runs = after.get("runs", [])
        run_record = runs[-1] if len(runs) > before_runs else {}
        candidates = best_candidates(run_record, args.report_candidates)
        semantic_candidates = best_candidates(run_record, args.report_candidates, semantic=True)
        after_qnn = after.get("active_memory", {}).get("qnn", {})
        best_cycle = candidates[0] if candidates else None
        best_semantic_cycle = semantic_candidates[0] if semantic_candidates else None
        if best_cycle is not None and (
            best_overall is None
            or (best_cycle["coverage"], best_cycle["score"]) > (best_overall["coverage"], best_overall["score"])
        ):
            best_overall = best_cycle
        if best_semantic_cycle is not None and (
            best_semantic_overall is None
            or (best_semantic_cycle["score"], best_semantic_cycle["coverage"]) > (
                best_semantic_overall["score"],
                best_semantic_overall["coverage"],
            )
        ):
            best_semantic_overall = best_semantic_cycle
        cycle_summary = {
            "cycle": cycle,
            "memory_runs_before": before_runs,
            "memory_runs_after": len(runs),
            "blocks_completed": len(run_record.get("block_stats", [])),
            "candidate_count": len(run_record.get("partial_candidates", [])),
            "best_coverage": best_cycle.get("coverage") if best_cycle else 0.0,
            "best_semantic_score": best_semantic_cycle.get("score") if best_semantic_cycle else None,
            "target_reached_this_cycle": bool(best_cycle and best_cycle.get("coverage", 0.0) >= args.target_coverage),
            "qnn": {
                "updates_before": before_qnn.get("updates"),
                "updates_after": after_qnn.get("updates"),
                "last_loss": after_qnn.get("last_loss"),
                "training": run_record.get("qnn_training"),
            },
            "global_reasoning": run_record.get("global_reasoning"),
            "lexical_lattice": run_record.get("lexical_lattice"),
            "top_word_hypotheses": run_record.get("top_word_hypotheses", []),
            "block_stats": run_record.get("block_stats", []),
            "global_refinement": run_record.get("global_refinement"),
            "best_candidates": candidates,
            "semantic_candidates": semantic_candidates,
        }
        report["cycles"].append(cycle_summary)
        report["latest_solver_diagnostics"] = {
            "global_reasoning": cycle_summary["global_reasoning"],
            "lexical_lattice": cycle_summary["lexical_lattice"],
            "top_word_hypotheses": cycle_summary["top_word_hypotheses"],
            "qnn": cycle_summary["qnn"],
        }
        if best_cycle and best_cycle.get("coverage", 0.0) >= args.target_coverage:
            report["target_reached"] = True
            break

    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    report["best_overall"] = best_overall
    report["best_semantic_overall"] = best_semantic_overall
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--cycle-minutes", type=float, default=0.05)
    parser.add_argument("--target-coverage", type=float, default=0.70)
    parser.add_argument("--memory", default="reports/dorabella_qnn_cycles_memory.json")
    parser.add_argument("--report", default="reports/dorabella_qnn_cycles_report.json")
    parser.add_argument("--report-candidates", type=int, default=8)
    parser.add_argument("--print-candidates", type=int, default=5)
    parser.add_argument("--quantum-mode", choices=["local", "qiskit-aer", "ibm-runtime"], default="local")
    parser.add_argument("--backend", default="aer_simulator")
    parser.add_argument("--shots", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=2)
    parser.add_argument("--keep-per-block", type=int, default=8)
    parser.add_argument("--beam-width", type=int, default=32)
    parser.add_argument("--report-candidates-per-block", type=int, default=16)
    parser.add_argument("--min-score", type=float, default=0.25)
    parser.add_argument("--word-hypotheses-per-window", type=int, default=24)
    parser.add_argument("--word-hypothesis-limit", type=int, default=2400)
    parser.add_argument("--global-prior-candidates", type=int, default=16)
    parser.add_argument("--segment-prior-candidates", type=int, default=40)
    parser.add_argument("--prior-letters-per-position", type=int, default=5)
    parser.add_argument("--orientation-variants", type=int, default=20)
    parser.add_argument("--diversity-prefix-len", type=int, default=12)
    parser.add_argument("--diversity-per-prefix", type=int, default=3)
    parser.add_argument("--initial-segment-seeds", type=int, default=48)
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
    parser.add_argument("--qnn-qubits", type=int, default=4)
    parser.add_argument("--qnn-layers", type=int, default=4)
    parser.add_argument("--qnn-train-limit", type=int, default=8)
    parser.add_argument("--report-word-hypotheses", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1897)
    parser.add_argument("--ibm-token-env", default="IBM_QUANTUM_TOKEN")
    parser.add_argument("--ibm-instance", default=None)
    args = parser.parse_args()
    report = run_cycles(args)
    print(json.dumps(console_summary(report, args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
