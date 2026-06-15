"""Active memory and lightweight neural feedback for Dorabella.

The goal is to turn daily runs into a search policy:
- reward position-letter assignments that appear in better candidates;
- penalize assignments that repeatedly produce noisy tokenizations;
- keep fragment hypotheses;
- expose an LLM reflection prompt that can be sent to an external model later.

This module is dependency-free. The neural component is a persistent MLP with
two hidden layers, online backpropagation, and a small replay buffer. It is not
meant to replace a large language model; it learns a search policy from the
solver's own runs.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .dorabella_constraints import normalize_text
from .dorabella_data import CANONICAL_LETTERS, END_ANCHOR, START_ANCHOR, DorabellaPosition
from .dorabella_language_model import score_partial_or_full_row


FEATURES = (
    "bias",
    "resolved_ratio",
    "token_score",
    "best_row_score",
    "worst_row_score",
    "unknown_penalty",
    "contraction_rate",
    "abbrev_rate",
    "single_letter_penalty",
    "avg_token_len",
    "long_word_rate",
    "row_balance",
    "complete_candidate",
    "semantic_score",
    "closure_consistency",
    "repeat_binding_density",
    "map_fill_ratio",
    "end_anchor",
    "start_anchor",
)

MLP_LAYERS = (len(FEATURES), 18, 10, 1)


@dataclass(frozen=True)
class MemorySignal:
    score: float
    features: dict[str, float]
    notes: str


def ensure_ai_memory(memory: dict[str, Any]) -> dict[str, Any]:
    ai = memory.setdefault(
        "active_memory",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "position_letter_scores": {},
            "symbol_letter_scores": {},
            "fragment_hypotheses": [],
            "dead_blocks": [],
            "neural": {
                "kind": "mlp",
                "features": list(FEATURES),
                "layers": list(MLP_LAYERS),
                "params": init_mlp_params(),
                "learning_rate": 0.025,
                "updates": 0,
                "replay_buffer": [],
                "replay_limit": 500,
            },
            "llm_reflections": [],
            "last_llm_prompt": "",
        },
    )
    neural = ai.setdefault("neural", {})
    if neural.get("kind") != "mlp" or "params" not in neural:
        neural.clear()
        neural.update(
            {
                "kind": "mlp",
                "features": list(FEATURES),
                "layers": list(MLP_LAYERS),
                "params": init_mlp_params(),
                "learning_rate": 0.025,
                "updates": 0,
                "replay_buffer": [],
                "replay_limit": 500,
            }
        )
    neural.setdefault("features", list(FEATURES))
    if tuple(neural.get("features", [])) != FEATURES:
        neural["features"] = list(FEATURES)
        neural["layers"] = list(MLP_LAYERS)
        neural["params"] = init_mlp_params()
        neural["updates"] = 0
        neural["replay_buffer"] = []
    neural.setdefault("layers", list(MLP_LAYERS))
    if not isinstance(neural.get("params"), dict) or "weights" not in neural.get("params", {}) or "biases" not in neural.get("params", {}):
        neural["params"] = init_mlp_params()
    neural.setdefault("learning_rate", 0.025)
    neural.setdefault("updates", 0)
    neural.setdefault("replay_buffer", [])
    neural.setdefault("replay_limit", 500)
    ai.setdefault("position_letter_scores", {})
    ai.setdefault("symbol_letter_scores", {})
    ai.setdefault("fragment_hypotheses", [])
    ai.setdefault("dead_blocks", [])
    ai.setdefault("llm_reflections", [])
    ai.setdefault("last_llm_prompt", "")
    ai.setdefault(
        "qnn",
        {
            "enabled": False,
            "qubits": 4,
            "layers": 4,
            "params": [],
            "updates": 0,
            "learning_rate": 0.015,
            "train_limit": 12,
            "last_loss": None,
            "training_history": [],
        },
    )
    return ai


def sigmoid(x: float) -> float:
    if x < -40:
        return 0.0
    if x > 40:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def init_mlp_params() -> dict[str, Any]:
    rng = random.Random(1897)
    params: dict[str, Any] = {"weights": [], "biases": []}
    for fan_in, fan_out in zip(MLP_LAYERS[:-1], MLP_LAYERS[1:]):
        scale = math.sqrt(2.0 / max(fan_in + fan_out, 1))
        weights = [[rng.uniform(-scale, scale) for _ in range(fan_in)] for _ in range(fan_out)]
        biases = [0.0 for _ in range(fan_out)]
        params["weights"].append(weights)
        params["biases"].append(biases)
    return params


def feature_vector(features: dict[str, float]) -> list[float]:
    return [float(features.get(name, 0.0)) for name in FEATURES]


def tanh(x: float) -> float:
    if x < -20:
        return -1.0
    if x > 20:
        return 1.0
    return math.tanh(x)


def mlp_forward(params: dict[str, Any], x: list[float]) -> tuple[float, list[list[float]], list[list[float]]]:
    activations = [x]
    preacts: list[list[float]] = []
    current = x
    weights = params["weights"]
    biases = params["biases"]
    for layer_idx, (w_layer, b_layer) in enumerate(zip(weights, biases)):
        z = []
        a = []
        is_output = layer_idx == len(weights) - 1
        for row, bias in zip(w_layer, b_layer):
            raw = sum(weight * val for weight, val in zip(row, current)) + bias
            z.append(raw)
            a.append(sigmoid(raw) if is_output else tanh(raw))
        preacts.append(z)
        activations.append(a)
        current = a
    return activations[-1][0], activations, preacts


def mlp_train(params: dict[str, Any], x: list[float], target: float, lr: float) -> float:
    pred, activations, preacts = mlp_forward(params, x)
    weights = params["weights"]
    biases = params["biases"]
    deltas: list[list[float]] = [[] for _ in weights]
    # Binary cross-entropy derivative through sigmoid output.
    deltas[-1] = [pred - target]
    for layer_idx in range(len(weights) - 2, -1, -1):
        next_delta = deltas[layer_idx + 1]
        next_weights = weights[layer_idx + 1]
        layer_delta = []
        for neuron_idx, z in enumerate(preacts[layer_idx]):
            downstream = sum(next_weights[k][neuron_idx] * next_delta[k] for k in range(len(next_delta)))
            deriv = 1.0 - tanh(z) ** 2
            layer_delta.append(downstream * deriv)
        deltas[layer_idx] = layer_delta

    for layer_idx, delta_layer in enumerate(deltas):
        prev_a = activations[layer_idx]
        for out_idx, delta in enumerate(delta_layer):
            clipped_delta = max(-2.0, min(2.0, delta))
            for in_idx, prev in enumerate(prev_a):
                weights[layer_idx][out_idx][in_idx] -= lr * clipped_delta * prev
            biases[layer_idx][out_idx] -= lr * clipped_delta
    return pred


def candidate_cipher_features(candidate: dict[str, Any]) -> dict[str, float]:
    raw_maps = candidate.get("bijection_maps") or []
    map_entries = 0
    if isinstance(raw_maps, list):
        for mapping in raw_maps:
            if isinstance(mapping, dict):
                map_entries += len(mapping)
    map_fill_ratio = map_entries / max(4 * len(CANONICAL_LETTERS), 1)

    closure = candidate.get("bijection_closure") or []
    closure_count = 0
    inconsistent = 0
    repeated = 0
    rendered_occurrences = 0
    if isinstance(closure, list):
        for item in closure:
            if not isinstance(item, dict):
                continue
            closure_count += 1
            if not item.get("exact_all_consistent", True):
                inconsistent += 1
            exact = item.get("exact_occurrences") or []
            possible = item.get("possible_ambiguous_occurrences") or []
            occurrence_count = len(exact) + len(possible)
            if occurrence_count >= 2:
                repeated += 1
            for occurrence in list(exact) + list(possible):
                if isinstance(occurrence, dict) and occurrence.get("rendered_letter") is not None:
                    rendered_occurrences += 1
    closure_consistency = 1.0 - inconsistent / max(closure_count, 1)
    repeat_binding_density = repeated / max(closure_count, 1)
    rendered_density = min(1.0, rendered_occurrences / max(closure_count * 2, 1))
    return {
        "closure_consistency": closure_consistency,
        "repeat_binding_density": repeat_binding_density * 0.75 + rendered_density * 0.25,
        "map_fill_ratio": min(1.0, map_fill_ratio),
    }


def candidate_features(candidate: dict[str, Any]) -> MemorySignal:
    rows = candidate.get("rows", [])
    total_unknowns = float(candidate.get("total_unknowns") or 0)
    resolved = float(candidate.get("resolved_unknowns") or 0)
    resolved_ratio = resolved / total_unknowns if total_unknowns else 0.0

    token_scores = []
    row_scores = []
    unknown_count = 0
    contraction_count = 0
    abbrev_count = 0
    single_letters = 0
    token_count = 0
    token_len_total = 0
    long_words = 0
    row_notes = []
    for row in rows:
        tok = score_partial_or_full_row(row)
        token_scores.append(tok.score)
        row_scores.append(tok.score)
        row_notes.append(tok.notes)
        for token in tok.tokens:
            token_count += 1
            token_len_total += len(token.text)
            if len(token.text) >= 5 and token.kind != "unknown":
                long_words += 1
            if token.kind == "unknown":
                unknown_count += 1
            if token.kind == "contraction":
                contraction_count += 1
            if token.kind == "abbrev":
                abbrev_count += 1
            if len(token.text) == 1 and token.text not in {"A", "I"}:
                single_letters += 1

    token_score = sum(token_scores) / max(len(token_scores), 1)
    best_row_score = max(row_scores) if row_scores else -1.0
    worst_row_score = min(row_scores) if row_scores else -1.0
    avg_token_len = token_len_total / max(token_count, 1)
    row_balance = 1.0 - min(1.0, (best_row_score - worst_row_score) / 6.0)
    joined = "".join(rows).upper()
    complete_candidate = 1.0 if total_unknowns and resolved >= total_unknowns else 0.0
    features = {
        "bias": 1.0,
        "resolved_ratio": resolved_ratio,
        "token_score": token_score,
        "best_row_score": best_row_score,
        "worst_row_score": worst_row_score,
        "unknown_penalty": -unknown_count / max(token_count, 1),
        "contraction_rate": contraction_count / max(token_count, 1),
        "abbrev_rate": abbrev_count / max(token_count, 1),
        "single_letter_penalty": -single_letters / max(token_count, 1),
        "avg_token_len": min(avg_token_len / 8.0, 1.0),
        "long_word_rate": long_words / max(token_count, 1),
        "row_balance": row_balance,
        "complete_candidate": complete_candidate,
        "semantic_score": tanh(float(candidate.get("score") or 0.0) / 4.0),
        "end_anchor": 1.0 if END_ANCHOR and joined.endswith(END_ANCHOR) else 0.0,
        "start_anchor": 1.0 if START_ANCHOR and joined.startswith(START_ANCHOR) else 0.0,
    }
    features.update(candidate_cipher_features(candidate))
    score = (
        0.35 * resolved_ratio
        + 0.45 * token_score
        + 0.10 * best_row_score
        + 0.08 * worst_row_score
        + 0.12 * features["contraction_rate"]
        + 0.04 * features["abbrev_rate"]
        + 0.10 * features["avg_token_len"]
        + 0.08 * features["long_word_rate"]
        + 0.08 * features["row_balance"]
        + 0.08 * features["complete_candidate"]
        + 0.10 * features["semantic_score"]
        + 0.08 * features["closure_consistency"]
        + 0.06 * features["repeat_binding_density"]
        + 0.03 * features["map_fill_ratio"]
        + 0.30 * features["unknown_penalty"]
        + 0.25 * features["single_letter_penalty"]
    )
    return MemorySignal(score=score, features=features, notes=" | ".join(row_notes))


def neural_predict(memory: dict[str, Any], features: dict[str, float]) -> float:
    ai = ensure_ai_memory(memory)
    params = ai["neural"]["params"]
    pred, _activations, _preacts = mlp_forward(params, feature_vector(features))
    return pred


def neural_update(memory: dict[str, Any], features: dict[str, float], target: float) -> float:
    ai = ensure_ai_memory(memory)
    neural = ai["neural"]
    pred = neural_predict(memory, features)
    lr = float(neural.get("learning_rate", 0.025))
    mlp_train(neural["params"], feature_vector(features), target, lr)
    neural["updates"] = int(neural.get("updates", 0)) + 1
    return pred


def replay_update(memory: dict[str, Any], features: dict[str, float], target: float) -> None:
    ai = ensure_ai_memory(memory)
    neural = ai["neural"]
    buffer = neural["replay_buffer"]
    buffer.append({"features": {name: float(features.get(name, 0.0)) for name in FEATURES}, "target": float(target)})
    limit = int(neural.get("replay_limit", 500))
    if len(buffer) > limit:
        del buffer[: len(buffer) - limit]
    # Deterministic mini-replay: recent examples plus a few spread through memory.
    if len(buffer) < 8:
        return
    step = max(1, len(buffer) // 8)
    replay = buffer[-4:] + buffer[::step][:8]
    for item in replay:
        neural_update(memory, item["features"], item["target"])


def rows_to_position_letters(rows: list[str], positions: list[DorabellaPosition]) -> dict[int, str]:
    text = normalize_text("".join(rows))
    out: dict[int, str] = {}
    for pos, ch in zip(positions, text):
        if ch and ch not in {"_", "?"}:
            out[pos.global_index] = ch
    return out


def bounded_add(scores: dict[str, float], key: str, delta: float) -> None:
    scores[key] = max(-5.0, min(5.0, float(scores.get(key, 0.0)) + delta))


def learn_from_candidate(memory: dict[str, Any], positions: list[DorabellaPosition], candidate: dict[str, Any]) -> None:
    ai = ensure_ai_memory(memory)
    signal = candidate_features(candidate)
    # Convert arbitrary linguistic score into a bounded learning target.
    target = sigmoid(signal.score)
    neural_update(memory, signal.features, target)
    replay_update(memory, signal.features, target)

    rows = candidate.get("rows", [])
    pos_letters = rows_to_position_letters(rows, positions)
    delta = max(-0.12, min(0.12, signal.score * 0.04))
    if candidate.get("resolved_unknowns") == candidate.get("total_unknowns"):
        delta *= 1.35
    for global_index, letter in pos_letters.items():
        pos = positions[global_index]
        bounded_add(ai["position_letter_scores"], f"{global_index + 1}:{letter}", delta)
        for sym in pos.symbol_options:
            bounded_add(ai["symbol_letter_scores"], f"A{pos.alphabet_index + 1}:{sym}:{letter}", delta / max(len(pos.symbol_options), 1))

    if signal.score > -0.35:
        fragment = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "score": signal.score,
            "resolved_unknowns": candidate.get("resolved_unknowns", 0),
            "rows": rows,
            "tokenization": signal.notes,
        }
        ai["fragment_hypotheses"].append(fragment)
        ai["fragment_hypotheses"] = sorted(
            ai["fragment_hypotheses"], key=lambda item: item["score"], reverse=True
        )[:100]


def learn_from_run(memory: dict[str, Any], positions: list[DorabellaPosition], run_record: dict[str, Any]) -> None:
    ai = ensure_ai_memory(memory)
    for candidate in run_record.get("partial_candidates", []):
        learn_from_candidate(memory, positions, candidate)

    for stat in run_record.get("block_stats", []):
        if stat.get("beam_in", 0) and stat.get("beam_out", 0) == 0:
            ai["dead_blocks"].append(
                {
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "block": stat.get("block"),
                    "block_positions": stat.get("block_positions", []),
                    "local_samples": stat.get("local_samples", 0),
                    "hypothesis": run_record.get("hypothesis"),
                }
            )
            ai["dead_blocks"] = ai["dead_blocks"][-100:]

    ai["last_llm_prompt"] = build_llm_memory_prompt(memory)
    try:
        from .dorabella_qnn import qnn_train_from_run

        qnn_train_from_run(memory, run_record)
    except Exception as exc:
        ai.setdefault("qnn", {})["last_error"] = str(exc)


def assignment_memory_bonus(
    memory: dict[str, Any],
    positions_by_index: dict[int, DorabellaPosition],
    assignment: dict[int, str],
) -> float:
    ai = ensure_ai_memory(memory)
    bonus = 0.0
    for global_index, letter in assignment.items():
        pos = positions_by_index.get(global_index)
        if pos is None:
            continue
        bonus += float(ai["position_letter_scores"].get(f"{global_index + 1}:{letter}", 0.0))
        for sym in pos.symbol_options:
            bonus += float(ai["symbol_letter_scores"].get(f"A{pos.alphabet_index + 1}:{sym}:{letter}", 0.0)) / max(len(pos.symbol_options), 1)
    return bonus / max(len(assignment), 1)


def candidate_neural_bonus(memory: dict[str, Any], candidate: dict[str, Any]) -> float:
    signal = candidate_features(candidate)
    pred = neural_predict(memory, signal.features)
    # Center around zero so an untrained net has little impact.
    return (pred - 0.5) * 2.0


def build_llm_memory_prompt(memory: dict[str, Any]) -> str:
    ai = ensure_ai_memory(memory)
    fragments = ai.get("fragment_hypotheses", [])[:12]
    dead_blocks = ai.get("dead_blocks", [])[-8:]
    neural = ai.get("neural", {})
    return (
        "You are a Dorabella cipher research agent. Review the active memory and "
        "suggest search-policy updates without breaking the 4-alphabet bijection.\n\n"
        f"Top fragment hypotheses:\n{fragments}\n\n"
        f"Recent dead blocks:\n{dead_blocks}\n\n"
        f"Neural architecture:\n{neural.get('layers')}\n\n"
        f"Neural updates:\n{neural.get('updates')}\n\n"
        "Return JSON with: likely_fragments, bad_branches, next_constraints, "
        "and one concise hypothesis for the plaintext."
    )
