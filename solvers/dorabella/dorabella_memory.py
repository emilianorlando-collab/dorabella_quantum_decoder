"""Persistent memory for short daily quantum runs."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .dorabella_constraints import normalize_text


def signature(rows: list[str]) -> str:
    raw = "|".join(normalize_text(row) for row in rows)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def load_memory(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "runs": [],
            "rejected_signatures": {},
            "best_candidates": [],
            "learned_penalties": {},
            "active_memory": {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "position_letter_scores": {},
                "symbol_letter_scores": {},
                "fragment_hypotheses": [],
                "dead_blocks": [],
                "neural": {
                    "kind": "mlp",
                    "features": [
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
                        "elgar_anchor",
                        "bella_anchor",
                    ],
                    "layers": [15, 18, 10, 1],
                    "params": {},
                    "learning_rate": 0.025,
                    "updates": 0,
                    "replay_buffer": [],
                    "replay_limit": 500,
                },
                "llm_reflections": [],
                "last_llm_prompt": "",
            },
        }
    return json.loads(path.read_text(encoding="utf-8"))


def save_memory(path: Path, memory: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(memory, ensure_ascii=False, indent=2), encoding="utf-8")


def remember_rejection(memory: dict[str, Any], rows: list[str], reason: str, score: float) -> None:
    memory["rejected_signatures"][signature(rows)] = {
        "reason": reason,
        "score": score,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def remember_candidate(memory: dict[str, Any], rows: list[str], score: float, notes: str) -> None:
    record = {
        "signature": signature(rows),
        "rows": rows,
        "score": score,
        "notes": notes,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    existing = [item for item in memory["best_candidates"] if item["signature"] != record["signature"]]
    existing.append(record)
    existing.sort(key=lambda item: item["score"], reverse=True)
    memory["best_candidates"] = existing[:100]
