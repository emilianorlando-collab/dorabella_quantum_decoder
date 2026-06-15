"""Hard symbolic constraints for the Dorabella quantum-hybrid search."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .dorabella_data import (
    ALPHABET1_MAP,
    CANONICAL_LETTERS,
    DorabellaPosition,
    canonicalize_letter,
    canonicalize_text,
)


def normalize_letter(ch: str) -> str:
    return canonicalize_letter(ch)


def normalize_text(text: str) -> str:
    return canonicalize_text(text)


def normalize_symbol(sym: str) -> str:
    return sym.strip().upper().replace("NO", "NW").replace("O", "W")


def parse_symbol_token(token: str) -> tuple[str, ...]:
    return tuple(normalize_symbol(part) for part in token.split("/"))


def parse_skeleton_line(line: str) -> list[frozenset[str] | None]:
    """Parse a line where '_' is unknown and slash-separated letters are alternatives."""
    s = line.replace(" ", "").upper()
    out: list[frozenset[str] | None] = []
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == "_":
            out.append(None)
            i += 1
            continue
        if ch.isalpha():
            opts = {normalize_letter(ch)}
            j = i + 1
            while j + 1 < len(s) and s[j] == "/" and s[j + 1].isalpha():
                opts.add(normalize_letter(s[j + 1]))
                j += 2
            out.append(frozenset(opts))
            i = j
            continue
        i += 1
    return out


def build_positions(
    row_symbols: Sequence[Sequence[str]], skeletons: Sequence[str]
) -> list[DorabellaPosition]:
    positions: list[DorabellaPosition] = []
    masks = [parse_skeleton_line(line) for line in skeletons]
    global_index = 0
    for row_idx, (symbols, mask) in enumerate(zip(row_symbols, masks), start=1):
        if len(symbols) != len(mask):
            raise ValueError(f"row {row_idx} has {len(symbols)} symbols but {len(mask)} mask slots")
        for col_idx, (token, allowed) in enumerate(zip(symbols, mask), start=1):
            alpha = global_index % 4
            positions.append(
                DorabellaPosition(
                    row=row_idx,
                    col=col_idx,
                    global_index=global_index,
                    alphabet_index=alpha,
                    symbol_options=parse_symbol_token(token),
                    allowed_letters=allowed,
                )
            )
            global_index += 1
    return positions


@dataclass
class CipherState:
    """Partial assignment with strict per-alphabet bijection.

    `maps[a][symbol] = letter` and `inverse[a][letter] = symbol` must both hold.
    Ambiguous orientations are represented by trying one symbol option at a time.
    """

    maps: list[dict[str, str]] = field(default_factory=lambda: [dict() for _ in range(4)])
    inverse: list[dict[str, str]] = field(default_factory=lambda: [dict() for _ in range(4)])

    def clone(self) -> "CipherState":
        return CipherState(
            maps=[m.copy() for m in self.maps],
            inverse=[m.copy() for m in self.inverse],
        )

    def bind(self, alphabet_index: int, symbol: str, letter: str) -> bool:
        letter = normalize_letter(letter)
        if letter not in CANONICAL_LETTERS:
            return False
        forward = self.maps[alphabet_index].get(symbol)
        reverse = self.inverse[alphabet_index].get(letter)
        if forward is not None and forward != letter:
            return False
        if reverse is not None and reverse != symbol:
            return False
        self.maps[alphabet_index][symbol] = letter
        self.inverse[alphabet_index][letter] = symbol
        return True


@dataclass(frozen=True)
class ValidationError:
    kind: str
    row: int
    col: int
    detail: str


def seed_alphabet1_constraints(positions: Iterable[DorabellaPosition], state: CipherState) -> bool:
    """Apply Elgar's documented alphabet to cycle positions 1,5,9,..."""
    for symbol, letter in ALPHABET1_MAP.items():
        if not state.bind(0, symbol, letter):
            return False
    for pos in positions:
        if pos.alphabet_index != 0:
            continue
        possible_letters = {
            ALPHABET1_MAP[sym] for sym in pos.symbol_options if sym in ALPHABET1_MAP
        }
        if not possible_letters:
            continue
        if pos.allowed_letters is not None:
            possible_letters &= set(pos.allowed_letters)
        if not possible_letters:
            return False
    return True


def seed_fixed_skeleton_constraints(positions: Iterable[DorabellaPosition], state: CipherState) -> bool:
    """Apply every non-ambiguous skeleton letter to the current cipher state."""
    for pos in positions:
        if pos.allowed_letters is None or len(pos.allowed_letters) != 1:
            continue
        letter = next(iter(pos.allowed_letters))
        placed = False
        for sym in pos.symbol_options:
            trial = state.clone()
            if trial.bind(pos.alphabet_index, sym, letter):
                state.maps = trial.maps
                state.inverse = trial.inverse
                placed = True
                break
        if not placed:
            return False
    return True


def validate_plaintext(
    rows: Sequence[str], positions: Sequence[DorabellaPosition]
) -> tuple[bool, list[ValidationError], CipherState]:
    text = "".join(normalize_text(row) for row in rows)
    errors: list[ValidationError] = []
    state = CipherState()
    if len(text) != len(positions):
        return False, [ValidationError("length", 0, 0, f"{len(text)} != {len(positions)}")], state

    seed_alphabet1_constraints(positions, state)
    seed_fixed_skeleton_constraints(positions, state)
    for ch, pos in zip(text, positions):
        if pos.allowed_letters is not None and ch not in pos.allowed_letters:
            errors.append(ValidationError("skeleton", pos.row, pos.col, f"{ch} not in {sorted(pos.allowed_letters)}"))
            continue
        bound = False
        for sym in pos.symbol_options:
            trial = state.clone()
            if trial.bind(pos.alphabet_index, sym, ch):
                state = trial
                bound = True
                break
        if not bound:
            errors.append(ValidationError("bijection", pos.row, pos.col, f"no symbol option can map to {ch}"))
    return len(errors) == 0, errors, state


def row_regex(mask: str) -> re.Pattern[str]:
    pieces: list[str] = []
    for slot in parse_skeleton_line(mask):
        if slot is None:
            pieces.append(f"[{CANONICAL_LETTERS}]")
        elif len(slot) == 1:
            pieces.append(next(iter(slot)))
        else:
            pieces.append("[" + "".join(sorted(slot)) + "]")
    return re.compile("^" + "".join(pieces) + "$")
