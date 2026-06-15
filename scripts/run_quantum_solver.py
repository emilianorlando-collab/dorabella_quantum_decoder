#!/usr/bin/env python3
"""Run the quantum-hybrid Dorabella solver from the package layout."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "solvers"))

from dorabella.dorabella_solver import main


if __name__ == "__main__":
    main()

