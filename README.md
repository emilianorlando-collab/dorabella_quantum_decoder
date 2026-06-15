# Dorabella Quantum Decoder

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Qiskit](https://img.shields.io/badge/Qiskit-Quantum-6929C4?logo=qiskit&logoColor=white)
![IBM Quantum](https://img.shields.io/badge/IBM%20Quantum-Runtime-052FAD?logo=ibm&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Numerics-013243?logo=numpy&logoColor=white)
![Status](https://img.shields.io/badge/status-research--prototype-orange)

Quantum-hybrid research toolkit for exploring the Dorabella cipher with strict
symbolic constraints, lexical search, persistent memory, and optional Qiskit /
IBM Quantum Runtime execution.

**Author:** Emiliano Orlando  
**Repository:** `emilianorlando-collab/dorabella_quantum_decoder`

This repository is an experimental cryptanalysis framework. It does not claim a
solution to the Dorabella cipher. Its purpose is to make hypotheses testable,
repeatable, and auditable while keeping the hard cryptographic constraints
separate from semantic ranking and quantum sampling.

## 🧩 What Is the Dorabella Cipher?

The Dorabella cipher is a short encrypted note associated with the English
composer Edward Elgar and addressed to Dora Penny, whom Elgar nicknamed
"Dorabella." The message is famous because it uses a compact alphabet of curved
symbols rather than ordinary letters, and because no universally accepted
solution has been established.

From a computational perspective, the cipher is attractive because it is small
enough to model exactly but ambiguous enough to create a large combinatorial
search space. That makes it a useful case study for hybrid cryptanalysis:
symbolic constraints can rule out impossible mappings, while statistical,
linguistic, neural, and quantum-inspired methods can prioritize the most
plausible remaining branches.

## ⚛️ Why Quantum-Hybrid Search?

The project uses quantum computing as a **structured sampling layer**, not as a
claim that a quantum processor can instantly brute-force the entire cipher.
Current public QPUs are powerful research devices, but the full Dorabella
problem still requires global constraints, semantic scoring, and careful
classical orchestration.

Quantum and quantum-inspired methods are useful here because they can support:

- compact block sampling over many letter assignments;
- future Grover-style oracle experiments for constraint-satisfying states;
- hybrid loops where a classical solver prepares small, meaningful subproblems;
- empirical comparison between local sampling, Qiskit Aer, and IBM Quantum
  Runtime backends.

In this repository, the symbolic engine remains the source of truth: no quantum
sample is accepted unless it satisfies the bijection and orientation rules.

## 🔎 What This Project Does

The Dorabella cipher is modeled as a sequence of oriented symbols arranged in
three rows. The solver treats the transcription as a constrained search problem:

- symbols may have one or more orientation interpretations;
- four substitution alphabets rotate by global position;
- every alphabet is validated as a strict bijection;
- ambiguous symbol readings are propagated consistently;
- candidate plaintexts are scored for English tokenization, grammar, period
  plausibility, and contextual relevance;
- repeated runs update a persistent memory that can guide future searches;
- quantum backends can be used as block samplers inside the larger symbolic
  search.

The repository intentionally ships with a neutral public skeleton. Private
plaintext hypotheses, anchors, notes, and generated reports are excluded from
version control.

## 🧱 Architecture

```text
scripts/
  run_quantum_solver.py      Main solver wrapper
  run_qnn_cycles.py          Repeated memory/QNN cycles with JSON reports
  run_phrase_solver.py       Phrase-first search experiment

solvers/dorabella/
  dorabella_data.py          Public transcription, alphabet cycle, config hook
  dorabella_constraints.py   Symbol parsing, normalization, bijection checks
  dorabella_quantum.py       Local, Qiskit Aer, and IBM Runtime block samplers
  dorabella_qnn.py           Lightweight variational quantum policy components
  dorabella_lexical_lattice.py
                             Whole-cryptogram lexical lattice under bijection
  dorabella_word_inference.py
                             Word-window priors and fragment inference
  dorabella_semantics.py     Tokenization, grammar, and semantic scoring
  dorabella_memory.py        Persistent candidate/rejection memory
  dorabella_memory_ai.py     Neural feedback and replay-buffer learning
```

## 🧠 Core Ideas

### Strict Bijection

The solver never accepts a candidate that maps one symbol to multiple letters
inside the same rotating alphabet, or maps two symbols to the same canonical
letter inside that alphabet. This is the core mathematical rule of the project.

### Rotating Alphabets

Positions are assigned to one of four alphabets by global index. The symbolic
layer validates assignments against this rotation before semantic scoring is
allowed to influence search.

### Ambiguous Orientation

Some symbols admit multiple orientation readings. The solver keeps those
possibilities explicit and propagates decisions through every later occurrence
of the same alphabet/symbol combination.

### Lexical Lattice

The lexical lattice proposes word-level paths across each row and across the
whole cryptogram, but every proposed word path is still filtered through the
same bijection engine. This prevents fluent-looking text from bypassing the
cryptographic constraints.

### Active Memory

Every run can write a JSON memory file containing candidates, rejections,
symbol-letter tendencies, neural replay data, and QNN training summaries. Later
runs can use this memory to avoid repeating low-quality branches and to promote
better-scoring structures.

### Quantum Runtime Integration

The quantum module supports:

- `local`: classical sampler with the same interface;
- `qiskit-aer`: local Qiskit simulator;
- `ibm-runtime`: IBM Quantum Runtime with backend-specific transpilation.

The current IBM integration is a block sampler, not a full semantic Grover
oracle. The symbolic and semantic layers remain responsible for global
consistency.

## 🔐 Private Hypotheses

Private hypotheses should live in a local file that is intentionally ignored by
git:

```text
solvers/dorabella/dorabella_private_data.py
```

Use `solvers/dorabella/dorabella_private_data.example.py` as a template. This
keeps research assumptions out of the public repository while preserving the
reproducible algorithmic framework.

## ⚙️ Installation

Python 3.10+ is recommended for current Qiskit Runtime support.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

For IBM Quantum Runtime:

```bash
pip install -U qiskit-ibm-runtime
```

## 🚀 Usage

Inspect the model without running a search:

```bash
python scripts/run_quantum_solver.py --dry-run
```

Run a local bounded search:

```bash
python scripts/run_qnn_cycles.py \
  --cycles 1 \
  --cycle-minutes 5 \
  --quantum-mode local \
  --memory reports/local_memory.json \
  --report reports/local_report.json
```

Run with Qiskit Aer:

```bash
python scripts/run_qnn_cycles.py \
  --cycles 1 \
  --cycle-minutes 5 \
  --quantum-mode qiskit-aer \
  --shots 128 \
  --memory reports/aer_memory.json \
  --report reports/aer_report.json
```

Run a small IBM Quantum Runtime smoke test:

```bash
export IBM_QUANTUM_TOKEN="..."
export IBM_QUANTUM_INSTANCE="..."

python scripts/run_qnn_cycles.py \
  --cycles 1 \
  --cycle-minutes 1 \
  --quantum-mode ibm-runtime \
  --backend ibm_fez \
  --ibm-instance "$IBM_QUANTUM_INSTANCE" \
  --shots 8 \
  --block-size 1 \
  --no-lexical-lattice \
  --global-refine-seconds 0 \
  --memory reports/ibm_smoke_memory.json \
  --report reports/ibm_smoke_report.json
```

## 📊 Reports

Reports are generated as JSON files and are ignored by git. Console output
includes a compact summary with:

- best technical candidate;
- best semantic candidate;
- candidate coverage;
- bijection consistency;
- tokenized rows;
- report and memory paths.

## 🛡️ Security

Do not commit API keys, notebook outputs, private notes, generated reports, or
private hypothesis files. The included `.gitignore` excludes these by default.

## 🧪 Status

This is a research codebase. It is designed to support systematic experiments,
not to assert a final plaintext. Contributions should preserve the separation
between:

1. hard cryptographic constraints;
2. private hypotheses;
3. semantic ranking;
4. quantum sampling;
5. generated experimental evidence.
