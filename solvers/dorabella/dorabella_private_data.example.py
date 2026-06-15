"""Example private hypothesis configuration.

Copy this file to ``dorabella_private_data.py`` and fill in your own research
assumptions locally. The real private file is ignored by git.
"""

# Optional plaintext masks. Use "_" for unknown positions and "A/B" for
# ambiguous plaintext choices. Keep each row length aligned with ROW_SYMBOLS.
SKELETONS = [
    "_" * 29,
    "_" * 31,
    "_" * 27,
]

# Optional anchors used by semantic scoring and protected beam prefixes.
START_ANCHOR = None
END_ANCHOR = None

# Optional overrides. Leave undefined unless your private experiment needs a
# different transcription or alphabet-1 map.
# ROW_SYMBOLS = [...]
# ALPHABET1_MAP = {...}
