"""
Convert a generated token sequence to a playable .mid file on disk.
"""
from __future__ import annotations
import numpy as np
from pathlib import Path

from src.preprocessing.midi_parser import token_seq_to_midi
from src.config import GENRES

# General-MIDI programs that sound approximately "genre-appropriate"
GENRE_PROGRAMS = {"classical": 0, "jazz": 32, "rock": 29,
                  "pop": 80,       "electronic": 81}


def save_tokens_as_midi(tokens: np.ndarray,
                        out_path: str | Path,
                        genre: str | None = None,
                        tempo: float = 120.0) -> Path:
    program = GENRE_PROGRAMS.get(genre, 0)
    pm      = token_seq_to_midi(tokens, tempo=tempo, program=program)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(out_path))
    return out_path
