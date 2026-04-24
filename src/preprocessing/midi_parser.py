"""
MIDI parser
-----------
Reads a `.mid` file with `pretty_midi` and converts it into the internal token
representation that all subsequent models consume.

Representation (monophonic, simplified):
    at every 16-th note step we pick the highest-velocity pitch that is
    sounding.  This yields a 1-D integer sequence of length `n_steps`.
    A value of REST_TOKEN (=1) means "no note at this step".
"""
from __future__ import annotations

import numpy as np
import pretty_midi
from pathlib import Path
from typing import List

from src.config import (PITCH_MIN, PITCH_MAX, STEPS_PER_BAR,
                        REST_TOKEN, NOTE_OFFSET)


def pitch_to_token(pitch: int) -> int:
    """Map raw MIDI pitch -> vocabulary token."""
    if pitch < PITCH_MIN or pitch > PITCH_MAX:
        return REST_TOKEN
    return pitch - PITCH_MIN + NOTE_OFFSET


def token_to_pitch(tok: int) -> int | None:
    """Reverse mapping, or None if the token is a special symbol."""
    if tok < NOTE_OFFSET:
        return None
    return tok - NOTE_OFFSET + PITCH_MIN


def midi_to_token_seq(midi_path: str | Path,
                      steps_per_bar: int = STEPS_PER_BAR,
                      bpm_default: float = 120.0) -> np.ndarray:
    """Parse `midi_path` into a 1-D numpy array of vocabulary tokens."""
    pm = pretty_midi.PrettyMIDI(str(midi_path))

    try:
        tempo = pm.get_tempo_changes()[1][0]
    except IndexError:
        tempo = bpm_default

    sec_per_beat = 60.0 / tempo
    step_sec     = sec_per_beat / (steps_per_bar / 4)   # 16-th of a beat

    end_sec  = pm.get_end_time()
    n_steps  = max(1, int(np.ceil(end_sec / step_sec)))
    tokens   = np.full(n_steps, REST_TOKEN, dtype=np.int64)

    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            start = int(note.start / step_sec)
            end   = min(n_steps, int(np.ceil(note.end / step_sec)))
            if start >= n_steps:
                continue
            tok = pitch_to_token(note.pitch)
            # keep the highest pitch at each step (simple monophonic pick)
            for t in range(start, end):
                if tokens[t] == REST_TOKEN or tok > tokens[t]:
                    tokens[t] = tok
    return tokens


def token_seq_to_midi(tokens: np.ndarray,
                      steps_per_bar: int = STEPS_PER_BAR,
                      tempo: float = 120.0,
                      program: int = 0) -> pretty_midi.PrettyMIDI:
    """Inverse: convert a token array back into a playable PrettyMIDI object."""
    pm   = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    inst = pretty_midi.Instrument(program=program)

    sec_per_beat = 60.0 / tempo
    step_sec     = sec_per_beat / (steps_per_bar / 4)

    t = 0
    while t < len(tokens):
        tok = int(tokens[t])
        pitch = token_to_pitch(tok)
        if pitch is None:
            t += 1
            continue
        # extend note while the same pitch token repeats
        j = t + 1
        while j < len(tokens) and int(tokens[j]) == tok:
            j += 1
        start_time = t   * step_sec
        end_time   = j   * step_sec
        inst.notes.append(pretty_midi.Note(
            velocity=90, pitch=pitch,
            start=start_time, end=end_time))
        t = j

    pm.instruments.append(inst)
    return pm


def parse_folder(folder: str | Path) -> List[np.ndarray]:
    """Parse every `.mid`/`.midi` file under `folder`; return list of token seqs."""
    folder = Path(folder)
    seqs   = []
    for p in sorted(list(folder.glob("*.mid")) + list(folder.glob("*.midi"))):
        try:
            seqs.append(midi_to_token_seq(p))
        except Exception as e:        # pragma: no cover - robustness only
            print(f"[warn] skipping {p.name}: {e}")
    return seqs
