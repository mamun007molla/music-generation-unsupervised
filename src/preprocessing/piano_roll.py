"""
Piano-roll helper
-----------------
Converts token sequences into a binary piano-roll (pitch x time) for
visualisation and certain baseline models (e.g. Markov chain matrices).
"""
import numpy as np
from src.config import NUM_PITCHES, NOTE_OFFSET, REST_TOKEN


def tokens_to_pianoroll(tokens: np.ndarray) -> np.ndarray:
    """Return a (NUM_PITCHES, T) binary piano-roll."""
    T = len(tokens)
    roll = np.zeros((NUM_PITCHES, T), dtype=np.uint8)
    for t, tok in enumerate(tokens):
        tok = int(tok)
        if tok >= NOTE_OFFSET:
            roll[tok - NOTE_OFFSET, t] = 1
    return roll


def pianoroll_to_tokens(roll: np.ndarray) -> np.ndarray:
    """Collapse a piano-roll back to the highest-pitch monophonic token stream."""
    T = roll.shape[1]
    tokens = np.full(T, REST_TOKEN, dtype=np.int64)
    for t in range(T):
        active = np.where(roll[:, t] > 0)[0]
        if active.size:
            tokens[t] = active.max() + NOTE_OFFSET
    return tokens
