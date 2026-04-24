"""
Tokeniser utilities
-------------------
Tiny helpers for turning long token streams into fixed-length training windows
and for one-hot / dense conversions used by the neural models.
"""
from __future__ import annotations
import numpy as np
from typing import List, Tuple

from src.config import SEQ_LEN, VOCAB_SIZE, PAD_TOKEN


def slice_sequences(tokens: np.ndarray,
                    seq_len: int = SEQ_LEN,
                    hop: int | None = None) -> np.ndarray:
    """Slide a fixed-length window across a token stream.

    Returns a 2-D array of shape `(n_windows, seq_len)`.
    The last (possibly incomplete) window is zero-padded with PAD_TOKEN.
    """
    hop = hop or seq_len // 2
    out = []
    if len(tokens) < seq_len:
        padded = np.full(seq_len, PAD_TOKEN, dtype=tokens.dtype)
        padded[:len(tokens)] = tokens
        return padded[None, :]

    for s in range(0, len(tokens) - seq_len + 1, hop):
        out.append(tokens[s:s + seq_len])
    return np.stack(out, axis=0)


def one_hot(seq: np.ndarray, vocab: int = VOCAB_SIZE) -> np.ndarray:
    """seq: (..., T) int -> (..., T, vocab) float."""
    out = np.zeros(seq.shape + (vocab,), dtype=np.float32)
    it = np.nditer(seq, flags=["multi_index"])
    for x in it:
        out[it.multi_index + (int(x),)] = 1.0
    return out


def train_test_split(windows: np.ndarray,
                     test_ratio: float = 0.2,
                     seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(windows))
    k   = int(len(windows) * (1 - test_ratio))
    return windows[idx[:k]], windows[idx[k:]]
