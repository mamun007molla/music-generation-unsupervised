"""
Helpers for building training tensors from the parsed MIDI corpus.
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Tuple

from src.config import RAW_MIDI_DIR, SEQ_LEN, GENRES, PAD_TOKEN
from src.preprocessing.midi_parser import midi_to_token_seq
from src.preprocessing.tokenizer   import slice_sequences


def load_corpus(root: Path = RAW_MIDI_DIR) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (windows (N,T), genre_idx (N,))."""
    all_wins, all_gen = [], []
    for g_idx, genre in enumerate(GENRES):
        g_dir = root / genre
        if not g_dir.exists():
            continue
        for mid in sorted(list(g_dir.glob("*.mid")) + list(g_dir.glob("*.midi"))):
            try:
                toks = midi_to_token_seq(mid)
                wins = slice_sequences(toks, SEQ_LEN)
                all_wins.append(wins)
                all_gen.extend([g_idx] * len(wins))
            except Exception as e:
                print(f"[warn] skipping {mid.name}: {e}")
    if not all_wins:
        raise RuntimeError(f"No MIDI data found under {root}")
    wins = np.concatenate(all_wins, axis=0).astype(np.int64)
    gens = np.asarray(all_gen,        dtype=np.int64)
    return wins, gens


def iterate_batches(X: np.ndarray, G: np.ndarray, batch_size: int,
                    rng: np.random.Generator):
    idx = rng.permutation(len(X))
    for s in range(0, len(X), batch_size):
        b = idx[s:s + batch_size]
        yield X[b], G[b]
