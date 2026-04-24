"""
Baseline generators
-------------------
Two simple, non-neural generators used for comparison:

1. Random-note generator : samples every step i.i.d. from the empirical
                           unigram distribution of the training corpus.
2. Markov chain          : first-order Markov model P(x_t | x_{t-1}).

Both baselines operate directly on the same token vocabulary as our neural
models, so their outputs can be exported to MIDI with exactly the same tools.
"""
from __future__ import annotations
import numpy as np
from src.config import VOCAB_SIZE, SEQ_LEN


class RandomGenerator:
    def __init__(self):
        self.p = None

    def fit(self, seqs: list[np.ndarray]):
        all_tok = np.concatenate([s for s in seqs])
        counts  = np.bincount(all_tok, minlength=VOCAB_SIZE).astype(np.float32)
        self.p  = counts / counts.sum()
        return self

    def sample(self, n_steps: int = SEQ_LEN,
               rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        return rng.choice(VOCAB_SIZE, size=n_steps, p=self.p).astype(np.int64)


class MarkovChain:
    """First-order Markov model with add-one smoothing."""
    def __init__(self, vocab: int = VOCAB_SIZE):
        self.V     = vocab
        self.trans = None           # (V, V) conditional probs
        self.start = None           # (V,)   initial distribution

    def fit(self, seqs: list[np.ndarray]):
        trans = np.ones((self.V, self.V), dtype=np.float32)     # +1 smoothing
        start = np.ones(self.V,          dtype=np.float32)
        for s in seqs:
            if len(s) == 0:
                continue
            start[int(s[0])] += 1
            for a, b in zip(s[:-1], s[1:]):
                trans[int(a), int(b)] += 1
        self.trans = trans / trans.sum(axis=1, keepdims=True)
        self.start = start / start.sum()
        return self

    def sample(self, n_steps: int = SEQ_LEN,
               rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        out = np.empty(n_steps, dtype=np.int64)
        out[0] = rng.choice(self.V, p=self.start)
        for t in range(1, n_steps):
            out[t] = rng.choice(self.V, p=self.trans[out[t-1]])
        return out
