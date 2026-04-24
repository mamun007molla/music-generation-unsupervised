"""
Quantitative evaluation metrics for symbolic music
--------------------------------------------------
Implements the four metrics requested by the rubric:

    (i)    Pitch histogram similarity  (L1 distance between pitch-class histos)
    (ii)   Rhythm diversity             #unique_durations / #total_notes
    (iii)  Repetition ratio             #repeated_patterns / #total_patterns
    (iv)   Perplexity                   exp(  -1/T * sum log p(x_t|x_{<t})  )
"""
from __future__ import annotations
import numpy as np
from src.config import NOTE_OFFSET, PAD_TOKEN


# ---------------------------------------------------------------------------
def _pitch_histogram(tokens: np.ndarray) -> np.ndarray:
    """12-bin normalised pitch-class histogram."""
    h = np.zeros(12, dtype=np.float32)
    for t in tokens:
        t = int(t)
        if t >= NOTE_OFFSET:
            pc = ((t - NOTE_OFFSET) + 21) % 12          # +21 because PITCH_MIN=21
            h[pc] += 1
    s = h.sum()
    return h / s if s > 0 else h


def pitch_histogram_similarity(a_tokens, b_tokens) -> float:
    """L1 distance between the pitch-class histograms (0 = identical)."""
    p = _pitch_histogram(a_tokens)
    q = _pitch_histogram(b_tokens)
    return float(np.sum(np.abs(p - q)))


# ---------------------------------------------------------------------------
def rhythm_diversity(tokens: np.ndarray) -> float:
    durs, cur, cur_dur = [], None, 0
    for t in tokens:
        t = int(t)
        if t == cur:
            cur_dur += 1
        else:
            if cur is not None and cur != PAD_TOKEN:
                durs.append(cur_dur)
            cur, cur_dur = t, 1
    if cur is not None and cur != PAD_TOKEN:
        durs.append(cur_dur)
    if not durs:
        return 0.0
    return len(set(durs)) / len(durs)


# ---------------------------------------------------------------------------
def repetition_ratio(tokens: np.ndarray, k: int = 4) -> float:
    if len(tokens) < 2 * k:
        return 0.0
    pats = [tuple(tokens[i:i+k]) for i in range(len(tokens) - k + 1)]
    return 1.0 - len(set(pats)) / len(pats)


# ---------------------------------------------------------------------------
def perplexity_from_loss(loss: float) -> float:
    return float(np.exp(loss))


# ---------------------------------------------------------------------------
def summarise(tokens_list, label="model") -> dict:
    """Aggregate metrics over a list of generated sequences."""
    rd   = np.mean([rhythm_diversity(t)  for t in tokens_list])
    rep  = np.mean([repetition_ratio(t) for t in tokens_list])
    # pitch histo similarity: compare every pair then take mean
    sims = []
    for i in range(len(tokens_list)):
        for j in range(i+1, len(tokens_list)):
            sims.append(pitch_histogram_similarity(tokens_list[i], tokens_list[j]))
    diversity_sim = float(np.mean(sims)) if sims else 0.0
    return {"label": label,
            "rhythm_diversity":    round(float(rd),  3),
            "repetition_ratio":    round(float(rep), 3),
            "pitch_hist_distance": round(diversity_sim, 3)}
