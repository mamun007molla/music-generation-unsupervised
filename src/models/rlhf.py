"""
Task 4 — Reinforcement Learning from Human Feedback (RLHF)
----------------------------------------------------------

We take the *pre-trained* Transformer from Task 3 as the policy
    pi_theta( x_t | x_{<t} )
and fine-tune it to maximise a scalar reward r(X) that reflects musical
preference.

REINFORCE objective (with baseline b for variance reduction):

    J(theta)  =  E_{X ~ pi_theta} [ r(X) ]
    grad_theta J  =  E [ (r - b) * sum_t grad_theta log pi_theta(x_t | x_{<t}) ]

Reward model
------------
Ideally r(X) is learned from pairwise human preference data collected with a
listening survey.  For reproducibility and because the survey size in a course
project is small, we use a *hybrid* reward:

    r(X) = 0.6 * r_rule(X)     # rule-based musicality proxy
         + 0.4 * r_human(X)    # learned from our listener survey

where r_rule rewards three musical priors (in-scale notes, rhythm variety and
low repetition) and r_human is a shallow linear regressor trained on the
survey scores collected from 10 listeners.
"""
from __future__ import annotations
import autograd.numpy as anp
from autograd import grad
import numpy as np

from src.config    import VOCAB_SIZE, NOTE_OFFSET, REST_TOKEN, PAD_TOKEN
from src.models    import transformer as tr


# ---------------------------------------------------------------------------
#  rule-based reward components
# ---------------------------------------------------------------------------
# major scale semitone offsets from tonic C = 60
_C_MAJOR = {0, 2, 4, 5, 7, 9, 11}


def _in_scale_ratio(tokens: np.ndarray) -> float:
    notes = [int(t) - NOTE_OFFSET + 21 for t in tokens
             if int(t) >= NOTE_OFFSET]
    if not notes:
        return 0.0
    return np.mean([(n % 12) in _C_MAJOR for n in notes])


def _rhythm_variety(tokens: np.ndarray) -> float:
    """Fraction of unique note-onset durations -> diversity proxy in [0,1]."""
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


def _repetition_penalty(tokens: np.ndarray, k: int = 4) -> float:
    """Penalise short-loop repetition:  higher -> worse."""
    if len(tokens) < 2 * k:
        return 0.0
    pats   = [tuple(tokens[i:i+k]) for i in range(len(tokens) - k + 1)]
    uniq   = len(set(pats))
    return 1.0 - uniq / len(pats)


def rule_reward(tokens: np.ndarray) -> float:
    return (0.5 * _in_scale_ratio(tokens)
          + 0.3 * _rhythm_variety(tokens)
          + 0.2 * (1.0 - _repetition_penalty(tokens)))


# ---------------------------------------------------------------------------
#  learned reward from survey
# ---------------------------------------------------------------------------
class SurveyRewardModel:
    """Linear regressor on 3 hand-crafted musicality features -> mean-opinion."""
    def __init__(self):
        self.w = np.zeros(3, dtype=np.float32)
        self.b = 0.0

    def features(self, tokens: np.ndarray) -> np.ndarray:
        return np.array([_in_scale_ratio(tokens),
                         _rhythm_variety(tokens),
                         1.0 - _repetition_penalty(tokens)],
                        dtype=np.float32)

    def fit(self, tokens_list, scores, lr=0.01, iters=300):
        X = np.stack([self.features(t) for t in tokens_list])   # (N,3)
        y = np.asarray(scores, dtype=np.float32)
        for _ in range(iters):
            pred = X @ self.w + self.b
            err  = pred - y
            self.w -= lr * (X.T @ err) / len(y)
            self.b -= lr * float(np.mean(err))
        return self

    def score(self, tokens: np.ndarray) -> float:
        feats = self.features(tokens)
        s = float(feats @ self.w + self.b)
        return (s - 1.0) / 4.0                          # rescale 1..5 -> 0..1


# ---------------------------------------------------------------------------
#  combined reward
# ---------------------------------------------------------------------------
def combined_reward(tokens: np.ndarray,
                    reward_model: SurveyRewardModel | None = None) -> float:
    r = rule_reward(tokens)
    if reward_model is not None:
        r = 0.6 * r + 0.4 * reward_model.score(tokens)
    return float(r)


# ---------------------------------------------------------------------------
#  REINFORCE update on transformer log-probabilities
# ---------------------------------------------------------------------------
def _seq_logp(params, meta, tokens, genres):
    """sum_t log p_theta(x_t | x_{<t})  for a single sampled sequence."""
    logits = tr.forward(params, meta, tokens, genres)[:, :-1, :]
    target = tokens[:, 1:]
    shifted = logits - anp.max(logits, axis=-1, keepdims=True)
    logp    = shifted - anp.log(anp.sum(anp.exp(shifted), axis=-1, keepdims=True))
    one_hot = anp.eye(meta["V"], dtype=anp.float32)[target]
    mask    = (target != PAD_TOKEN).astype(anp.float32)
    return anp.sum(one_hot * logp, axis=-1) * mask          # (B, T-1)


def pg_objective(params, meta, tokens_batch, genres_batch, advantage):
    """NEGATIVE of J(theta) so we can minimise it."""
    log_p = _seq_logp(params, meta, tokens_batch, genres_batch)   # (B, T-1)
    seq_logp = anp.sum(log_p, axis=-1)                            # (B,)
    return -anp.mean(advantage * seq_logp)


pg_grad = grad(pg_objective, argnum=0)
