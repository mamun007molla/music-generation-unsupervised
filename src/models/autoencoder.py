"""
Task 1 — LSTM Autoencoder for single-genre music generation
-----------------------------------------------------------

Architecture
    Embedding (V -> D)
        |
    LSTM encoder  (D -> H)
        |
    latent code   z = W_z h_T          (R^L)
        |
    LSTM decoder  (z fed at every step) -> hidden (L -> H)
        |
    Linear head   (H -> V)  -> per-step softmax over vocabulary

Loss
    L_AE  =  - (1/T) * sum_t  log p_theta( x_t | z )          (reconstruction CE)

Backpropagation is handled by the `autograd` library, which lets us write the
forward pass with `autograd.numpy` and obtain analytical gradients for every
parameter automatically.  All code paths are pure-python — no GPU required.
"""
from __future__ import annotations
import autograd.numpy as anp
from autograd import grad
import numpy as np

from src.config import VOCAB_SIZE, PAD_TOKEN, SEQ_LEN


# ---------------------------------------------------------------------------
#  initialisation helpers
# ---------------------------------------------------------------------------
def _xavier(shape, rng):
    lim = np.sqrt(6.0 / (shape[0] + shape[-1]))
    return rng.uniform(-lim, lim, shape).astype(np.float32)


def init_params(vocab: int, d_emb: int, hidden: int, latent: int,
                seed: int = 0) -> dict:
    """Create every weight tensor the autoencoder needs."""
    rng = np.random.default_rng(seed)
    p   = {}
    p["E"]    = _xavier((vocab, d_emb), rng)                      # embedding

    # encoder LSTM gates:  i,f,g,o   (concatenated)
    p["Wxh_e"] = _xavier((d_emb,  4 * hidden), rng)
    p["Whh_e"] = _xavier((hidden, 4 * hidden), rng)
    p["bh_e"]  = np.zeros(4 * hidden, dtype=np.float32)

    # latent projection
    p["Wz"]   = _xavier((hidden, latent), rng)
    p["bz"]   = np.zeros(latent, dtype=np.float32)

    # decoder LSTM gates  (input = concat[emb(prev_out), z])
    p["Wxh_d"] = _xavier((d_emb + latent, 4 * hidden), rng)
    p["Whh_d"] = _xavier((hidden,         4 * hidden), rng)
    p["bh_d"]  = np.zeros(4 * hidden, dtype=np.float32)

    # output head
    p["Wout"] = _xavier((hidden, vocab), rng)
    p["bout"] = np.zeros(vocab, dtype=np.float32)
    return p


# ---------------------------------------------------------------------------
#  basic building blocks
# ---------------------------------------------------------------------------
def _sigmoid(x):  return 1.0 / (1.0 + anp.exp(-x))
def _softmax(x, axis=-1):
    x = x - anp.max(x, axis=axis, keepdims=True)
    e = anp.exp(x)
    return e / anp.sum(e, axis=axis, keepdims=True)


def lstm_step(x, h, c, Wx, Wh, b):
    """One LSTM time-step (Hochreiter '97)."""
    gates = anp.dot(x, Wx) + anp.dot(h, Wh) + b
    H     = gates.shape[-1] // 4
    i = _sigmoid(gates[..., 0*H:1*H])
    f = _sigmoid(gates[..., 1*H:2*H])
    g = anp.tanh(gates[..., 2*H:3*H])
    o = _sigmoid(gates[..., 3*H:4*H])
    c_new = f * c + i * g
    h_new = o * anp.tanh(c_new)
    return h_new, c_new


# ---------------------------------------------------------------------------
#  full forward pass
# ---------------------------------------------------------------------------
def forward(params, batch_tokens):
    """
    batch_tokens : (B, T) int
    returns      : (logits  (B, T, V), latent (B, L))
    """
    B, T       = batch_tokens.shape
    H          = params["Whh_e"].shape[0]
    emb        = params["E"][batch_tokens]                    # (B,T,D)

    # ---- encoder ---------------------------------------------------------
    h = anp.zeros((B, H));  c = anp.zeros((B, H))
    for t in range(T):
        h, c = lstm_step(emb[:, t, :], h, c,
                         params["Wxh_e"], params["Whh_e"], params["bh_e"])
    z = anp.dot(h, params["Wz"]) + params["bz"]               # (B,L)

    # ---- decoder ---------------------------------------------------------
    h = anp.zeros((B, H));  c = anp.zeros((B, H))
    prev_tok = anp.zeros((B, params["E"].shape[1]))           # start token = 0
    logits_list = []
    for t in range(T):
        dec_in = anp.concatenate([prev_tok, z], axis=-1)
        h, c   = lstm_step(dec_in, h, c,
                           params["Wxh_d"], params["Whh_d"], params["bh_d"])
        logit  = anp.dot(h, params["Wout"]) + params["bout"]   # (B,V)
        logits_list.append(logit)
        # teacher forcing with the *true* previous token keeps training stable
        prev_tok = params["E"][batch_tokens[:, t]]
    logits = anp.stack(logits_list, axis=1)                   # (B,T,V)
    return logits, z


# ---------------------------------------------------------------------------
#  loss
# ---------------------------------------------------------------------------
def ae_loss(params, batch_tokens):
    logits, _ = forward(params, batch_tokens)                 # (B,T,V)
    log_probs = logits - anp.log(anp.sum(anp.exp(
                   logits - anp.max(logits, axis=-1, keepdims=True)),
                   axis=-1, keepdims=True)) \
                - anp.max(logits, axis=-1, keepdims=True)
    B, T, V   = logits.shape
    idx       = batch_tokens
    mask      = (idx != PAD_TOKEN).astype(anp.float32)
    # gather log prob of true class  ->  (B,T)
    one_hot   = anp.eye(V, dtype=anp.float32)[idx]
    nll       = -anp.sum(one_hot * log_probs, axis=-1)
    return anp.sum(nll * mask) / (anp.sum(mask) + 1e-8)


ae_loss_grad = grad(ae_loss)


# ---------------------------------------------------------------------------
#  sampling
# ---------------------------------------------------------------------------
def sample_from_latent(params, z, T: int = SEQ_LEN,
                       temperature: float = 1.0,
                       rng: np.random.Generator | None = None) -> np.ndarray:
    """Generate a token sequence conditioned on a latent code z  (L,) or (B,L)."""
    if rng is None:
        rng = np.random.default_rng()
    z = np.atleast_2d(z).astype(np.float32)
    B = z.shape[0]
    H = params["Whh_d"].shape[0]
    h = np.zeros((B, H), dtype=np.float32)
    c = np.zeros((B, H), dtype=np.float32)
    prev_tok = np.zeros((B, params["E"].shape[1]), dtype=np.float32)

    out = np.zeros((B, T), dtype=np.int64)
    for t in range(T):
        dec_in = np.concatenate([prev_tok, z], axis=-1)
        h, c   = lstm_step(dec_in, h, c,
                           params["Wxh_d"], params["Whh_d"], params["bh_d"])
        logit  = np.asarray(np.dot(h, params["Wout"]) + params["bout"])
        probs  = np.exp((logit - logit.max(axis=-1, keepdims=True)) / temperature)
        probs /= probs.sum(axis=-1, keepdims=True)
        # sample
        tok    = np.array([rng.choice(logit.shape[-1], p=probs[b])
                           for b in range(B)])
        out[:, t] = tok
        prev_tok  = params["E"][tok]
    return out if B > 1 else out[0]
