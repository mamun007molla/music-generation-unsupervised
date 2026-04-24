"""
Task 3 — Transformer decoder for long-horizon music generation
--------------------------------------------------------------

A small GPT-style decoder-only Transformer:

    Embedding(V, D) + PositionalEmbedding + GenreEmbedding
        |
    [ MaskedMultiHeadAttn  ->  LayerNorm  ->  FFN  ->  LayerNorm ] x N
        |
    Linear head (D -> V)

Loss  —  standard auto-regressive log-likelihood (next-token prediction):

    L_TR  =  - (1/T) * sum_t log p_theta( x_t | x_{<t} )

Perplexity  =  exp( L_TR )
"""
from __future__ import annotations
import autograd.numpy as anp
from autograd import grad
import numpy as np

from src.config import VOCAB_SIZE, SEQ_LEN, NUM_GENRES, PAD_TOKEN


# ---------------------------------------------------------------------------
#  initialisation
# ---------------------------------------------------------------------------
def _randn(shape, rng, scale=0.02):
    return (rng.standard_normal(shape) * scale).astype(np.float32)


def init_params(vocab: int = VOCAB_SIZE, T: int = SEQ_LEN,
                d_model: int = 64, n_heads: int = 4, n_layers: int = 2,
                n_genres: int = NUM_GENRES, seed: int = 2):
    """Returns (params, meta); autograd differentiates only w.r.t. params."""
    assert d_model % n_heads == 0
    rng  = np.random.default_rng(seed)
    p    = {}
    p["E_tok"]   = _randn((vocab,    d_model), rng)
    p["E_pos"]   = _randn((T,        d_model), rng)
    p["E_gen"]   = _randn((n_genres, d_model), rng)
    for l in range(n_layers):
        p[f"Wq_{l}"] = _randn((d_model, d_model), rng)
        p[f"Wk_{l}"] = _randn((d_model, d_model), rng)
        p[f"Wv_{l}"] = _randn((d_model, d_model), rng)
        p[f"Wo_{l}"] = _randn((d_model, d_model), rng)
        p[f"W1_{l}"] = _randn((d_model, 4*d_model), rng)
        p[f"b1_{l}"] = np.zeros(4*d_model, dtype=np.float32)
        p[f"W2_{l}"] = _randn((4*d_model, d_model), rng)
        p[f"b2_{l}"] = np.zeros(d_model,   dtype=np.float32)
        p[f"ln1_g_{l}"] = np.ones(d_model, dtype=np.float32)
        p[f"ln1_b_{l}"] = np.zeros(d_model, dtype=np.float32)
        p[f"ln2_g_{l}"] = np.ones(d_model, dtype=np.float32)
        p[f"ln2_b_{l}"] = np.zeros(d_model, dtype=np.float32)
    p["W_head"] = _randn((d_model, vocab), rng)
    p["b_head"] = np.zeros(vocab, dtype=np.float32)
    meta = dict(T=T, V=vocab, D=d_model, H=n_heads, L=n_layers, G=n_genres)
    return p, meta


# ---------------------------------------------------------------------------
#  building blocks
# ---------------------------------------------------------------------------
def layernorm(x, g, b, eps=1e-5):
    mu  = anp.mean(x, axis=-1, keepdims=True)
    var = anp.mean((x - mu) ** 2, axis=-1, keepdims=True)
    return (x - mu) / anp.sqrt(var + eps) * g + b


def masked_softmax(scores, mask):
    """scores,mask : (B,h,T,T). mask = 1 means keep."""
    scores = anp.where(mask > 0.5, scores, -1e9)
    m = anp.max(scores, axis=-1, keepdims=True)
    e = anp.exp(scores - m)
    return e / anp.sum(e, axis=-1, keepdims=True)


def multihead_attn(x, Wq, Wk, Wv, Wo, n_heads, causal_mask):
    B, T, D = x.shape
    H = n_heads;  Dh = D // H
    q = anp.transpose(anp.dot(x, Wq).reshape(B, T, H, Dh), (0, 2, 1, 3))   # (B,H,T,Dh)
    k = anp.transpose(anp.dot(x, Wk).reshape(B, T, H, Dh), (0, 2, 1, 3))
    v = anp.transpose(anp.dot(x, Wv).reshape(B, T, H, Dh), (0, 2, 1, 3))
    scores = anp.matmul(q, anp.transpose(k, (0, 1, 3, 2))) / anp.sqrt(Dh)
    attn   = masked_softmax(scores, causal_mask)
    ctx    = anp.transpose(anp.matmul(attn, v), (0, 2, 1, 3)).reshape(B, T, D)
    return anp.dot(ctx, Wo)


def _causal_mask(T):
    m = anp.tri(T, T)                           # 1 on & below diagonal
    return m.reshape(1, 1, T, T)


def forward(params, meta, tokens, genres):
    """tokens (B,T) int ; genres (B,) int  ->  logits (B,T,V)"""
    B, T = tokens.shape
    x    = params["E_tok"][tokens] \
         + params["E_pos"][None, :T, :] \
         + params["E_gen"][genres][:, None, :]
    mask = _causal_mask(T)
    for l in range(meta["L"]):
        # attention sub-layer with residual + LN
        a = multihead_attn(x, params[f"Wq_{l}"], params[f"Wk_{l}"],
                              params[f"Wv_{l}"], params[f"Wo_{l}"],
                              meta["H"], mask)
        x = layernorm(x + a, params[f"ln1_g_{l}"], params[f"ln1_b_{l}"])
        # FFN sub-layer with residual + LN
        h = anp.maximum(0.0, anp.dot(x, params[f"W1_{l}"]) + params[f"b1_{l}"])
        h = anp.dot(h, params[f"W2_{l}"]) + params[f"b2_{l}"]
        x = layernorm(x + h, params[f"ln2_g_{l}"], params[f"ln2_b_{l}"])
    logits = anp.dot(x, params["W_head"]) + params["b_head"]      # (B,T,V)
    return logits


def tr_loss(params, meta, tokens, genres):
    """Auto-regressive CE on positions 1..T-1 (next-token prediction)."""
    logits = forward(params, meta, tokens, genres)[:, :-1, :]
    target = tokens[:, 1:]
    shifted = logits - anp.max(logits, axis=-1, keepdims=True)
    logp    = shifted - anp.log(anp.sum(anp.exp(shifted), axis=-1, keepdims=True))
    one_hot = anp.eye(meta["V"], dtype=anp.float32)[target]
    mask    = (target != PAD_TOKEN).astype(anp.float32)
    nll     = -anp.sum(one_hot * logp, axis=-1)
    return anp.sum(nll * mask) / (anp.sum(mask) + 1e-8)


tr_loss_grad = grad(tr_loss, argnum=0)


# ---------------------------------------------------------------------------
#  sampling
# ---------------------------------------------------------------------------
def sample(params, meta, genre_idx: int, n_steps: int,
           temperature: float = 1.0,
           rng: np.random.Generator | None = None,
           prefix: np.ndarray | None = None) -> np.ndarray:
    """Auto-regressive generation up to `n_steps` tokens (<= T)."""
    if rng is None:
        rng = np.random.default_rng()
    T = meta["T"]
    n_steps = min(n_steps, T)

    tokens = np.zeros((1, T), dtype=np.int64)
    if prefix is not None:
        k = min(len(prefix), T)
        tokens[0, :k] = prefix[:k]
        start = k
    else:
        start = 1                           # keep position 0 as <PAD>=start

    genres = np.array([genre_idx], dtype=np.int64)
    for t in range(start, n_steps):
        logits = np.asarray(forward(params, meta, tokens, genres))      # (1,T,V)
        logit  = logits[0, t - 1] / temperature
        p      = np.exp(logit - logit.max());  p /= p.sum()
        tokens[0, t] = rng.choice(meta["V"], p=p)
    return tokens[0, :n_steps]
