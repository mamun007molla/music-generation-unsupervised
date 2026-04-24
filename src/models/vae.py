"""
Task 2 — Variational Autoencoder for multi-genre generation
-----------------------------------------------------------

We use an MLP-VAE over a *flattened* one-hot-ish token sequence.
Although the input/output stays sequence-structured (T time-steps), the
encoder/decoder themselves are fully-connected — this keeps the math clean
and lets us clearly demonstrate:

  q_phi(z|x) = N( mu(x), diag(sigma(x)^2) )
  z = mu + sigma * eps,   eps ~ N(0, I)
  p_theta(x|z) = softmax(decoder(z))
  L_VAE = E_q[-log p(x|z)] + beta * D_KL( q(z|x) || N(0,I) )

A lightweight genre-conditioning vector (one-hot over 5 genres) is concatenated
to z before decoding, which allows controllable multi-genre sampling at
inference time without changing the unsupervised reconstruction objective.
"""
from __future__ import annotations
import autograd.numpy as anp
from autograd import grad
import numpy as np

from src.config import VOCAB_SIZE, SEQ_LEN, NUM_GENRES, PAD_TOKEN


def _xavier(shape, rng):
    lim = np.sqrt(6.0 / (shape[0] + shape[-1]))
    return rng.uniform(-lim, lim, shape).astype(np.float32)


def init_params(vocab: int = VOCAB_SIZE, T: int = SEQ_LEN,
                hidden: int = 128, latent: int = 32,
                n_genres: int = NUM_GENRES, seed: int = 1):
    """Returns (params, meta) — autograd only sees `params`."""
    rng   = np.random.default_rng(seed)
    d_in  = T * vocab
    p     = {}
    # ---- encoder ---------------------------------------------------------
    p["W1"]   = _xavier((d_in, hidden), rng)
    p["b1"]   = np.zeros(hidden, dtype=np.float32)
    p["W_mu"] = _xavier((hidden, latent), rng)
    p["b_mu"] = np.zeros(latent, dtype=np.float32)
    p["W_lv"] = _xavier((hidden, latent), rng)
    p["b_lv"] = np.zeros(latent, dtype=np.float32)
    # ---- decoder (concat z with one-hot genre) --------------------------
    p["W2"]   = _xavier((latent + n_genres, hidden), rng)
    p["b2"]   = np.zeros(hidden, dtype=np.float32)
    p["W3"]   = _xavier((hidden, d_in), rng)
    p["b3"]   = np.zeros(d_in, dtype=np.float32)
    meta = dict(T=T, V=vocab, G=n_genres, L=latent)
    return p, meta


def _one_hot(idx, V):
    eye = anp.eye(V, dtype=anp.float32)
    return eye[idx]


def encode(params, meta, x_tokens):
    """x_tokens : (B, T)  ->  mu, logvar  both (B, L)"""
    V   = meta["V"]
    xoh = _one_hot(x_tokens, V).reshape(x_tokens.shape[0], -1)   # (B, T*V)
    h   = anp.tanh(anp.dot(xoh, params["W1"]) + params["b1"])
    mu  = anp.dot(h, params["W_mu"]) + params["b_mu"]
    lv  = anp.dot(h, params["W_lv"]) + params["b_lv"]
    return mu, lv, xoh


def decode(params, meta, z, genre_onehot):
    """z : (B, L), genre_onehot : (B, G)  ->  logits (B, T, V)"""
    T, V = meta["T"], meta["V"]
    zg   = anp.concatenate([z, genre_onehot], axis=-1)
    h    = anp.tanh(anp.dot(zg, params["W2"]) + params["b2"])
    flat = anp.dot(h, params["W3"]) + params["b3"]
    return flat.reshape(-1, T, V)


def reparameterize(mu, logvar, rng):
    eps = rng.standard_normal(mu.shape).astype(np.float32)
    return mu + anp.exp(0.5 * logvar) * eps


def vae_loss(params, meta, x_tokens, genre_onehot, eps, beta: float = 0.05):
    mu, lv, xoh = encode(params, meta, x_tokens)
    z           = mu + anp.exp(0.5 * lv) * eps                 # deterministic w.r.t eps
    logits      = decode(params, meta, z, genre_onehot)        # (B,T,V)
    # reconstruction — multinomial cross-entropy
    shifted = logits - anp.max(logits, axis=-1, keepdims=True)
    log_probs = shifted - anp.log(anp.sum(anp.exp(shifted), axis=-1, keepdims=True))
    one_hot = anp.eye(meta["V"], dtype=anp.float32)[x_tokens]
    mask    = (x_tokens != PAD_TOKEN).astype(anp.float32)
    nll_per = -anp.sum(one_hot * log_probs, axis=-1)           # (B,T)
    recon   = anp.sum(nll_per * mask) / (anp.sum(mask) + 1e-8)
    # KL  D_KL( q || N(0,I) )
    kl = -0.5 * anp.mean(anp.sum(1.0 + lv - mu**2 - anp.exp(lv), axis=-1))
    return recon + beta * kl, recon, kl


def vae_total_loss(params, meta, x_tokens, genre_onehot, eps, beta=0.05):
    total, _, _ = vae_loss(params, meta, x_tokens, genre_onehot, eps, beta)
    return total


# autograd differentiates only w.r.t. argument 0 (params)
vae_loss_grad = grad(vae_total_loss, argnum=0)


def sample(params, meta, genre_idx: int, n: int = 1,
           rng: np.random.Generator | None = None,
           temperature: float = 1.0) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    z     = rng.standard_normal((n, meta["L"])).astype(np.float32)
    gvec  = np.zeros((n, meta["G"]), dtype=np.float32)
    gvec[:, genre_idx] = 1.0
    logits = np.asarray(decode(params, meta, z, gvec))         # (n,T,V)
    logits /= temperature
    probs  = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True)
    T, V   = probs.shape[1], probs.shape[2]
    out    = np.zeros((n, T), dtype=np.int64)
    for b in range(n):
        for t in range(T):
            out[b, t] = rng.choice(V, p=probs[b, t])
    return out


def latent_interpolate(params, meta, z1, z2, genre_idx: int, steps: int = 5) -> np.ndarray:
    """Linear interpolation between two latent codes — for the interp experiment."""
    z1, z2 = np.asarray(z1), np.asarray(z2)
    alphas = np.linspace(0, 1, steps)
    zs     = np.stack([(1 - a) * z1 + a * z2 for a in alphas], axis=0)
    gvec   = np.zeros((steps, meta["G"]), dtype=np.float32)
    gvec[:, genre_idx] = 1.0
    logits = np.asarray(decode(params, meta, zs, gvec))
    return np.argmax(logits, axis=-1)                           # greedy decode
