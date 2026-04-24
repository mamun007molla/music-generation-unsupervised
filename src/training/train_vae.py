"""
Train Task 2 : Variational Autoencoder for multi-genre music generation.
Produces
    outputs/plots/task2_vae_loss.png
    outputs/generated_midis/task2_vae_{genre}_{i}.mid   (8 total)
    outputs/plots/task2_latent_interp.png
    outputs/task2_vae_params.pkl
"""
from __future__ import annotations
import numpy as np, pickle, time
from pathlib import Path

from src.config import (SEED, VAE_HIDDEN, VAE_LATENT, VAE_EPOCHS, VAE_LR,
                        VAE_BATCH, VAE_BETA, SEQ_LEN, VOCAB_SIZE,
                        NUM_GENRES, MIDI_OUT_DIR, PLOT_DIR, OUTPUT_DIR, GENRES)
from src.preprocessing.loader import load_corpus, iterate_batches
from src.models import vae as VAE
from src.training.optim import Adam
from src.generation.midi_export import save_tokens_as_midi


def train_vae():
    print("="*70); print(" Task 2 : Variational Autoencoder (multi-genre)")
    print("="*70)
    rng = np.random.default_rng(SEED + 1)

    X, G = load_corpus()
    print(f"  total windows : {len(X)}  ({NUM_GENRES} genres)")

    vocab  = int(X.max()) + 1 if X.max() >= 90 else 90
    params, meta = VAE.init_params(vocab=vocab, T=SEQ_LEN,
                             hidden=VAE_HIDDEN, latent=VAE_LATENT,
                             n_genres=NUM_GENRES, seed=SEED + 1)
    opt    = Adam(lr=VAE_LR)

    loss_hist, recon_hist, kl_hist = [], [], []
    t0 = time.time()
    for epoch in range(1, VAE_EPOCHS + 1):
        ep_l, ep_r, ep_k = [], [], []
        for xb, gb in iterate_batches(X, G, VAE_BATCH, rng):
            # one-hot genre & fresh epsilon each step
            goh = np.eye(NUM_GENRES, dtype=np.float32)[gb]
            eps = rng.standard_normal((len(xb), VAE_LATENT)).astype(np.float32)
            total, recon, kl = VAE.vae_loss(params, meta, xb, goh, eps, VAE_BETA)
            grads  = VAE.vae_loss_grad(params, meta, xb, goh, eps, VAE_BETA)
            params = opt.step(params, grads)
            ep_l.append(float(total)); ep_r.append(float(recon)); ep_k.append(float(kl))
        loss_hist.append(np.mean(ep_l))
        recon_hist.append(np.mean(ep_r))
        kl_hist.append(np.mean(ep_k))
        print(f"  epoch {epoch:3d}/{VAE_EPOCHS}  "
              f"total={loss_hist[-1]:.3f}  recon={recon_hist[-1]:.3f}  "
              f"KL={kl_hist[-1]:.3f}")
    print(f"  trained in {time.time()-t0:.1f}s")

    # ---- checkpoint ----------------------------------------------------
    with open(OUTPUT_DIR / "task2_vae_params.pkl", "wb") as f:
        pickle.dump({"params": params, "meta": meta}, f)

    # ---- plots ---------------------------------------------------------
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.5))
    for a, y, t in zip(ax, [loss_hist, recon_hist, kl_hist],
                           ["total", "reconstruction", "KL divergence"]):
        a.plot(y, lw=2); a.set_title(t); a.set_xlabel("epoch")
        a.grid(alpha=0.3)
    fig.suptitle("Task 2 — β-VAE training curves")
    plt.tight_layout(); plt.savefig(PLOT_DIR / "task2_vae_loss.png", dpi=150)
    plt.close()

    # ---- 8 samples across all genres ----------------------------------
    print("  generating 8 multi-genre samples …")
    per_genre = {"classical": 2, "jazz": 2, "rock": 2, "pop": 1, "electronic": 1}
    s = 0
    for genre, count in per_genre.items():
        for i in range(count):
            toks = VAE.sample(params, meta, genre_idx=GENRES.index(genre),
                              n=1, rng=rng, temperature=0.9)[0]
            save_tokens_as_midi(toks,
                MIDI_OUT_DIR / f"task2_vae_{genre}_{i+1}.mid", genre=genre)
            s += 1
    print(f"    saved {s} files")

    # ---- latent interpolation experiment ------------------------------
    print("  running latent interpolation …")
    z1 = rng.standard_normal(VAE_LATENT).astype(np.float32)
    z2 = rng.standard_normal(VAE_LATENT).astype(np.float32)
    interp_tokens = VAE.latent_interpolate(params, meta, z1, z2,
                                           genre_idx=GENRES.index("pop"), steps=5)
    # visualise as pianoroll strips
    fig, axes = plt.subplots(5, 1, figsize=(10, 6), sharex=True)
    for ax, toks, a in zip(axes, interp_tokens,
                            np.linspace(0, 1, len(interp_tokens))):
        roll = np.zeros((vocab, SEQ_LEN))
        for t, tok in enumerate(toks):
            roll[int(tok), t] = 1
        ax.imshow(roll, aspect="auto", cmap="Purples", origin="lower")
        ax.set_ylabel(f"α={a:.2f}"); ax.set_yticks([])
    axes[-1].set_xlabel("time step")
    fig.suptitle("Task 2 — Latent interpolation between two z vectors")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "task2_latent_interp.png", dpi=150)
    plt.close()

    return loss_hist, params


if __name__ == "__main__":
    train_vae()
