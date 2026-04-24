"""
Train Task 1 : LSTM Autoencoder on a single-genre (classical) slice.
Produces
    outputs/plots/task1_ae_loss.png
    outputs/generated_midis/task1_ae_sample_{1..5}.mid
    outputs/task1_ae_params.npz   (checkpoint)
"""
from __future__ import annotations
import numpy as np, pickle, time
from pathlib import Path

from src.config import (SEED, AE_HIDDEN, AE_LATENT, AE_EPOCHS, AE_LR, AE_BATCH,
                        SEQ_LEN, MIDI_OUT_DIR, PLOT_DIR, OUTPUT_DIR, GENRES)
from src.preprocessing.loader   import load_corpus, iterate_batches
from src.models import autoencoder as AE
from src.training.optim         import Adam
from src.generation.midi_export import save_tokens_as_midi


def train_ae():
    print("="*70); print(" Task 1 : LSTM Autoencoder (single genre = classical)")
    print("="*70)
    rng = np.random.default_rng(SEED)

    # ---- data: classical only ------------------------------------------
    X, G = load_corpus()
    mask = (G == GENRES.index("classical"))
    X    = X[mask]
    print(f"  classical windows : {len(X)}  (shape {X.shape})")

    params = AE.init_params(vocab=int(X.max()) + 1 if X.max() >= 90 else 90,
                             d_emb=32, hidden=AE_HIDDEN, latent=AE_LATENT,
                             seed=SEED)
    opt    = Adam(lr=AE_LR)

    loss_hist = []
    t0 = time.time()
    for epoch in range(1, AE_EPOCHS + 1):
        epoch_losses = []
        for xb, _ in iterate_batches(X, np.zeros(len(X)), AE_BATCH, rng):
            loss_val = float(AE.ae_loss(params, xb))
            grads    = AE.ae_loss_grad(params, xb)
            params   = opt.step(params, grads)
            epoch_losses.append(loss_val)
        ep_loss = float(np.mean(epoch_losses))
        loss_hist.append(ep_loss)
        print(f"  epoch {epoch:3d}/{AE_EPOCHS}  loss={ep_loss:.4f}")
    print(f"  trained in {time.time()-t0:.1f}s")

    # ---- save checkpoint -----------------------------------------------
    ckpt_path = OUTPUT_DIR / "task1_ae_params.pkl"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ckpt_path, "wb") as f:
        pickle.dump(params, f)

    # ---- loss curve ----------------------------------------------------
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,3.5))
    plt.plot(loss_hist, lw=2, color="#1f77b4")
    plt.xlabel("epoch"); plt.ylabel("reconstruction loss"); plt.grid(alpha=0.3)
    plt.title("Task 1 — LSTM Autoencoder training loss")
    plt.tight_layout(); plt.savefig(PLOT_DIR / "task1_ae_loss.png", dpi=150)
    plt.close()

    # ---- generate 5 samples --------------------------------------------
    print("  generating 5 MIDI samples …")
    for i in range(5):
        z   = rng.standard_normal(AE_LATENT).astype(np.float32) * 0.8
        toks = AE.sample_from_latent(params, z, T=SEQ_LEN,
                                     temperature=0.9, rng=rng)
        save_tokens_as_midi(toks,
            MIDI_OUT_DIR / f"task1_ae_sample_{i+1}.mid",
            genre="classical")
    return loss_hist, params


if __name__ == "__main__":
    train_ae()
