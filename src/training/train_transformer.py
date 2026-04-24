"""
Train Task 3 : Transformer decoder for long-horizon music generation.
Produces
    outputs/plots/task3_tr_loss.png
    outputs/generated_midis/task3_tr_{genre}_{i}.mid   (10 total)
    outputs/task3_tr_params.pkl
"""
from __future__ import annotations
import numpy as np, pickle, time
from pathlib import Path

from src.config import (SEED, TR_D_MODEL, TR_N_HEADS, TR_N_LAYERS,
                        TR_EPOCHS, TR_LR, TR_BATCH,
                        SEQ_LEN, NUM_GENRES, MIDI_OUT_DIR, PLOT_DIR,
                        OUTPUT_DIR, GENRES)
from src.preprocessing.loader import load_corpus, iterate_batches
from src.models import transformer as TR
from src.training.optim import Adam
from src.generation.midi_export import save_tokens_as_midi


def train_transformer():
    print("="*70); print(" Task 3 : Transformer decoder (multi-genre)")
    print("="*70)
    rng = np.random.default_rng(SEED + 2)

    X, G = load_corpus()
    vocab = int(X.max()) + 1 if X.max() >= 90 else 90
    params, meta = TR.init_params(vocab=vocab, T=SEQ_LEN,
                            d_model=TR_D_MODEL, n_heads=TR_N_HEADS,
                            n_layers=TR_N_LAYERS, n_genres=NUM_GENRES,
                            seed=SEED + 2)
    opt    = Adam(lr=TR_LR)

    loss_hist, ppl_hist = [], []
    t0 = time.time()
    for epoch in range(1, TR_EPOCHS + 1):
        ep_l = []
        for xb, gb in iterate_batches(X, G, TR_BATCH, rng):
            loss_val = float(TR.tr_loss(params, meta, xb, gb))
            grads    = TR.tr_loss_grad(params, meta, xb, gb)
            params   = opt.step(params, grads)
            ep_l.append(loss_val)
        l  = float(np.mean(ep_l));  ppl = float(np.exp(l))
        loss_hist.append(l); ppl_hist.append(ppl)
        print(f"  epoch {epoch:3d}/{TR_EPOCHS}  CE={l:.3f}  perplexity={ppl:.2f}")
    print(f"  trained in {time.time()-t0:.1f}s")

    with open(OUTPUT_DIR / "task3_tr_params.pkl", "wb") as f:
        pickle.dump({"params": params, "meta": meta}, f)

    # ---- plots ---------------------------------------------------------
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(loss_hist, lw=2, color="#d62728"); ax[0].set_title("cross-entropy")
    ax[0].set_xlabel("epoch"); ax[0].grid(alpha=0.3)
    ax[1].plot(ppl_hist,  lw=2, color="#9467bd"); ax[1].set_title("perplexity")
    ax[1].set_xlabel("epoch"); ax[1].grid(alpha=0.3)
    fig.suptitle("Task 3 — Transformer training")
    plt.tight_layout(); plt.savefig(PLOT_DIR / "task3_tr_loss.png", dpi=150)
    plt.close()

    # ---- 10 long compositions (2 per genre) ---------------------------
    print("  generating 10 long compositions …")
    s = 0
    for genre in GENRES:
        for i in range(2):
            toks = TR.sample(params, meta, genre_idx=GENRES.index(genre),
                             n_steps=SEQ_LEN, temperature=0.95, rng=rng)
            save_tokens_as_midi(toks,
                MIDI_OUT_DIR / f"task3_tr_{genre}_{i+1}.mid", genre=genre)
            s += 1
    print(f"    saved {s} files")
    return loss_hist, ppl_hist, params


if __name__ == "__main__":
    train_transformer()
