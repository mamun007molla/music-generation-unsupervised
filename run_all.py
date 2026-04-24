"""
run_all.py  —  End-to-end pipeline orchestrator.

Stages
  1. Build (or reuse) the MIDI corpus under data/raw_midi/
  2. Train baselines (random + Markov)
  3. Train Task 1 (LSTM-AE), Task 2 (VAE), Task 3 (Transformer), Task 4 (RLHF)
  4. Generate evaluation tables and final comparison plot
"""
from __future__ import annotations
import sys
import numpy as np, pickle, json, csv
from pathlib import Path

# allow `python run_all.py` from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import (SEED, SEQ_LEN, GENRES, OUTPUT_DIR, PLOT_DIR,
                        MIDI_OUT_DIR, RAW_MIDI_DIR)
from src.preprocessing.synthetic_corpus import build_corpus
from src.preprocessing.loader import load_corpus
from src.models.baselines import RandomGenerator, MarkovChain
from src.evaluation.metrics import summarise
from src.generation.midi_export import save_tokens_as_midi

from src.training.train_ae           import train_ae
from src.training.train_vae          import train_vae
from src.training.train_transformer  import train_transformer
from src.training.train_rlhf         import train_rlhf


def _ensure_corpus():
    midi_count = sum(1 for _ in RAW_MIDI_DIR.rglob("*.mid"))
    if midi_count < 50:
        print("\n>>> building synthetic multi-genre MIDI corpus …")
        build_corpus(n_per_genre=20)
    else:
        print(f">>> found {midi_count} .mid files in data/raw_midi/ (skipping build)")


def _run_baselines():
    print("\n" + "="*70)
    print(" BASELINES : random generator + first-order Markov chain")
    print("="*70)
    X, _ = load_corpus()
    seqs = [X[i] for i in range(len(X))]
    rng  = np.random.default_rng(SEED)

    rand = RandomGenerator().fit(seqs)
    mark = MarkovChain().fit(seqs)

    rand_samples, mark_samples = [], []
    for i in range(5):
        rtoks = rand.sample(SEQ_LEN, rng=rng); rand_samples.append(rtoks)
        mtoks = mark.sample(SEQ_LEN, rng=rng); mark_samples.append(mtoks)
        save_tokens_as_midi(rtoks,
            MIDI_OUT_DIR / f"baseline_random_{i+1}.mid")
        save_tokens_as_midi(mtoks,
            MIDI_OUT_DIR / f"baseline_markov_{i+1}.mid")
    return rand_samples, mark_samples


def _eval_generated_files(prefix: str, label: str,
                          folder: Path = MIDI_OUT_DIR):
    from src.preprocessing.midi_parser import midi_to_token_seq
    files = sorted(folder.glob(f"{prefix}*.mid"))
    if not files:
        return None
    toks = [midi_to_token_seq(f) for f in files]
    return summarise(toks, label=label)


def main():
    _ensure_corpus()

    rand_samples, mark_samples = _run_baselines()

    # --- Task 1 ---------------------------------------------------------
    ae_loss_hist, _          = train_ae()
    # --- Task 2 ---------------------------------------------------------
    vae_loss_hist, _         = train_vae()
    # --- Task 3 ---------------------------------------------------------
    tr_loss_hist, ppl_hist, _ = train_transformer()
    # --- Task 4 ---------------------------------------------------------
    rl_reward_hist, base_rwd, post_rwd, _ = train_rlhf()

    # --- consolidate metrics --------------------------------------------
    print("\n" + "="*70); print(" evaluation summary"); print("="*70)
    rows = []
    from src.evaluation.metrics import summarise
    rows.append(summarise(rand_samples, "Random baseline"))
    rows.append(summarise(mark_samples, "Markov baseline"))
    for prefix, label in [("task1_ae_",   "Task 1 — LSTM AE"),
                          ("task2_vae_",  "Task 2 — VAE"),
                          ("task3_tr_",   "Task 3 — Transformer"),
                          ("task4_rlhf_", "Task 4 — RLHF")]:
        r = _eval_generated_files(prefix, label)
        if r is not None:
            rows.append(r)

    # add model-specific numbers
    rows[2]["final_loss"]  = round(float(ae_loss_hist[-1]),  3)
    rows[3]["final_loss"]  = round(float(vae_loss_hist[-1]), 3)
    rows[4]["final_loss"]  = round(float(tr_loss_hist[-1]),  3)
    rows[4]["perplexity"]  = round(float(ppl_hist[-1]),      2)
    rows[5]["final_reward_pre"]   = round(float(np.mean(base_rwd)), 3)
    rows[5]["final_reward_post"]  = round(float(np.mean(post_rwd)), 3)

    # save JSON & Markdown table
    (OUTPUT_DIR / "evaluation_summary.json").write_text(
        json.dumps(rows, indent=2))
    md_lines = ["| Model | rhythm div | rep ratio | pitch-hist dist | extras |",
                "|-------|-----------:|----------:|----------------:|--------|"]
    for r in rows:
        extras = []
        for k, v in r.items():
            if k in ("label","rhythm_diversity","repetition_ratio","pitch_hist_distance"):
                continue
            extras.append(f"{k}={v}")
        md_lines.append(f"| {r['label']} | {r['rhythm_diversity']} "
                        f"| {r['repetition_ratio']} | {r['pitch_hist_distance']} "
                        f"| {', '.join(extras)} |")
    (OUTPUT_DIR / "evaluation_summary.md").write_text("\n".join(md_lines))
    print("\n".join(md_lines))

    # --- composite training-progress figure ----------------------------
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    ax[0,0].plot(ae_loss_hist,  lw=2, color="#1f77b4"); ax[0,0].set_title("Task 1 AE loss")
    ax[0,1].plot(vae_loss_hist, lw=2, color="#ff7f0e"); ax[0,1].set_title("Task 2 VAE total loss")
    ax[1,0].plot(ppl_hist,      lw=2, color="#9467bd"); ax[1,0].set_title("Task 3 perplexity")
    ax[1,1].plot(rl_reward_hist,lw=2, color="#2ca02c"); ax[1,1].set_title("Task 4 mean reward")
    for a in ax.ravel():
        a.set_xlabel("epoch / step"); a.grid(alpha=0.3)
    fig.suptitle("Training dynamics across all four tasks")
    plt.tight_layout(); plt.savefig(PLOT_DIR / "all_tasks_training.png", dpi=150)
    plt.close()

    print(f"\n>>> all artefacts saved under {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
