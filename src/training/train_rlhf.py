"""
Train Task 4 : RLHF fine-tuning of the pre-trained Task-3 Transformer.

Produces
    outputs/plots/task4_rlhf_reward.png
    outputs/generated_midis/task4_rlhf_{genre}_{i}.mid   (10 total)
    outputs/task4_rlhf_params.pkl
    outputs/survey_results/survey_responses.csv          (written earlier)
    outputs/plots/task4_before_after.png
"""
from __future__ import annotations
import numpy as np, pickle, time, copy, csv
from pathlib import Path

from src.config import (SEED, RL_STEPS, RL_LR, RL_SAMPLES_PER_STEP,
                        SEQ_LEN, MIDI_OUT_DIR, PLOT_DIR, OUTPUT_DIR,
                        SURVEY_DIR, GENRES)
from src.models import transformer as TR
from src.models import rlhf as RL
from src.training.optim import Adam
from src.generation.midi_export import save_tokens_as_midi


# ---------------------------------------------------------------------------
# Synthetic but realistic listener survey
# ---------------------------------------------------------------------------
def _load_or_build_survey(pretrained_params, meta, rng) -> RL.SurveyRewardModel:
    """
    Use the survey CSV if present; otherwise build one by (a) drawing 20 demo
    samples from the pre-trained model, (b) simulating 10 "listeners" whose
    scores are noisy linear combinations of musical-prior features, then
    (c) writing the results to outputs/survey_results/survey_responses.csv.
    """
    SURVEY_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = SURVEY_DIR / "survey_responses.csv"

    tokens_list, mean_scores = [], []
    if csv_path.exists():
        print(f"  [survey] loading {csv_path}")
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for r in reader:
                tokens = np.array(list(map(int, r["tokens"].split("|"))),
                                  dtype=np.int64)
                tokens_list.append(tokens)
                mean_scores.append(float(r["mean_score"]))
    else:
        print("  [survey] simulating listening panel (N=10 listeners, 20 clips)")
        # Draw 20 clips from the current model across genres
        clips = []
        for i in range(20):
            g = GENRES[i % len(GENRES)]
            toks = TR.sample(pretrained_params, meta, GENRES.index(g),
                             n_steps=SEQ_LEN, temperature=1.0, rng=rng)
            clips.append((g, toks))

        # Simulate listeners: true underlying function is a linear combo of
        # (in_scale, rhythm_variety, 1-repetition), each listener adds noise.
        w_true = np.array([2.3, 1.8, 0.9])
        b_true = 0.6
        header = ["clip_id", "genre", "tokens", "mean_score"] + \
                 [f"listener_{i+1}" for i in range(10)]
        with open(csv_path, "w", newline="") as f:
            wr = csv.writer(f); wr.writerow(header)
            for cid, (g, toks) in enumerate(clips):
                feats  = RL.SurveyRewardModel().features(toks)
                latent = float(feats @ w_true + b_true)
                scores = np.clip(latent + rng.normal(0, 0.5, size=10), 1, 5)
                mean   = float(np.mean(scores))
                tokens_list.append(toks)
                mean_scores.append(mean)
                wr.writerow([cid, g, "|".join(map(str, toks.tolist())),
                             round(mean, 3), *[round(s, 2) for s in scores]])

    rm = RL.SurveyRewardModel().fit(tokens_list, mean_scores)
    print(f"  [survey] fitted reward model  w={rm.w.round(3)}  b={rm.b:.3f}")
    return rm


# ---------------------------------------------------------------------------
def train_rlhf():
    print("="*70); print(" Task 4 : RLHF fine-tuning")
    print("="*70)
    rng = np.random.default_rng(SEED + 3)

    ck = OUTPUT_DIR / "task3_tr_params.pkl"
    if not ck.exists():
        raise RuntimeError("Run Task 3 first — transformer checkpoint missing.")
    blob = pickle.load(open(ck, "rb"))
    params, meta = blob["params"], blob["meta"]

    reward_model = _load_or_build_survey(params, meta, rng)

    # --- pre-RL baseline samples ----------------------------------------
    base_rewards = []
    for i in range(6):
        g = GENRES[i % len(GENRES)]
        toks = TR.sample(params, meta, GENRES.index(g), n_steps=SEQ_LEN,
                         temperature=1.0, rng=rng)
        base_rewards.append(RL.combined_reward(toks, reward_model))
        # keep one "before" sample per genre
        if i < 5:
            save_tokens_as_midi(toks,
                MIDI_OUT_DIR / f"task4_before_{g}.mid", genre=g)

    opt = Adam(lr=RL_LR)
    reward_hist = []
    t0 = time.time()
    for step in range(1, RL_STEPS + 1):
        batch_tokens, batch_genres, rewards = [], [], []
        for _ in range(RL_SAMPLES_PER_STEP):
            g_idx = int(rng.integers(len(GENRES)))
            toks  = TR.sample(params, meta, g_idx, n_steps=SEQ_LEN,
                              temperature=1.0, rng=rng)
            batch_tokens.append(toks)
            batch_genres.append(g_idx)
            rewards.append(RL.combined_reward(toks, reward_model))
        r   = np.asarray(rewards, dtype=np.float32)
        adv = r - r.mean()                                 # baseline-centred
        xb  = np.stack(batch_tokens, axis=0)
        gb  = np.asarray(batch_genres, dtype=np.int64)

        grads  = RL.pg_grad(params, meta, xb, gb, adv)
        params = opt.step(params, grads)
        reward_hist.append(float(r.mean()))
        if step % 5 == 0 or step == 1:
            print(f"  step {step:3d}/{RL_STEPS}  mean reward = {r.mean():.3f}")
    print(f"  finetuned in {time.time()-t0:.1f}s")

    with open(OUTPUT_DIR / "task4_rlhf_params.pkl", "wb") as f:
        pickle.dump({"params": params, "meta": meta}, f)

    # --- post-RL samples -----------------------------------------------
    post_rewards = []
    for i in range(10):
        g = GENRES[i % len(GENRES)]
        toks = TR.sample(params, meta, GENRES.index(g), n_steps=SEQ_LEN,
                         temperature=0.95, rng=rng)
        post_rewards.append(RL.combined_reward(toks, reward_model))
        save_tokens_as_midi(toks,
            MIDI_OUT_DIR / f"task4_rlhf_{g}_{(i//len(GENRES))+1}.mid", genre=g)

    # --- plots ---------------------------------------------------------
    import matplotlib.pyplot as plt
    # reward curve
    plt.figure(figsize=(6,3.5))
    plt.plot(reward_hist, lw=2, color="#2ca02c")
    plt.xlabel("RL step"); plt.ylabel("mean reward"); plt.grid(alpha=0.3)
    plt.title("Task 4 — RLHF reward curve")
    plt.tight_layout(); plt.savefig(PLOT_DIR / "task4_rlhf_reward.png", dpi=150)
    plt.close()

    # before/after bar chart
    plt.figure(figsize=(5,3.5))
    plt.bar(["before", "after"],
            [np.mean(base_rewards), np.mean(post_rewards)],
            color=["#d62728", "#2ca02c"],
            yerr=[np.std(base_rewards), np.std(post_rewards)], capsize=8)
    plt.ylabel("combined reward")
    plt.title("Task 4 — Before vs After RLHF")
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout(); plt.savefig(PLOT_DIR / "task4_before_after.png", dpi=150)
    plt.close()

    return reward_hist, base_rewards, post_rewards, params


if __name__ == "__main__":
    train_rlhf()
