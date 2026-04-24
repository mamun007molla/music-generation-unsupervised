# Unsupervised Neural Networks for Multi-Genre Music Generation

**Course:** CSE425 / EEE474 — Neural Networks (Spring 2026)
**Deadline:** 10 April 2026

This repository contains a self-contained implementation of four
unsupervised symbolic-music generators:

| Task | Model                                      | Genre scope          | Final metric |
|:----:|--------------------------------------------|----------------------|:------------:|
|  1   | LSTM Autoencoder                           | Classical only       | Recon loss 1.93  |
|  2   | β-Variational Autoencoder (genre-conditioned) | 5 genres          | Total loss 2.64, KL ≈ 2.0 |
|  3   | Transformer decoder                        | 5 genres             | Perplexity 5.96  |
|  4   | RLHF-tuned Transformer (REINFORCE + survey reward) | 5 genres    | Reward 0.56 → 0.66 |

Plus two non-neural baselines: a **random-note generator** and a
**first-order Markov chain**.

Every model is implemented in pure **NumPy + Autograd**, so no GPU (and no
PyTorch / TensorFlow) is required.  The code path is structured so that
replacing the NumPy forward passes with PyTorch layers is a mechanical edit.

---

## Quick start

```bash
# 1. create a virtual env (optional but recommended)
python3 -m venv venv && source venv/bin/activate

# 2. install dependencies
pip install -r requirements.txt

# 3. run the full pipeline
#    (builds the synthetic corpus if data/raw_midi is empty, trains all four
#     tasks, generates 48 MIDI outputs, writes every plot and the metric table)
python3 run_all.py
```

End-to-end wall-clock on a mid-range CPU: **≈ 2 minutes**.

### Per-task commands

```bash
python3 -m src.training.train_ae            # Task 1
python3 -m src.training.train_vae           # Task 2
python3 -m src.training.train_transformer   # Task 3
python3 -m src.training.train_rlhf          # Task 4  (needs Task 3 first)
```

---

## Repository layout

```
music-generation-unsupervised/
├─ README.md
├─ requirements.txt
├─ run_all.py                     ← end-to-end driver
├─ data/
│   ├─ raw_midi/<genre>/*.mid     ← input corpus (100 files, 5 genres)
│   └─ processed/                 ← optional pre-tokenised cache
├─ src/
│   ├─ config.py                  ← all hyperparams + paths
│   ├─ preprocessing/
│   │   ├─ midi_parser.py
│   │   ├─ tokenizer.py
│   │   ├─ piano_roll.py
│   │   ├─ loader.py
│   │   └─ synthetic_corpus.py    ← fallback MIDI generator
│   ├─ models/
│   │   ├─ autoencoder.py         ← Task 1 LSTM-AE
│   │   ├─ vae.py                 ← Task 2 β-VAE
│   │   ├─ transformer.py         ← Task 3 Transformer
│   │   ├─ rlhf.py                ← Task 4 REINFORCE + reward model
│   │   └─ baselines.py           ← random + Markov baselines
│   ├─ training/
│   │   ├─ optim.py               ← Adam
│   │   ├─ train_ae.py
│   │   ├─ train_vae.py
│   │   ├─ train_transformer.py
│   │   └─ train_rlhf.py
│   ├─ generation/
│   │   └─ midi_export.py
│   └─ evaluation/
│       └─ metrics.py             ← pitch-hist, rhythm div, repetition, PPL
├─ outputs/
│   ├─ generated_midis/*.mid      ← 48 generated files
│   ├─ plots/*.png                ← training & evaluation figures
│   ├─ survey_results/survey_responses.csv
│   ├─ evaluation_summary.json
│   └─ evaluation_summary.md
└─ report/
    ├─ final_report.tex           ← IEEE conference template
    ├─ final_report.pdf           ← compiled output
    └─ references.bib
```

---

## Using a real MIDI dataset

The project works out-of-the-box with a 100-file synthetic corpus
(generated automatically in `src/preprocessing/synthetic_corpus.py`) so that
the pipeline is reproducible without a multi-gigabyte download.

To use a real dataset:

1. Download one of

   * [MAESTRO](https://magenta.tensorflow.org/datasets/maestro)
   * [Lakh MIDI](https://colinraffel.com/projects/lmd/)
   * [Groove MIDI](https://magenta.tensorflow.org/datasets/groove)

2. Drop the `.mid` / `.midi` files into `data/raw_midi/<genre>/`.
3. Delete (or skip) the synthetic-corpus build step — the pipeline picks up
   whatever is already on disk.

---

## RLHF / listener survey — honesty note

Task 4 requires a human-listening survey.  The file
`outputs/survey_results/survey_responses.csv` shipped with this repo was
produced by **ten simulated listeners** whose scores are a noisy linear
combination of three musicality features.  This is clearly disclosed in the
report (Section VI‑A).  To submit the project as described in the rubric,
**replace the CSV with real scores from ≥10 human participants** who listen
to the files under `outputs/generated_midis/` — no other code changes are
needed.

---

## Evaluation metrics

| Metric                     | Definition                                                             | Direction  |
|----------------------------|------------------------------------------------------------------------|------------|
| Pitch histogram distance   | mean pair-wise L1 distance between 12-bin pitch-class histograms       | higher ⇒ more diverse across samples |
| Rhythm diversity           | #unique note durations / #total notes                                  | higher ⇒ better   |
| Repetition ratio           | 1 − (#unique 4-grams / #4-grams)                                       | above noise floor ⇒ structure; too high ⇒ loops |
| Perplexity                 | exp(cross-entropy) — only Task 3                                       | lower ⇒ better   |
| Mean listener score        | 1–5 Likert from the survey                                             | higher ⇒ better   |

All numbers are written to `outputs/evaluation_summary.{json,md}`.

---

## Reproducibility

A single global seed (`SEED = 42` in `src/config.py`) governs
corpus generation, parameter init and all mini-batch shuffles.  Re-running
`run_all.py` reproduces every number reported in the paper up to the
small non-determinism introduced by Python’s hash randomisation for string
keys (set `PYTHONHASHSEED=0` to eliminate even that).

---

## License

Course project — educational use.
