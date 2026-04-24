"""
Global configuration for the Unsupervised Multi-Genre Music Generation project.

All hyperparameters, paths and constants are centralised here so that every
training / evaluation script reads exactly the same settings.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
DATA_DIR       = PROJECT_ROOT / "data"
RAW_MIDI_DIR   = DATA_DIR / "raw_midi"
PROCESSED_DIR  = DATA_DIR / "processed"
SPLIT_DIR      = DATA_DIR / "train_test_split"
OUTPUT_DIR     = PROJECT_ROOT / "outputs"
MIDI_OUT_DIR   = OUTPUT_DIR / "generated_midis"
PLOT_DIR       = OUTPUT_DIR / "plots"
SURVEY_DIR     = OUTPUT_DIR / "survey_results"

# ---------------------------------------------------------------------------
# Musical / tokenisation constants
# ---------------------------------------------------------------------------
PITCH_MIN      = 21          # A0  (lowest MIDI note we encode)
PITCH_MAX      = 108         # C8  (highest MIDI note we encode)
NUM_PITCHES    = PITCH_MAX - PITCH_MIN + 1   # 88 piano keys

STEPS_PER_BAR  = 16          # 16-th note resolution
BARS_PER_SEQ   = 4           # each training segment covers 4 bars
SEQ_LEN        = STEPS_PER_BAR * BARS_PER_SEQ    # 64 time-steps per sample

VOCAB_SIZE     = NUM_PITCHES + 2     # + <PAD> and <REST>
PAD_TOKEN      = 0
REST_TOKEN     = 1
NOTE_OFFSET    = 2                    # note tokens start at 2

GENRES         = ["classical", "jazz", "rock", "pop", "electronic"]
NUM_GENRES     = len(GENRES)

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
# Task 1 – LSTM Autoencoder
AE_HIDDEN      = 64
AE_LATENT      = 32
AE_EPOCHS      = 25
AE_LR          = 5e-3
AE_BATCH       = 16

# Task 2 – Variational Autoencoder
VAE_HIDDEN     = 64
VAE_LATENT     = 32
VAE_EPOCHS     = 30
VAE_LR         = 3e-3
VAE_BATCH      = 16
VAE_BETA       = 0.05         # KL weight (β-VAE)

# Task 3 – Transformer
TR_D_MODEL     = 48
TR_N_HEADS     = 4
TR_N_LAYERS    = 2
TR_EPOCHS      = 10
TR_LR          = 2e-3
TR_BATCH       = 16

# Task 4 – RLHF policy gradient
RL_STEPS       = 20
RL_LR          = 3e-4
RL_SAMPLES_PER_STEP = 4

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
