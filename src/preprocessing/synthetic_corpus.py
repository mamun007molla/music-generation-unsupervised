"""
Synthetic-MIDI generator
------------------------
Because the MAESTRO / Lakh / Groove datasets are many gigabytes, this script
produces a small *surrogate* corpus that captures genre-specific statistics
(scale, tempo, rhythmic density, chord motion).  It is enough to train the
small demo models in this project and, critically, every pipeline stage
(parsing -> training -> generation -> evaluation) runs on *real* MIDI files.

When the student has downloaded a real dataset, they simply drop the `.mid`
files into `data/raw_midi/<genre>/` and delete this synthesiser.
"""
from __future__ import annotations
import numpy as np
import pretty_midi
from pathlib import Path

from src.config import GENRES, RAW_MIDI_DIR, SEED

# ---- genre templates ------------------------------------------------------
#  scale  : list of semitone offsets from tonic
#  tempo  : range (bpm)
#  rhythm : probability of a note on each 16-th step
#  program: General-MIDI instrument to write into the file
GENRE_TEMPLATES = {
    "classical":  dict(scale=[0, 2, 4, 5, 7, 9, 11],          # major
                       tempo=(70, 110), rhythm=0.55, program=0,  # piano
                       tonic_choices=[60, 62, 65, 67]),
    "jazz":       dict(scale=[0, 2, 3, 5, 7, 9, 10],          # dorian
                       tempo=(100, 160), rhythm=0.70, program=32, # acoustic bass
                       tonic_choices=[58, 60, 63, 65]),
    "rock":       dict(scale=[0, 3, 5, 6, 7, 10],             # blues
                       tempo=(110, 150), rhythm=0.60, program=29, # overdriven guitar
                       tonic_choices=[52, 55, 57, 59]),
    "pop":        dict(scale=[0, 2, 4, 5, 7, 9, 11],          # major
                       tempo=(90, 130), rhythm=0.65, program=80,  # lead synth
                       tonic_choices=[60, 62, 64, 67]),
    "electronic": dict(scale=[0, 2, 3, 5, 7, 8, 10],          # natural minor
                       tempo=(120, 150), rhythm=0.80, program=81, # saw lead
                       tonic_choices=[60, 63, 65, 67]),
}


def _make_one_piece(genre: str, rng: np.random.Generator,
                    n_bars: int = 8, steps_per_bar: int = 16) -> pretty_midi.PrettyMIDI:
    t       = GENRE_TEMPLATES[genre]
    tempo   = rng.integers(*t["tempo"])
    pm      = pretty_midi.PrettyMIDI(initial_tempo=float(tempo))
    inst    = pretty_midi.Instrument(program=t["program"])
    sec_per_beat = 60.0 / tempo
    step_sec = sec_per_beat / (steps_per_bar / 4)

    tonic = int(rng.choice(t["tonic_choices"]))
    scale = t["scale"]

    current_pitch = tonic
    step          = 0
    total_steps   = n_bars * steps_per_bar

    while step < total_steps:
        if rng.random() < t["rhythm"]:
            # pick a scale degree, but restricted to a small step from current pitch
            degree_shift = rng.choice([-2, -1, -1, 0, 1, 1, 2])
            new_pitch    = tonic + rng.choice(scale) + 12 * rng.choice([0, 0, 1])
            current_pitch = int(np.clip(new_pitch, 36, 84))

            duration_steps = int(rng.choice([1, 1, 2, 2, 4]))
            start = step     * step_sec
            end   = (step + duration_steps) * step_sec
            velocity = int(rng.integers(70, 110))
            inst.notes.append(pretty_midi.Note(
                velocity=velocity, pitch=current_pitch,
                start=start, end=end))
            step += duration_steps
        else:
            step += 1                                   # rest
    pm.instruments.append(inst)
    return pm


def build_corpus(n_per_genre: int = 20, out_root: Path = RAW_MIDI_DIR):
    rng = np.random.default_rng(SEED)
    out_root.mkdir(parents=True, exist_ok=True)
    summary = {}
    for genre in GENRES:
        g_dir = out_root / genre
        g_dir.mkdir(exist_ok=True)
        for i in range(n_per_genre):
            pm   = _make_one_piece(genre, rng)
            path = g_dir / f"{genre}_{i:03d}.mid"
            pm.write(str(path))
        summary[genre] = n_per_genre
        print(f"  [corpus]  {genre:<11} -> {n_per_genre} files  ({g_dir})")
    return summary


if __name__ == "__main__":
    build_corpus()
