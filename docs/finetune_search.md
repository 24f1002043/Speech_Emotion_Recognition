# Choosing the fine-tuning hyper-parameters

The first fine-tuning attempt was *worse* than the frozen model: with encoder LR 5e-5 and no warm-up, the
training loss reached ~0.00 by epoch 10 while validation UA drifted around 0.65–0.73. With 1,080 training clips
and a 94M-parameter encoder, that is straightforward overfitting.

Four configurations were therefore compared on the **actor split, seed 0, validation set only**
(actors 19–20). The test actors (21–24) played no part in this choice; test numbers are shown here only
afterwards, for the record.

| Config | Encoder LR | Head-only warm-up epochs | Epochs | **Val UA** | Test UA (not used for the choice) |
|---|---|---|---|---|---|
| A | 5e-5 | 0 | 20 | 0.734 | 0.566 |
| B | 1e-5 | 1 | 12 | 0.742 | 0.734 |
| C | 1e-5 | 2 | 20 | 0.773 | 0.770 |
| **D (chosen)** | **3e-5** | **3** | **12** | **0.781** | 0.812 |

Config D is the default in `scripts/run_experiments.py` (`--ft-lr 3e-5 --ft-freeze-epochs 3 --ft-epochs 12`)
and is what every reported fine-tuning run uses, for both splits and the CREMA-D model.

Two things mattered: a **head-only warm-up** (the randomly initialised classifier is not allowed to push large
gradients into the pretrained encoder before it is itself sensible) and a **shorter schedule**, which limits how
far the encoder can drift from the pretrained weights on a small corpus.

Note also that config A, despite being poor on RAVDESS test actors (0.566), was the *best* of these on the
cross-corpus CREMA-D test (0.445 vs 0.310 for the frozen model). Aggressive fine-tuning moved the encoder away
from RAVDESS's 2-sentence idiosyncrasies, which hurt in-corpus accuracy but helped transfer. This is a
side observation from a single seed, not a controlled result.
