"""Assemble a self-contained Hugging Face Space in ./space (then upload it with `hf upload`)."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SPACE_README = """---
title: Speech Emotion Recognition
emoji: 🎙️
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
license: cc-by-nc-sa-4.0
short_description: 8-class speech emotion recognition, tested on unseen actors
---

# Speech Emotion Recognition (RAVDESS, actor-independent)

Record a sentence and see which of 8 emotions (neutral, calm, happy, sad, angry, fearful, disgust, surprised)
a WavLM Base+ model hears. The model never saw the test actors during training.

Code, training and full results: see the project repository.

Data: RAVDESS, Livingstone SR & Russo FA (2018), PLoS ONE 13(5): e0196391, CC BY-NC-SA 4.0.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None,
                    help="default: results/checkpoints/wavlm_finetuned.pt if present, else wavlm_frozen_head.pt")
    ap.add_argument("--out", default=str(ROOT / "space"))
    args = ap.parse_args()
    ckdir = ROOT / "results" / "checkpoints"
    ck = Path(args.checkpoint) if args.checkpoint else next(
        p for p in [ckdir / "wavlm_finetuned.pt", ckdir / "wavlm_frozen_head.pt"] if p.exists())
    out = Path(args.out)
    shutil.rmtree(out, ignore_errors=True)
    (out / "ser").mkdir(parents=True)
    (out / "checkpoints").mkdir()
    shutil.copy(ROOT / "app" / "app.py", out / "app.py")
    shutil.copy(ROOT / "app" / "requirements.txt", out / "requirements.txt")
    for f in ["__init__.py", "data.py", "models.py"]:
        shutil.copy(ROOT / "ser" / f, out / "ser" / f)
    shutil.copy(ck, out / "checkpoints" / ck.name)
    (out / "README.md").write_text(SPACE_README, encoding="utf-8")
    print(f"Space assembled in {out} using {ck.name}")


if __name__ == "__main__":
    main()
