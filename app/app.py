"""Gradio demo: record your voice, see which of 8 emotions the model hears.

Runs from the repo (`python app/app.py`) or as a Hugging Face Space built with
`python tools/export_space.py`.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent):  # flat Space checkout or repo layout
    if (p / "ser").is_dir():
        sys.path.insert(0, str(p))
        break

import gradio as gr
import numpy as np
import torch

from ser.data import SR, preprocess_wave
from ser.models import SSLClassifier

CANDIDATES = [HERE / "checkpoints" / "wavlm_finetuned.pt", HERE / "checkpoints" / "wavlm_frozen_head.pt",
              HERE.parent / "results" / "checkpoints" / "wavlm_finetuned.pt",
              HERE.parent / "results" / "checkpoints" / "wavlm_frozen_head.pt"]
CKPT = Path(os.environ.get("SER_CHECKPOINT") or next((c for c in CANDIDATES if c.exists()), CANDIDATES[1]))
MAX_SECONDS = 10


def load_model(path: Path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = SSLClassifier(len(ck["classes"]), ck["model_name"])
    if ck["mode"] == "frozen":  # pretrained encoder from the Hub + our small trained head
        model.head.load_state_dict(ck["state_dict"])
    else:
        model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ck["state_dict"].items()})
    return model.eval(), ck


model, ck = load_model(CKPT)
CLASSES = ck["classes"]
KIND = "fine-tuned" if ck["mode"] == "finetune" else "frozen encoder + trained head"


def predict(audio):
    if audio is None:
        return None, "Record or upload a clip first."
    sr, y = audio
    y = np.asarray(y)
    if np.issubdtype(y.dtype, np.integer):
        y = y.astype(np.float32) / np.iinfo(y.dtype).max
    wave = preprocess_wave(y, sr)[: MAX_SECONDS * SR]
    if len(wave) < 0.3 * SR:
        return None, "That clip is too short. Try a full sentence (1–5 seconds)."
    with torch.inference_mode():
        probs = torch.softmax(model(torch.from_numpy(wave)[None]), -1)[0].numpy()
    return {c: float(p) for c, p in zip(CLASSES, probs)}, f"Analysed {len(wave) / SR:.1f} s of speech."


DESCRIPTION = f"""
**A model that recognises 8 emotions from voice, tested only on actors it never heard.**

WavLM Base+ ({KIND}) trained on RAVDESS actors 1–18, model selection on actors 19–20.
Unweighted accuracy on the 4 held-out actors (21–24): **{100 * ck['test_ua']:.1f}%** (chance 12.5%).

Try saying *"Kids are talking by the door"* or *"Dogs are sitting by the door"* the way you would if you were
happy, angry, sad, afraid... RAVDESS is acted North-American English speech, so expect the model to be less
reliable on other accents, microphones and on natural (non-acted) emotion.
"""

with gr.Blocks(title="Speech Emotion Recognition") as demo:
    gr.Markdown("# Speech Emotion Recognition")
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column():
            audio = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Your voice")
            button = gr.Button("Recognise emotion", variant="primary")
        with gr.Column():
            label = gr.Label(num_top_classes=len(CLASSES), label="Predicted emotion")
            info = gr.Markdown()
    button.click(predict, inputs=audio, outputs=[label, info])
    audio.stop_recording(predict, inputs=audio, outputs=[label, info])
    gr.Markdown("Data: RAVDESS (Livingstone & Russo, 2018), CC BY-NC-SA 4.0. Non-commercial research demo. "
                "Audio is processed in memory and not stored.")

if __name__ == "__main__":
    demo.launch()
