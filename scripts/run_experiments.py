"""Run every experiment in the project and write one JSON file per run.

Stages
  prepare   download RAVDESS (+ CREMA-D) and cache 16 kHz trimmed audio
  cnn       MFCC-CNN baseline            RAVDESS, actor + random split
  frozen    frozen WavLM + layer-weighted head      "
  finetune  fine-tuned WavLM                        "
  cremad    cross-corpus: WavLM trained on CREMA-D (speaker-independent),
            tested on unseen CREMA-D speakers and on all of RAVDESS

Runs that already have a result file are skipped, so the script can be
re-launched after a Colab disconnect. Test sets are only scored once, with the
checkpoint chosen on validation UA.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from ser import data as D
from ser.models import SSL_MODEL, mfcc_features
from ser.train import (CNNTrainer, FinetuneTrainer, HeadTrainer, extract_pooled, metrics, restricted_argmax)

RAV_SHARED = [D.RAVDESS_EMOTIONS.index(e) for e in D.SHARED_EMOTIONS]


class Workspace:
    def __init__(self, args):
        self.args = args
        self.root = Path(args.data_root)
        self.cache = self.root / "cache"
        self.out = Path(args.out)
        (self.out / "runs").mkdir(parents=True, exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.slug = args.model_name.split("/")[-1]
        self._pooled = {}
        self._encoder = None
        self.quick_take = (lambda idx, k: np.sort(np.random.default_rng(0).permutation(idx)[:k])) if args.quick else None

    # ------------------------------------------------------------ data
    def ravdess(self):
        if not hasattr(self, "rav"):
            self.rav = D.ravdess_metadata(D.download_ravdess(self.root))
            self.rav_y = self.rav.emotion.map(D.RAVDESS_EMOTIONS.index).to_numpy()
            self.rav_waves = D.load_waves(self.rav.path.tolist(), self.cache / "ravdess_waves.pkl")
            shared = self.rav.emotion.isin(D.SHARED_EMOTIONS).to_numpy()
            self.rav6_idx = np.flatnonzero(shared)
            self.rav6_y = self.rav.emotion[shared].map(D.SHARED_EMOTIONS.index).to_numpy()
        return self.rav

    def cremad(self):
        if not hasattr(self, "crem"):
            root = D.download_cremad(self.root, limit=300 if self.args.quick else None)
            self.crem = D.cremad_metadata(root)
            self.crem_y = self.crem.emotion.map(D.SHARED_EMOTIONS.index).to_numpy()
            name = "cremad_waves_quick.pkl" if self.args.quick else "cremad_waves.pkl"
            self.crem_waves = D.load_waves(self.crem.path.tolist(), self.cache / name)
            self.crem_split = D.cremad_split(self.crem)
        return self.crem

    def split(self, kind):
        s = D.ravdess_split(self.ravdess(), kind) if kind != "cremad" else self.crem_split
        if self.quick_take:
            s = {k: self.quick_take(v, {"train": 96, "val": 24, "test": 32}[k]) for k, v in s.items()}
        return s

    def use_cremad(self):
        return not self.args.no_cremad

    # ------------------------------------------------------------ SSL features
    def encoder(self):
        if self._encoder is None:
            from transformers import AutoModel
            self._encoder = AutoModel.from_pretrained(self.args.model_name)
        return self._encoder

    def pooled(self, dataset):
        """(N, L, H) cached frozen features for every clip of a dataset."""
        if dataset not in self._pooled:
            suffix = "_quick" if self.args.quick else ""
            f = self.cache / f"{dataset}_pooled_{self.slug}{suffix}.npy"
            waves = self.rav_waves if dataset == "ravdess" else self.crem_waves
            if f.exists() and len(np.load(f, mmap_mode="r")) == len(waves):
                self._pooled[dataset] = np.load(f)
            else:
                print(f"Extracting frozen {self.slug} features for {len(waves)} {dataset} clips...", flush=True)
                if self.args.quick:  # only the clips a quick run touches
                    need = set()
                    for kind in (["actor", "random"] if dataset == "ravdess" else ["cremad"]):
                        for v in self.split(kind).values():
                            need.update(v.tolist())
                    if dataset == "ravdess":
                        need.update(self.rav6_idx[:32].tolist())
                    X = np.zeros((len(waves), self.encoder().config.num_hidden_layers + 1,
                                  self.encoder().config.hidden_size), np.float16)
                    need = sorted(need)
                    X[need] = extract_pooled(self.encoder(), [waves[i] for i in need], self.device)
                else:
                    X = extract_pooled(self.encoder(), waves, self.device)
                f.parent.mkdir(parents=True, exist_ok=True)
                np.save(f, X)
                self._pooled[dataset] = X
        return self._pooled[dataset]

    # ------------------------------------------------------------ bookkeeping
    def path(self, name):
        return self.out / "runs" / f"{name}.json"

    def done(self, name):
        return self.path(name).exists()

    def save(self, name, rec):
        rec = dict(name=name, **rec, quick=self.args.quick, device=self.device,
                   finished=time.strftime("%Y-%m-%d %H:%M:%S"))
        self.path(name).write_text(json.dumps(rec, indent=1))
        c = " ".join(f"{k}={v['ua']:.3f}" for k, v in rec.get("cross", {}).items())
        print(f"[done] {name}: val UA {rec['val_ua']:.3f} | test UA {rec['test']['ua']:.3f} "
              f"WA {rec['test']['wa']:.3f} {c}", flush=True)
        # The full record also goes to the log, so a notebook's saved output keeps every
        # result even if the machine that wrote results/runs/ is recycled.
        print("RESULT_JSON " + json.dumps(rec, separators=(",", ":")), flush=True)


def cross_eval_ravdess_model(ws, logits_fn):
    """8-class RAVDESS model scored on unseen CREMA-D speakers (6 shared emotions)."""
    if not ws.use_cremad():
        return {}
    ws.cremad()
    te = ws.split("cremad")["test"]
    pred = restricted_argmax(logits_fn(te), RAV_SHARED)
    return {"cremad_test": metrics(ws.crem_y[te], pred, 6)}


# ================================================================ stages
def stage_prepare(ws):
    df = ws.ravdess()
    print(f"RAVDESS: {len(df)} clips, {df.speaker.nunique()} actors")
    for kind in ["actor", "random"]:
        s = ws.split(kind)
        print(f"  {kind:6s} split sizes: " + ", ".join(f"{k}={len(v)}" for k, v in s.items()))
    if ws.use_cremad():
        c = ws.cremad()
        s = ws.split("cremad")
        print(f"CREMA-D: {len(c)} clips, {c.speaker.nunique()} speakers; "
              + ", ".join(f"{k}={len(v)} clips/{c.speaker[v].nunique()} spk" for k, v in s.items()))


def stage_cnn(ws):
    ws.ravdess()
    f = ws.cache / "ravdess_mfcc.pkl"
    if f.exists():
        feats = pickle.loads(f.read_bytes())
    else:
        print("Computing MFCCs...", flush=True)
        feats = [mfcc_features(w) for w in ws.rav_waves]
        f.write_bytes(pickle.dumps(feats))
    crem_feats = None
    for kind in ws.args.splits:
        for seed in ws.args.seeds:
            name = f"ravdess_{kind}_mfcc_cnn_seed{seed}"
            if ws.done(name):
                continue
            s, t0 = ws.split(kind), time.time()
            tr = CNNTrainer(8, ws.device, epochs=3 if ws.args.quick else 60)
            info = tr.fit([feats[i] for i in s["train"]], ws.rav_y[s["train"]],
                          [feats[i] for i in s["val"]], ws.rav_y[s["val"]], seed)
            test = metrics(ws.rav_y[s["test"]], tr.logits([feats[i] for i in s["test"]]).argmax(1), 8)
            cross = {}
            if kind == "actor":
                if ws.use_cremad() and crem_feats is None:
                    ws.cremad()
                    crem_feats = {i: mfcc_features(ws.crem_waves[i]) for i in ws.split("cremad")["test"]}
                cross = cross_eval_ravdess_model(ws, lambda idx: tr.logits([crem_feats[i] for i in idx]))
            ws.save(name, dict(dataset="ravdess", split=kind, model="mfcc_cnn", seed=seed, classes=D.RAVDESS_EMOTIONS,
                               sizes={k: len(v) for k, v in s.items()}, test=test, cross=cross,
                               train_seconds=time.time() - t0, **info))


def stage_frozen(ws):
    ws.ravdess()
    X = ws.pooled("ravdess")
    L, H = X.shape[1:]
    for kind in ws.args.splits:
        for seed in ws.args.seeds:
            name = f"ravdess_{kind}_wavlm_frozen_seed{seed}"
            if ws.done(name):
                continue
            s, t0 = ws.split(kind), time.time()
            tr = HeadTrainer(L, H, 8, ws.device, epochs=5 if ws.args.quick else 150)
            info = tr.fit(X[s["train"]], ws.rav_y[s["train"]], X[s["val"]], ws.rav_y[s["val"]], seed)
            test = metrics(ws.rav_y[s["test"]], tr.logits(X[s["test"]]).argmax(1), 8)
            cross = {}
            if kind == "actor":
                cross = cross_eval_ravdess_model(ws, lambda idx: tr.logits(ws.pooled("cremad")[idx]))
                if seed == ws.args.seeds[0]:
                    ck = ws.out / "checkpoints" / "wavlm_frozen_head.pt"
                    ck.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(dict(mode="frozen", model_name=ws.args.model_name, classes=D.RAVDESS_EMOTIONS,
                                    state_dict=tr.model.state_dict(), test_ua=test["ua"], split="actor",
                                    seed=seed), ck)
            ws.save(name, dict(dataset="ravdess", split=kind, model="wavlm_frozen", seed=seed,
                               classes=D.RAVDESS_EMOTIONS, sizes={k: len(v) for k, v in s.items()}, test=test,
                               cross=cross, train_seconds=time.time() - t0, **info))


def stage_finetune(ws):
    ws.ravdess()
    W = ws.rav_waves
    for kind in ws.args.splits:
        for seed in ws.args.ft_seeds:
            name = f"ravdess_{kind}_wavlm_finetune{ws.args.tag}_seed{seed}"
            if ws.done(name):
                continue
            print(f"== {name}", flush=True)
            s, t0 = ws.split(kind), time.time()
            tr = FinetuneTrainer(8, ws.device, ws.args.model_name, epochs=1 if ws.args.quick else ws.args.ft_epochs,
                                 lr_encoder=ws.args.ft_lr, freeze_epochs=ws.args.ft_freeze_epochs)
            info = tr.fit([W[i] for i in s["train"]], ws.rav_y[s["train"]],
                          [W[i] for i in s["val"]], ws.rav_y[s["val"]], seed)
            test = metrics(ws.rav_y[s["test"]], tr.logits([W[i] for i in s["test"]]).argmax(1), 8)
            cross = {}
            if kind == "actor":
                cross = cross_eval_ravdess_model(ws, lambda idx: tr.logits([ws.crem_waves[i] for i in idx]))
                if ws.args.save_ft_checkpoint and seed == ws.args.ft_seeds[0]:
                    ck = ws.out / "checkpoints" / "wavlm_finetuned.pt"
                    ck.parent.mkdir(parents=True, exist_ok=True)
                    sd = {k: (v.half() if v.is_floating_point() else v) for k, v in tr.model.state_dict().items()}
                    torch.save(dict(mode="finetune", model_name=ws.args.model_name, classes=D.RAVDESS_EMOTIONS,
                                    state_dict=sd, test_ua=test["ua"], split="actor", seed=seed), ck)
            ws.save(name, dict(dataset="ravdess", split=kind, model="wavlm_finetune" + ws.args.tag, seed=seed,
                               hparams=dict(epochs=ws.args.ft_epochs, lr_encoder=ws.args.ft_lr,
                                            freeze_epochs=ws.args.ft_freeze_epochs),
                               classes=D.RAVDESS_EMOTIONS, sizes={k: len(v) for k, v in s.items()}, test=test,
                               cross=cross, train_seconds=time.time() - t0, **info))
            del tr
            torch.cuda.empty_cache()


def stage_cremad(ws):
    """Train on CREMA-D (speaker-independent), test on unseen speakers and on RAVDESS."""
    ws.ravdess()
    ws.cremad()
    s = ws.split("cremad")
    y = ws.crem_y
    rav6 = ws.rav6_idx[:32] if ws.args.quick else ws.rav6_idx
    ry = ws.rav6_y[: len(rav6)]
    sizes = {k: len(v) for k, v in s.items()}

    X, RX = ws.pooled("cremad"), ws.pooled("ravdess")
    for seed in ws.args.seeds:
        name = f"cremad_speaker_wavlm_frozen_seed{seed}"
        if ws.done(name):
            continue
        t0 = time.time()
        tr = HeadTrainer(X.shape[1], X.shape[2], 6, ws.device, epochs=5 if ws.args.quick else 150)
        info = tr.fit(X[s["train"]], y[s["train"]], X[s["val"]], y[s["val"]], seed)
        test = metrics(y[s["test"]], tr.logits(X[s["test"]]).argmax(1), 6)
        cross = {"ravdess_all_actors": metrics(ry, tr.logits(RX[rav6]).argmax(1), 6)}
        ws.save(name, dict(dataset="cremad", split="speaker", model="wavlm_frozen", seed=seed,
                           classes=D.SHARED_EMOTIONS, sizes=sizes, test=test, cross=cross,
                           train_seconds=time.time() - t0, **info))

    Wc = ws.crem_waves
    for seed in ws.args.cremad_ft_seeds:
        name = f"cremad_speaker_wavlm_finetune{ws.args.tag}_seed{seed}"
        if ws.done(name):
            continue
        print(f"== {name}", flush=True)
        t0 = time.time()
        tr = FinetuneTrainer(6, ws.device, ws.args.model_name, epochs=1 if ws.args.quick else ws.args.cremad_ft_epochs,
                             lr_encoder=ws.args.ft_lr, freeze_epochs=ws.args.ft_freeze_epochs)
        info = tr.fit([Wc[i] for i in s["train"]], y[s["train"]], [Wc[i] for i in s["val"]], y[s["val"]], seed)
        test = metrics(y[s["test"]], tr.logits([Wc[i] for i in s["test"]]).argmax(1), 6)
        cross = {"ravdess_all_actors": metrics(ry, tr.logits([ws.rav_waves[i] for i in rav6]).argmax(1), 6)}
        ws.save(name, dict(dataset="cremad", split="speaker", model="wavlm_finetune" + ws.args.tag, seed=seed,
                           classes=D.SHARED_EMOTIONS, sizes=sizes, test=test, cross=cross,
                           train_seconds=time.time() - t0, **info))
        del tr
        torch.cuda.empty_cache()


STAGES = dict(prepare=stage_prepare, cnn=stage_cnn, frozen=stage_frozen, finetune=stage_finetune,
              cremad=stage_cremad)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", nargs="+", default=list(STAGES), choices=list(STAGES))
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--splits", nargs="+", default=["actor", "random"], choices=["actor", "random"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="seeds for CNN and frozen runs")
    ap.add_argument("--ft-seeds", nargs="+", type=int, default=[0, 1, 2], help="seeds for RAVDESS fine-tuning")
    ap.add_argument("--cremad-ft-seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--model-name", default=SSL_MODEL)
    # Fine-tuning defaults were chosen on validation UA only (see docs/finetune_search.md).
    ap.add_argument("--ft-epochs", type=int, default=12)
    ap.add_argument("--ft-lr", type=float, default=3e-5, help="encoder learning rate")
    ap.add_argument("--ft-freeze-epochs", type=int, default=3, help="head-only epochs before unfreezing")
    ap.add_argument("--tag", default="", help="suffix for run names (keeps search runs separate)")
    ap.add_argument("--cremad-ft-epochs", type=int, default=8)
    ap.add_argument("--no-cremad", action="store_true", help="skip CREMA-D download and cross-corpus tests")
    ap.add_argument("--save-ft-checkpoint", action="store_true")
    ap.add_argument("--quick", action="store_true", help="tiny subsets and 1-5 epochs, for smoke tests only")
    args = ap.parse_args()
    ws = Workspace(args)
    print(f"device={ws.device} torch={torch.__version__}", flush=True)
    for st in args.stages:
        t0 = time.time()
        print(f"\n######## stage: {st}", flush=True)
        STAGES[st](ws)
        print(f"######## stage {st} finished in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
