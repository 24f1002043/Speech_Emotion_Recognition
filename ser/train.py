"""Training / evaluation loops. Model selection uses validation UA only."""
from __future__ import annotations

import copy
import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

from .data import SR
from .models import MFCCCNN, SSLClassifier


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def metrics(y_true, y_pred, n_classes: int) -> dict:
    """UA = unweighted accuracy = mean per-class recall; WA = plain accuracy."""
    return dict(ua=float(balanced_accuracy_score(y_true, y_pred)),
                wa=float(accuracy_score(y_true, y_pred)),
                macro_f1=float(f1_score(y_true, y_pred, average="macro")),
                confusion=confusion_matrix(y_true, y_pred, labels=list(range(n_classes))).tolist())


def class_weights(y: np.ndarray, n_classes: int, device) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    w = counts.sum() / (n_classes * np.maximum(counts, 1))
    return torch.tensor(w, device=device)


def restricted_argmax(logits: np.ndarray, keep: list[int]) -> np.ndarray:
    """Predict only among `keep` columns; returns positions within `keep`."""
    return logits[:, keep].argmax(1)


# ================================================================ MFCC-CNN
CNN_FRAMES = 350  # 3.5 s


def _crop_pad(x: np.ndarray, T: int, rng: np.random.Generator | None) -> np.ndarray:
    t = x.shape[-1]
    if t >= T:
        s = int(rng.integers(0, t - T + 1)) if rng is not None else (t - T) // 2
        return x[..., s:s + T]
    out = np.zeros((*x.shape[:-1], T), np.float32)
    s = int(rng.integers(0, T - t + 1)) if rng is not None else 0
    out[..., s:s + t] = x
    return out


def _spec_augment(x: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    B, _, F_, T = x.shape
    x = x.clone()
    for b in range(B):
        f = int(rng.integers(0, 7))
        f0 = int(rng.integers(0, F_ - f + 1))
        x[b, :, f0:f0 + f, :] = 0
        for _ in range(2):
            t = int(rng.integers(0, 41))
            t0 = int(rng.integers(0, T - t + 1))
            x[b, :, :, t0:t0 + t] = 0
    return x


class CNNTrainer:
    def __init__(self, n_classes: int, device, epochs: int = 60, patience: int = 15, batch_size: int = 32,
                 lr: float = 1e-3):
        self.n_classes, self.device, self.epochs, self.patience = n_classes, device, epochs, patience
        self.batch_size, self.lr = batch_size, lr

    def _norm(self, feats):
        return [(f - self.mean) / self.std for f in feats]

    def _batch(self, feats, idx, rng):
        return torch.from_numpy(np.stack([_crop_pad(feats[i], CNN_FRAMES, rng) for i in idx])).to(self.device)

    def logits(self, feats) -> np.ndarray:
        self.model.eval()
        feats, out = self._norm(feats), []
        with torch.no_grad():
            for s in range(0, len(feats), 64):
                out.append(self.model(self._batch(feats, range(s, min(s + 64, len(feats))), None)).cpu().numpy())
        return np.concatenate(out)

    def fit(self, tr_feats, tr_y, va_feats, va_y, seed: int) -> dict:
        set_seed(seed)
        rng = np.random.default_rng(seed)
        cat = np.concatenate(tr_feats, axis=-1)  # normalisation stats from training data only
        self.mean, self.std = cat.mean(-1, keepdims=True), cat.std(-1, keepdims=True) + 1e-5
        tr = self._norm(tr_feats)
        self.model = MFCCCNN(self.n_classes).to(self.device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights(tr_y, self.n_classes, self.device))
        y_t = torch.from_numpy(tr_y).to(self.device)
        best, best_state, best_epoch, bad, hist = -1.0, None, 0, 0, []
        for ep in range(self.epochs):
            self.model.train()
            perm = rng.permutation(len(tr))
            for s in range(0, len(perm), self.batch_size):
                idx = perm[s:s + self.batch_size]
                x = _spec_augment(self._batch(tr, idx, rng), rng)
                loss = loss_fn(self.model(x), y_t[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
            sched.step()
            va_ua = metrics(va_y, self.logits(va_feats).argmax(1), self.n_classes)["ua"]
            hist.append(va_ua)
            if va_ua > best:
                best, best_epoch, bad = va_ua, ep, 0
                best_state = copy.deepcopy(self.model.state_dict())
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.model.load_state_dict(best_state)
        return dict(val_ua=best, best_epoch=best_epoch, epochs_run=len(hist), val_history=hist)


# ================================================================ SSL: frozen
@torch.no_grad()
def extract_pooled(encoder, waves, device, max_sec: float = 10.0, log_every: int = 500) -> np.ndarray:
    """Per-layer time-averaged hidden states, one clip at a time (no padding)."""
    model = SSLClassifier(2, encoder=encoder).to(device).eval()
    use_amp = torch.device(device).type == "cuda"
    out = []
    for i, w in enumerate(waves):
        x = torch.from_numpy(w[: int(max_sec * SR)])[None].to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            out.append(model.pooled(x).squeeze(0).cpu().numpy().astype(np.float16))
        if log_every and (i + 1) % log_every == 0:
            print(f"  extracted {i + 1}/{len(waves)}", flush=True)
    return np.stack(out)


class HeadTrainer:
    def __init__(self, n_layers: int, hidden: int, n_classes: int, device, epochs: int = 150, patience: int = 25,
                 batch_size: int = 32, lr: float = 1e-3):
        from .models import LayerWeightedHead
        self.make = lambda: LayerWeightedHead(n_layers, hidden, n_classes)
        self.n_classes, self.device, self.epochs, self.patience = n_classes, device, epochs, patience
        self.batch_size, self.lr = batch_size, lr

    def logits(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.from_numpy(X).float().to(self.device)).cpu().numpy()

    def fit(self, Xtr, ytr, Xva, yva, seed: int) -> dict:
        set_seed(seed)
        rng = np.random.default_rng(seed)
        self.model = self.make().to(self.device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights(ytr, self.n_classes, self.device))
        X = torch.from_numpy(Xtr).float().to(self.device)
        y = torch.from_numpy(ytr).to(self.device)
        best, best_state, best_epoch, bad, hist = -1.0, None, 0, 0, []
        for ep in range(self.epochs):
            self.model.train()
            perm = rng.permutation(len(X))
            for s in range(0, len(perm), self.batch_size):
                idx = torch.from_numpy(perm[s:s + self.batch_size]).to(self.device)
                loss = loss_fn(self.model(X[idx]), y[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
            va_ua = metrics(yva, self.logits(Xva).argmax(1), self.n_classes)["ua"]
            hist.append(va_ua)
            if va_ua > best:
                best, best_epoch, bad = va_ua, ep, 0
                best_state = copy.deepcopy(self.model.state_dict())
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.model.load_state_dict(best_state)
        return dict(val_ua=best, best_epoch=best_epoch, epochs_run=len(hist), val_history=hist,
                    layer_weights=self.model.layer_weights())


# ================================================================ SSL: fine-tuning
def _length_bucketed_batches(lengths: np.ndarray, batch_size: int, rng: np.random.Generator) -> list:
    perm = rng.permutation(len(lengths))
    batches = []
    for s in range(0, len(perm), batch_size * 50):
        chunk = perm[s:s + batch_size * 50]
        chunk = chunk[np.argsort(lengths[chunk])]
        batches += [chunk[i:i + batch_size] for i in range(0, len(chunk), batch_size)]
    rng.shuffle(batches)
    return batches


class FinetuneTrainer:
    def __init__(self, n_classes: int, device, model_name: str, epochs: int = 20, batch_size: int = 8,
                 lr_encoder: float = 5e-5, lr_head: float = 1e-3, max_train_sec: float = 5.0, eval_every: int = 1,
                 freeze_epochs: int = 0):
        self.n_classes, self.device, self.model_name, self.epochs = n_classes, device, model_name, epochs
        self.batch_size, self.lr_encoder, self.lr_head = batch_size, lr_encoder, lr_head
        self.freeze_epochs = freeze_epochs  # train the head alone for this many epochs first
        self.max_train = int(max_train_sec * SR)
        self.eval_every = eval_every
        self.amp = torch.device(device).type == "cuda"

    @torch.no_grad()
    def logits(self, waves, max_sec: float = 10.0) -> np.ndarray:
        """Batch size 1: no padding, identical to how the demo scores a recording."""
        self.model.eval()
        out = []
        for w in waves:
            x = torch.from_numpy(w[: int(max_sec * SR)])[None].to(self.device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
                out.append(self.model(x).float().cpu().numpy())
        return np.concatenate(out)

    def _collate(self, waves, idx, rng):
        clips = []
        for i in idx:
            w = waves[i]
            if len(w) > self.max_train:
                s = int(rng.integers(0, len(w) - self.max_train + 1))
                w = w[s:s + self.max_train]
            clips.append(w)
        T = max(len(c) for c in clips)
        x = np.zeros((len(clips), T), np.float32)
        m = np.zeros((len(clips), T), np.int64)
        for j, c in enumerate(clips):
            x[j, :len(c)], m[j, :len(c)] = c, 1
        return torch.from_numpy(x).to(self.device), torch.from_numpy(m).to(self.device)

    def fit(self, tr_waves, tr_y, va_waves, va_y, seed: int, log=print) -> dict:
        set_seed(seed)
        rng = np.random.default_rng(seed)
        self.model = SSLClassifier(self.n_classes, self.model_name).to(self.device)
        self.model.freeze_feature_encoder()
        enc = [p for n, p in self.model.named_parameters() if n.startswith("encoder.") and p.requires_grad]
        opt = torch.optim.AdamW([{"params": enc, "lr": self.lr_encoder},
                                 {"params": self.model.head.parameters(), "lr": self.lr_head}], weight_decay=0.01)
        lengths = np.array([min(len(w), self.max_train) for w in tr_waves])
        steps_per_epoch = math.ceil(len(tr_waves) / self.batch_size)
        total, warmup = steps_per_epoch * self.epochs, max(1, int(0.1 * steps_per_epoch * self.epochs))
        freeze_steps = self.freeze_epochs * steps_per_epoch

        def shape(s):  # linear warm-up then linear decay
            return (s + 1) / warmup if s < warmup else max(0.0, (total - s) / max(1, total - warmup))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, [lambda s: 0.0 if s < freeze_steps else shape(s), shape])
        scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights(tr_y, self.n_classes, self.device))
        y_t = torch.from_numpy(tr_y).to(self.device)
        best, best_state, best_epoch, hist = -1.0, None, 0, []
        for ep in range(self.epochs):
            self.model.train()
            for p in enc:  # head-only warm-up keeps the pretrained encoder intact at first
                p.requires_grad = ep >= self.freeze_epochs
            t0, run_loss = time.time(), 0.0
            batches = _length_bucketed_batches(lengths, self.batch_size, rng)
            for idx in batches:
                x, m = self._collate(tr_waves, idx, rng)
                with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
                    logits = self.model(x, m)
                loss = loss_fn(logits.float(), y_t[idx])
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                run_loss += loss.item()
            if (ep + 1) % self.eval_every and ep != self.epochs - 1:
                continue
            va_ua = metrics(va_y, self.logits(va_waves).argmax(1), self.n_classes)["ua"]
            hist.append(va_ua)
            log(f"  epoch {ep + 1}/{self.epochs} loss {run_loss / len(batches):.3f} val UA {va_ua:.3f} "
                f"({time.time() - t0:.0f}s)")
            if va_ua > best:
                best, best_epoch = va_ua, ep
                best_state = {k: v.detach().to("cpu", copy=True) for k, v in self.model.state_dict().items()}
        self.model.load_state_dict(best_state)
        return dict(val_ua=best, best_epoch=best_epoch, epochs_run=self.epochs, val_history=hist,
                    layer_weights=self.model.head.layer_weights())
