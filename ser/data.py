"""Dataset download, metadata parsing, audio loading and speaker-independent splits."""
from __future__ import annotations

import json
import pickle
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SR = 16000
SPLIT_SEED = 1234  # fixed once, never tuned

# ---------------------------------------------------------------- RAVDESS
RAVDESS_URL = "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"
# Filename field 3 is the emotion code 01..08, in this order.
RAVDESS_EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
ACTOR_SPLIT = {"train": range(1, 19), "val": range(19, 21), "test": range(21, 25)}

# ---------------------------------------------------------------- CREMA-D
CREMAD_REPO = "CheyneyComputerScience/CREMA-D"
CREMAD_MEDIA = f"https://media.githubusercontent.com/media/{CREMAD_REPO}/master/AudioWAV/"
# Fallback: byte-identical AudioWAV files packed as one archive on the Hugging Face Hub.
CREMAD_MIRROR = "https://huggingface.co/datasets/myleslinder/crema-d/resolve/main/data/crema_d.tar.gz"
CREMAD_CODES = {"NEU": "neutral", "HAP": "happy", "SAD": "sad", "ANG": "angry", "FEA": "fearful", "DIS": "disgust"}
# The six emotions CREMA-D and RAVDESS have in common (RAVDESS adds calm and surprised).
SHARED_EMOTIONS = ["neutral", "happy", "sad", "angry", "fearful", "disgust"]
CREMAD_SPEAKER_SPLIT = (64, 9, 18)  # train / val / test speakers out of 91


def _download(url: str, dest: Path, retries: int = 6) -> None:
    for attempt in range(retries):
        try:
            tmp = dest.with_suffix(dest.suffix + ".part")
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(dest)
            return
        except Exception:  # network hiccup: back off and retry
            if attempt == retries - 1:
                raise
            time.sleep(min(2 ** attempt, 20))


def download_ravdess(root: str | Path) -> Path:
    root = Path(root)
    out = root / "ravdess"
    if len(list(out.glob("Actor_*/*.wav"))) == 1440:
        return out
    root.mkdir(parents=True, exist_ok=True)
    zpath = root / "ravdess_speech.zip"
    if not zpath.exists():
        print("Downloading RAVDESS speech audio from Zenodo (~200 MB)...")
        _download(RAVDESS_URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(out)
    return out


def ravdess_metadata(root: str | Path) -> pd.DataFrame:
    rows = []
    for p in sorted(Path(root).glob("Actor_*/*.wav")):
        f = p.stem.split("-")  # modality-channel-emotion-intensity-statement-repetition-actor
        actor = int(f[6])
        rows.append(dict(path=str(p), dataset="ravdess", speaker=actor,
                         emotion=RAVDESS_EMOTIONS[int(f[2]) - 1], intensity=int(f[3]),
                         statement=int(f[4]), gender="male" if actor % 2 else "female"))
    df = pd.DataFrame(rows)
    assert len(df) == 1440, f"expected 1440 RAVDESS clips, found {len(df)}"
    return df


def ravdess_split(df: pd.DataFrame, kind: str) -> dict[str, np.ndarray]:
    """Return index arrays for train/val/test.

    actor : actors 1-18 train, 19-20 val, 21-24 test (no speaker overlap)
    random: stratified random split with exactly the same sizes (1080/120/240)
    """
    if kind == "actor":
        return {k: df.index[df.speaker.isin(list(v))].to_numpy() for k, v in ACTOR_SPLIT.items()}
    if kind == "random":
        idx = df.index.to_numpy()
        trval, test = train_test_split(idx, test_size=240, stratify=df.emotion, random_state=SPLIT_SEED)
        train, val = train_test_split(trval, test_size=120, stratify=df.emotion[trval], random_state=SPLIT_SEED)
        return {"train": np.sort(train), "val": np.sort(val), "test": np.sort(test)}
    raise ValueError(kind)


def download_cremad(root: str | Path, workers: int = 8, limit: int | None = None) -> Path:
    """Fetch the 7,442 CREMA-D WAV files from the official GitHub LFS store.

    `limit` keeps an evenly spaced subset (smoke tests only); the metadata
    reader then only sees the files that were downloaded.
    """
    out = Path(root) / "cremad"
    out.mkdir(parents=True, exist_ok=True)
    api = f"https://api.github.com/repos/{CREMAD_REPO}"
    listing = out / "_listing.json"
    if not listing.exists():
        with urllib.request.urlopen(f"{api}/contents/") as r:
            sha = next(d["sha"] for d in json.load(r) if d["name"] == "AudioWAV")
        with urllib.request.urlopen(f"{api}/git/trees/{sha}") as r:
            names = [t["path"] for t in json.load(r)["tree"] if t["path"].endswith(".wav")]
        listing.write_text(json.dumps(names))
    names = json.loads(listing.read_text())
    if limit:
        names = names[:: max(1, len(names) // limit)]
    todo = [n for n in names if not (out / n).exists() or (out / n).stat().st_size < 1000]
    if todo:
        print(f"Downloading {len(todo)} CREMA-D files from GitHub LFS...", flush=True)
        try:
            with ThreadPoolExecutor(workers) as ex:
                list(ex.map(lambda n: _download(CREMAD_MEDIA + n, out / n), todo))
        except Exception as e:  # GitHub LFS sometimes answers 503 under load
            print(f"GitHub LFS failed ({e}); using the Hugging Face mirror of the same files.", flush=True)
            _cremad_from_mirror(out)
    return out


def _cremad_from_mirror(out: Path) -> None:
    import tarfile
    tgz = out.parent / "crema_d.tar.gz"
    if not tgz.exists():
        _download(CREMAD_MIRROR, tgz)
    with tarfile.open(tgz) as tar:
        for m in tar:
            if m.isfile() and m.name.endswith(".wav"):
                m.name = Path(m.name).name
                tar.extract(m, out, filter="data")


def cremad_metadata(root: str | Path) -> pd.DataFrame:
    rows = []
    listing = Path(root) / "_listing.json"
    official = set(json.loads(listing.read_text())) if listing.exists() else None
    for p in sorted(Path(root).glob("*.wav")):
        if official is not None and p.name not in official:  # e.g. a misnamed duplicate in the mirror
            continue
        spk, sent, emo, level = p.stem.split("_")
        if p.stat().st_size < 1000:  # a handful of CREMA-D files are empty
            continue
        rows.append(dict(path=str(p), dataset="cremad", speaker=int(spk), emotion=CREMAD_CODES[emo],
                         intensity=level, statement=sent))
    return pd.DataFrame(rows)


def cremad_split(df: pd.DataFrame) -> dict[str, np.ndarray]:
    speakers = np.array(sorted(df.speaker.unique()))
    perm = np.random.default_rng(SPLIT_SEED).permutation(speakers)
    n_tr, n_va, _ = CREMAD_SPEAKER_SPLIT
    groups = {"train": perm[:n_tr], "val": perm[n_tr:n_tr + n_va], "test": perm[n_tr + n_va:]}
    return {k: df.index[df.speaker.isin(v)].to_numpy() for k, v in groups.items()}


# ---------------------------------------------------------------- audio
def preprocess_wave(y: np.ndarray, sr: int, top_db: float = 40.0) -> np.ndarray:
    """Mono 16 kHz, leading/trailing silence trimmed, peak-normalised.

    The same function is used for training data and for the live demo, so a
    microphone recording goes through exactly the pipeline the model saw.
    """
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=-1) if y.shape[-1] <= 8 else y.mean(axis=0)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    if len(trimmed) >= 0.3 * SR:
        y = trimmed
    peak = np.abs(y).max()
    if peak > 0:
        y = 0.95 * y / peak
    return y.astype(np.float32)


def load_wave(path: str) -> np.ndarray:
    y, sr = librosa.load(path, sr=None, mono=True)
    return preprocess_wave(y, sr)


def load_waves(paths: list[str], cache: str | Path | None = None, workers: int = 4) -> list[np.ndarray]:
    """Load (and cache as int16) every clip; returns float32 arrays."""
    if cache is not None and Path(cache).exists():
        with open(cache, "rb") as f:
            stored = pickle.load(f)
        if stored["paths"] == list(paths):
            return [w.astype(np.float32) / 32767.0 for w in stored["waves"]]
    with ThreadPoolExecutor(workers) as ex:
        waves = list(ex.map(load_wave, paths))
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "wb") as f:
            pickle.dump({"paths": list(paths), "waves": [(w * 32767).astype(np.int16) for w in waves]}, f)
    return waves
