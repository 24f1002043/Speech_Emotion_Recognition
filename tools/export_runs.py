"""Pack/unpack results/runs as one base64 blob, to move results off a Colab VM without a file download.

On the training machine:   python tools/export_runs.py pack
Then, locally:             python tools/export_runs.py unpack blob.txt
"""
from __future__ import annotations

import base64
import json
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs"


def pack(chunk: int = 3000) -> None:
    runs = {p.stem: json.loads(p.read_text()) for p in sorted(RUNS.glob("*.json"))}
    blob = base64.b64encode(zlib.compress(json.dumps(runs, separators=(",", ":")).encode(), 9)).decode()
    print(f"RUNS_BLOB_START {len(runs)} runs {len(blob)} chars")
    for i in range(0, len(blob), chunk):
        print(blob[i:i + chunk])
    print("RUNS_BLOB_END")


def unpack(path: str) -> None:
    text = "".join(Path(path).read_text().split())
    runs = json.loads(text) if text.startswith("{") else json.loads(zlib.decompress(base64.b64decode(text)))
    RUNS.mkdir(parents=True, exist_ok=True)
    for name, rec in runs.items():
        (RUNS / f"{name}.json").write_text(json.dumps(rec, indent=1))
    print(f"wrote {len(runs)} run files to {RUNS}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "unpack":
        unpack(sys.argv[2])
    else:
        pack()
