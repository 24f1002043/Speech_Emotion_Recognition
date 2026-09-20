"""Aggregate results/runs/*.json into tables (Markdown + JSON) and figures."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

MODELS = {"mfcc_cnn": "MFCC-CNN (baseline)", "wavlm_frozen": "WavLM Base+ frozen",
          "wavlm_finetune": "WavLM Base+ fine-tuned"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
ACTOR, RANDOM = "#2a78d6", "#eb6834"  # categorical slots 1-2 (validated adjacent pair)
BLUES = LinearSegmentedColormap.from_list("seq_blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def pct(m, s=None, n=1):
    return f"{100 * m:.1f}" if s is None or n < 2 else f"{100 * m:.1f} ± {100 * s:.1f}"


def aggregate(runs):
    groups = defaultdict(list)
    for r in runs:
        groups[(r["dataset"], r["split"], r["model"])].append(r)
    agg = {}
    for key, rs in groups.items():
        rs = sorted(rs, key=lambda r: r["seed"])
        a = dict(n=len(rs), seeds=[r["seed"] for r in rs], classes=rs[0]["classes"], sizes=rs[0]["sizes"])
        for m in ["ua", "wa", "macro_f1"]:
            v = np.array([r["test"][m] for r in rs])
            a[m], a[m + "_std"], a[m + "_runs"] = float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0, v.tolist()
        a["val_ua"] = float(np.mean([r["val_ua"] for r in rs]))
        a["confusion"] = np.sum([r["test"]["confusion"] for r in rs], axis=0).tolist()
        cross = defaultdict(list)
        for r in rs:
            for k, v in r.get("cross", {}).items():
                cross[k].append(v)
        a["cross"] = {k: dict(ua=float(np.mean([x["ua"] for x in v])), ua_std=float(np.std([x["ua"] for x in v], ddof=1))
                              if len(v) > 1 else 0.0, wa=float(np.mean([x["wa"] for x in v])), n=len(v),
                              confusion=np.sum([x["confusion"] for x in v], axis=0).tolist())
                      for k, v in cross.items()}
        if "layer_weights" in rs[0]:
            a["layer_weights"] = np.mean([r["layer_weights"] for r in rs], axis=0).tolist()
        agg["/".join(key)] = a
    return agg


def fig_split_comparison(agg, out):
    models = [m for m in MODELS if f"ravdess/actor/{m}" in agg and f"ravdess/random/{m}" in agg]
    if not models:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    x, w = np.arange(len(models)), 0.36
    for off, split, color, label in [(-w / 2 - 0.01, "random", RANDOM, "Random split (speakers leak into test)"),
                                     (w / 2 + 0.01, "actor", ACTOR, "Actor split (unseen actors 21–24)")]:
        a = [agg[f"ravdess/{split}/{m}"] for m in models]
        vals = np.array([r["ua"] for r in a]) * 100
        errs = np.array([r["ua_std"] for r in a]) * 100
        ax.bar(x + off, vals, w, color=color, label=label, zorder=2)
        ax.errorbar(x + off, vals, yerr=errs, fmt="none", ecolor=INK2, elinewidth=1, capsize=3, zorder=3)
        for xi, v, e in zip(x + off, vals, errs):
            ax.text(xi, v + e + 1.2, f"{v:.1f}", ha="center", va="bottom", fontsize=9, color=INK)
    ax.axhline(12.5, color=INK2, lw=1, ls=(0, (3, 3)), zorder=1)
    ax.text(len(models) - 0.5, 13.5, "chance (8 classes)", ha="right", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(x, [MODELS[m] for m in models])
    ax.set_ylabel("Unweighted accuracy (%)")
    ax.set_ylim(0, 105)
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(length=0)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.set_title("RAVDESS, 8 emotions: random vs actor-independent split (mean ± sd over seeds)",
                 fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_confusion(conf, classes, title, out):
    conf = np.array(conf, dtype=float)
    norm = conf / conf.sum(1, keepdims=True).clip(min=1) * 100
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.imshow(norm, cmap=BLUES, vmin=0, vmax=100)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{norm[i, j]:.0f}", ha="center", va="center", fontsize=8,
                    color="#ffffff" if norm[i, j] > 55 else INK)
    ax.set_xticks(range(len(classes)), classes, rotation=40, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=10, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--include-quick", action="store_true")
    args = ap.parse_args()
    root = Path(args.results)
    runs = [json.loads(p.read_text()) for p in sorted((root / "runs").glob("*.json"))]
    runs = [r for r in runs if args.include_quick or not r.get("quick")]
    if not runs:
        raise SystemExit("no runs found")
    agg = aggregate(runs)
    (root / "summary.json").write_text(json.dumps(agg, indent=1))
    figs = root / "figures"
    figs.mkdir(exist_ok=True)

    L = ["# Results", "",
         "All numbers are on held-out test data, scored once with the checkpoint picked on validation UA.",
         "UA = unweighted accuracy (mean per-class recall); WA = plain accuracy. Mean ± sd over seeds.", "",
         "## RAVDESS (8 emotions): random split vs actor-independent split", "",
         "| Model | Random split UA | Actor split UA (unseen actors) | Gap (pp) | Actor split WA | Actor split macro-F1 | Seeds |",
         "|---|---|---|---|---|---|---|"]
    for m, label in MODELS.items():
        a, r = agg.get(f"ravdess/actor/{m}"), agg.get(f"ravdess/random/{m}")
        if not a:
            continue
        gap = f"{100 * (r['ua'] - a['ua']):+.1f}" if r else "–"
        L.append(f"| {label} | {pct(r['ua'], r['ua_std'], r['n']) if r else '–'} | **{pct(a['ua'], a['ua_std'], a['n'])}** "
                 f"| {gap} | {pct(a['wa'], a['wa_std'], a['n'])} | {pct(a['macro_f1'], a['macro_f1_std'], a['n'])} | {a['n']} |")
    L += ["", "Split sizes (both splits): train 1080 / val 120 / test 240 clips. "
          "Actor split: actors 1–18 train, 19–20 val, 21–24 test.", ""]

    if fig_split_comparison(agg, figs / "split_comparison.png"):
        L += ["![split comparison](figures/split_comparison.png)", ""]

    L += ["## Per-class recall on unseen actors (actor split, pooled over seeds)", ""]
    first = next((agg[k] for k in (f"ravdess/actor/{m}" for m in MODELS) if k in agg), None)
    if first:
        classes = first["classes"]
        L += ["| Model | " + " | ".join(classes) + " |", "|---|" + "---|" * len(classes)]
        for m, label in MODELS.items():
            a = agg.get(f"ravdess/actor/{m}")
            if a:
                c = np.array(a["confusion"], float)
                rec = np.diag(c) / c.sum(1).clip(min=1)
                L.append(f"| {label} | " + " | ".join(f"{100 * v:.0f}" for v in rec) + " |")
                fig_confusion(a["confusion"], classes, f"{label}: unseen actors (row %)",
                              figs / f"confusion_actor_{m}.png")
        L += [""] + [f"![{m}](figures/confusion_actor_{m}.png)" for m in MODELS if f"ravdess/actor/{m}" in agg] + [""]

    cross_rows = []
    for m, label in MODELS.items():
        a = agg.get(f"ravdess/actor/{m}")
        if a and "cremad_test" in a["cross"]:
            c = a["cross"]["cremad_test"]
            cross_rows.append(f"| {label} | RAVDESS actors 1–18 | RAVDESS actors 21–24 (8-class) | {pct(a['ua'], a['ua_std'], a['n'])} "
                              f"| CREMA-D, 18 unseen speakers (6-class) | {pct(c['ua'], c['ua_std'], c['n'])} |")
    for m, label in MODELS.items():
        a = agg.get(f"cremad/speaker/{m}")
        if a:
            c = a["cross"]["ravdess_all_actors"]
            cross_rows.append(f"| {label} | CREMA-D, 64 speakers | CREMA-D, 18 unseen speakers (6-class) | {pct(a['ua'], a['ua_std'], a['n'])} "
                              f"| RAVDESS, all 24 actors (6-class) | {pct(c['ua'], c['ua_std'], c['n'])} |")
            fig_confusion(c["confusion"], a["classes"], f"{label}: trained CREMA-D → tested RAVDESS",
                          figs / f"confusion_cremad_to_ravdess_{m}.png")
    if cross_rows:
        L += ["## Cross-corpus generalisation (RAVDESS ↔ CREMA-D)", "",
              "Only the 6 emotions both corpora share are scored across corpora (neutral, happy, sad, angry, fearful, disgust); "
              "an 8-class RAVDESS model is restricted to those 6 outputs. Chance = 16.7%.", "",
              "| Model | Trained on | In-corpus test | UA | Cross-corpus test | UA |", "|---|---|---|---|---|---|"] + cross_rows + [""]

    lw = agg.get("ravdess/actor/wavlm_frozen", {}).get("layer_weights")
    if lw:
        top = np.argsort(lw)[::-1][:3]
        L += ["## Which WavLM layers matter (frozen model, learned layer weights)", "",
              "Layer 0 = CNN/embedding output, 12 = last transformer layer. Top layers: "
              + ", ".join(f"{i} ({100 * lw[i]:.0f}%)" for i in top) + ".", ""]
    (root / "RESULTS.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
