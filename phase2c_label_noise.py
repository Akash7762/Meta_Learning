"""
phase2c_label_noise.py
----------------------
How noisy are the selector training labels?

Reads results/phase2c_matrix.json (the TRAINING matrix) only. No optimiser is
run, and the evaluation matrix is not read.

Each row's training label is its best algorithm: the lowest median over the
5 stored runs per cell. To see how stable that label is, every cell's 5 runs
are resampled with replacement (bootstrap, B times, seeded), the medians are
recomputed, and the best algorithm is re-picked.

  label agreement   share of resamples whose re-picked algorithm is one of the
                    row's original best (exact ties at the best median all
                    count as agreeing)
  harmful flip      a re-pick more than 1% worse than the original best, on
                    the row's min-max scale of the original medians. A flip
                    between two nearly equal algorithms is not harmful.

Rows are grouped as stable (agreement >= 90%), shaky (50-90%) and unstable
(< 50%).

With only 5 runs per cell the bootstrap is coarse. It shows how much the
labels move under re-measurement, not an exact error rate.

Outputs:
  results/phase2c_label_noise.png           the figure
  results/phase2c_label_noise.txt           summary
  results/phase2c_label_noise_rows.csv      per-row values (table view)
  results/phase2c_label_noise_functions.csv per-function values
"""
import argparse
import csv
import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.patches import Patch, PathPatch       # noqa: E402
from matplotlib.path import Path                      # noqa: E402

MATRIX = "results/phase2c_matrix.json"
OUT_STEM = "results/phase2c_label_noise"
B = 1000
SEED = 20261005
HARM = 0.01
STABLE, UNSTABLE = 0.90, 0.50

# Colours: dataviz reference palette, validated (single series; 3-step ordinal ramp).
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SERIES = "#2a78d6"
ORDINAL = ["#86b6ef", "#2a78d6", "#104281"]          # stable, shaky, unstable

DPI = 150
PX = DPI / 96.0                                       # one CSS px in output pixels
R_PX, GAP_PX, MAX_BAR_PX = 4 * PX, 2 * PX, 24 * PX


def load(path):
    d = json.load(open(path, encoding="utf-8"))
    algos = sorted(d["meta"]["algorithms"])
    rows = sorted(d["features"])
    R = np.array([[d["cells"][r + "|" + a]["fitnesses"] for a in algos] for r in rows], float)
    MED = np.array([[d["cells"][r + "|" + a]["median"] for a in algos] for r in rows], float)
    c0 = [d["cells"][r + "|" + algos[0]] for r in rows]
    funcs = np.array([c["function"] for c in c0])
    dims = np.array([c["dim"] for c in c0])
    seeds = np.array([c["pseudo_instance_seed"] for c in c0])
    assert np.isfinite(R).all(), "non-finite run in matrix"
    assert np.allclose(np.median(R, axis=2), MED), "stored medians do not match stored runs"
    return d["meta"], algos, rows, R, MED, funcs, dims, seeds


def bootstrap(R, MED, rng):
    n_row, n_alg, n_runs = R.shape
    lo, hi = MED.min(1), MED.max(1)
    span = np.where(hi - lo == 0, 1.0, hi - lo)
    agree, harm, n_tied = np.zeros(n_row), np.zeros(n_row), np.zeros(n_row, int)
    cols = np.arange(n_alg)[None, :, None]
    for i in range(n_row):
        best_set = np.flatnonzero(MED[i] == lo[i])
        n_tied[i] = len(best_set)
        idx = rng.integers(0, n_runs, size=(B, n_alg, n_runs))
        picks = np.median(R[i][cols, idx], axis=2).argmin(axis=1)
        agree[i] = np.isin(picks, best_set).mean()
        harm[i] = (((MED[i, picks] - lo[i]) / span[i]) > HARM).mean()
    return agree, harm, n_tied


def classify(a):
    return np.where(a >= STABLE, 0, np.where(a >= UNSTABLE, 1, 2))


# --------------------------------------------------------------------------
# Drawing helpers: thin bars, 4px rounded data end, square at the baseline.
# --------------------------------------------------------------------------

def _scale(ax):
    bb = ax.get_window_extent()
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    return abs(bb.width / (x1 - x0)), abs(bb.height / (y1 - y0))


def bar(ax, x0, x1, y0, y1, color, end):
    """Filled bar in data coords. end: 'top', 'right' or 'none' (square)."""
    if x1 <= x0 or y1 <= y0:
        return
    sx, sy = _scale(ax)
    rx, ry, k = R_PX / sx, R_PX / sy, 0.5523
    M, L, C, Z = Path.MOVETO, Path.LINETO, Path.CURVE4, Path.CLOSEPOLY
    if end == "top":
        rx, ry = min(rx, (x1 - x0) / 2), min(ry, y1 - y0)
        v = [(x0, y0), (x1, y0), (x1, y1 - ry), (x1, y1 - ry + k * ry), (x1 - rx + k * rx, y1),
             (x1 - rx, y1), (x0 + rx, y1), (x0 + rx - k * rx, y1), (x0, y1 - ry + k * ry),
             (x0, y1 - ry), (x0, y0)]
        c = [M, L, L, C, C, C, L, C, C, C, Z]
    elif end == "right":
        rx, ry = min(rx, x1 - x0), min(ry, (y1 - y0) / 2)
        v = [(x0, y0), (x1 - rx, y0), (x1 - rx + k * rx, y0), (x1, y0 + ry - k * ry), (x1, y0 + ry),
             (x1, y1 - ry), (x1, y1 - ry + k * ry), (x1 - rx + k * rx, y1), (x1 - rx, y1),
             (x0, y1), (x0, y0)]
        c = [M, L, C, C, C, L, C, C, C, L, Z]
    else:
        v = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
        c = [M, L, L, L, Z]
    ax.add_patch(PathPatch(Path(v, c), facecolor=color, edgecolor="none", zorder=3))


def style(ax, grid_axis):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    base = "bottom" if grid_axis == "y" else "left"
    ax.spines[base].set_visible(True)
    ax.spines[base].set_color(AXIS)
    ax.spines[base].set_linewidth(1)
    ax.grid(axis=grid_axis, color=GRID, linewidth=1, linestyle="-", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelcolor=INK2, length=0, labelsize=9)


def title(ax, head, sub):
    for txt, dy, kw in ((head, 36, dict(color=INK, fontsize=11.5, fontweight="bold")),
                        (sub, 20, dict(color=MUTED, fontsize=9))):
        ax.annotate(txt, xy=(0, 1), xycoords="axes fraction", xytext=(0, dy),
                    textcoords="offset points", va="bottom", ha="left", **kw)


def short(fn):
    return fn.replace("BBOB_", "").replace("_", " ", 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--matrix", default=MATRIX)
    ap.add_argument("--out-stem", default=OUT_STEM)
    args = ap.parse_args()

    meta, algos, rows, R, MED, funcs, dims, seeds = load(args.matrix)
    rng = np.random.default_rng(SEED)
    agree, harm, n_tied = bootstrap(R, MED, rng)
    cls = classify(agree)
    n_row = len(rows)
    udims = sorted(set(dims.tolist()))
    ufuncs = sorted(set(funcs.tolist()))

    # ---------------- summary text ----------------
    def shares(mask):
        c = cls[mask]
        return [100.0 * (c == j).mean() for j in range(3)]

    L = []
    w = L.append
    w("=" * 78)
    w("  PHASE 2c TRAINING LABELS -- HOW NOISY ARE THEY?")
    w("=" * 78)
    w("  matrix          : %s" % args.matrix)
    w("  rows            : %d   algorithms: %d   runs per cell: %d" % (n_row, len(algos), R.shape[2]))
    w("  dims / seeds    : %s / %s" % (udims, sorted(set(seeds.tolist()))))
    w("  bootstrap       : B=%d resamples per row, seed %d" % (B, SEED))
    w("  harmful flip    : re-pick more than %.0f%% worse than the original best" % (100 * HARM))
    w("  rows with a tie at the best median: %d" % int((n_tied > 1).sum()))
    w("")
    w("  label agreement over all rows: mean %.1f%%, median %.1f%%" % (100 * agree.mean(), 100 * np.median(agree)))
    w("  harmful flips over all resamples: %.1f%%  (so %.1f%% of re-picks land within 1%% of the best)"
      % (100 * harm.mean(), 100 * (1 - harm.mean())))
    w("")
    w("  %-10s %8s %8s %10s %10s %14s" % ("group", "rows", "stable", "shaky", "unstable", "harmful flips"))
    for name, mask in [("all", np.ones(n_row, bool))] + [("dim %d" % dd, dims == dd) for dd in udims]:
        s = shares(mask)
        w("  %-10s %8d %7.1f%% %9.1f%% %9.1f%% %13.1f%%" % (name, mask.sum(), s[0], s[1], s[2], 100 * harm[mask].mean()))
    w("")
    w("  stable = agreement >= 90%, shaky = 50-90%, unstable = below 50%")
    w("=" * 78)
    txt = "\n".join(L)
    print(txt)
    with open(args.out_stem + ".txt", "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")

    # ---------------- table views ----------------
    labels = ["stable", "shaky", "unstable"]
    with open(args.out_stem + "_rows.csv", "w", newline="", encoding="utf-8") as fh:
        cw = csv.writer(fh)
        cw.writerow(["row", "function", "seed", "dim", "original_best", "n_tied_best",
                     "label_agreement", "harmful_flip_rate", "class"])
        for i, r in enumerate(rows):
            cw.writerow([r, funcs[i], int(seeds[i]), int(dims[i]), algos[int(MED[i].argmin())],
                         int(n_tied[i]), "%.4f" % agree[i], "%.4f" % harm[i], labels[cls[i]]])
    fstats = []
    for f in ufuncs:
        m = funcs == f
        fstats.append((f, agree[m].mean(), harm[m].mean(), int((cls[m] == 2).sum()), int(m.sum())))
    with open(args.out_stem + "_functions.csv", "w", newline="", encoding="utf-8") as fh:
        cw = csv.writer(fh)
        cw.writerow(["function", "mean_label_agreement", "mean_harmful_flip_rate", "unstable_rows", "rows"])
        for f, a, h, u, n in fstats:
            cw.writerow([f, "%.4f" % a, "%.4f" % h, u, n])

    # ---------------- figure ----------------
    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]
    fig = plt.figure(figsize=(13, 8.6), dpi=DPI, facecolor=SURFACE)
    fig.text(0.06, 0.955, "How noisy are the training labels?", color=INK, fontsize=15, fontweight="bold")
    fig.text(0.06, 0.925, "Each row's best algorithm, re-picked from %s bootstrap resamples of its 5 stored runs. "
             "Training matrix: %d rows, %d algorithms." % (format(B, ","), n_row, len(algos)),
             color=INK2, fontsize=10)

    # Panel A: distribution of label agreement
    axA = fig.add_axes([0.07, 0.56, 0.39, 0.25])
    edges = np.linspace(0, 1, 21)
    counts, _ = np.histogram(agree, bins=edges)
    axA.set_xlim(0, 100)
    axA.set_ylim(0, max(counts) * 1.12)
    style(axA, "y")
    fig.canvas.draw()
    gap = GAP_PX / _scale(axA)[0]
    for c_, a_, b_ in zip(counts, edges[:-1], edges[1:]):
        bar(axA, 100 * a_ + gap / 2, 100 * b_ - gap / 2, 0, c_, SERIES, "top")
    for xv in (100 * UNSTABLE, 100 * STABLE):
        axA.axvline(xv, color=AXIS, linewidth=1, zorder=2)
    for xc, lab in ((25, "unstable"), (70, "shaky"), (95, "stable")):
        axA.annotate(lab, xy=(xc, 1), xycoords=("data", "axes fraction"), xytext=(0, 4),
                     textcoords="offset points", ha="center", va="bottom", color=MUTED, fontsize=8.5)
    axA.set_xticks(range(0, 101, 10))
    axA.set_xticklabels(["%d%%" % v for v in range(0, 101, 10)])
    axA.set_xlabel("Label agreement", color=INK2, fontsize=9.5)
    axA.set_ylabel("Rows", color=INK2, fontsize=9.5)
    title(axA, "Label agreement per row",
          "Mean %.1f%% across %d rows" % (100 * agree.mean(), n_row))

    # Panel B: share of rows per reliability class, overall and by dimension
    axB = fig.add_axes([0.12, 0.19, 0.34, 0.19])
    groups = [("All rows", np.ones(n_row, bool))] + [("Dim %d" % dd, dims == dd) for dd in udims]
    axB.set_xlim(0, 100)
    axB.set_ylim(-0.6, len(groups) - 0.4)
    style(axB, "x")
    axB.invert_yaxis()
    fig.canvas.draw()
    sx, sy = _scale(axB)
    half = min(0.31, MAX_BAR_PX / sy / 2)
    gx = GAP_PX / sx
    renderer = fig.canvas.get_renderer()
    for gi, (gname, mask) in enumerate(groups):
        s = shares(mask)
        nz = [j for j in range(3) if s[j] > 0]
        x = 0.0
        for j in range(3):
            if s[j] <= 0:
                continue
            last = j == nz[-1]
            x0 = x + (gx / 2 if x > 0 else 0)
            x1 = x + s[j] - (0 if last else gx / 2)
            bar(axB, x0, x1, gi - half, gi + half, ORDINAL[j], "right" if last else "none")
            t = axB.text((x0 + x1) / 2, gi, "%.0f%%" % s[j], ha="center", va="center", fontsize=8.5,
                         color=("#ffffff" if j > 0 else INK), zorder=4)
            if t.get_window_extent(renderer).width + 8 * PX > (x1 - x0) * sx:
                t.remove()
            x += s[j]
    axB.set_yticks(range(len(groups)))
    axB.set_yticklabels(["%s (%d)" % (g, m.sum()) for g, m in groups])
    axB.set_xticks(range(0, 101, 20))
    axB.set_xticklabels(["%d%%" % v for v in range(0, 101, 20)])
    axB.set_xlabel("Share of rows", color=INK2, fontsize=9.5)
    title(axB, "Rows by label reliability", "Darker means a noisier label")
    axB.legend(handles=[Patch(facecolor=ORDINAL[0], label="Stable (90% or more)"),
                        Patch(facecolor=ORDINAL[1], label="Shaky (50-90%)"),
                        Patch(facecolor=ORDINAL[2], label="Unstable (below 50%)")],
               loc="upper left", bbox_to_anchor=(-0.16, -0.30), ncol=3, frameon=False, fontsize=8.5,
               labelcolor=INK2, handlelength=1.0, handleheight=1.0, columnspacing=1.4)

    # Panel C: mean label agreement by function, noisiest at the top
    axC = fig.add_axes([0.62, 0.19, 0.34, 0.62])
    order = sorted(fstats, key=lambda t: t[1])
    n_f = len(order)
    axC.set_xlim(0, 100)
    axC.set_ylim(-0.6, n_f - 0.4)
    style(axC, "x")
    axC.invert_yaxis()
    fig.canvas.draw()
    sx, sy = _scale(axC)
    half = min(0.31, MAX_BAR_PX / sy / 2)
    for k_, (f, a, h, u, n) in enumerate(order):
        bar(axC, 0, 100 * a, k_ - half, k_ + half, SERIES, "right")
    for k_ in list(range(3)) + [n_f - 1]:
        a = order[k_][1]
        axC.text(100 * a + 1.2, k_, "%.0f%%" % (100 * a), va="center", ha="left", color=INK2, fontsize=8.5)
    axC.set_yticks(range(n_f))
    axC.set_yticklabels([short(t[0]) for t in order], fontsize=8.5)
    axC.set_xticks(range(0, 101, 20))
    axC.set_xticklabels(["%d%%" % v for v in range(0, 101, 20)])
    axC.set_xlabel("Mean label agreement", color=INK2, fontsize=9.5)
    title(axC, "Mean label agreement by function", "Noisiest functions at the top; 30 rows each")

    fig.text(0.07, 0.040,
             "Source: %s. Bootstrap B=%d, seed %d. Ties at the best median count as agreement. "
             "%.1f%% of all re-picks land within 1%% of the original best."
             % (args.matrix, B, SEED, 100 * (1 - harm.mean())), color=MUTED, fontsize=8)
    fig.text(0.07, 0.018, "Values per row and per function: %s_rows.csv, %s_functions.csv."
             % (args.out_stem, args.out_stem), color=MUTED, fontsize=8)
    fig.savefig(args.out_stem + ".png", dpi=DPI, facecolor=SURFACE)
    print("saved: %s.png, .txt, _rows.csv, _functions.csv" % args.out_stem)


if __name__ == "__main__":
    main()
