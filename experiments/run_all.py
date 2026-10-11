"""Run every sparse-MoE trade-off experiment and write the plots.

Usage:  python experiments/run_all.py [--steps 300] [--quick]
"""

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sparse_moe_tradeoffs import (CharData, CharLM, active_expert_param_count,
                                  build_corpus, domain_routing_counts,
                                  expert_param_count, make_val_batches,
                                  matched_family, train_model, variant)
from sparse_moe_tradeoffs.corpus import DOMAINS

PLOTS = ROOT / "plots"

# ------------------------------------------------------------------ palette
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#e34948"]   # fixed order, never cycled
SURFACE, INK, SEC, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
SEQ_CMAP = LinearSegmentedColormap.from_list(
    "seqblue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf",
                "#184f95", "#0d366b"])


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "text.color": INK,
        "axes.edgecolor": BASELINE, "axes.labelcolor": SEC,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "axes.titlesize": 10.5, "figure.titlesize": 13,
        "legend.frameon": False,
    })


def smooth(a: np.ndarray, w: int) -> np.ndarray:
    """Column-wise moving average (same length)."""
    if w <= 1:
        return a
    kern = np.ones(w) / w
    return np.stack([np.convolve(a[:, j], kern, mode="same")
                     for j in range(a.shape[1])], axis=1)


def stacked_loads(ax, loads: np.ndarray, w: int = 15, title: str = ""):
    steps = np.arange(loads.shape[0])
    sm = smooth(loads, w)
    sm = sm / sm.sum(axis=1, keepdims=True)
    ax.stackplot(steps, sm.T, colors=SERIES[:loads.shape[1]],
                 edgecolor=SURFACE, linewidth=0.6)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, len(steps) - 1)
    ax.set_title(title)
    ax.grid(False)


def bar_panel(ax, names, values, color=SERIES[0], fmt="{:.3f}"):
    xs = np.arange(len(names))
    bars = ax.bar(xs, values, width=0.55, color=color, edgecolor=SURFACE,
                  linewidth=1.5)
    ax.set_xticks(xs, names)
    for b, v in zip(bars, values):
        ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=9, color=INK)
    ax.grid(axis="x", visible=False)
    return bars


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--quick", action="store_true",
                    help="tiny run for CI (steps=40, small corpus)")
    args = ap.parse_args()
    steps = 40 if args.quick else args.steps
    n_chars = 30_000 if args.quick else 150_000

    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(2)
    style()
    PLOTS.mkdir(exist_ok=True)
    t_start = time.time()

    text, tags = build_corpus(n_chars=n_chars, seed=0)
    data = CharData(text, tags, block_size=128)
    fam = matched_family()
    val_batches = make_val_batches(data, n_batches=6, batch_size=24, seed=123)
    print(f"corpus {len(text)} chars, vocab {data.vocab_size}")

    D = 96
    runs, models = {}, {}

    def run(name, spec=None, mtp=False, label=None):
        torch.manual_seed(0)
        model = CharLM(data.vocab_size, data.block_size, d=D, moe_spec=spec,
                       dense_hidden=fam["dense_hidden"], mtp=mtp)
        t0 = time.time()
        hist = train_model(model, data, val_batches, steps=steps,
                           batch_size=args.batch_size, eval_every=25, seed=0)
        hist["label"] = label or name
        hist["wall"] = time.time() - t0
        runs[name], models[name] = hist, model
        extra = ""
        if "mtp_acc" in hist["final"]:
            extra = f"  t+2 acc {hist['final']['mtp_acc']:.3f}"
        print(f"[{hist['wall']:5.1f}s] {name:12s} val {hist['final']['val_loss']:.4f}"
              f"  params {model.n_params()}{extra}")
        return hist

    flag = fam["flagship"]
    run("dense", None, label="dense FFN")
    run("coarse", fam["coarse"], label="2 big, top-1")
    run("fine_iso", fam["fine_iso"], label="8 fine, top-2")
    run("flagship", flag, label="8 fine + shared")
    run("aux", variant(flag, balance="aux", aux_coef=0.2), label="aux loss (0.2)")
    run("none", variant(flag, balance="none"), label="no balancing")
    run("mtp", flag, mtp=True, label="flagship + MTP")
    run("gamma50", variant(flag, gamma=0.5), label="gamma x50")
    run("no_shared", variant(flag, shared_hidden=0), label="shared removed")
    run("no_renorm", variant(flag, renormalize=False), label="no renorm")

    # ---------------------------------------------- 1. iso-active quality bar
    order = ["dense", "coarse", "fine_iso", "flagship"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    vals = [runs[k]["final"]["val_loss"] for k in order]
    names = []
    for k in order:
        spec = {"dense": None, "coarse": fam["coarse"],
                "fine_iso": fam["fine_iso"], "flagship": flag}[k]
        if spec is None:
            tot = act = 2 * D * fam["dense_hidden"]
        else:
            tot = expert_param_count(spec, D)
            act = active_expert_param_count(spec, D)
        names.append(f"{runs[k]['label']}\ntotal {tot//1000}k / active {act//1000}k")
    bar_panel(ax, names, vals)
    ax.set_ylabel("validation loss (nats/char)")
    ax.set_ylim(min(vals) * 0.96, max(vals) * 1.04)
    ax.set_title("Iso-ACTIVE-parameter quality (FFN params per layer under each bar)")
    fig.tight_layout()
    fig.savefig(PLOTS / "iso_active_quality.png", dpi=150)
    plt.close(fig)

    # ------------------------------------- 2. load balancing: stacked areas
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for ax, key in zip(axes, ["aux", "flagship", "none"]):
        h = runs[key]
        stacked_loads(ax, h["loads"][1],
                      title=f"{h['label']} — final val {h['final']['val_loss']:.3f}")
        ax.set_xlabel("training step")
    axes[0].set_ylabel("expert load share (layer 2)")
    fig.legend(handles=[plt.Line2D([], [], color=SERIES[e], lw=6,
                                   label=f"expert {e}") for e in range(8)],
               loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=8,
               fontsize=8)
    fig.suptitle("Load balancing: strong aux loss vs aux-FREE bias vs nothing")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(PLOTS / "load_balance_stackedarea.png", dpi=150)
    plt.close(fig)

    # -------------------------------------------- 3. aux-free bias trajectories
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for li, ax in enumerate(axes):
        b = runs["flagship"]["biases"][li]
        for e in range(b.shape[1]):
            ax.plot(b[:, e], color=SERIES[e], lw=2, label=f"expert {e}")
        ax.set_title(f"layer {li + 1}")
        ax.set_xlabel("training step")
    axes[0].set_ylabel("selection bias $b_e$")
    axes[1].legend(fontsize=8, ncol=2)
    fig.suptitle("Aux-free balancing: per-expert selection-bias trajectories "
                 r"($b_e$ += $\gamma\,\mathrm{sign}(\bar{c} - c_e)$)")
    fig.tight_layout()
    fig.savefig(PLOTS / "auxfree_bias_trajectories.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------- 4. MTP effect
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8),
                                   gridspec_kw={"width_ratios": [2, 1]})
    for key, color in (("flagship", SERIES[0]), ("mtp", SERIES[1])):
        h = runs[key]
        ax1.plot(h["val_steps"], h["val_loss"], color=color, lw=2,
                 label=h["label"])
    ax1.set_xlabel("training step")
    ax1.set_ylabel("next-token val loss")
    ax1.set_title("Effect of the t+2 head on next-token quality")
    ax1.legend()
    acc = runs["mtp"]["final"]["mtp_acc"]
    bar_panel(ax2, ["t+2 top-1\naccuracy"], [acc], color=SERIES[1], fmt="{:.1%}")
    ax2.set_ylim(0, 1)
    ax2.set_title(f"MTP head (CE {runs['mtp']['final']['mtp_loss']:.3f})")
    fig.tight_layout()
    fig.savefig(PLOTS / "mtp_effect.png", dpi=150)
    plt.close(fig)

    # -------------------------------------- 5. domain specialization heatmap
    counts = domain_routing_counts(models["flagship"], data)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for li, ax in enumerate(axes):
        frac = counts[li] / counts[li].sum(axis=1, keepdims=True)
        im = ax.imshow(frac, cmap=SEQ_CMAP, vmin=0, vmax=frac.max(),
                       aspect="auto")
        ax.set_xticks(range(8), [f"E{e}" for e in range(8)])
        ax.set_yticks(range(3), DOMAINS if li == 0 else [""] * 3)
        for i in range(3):
            for j in range(8):
                ax.annotate(f"{frac[i, j]:.2f}", (j, i), ha="center",
                            va="center", fontsize=7.5,
                            color=SURFACE if frac[i, j] > 0.6 * frac.max() else INK)
        ax.set_title(f"layer {li + 1}")
        ax.grid(False)
    fig.colorbar(im, ax=axes, shrink=0.8, label="routed share")
    fig.suptitle("Which experts see which domain (flagship, share of routed slots)",
                 y=1.04)
    fig.savefig(PLOTS / "specialization_heatmap.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # ------------------------------- 6. break-it: gamma 50x too high (thrash)
    fig, axes = plt.subplots(2, 2, figsize=(11, 6))
    stacked_loads(axes[0, 0], runs["gamma50"]["loads"][1], w=1,
                  title=f"gamma x50: loads, layer 2 (val "
                        f"{runs['gamma50']['final']['val_loss']:.3f})")
    axes[0, 0].set_xlabel("training step")
    zoom = slice(max(0, steps // 2), min(steps, steps // 2 + 60))
    loads_z = runs["gamma50"]["loads"][1][zoom]
    for e in range(8):
        axes[0, 1].plot(range(zoom.start, zoom.start + loads_z.shape[0]),
                        loads_z[:, e], color=SERIES[e], lw=1.5)
    axes[0, 1].set_title("gamma x50: per-expert load, 60-step zoom (oscillation)")
    axes[0, 1].set_xlabel("training step")
    axes[0, 1].set_ylabel("load share")
    for e in range(8):
        axes[1, 0].plot(runs["gamma50"]["biases"][1][:, e], color=SERIES[e],
                        lw=1.5)
    axes[1, 0].set_title("gamma x50: bias trajectories (thrash)")
    axes[1, 0].set_xlabel("training step")
    axes[1, 0].set_ylabel("selection bias $b_e$")
    for e in range(8):
        axes[1, 1].plot(runs["flagship"]["biases"][1][:, e], color=SERIES[e],
                        lw=1.5)
    axes[1, 1].set_title(f"healthy gamma for contrast (val "
                         f"{runs['flagship']['final']['val_loss']:.3f})")
    axes[1, 1].set_xlabel("training step")
    fig.suptitle("Break-it: aux-free update rate 50x too high")
    fig.tight_layout()
    fig.savefig(PLOTS / "break_gamma_thrash.png", dpi=150)
    plt.close(fig)

    # --------------------------------- 7. break-it: shared expert removed
    fig, ax = plt.subplots(figsize=(5.2, 4))
    vals = [runs["flagship"]["final"]["val_loss"],
            runs["no_shared"]["final"]["val_loss"]]
    act_f = active_expert_param_count(flag, D) / 1000
    act_n = active_expert_param_count(variant(flag, shared_hidden=0), D) / 1000
    bar_panel(ax, [f"8 fine + shared\n({act_f:.1f}k FFN active/layer)",
                   f"shared removed\n({act_n:.1f}k FFN active/layer)"], vals)
    ax.set_ylabel("validation loss (nats/char)")
    ax.set_ylim(0, max(vals) * 1.15)
    ax.set_title("Break-it: delete the always-active shared expert")
    fig.tight_layout()
    fig.savefig(PLOTS / "break_shared_removed.png", dpi=150)
    plt.close(fig)

    # ------------------------------ 8. break-it: forgot gate renormalization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8),
                                   gridspec_kw={"width_ratios": [2, 1]})
    for key, color in (("flagship", SERIES[0]), ("no_renorm", SERIES[1])):
        h = runs[key]
        ax1.plot(h["val_steps"], h["val_loss"], color=color, lw=2,
                 label=h["label"])
    ax1.set_xlabel("training step")
    ax1.set_ylabel("next-token val loss")
    ax1.legend()
    ax1.set_title("Route top-2 but forget to renormalize the two gates")
    bar_panel(ax2, ["renorm", "no renorm"],
              [runs["flagship"]["final"]["val_loss"],
               runs["no_renorm"]["final"]["val_loss"]])
    ax2.set_title("final val loss")
    fig.tight_layout()
    fig.savefig(PLOTS / "break_no_renorm.png", dpi=150)
    plt.close(fig)

    # --------------------------------------------------------------- summary
    summary = {k: {"label": h["label"], "val_loss": round(h["final"]["val_loss"], 4),
                   "params": models[k].n_params(),
                   **({"mtp_acc": round(h["final"]["mtp_acc"], 4),
                       "mtp_loss": round(h["final"]["mtp_loss"], 4)}
                      if "mtp_acc" in h["final"] else {})}
               for k, h in runs.items()}
    summary["_meta"] = {"steps": steps, "batch_size": args.batch_size,
                        "wall_seconds": round(time.time() - t_start, 1),
                        "peak_rss_mb": round(resource.getrusage(
                            resource.RUSAGE_SELF).ru_maxrss / 1024, 1)}
    (ROOT / "results.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"total wall time {time.time() - t_start:.1f}s; plots in {PLOTS}/")


if __name__ == "__main__":
    main()
