"""Figures for the README, drawn from results/generations/*.json and results/logs/train_*.log -> results/figures/*.png.

  python src/make_figures.py
Colors: categorical slots 1-2 of the reference palette (blue = Qwen2.5-1.5B, orange = Llama-3.2-3B), fixed order.
"""
import ast, os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from metrics import esconv, goal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # bridge/
GEN = os.path.join(ROOT, "results", "generations"); LOGS = os.path.join(ROOT, "results", "logs")
OUT = os.path.join(ROOT, "results", "figures"); os.makedirs(OUT, exist_ok=True)
GOLD = os.path.join(ROOT, "..", "data", "feedback_qesconv_labeledsplit", "test.json")
EVAL = os.path.join(ROOT, "data", "goal_eval_set.json")
BLUE, ORANGE, INK, MUTED, GRID, SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e3e2de", "#fcfcfb"
MAJORITY = 0.583  # always-predict-majority ESConv accuracy (results/paired_bootstrap.txt)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "axes.spines.top": False, "axes.spines.right": False})

def score(run):
    e = esconv(GOLD, os.path.join(GEN, f"{run}_esconv.json")); g = goal(EVAL, os.path.join(GEN, f"{run}_goal.json"))
    return {"acc": e["perfect_acc"], "acc_ci": e["perfect_acc_ci95"], "det": g["detection_balanced_acc"], "det_ci": g["detection_ci95"]}

RUNS = {r: score(r) for r in ["sft_mix0", "sft_mix0.02", "sft_mix0.05", "sft_mix0.10", "llama_mix0", "llama_mix0.10", "seq_noreplay", "seq_replay1000"]}
QWEN, LLAMA = "Qwen2.5-1.5B", "Llama-3.2-3B"

def style(ax, title, ylabel):
    ax.set_title(title, loc="left", fontsize=10.5, color=INK); ax.set_ylabel(ylabel)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)

def errs(s, key):
    lo, hi = s[key + "_ci"]; return [[s[key] - lo], [hi - s[key]]]

# 1. headline: both models, baseline vs 10% goal mix, on both tasks
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
groups = [("Baseline (mix 0)", "sft_mix0", "llama_mix0"), ("10% goal mix", "sft_mix0.10", "llama_mix0.10")]
for ax, key, title in [(axes[0], "acc", "Old task: FeedbackESConv held-out\nperfect-flag accuracy"),
                       (axes[1], "det", "New task: goal-setting eval\ndetection balanced accuracy")]:
    for gi, (label, q, l) in enumerate(groups):
        for off, run, color in [(-0.19, q, BLUE), (0.19, l, ORANGE)]:
            s = RUNS[run]
            ax.bar(gi + off, s[key], 0.34, color=color, yerr=errs(s, key), error_kw={"ecolor": MUTED, "elinewidth": 1, "capsize": 3})
            ax.text(gi + off, min(s[key + "_ci"][1], 1.0) + 0.025, f"{s[key]:.3f}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([g[0] for g in groups]); ax.set_ylim(0, 1.12)
    style(ax, title, "score")
axes[0].axhline(MAJORITY, color=MUTED, linestyle=(0, (4, 3)), linewidth=1); axes[1].axhline(0.5, color=MUTED, linestyle=(0, (4, 3)), linewidth=1)
ref = plt.Line2D([0], [0], color=MUTED, linestyle=(0, (4, 3)), linewidth=1)
fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=ORANGE), ref], labels=[QWEN, LLAMA, "reference: majority class 0.583 (left) / chance 0.5 (right)"], loc="lower center", ncol=3, frameon=False, fontsize=9)
fig.suptitle("Two models, two tasks: the 10% goal mix lifts goal-setting detection; ESConv moves by about a point or less (bars: 95% CI)", x=0.01, ha="left", fontsize=11)
fig.tight_layout(rect=(0, 0.06, 1, 0.94)); fig.savefig(os.path.join(OUT, "fig1_model_comparison.png"), dpi=160); plt.close(fig)

# 2. mixing-ratio sweep
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
q = [("sft_mix0", 0), ("sft_mix0.02", 2), ("sft_mix0.05", 5), ("sft_mix0.10", 10)]; l = [("llama_mix0", 0), ("llama_mix0.10", 10)]
for ax, key, title in [(axes[0], "acc", "Old task: ESConv perfect-flag accuracy"), (axes[1], "det", "New task: goal detection balanced accuracy")]:
    for pts, color, name in [(q, BLUE, QWEN), (l, ORANGE, LLAMA)]:
        xs = [x for _, x in pts]; ys = [RUNS[r][key] for r, _ in pts]
        lo = [RUNS[r][key + "_ci"][0] for r, _ in pts]; hi = [RUNS[r][key + "_ci"][1] for r, _ in pts]
        ax.fill_between(xs, lo, hi, color=color, alpha=0.15, linewidth=0)
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=7, markeredgecolor=SURF, markeredgewidth=1.5)
        dy = {"acc": {QWEN: 0.012, LLAMA: -0.012}, "det": {QWEN: 0, LLAMA: 0}}[key][name]
        ax.text(xs[-1] + 0.3, ys[-1] + dy, name, color=INK, fontsize=9, va="center")
    ax.set_xlim(-0.5, 14); ax.set_xticks([0, 2, 5, 10]); ax.set_xticklabels(["0%", "2%", "5%", "10%"]); ax.set_xlabel("goal-setting rows as % of ESConv rows")
    style(ax, title, "score")
axes[0].set_ylim(0.8, 0.95); axes[1].set_ylim(0.45, 1.0)
fig.suptitle("Mixing-ratio sweep (shaded = 95% CI). Llama was trained at 0% and 10% only", x=0.01, ha="left", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(os.path.join(OUT, "fig2_mix_sweep.png"), dpi=160); plt.close(fig)

# 3. sequential fine-tuning vs mixed SFT (Qwen2.5-1.5B)
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
names = [("Baseline\n(mix 0)", "sft_mix0"), ("Mixed SFT\n10%", "sft_mix0.10"), ("Sequential\nno replay", "seq_noreplay"), ("Sequential\n+1000 replay", "seq_replay1000")]
for ax, key, title in [(axes[0], "acc", "Old task: ESConv perfect-flag accuracy"), (axes[1], "det", "New task: goal detection balanced accuracy")]:
    for i, (_, r) in enumerate(names):
        s = RUNS[r]; ax.bar(i, s[key], 0.6, color=BLUE, yerr=errs(s, key), error_kw={"ecolor": MUTED, "elinewidth": 1, "capsize": 3})
        ax.text(i, min(s[key + "_ci"][1], 1.0) + 0.025, f"{s[key]:.3f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(names))); ax.set_xticklabels([n for n, _ in names], fontsize=9); ax.set_ylim(0, 1.12)
    style(ax, title, "score")
axes[0].axhline(MAJORITY, color=MUTED, linestyle=(0, (4, 3)), linewidth=1)
fig.legend(handles=[plt.Line2D([0], [0], color=MUTED, linestyle=(0, (4, 3)), linewidth=1)], labels=["majority-class accuracy 0.583 (left panel)"], loc="lower center", frameon=False, fontsize=9)
fig.suptitle("Forgetting test (Qwen2.5-1.5B): sequential fine-tuning without replay forgets much of the old task", x=0.01, ha="left", fontsize=11)
fig.tight_layout(rect=(0, 0.05, 1, 0.94)); fig.savefig(os.path.join(OUT, "fig3_forgetting.png"), dpi=160); plt.close(fig)

# 4. training loss from the Modal logs
def losses(log):
    pts = []
    for line in open(os.path.join(LOGS, log), encoding="utf8", errors="ignore"):
        m = re.match(r"\{'loss'.*\}", line.strip())
        if m:
            try: d = ast.literal_eval(m.group(0)); pts.append((d["epoch"], d["loss"]))
            except Exception: pass
    return pts
fig, ax = plt.subplots(figsize=(8.5, 4.4))
for log, color, dash, label in [("train_mix0.log", BLUE, (0, (5, 3)), f"{QWEN}, mix 0"), ("train_sft_mix0.10.log", BLUE, "-", f"{QWEN}, mix 10%"),
                                ("train_llama_mix0.log", ORANGE, (0, (5, 3)), f"{LLAMA}, mix 0"), ("train_llama_mix0.10.log", ORANGE, "-", f"{LLAMA}, mix 10%")]:
    p = losses(log); ax.plot([x for x, _ in p], [y for _, y in p], color=color, linestyle=dash, linewidth=1.8, label=label)
ax.set_ylim(0, 1.6); ax.set_xlabel("epoch"); style(ax, "Training loss (response tokens only)", "loss")
ax.legend(frameon=False, fontsize=9)
fig.text(0.01, 0.01, "Mixed runs include extra goal-setting rows, so their loss is not directly comparable to mix 0.", fontsize=8.5, color=MUTED)
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(os.path.join(OUT, "fig4_training_loss.png"), dpi=160); plt.close(fig)
print("wrote", sorted(os.listdir(OUT)))
