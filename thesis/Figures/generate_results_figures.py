from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parent
BLUE = "#2878B5"
ORANGE = "#F28E2B"
GREEN = "#2E8B57"
RED = "#C44E52"


def finish(fig, filename):
    fig.tight_layout()
    fig.savefig(OUT / filename, bbox_inches="tight")
    plt.close(fig)


plt.rcParams.update({
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

frames = np.arange(8)
step250 = np.array([0.9921, 0.9889, 0.9912, 0.9911, 0.9889, 0.9874, 0.9872, 0.9921])
step6000 = np.array([0.9938, 0.9915, 0.9918, 0.9916, 0.9896, 0.9885, 0.9886, 0.9952])
fig, ax = plt.subplots(figsize=(6.4, 3.7))
ax.plot(frames, step250, "o-", color=BLUE, label="Step 250")
ax.plot(frames, step6000, "o-", color=ORANGE, label="Step 6,000")
ax.set(xlabel="Frame", ylabel="Cosine similarity", xticks=frames, ylim=(0.985, 0.997))
ax.grid(axis="y", alpha=0.25)
ax.legend(frameon=False)
finish(fig, "fig1_fidelity_convergence.png")

clips = [
    "bike-packing", "blackswan", "bmx-trees", "breakdance", "camel",
    "car-roundabout", "car-shadow", "cows", "dance-twirl", "dog",
    "dogs-jump", "drift-chicane", "drift-straight", "goat", "gold-fish",
    "horsejump-high", "india", "judo", "kite-surf", "lab-coat", "libby",
    "loading", "mbike-trick", "motocross-jump", "paragliding-launch",
    "parkour", "pigs", "scooter-black", "shooting", "soapbox",
]
temporal_2d = np.array([
    0.879, 0.917, 0.810, 0.875, 0.902, 0.893, 0.927, 0.914, 0.869, 0.807,
    0.916, 0.906, 0.812, 0.859, 0.911, 0.895, 0.764, 0.901, 0.891, 0.858,
    0.856, 0.846, 0.886, 0.681, 0.909, 0.819, 0.908, 0.867, 0.776, 0.826,
])
temporal_3d = np.array([
    0.974, 0.971, 0.928, 0.978, 0.973, 0.950, 0.967, 0.978, 0.952, 0.925,
    0.975, 0.975, 0.901, 0.952, 0.976, 0.961, 0.896, 0.986, 0.958, 0.932,
    0.935, 0.933, 0.964, 0.805, 0.966, 0.910, 0.976, 0.924, 0.885, 0.908,
])
temporal_delta = temporal_3d - temporal_2d
fig, ax = plt.subplots(figsize=(5.2, 4.6))
ax.scatter(temporal_2d, temporal_3d, color=BLUE, alpha=0.85)
limits = (0.65, 1.0)
ax.plot(limits, limits, "--", color="0.45", linewidth=1, label="$y=x$")
for name in ["india", "motocross-jump", "bmx-trees", "dog"]:
    i = clips.index(name)
    ax.annotate(name, (temporal_2d[i], temporal_3d[i]), xytext=(4, 4),
                textcoords="offset points", fontsize=7)
ax.set(xlabel="2D AnyUp temporal consistency", ylabel="AnyUp3D temporal consistency",
       xlim=limits, ylim=limits)
ax.grid(alpha=0.2)
ax.legend(frameon=False)
finish(fig, "fig2_temporal_scatter.pdf")

order = np.argsort(temporal_delta)
fig, ax = plt.subplots(figsize=(6.5, 7.2))
colors = [ORANGE if clips[i] in {"india", "motocross-jump", "bmx-trees", "dog"} else BLUE for i in order]
ax.barh(np.array(clips)[order], temporal_delta[order], color=colors)
ax.set(xlabel="Improvement in temporal consistency")
ax.axvline(0, color="0.3", linewidth=0.8)
ax.grid(axis="x", alpha=0.2)
finish(fig, "fig3_temporal_delta_bar.pdf")

metrics = ["J", "F", "J&F"]
bilinear_agg = np.array([0.272, 0.260, 0.266])
anyup3d_agg = np.array([0.330, 0.365, 0.347])
x = np.arange(len(metrics))
fig, ax = plt.subplots(figsize=(5.8, 3.8))
width = 0.34
ax.bar(x - width / 2, bilinear_agg, width, label="Bilinear", color=BLUE)
ax.bar(x + width / 2, anyup3d_agg, width, label="AnyUp3D", color=ORANGE)
ax.set(ylabel="Score", xticks=x, xticklabels=metrics, ylim=(0, 0.42))
ax.grid(axis="y", alpha=0.2)
ax.legend(frameon=False)
finish(fig, "fig4_jf_aggregate_bar.pdf")

bilinear_jf = np.array([
    0.324, 0.506, 0.114, 0.260, 0.397, 0.566, 0.522, 0.522, 0.300, 0.437,
    0.163, 0.234, 0.154, 0.251, 0.278, 0.157, 0.216, 0.392, 0.136, 0.182,
    0.250, 0.275, 0.078, 0.194, 0.143, 0.112, 0.324, 0.215, 0.233, 0.122,
])
anyup3d_jf = np.array([
    0.413, 0.550, 0.193, 0.291, 0.538, 0.608, 0.488, 0.636, 0.457, 0.517,
    0.212, 0.153, 0.162, 0.313, 0.458, 0.275, 0.304, 0.399, 0.199, 0.216,
    0.427, 0.275, 0.158, 0.292, 0.263, 0.242, 0.449, 0.298, 0.299, 0.186,
])
jf_delta = anyup3d_jf - bilinear_jf
fig, ax = plt.subplots(figsize=(5.2, 4.6))
point_colors = np.where(jf_delta < 0, RED, np.where(np.isclose(jf_delta, 0), "0.5", GREEN))
ax.scatter(bilinear_jf, anyup3d_jf, c=point_colors, alpha=0.9)
limits = (0.05, 0.66)
ax.plot(limits, limits, "--", color="0.45", linewidth=1)
for name in ["car-shadow", "drift-chicane", "loading"]:
    i = clips.index(name)
    ax.annotate(name, (bilinear_jf[i], anyup3d_jf[i]), xytext=(4, 4),
                textcoords="offset points", fontsize=7)
ax.set(xlabel="Bilinear J&F", ylabel="AnyUp3D J&F", xlim=limits, ylim=limits)
ax.grid(alpha=0.2)
finish(fig, "fig5_jf_scatter.pdf")

order = np.argsort(jf_delta)
fig, ax = plt.subplots(figsize=(6.5, 7.2))
colors = np.where(jf_delta[order] < 0, RED, GREEN)
ax.barh(np.array(clips)[order], jf_delta[order], color=colors)
ax.axvline(0, color="0.3", linewidth=0.8)
ax.set(xlabel="Change in J&F (AnyUp3D − bilinear)")
ax.grid(axis="x", alpha=0.2)
finish(fig, "fig6_jf_delta_bar.pdf")

fig, ax = plt.subplots(figsize=(4.8, 3.7))
params = np.array([878080, 1540736]) / 1e6
bars = ax.bar(["2D AnyUp", "AnyUp3D"], params, color=[BLUE, ORANGE])
ax.bar_label(bars, fmt="%.3f M", padding=3)
ax.set(ylabel="Learnable parameters (millions)", ylim=(0, 1.75))
ax.grid(axis="y", alpha=0.2)
finish(fig, "fig7_cost_params.pdf")

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))
latency = [439.4, 1189.4]
memory = [0.51, 5.39]
bars = axes[0].bar(["2D AnyUp", "AnyUp3D"], latency, color=[BLUE, ORANGE])
axes[0].bar_label(bars, fmt="%.1f", padding=3)
axes[0].set(ylabel="Latency per 8-frame clip (ms)", ylim=(0, 1350))
bars = axes[1].bar(["2D AnyUp", "AnyUp3D"], memory, color=[BLUE, ORANGE])
axes[1].bar_label(bars, fmt="%.2f", padding=3)
axes[1].set(ylabel="Peak GPU memory (GB)", ylim=(0, 6.1))
for ax in axes:
    ax.grid(axis="y", alpha=0.2)
finish(fig, "fig8_cost_latency_memory.pdf")