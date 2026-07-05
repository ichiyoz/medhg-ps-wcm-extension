"""Conceptual visual of node DEGREE and hub dilution. Synthetic — no patient data.
Renders degree_dilution.png.   PYTHONPATH=. python analysis/plot_degree.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

RED = "#d9534f"      # readmitted patient
GRAY = "#9aa6b2"     # not readmitted
MUD = "#8c7b66"      # washed-out average
HUB = {"surgeon": "#e3b23c", "cpt": "#5fae73", "sex": "#5a8fc7"}


def spokes(ax, center, n_draw, color, label, degree, r=1.0):
    """Draw a hub node with n_draw spokes; label its true degree."""
    cx, cy = center
    ax.scatter([cx], [cy], s=1700, c=color, edgecolors="#333", linewidths=1.5, zorder=3)
    rng = np.random.default_rng(len(label))
    for k in range(n_draw):
        a = 2 * np.pi * k / n_draw
        x, y = cx + r * np.cos(a), cy + r * np.sin(a)
        ax.plot([cx, x], [cy, y], color="#bbb", lw=0.7, zorder=1)
        c = RED if rng.random() < 0.1 else GRAY
        ax.scatter([x], [y], s=42, c=c, edgecolors="none", zorder=2)
    ax.text(cx, cy, label, ha="center", va="center", fontsize=10, fontweight="bold")
    ax.text(cx, cy - r - 0.55, f"degree ≈ {degree:,}", ha="center", fontsize=12,
            fontweight="bold", color="#333")
    ax.text(cx, cy - r - 0.92, f"{n_draw} spokes shown", ha="center", fontsize=8, color="#888")


fig = plt.figure(figsize=(14, 9))

# ---- Row 1: degree = number of connections ---------------------------------
ax1 = fig.add_subplot(2, 1, 1)
ax1.set_title("Degree = how many patients connect to a node  (each dot = 1 patient,  ● = readmitted)",
              fontsize=14, fontweight="bold", loc="left")
spokes(ax1, (-4.2, 0), 13, HUB["surgeon"], "one\nsurgeon", 13, r=1.15)
spokes(ax1, (0.4, 0), 40, HUB["cpt"], "one\nCPT", 4000, r=1.35)
spokes(ax1, (5.2, 0), 70, HUB["sex"], "'male'\nnode", 22000, r=1.6)
ax1.text(-4.2, 1.9, "small hub", ha="center", fontsize=11, color="#2a7")
ax1.text(0.4, 2.2, "big hub", ha="center", fontsize=11, color="#c80")
ax1.text(5.2, 2.5, "giant hub", ha="center", fontsize=11, color="#c33")
ax1.set_xlim(-6.2, 7.4); ax1.set_ylim(-3.0, 3.0); ax1.axis("off")
ax1.set_aspect("equal")

# ---- Row 2: dilution = averaging washes the signal out ---------------------
ax2 = fig.add_subplot(2, 1, 2)
ax2.set_title("Why high degree hurts: the node AVERAGES everyone, then hands the mush back",
              fontsize=14, fontweight="bold", loc="left")

# left: incoming colored dots
rng = np.random.default_rng(7)
xs = np.linspace(-9, -6.5, 1);
ys = np.linspace(2.4, -2.4, 12)
for i, y in enumerate(ys):
    c = RED if i in (2, 9) else GRAY
    ax2.scatter([-8.5], [y], s=80, c=c, edgecolors="none")
    ax2.add_patch(FancyArrowPatch((-8.2, y), (-4.6, 0), arrowstyle="-", color="#ccc", lw=0.7))
ax2.text(-8.5, 3.0, "4,000 patients\n(mostly not readmitted)", ha="center", fontsize=10)

# middle: the hub = muddy average
ax2.scatter([-4.0], [0], s=2600, c=MUD, edgecolors="#333", linewidths=1.5)
ax2.text(-4.0, 0, "average\n= mud", ha="center", va="center", fontsize=10, fontweight="bold", color="white")
ax2.text(-4.0, -2.4, "CPT node pools\nall 4,000 into ONE vector", ha="center", fontsize=10)

# right: mush sent back to everyone
for i, y in enumerate(ys):
    ax2.add_patch(FancyArrowPatch((-3.4, 0), (0.2, y), arrowstyle="->", color=MUD, lw=1.0,
                                  mutation_scale=10))
    ax2.scatter([0.5], [y], s=80, c=MUD, edgecolors="none")
ax2.text(0.5, 3.0, "every patient now\nlooks the same", ha="center", fontsize=10, color=MUD)

# far right: contrast with the tree
ax2.plot([4.2, 4.2], [-2.4, 2.4], color="#333", lw=1.0)
ax2.text(5.6, 2.0, "A TREE instead:", ha="center", fontsize=11, fontweight="bold")
ax2.text(5.6, 1.2, "reads 'CPT = 43775'\nas a sharp split —", ha="center", fontsize=10)
ax2.text(5.6, 0.2, "keeps each patient's\nsignal intact.", ha="center", fontsize=10)
ax2.scatter([4.9, 5.3, 5.7, 6.1], [-1.0]*4, s=80,
            c=[RED, GRAY, GRAY, RED], edgecolors="none")
ax2.text(5.6, -1.8, "no averaging", ha="center", fontsize=9, color="#2a7")

ax2.set_xlim(-10.5, 7.5); ax2.set_ylim(-3.4, 3.4); ax2.axis("off")

# legend
from matplotlib.lines import Line2D
fig.legend(handles=[
    Line2D([0], [0], marker="o", color="w", markerfacecolor=RED, markersize=11, label="readmitted patient"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=GRAY, markersize=11, label="not readmitted"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=MUD, markersize=11, label="washed-out average"),
], loc="lower center", ncol=3, fontsize=10, frameon=False)

plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig("degree_dilution.png", dpi=150, bbox_inches="tight")
print("wrote degree_dilution.png")
