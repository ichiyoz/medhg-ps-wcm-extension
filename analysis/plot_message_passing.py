"""Cartoon of message passing in 4 frames. Synthetic — no data.
Renders message_passing.png.   PYTHONPATH=. python3 analysis/plot_message_passing.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch

P = ["#d9534f", "#5a8fc7", "#5fae73"]   # patient card colors
GREEN = "#5fae73"
GRAY = "#cfd6dd"
NAMES = ["Ann", "Bob", "Cy"]


def card(ax, x, y, colors, w=0.62, h=0.34, ec="#333"):
    """A little 'data card' = one or more colored stripes."""
    n = len(colors)
    for i, c in enumerate(colors):
        ax.add_patch(FancyBboxPatch((x - w/2 + i*w/n, y - h/2), w/n, h,
                     boxstyle="square,pad=0", fc=c, ec="none", zorder=4))
    ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="square,pad=0",
                 fc="none", ec=ec, lw=1.3, zorder=5))


def node(ax, x, y, color, label, r=0.30):
    ax.add_patch(Circle((x, y), r, fc=color, ec="#333", lw=1.4, zorder=3))
    ax.text(x, y, label, ha="center", va="center", fontsize=9, fontweight="bold", zorder=6)


def edge(ax, p1, p2, color="#bbb", lw=1.0, arrow=False, style="-"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>" if arrow else "-",
                 color=color, lw=lw, mutation_scale=14, zorder=2,
                 connectionstyle="arc3", linestyle=style))


CPT = (0, 1.7)
PTS = [(-1.7, -0.2), (0, -0.2), (1.7, -0.2)]
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

def base(ax, title):
    node(ax, *CPT, GRAY, "CPT")
    ax.text(CPT[0], CPT[1] + 0.55, "shared procedure node", ha="center", fontsize=9, color="#777")
    for (x, y), nm in zip(PTS, NAMES):
        node(ax, x, y, "#9aa6b2", nm)
    ax.set_title(title, fontsize=13, fontweight="bold", loc="left")
    ax.set_xlim(-2.7, 2.7); ax.set_ylim(-1.5, 2.6); ax.axis("off"); ax.set_aspect("equal")

# Frame 1 — everyone holds their own card
ax = axes[0][0]; base(ax, "1.  Each patient holds its own data card")
for (x, y), c in zip(PTS, P):
    edge(ax, (x, y + 0.32), (CPT[0], CPT[1] - 0.30))
    card(ax, x, y - 0.62, [c])
ax.text(0, -1.35, "Ann was readmitted (red); Bob & Cy unknown", ha="center", fontsize=9, color="#555")

# Frame 2 — STEP 1: cards travel UP the edges into the node
ax = axes[0][1]; base(ax, "2.  Layer 1 — patients send cards UP")
for (x, y), c in zip(PTS, P):
    edge(ax, (x, y + 0.32), (CPT[0], CPT[1] - 0.30), color=c, lw=2.2, arrow=True)
    card(ax, (x + CPT[0])/2, (y + CPT[1])/2 + 0.1, [c], w=0.42, h=0.24)
ax.text(0, -1.35, "the node GATHERS every patient's card", ha="center", fontsize=9, color="#555")

# Frame 3 — node now holds the BLEND, sends it DOWN
ax = axes[1][0]; base(ax, "3.  Layer 2 — node averages, sends the blend DOWN")
card(ax, CPT[0] + 0.95, CPT[1], P, w=0.7, h=0.34)
ax.text(CPT[0] + 0.95, CPT[1] + 0.4, "blend", ha="center", fontsize=8, color="#777")
for (x, y), c in zip(PTS, P):
    edge(ax, (CPT[0], CPT[1] - 0.30), (x, y + 0.32), color="#8c7b66", lw=2.0, arrow=True)
    card(ax, x, y - 0.62, [c] + P)   # own card + the blend appended
ax.text(0, -1.35, "every patient now carries the blend of ALL peers", ha="center", fontsize=9, color="#555")

# Frame 4 — result: Bob now knows about Ann
ax = axes[1][1]; base(ax, "4.  Result — Bob's card now contains a hint of Ann")
for (x, y), c in zip(PTS, P):
    edge(ax, (x, y + 0.32), (CPT[0], CPT[1] - 0.30))
    card(ax, x, y - 0.62, [c] + P)
# highlight Bob
bx, by = PTS[1]
ax.add_patch(Circle((bx, by - 0.62), 0.0))
ax.annotate("Bob 'sees' Ann's red\nthrough the shared CPT",
            xy=(bx + 0.31, by - 0.62), xytext=(bx + 0.5, by - 1.25),
            fontsize=9, color="#c33",
            arrowprops=dict(arrowstyle="->", color="#c33"))
ax.text(0, 2.35, "↑ that cross-patient hint is the ENTIRE point of a GNN",
        ha="center", fontsize=9, color="#2a7", fontweight="bold")

plt.suptitle("Message passing = cards flow UP into shared nodes, then the blend flows back DOWN",
             fontsize=15, fontweight="bold", y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("message_passing.png", dpi=150, bbox_inches="tight")
print("wrote message_passing.png")
