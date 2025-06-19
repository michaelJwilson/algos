import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

# Load the CSV file (remove '#' if present in header)
df = pd.read_csv(
    "data/node_marginals.csv",
    comment="#",
    sep="\t",
    names=["id", "x", "y", "bp_marginal", "felsenstein_marginal"],
)


def parse_vec(s):
    # Remove brackets and split
    return np.array([float(x) for x in s.strip("[]").replace("\t", ",").split(",")])


# exit(0)

df["bp_marginal"] = df["bp_marginal"].apply(parse_vec)
df["felsenstein_marginal"] = df["felsenstein_marginal"].apply(parse_vec)

# Infer nleaves and nancestors
nleaves = sum(df["y"] == 0)
nancestors = len(df) - nleaves


def compute_edges(nleaves, nancestors):
    nnodes = nleaves + nancestors
    edges = []
    for p in range(nleaves, nnodes):
        left = 2 * (p - nleaves)
        right = 2 * (p - nleaves) + 1
        if left < nnodes:
            edges.append((p, left))
        if right < nnodes:
            edges.append((p, right))
    return edges


edges = compute_edges(nleaves, nancestors)

# Set up figure
fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharey=True, constrained_layout=True)

titles = ["Belief Propagation", "Felsenstein"]
cols = ["bp_marginal", "felsenstein_marginal"]

for ax, col, title in zip(axes, cols, titles):
    for parent, child in edges:
        x0, y0 = df.loc[parent, ["x", "y"]]
        x1, y1 = df.loc[child, ["x", "y"]]

        ax.plot([x0, x1], [y0, y1], color="gray", lw=1.5, zorder=1, alpha=0.7)

    for _, row in df.iterrows():
        x, y = row["x"], row["y"]

        rgb = row[col][:3]
        rgb = np.clip(rgb, 0, 1)

        ax.scatter(x, y, color=rgb, s=180, edgecolor="k", linewidth=1.2, zorder=2)

    ax.set_title(title, fontsize=16, pad=12)
    ax.set_xlabel("$x$", fontsize=14)

    ax.tick_params(axis="both", which="major", labelsize=12)

    ax.grid(False)

axes[0].set_ylabel("$y$", fontsize=14)

fig.suptitle("Node Marginals on (Ultrametric) Binary Tree", fontsize=18, y=1.05)

for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.savefig("plots/tree_marginals.pdf", bbox_inches="tight")
