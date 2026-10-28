import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

BLACK = "#0B0B0F"
PANEL = "#14141B"
CRIMSON = "#C1121F"
BLOOD = "#7A0C16"
ROSE = "#E63946"
GOLD = "#D4AF37"
IVORY = "#F1E9E4"
GREY = "#8A8A93"

PALETTE = [CRIMSON, GOLD, ROSE, BLOOD, IVORY, GREY]
HEAT_CMAP = LinearSegmentedColormap.from_list("drama_heat", [BLACK, BLOOD, CRIMSON, GOLD])
DIVERGING_CMAP = LinearSegmentedColormap.from_list("drama_div", [GOLD, BLACK, CRIMSON])


def set_theme():
    mpl.rcParams.update({
        "figure.facecolor": BLACK,
        "axes.facecolor": PANEL,
        "savefig.facecolor": BLACK,
        "axes.edgecolor": GREY,
        "axes.labelcolor": IVORY,
        "axes.titlecolor": GOLD,
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
        "grid.color": "#2A2A33",
        "grid.linestyle": "--",
        "grid.alpha": 0.6,
        "text.color": IVORY,
        "xtick.color": IVORY,
        "ytick.color": IVORY,
        "font.family": "serif",
        "legend.facecolor": PANEL,
        "legend.edgecolor": GREY,
        "legend.labelcolor": IVORY,
        "figure.dpi": 120,
    })
    sns.set_palette(PALETTE)


def headline(ax, title, subtitle=None):
    ax.set_title(title, loc="left", pad=24)
    if subtitle:
        ax.text(0, 1.03, subtitle, transform=ax.transAxes, color=GREY, fontsize=10, style="italic")


if __name__ == "__main__":
    set_theme()
    rng = np.random.default_rng(7)
    sample = rng.lognormal(mean=4, sigma=1, size=800)
    corr = np.corrcoef(rng.normal(size=(5, 100)))
    labels = ["pages", "year", "editions", "ratings", "want-to-read"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.hist(sample, bins=40, color=CRIMSON, edgecolor=BLACK)
    headline(ax1, "Who Gets Wanted?", "Sample preview of the project theme")
    ax1.set_xlabel("want-to-read count")
    ax1.set_ylabel("books")

    sns.heatmap(corr, cmap=DIVERGING_CMAP, annot=True, fmt=".2f", cbar=False,
                xticklabels=labels, yticklabels=labels, linewidths=1, linecolor=BLACK,
                annot_kws={"color": IVORY}, ax=ax2)
    headline(ax2, "Correlation of Desire", "Sample preview of the project theme")

    fig.tight_layout()
    fig.savefig("reports/theme_preview.png")
    print("Saved reports/theme_preview.png")