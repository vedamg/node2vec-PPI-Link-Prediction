
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Locate the project and results files
PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_DIR / "results" / "repeated_runs"
FIGURES_DIR = PROJECT_DIR / "results" / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

summary_file = RESULTS_DIR / "repeated_run_mean_sd.csv"
df = pd.read_csv(summary_file)


# Set a consistent order and readable labels
method_order = [
    "baseline",
    "local",
    "outward",
    "common_neighbors"
]

method_labels = [
    "Baseline\n(p=1, q=1)",
    "Local\n(p=1, q=2)",
    "Outward\n(p=1, q=0.5)",
    "Common\nNeighbors"
]

sampling_order = ["random", "degree_matched"]

sampling_labels = [
    "Random negatives",
    "Degree-matched negatives"
]


def plot_metric(mean_column, std_column, ylabel, title, filename):
    """Plot mean performance with standard-deviation error bars."""

    x = np.arange(len(method_order))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, strategy in enumerate(sampling_order):
        subset = (
            df[df["Sampling_Strategy"] == strategy]
            .set_index("Experiment")
            .reindex(method_order)
        )

        means = subset[mean_column].to_numpy()
        stds = subset[std_column].to_numpy()

        positions = x + (i - 0.5) * width

        ax.bar(
            positions,
            means,
            width,
            yerr=stds,
            capsize=5,
            label=sampling_labels[i]
        )

    ax.set_xlabel("Method")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(method_labels)
    ax.set_ylim(0.88, 0.97)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=300)
    plt.close(fig)

    print(f"Saved: {FIGURES_DIR / filename}")


# Figure 1: ROC-AUC
plot_metric(
    mean_column="ROC_AUC_mean",
    std_column="ROC_AUC_std",
    ylabel="ROC-AUC (mean ± SD)",
    title="Link Prediction Performance: ROC-AUC",
    filename="repeated_run_roc_auc.png"
)

# Figure 2: Average Precision
plot_metric(
    mean_column="Average_Precision_mean",
    std_column="Average_Precision_std",
    ylabel="Average Precision (mean ± SD)",
    title="Link Prediction Performance: Average Precision",
    filename="repeated_run_average_precision.png"
)

print("\nAll repeated-run figures have been generated.")
