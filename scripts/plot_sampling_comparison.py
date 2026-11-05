
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

# Load both experiment results
random_results = pd.read_csv(
    RESULTS_DIR / "node2vec_comparison.csv"
)

matched_results = pd.read_csv(
    RESULTS_DIR / "degree_matched_comparison.csv"
)

# Standardize method names for merging
random_results["Experiment"] = (
    random_results["Experiment"].str.lower()
)

matched_results["Experiment"] = (
    matched_results["Experiment"].str.lower()
)

methods = [
    "baseline",
    "local",
    "outward",
    "common_neighbors"
]

labels = [
    "node2vec baseline",
    "node2vec local",
    "node2vec outward",
    "Common Neighbors"
]

metrics = [
    ("ROC_AUC", "ROC-AUC"),
    ("Average_Precision", "Average Precision")
]


# ==========================================
# 1. COMPARE RANDOM VS DEGREE-MATCHED
# ==========================================

for metric, metric_label in metrics:

    random_values = [
        random_results.loc[
            random_results["Experiment"] == method,
            metric
        ].iloc[0]
        for method in methods
    ]

    matched_values = [
        matched_results.loc[
            matched_results["Experiment"] == method,
            metric
        ].iloc[0]
        for method in methods
    ]

    x = np.arange(len(methods))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(
        x - width / 2,
        random_values,
        width,
        label="Random negatives"
    )

    ax.bar(
        x + width / 2,
        matched_values,
        width,
        label="Degree-matched negatives"
    )

    ax.set_ylabel(metric_label)
    ax.set_title(
        f"{metric_label}: Random vs Degree-Matched Negatives"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")

    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for i, value in enumerate(random_values):
        ax.text(
            i - width / 2,
            value + 0.008,
            f"{value:.3f}",
            ha="center",
            fontsize=8
        )

    for i, value in enumerate(matched_values):
        ax.text(
            i + width / 2,
            value + 0.008,
            f"{value:.3f}",
            ha="center",
            fontsize=8
        )

    plt.tight_layout()

    filename = (
        f"sampling_comparison_{metric.lower()}.png"
    )

    plt.savefig(
        RESULTS_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()


# ==========================================
# 2. SAVE COMBINED COMPARISON TABLE
# ==========================================

combined = random_results[
    ["Experiment", "ROC_AUC", "Average_Precision"]
].merge(
    matched_results[
        ["Experiment", "ROC_AUC", "Average_Precision"]
    ],
    on="Experiment",
    suffixes=("_random", "_degree_matched")
)

combined["ROC_AUC_change"] = (
    combined["ROC_AUC_degree_matched"]
    - combined["ROC_AUC_random"]
)

combined["AP_change"] = (
    combined["Average_Precision_degree_matched"]
    - combined["Average_Precision_random"]
)

combined.to_csv(
    RESULTS_DIR / "combined_sampling_comparison.csv",
    index=False
)

print("\n========== COMBINED COMPARISON ==========")
print(combined.to_string(index=False))

print("\nPlots and comparison table saved successfully!")
