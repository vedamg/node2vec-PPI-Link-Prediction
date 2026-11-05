
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================
# 1. LOAD RESULTS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

df = pd.read_csv(
    RESULTS_DIR / "node2vec_comparison.csv"
)

# Arrange methods in performance order
df = df.sort_values(
    by="ROC_AUC",
    ascending=False
)

print(df.round(4).to_string(index=False))


# ==========================================
# 2. ROC-AUC COMPARISON
# ==========================================

plt.figure(figsize=(9, 5))

plt.bar(
    df["Experiment"],
    df["ROC_AUC"]
)

plt.ylabel("ROC-AUC")
plt.xlabel("Method")
plt.title("ROC-AUC Comparison of Link Prediction Methods")
plt.ylim(0, 1)

plt.xticks(rotation=20)
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "roc_auc_comparison.png",
    dpi=300
)

plt.show()


# ==========================================
# 3. AVERAGE PRECISION COMPARISON
# ==========================================

plt.figure(figsize=(9, 5))

plt.bar(
    df["Experiment"],
    df["Average_Precision"]
)

plt.ylabel("Average Precision")
plt.xlabel("Method")
plt.title("Average Precision Comparison of Link Prediction Methods")
plt.ylim(0, 1)

plt.xticks(rotation=20)
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "average_precision_comparison.png",
    dpi=300
)

plt.show()


# ==========================================
# 4. NODE2VEC PARAMETER COMPARISON
# ==========================================

node2vec_df = df[
    df["Experiment"].isin(
        ["baseline", "local", "outward"]
    )
].copy()

plt.figure(figsize=(8, 5))

plt.bar(
    node2vec_df["Experiment"],
    node2vec_df["ROC_AUC"]
)

plt.ylabel("ROC-AUC")
plt.xlabel("node2vec configuration")
plt.title("Effect of Random-Walk Parameters on node2vec")
plt.ylim(0, 1)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "node2vec_parameter_comparison.png",
    dpi=300
)

plt.show()

print("\nAll comparison plots saved successfully!")
