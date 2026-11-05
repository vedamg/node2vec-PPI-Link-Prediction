
import networkx as nx
import pandas as pd

from pathlib import Path
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score
)

# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

# ==========================================
# 2. LOAD TRAINING NETWORK
# ==========================================

print("Loading training interactions...")

train_edges = pd.read_csv(
    RESULTS_DIR / "train_edges.csv"
)

G = nx.from_pandas_edgelist(
    train_edges,
    source="protein1",
    target="protein2"
)

print("Training proteins:", G.number_of_nodes())
print("Training interactions:", G.number_of_edges())

# ==========================================
# 3. LOAD TEST DATA
# ==========================================

test_positive = pd.read_csv(
    RESULTS_DIR / "test_positive_edges.csv"
)

test_negative = pd.read_csv(
    RESULTS_DIR / "test_negative_edges.csv"
)

test_edges = pd.concat(
    [test_positive, test_negative],
    ignore_index=True
)

y_test = [
    1
] * len(test_positive) + [
    0
] * len(test_negative)

print("Positive test pairs:", len(test_positive))
print("Negative test pairs:", len(test_negative))

# ==========================================
# 4. CALCULATE COMMON NEIGHBOURS
# ==========================================

print("\nCalculating Common Neighbors scores...")

scores = []

for row in test_edges.itertuples(index=False):

    protein_a = str(row.protein1)
    protein_b = str(row.protein2)

    # Number of shared neighbours in training graph
    score = len(
        list(nx.common_neighbors(G, protein_a, protein_b))
    )

    scores.append(score)

# ==========================================
# 5. EVALUATE PERFORMANCE
# ==========================================

# Predict positive when at least one common neighbour exists.
y_pred = [
    1 if score > 0 else 0
    for score in scores
]

roc_auc = roc_auc_score(y_test, scores)

average_precision = average_precision_score(
    y_test, scores
)

accuracy = accuracy_score(
    y_test, y_pred
)

f1 = f1_score(
    y_test, y_pred
)

print("\n========== COMMON NEIGHBORS RESULTS ==========")

print("ROC-AUC:", round(roc_auc, 4))
print("Average Precision:", round(average_precision, 4))
print("Accuracy:", round(accuracy, 4))
print("F1-score:", round(f1, 4))

# ==========================================
# 6. SAVE PREDICTIONS
# ==========================================

predictions = test_edges.copy()

predictions["actual_label"] = y_test
predictions["common_neighbors_score"] = scores
predictions["predicted_label"] = y_pred

predictions.to_csv(
    RESULTS_DIR / "predictions_common_neighbors.csv",
    index=False
)

# ==========================================
# 7. SAVE COMPARISON RESULTS
# ==========================================

new_result = pd.DataFrame([{
    "Experiment": "common_neighbors",
    "ROC_AUC": roc_auc,
    "Average_Precision": average_precision,
    "Accuracy": accuracy,
    "F1_Score": f1
}])

comparison_file = RESULTS_DIR / "node2vec_comparison.csv"

existing_results = pd.read_csv(comparison_file)

# Avoid duplicate Common Neighbors rows if rerun
existing_results = existing_results[
    existing_results["Experiment"] != "common_neighbors"
]

updated_results = pd.concat(
    [existing_results, new_result],
    ignore_index=True
)

updated_results.to_csv(
    comparison_file,
    index=False
)

print("\nUpdated comparison:")
print(updated_results.round(4).to_string(index=False))

print("\nSaved predictions and updated comparison successfully.")
