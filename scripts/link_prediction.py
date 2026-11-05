
import random
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ==========================================
# 1. CONFIGURATION
# ==========================================

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

# The three embedding experiments
EXPERIMENTS = [
    "baseline",
    "local",
    "outward"
]


# ==========================================
# 2. LOAD TRAINING AND TEST EDGES
# ==========================================

print("Loading edge datasets...")

train_edges = pd.read_csv(
    RESULTS_DIR / "train_edges.csv"
)

test_positive = pd.read_csv(
    RESULTS_DIR / "test_positive_edges.csv"
)

test_negative = pd.read_csv(
    RESULTS_DIR / "test_negative_edges.csv"
)

print("Training positive edges:", len(train_edges))
print("Test positive edges:", len(test_positive))
print("Test negative edges:", len(test_negative))


# ==========================================
# 3. GENERATE TRAINING NEGATIVE EDGES
# ==========================================

# Combine training and held-out positive edges.
# These represent all known positive interactions.

all_positive_df = pd.concat(
    [train_edges, test_positive],
    ignore_index=True
)

all_positive_edges = {
    tuple(sorted((str(row.protein1), str(row.protein2))))
    for row in all_positive_df.itertuples(index=False)
}

test_negative_edges = {
    tuple(sorted((str(row.protein1), str(row.protein2))))
    for row in test_negative.itertuples(index=False)
}

# Use the same protein set as the training network.
nodes = sorted(
    set(train_edges["protein1"].astype(str))
    | set(train_edges["protein2"].astype(str))
)

# Sample training negatives separately from test negatives.
rng = random.Random(RANDOM_SEED)

train_negative_edges = set()

target_count = len(train_edges)

while len(train_negative_edges) < target_count:

    u, v = rng.sample(nodes, 2)
    pair = tuple(sorted((u, v)))

    if pair in all_positive_edges:
        continue

    if pair in test_negative_edges:
        continue

    train_negative_edges.add(pair)

train_negative = pd.DataFrame(
    list(train_negative_edges),
    columns=["protein1", "protein2"]
)

train_negative.to_csv(
    RESULTS_DIR / "train_negative_edges.csv",
    index=False
)

print("\nTraining negative edges generated:", len(train_negative))


# ==========================================
# 4. CREATE EDGE FEATURES
# ==========================================

def create_edge_features(edge_df, embeddings):
    """
    Create Hadamard-product features for protein pairs.

    Each edge feature is the element-wise product
    of the two protein embedding vectors.
    """

    features = []

    for row in edge_df.itertuples(index=False):

        protein_a = str(row.protein1)
        protein_b = str(row.protein2)

        if protein_a not in embeddings.index:
            raise ValueError(
                f"Embedding missing for {protein_a}"
            )

        if protein_b not in embeddings.index:
            raise ValueError(
                f"Embedding missing for {protein_b}"
            )

        vector_a = embeddings.loc[protein_a].to_numpy(
            dtype=float
        )

        vector_b = embeddings.loc[protein_b].to_numpy(
            dtype=float
        )

        # Hadamard product
        edge_vector = vector_a * vector_b

        features.append(edge_vector)

    return np.array(features)


# ==========================================
# 5. PREPARE LABELS
# ==========================================

# Training:
# 1 = known interaction
# 0 = sampled unobserved pair

X_train_edges = pd.concat(
    [train_edges, train_negative],
    ignore_index=True
)

y_train = np.concatenate([
    np.ones(len(train_edges)),
    np.zeros(len(train_negative))
])

# Testing:
# 1 = held-out known interaction
# 0 = sampled unobserved pair

X_test_edges = pd.concat(
    [test_positive, test_negative],
    ignore_index=True
)

y_test = np.concatenate([
    np.ones(len(test_positive)),
    np.zeros(len(test_negative))
])


# ==========================================
# 6. RUN LINK PREDICTION EXPERIMENTS
# ==========================================

all_results = []

for experiment in EXPERIMENTS:

    print("\n===================================")
    print("Evaluating:", experiment)
    print("===================================")

    embedding_file = (
        RESULTS_DIR / f"embeddings_{experiment}.csv"
    )

    embeddings = pd.read_csv(
        embedding_file,
        index_col=0
    )

    embeddings.index = embeddings.index.astype(str)

    print("Loaded embeddings:", embeddings.shape)

    # Generate Hadamard edge features
    X_train = create_edge_features(
        X_train_edges,
        embeddings
    )

    X_test = create_edge_features(
        X_test_edges,
        embeddings
    )

    print("Training feature shape:", X_train.shape)
    print("Testing feature shape:", X_test.shape)

    # Standardize features and train logistic regression
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_SEED
        )
    )

    classifier.fit(X_train, y_train)

    # Predict probabilities
    y_prob = classifier.predict_proba(X_test)[:, 1]

    # Convert probabilities into class predictions
    y_pred = (y_prob >= 0.5).astype(int)

    # Evaluation metrics
    roc_auc = roc_auc_score(y_test, y_prob)

    average_precision = average_precision_score(
        y_test,
        y_prob
    )

    accuracy = accuracy_score(y_test, y_pred)

    f1 = f1_score(y_test, y_pred)

    print("\nResults:")
    print("ROC-AUC:", round(roc_auc, 4))
    print("Average Precision:", round(average_precision, 4))
    print("Accuracy:", round(accuracy, 4))
    print("F1-score:", round(f1, 4))

    all_results.append({
        "Experiment": experiment,
        "ROC_AUC": roc_auc,
        "Average_Precision": average_precision,
        "Accuracy": accuracy,
        "F1_Score": f1
    })

    # Save test predictions
    predictions = X_test_edges.copy()
    predictions["actual_label"] = y_test
    predictions["predicted_probability"] = y_prob
    predictions["predicted_label"] = y_pred

    predictions.to_csv(
        RESULTS_DIR / f"predictions_{experiment}.csv",
        index=False
    )


# ==========================================
# 7. SAVE AND DISPLAY COMPARISON
# ==========================================

results_df = pd.DataFrame(all_results)

results_df.to_csv(
    RESULTS_DIR / "node2vec_comparison.csv",
    index=False
)

print("\n===================================")
print("FINAL NODE2VEC COMPARISON")
print("===================================")

print(
    results_df.round(4).to_string(index=False)
)

print("\nResults saved to:")
print(RESULTS_DIR / "node2vec_comparison.csv")
