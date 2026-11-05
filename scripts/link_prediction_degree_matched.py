
import random
from pathlib import Path

import pandas as pd
import networkx as nx

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score
)


# ==========================================
# 1. CONFIGURATION
# ==========================================

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"


# ==========================================
# 2. LOAD DATA
# ==========================================

print("Loading datasets...")

G = nx.read_graphml(
    RESULTS_DIR / "training_network.graphml"
)

train_positive = pd.read_csv(
    RESULTS_DIR / "train_edges.csv"
)

train_negative = pd.read_csv(
    RESULTS_DIR / "train_negative_degree_matched.csv"
)

test_positive = pd.read_csv(
    RESULTS_DIR / "test_positive_edges.csv"
)

test_negative = pd.read_csv(
    RESULTS_DIR / "test_negative_degree_matched.csv"
)

print("Training network proteins:", G.number_of_nodes())
print("Training network interactions:", G.number_of_edges())

print("Training positive pairs:", len(train_positive))
print("Training negative pairs:", len(train_negative))
print("Test positive pairs:", len(test_positive))
print("Test negative pairs:", len(test_negative))


# ==========================================
# 3. PREPARE TRAINING AND TEST LABELS
# ==========================================

train_edges = pd.concat(
    [train_positive, train_negative],
    ignore_index=True
)

y_train = [
    1
] * len(train_positive) + [
    0
] * len(train_negative)

test_edges = pd.concat(
    [test_positive, test_negative],
    ignore_index=True
)

y_test = [
    1
] * len(test_positive) + [
    0
] * len(test_negative)


# ==========================================
# 4. NODE2VEC LINK PREDICTION
# ==========================================

def load_embeddings(filename):
    """Load a protein embedding CSV."""

    df = pd.read_csv(
        RESULTS_DIR / filename,
        index_col=0
    )

    df.index = df.index.astype(str)

    return df


def create_edge_features(edges, embeddings):
    """
    Create edge features using the Hadamard product
    of the two endpoint protein embeddings.
    """

    features = []

    for row in edges.itertuples(index=False):

        u = str(row.protein1)
        v = str(row.protein2)

        if u not in embeddings.index or v not in embeddings.index:
            raise ValueError(
                f"Missing embedding for protein pair: {u}, {v}"
            )

        vector_u = embeddings.loc[u].to_numpy(dtype=float)
        vector_v = embeddings.loc[v].to_numpy(dtype=float)

        features.append(vector_u * vector_v)

    return features


def evaluate_node2vec(experiment, embedding_file):

    print(f"\n========== {experiment.upper()} ==========")

    embeddings = load_embeddings(embedding_file)

    X_train = create_edge_features(
        train_edges,
        embeddings
    )

    X_test = create_edge_features(
        test_edges,
        embeddings
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_SEED
        ))
    ])

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    auc = roc_auc_score(y_test, probabilities)
    ap = average_precision_score(y_test, probabilities)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print("ROC-AUC:", round(auc, 4))
    print("Average Precision:", round(ap, 4))
    print("Accuracy:", round(accuracy, 4))
    print("F1-score:", round(f1, 4))

    prediction_df = test_edges.copy()
    prediction_df["true_label"] = y_test
    prediction_df["predicted_probability"] = probabilities
    prediction_df["predicted_label"] = predictions

    output_file = (
        RESULTS_DIR /
        f"predictions_degree_matched_{experiment}.csv"
    )

    prediction_df.to_csv(output_file, index=False)

    return {
        "Experiment": experiment,
        "ROC_AUC": auc,
        "Average_Precision": ap,
        "Accuracy": accuracy,
        "F1_Score": f1
    }


# ==========================================
# 5. RUN THREE NODE2VEC EXPERIMENTS
# ==========================================

results = []

results.append(
    evaluate_node2vec(
        "baseline",
        "embeddings_baseline.csv"
    )
)

results.append(
    evaluate_node2vec(
        "local",
        "embeddings_local.csv"
    )
)

results.append(
    evaluate_node2vec(
        "outward",
        "embeddings_outward.csv"
    )
)


# ==========================================
# 6. COMMON NEIGHBORS BASELINE
# ==========================================

print("\n========== COMMON NEIGHBORS ==========")

def common_neighbor_scores(edges):

    scores = []

    for row in edges.itertuples(index=False):

        u = str(row.protein1)
        v = str(row.protein2)

        score = len(list(nx.common_neighbors(G, u, v)))

        scores.append(score)

    return scores


cn_scores = common_neighbor_scores(test_edges)

cn_auc = roc_auc_score(y_test, cn_scores)
cn_ap = average_precision_score(y_test, cn_scores)

# A positive prediction means at least one common neighbor.
cn_predictions = [
    1 if score > 0 else 0
    for score in cn_scores
]

cn_accuracy = accuracy_score(y_test, cn_predictions)
cn_f1 = f1_score(y_test, cn_predictions)

print("ROC-AUC:", round(cn_auc, 4))
print("Average Precision:", round(cn_ap, 4))
print("Accuracy:", round(cn_accuracy, 4))
print("F1-score:", round(cn_f1, 4))

cn_prediction_df = test_edges.copy()
cn_prediction_df["true_label"] = y_test
cn_prediction_df["common_neighbors"] = cn_scores
cn_prediction_df["predicted_label"] = cn_predictions

cn_prediction_df.to_csv(
    RESULTS_DIR / "predictions_degree_matched_common_neighbors.csv",
    index=False
)

results.append({
    "Experiment": "common_neighbors",
    "ROC_AUC": cn_auc,
    "Average_Precision": cn_ap,
    "Accuracy": cn_accuracy,
    "F1_Score": cn_f1
})


# ==========================================
# 7. SAVE COMPARISON RESULTS
# ==========================================

comparison = pd.DataFrame(results)

comparison = comparison.sort_values(
    by="ROC_AUC",
    ascending=False
)

comparison.to_csv(
    RESULTS_DIR / "degree_matched_comparison.csv",
    index=False
)

print("\n========== FINAL DEGREE-MATCHED COMPARISON ==========")
print(comparison.to_string(index=False))

print("\nResults saved to:")
print(RESULTS_DIR / "degree_matched_comparison.csv")
print("\nDegree-matched link prediction completed!")
