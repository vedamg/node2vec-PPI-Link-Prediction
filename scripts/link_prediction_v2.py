
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

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

SEEDS = [42, 123, 2026]

EXPERIMENTS = ["baseline", "local", "outward"]

BASE_DIR = Path(__file__).resolve().parent.parent

REPEATED_DIR = BASE_DIR / "results" / "repeated_runs"

all_results = []


# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def load_edges(filepath):
    """Load protein interaction pairs as strings."""

    df = pd.read_csv(filepath)

    df["protein1"] = df["protein1"].astype(str)
    df["protein2"] = df["protein2"].astype(str)

    return df


def create_edge_features(edge_df, embeddings):
    """
    Create Hadamard-product features for protein pairs.
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

        features.append(vector_a * vector_b)

    return np.asarray(features)


def calculate_metrics(y_true, y_prob, threshold=0.5):
    """Calculate classification performance metrics."""

    y_pred = (y_prob >= threshold).astype(int)

    return {
        "ROC_AUC": roc_auc_score(y_true, y_prob),
        "Average_Precision": average_precision_score(
            y_true, y_prob
        ),
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1_Score": f1_score(y_true, y_pred)
    }


def generate_random_training_negatives(
    train_positive,
    test_negative,
    seed
):
    """
    Generate random training negatives.

    Excludes all known positive interactions and
    the random test-negative pairs.
    """

    rng = random.Random(seed)

    all_positive_df = pd.read_csv(
        SEED_DIR / "test_positive_edges.csv"
    )

    all_positive_df = pd.concat(
        [train_positive, all_positive_df],
        ignore_index=True
    )

    all_positive = {
        tuple(sorted((str(row.protein1), str(row.protein2))))
        for row in all_positive_df.itertuples(index=False)
    }

    test_negative_set = {
        tuple(sorted((str(row.protein1), str(row.protein2))))
        for row in test_negative.itertuples(index=False)
    }

    nodes = sorted(
        set(train_positive["protein1"])
        | set(train_positive["protein2"])
    )

    negatives = set()
    target_count = len(train_positive)

    while len(negatives) < target_count:

        u, v = rng.sample(nodes, 2)

        pair = tuple(sorted((u, v)))

        if pair in all_positive:
            continue

        if pair in test_negative_set:
            continue

        negatives.add(pair)

    negative_df = pd.DataFrame(
        list(negatives),
        columns=["protein1", "protein2"]
    )

    return negative_df


def evaluate_node2vec(
    seed,
    experiment,
    strategy,
    train_positive,
    train_negative,
    test_positive,
    test_negative,
    embeddings
):
    """Train and evaluate one node2vec configuration."""

    X_train_edges = pd.concat(
        [train_positive, train_negative],
        ignore_index=True
    )

    y_train = np.concatenate([
        np.ones(len(train_positive)),
        np.zeros(len(train_negative))
    ])

    X_test_edges = pd.concat(
        [test_positive, test_negative],
        ignore_index=True
    )

    y_test = np.concatenate([
        np.ones(len(test_positive)),
        np.zeros(len(test_negative))
    ])

    X_train = create_edge_features(
        X_train_edges,
        embeddings
    )

    X_test = create_edge_features(
        X_test_edges,
        embeddings
    )

    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            random_state=seed
        )
    )

    classifier.fit(X_train, y_train)

    y_prob = classifier.predict_proba(X_test)[:, 1]

    y_pred = (y_prob >= 0.5).astype(int)

    metrics = calculate_metrics(y_test, y_prob)

    # Save predictions for this experiment.
    predictions = X_test_edges.copy()

    predictions["actual_label"] = y_test
    predictions["predicted_probability"] = y_prob
    predictions["predicted_label"] = y_pred

    predictions.to_csv(
        SEED_DIR / (
            f"predictions_{strategy}_{experiment}.csv"
        ),
        index=False
    )

    return metrics


def evaluate_common_neighbors(
    G_train,
    test_positive,
    test_negative
):
    """
    Common Neighbors baseline.

    Score = number of shared neighbors between
    the two proteins in the training network.
    """

    test_edges = pd.concat(
        [test_positive, test_negative],
        ignore_index=True
    )

    y_test = np.concatenate([
        np.ones(len(test_positive)),
        np.zeros(len(test_negative))
    ])

    scores = []

    for row in test_edges.itertuples(index=False):

        score = len(
            list(
                nx.common_neighbors(
                    G_train,
                    str(row.protein1),
                    str(row.protein2)
                )
            )
        )

        scores.append(score)

    scores = np.asarray(scores)

    # Common Neighbors uses a positive score
    # as the classification threshold.
    y_pred = (scores > 0).astype(int)

    metrics = {
        "ROC_AUC": roc_auc_score(y_test, scores),
        "Average_Precision": average_precision_score(
            y_test, scores
        ),
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1_Score": f1_score(y_test, y_pred)
    }

    predictions = test_edges.copy()

    predictions["actual_label"] = y_test
    predictions["common_neighbors_score"] = scores
    predictions["predicted_label"] = y_pred

    return metrics, predictions


# ==========================================
# 3. LOOP THROUGH THE THREE SEEDS
# ==========================================

for seed in SEEDS:

    print("\n" + "=" * 55)
    print("PROCESSING SEED:", seed)
    print("=" * 55)

    SEED_DIR = REPEATED_DIR / f"seed_{seed}"

    if not SEED_DIR.exists():
        raise FileNotFoundError(
            f"Seed directory not found: {SEED_DIR}"
        )

    # Load graph and edge datasets.
    G_train = nx.read_graphml(
        SEED_DIR / "training_network.graphml"
    )

    train_positive = load_edges(
        SEED_DIR / "train_edges.csv"
    )

    test_positive = load_edges(
        SEED_DIR / "test_positive_edges.csv"
    )

    test_negative_random = load_edges(
        SEED_DIR / "test_negative_edges.csv"
    )

    test_negative_matched = load_edges(
        SEED_DIR / "test_negative_degree_matched.csv"
    )

    # Generate random training negatives.
    train_negative_random = generate_random_training_negatives(
        train_positive,
        test_negative_random,
        seed
    )

    train_negative_random.to_csv(
        SEED_DIR / "train_negative_edges.csv",
        index=False
    )

    # Load degree-matched training negatives.
    train_negative_matched = load_edges(
        SEED_DIR / "train_negative_degree_matched.csv"
    )

    print("Training positives:", len(train_positive))
    print("Training random negatives:", len(train_negative_random))
    print("Training matched negatives:", len(train_negative_matched))
    print("Test positives:", len(test_positive))
    print("Test random negatives:", len(test_negative_random))
    print("Test matched negatives:", len(test_negative_matched))

    # Two separate evaluation strategies.
    strategies = {
        "random": (
            train_negative_random,
            test_negative_random
        ),
        "degree_matched": (
            train_negative_matched,
            test_negative_matched
        )
    }

    seed_results = []

    # ==========================================
    # 4. EVALUATE NODE2VEC CONFIGURATIONS
    # ==========================================

    for experiment in EXPERIMENTS:

        embedding_file = (
            SEED_DIR / f"embeddings_{experiment}.csv"
        )

        if not embedding_file.exists():
            raise FileNotFoundError(
                f"Embedding file not found: {embedding_file}"
            )

        embeddings = pd.read_csv(
            embedding_file,
            index_col=0
        )

        embeddings.index = embeddings.index.astype(str)

        print("\nLoaded:", experiment, embeddings.shape)

        for strategy, datasets in strategies.items():

            train_negative, test_negative = datasets

            print(
                "\nEvaluating:",
                experiment,
                "| Strategy:",
                strategy
            )

            metrics = evaluate_node2vec(
                seed,
                experiment,
                strategy,
                train_positive,
                train_negative,
                test_positive,
                test_negative,
                embeddings
            )

            result = {
                "Seed": seed,
                "Sampling_Strategy": strategy,
                "Experiment": experiment,
                **metrics
            }

            seed_results.append(result)
            all_results.append(result)

            print(
                "ROC-AUC:",
                round(metrics["ROC_AUC"], 4),
                "| AP:",
                round(metrics["Average_Precision"], 4)
            )

    # ==========================================
    # 5. COMMON NEIGHBORS BASELINE
    # ==========================================

    for strategy, datasets in strategies.items():

        train_negative, test_negative = datasets

        metrics, predictions = evaluate_common_neighbors(
            G_train,
            test_positive,
            test_negative
        )

        predictions.to_csv(
            SEED_DIR / f"predictions_{strategy}_common_neighbors.csv",
            index=False
        )

        result = {
            "Seed": seed,
            "Sampling_Strategy": strategy,
            "Experiment": "common_neighbors",
            **metrics
        }

        seed_results.append(result)
        all_results.append(result)

        print(
            "\nCommon Neighbors |",
            strategy,
            "| ROC-AUC:",
            round(metrics["ROC_AUC"], 4),
            "| AP:",
            round(metrics["Average_Precision"], 4)
        )

    # Save results for this seed.
    seed_results_df = pd.DataFrame(seed_results)

    seed_results_df.to_csv(
        SEED_DIR / "repeated_run_comparison.csv",
        index=False
    )

    print("\nSaved results for seed:", seed)


# ==========================================
# 6. SAVE ALL INDIVIDUAL RUN RESULTS
# ==========================================

all_results_df = pd.DataFrame(all_results)

all_results_df.to_csv(
    REPEATED_DIR / "all_repeated_run_results.csv",
    index=False
)


# ==========================================
# 7. CALCULATE MEAN AND STANDARD DEVIATION
# ==========================================

metrics_to_summarize = [
    "ROC_AUC",
    "Average_Precision",
    "Accuracy",
    "F1_Score"
]

summary = (
    all_results_df
    .groupby(["Sampling_Strategy", "Experiment"])[
        metrics_to_summarize
    ]
    .agg(["mean", "std"])
)

summary.columns = [
    f"{metric}_{statistic}"
    for metric, statistic in summary.columns
]

summary = summary.reset_index()

summary.to_csv(
    REPEATED_DIR / "repeated_run_mean_sd.csv",
    index=False
)


# ==========================================
# 8. DISPLAY FINAL RESULTS
# ==========================================

print("\n" + "=" * 65)
print("ALL REPEATED EXPERIMENTS COMPLETED")
print("=" * 65)

print("\nIndividual run results:")

print(
    all_results_df.round(4).to_string(index=False)
)

print("\nMEAN AND STANDARD DEVIATION ACROSS THREE SEEDS:")

print(
    summary.round(4).to_string(index=False)
)

print("\nSaved files:")
print(REPEATED_DIR / "all_repeated_run_results.csv")
print(REPEATED_DIR / "repeated_run_mean_sd.csv")
