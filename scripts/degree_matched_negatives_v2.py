
import random
import math
import sys
from pathlib import Path

import pandas as pd
import networkx as nx


# ==========================================
# 1. CONFIGURATION
# ==========================================

RANDOM_SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42

rng = random.Random(RANDOM_SEED)

BASE_DIR = Path(__file__).resolve().parent.parent

# Separate folder for each seed.
RESULTS_DIR = (
    BASE_DIR / "results" / "repeated_runs" / f"seed_{RANDOM_SEED}"
)

BIN_TOLERANCE = 1

print("Random seed:", RANDOM_SEED)
print("Results directory:", RESULTS_DIR)


# ==========================================
# 2. LOAD NETWORK AND EDGE DATA
# ==========================================

print("\nLoading training network and edge datasets...")

G = nx.read_graphml(
    RESULTS_DIR / "training_network.graphml"
)

train_positive = pd.read_csv(
    RESULTS_DIR / "train_edges.csv"
)

test_positive = pd.read_csv(
    RESULTS_DIR / "test_positive_edges.csv"
)

# All known positive interactions, including held-out edges.
all_positive_df = pd.concat(
    [train_positive, test_positive],
    ignore_index=True
)

all_positive = {
    tuple(sorted((str(row.protein1), str(row.protein2))))
    for row in all_positive_df.itertuples(index=False)
}

# Exclude original random test negatives as well.
old_test_negative = pd.read_csv(
    RESULTS_DIR / "test_negative_edges.csv"
)

excluded_negative = {
    tuple(sorted((str(row.protein1), str(row.protein2))))
    for row in old_test_negative.itertuples(index=False)
}

nodes = list(G.nodes())


# ==========================================
# 3. CALCULATE DEGREE BINS
# ==========================================

degrees = dict(G.degree())

degree_bins = {
    node: int(math.floor(math.log2(degrees[node])))
    for node in nodes
}

bin_groups = {}

for node in nodes:
    b = degree_bins[node]
    bin_groups.setdefault(b, []).append(node)

print("Training network proteins:", len(nodes))
print("Degree bins:", len(bin_groups))


# ==========================================
# 4. DEGREE-MATCHED NEGATIVE SAMPLING
# ==========================================

# Track all generated and excluded negatives.
# This prevents duplicates across training and test sets.
used_negatives = set(excluded_negative)


def sample_matched_negative(protein_a, protein_b):

    """
    Sample a non-interacting pair with similar
    logarithmic degree bins to a positive pair.

    Each endpoint can differ by at most one bin.
    """

    target_bins = sorted([
        degree_bins[str(protein_a)],
        degree_bins[str(protein_b)]
    ])

    candidate_bins_1 = [
        b for b in bin_groups
        if abs(b - target_bins[0]) <= BIN_TOLERANCE
    ]

    candidate_bins_2 = [
        b for b in bin_groups
        if abs(b - target_bins[1]) <= BIN_TOLERANCE
    ]

    pool_1 = [
        node
        for b in candidate_bins_1
        for node in bin_groups[b]
    ]

    pool_2 = [
        node
        for b in candidate_bins_2
        for node in bin_groups[b]
    ]

    for attempt in range(10000):

        u = rng.choice(pool_1)
        v = rng.choice(pool_2)

        if u == v:
            continue

        pair = tuple(sorted((u, v)))

        candidate_bins = sorted([
            degree_bins[u],
            degree_bins[v]
        ])

        # Check degree-bin similarity.
        if any(
            abs(candidate_bins[i] - target_bins[i])
            > BIN_TOLERANCE
            for i in range(2)
        ):
            continue

        # Exclude known positive interactions.
        if pair in all_positive:
            continue

        # Exclude previously generated negatives.
        if pair in used_negatives:
            continue

        used_negatives.add(pair)

        return pair

    raise RuntimeError(
        f"Could not find a matched negative for "
        f"{protein_a}, {protein_b}. "
        "Try increasing BIN_TOLERANCE."
    )


# ==========================================
# 5. GENERATE TEST NEGATIVES
# ==========================================

print("\nGenerating degree-matched TEST negatives...")

test_negative_matched = []

for row in test_positive.itertuples(index=False):

    pair = sample_matched_negative(
        str(row.protein1),
        str(row.protein2)
    )

    test_negative_matched.append(pair)

    if len(test_negative_matched) % 5000 == 0:
        print(
            "Test negatives generated:",
            len(test_negative_matched)
        )


# ==========================================
# 6. GENERATE TRAINING NEGATIVES
# ==========================================

print("\nGenerating degree-matched TRAINING negatives...")

train_negative_matched = []

for row in train_positive.itertuples(index=False):

    pair = sample_matched_negative(
        str(row.protein1),
        str(row.protein2)
    )

    train_negative_matched.append(pair)

    if len(train_negative_matched) % 10000 == 0:
        print(
            "Training negatives generated:",
            len(train_negative_matched)
        )


# ==========================================
# 7. SAVE NEGATIVE DATASETS
# ==========================================

test_negative_df = pd.DataFrame(
    test_negative_matched,
    columns=["protein1", "protein2"]
)

train_negative_df = pd.DataFrame(
    train_negative_matched,
    columns=["protein1", "protein2"]
)

test_negative_df.to_csv(
    RESULTS_DIR / "test_negative_degree_matched.csv",
    index=False
)

train_negative_df.to_csv(
    RESULTS_DIR / "train_negative_degree_matched.csv",
    index=False
)


# ==========================================
# 8. VALIDATION
# ==========================================

# Convert generated datasets to sets for validation.
train_negative_set = set(train_negative_matched)
test_negative_set = set(test_negative_matched)

known_positive_overlap = sum(
    pair in all_positive
    for pair in train_negative_set | test_negative_set
)

# Check that training and test negatives are disjoint.
train_test_overlap = len(
    train_negative_set & test_negative_set
)

print("\n========== DEGREE-MATCHED DATA ==========")

print("Training positive pairs:", len(train_positive))
print("Training matched negatives:", len(train_negative_df))

print("Test positive pairs:", len(test_positive))
print("Test matched negatives:", len(test_negative_df))

print(
    "Training negative duplicates:",
    train_negative_df.duplicated().sum()
)

print(
    "Test negative duplicates:",
    test_negative_df.duplicated().sum()
)

print(
    "Training-test negative overlap:",
    train_test_overlap
)

print(
    "Negative pairs overlapping known positives:",
    known_positive_overlap
)

print("\nSaved degree-matched negative datasets successfully.")
