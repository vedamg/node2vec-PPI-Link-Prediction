
import pandas as pd
import networkx as nx
import random
from pathlib import Path

# ==========================================
# 1. CONFIGURATION
# ==========================================

RANDOM_SEED = 42
TEST_FRACTION = 0.20

random.seed(RANDOM_SEED)

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "results" / "cleaned_interactions.csv"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)


# ==========================================
# 2. LOAD CLEANED PPI DATA
# ==========================================

df = pd.read_csv(INPUT_FILE)

G = nx.from_pandas_edgelist(
    df,
    source="protein1",
    target="protein2",
    edge_attr="combined_score"
)

print("Original network:")
print("Proteins:", G.number_of_nodes())
print("Interactions:", G.number_of_edges())


# ==========================================
# 3. EXTRACT LARGEST CONNECTED COMPONENT
# ==========================================

largest_component = max(
    nx.connected_components(G),
    key=len
)

G_lcc = G.subgraph(largest_component).copy()

print("\nLargest connected component:")
print("Proteins:", G_lcc.number_of_nodes())
print("Interactions:", G_lcc.number_of_edges())


# ==========================================
# 4. SELECT POSITIVE EDGES FOR TESTING
# ==========================================

# Store all original positive interactions.
# Each pair is sorted to avoid duplicate representations.

all_positive_edges = {
    tuple(sorted((u, v)))
    for u, v in G_lcc.edges()
}

target_test_edges = int(
    TEST_FRACTION * G_lcc.number_of_edges()
)

# Shuffle edges reproducibly.
candidate_edges = list(G_lcc.edges())
random.shuffle(candidate_edges)

# Create the training graph.
G_train = G_lcc.copy()

test_positive_edges = []

# Remove an edge only if both endpoints retain
# at least one training connection.

for u, v in candidate_edges:

    if len(test_positive_edges) >= target_test_edges:
        break

    if G_train.has_edge(u, v):

        if G_train.degree(u) > 1 and G_train.degree(v) > 1:

            G_train.remove_edge(u, v)
            test_positive_edges.append(tuple(sorted((u, v))))


# ==========================================
# 5. GENERATE NEGATIVE TEST EXAMPLES
# ==========================================

# Negative examples are protein pairs that have
# no recorded interaction in the original network.
# They are NOT confirmed biological non-interactions.

nodes = list(G_lcc.nodes())

negative_edges = set()

number_of_negatives = len(test_positive_edges)

while len(negative_edges) < number_of_negatives:

    u, v = random.sample(nodes, 2)

    pair = tuple(sorted((u, v)))

    if pair not in all_positive_edges:
        negative_edges.add(pair)


# ==========================================
# 6. SAVE TRAINING AND TEST DATA
# ==========================================

train_df = pd.DataFrame(
    [
        (u, v)
        for u, v in G_train.edges()
    ],
    columns=["protein1", "protein2"]
)

positive_df = pd.DataFrame(
    test_positive_edges,
    columns=["protein1", "protein2"]
)

negative_df = pd.DataFrame(
    list(negative_edges),
    columns=["protein1", "protein2"]
)

train_df.to_csv(
    RESULTS_DIR / "train_edges.csv",
    index=False
)

positive_df.to_csv(
    RESULTS_DIR / "test_positive_edges.csv",
    index=False
)

negative_df.to_csv(
    RESULTS_DIR / "test_negative_edges.csv",
    index=False
)

nx.write_graphml(
    G_train,
    RESULTS_DIR / "training_network.graphml"
)


# ==========================================
# 7. CHECK THE SPLIT
# ==========================================

isolated_nodes = list(nx.isolates(G_train))

training_components = nx.number_connected_components(G_train)

# Confirm no held-out positive edge remains in training.
leakage = sum(
    G_train.has_edge(u, v)
    for u, v in test_positive_edges
)

# Check that every test endpoint is present in training.
missing_test_nodes = {
    node
    for edge in test_positive_edges
    for node in edge
    if node not in G_train
}

print("\n========== FINAL DATA SPLIT ==========")

print("Training interactions:", len(train_df))
print("Positive test interactions:", len(positive_df))
print("Negative test interactions:", len(negative_df))

print(
    "Actual test fraction:",
    round(
        len(positive_df) / G_lcc.number_of_edges() * 100,
        2
    ),
    "%"
)

print("Isolated training proteins:", len(isolated_nodes))
print("Training connected components:", training_components)
print("Test-edge leakage:", leakage)
print("Missing test endpoint proteins:", len(missing_test_nodes))

print("\nFiles saved successfully in the results folder.")
