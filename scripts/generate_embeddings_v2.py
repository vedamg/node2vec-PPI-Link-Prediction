
import sys
import random
from pathlib import Path

import numpy as np
import networkx as nx
import pandas as pd
from node2vec import Node2Vec


# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

RANDOM_SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42

# Each seed gets its own folder.
RESULTS_DIR = (
    BASE_DIR / "results" / "repeated_runs" / f"seed_{RANDOM_SEED}"
)

INPUT_FILE = RESULTS_DIR / "training_network.graphml"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Set random seeds.
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ==========================================
# 2. NODE2VEC EXPERIMENTS
# ==========================================

experiments = {
    "baseline": {"p": 1, "q": 1},
    "local": {"p": 1, "q": 2},
    "outward": {"p": 1, "q": 0.5}
}


# ==========================================
# 3. LOAD TRAINING NETWORK
# ==========================================

print("Loading training network...")
print("Random seed:", RANDOM_SEED)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Training network not found: {INPUT_FILE}\n"
        "Run the updated train/test preparation script first."
    )

G = nx.read_graphml(INPUT_FILE)

# Ensure edge weights are numeric.
for u, v, data in G.edges(data=True):
    data["combined_score"] = float(data["combined_score"])

print("\nTraining network loaded!")
print("Proteins:", G.number_of_nodes())
print("Interactions:", G.number_of_edges())

components = sorted(
    nx.connected_components(G),
    key=len,
    reverse=True
)

print("Connected components:", len(components))
print("Largest training component:", len(components[0]))


# ==========================================
# 4. NODE2VEC PARAMETERS
# ==========================================

DIMENSIONS = 128
WALK_LENGTH = 80
NUM_WALKS = 10
WINDOW_SIZE = 10


# ==========================================
# 5. GENERATE EMBEDDINGS
# ==========================================

for experiment_name, params in experiments.items():

    p = params["p"]
    q = params["q"]

    print("\n===================================")
    print("Experiment:", experiment_name)
    print("p =", p, "| q =", q)
    print("Seed =", RANDOM_SEED)
    print("===================================")

    node2vec_model = Node2Vec(
        G,
        dimensions=DIMENSIONS,
        walk_length=WALK_LENGTH,
        num_walks=NUM_WALKS,
        p=p,
        q=q,
        weight_key="combined_score",
        workers=1,
        seed=RANDOM_SEED,
        quiet=False
    )

    model = node2vec_model.fit(
        window=WINDOW_SIZE,
        min_count=1,
        batch_words=128,
        workers=1,
        seed=RANDOM_SEED
    )

    embeddings = model.wv

    embedding_df = pd.DataFrame.from_dict(
        {
            protein: embeddings[protein]
            for protein in embeddings.index_to_key
        },
        orient="index"
    )

    embedding_df.index.name = "protein"

    output_file = (
        RESULTS_DIR /
        f"embeddings_{experiment_name}.csv"
    )

    embedding_df.to_csv(output_file)

    print("\nEmbedding generation completed!")
    print("Number of protein embeddings:", len(embedding_df))
    print("Embedding dimensions:", embedding_df.shape[1])
    print("Saved:", output_file)


print("\nAll node2vec experiments completed!")
print("Results directory:", RESULTS_DIR)
