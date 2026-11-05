import networkx as nx
import pandas as pd
from node2vec import Node2Vec
from pathlib import Path

# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "results" / "training_network.graphml"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)

RANDOM_SEED = 42

# Three node2vec experiments
experiments = {
    "baseline": {"p": 1, "q": 1},
    "local": {"p": 1, "q": 2},
    "outward": {"p": 1, "q": 0.5}
}

# ==========================================
# 2. LOAD TRAINING NETWORK
# ==========================================

print("Loading training network...")

G = nx.read_graphml(INPUT_FILE)

# Ensure edge weights are numeric
for u, v, data in G.edges(data=True):
    data["combined_score"] = float(data["combined_score"])

print("\nTraining network loaded!")
print("Proteins:", G.number_of_nodes())
print("Interactions:", G.number_of_edges())

# Examine connected components
components = sorted(
    nx.connected_components(G),
    key=len,
    reverse=True
)

print("Connected components:", len(components))
print("Largest training component:", len(components[0]))

# ==========================================
# 3. NODE2VEC PARAMETERS
# ==========================================

DIMENSIONS = 128
WALK_LENGTH = 80
NUM_WALKS = 10
WINDOW_SIZE = 10

# ==========================================
# 4. RUN NODE2VEC EXPERIMENTS
# ==========================================

for experiment_name, params in experiments.items():

    p = params["p"]
    q = params["q"]

    print("\n===================================")
    print("Experiment:", experiment_name)
    print("p =", p, "| q =", q)
    print("===================================")

    # Generate biased random walks
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

    # Train Skip-gram model
    model = node2vec_model.fit(
        window=WINDOW_SIZE,
        min_count=1,
        batch_words=128,
        workers=1,
        seed=RANDOM_SEED
    )

    # Extract protein embeddings
    embeddings = model.wv

    # Convert embeddings into a DataFrame
    embedding_df = pd.DataFrame.from_dict(
        {
            protein: embeddings[protein]
            for protein in embeddings.index_to_key
        },
        orient="index"
    )

    embedding_df.index.name = "protein"

    # Save embeddings
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
