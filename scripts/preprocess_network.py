
"""
Preprocess STRING human physical protein interaction data.

Tasks:
1. Load the downloaded STRING dataset.
2. Validate and clean interaction records.
3. Construct an undirected PPI network.
4. Calculate network-level statistics.
5. Save the cleaned network and statistics.
"""

from pathlib import Path
import gzip
import pandas as pd
import networkx as nx


# --------------------------------------------------
# 1. Define project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = (
    DATA_DIR / "9606.protein.physical.links.v12.5.txt.gz"
)


# --------------------------------------------------
# 2. Load the STRING dataset
# --------------------------------------------------

print("Loading STRING interaction dataset...")

df = pd.read_csv(
    INPUT_FILE,
    sep=" ",
    compression="gzip"
)

print("\nOriginal dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())


# --------------------------------------------------
# 3. Clean the interaction data
# --------------------------------------------------

# Keep only the required columns
df = df[["protein1", "protein2", "combined_score"]].copy()

# Remove records with missing values
df = df.dropna()

# Ensure interaction scores are numeric
df["combined_score"] = pd.to_numeric(
    df["combined_score"],
    errors="coerce"
)
df = df.dropna(subset=["combined_score"])

# Keep interactions with confidence score >= 700
df = df[df["combined_score"] >= 700]

# Remove self-interactions
df = df[df["protein1"] != df["protein2"]]

# Remove duplicate interactions, treating A-B and B-A as identical
df["edge_key"] = df.apply(
    lambda row: tuple(sorted((row["protein1"], row["protein2"]))),
    axis=1
)

df = df.drop_duplicates(subset="edge_key")
df = df.drop(columns="edge_key")

print("\nCleaned dataset shape:", df.shape)


# --------------------------------------------------
# 4. Construct an undirected PPI network
# --------------------------------------------------

print("\nConstructing PPI network...")

G = nx.from_pandas_edgelist(
    df,
    source="protein1",
    target="protein2",
    edge_attr="combined_score",
    create_using=nx.Graph()
)

print("Network construction completed!")


# --------------------------------------------------
# 5. Calculate network statistics
# --------------------------------------------------

num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()

degrees = dict(G.degree())

average_degree = (
    sum(degrees.values()) / num_nodes if num_nodes > 0 else 0
)

density = nx.density(G)

components = list(nx.connected_components(G))
num_components = len(components)

largest_component = (
    max(len(component) for component in components)
    if components else 0
)

average_clustering = nx.average_clustering(G)

statistics = {
    "Number of proteins (nodes)": num_nodes,
    "Number of interactions (edges)": num_edges,
    "Average degree": average_degree,
    "Network density": density,
    "Connected components": num_components,
    "Largest component size": largest_component,
    "Average clustering coefficient": average_clustering
}

print("\n========== NETWORK STATISTICS ==========")

for key, value in statistics.items():
    print(f"{key}: {value}")


# --------------------------------------------------
# 6. Save results
# --------------------------------------------------

# Save cleaned interactions
cleaned_file = RESULTS_DIR / "cleaned_interactions.csv"
df.to_csv(cleaned_file, index=False)

# Save network statistics
stats_file = RESULTS_DIR / "network_statistics.csv"

pd.DataFrame(
    statistics.items(),
    columns=["Metric", "Value"]
).to_csv(stats_file, index=False)

# Save graph for subsequent analysis
graph_file = RESULTS_DIR / "ppi_network.graphml"
nx.write_graphml(G, graph_file)

print("\nSaved files:")
print(cleaned_file)
print(stats_file)
print(graph_file)

print("\nPreprocessing completed successfully!")
