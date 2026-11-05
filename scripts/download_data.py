
"""
Download human physical protein-protein interaction data
from STRING database version 12.5.

Organism: Homo sapiens
Taxonomy ID: 9606
"""

from pathlib import Path
import requests
from tqdm import tqdm

# Locate the main project directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Create data directory if it does not exist
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# STRING dataset URL
URL = (
    "https://stringdb-downloads.org/download/stream/"
    "protein.physical.links.v12.5/"
    "9606.protein.physical.links.v12.5.min700.onlyAB.txt.gz"
)

OUTPUT_FILE = DATA_DIR / "9606.protein.physical.links.v12.5.txt.gz"


def download_file(url, output_path):

    # Avoid downloading the same file repeatedly
    if output_path.exists():
        print("Dataset already exists:", output_path)
        return

    print("Downloading STRING dataset...")

    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    with open(output_path, "wb") as file:
        with tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            desc="Download"
        ) as progress:

            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    progress.update(len(chunk))

    print("Download completed!")
    print("Saved at:", output_path)


if __name__ == "__main__":
    download_file(URL, OUTPUT_FILE)
