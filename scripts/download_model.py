"""Download BGE-M3 embedding model to the project cache directory.

Run this script once to enable semantic embeddings (higher quality similarity).
When the model is not downloaded, the system falls back to keyword matching.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.rag.embeddings import CACHE_DIR

print(f"Downloading BGE-M3 model to: {CACHE_DIR}")
print("This will download approximately 2.2GB on first run.")
print()

try:
    from sentence_transformers import SentenceTransformer
    import os
    os.environ["HF_HUB_OFFLINE"] = "0"

    model = SentenceTransformer("BAAI/bge-m3", device="cpu", trust_remote_code=True)
    print("Model downloaded successfully!")

    vec = model.encode(["test query"], normalize_embeddings=True)
    print(f"Test embedding dimension: {len(vec[0])}")
    print("Model is ready for use.")
except Exception as e:
    print(f"Error downloading model: {e}")
    print()
    print("The system will continue to work using keyword-based fallback.")
    sys.exit(1)
