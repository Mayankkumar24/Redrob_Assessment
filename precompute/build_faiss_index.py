"""
Step 1.6 — Vector Index Construction

Builds a FAISS index over the candidate embeddings so that at runtime,
finding the top-K semantically similar candidates to the JD query takes
milliseconds instead of a brute-force 100K-way loop.

Index choice: IndexFlatIP (exact inner-product search). At 100K candidates
x 384 dims, an exact flat index is still fast enough on CPU (sub-second
search) - no need for the complexity/recall tradeoff of an approximate
index like IVF or HNSW. Since embeddings are L2-normalized (done in Step
1.5), inner product IS cosine similarity, so IndexFlatIP gives exact cosine
similarity search.

This step needs NO network access and NO embedding model - it only
operates on the .npy arrays already saved by Step 1.5. Fully testable
offline, including in network-restricted environments.
"""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np


def build_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build an exact inner-product FAISS index. Embeddings must already be
    L2-normalized (Step 1.5 does this) for inner product to equal cosine
    similarity.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def save_index(index: faiss.Index, path: str = "data/processed/candidate_index.faiss"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))
    print(f"[build_faiss_index] Saved index ({index.ntotal} vectors, "
          f"dim={index.d}) to {path}")


def load_index(path: str = "data/processed/candidate_index.faiss") -> faiss.Index:
    return faiss.read_index(str(path))


def search(
    index: faiss.Index, query_vector: np.ndarray, k: int = 5000
) -> tuple[np.ndarray, np.ndarray]:
    """
    query_vector: shape (384,) or (1, 384), L2-normalized.
    Returns (scores, indices) each of shape (k,) - scores are cosine
    similarities in [-1, 1] (in practice [0, 1] for normalized text
    embeddings of similar content), indices are positions into the
    embeddings array used to build the index (map back via candidate_ids.npy).
    """
    if query_vector.ndim == 1:
        query_vector = query_vector.reshape(1, -1)
    k = min(k, index.ntotal)
    scores, indices = index.search(query_vector.astype(np.float32), k)
    return scores[0], indices[0]


def run(embeddings_path: str = "data/processed/candidate_embeddings.npy",output_path: str = "data/processed/candidate_index.faiss"):
    embeddings = np.load(embeddings_path)
    index = build_index(embeddings)
    save_index(index, output_path)
    return index;
#     output_path: str = "data/processed/candidate_index.faiss",
# ):
#     embeddings = np.load(embeddings_path)
#     index = build_index(embeddings)
#     save_index(index, output_path)
#     return index


if __name__ == "__main__":
    # Self-contained test using mock embeddings, so this is fully verifiable
    # without needing Step 1.5's real model output first.
    print("=== Testing FAISS index build + search with mock embeddings ===\n")

    rng = np.random.default_rng(7)
    n, dim = 50, 384
    mock_embeddings = rng.normal(size=(n, dim)).astype(np.float32)
    mock_embeddings /= np.linalg.norm(mock_embeddings, axis=1, keepdims=True)
    candidate_ids = [f"CAND_{i:07d}" for i in range(1, n + 1)]

    index = build_index(mock_embeddings)
    print(f"Index built: {index.ntotal} vectors, dim={index.d}")

    save_index(index, "data/processed/_test_index.faiss")
    loaded = load_index("data/processed/_test_index.faiss")
    print(f"Re-loaded index: {loaded.ntotal} vectors")

    # query with the 5th candidate's own embedding -> should retrieve itself
    # as the #1 (highest similarity) result, score should be ~1.0
    query = mock_embeddings[4]
    scores, indices = search(loaded, query, k=5)
    print(f"\nSelf-similarity sanity check (querying with candidate #5's own vector):")
    print(f"  Top result index: {indices[0]} (expected 4), score: {scores[0]:.4f} (expected ~1.0)")
    print(f"  Top-5 candidate_ids: {[candidate_ids[i] for i in indices]}")
    print(f"  Top-5 scores: {[round(s, 4) for s in scores]}")

    assert indices[0] == 4, "Self-similarity check failed - index not returning itself as top match"
    assert scores[0] > 0.99, "Self-similarity score should be ~1.0"
    print("\nSelf-similarity check: PASSED")

    Path("data/processed/_test_index.faiss").unlink()


"""
Step 1.6 — Vector Index Construction

Builds a FAISS index over the candidate embeddings so that at runtime,
finding the top-K semantically similar candidates to the JD query takes
milliseconds instead of a brute-force 100K-way loop.
"""

# from __future__ import annotations

# from pathlib import Path

# import faiss
# import numpy as np


# def build_index(embeddings: np.ndarray) -> faiss.Index:
#     """
#     Build an exact inner-product FAISS index. Embeddings must already be
#     L2-normalized (Step 1.5 does this) for inner product to equal cosine
#     similarity.
#     """
#     dim = embeddings.shape[1]
#     index = faiss.IndexFlatIP(dim)
#     index.add(embeddings)
#     return index


# # DEFAULT PATH UPDATED TO GOOGLE DRIVE
# def save_index(index: faiss.Index, path: str = "candidate_index.faiss"):
#     Path(path).parent.mkdir(parents=True, exist_ok=True)
#     faiss.write_index(index, str(path))
#     print(f"[build_faiss_index] Saved index ({index.ntotal} vectors, "
#           f"dim={index.d}) to {path}")


# # DEFAULT PATH UPDATED TO GOOGLE DRIVE
# def load_index(path: str = "candidate_index.faiss") -> faiss.Index:
#     return faiss.read_index(str(path))


# def search(
#     index: faiss.Index, query_vector: np.ndarray, k: int = 5000
# ) -> tuple[np.ndarray, np.ndarray]:
#     """
#     query_vector: shape (384,) or (1, 384), L2-normalized.
#     Returns (scores, indices) each of shape (k,) - scores are cosine
#     similarities.
#     """
#     if query_vector.ndim == 1:
#         query_vector = query_vector.reshape(1, -1)
#     k = min(k, index.ntotal)
#     scores, indices = index.search(query_vector.astype(np.float32), k)
#     return scores[0], indices[0]


# # DEFAULT PATHS UPDATED TO GOOGLE DRIVE
# def run(
#     embeddings_path: str = "C:\\Users\\Neeraj\\Downloads\\redrob-ranker-step1\\redrob-ranker\\data\\processed\\candidate_embeddings.npy",
#     output_path: str = "C:\\Users\\Neeraj\\Downloads\\redrob-ranker-step1\\redrob-ranker\\data\\processed\\candidate_index.faiss",
# ):
#     print(f"Loading embeddings directly from: {embeddings_path}...")
#     embeddings = np.load(embeddings_path)
    
#     print("Building FAISS index...")
#     index = build_index(embeddings)
    
#     save_index(index, output_path)
#     return index

# if __name__ == "__main__":
#     print("Starting ACTUAL FAISS index build for local files...")
    
#     # Yeh function aapke asli data ko load karke index banayega aur save karega
#     run() 
    
#     print("Process Completed Successfully!")


# import json;

# path = "C:\\Users\\Neeraj\\Downloads\\redrob-ranker-step1\\redrob-ranker\\data\\processed\\production_experience_scores.json"
# max_score = -11
# candidates_with_max_score = []

# with open(path,'r') as f:
#     data = json.load(f)

# sorted_items = sorted(data.items(),key = lambda x : x[1],reverse = True)
# top_10 = sorted_items[:10]

# for key,val in top_10:
#     print(f"Candidate ID : {key}, Score : {val}")

