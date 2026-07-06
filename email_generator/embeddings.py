"""
embeddings.py — Phase 4, step 1.

Encodes clean_text into fixed-length vectors using a pretrained
Sentence Transformer (all-MiniLM-L6-v2: 384 dims, fast, good default —
no training required, per the strategy discussion: fine-tuning your own
model isn't worth it on synthetic/template-heavy data).

Usage:
    python embeddings.py generated_emails labels.csv
"""

import sys

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from loader import load_emails
from cleaner import clean_dataframe

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDINGS_PATH = "email_embeddings.npy"
IDS_PATH = "email_embedding_ids.csv"


def embed_texts(texts, model=None, batch_size=64, show_progress=True):
    """Encode a list/Series of strings into a (n, dim) numpy array."""
    if model is None:
        model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        list(texts), batch_size=batch_size, show_progress_bar=show_progress,
        normalize_embeddings=True,  # unit-norm vectors -> cosine similarity == dot product
    )
    return embeddings, model


def embed_dataframe(df: pd.DataFrame, text_col: str = "search_text"):
    """Embed every row's text_col, return (embeddings array, model).
    Default is search_text, not clean_text: search/clustering needs the
    quoted content in thread replies that clean_text deliberately strips
    for classifier training. Using clean_text here caused thread replies
    across different categories to collapse into generic near-duplicate
    vectors (e.g. 'Confirmed, thank you.' for Travel, Finance, College
    alike) -- see the semantic_search.py bug this fixes."""
    embeddings, model = embed_texts(df[text_col])
    return embeddings, model


def save_embeddings(embeddings: np.ndarray, filenames: pd.Series,
                     embeddings_path: str = EMBEDDINGS_PATH, ids_path: str = IDS_PATH):
    """Store vectors as .npy (fast, compact) with a parallel CSV of
    filenames so a vector at row i can be traced back to its email."""
    np.save(embeddings_path, embeddings)
    pd.DataFrame({"filename": filenames.values}).to_csv(ids_path, index=False)
    print(f"Saved {embeddings.shape} embeddings to {embeddings_path}")
    print(f"Saved {len(filenames)} filename mappings to {ids_path}")


def load_embeddings(embeddings_path: str = EMBEDDINGS_PATH, ids_path: str = IDS_PATH):
    embeddings = np.load(embeddings_path)
    ids = pd.read_csv(ids_path)["filename"]
    return embeddings, ids


# --- Sanity check: the analogy test from the original Phase 3 plan -----------

def analogy_sanity_check(model):
    """Confirm semantically similar phrases land close together in vector
    space BEFORE trusting these embeddings for clustering/search/anything.
    This is not optional -- an embedding model that fails this test is not
    safe to build on top of."""
    phrases = [
        "Assignment submission",
        "Assignment deadline",
        "Homework due",
        "Project upload",
        "Flight booking confirmed",       # unrelated control phrase
        "Your credit card statement",     # unrelated control phrase
    ]
    embeddings = model.encode(phrases, normalize_embeddings=True)
    sims = embeddings @ embeddings.T

    print("\nANALOGY SANITY CHECK (cosine similarity)")
    print("Phrases:")
    for i, p in enumerate(phrases):
        print(f"  [{i}] {p}")
    print()
    header = "      " + "".join(f"[{i}]   " for i in range(len(phrases)))
    print(header)
    for i, row in enumerate(sims):
        print(f"[{i}] " + " ".join(f"{v:.2f}" for v in row))

    # The four "assignment/deadline" phrases (0-3) should be mutually close;
    # the two control phrases (4-5) should be far from that cluster.
    cluster_sim = sims[np.ix_([0, 1, 2, 3], [0, 1, 2, 3])]
    cross_sim = sims[np.ix_([0, 1, 2, 3], [4, 5])]
    avg_within = cluster_sim[np.triu_indices(4, k=1)].mean()
    avg_across = cross_sim.mean()

    print(f"\nAvg similarity WITHIN assignment/deadline cluster: {avg_within:.3f}")
    print(f"Avg similarity ACROSS to unrelated controls:        {avg_across:.3f}")
    passed = avg_within > avg_across + 0.1  # require a meaningful margin, not just any gap
    print(f"PASS (within > across by meaningful margin): {passed}")
    if not passed:
        print("WARNING: embeddings did not separate related from unrelated phrases "
              "as expected -- do not trust downstream clustering/search until this "
              "is investigated.")
    return passed


if __name__ == "__main__":
    emails_dir = sys.argv[1] if len(sys.argv) > 1 else "generated_emails"
    labels_csv = sys.argv[2] if len(sys.argv) > 2 else "labels.csv"

    df = load_emails(emails_dir, labels_csv)
    df = clean_dataframe(df)

    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    passed = analogy_sanity_check(model)
    if not passed:
        print("\nAborting embedding generation -- fix the sanity check first.")
        sys.exit(1)

    print(f"\nEmbedding {len(df)} emails...")
    embeddings, _ = embed_texts(df["search_text"], model=model)
    save_embeddings(embeddings, df["filename"])

    print(f"\nEmbedding shape: {embeddings.shape}")
    print(f"Sample vector norm (should be ~1.0 due to normalize_embeddings): "
          f"{np.linalg.norm(embeddings[0]):.4f}")