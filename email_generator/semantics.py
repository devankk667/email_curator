"""
semantic_search.py — Phase 4, step 2.

Implements the "AI search" feature from the original roadmap: search by
meaning, not keywords. Encodes a natural-language query with the same
model used to embed the corpus, then ranks emails by cosine similarity.

Since embeddings are unit-normalized (normalize_embeddings=True in
embeddings.py), cosine similarity is just a dot product -- no need for
sklearn's cosine_similarity, a plain matrix multiply is enough and is
faster at this scale.

Usage:
    python semantic_search.py "flight booking confirmation"
    python semantic_search.py "job interview invitation" --top_k 10
"""

import sys
import argparse

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from loader import load_emails
from cleaner import clean_dataframe
from embeddings import load_embeddings, MODEL_NAME


class SemanticSearchIndex:
    def __init__(self, embeddings: np.ndarray, meta_df: pd.DataFrame, model: SentenceTransformer = None):
        """
        embeddings: (n, dim) array aligned row-for-row with meta_df.
        meta_df: DataFrame with at least 'filename', 'subject', 'category',
                 'search_text' columns, in the SAME row order as embeddings.
        """
        assert len(embeddings) == len(meta_df), (
            f"Embeddings ({len(embeddings)}) and metadata ({len(meta_df)}) row count mismatch -- "
            f"they must be aligned. Did you filter/sort meta_df after generating embeddings?"
        )
        self.embeddings = embeddings
        self.meta_df = meta_df.reset_index(drop=True)
        self.model = model or SentenceTransformer(MODEL_NAME)

    def search(self, query: str, top_k: int = 5, category_filter: str = None):
        query_vec = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self.embeddings @ query_vec  # cosine similarity via dot product (unit-norm vectors)

        candidate_idx = np.arange(len(scores))
        if category_filter:
            mask = (self.meta_df["category"] == category_filter).values
            candidate_idx = candidate_idx[mask]

        top_idx = candidate_idx[np.argsort(scores[candidate_idx])[::-1][:top_k]]

        results = self.meta_df.iloc[top_idx][["filename", "category", "subject", "search_text"]].copy()
        results["score"] = scores[top_idx]
        return results.reset_index(drop=True)


def build_index_from_disk(emails_dir: str, labels_csv: str):
    """Convenience loader: rebuilds the metadata DataFrame and aligns it
    with the saved embeddings by filename (defensive against any row-order
    drift between when embeddings were generated and now)."""
    df = load_emails(emails_dir, labels_csv)
    df = clean_dataframe(df)

    embeddings, ids = load_embeddings()
    id_order = pd.DataFrame({"filename": ids})
    aligned_df = id_order.merge(df, on="filename", how="left")

    missing = aligned_df["category"].isna().sum()
    if missing:
        print(f"WARNING: {missing} embedded filenames not found in current metadata -- "
              f"embeddings may be stale, consider regenerating.")

    return SemanticSearchIndex(embeddings, aligned_df)


def print_results(query, results):
    print(f"\nQuery: {query!r}")
    print("-" * 70)
    for _, row in results.iterrows():
        snippet = row["search_text"][:90].replace("\n", " ")
        print(f"[{row['score']:.3f}] ({row['category']:10s}) {row['subject']}")
        print(f"          {snippet}...")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", default=None)
    parser.add_argument("--emails_dir", default="generated_emails")
    parser.add_argument("--labels_csv", default="labels.csv")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--category", default=None)
    args = parser.parse_args()

    index = build_index_from_disk(args.emails_dir, args.labels_csv)

    if args.query:
        results = index.search(args.query, top_k=args.top_k, category_filter=args.category)
        print_results(args.query, results)
    else:
        # No query given -- run a demo set covering different categories,
        # so you can eyeball whether search quality is sane end-to-end.
        demo_queries = [
            "flight booking confirmation",
            "job interview invitation",
            "credit card payment due",
            "assignment deadline reminder",
            "you have won a prize",
            "someone tagged me in a photo",
        ]
        for q in demo_queries:
            results = index.search(q, top_k=3)
            print_results(q, results)