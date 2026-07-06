"""
semantic_folders.py — Gmail curator capability #1: Semantic folders.

"Automatically group similar emails without predefined categories."

Clusters emails purely by embedding similarity, then auto-generates
human-readable folder names per cluster. The existing 'category' label
is used ONLY as a validation check afterward -- never as an input to
clustering itself.

Improvements over basic KMeans:
- Optional HDBSCAN for density-based clustering (no need to specify k)
- UMAP visualization for understanding cluster structure
- Better cluster naming using TF-IDF + closest-to-centroid emails
- Outlier detection and handling
- Model persistence for assigning new emails to existing clusters
- Multiple clustering metrics for better evaluation

Usage:
    python semantic_folders.py generated_emails labels.csv --k 12
    python semantic_folders.py generated_emails labels.csv --scan
    python semantic_folders.py generated_emails labels.csv --hdbscan
    python semantic_folders.py generated_emails labels.csv --k 12 --save-model
"""

import sys
import argparse
import pickle

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import (
    silhouette_score, 
    adjusted_rand_score, 
    calinski_harabasz_score,
    davies_bouldin_score
)
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: UMAP not available. Install with: pip install umap-learn")

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    print("Warning: HDBSCAN not available. Install with: pip install hdbscan")

from loader import load_emails
from cleaner import clean_dataframe
from embeddings import load_embeddings


def align_embeddings_to_df(df: pd.DataFrame, embeddings: np.ndarray, ids: pd.Series):
    """Align embeddings to dataframe, filtering out missing entries."""
    id_to_row = {f: i for i, f in enumerate(ids)}
    valid_mask = df["filename"].isin(id_to_row)
    if not valid_mask.any():
        raise ValueError("No matching filenames found between dataframe and embeddings.")
    
    df = df[valid_mask].reset_index(drop=True)
    embeddings = embeddings[[id_to_row[f] for f in df["filename"]]]
    return df, embeddings


def scan_k(embeddings: np.ndarray, k_values=(6, 8, 10, 12, 16, 20, 25, 30)):
    """Try several cluster counts, report multiple metrics for each."""
    print("Scanning cluster counts (multiple metrics):")
    print("  Silhouette: higher = better-separated (-1 to 1)")
    print("  Calinski-Harabasz: higher = better-defined clusters")
    print("  Davies-Bouldin: lower = better separation\n")
    
    scores = {"silhouette": {}, "calinski": {}, "davies": {}}
    
    # Subsample for faster evaluation
    sample_size = min(5000, len(embeddings))
    sample_idx = np.random.RandomState(42).choice(len(embeddings), sample_size, replace=False)
    sample_embeddings = embeddings[sample_idx]
    
    for k in k_values:
        if k >= len(embeddings):
            print(f"  k={k:3d}   SKIPPED (k >= n_samples)")
            continue
            
        # Use MiniBatchKMeans for speed on large datasets
        if len(embeddings) > 10000:
            km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=5, batch_size=1024)
        else:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            
        labels = km.fit_predict(embeddings)
        
        # Compute metrics on subsample
        sil = silhouette_score(sample_embeddings, labels[sample_idx])
        cal = calinski_harabasz_score(sample_embeddings, labels[sample_idx])
        dav = davies_bouldin_score(sample_embeddings, labels[sample_idx])
        
        scores["silhouette"][k] = sil
        scores["calinski"][k] = cal
        scores["davies"][k] = dav
        
        print(f"  k={k:3d}   silhouette={sil:.3f}   calinski={cal:.0f}   davies={dav:.2f}")

    # Plot all metrics
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(list(scores["silhouette"].keys()), list(scores["silhouette"].values()), marker="o")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Silhouette Score")
    axes[0].set_title("Silhouette (higher = better)")
    axes[0].grid(alpha=0.3)
    
    axes[1].plot(list(scores["calinski"].keys()), list(scores["calinski"].values()), marker="o", color="green")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Calinski-Harabasz Score")
    axes[1].set_title("Calinski-Harabasz (higher = better)")
    axes[1].grid(alpha=0.3)
    
    axes[2].plot(list(scores["davies"].keys()), list(scores["davies"].values()), marker="o", color="red")
    axes[2].set_xlabel("k")
    axes[2].set_ylabel("Davies-Bouldin Score")
    axes[2].set_title("Davies-Bouldin (lower = better)")
    axes[2].grid(alpha=0.3)
    
    fig.tight_layout()
    fig.savefig("semantic_folders_k_scan.png", dpi=120)
    plt.close(fig)
    print("\nSaved scan plot to semantic_folders_k_scan.png")

    # Pick best k by silhouette (most interpretable metric)
    best_k = max(scores["silhouette"], key=scores["silhouette"].get)
    print(f"\nBest k by silhouette: {best_k} (score={scores['silhouette'][best_k]:.3f})")
    return best_k


def name_cluster_tfidf(texts: pd.Series, top_n: int = 3) -> str:
    """Generate folder name from most distinctive TF-IDF bigrams."""
    if len(texts) < 2:
        return "misc"
    
    vec = TfidfVectorizer(
        stop_words="english", 
        ngram_range=(2, 2), 
        max_features=100, 
        min_df=2,
        max_df=0.9  # Ignore very common terms
    )
    
    try:
        matrix = vec.fit_transform(texts)
    except ValueError:
        return "misc"
    
    # Get average TF-IDF scores across all documents in cluster
    avg_scores = np.asarray(matrix.mean(axis=0)).flatten()
    feature_names = vec.get_feature_names_out()
    
    # Sort by average TF-IDF score
    top_indices = avg_scores.argsort()[-top_n:][::-1]
    top_terms = [feature_names[i] for i in top_indices if avg_scores[i] > 0]
    
    return " / ".join(top_terms) if top_terms else "misc"


def name_cluster_from_centroid_emails(
    df: pd.DataFrame, 
    embeddings: np.ndarray, 
    centroid: np.ndarray, 
    top_n: int = 3
) -> str:
    """Generate folder name from subjects of emails closest to cluster centroid."""
    # Find emails closest to centroid
    distances = np.linalg.norm(embeddings - centroid, axis=1)
    closest_indices = distances.argsort()[:min(10, len(embeddings))]
    
    # Extract key terms from their subjects
    closest_subjects = df.iloc[closest_indices]["subject"].tolist()
    all_text = " ".join(closest_subjects)
    
    # Use TF-IDF on these representative subjects
    vec = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=50,
        min_df=1
    )
    
    try:
        matrix = vec.fit_transform([all_text])
        scores = np.asarray(matrix).flatten()
        features = vec.get_feature_names_out()
        top_indices = scores.argsort()[-top_n:][::-1]
        top_terms = [features[i] for i in top_indices if scores[i] > 0]
        return " / ".join(top_terms) if top_terms else "misc"
    except:
        return "misc"


def build_folders_kmeans(
    df: pd.DataFrame, 
    embeddings: np.ndarray, 
    k: int,
    use_minibatch: bool = False
) -> pd.DataFrame:
    """Cluster using KMeans and generate folder names."""
    print(f"\nClustering with KMeans (k={k})...")
    
    if use_minibatch or len(embeddings) > 10000:
        km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=5, batch_size=1024)
        print("  Using MiniBatchKMeans for speed")
    else:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
    
    df = df.copy()
    df["cluster"] = km.fit_predict(embeddings)
    df["distance_to_centroid"] = np.linalg.norm(
        embeddings - km.cluster_centers_[df["cluster"]], axis=1
    )
    
    # Identify outliers (emails far from their cluster centroid)
    outlier_threshold = np.percentile(df["distance_to_centroid"], 95)
    df["is_outlier"] = df["distance_to_centroid"] > outlier_threshold
    
    print(f"  Identified {df['is_outlier'].sum()} outliers (top 5% distance)")
    
    return df, km


def build_folders_hdbscan(df: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
    """Cluster using HDBSCAN (density-based, no need to specify k)."""
    if not HDBSCAN_AVAILABLE:
        raise RuntimeError("HDBSCAN not available. Install with: pip install hdbscan")
    
    print("\nClustering with HDBSCAN (density-based)...")
    
    # UMAP for dimensionality reduction (HDBSCAN works better in lower dimensions)
    if UMAP_AVAILABLE:
        print("  Reducing dimensions with UMAP...")
        reducer = umap.UMAP(n_components=10, random_state=42, metric="cosine")
        embeddings_reduced = reducer.fit_transform(embeddings)
    else:
        embeddings_reduced = embeddings
    
    clusterer = hdbscan.HDBSCAN(min_cluster_size=20, min_samples=10, metric="euclidean")
    df = df.copy()
    df["cluster"] = clusterer.fit_predict(embeddings_reduced)
    
    # HDBSCAN assigns -1 to noise/outliers
    n_outliers = (df["cluster"] == -1).sum()
    print(f"  Found {len(df[df['cluster'] != -1]['cluster'].unique())} clusters")
    print(f"  Identified {n_outliers} outliers (noise points)")
    
    return df, clusterer


def build_folders(df: pd.DataFrame, embeddings: np.ndarray, k: int, use_hdbscan: bool = False):
    """Main clustering function with folder naming."""
    if use_hdbscan:
        df, model = build_folders_hdbscan(df, embeddings)
    else:
        df, model = build_folders_kmeans(df, embeddings, k)
    
    print(f"\n{'=' * 70}\nSEMANTIC FOLDERS\n{'=' * 70}")
    
    cluster_names = {}
    unique_clusters = sorted(df["cluster"].unique())
    
    # Handle HDBSCAN noise cluster
    if use_hdbscan and -1 in unique_clusters:
        cluster_names[-1] = "unclassified / outliers"
        unique_clusters = [c for c in unique_clusters if c != -1]
    
    for cluster_id in unique_clusters:
        subset = df[df["cluster"] == cluster_id]
        
        # Try two naming strategies and pick the better one
        name_tfidf = name_cluster_tfidf(subset["search_text"])
        
        # For KMeans, use centroid-based naming
        if hasattr(model, "cluster_centers_"):
            name_centroid = name_cluster_from_centroid_emails(
                subset, 
                embeddings[df["cluster"] == cluster_id],
                model.cluster_centers_[cluster_id]
            )
            # Prefer centroid-based if it's more specific
            name = name_centroid if len(name_centroid) > len(name_tfidf) else name_tfidf
        else:
            name = name_tfidf
        
        cluster_names[cluster_id] = name
        
        # Cluster statistics
        dominant_categories = subset["category"].value_counts(normalize=True).head(3)
        cat_str = ", ".join(f"{c} {p:.0%}" for c, p in dominant_categories.items())
        
        avg_distance = subset["distance_to_centroid"].mean() if "distance_to_centroid" in subset.columns else 0
        outlier_count = subset["is_outlier"].sum() if "is_outlier" in subset.columns else 0
        
        print(f"\nFolder {cluster_id}: \"{name}\"  ({len(subset)} emails)")
        print(f"  dominant true categories: {cat_str}")
        print(f"  avg distance to centroid: {avg_distance:.3f}  |  outliers: {outlier_count}")
        print(f"  sample subjects:")
        for subj in subset["subject"].drop_duplicates().head(3):
            print(f"    - {subj}")
    
    df["folder_name"] = df["cluster"].map(cluster_names)
    return df, model


def visualize_clusters(df: pd.DataFrame, embeddings: np.ndarray):
    """Create 2D visualization of clusters using UMAP."""
    if not UMAP_AVAILABLE:
        print("\nSkipping visualization (UMAP not available)")
        return
    
    print("\nCreating 2D visualization with UMAP...")
    reducer = umap.UMAP(n_components=2, random_state=42, metric="cosine")
    embeddings_2d = reducer.fit_transform(embeddings)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Color by cluster
    unique_clusters = sorted(df["cluster"].unique())
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))
    
    for i, cluster_id in enumerate(unique_clusters):
        mask = df["cluster"] == cluster_id
        ax.scatter(
            embeddings_2d[mask, 0], 
            embeddings_2d[mask, 1],
            c=[colors[i]],
            label=f"Cluster {cluster_id}",
            alpha=0.6,
            s=20
        )
    
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    ax.set_title("Semantic Email Clusters (UMAP Visualization)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    
    fig.tight_layout()
    fig.savefig("semantic_folders_visualization.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved visualization to semantic_folders_visualization.png")


def validate_against_categories(df: pd.DataFrame):
    """Validate clustering against known categories."""
    print(f"\n{'=' * 70}\nVALIDATION: clusters vs. known categories\n{'=' * 70}")
    print("(Clustering did NOT use 'category' as input)\n")
    
    # Filter out outliers/noise for ARI calculation
    df_valid = df[df["cluster"] != -1] if "cluster" in df.columns and -1 in df["cluster"].values else df
    
    ari = adjusted_rand_score(df_valid["category"], df_valid["cluster"])
    print(f"Adjusted Rand Index: {ari:.3f}")
    print("(1.0 = perfect rediscovery, 0.0 = no better than random)")
    
    # Cluster purity (how homogeneous each cluster is)
    print("\nCluster purity (fraction of dominant category in each cluster):")
    purities = []
    for cluster_id in sorted(df_valid["cluster"].unique()):
        subset = df_valid[df_valid["cluster"] == cluster_id]
        dominant_cat = subset["category"].value_counts(normalize=True).iloc[0]
        purities.append(dominant_cat)
        print(f"  Cluster {cluster_id}: {dominant_cat:.1%}")
    
    avg_purity = np.mean(purities)
    print(f"\nAverage cluster purity: {avg_purity:.1%}")
    
    print("\nCrosstab: cluster x category (row-normalized %):")
    crosstab = pd.crosstab(df_valid["cluster"], df_valid["category"], normalize="index") * 100
    print(crosstab.round(1))
    
    print("\nClusters with no single category above 70% (mixed/cross-category):")
    max_share = crosstab.max(axis=1)
    mixed = max_share[max_share < 70]
    if len(mixed):
        print(mixed.round(1))
    else:
        print("None -- every cluster is dominated by one category.")


def save_model(model, filepath: str):
    """Save clustering model for future predictions."""
    with open(filepath, "wb") as f:
        pickle.dump(model, f)
    print(f"\nSaved clustering model to {filepath}")


def main(emails_dir, labels_csv, k=None, scan=False, use_hdbscan=False, save_model_flag=False):
    df = load_emails(emails_dir, labels_csv)
    df = clean_dataframe(df)
    
    print(f"Loaded {len(df)} emails")

    embeddings, ids = load_embeddings()
    df, embeddings = align_embeddings_to_df(df, embeddings, ids)
    print(f"Matched {len(df)} emails with embeddings")

    # Determine clustering method and parameters
    if use_hdbscan:
        if not HDBSCAN_AVAILABLE:
            print("Error: HDBSCAN not available")
            return df
        model_type = "hdbscan"
    else:
        if scan or k is None:
            best_k = scan_k(embeddings)
            if k is None:
                k = best_k
        model_type = "kmeans"

    # Build clusters
    df, model = build_folders(df, embeddings, k if not use_hdbscan else None, use_hdbscan)
    
    # Visualize
    visualize_clusters(df, embeddings)
    
    # Validate
    validate_against_categories(df)

    # Save results
    df.to_csv("emails_with_semantic_folders.csv", index=False)
    print(f"\nSaved full results to emails_with_semantic_folders.csv")
    
    # Save model if requested
    if save_model_flag:
        model_filename = f"semantic_folder_model_{model_type}.pkl"
        save_model(model, model_filename)

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic email folder clustering")
    parser.add_argument("emails_dir", nargs="?", default="generated_emails")
    parser.add_argument("labels_csv", nargs="?", default="labels.csv")
    parser.add_argument("--k", type=int, default=None, help="number of clusters (default: auto-scan)")
    parser.add_argument("--scan", action="store_true", help="run multi-metric scan across several k values")
    parser.add_argument("--hdbscan", action="store_true", help="use HDBSCAN instead of KMeans (density-based)")
    parser.add_argument("--save-model", action="store_true", help="save clustering model for future predictions")
    args = parser.parse_args()

    main(
        args.emails_dir, 
        args.labels_csv, 
        k=args.k, 
        scan=args.scan, 
        use_hdbscan=args.hdbscan,
        save_model_flag=args.save_model
    )