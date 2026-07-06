"""
split.py — Phase 3, step 0 (do this before any feature engineering).

EDA found that 55% of emails share a digit-normalized template signature
with at least other email, and 100% of threaded replies collapse to a
handful of generic signatures after cleaning. A random train/test split
would put near-identical (or literally identical) rows on both sides,
inflating every metric downstream. This module prevents that by
splitting on template signature as the grouping key -- all rows sharing
a signature go entirely into train OR entirely into test, never both.

Usage:
    from split import group_aware_split
    train_df, test_df = group_aware_split(df)
"""

import re

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def template_signature(text: str) -> str:
    """Same normalization used in eda.py -- digits collapsed so template
    reuse groups together regardless of the specific amount/date/name."""
    sig = re.sub(r"\d+", "#", str(text).lower())
    sig = re.sub(r"\s+", " ", sig).strip()
    return sig


def group_aware_split(df: pd.DataFrame, test_size: float = 0.2, n_splits: int = 5, random_state: int = 42):
    """
    Split df into train/test such that:
    - no template signature appears in both train and test (prevents leakage)
    - category proportions are approximately preserved (stratification)

    Implementation: StratifiedGroupKFold gives n_splits folds; we take one
    fold as test and the rest as train. n_splits=5 -> ~20% test, matching
    the usual 80/20 convention.
    """
    df = df.copy()
    df["_template_sig"] = df["clean_text"].apply(template_signature)

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    train_idx, test_idx = next(sgkf.split(df, df["category"], groups=df["_template_sig"]))

    train_df = df.iloc[train_idx].drop(columns=["_template_sig"]).reset_index(drop=True)
    test_df = df.iloc[test_idx].drop(columns=["_template_sig"]).reset_index(drop=True)

    return train_df, test_df


def verify_no_leakage(train_df: pd.DataFrame, test_df: pd.DataFrame) -> bool:
    """Sanity check: assert zero template-signature overlap between splits."""
    train_sigs = set(train_df["clean_text"].apply(template_signature))
    test_sigs = set(test_df["clean_text"].apply(template_signature))
    overlap = train_sigs & test_sigs
    if overlap:
        print(f"LEAKAGE DETECTED: {len(overlap)} template signatures appear in both splits")
        return False
    print("No leakage: zero template signatures shared between train and test.")
    return True


if __name__ == "__main__":
    import sys
    from loader import load_emails
    from cleaner import clean_dataframe

    emails_dir = sys.argv[1] if len(sys.argv) > 1 else "generated_emails"
    labels_csv = sys.argv[2] if len(sys.argv) > 2 else "labels.csv"

    df = load_emails(emails_dir, labels_csv)
    df = clean_dataframe(df)

    train_df, test_df = group_aware_split(df)

    print(f"Train: {len(train_df)} rows ({len(train_df)/len(df):.1%})")
    print(f"Test:  {len(test_df)} rows ({len(test_df)/len(df):.1%})")
    print()
    verify_no_leakage(train_df, test_df)
    print()
    print("Category balance -- train vs test:")
    balance = pd.DataFrame({
        "train_pct": (train_df["category"].value_counts(normalize=True) * 100).round(1),
        "test_pct": (test_df["category"].value_counts(normalize=True) * 100).round(1),
    })
    print(balance)