"""
spamassassin_analysis.py

PART 1: Validates heuristic_labels.py's keyword-based spam_score heuristic
against REAL spam/ham -- not just the synthetic data it was implicitly
tuned against. Answers: does the heuristic actually generalize, or was
it just overfit to templates it wrote itself?

PART 2: Trains a standalone spam/ham classifier on real SpamAssassin
data, using a DUPLICATE-SAFE split. Real spam campaigns resend
byte-identical messages (your corpus has a 20.5% exact-duplicate rate)
-- a naive random split would leak identical messages across train/test,
the same risk your synthetic dataset had with template duplication,
just via exact repeats instead of templated variation.

This is kept STANDALONE from your synthetic 8-category classifier --
real 2002 ham/spam has no topical relationship to your synthetic
College/Shopping/Travel/etc. categories.

Usage:
    python spamassassin_analysis.py spamassassin_emails.csv
"""

import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, classification_report, confusion_matrix,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cleaner import clean_email_body
from heuristic_labels import compute_labels


def clean_corpus(df: pd.DataFrame) -> pd.DataFrame:
    """Reuses the exact same cleaner.py pipeline as the synthetic dataset --
    important for a fair test: the heuristic and classifier see real mail
    processed identically to how it processes synthetic mail."""
    records = df.apply(
        lambda r: clean_email_body(r["body_text"], r.get("body_html", "")), axis=1
    )
    clean_df = pd.DataFrame(list(records))
    return pd.concat([df.reset_index(drop=True), clean_df], axis=1)


def validate_heuristic(df: pd.DataFrame) -> pd.DataFrame:
    print("=" * 70)
    print("PART 1: Validating labeling.py spam_score heuristic against REAL spam/ham")
    print("=" * 70)

    # category="Unknown" -- not in CATEGORY_SPAM_BASE, so it falls back to
    # the dict's default (5), meaning this tests ONLY the keyword-matching
    # component of the heuristic, not the category base-rate component
    # (which has no meaning for real mail with no category label).
    scores = df.apply(
        lambda r: compute_labels("Unknown", r["subject"], r["clean_text"])[0], axis=1
    )
    df["heuristic_spam_score"] = scores

    print("\nheuristic_spam_score by real label:")
    print(df.groupby("label")["heuristic_spam_score"].describe()[["mean", "std", "min", "max"]].round(1))

    y_true = (df["label"] == "spam").astype(int)
    auc = roc_auc_score(y_true, df["heuristic_spam_score"])
    print(f"\nROC AUC (heuristic spam_score as a classifier score): {auc:.3f}")
    print("(0.5 = no better than random guessing, 1.0 = perfect separation)")
    print("This is the honest answer to: does a keyword list built while looking at\n"
          "synthetic spam actually generalize to real spam, or was it just pattern-\n"
          "matching its own templates?")

    fig, ax = plt.subplots(figsize=(7, 4))
    df[df["label"] == "spam"]["heuristic_spam_score"].plot(
        kind="hist", bins=30, alpha=0.6, label="real spam", ax=ax, color="crimson")
    df[df["label"] == "ham"]["heuristic_spam_score"].plot(
        kind="hist", bins=30, alpha=0.6, label="real ham", ax=ax, color="steelblue")
    ax.set_xlabel("heuristic_spam_score (from heuristic_labels.py, unmodified)")
    ax.set_title("heuristic_labels.py spam heuristic: real spam vs. real ham")
    ax.legend()
    fig.tight_layout()
    fig.savefig("spamassassin_heuristic_validation.png", dpi=120)
    plt.close(fig)
    print("Saved spamassassin_heuristic_validation.png")

    return df


def duplicate_safe_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Groups by exact clean_text (not the digit-normalized template
    signature used for the synthetic dataset) -- real spam campaigns
    repeat byte-for-byte, they don't vary by template placeholders."""
    df = df.copy()
    df["_dup_group"] = df["clean_text"].fillna("").apply(hash)
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(df, groups=df["_dup_group"]))
    train_df = df.iloc[train_idx].drop(columns=["_dup_group"]).reset_index(drop=True)
    test_df = df.iloc[test_idx].drop(columns=["_dup_group"]).reset_index(drop=True)
    return train_df, test_df


def train_classifier(df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("PART 2: Training a real-world spam/ham classifier")
    print("=" * 70)

    train_df, test_df = duplicate_safe_split(df)

    overlap = set(train_df["clean_text"]) & set(test_df["clean_text"])
    print(f"Train: {len(train_df)}  Test: {len(test_df)}  "
          f"Exact-text overlap: {len(overlap)} (should be 0)")
    if overlap:
        print("WARNING: leakage detected -- do not trust the metrics below until fixed.")

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english", min_df=2)
    X_train = vectorizer.fit_transform(train_df["clean_text"])
    X_test = vectorizer.transform(test_df["clean_text"])
    y_train = (train_df["label"] == "spam").astype(int)
    y_test = (test_df["label"] == "spam").astype(int)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    if y_test.nunique() < 2:
        print("WARNING: test split contains only one class -- ROC AUC undefined, "
              "and metrics below are not meaningful. This can happen with very "
              "small/highly-duplicated inputs; unlikely on the full real corpus.")
        auc = float("nan")
    else:
        auc = roc_auc_score(y_test, y_proba)

    print(f"\nAccuracy: {acc:.3f}   F1: {f1:.3f}   ROC AUC: {auc:.3f}")
    print(classification_report(y_test, y_pred, labels=[0, 1],
                                 target_names=["ham", "spam"], digits=3, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion matrix (rows=actual, cols=predicted, [ham,spam]):\n{cm}")

    # Which words actually drove real-world spam detection -- compare
    # this by eye against labeling.py's hand-picked SPAM_KEYWORDS list.
    feature_names = vectorizer.get_feature_names_out()
    top_spam_idx = np.argsort(clf.coef_[0])[-15:][::-1]
    print("\nTop 15 real-spam-indicating terms (learned from data, not hand-picked):")
    print([feature_names[i] for i in top_spam_idx])

    return clf, vectorizer


def main(csv_path: str):
    df = pd.read_csv(csv_path)
    df["body_text"] = df["body_text"].fillna("")
    if "body_html" not in df.columns:
        df["body_html"] = ""
    df["body_html"] = df["body_html"].fillna("")
    df["subject"] = df["subject"].fillna("")

    df = clean_corpus(df)
    df = validate_heuristic(df)
    train_classifier(df)


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "spamassassin_emails.csv"
    main(csv_path)