"""
retrain_on_real_mail.py — Hybrid training pipeline.

Combines three data sources to retrain the category classifier:
  1. Synthetic emails (generated_emails/ + labels.csv)  -- 10,000 emails, balanced
  2. Real Gmail emails (gmail_emails/)                   -- ~50 emails, auto-labeled
  3. SpamAssassin corpus (../Spam_Assassin)              -- ~6,000 real emails, spam/ham

The combined corpus gives the model:
  - Real email vocabulary (vs. templated synthetic text)
  - Real sender-domain patterns
  - Real spam writing styles (SpamAssassin)

Outputs:
  - feature_builder.joblib   (fitted TF-IDF + engineered features pipeline)
  - classifier.joblib        (trained LogisticRegression model)
  - gmail_labels.csv         (auto-generated labels for Gmail emails)
  - retrain_report.txt       (accuracy / F1 scores on hold-out split)

Usage:
    cd email_generator
    python retrain_on_real_mail.py
"""

import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from cleaner import clean_dataframe, clean_email_body
from features import FeatureBuilder
from loader import load_emails
from split import group_aware_split

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SYNTHETIC_DIR = "generated_emails"
SYNTHETIC_LABELS = "labels.csv"
GMAIL_DIR = "gmail_emails"
GMAIL_LABELS_CSV = "gmail_labels.csv"
SPAMASSASSIN_DIR = Path("..") / "Spam_Assassin"
SPAMASSASSIN_CSV = "spamassassin_emails.csv"

FEATURE_BUILDER_PATH = "feature_builder.joblib"
CLASSIFIER_PATH = "classifier.joblib"
REPORT_PATH = "retrain_report.txt"


# ---------------------------------------------------------------------------
# 1. Synthetic dataset
# ---------------------------------------------------------------------------

def load_synthetic_data() -> pd.DataFrame:
    """Load and clean the full synthetic email dataset with labels."""
    if not Path(SYNTHETIC_DIR).exists() or not Path(SYNTHETIC_LABELS).exists():
        logger.warning(f"Synthetic dataset not found ({SYNTHETIC_DIR}/ + {SYNTHETIC_LABELS}). Skipping.")
        return pd.DataFrame()

    logger.info("Loading synthetic emails...")
    df = load_emails(SYNTHETIC_DIR, SYNTHETIC_LABELS)
    df = clean_dataframe(df)
    df["source"] = "synthetic"
    logger.info(f"  Synthetic: {len(df)} emails, categories: {sorted(df['category'].unique())}")
    return df


# ---------------------------------------------------------------------------
# 2. Real Gmail emails (auto-labeled)
# ---------------------------------------------------------------------------

def load_gmail_data() -> pd.DataFrame:
    """Auto-label Gmail emails and return as a training-ready DataFrame."""
    if not Path(GMAIL_DIR).exists():
        logger.warning(f"Gmail emails directory '{GMAIL_DIR}/' not found. Skipping.")
        return pd.DataFrame()

    # Auto-label using heuristic labeler
    try:
        from gmail_auto_labeler import label_gmail_emails
        labels_df = label_gmail_emails(GMAIL_DIR, GMAIL_LABELS_CSV, min_confidence=0.0)
    except Exception as e:
        logger.error(f"Auto-labeling failed: {e}")
        return pd.DataFrame()

    if len(labels_df) == 0:
        return pd.DataFrame()

    # Load raw emails
    df = load_emails(GMAIL_DIR)
    df = clean_dataframe(df)

    # Add required columns that FeatureBuilder expects
    if "spam_score" not in df.columns:
        df["spam_score"] = 0.0
    if "has_attachment" not in df.columns:
        df["has_attachment"] = df["num_attachments"] > 0
    if "is_thread_reply" not in df.columns:
        df["is_thread_reply"] = df["subject"].str.lower().str.startswith(("re:", "fwd:", "fw:"))

    # Merge auto-labels
    df = df.merge(labels_df[["filename", "category"]], on="filename", how="inner")
    df["source"] = "gmail"

    logger.info(f"  Gmail: {len(df)} emails, categories: {sorted(df['category'].unique())}")
    logger.info(f"  Gmail label distribution:\n{df['category'].value_counts().to_string()}")
    return df


# ---------------------------------------------------------------------------
# 3. SpamAssassin corpus (real-world spam/ham)
# ---------------------------------------------------------------------------

def _map_spamassassin_label(row: pd.Series) -> str:
    """
    Map SpamAssassin binary labels to the synthetic category space.

    spam  → Spam  (direct mapping)
    ham   → use keyword scoring to assign a realistic non-Spam category
    """
    if row.get("label") == "spam":
        return "Spam"

    # For ham, use keyword scoring to find a meaningful category
    text = f"{row.get('subject', '')} {row.get('body_text', '')[:500]}".lower()

    keyword_scores = {
        "Finance": sum(1 for kw in ["invoice", "bank", "payment", "account", "credit", "transaction"] if kw in text),
        "Job": sum(1 for kw in ["job", "position", "interview", "resume", "hiring", "career"] if kw in text),
        "Travel": sum(1 for kw in ["flight", "hotel", "booking", "reservation", "trip", "travel"] if kw in text),
        "Shopping": sum(1 for kw in ["order", "shipped", "delivery", "product", "purchase", "buy"] if kw in text),
        "College": sum(1 for kw in ["course", "assignment", "university", "grade", "exam", "lecture"] if kw in text),
        "Government": sum(1 for kw in ["government", "tax", "official", "ministry", "notice", "court"] if kw in text),
    }

    best_category = max(keyword_scores, key=keyword_scores.get)
    if keyword_scores[best_category] == 0:
        return "Social"  # generic non-spam fallback

    return best_category


def load_spamassassin_data(max_per_class: int = 300) -> pd.DataFrame:
    """Load SpamAssassin corpus and map to the 8-category label space."""
    # Use pre-parsed CSV if available (much faster than re-parsing all files)
    if Path(SPAMASSASSIN_CSV).exists():
        logger.info(f"Loading SpamAssassin from cached CSV ({SPAMASSASSIN_CSV})...")
        df = pd.read_csv(SPAMASSASSIN_CSV)
    elif SPAMASSASSIN_DIR.exists():
        logger.info(f"Loading SpamAssassin from raw corpus ({SPAMASSASSIN_DIR})...")
        try:
            from spamassassin_loader import load_spamassassin_corpus
            df = load_spamassassin_corpus(str(SPAMASSASSIN_DIR))
        except Exception as e:
            logger.error(f"SpamAssassin load error: {e}")
            return pd.DataFrame()
    else:
        logger.warning(f"SpamAssassin corpus not found at {SPAMASSASSIN_DIR}. Skipping.")
        return pd.DataFrame()

    if len(df) == 0:
        return pd.DataFrame()

    # Sample to avoid overwhelming the synthetic data
    spam_df = df[df["label"] == "spam"].head(max_per_class).copy()
    ham_df = df[df["label"] == "ham"].head(max_per_class * 2).copy()
    df = pd.concat([spam_df, ham_df], ignore_index=True)

    # Map to category space
    logger.info("Mapping SpamAssassin labels to category space...")
    df["category"] = df.apply(_map_spamassassin_label, axis=1)

    # Ensure body_text exists
    if "body_text" not in df.columns:
        df["body_text"] = ""
    if "body_html" not in df.columns:
        df["body_html"] = ""
    if "subject" not in df.columns:
        df["subject"] = ""
    if "from_address" not in df.columns:
        df["from_address"] = ""
    if "num_attachments" not in df.columns:
        df["num_attachments"] = 0

    # Fill NaN string columns before cleaning (CSV parsing turns missing values
    # into float NaN, and clean_email_body calls .strip() which fails on float)
    for col in ("body_text", "body_html", "subject", "from_address"):
        if col in df.columns:
            df[col] = df[col].fillna("")

    # Clean text
    df = clean_dataframe(df)


    # Add required feature columns
    if "spam_score" not in df.columns:
        df["spam_score"] = df["label"].map({"spam": 80.0, "ham": 5.0}).fillna(5.0)
    if "has_attachment" not in df.columns:
        df["has_attachment"] = df["num_attachments"] > 0
    if "is_thread_reply" not in df.columns:
        df["is_thread_reply"] = df["subject"].str.lower().str.startswith(("re:", "fwd:", "fw:"))

    df["source"] = "spamassassin"
    logger.info(f"  SpamAssassin: {len(df)} emails, categories:\n{df['category'].value_counts().to_string()}")
    return df


# ---------------------------------------------------------------------------
# 4. Hybrid training
# ---------------------------------------------------------------------------

REQUIRED_FEATURE_COLS = [
    "filename", "subject", "clean_text", "search_text",
    "from_address", "word_count", "char_count", "url_count",
    "num_attachments", "has_url", "has_attachment", "is_thread_reply",
    "used_html_fallback", "spam_score", "category", "source",
]


def align_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all required columns exist with sensible defaults."""
    if "subject" not in df.columns:
        df["subject"] = ""
    if "from_address" not in df.columns:
        df["from_address"] = ""
    if "spam_score" not in df.columns:
        df["spam_score"] = 0.0
    if "has_attachment" not in df.columns:
        df["has_attachment"] = df.get("num_attachments", pd.Series(0)) > 0
    if "is_thread_reply" not in df.columns:
        df["is_thread_reply"] = df.get("subject", pd.Series("")).str.lower().str.startswith(("re:", "fwd:", "fw:"))
    if "source" not in df.columns:
        df["source"] = "unknown"

    # Drop emails with missing category
    df = df.dropna(subset=["category"])
    return df


def retrain(
    synthetic_dir: str = SYNTHETIC_DIR,
    synthetic_labels: str = SYNTHETIC_LABELS,
    gmail_dir: str = GMAIL_DIR,
    feature_builder_path: str = FEATURE_BUILDER_PATH,
    classifier_path: str = CLASSIFIER_PATH,
    report_path: str = REPORT_PATH,
) -> dict:
    """
    Full hybrid retrain pipeline.
    Returns a dict with accuracy, macro_f1, report string.
    """
    logger.info("=" * 60)
    logger.info("HYBRID RETRAINING PIPELINE")
    logger.info("=" * 60)

    # --- Load all data sources ---
    dfs = []

    synthetic_df = load_synthetic_data()
    if len(synthetic_df) > 0:
        synthetic_df = align_columns(synthetic_df)
        dfs.append(synthetic_df)

    gmail_df = load_gmail_data()
    if len(gmail_df) > 0:
        gmail_df = align_columns(gmail_df)
        dfs.append(gmail_df)

    spamassassin_df = load_spamassassin_data(max_per_class=300)
    if len(spamassassin_df) > 0:
        spamassassin_df = align_columns(spamassassin_df)
        dfs.append(spamassassin_df)

    if not dfs:
        logger.error("No training data found. Aborting.")
        sys.exit(1)

    combined_df = pd.concat(dfs, ignore_index=True)

    logger.info(f"\nCombined training set: {len(combined_df)} emails")
    logger.info(f"Sources: {combined_df['source'].value_counts().to_dict()}")
    logger.info(f"Category distribution:\n{combined_df['category'].value_counts().to_string()}")

    # --- Train/test split ---
    # Use group_aware_split only for synthetic (which has message group IDs);
    # for the combined corpus, a simple stratified split is safer.
    train_df, test_df = train_test_split(
        combined_df,
        test_size=0.15,
        random_state=42,
        stratify=combined_df["category"],
    )
    logger.info(f"\nTrain: {len(train_df)}, Test: {len(test_df)}")

    # --- Feature engineering ---
    logger.info("\nBuilding features (subject + clean_text TF-IDF + engineered)...")
    fb = FeatureBuilder(
        max_tfidf_features=5000,
        ngram_range=(1, 2),
        text_fields=("subject", "clean_text"),
        include_engineered=True,
        include_domain=True,
    )

    X_train, y_train = fb.fit_transform(train_df)
    X_test, y_test = fb.transform(test_df)
    logger.info(f"Feature matrix shape: {X_train.shape}")

    # --- Train LogisticRegression ---
    logger.info("\nTraining LogisticRegression (C=2.0, max_iter=2000)...")
    clf = LogisticRegression(
        C=2.0,
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # --- Evaluate ---
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    report = classification_report(y_test, y_pred, zero_division=0)

    logger.info(f"\n{'='*60}")
    logger.info("EVALUATION RESULTS (15% hold-out test set)")
    logger.info(f"{'='*60}")
    logger.info(f"Accuracy : {accuracy:.3f}")
    logger.info(f"Macro F1 : {macro_f1:.3f}")
    logger.info(f"\n{report}")

    # Source-wise breakdown
    test_df = test_df.copy()
    test_df["predicted"] = y_pred
    logger.info("\nBreakdown by source:")
    for src in test_df["source"].unique():
        src_mask = test_df["source"] == src
        src_acc = accuracy_score(y_test[src_mask.values], y_pred[src_mask.values])
        logger.info(f"  {src:15s}: accuracy={src_acc:.3f}  n={src_mask.sum()}")

    # --- Save artefacts ---
    fb.save(feature_builder_path)
    joblib.dump(clf, classifier_path)
    logger.info(f"Saved feature builder -> {feature_builder_path}")
    logger.info(f"Saved classifier      -> {classifier_path}")

    result = {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "report": report,
        "train_size": len(train_df),
        "test_size": len(test_df),
        "classes": list(clf.classes_),
    }

    # Save report to disk
    report_lines = [
        "HYBRID RETRAIN REPORT",
        "=" * 60,
        f"Train size : {len(train_df)}",
        f"Test size  : {len(test_df)}",
        f"Sources    : {combined_df['source'].value_counts().to_dict()}",
        f"Accuracy   : {accuracy:.3f}",
        f"Macro F1   : {macro_f1:.3f}",
        "",
        report,
    ]
    Path(report_path).write_text("\n".join(report_lines), encoding="utf-8")
    logger.info(f"Saved report          → {report_path}")

    return result


if __name__ == "__main__":
    result = retrain()
    print(f"\n[DONE] Retraining complete!")
    print(f"   Accuracy : {result['accuracy']:.3f}")
    print(f"   Macro F1 : {result['macro_f1']:.3f}")
    print(f"   Classes  : {result['classes']}")
    print(f"\nModels saved: {FEATURE_BUILDER_PATH}, {CLASSIFIER_PATH}")
    print(f"Report saved: {REPORT_PATH}")

