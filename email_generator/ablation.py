"""
ablation_experiments.py — Phase 3, step 3.

Runs the same classifiers across three feature configurations to see
how much of the 99.8% "full feature" accuracy is genuine text
understanding versus metadata shortcuts (mainly sender_domain, which
EDA showed is near-perfectly category-partitioned in this synthetic
dataset and won't be this clean on real Gmail data).

Configs:
    A. Full        -- subject+body text + engineered features + sender_domain
    B. No domain   -- subject+body text + engineered features (no domain)
    C. Body only   -- body text only, no metadata at all

Usage:
    python ablation_experiments.py generated_emails labels.csv
"""

import sys

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report

from loader import load_emails
from cleaner import clean_dataframe
from split import group_aware_split, verify_no_leakage
from features import FeatureBuilder


CONFIGS = {
    "A. Full (text + metadata + domain)": dict(
        text_fields=("subject", "clean_text"), include_engineered=True, include_domain=True,
    ),
    "B. No domain (text + metadata)": dict(
        text_fields=("subject", "clean_text"), include_engineered=True, include_domain=False,
    ),
    "C. Body only (no metadata at all)": dict(
        text_fields=("clean_text",), include_engineered=False, include_domain=False,
    ),
}

MODELS = {
    "Naive Bayes": lambda: MultinomialNB(alpha=0.1),
    "Logistic Regression": lambda: LogisticRegression(max_iter=1000, C=1.0),
    "Linear SVC": lambda: LinearSVC(C=1.0, max_iter=5000),
}


def run_config(config_name, config_kwargs, train_df, test_df):
    fb = FeatureBuilder(**config_kwargs)
    X_train, y_train = fb.fit_transform(train_df)
    X_test, y_test = fb.transform(test_df)

    print(f"\n{'#' * 70}")
    print(f"# {config_name}")
    print(f"{'#' * 70}")
    print(f"Feature count: {X_train.shape[1]}")

    results = {}
    for model_name, model_fn in MODELS.items():
        model = model_fn()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro")
        results[model_name] = {"accuracy": acc, "macro_f1": macro_f1}
        print(f"  {model_name:22s} accuracy={acc:.3f}  macro_f1={macro_f1:.3f}")

        if model_name == "Logistic Regression":
            print(f"\n  --- {model_name} per-class report ---")
            print(classification_report(y_test, y_pred, zero_division=0, digits=3))

    return results


def main(emails_dir, labels_csv):
    df = load_emails(emails_dir, labels_csv)
    df = clean_dataframe(df)
    train_df, test_df = group_aware_split(df)
    verify_no_leakage(train_df, test_df)

    all_results = {}
    for config_name, config_kwargs in CONFIGS.items():
        all_results[config_name] = run_config(config_name, config_kwargs, train_df, test_df)

    print(f"\n{'=' * 70}")
    print("SUMMARY -- accuracy / macro_f1 by config x model")
    print(f"{'=' * 70}")
    header = f"{'Config':38s}" + "".join(f"{m:22s}" for m in MODELS)
    print(header)
    for config_name in CONFIGS:
        row = f"{config_name:38s}"
        for model_name in MODELS:
            r = all_results[config_name][model_name]
            row += f"{r['accuracy']:.3f} / {r['macro_f1']:.3f}    "
        print(row)

    return all_results


if __name__ == "__main__":
    emails_dir = sys.argv[1] if len(sys.argv) > 1 else "generated_emails"
    labels_csv = sys.argv[2] if len(sys.argv) > 2 else "labels.csv"
    main(emails_dir, labels_csv)