"""
baseline_classifier.py — Phase 3, step 2.

Trains two classical baselines on the TF-IDF + engineered feature
matrix from features.py, using the leakage-safe split from split.py.
This produces the benchmark number that Phase 4's transformer
embeddings need to beat to justify their extra complexity/cost.

Usage:
    python baseline_classifier.py generated_emails labels.csv
"""

import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from loader import load_emails
from cleaner import clean_dataframe
from split import group_aware_split, verify_no_leakage
from features import FeatureBuilder


def train_and_evaluate(model, model_name, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print(f"\n{'=' * 60}")
    print(f"{model_name}")
    print(f"{'=' * 60}")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Macro F1:  {macro_f1:.3f}   (unweighted average across classes -- "
          f"penalizes ignoring small classes like Government, unlike accuracy)")
    print()
    print(classification_report(y_test, y_pred, zero_division=0))

    return {"model": model, "name": model_name, "accuracy": acc, "macro_f1": macro_f1,
            "y_pred": y_pred}


def plot_confusion_matrix(y_test, y_pred, labels, title, filename):
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(filename, dpi=120)
    plt.close(fig)


def top_features_per_class(model, feature_names, class_labels, top_n=8):
    """For Logistic Regression: which features drove each category's predictions."""
    print("\nTop features per category (Logistic Regression coefficients):")
    for i, cls in enumerate(class_labels):
        coefs = model.coef_[i]
        top_idx = np.argsort(coefs)[-top_n:][::-1]
        top_feats = [(feature_names[j], round(coefs[j], 2)) for j in top_idx]
        print(f"  {cls:12s}: {top_feats}")


def main(emails_dir, labels_csv):
    df = load_emails(emails_dir, labels_csv)
    df = clean_dataframe(df)
    train_df, test_df = group_aware_split(df)
    verify_no_leakage(train_df, test_df)

    fb = FeatureBuilder()
    X_train, y_train = fb.fit_transform(train_df)
    X_test, y_test = fb.transform(test_df)

    fb.save("feature_builder.joblib")

    print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

    results = []

    nb = MultinomialNB(alpha=0.1)
    results.append(train_and_evaluate(nb, "Multinomial Naive Bayes", X_train, y_train, X_test, y_test))

    lr = LogisticRegression(max_iter=1000, C=1.0)
    results.append(train_and_evaluate(lr, "Logistic Regression", X_train, y_train, X_test, y_test))

    svc = LinearSVC(C=1.0, max_iter=5000)
    results.append(train_and_evaluate(svc, "Linear SVC", X_train, y_train, X_test, y_test))

    labels = sorted(df["category"].unique())
    for r in results:
        safe_name = r["name"].lower().replace(" ", "_")
        plot_confusion_matrix(y_test, r["y_pred"], labels,
                               f"{r['name']} — confusion matrix",
                               f"confusion_matrix_{safe_name}.png")

    top_features_per_class(lr, fb.feature_names_, lr.classes_)

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for r in results:
        print(f"{r['name']:30s} accuracy={r['accuracy']:.3f}  macro_f1={r['macro_f1']:.3f}")

    return results


if __name__ == "__main__":
    emails_dir = sys.argv[1] if len(sys.argv) > 1 else "generated_emails"
    labels_csv = sys.argv[2] if len(sys.argv) > 2 else "labels.csv"
    main(emails_dir, labels_csv)