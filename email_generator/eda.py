"""
eda.py — Phase 2, step 3 (full version).

One function per research question, each self-contained and runnable
standalone. Produces a text report + 12 saved figures in eda_figures/.

Usage:
    python eda.py generated_emails labels.csv
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from loader import load_emails
from cleaner import clean_dataframe

FIGURES_DIR = Path("eda_figures")


# --- 1. Dataset summary -----------------------------------------------------

def dataset_summary(df: pd.DataFrame):
    print("1. DATASET SUMMARY")
    print(f"Rows: {len(df)}  Columns: {df.shape[1]}")
    print(f"Date range: {df['date'].min()} -> {df['date'].max()}")
    print(f"Categories: {df['category'].nunique()}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    print()


# --- 2. Missing value report -------------------------------------------------

def missing_value_report(df: pd.DataFrame):
    print("2. MISSING VALUE REPORT")
    missing = {}
    for col in df.columns:
        if df[col].dtype == object:
            n_missing = (df[col].isna() | (df[col] == "")).sum()
        else:
            n_missing = df[col].isna().sum()
        if n_missing > 0:
            missing[col] = n_missing
    if not missing:
        print("No missing values in any column.")
    else:
        report = pd.Series(missing).sort_values(ascending=False)
        print(pd.DataFrame({"missing_count": report, "pct": (report / len(df) * 100).round(1)}))
    print()


# --- 3. Class balance --------------------------------------------------------

def class_balance(df: pd.DataFrame):
    counts = df["category"].value_counts()
    pct = (counts / len(df) * 100).round(1)
    print("3. CLASS BALANCE")
    print(pd.DataFrame({"count": counts, "pct": pct}))

    fig, ax = plt.subplots(figsize=(7, 4))
    counts.plot(kind="bar", ax=ax, color="#4C72B0")
    ax.set_ylabel("count")
    ax.set_title("Email count by category")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "01_class_balance.png", dpi=120)
    plt.close(fig)
    print()


# --- 4. Text length analysis --------------------------------------------------

def text_length_analysis(df: pd.DataFrame):
    print("4. TEXT LENGTH ANALYSIS (word_count, by category)")
    stats = df.groupby("category")["word_count"].agg(["mean", "std", "min", "max"]).round(1)
    print(stats.sort_values("mean", ascending=False))

    fig, ax = plt.subplots(figsize=(8, 4))
    df.boxplot(column="word_count", by="category", ax=ax, rot=45)
    ax.set_title("Word count by category")
    plt.suptitle("")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "02_text_length_by_category.png", dpi=120)
    plt.close(fig)
    print()


# --- 5. Subject analysis -------------------------------------------------------

def subject_analysis(df: pd.DataFrame):
    print("5. SUBJECT ANALYSIS")
    subj_len = df["subject"].str.split().str.len()
    print(f"Subject word count: mean={subj_len.mean():.1f}, min={subj_len.min()}, max={subj_len.max()}")

    thread_prefix_rate = df["subject"].str.startswith(("Re:", "Fwd:")).mean()
    print(f"Subjects with Re:/Fwd: prefix: {thread_prefix_rate:.1%}")

    dup_subject_rate = 1 - (df["subject"].nunique() / len(df))
    print(f"Exact duplicate subjects (rate): {dup_subject_rate:.1%}")

    fig, ax = plt.subplots(figsize=(7, 4))
    subj_len.plot(kind="hist", bins=15, ax=ax, color="#C44E52")
    ax.set_title("Subject word count distribution")
    ax.set_xlabel("words")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "03_subject_length_distribution.png", dpi=120)
    plt.close(fig)
    print()


# --- 6 & 7. Duplicate detection + template duplication ------------------------

def _template_signature(text: str) -> str:
    sig = re.sub(r"\d+", "#", str(text).lower())
    sig = re.sub(r"\s+", " ", sig).strip()
    return sig


def duplicate_detection(df: pd.DataFrame):
    print("6. DUPLICATE DETECTION (exact clean_text matches)")
    exact_dup_rate = 1 - (df["clean_text"].nunique() / len(df))
    print(f"Exact duplicate clean_text rate: {exact_dup_rate:.1%}")
    print()


def template_duplication(df: pd.DataFrame):
    print("7. TEMPLATE DUPLICATION (digit-normalized signature reuse)")
    df = df.copy()
    df["_sig"] = df["clean_text"].apply(_template_signature)
    n_unique = df["_sig"].nunique()
    dup_rate = 1 - (n_unique / len(df))
    print(f"Unique template signatures: {n_unique} / {len(df)} ({dup_rate:.1%} share a signature)")

    dup_rate_by_cat = df.groupby("category").apply(
        lambda g: 1 - (g["_sig"].nunique() / len(g)), include_groups=False
    ).sort_values(ascending=False)
    print("\nTemplate duplication rate by category:")
    print((dup_rate_by_cat * 100).round(1))

    thread_dup_rate = df.loc[df["is_thread_reply"], "_sig"].duplicated(keep=False).mean() if df["is_thread_reply"].any() else 0
    print(f"\nThread-reply rows that share a signature with another row: {thread_dup_rate:.1%}")

    fig, ax = plt.subplots(figsize=(7, 4))
    (dup_rate_by_cat * 100).plot(kind="bar", ax=ax, color="#8172B2")
    ax.set_ylabel("% duplicated")
    ax.set_title("Template duplication rate by category")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "04_template_duplication_by_category.png", dpi=120)
    plt.close(fig)
    print()


# --- 8. URL analysis -------------------------------------------------------------

def url_analysis(df: pd.DataFrame):
    print("8. URL ANALYSIS")
    rate_by_cat = df.groupby("category")["has_url"].mean().sort_values(ascending=False) * 100
    print("has_url rate by category (%):")
    print(rate_by_cat.round(1))
    print(f"\nAvg url_count when present: {df.loc[df['has_url'], 'url_count'].mean():.2f}")
    print(f"spam_score: has_url=True mean={df.loc[df['has_url'],'spam_score'].mean():.1f}, "
          f"has_url=False mean={df.loc[~df['has_url'],'spam_score'].mean():.1f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    rate_by_cat.plot(kind="bar", ax=ax, color="#64B5CD")
    ax.set_ylabel("% with URL")
    ax.set_title("URL presence rate by category")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "05_url_rate_by_category.png", dpi=120)
    plt.close(fig)
    print()


# --- 9. Attachment analysis ------------------------------------------------------

def attachment_analysis(df: pd.DataFrame):
    print("9. ATTACHMENT ANALYSIS")
    rate_by_cat = df.groupby("category")["has_attachment"].mean().sort_values(ascending=False) * 100
    print("has_attachment rate by category (%):")
    print(rate_by_cat.round(1))

    fig, ax = plt.subplots(figsize=(7, 4))
    rate_by_cat.plot(kind="bar", ax=ax, color="#CCB974")
    ax.set_ylabel("% with attachment")
    ax.set_title("Attachment rate by category")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "06_attachment_rate_by_category.png", dpi=120)
    plt.close(fig)
    print()


# --- 10. Thread analysis ----------------------------------------------------------

def thread_analysis(df: pd.DataFrame):
    print("10. THREAD ANALYSIS")
    thread_rate = df["is_thread_reply"].mean()
    print(f"Overall thread-reply rate: {thread_rate:.1%}")

    rate_by_cat = df.groupby("category")["is_thread_reply"].mean().sort_values(ascending=False) * 100
    print("\nThread-reply rate by category (%):")
    print(rate_by_cat.round(1))

    thread_word_count = df.loc[df["is_thread_reply"], "word_count"].mean()
    non_thread_word_count = df.loc[~df["is_thread_reply"], "word_count"].mean()
    print(f"\nAvg word_count -- threads: {thread_word_count:.1f}, non-threads: {non_thread_word_count:.1f}")
    print("(threads are shorter after cleaning because quoted original content is stripped --"
          " see template_duplication findings for why this matters)")

    fig, ax = plt.subplots(figsize=(6, 4))
    df["is_thread_reply"].value_counts().plot(kind="pie", ax=ax, autopct="%1.1f%%",
                                                labels=["Original", "Thread reply"], colors=["#4C72B0", "#DD8452"])
    ax.set_ylabel("")
    ax.set_title("Thread reply share")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "07_thread_reply_share.png", dpi=120)
    plt.close(fig)
    print()


# --- 11. Sender domain analysis -----------------------------------------------------

def sender_domain_analysis(df: pd.DataFrame):
    print("11. SENDER DOMAIN ANALYSIS")
    domains = df["from_address"].str.split("@").str[-1]
    top_domains = domains.value_counts().head(12)
    print("Top 12 sender domains:")
    print(top_domains)

    domains_per_category = domains.groupby(df["category"]).nunique()
    print("\nUnique domains used per category:")
    print(domains_per_category.sort_values(ascending=False))

    fig, ax = plt.subplots(figsize=(8, 4))
    top_domains.plot(kind="barh", ax=ax, color="#937860")
    ax.set_xlabel("count")
    ax.set_title("Top sender domains")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "08_top_sender_domains.png", dpi=120)
    plt.close(fig)
    print()


# --- 12. HTML fallback analysis --------------------------------------------------------

def html_fallback_analysis(df: pd.DataFrame):
    print("12. HTML FALLBACK ANALYSIS")
    fallback_count = df["used_html_fallback"].sum()
    print(f"Emails that used HTML->text fallback (empty plain-text part): {fallback_count} "
          f"({fallback_count/len(df):.1%})")
    if fallback_count == 0:
        print("None -- every synthetic email has a populated text/plain part. "
              "This code path remains untested until real-world HTML-only mail is loaded.")
    print()


# --- 13. Vocabulary analysis (unigrams + bigrams) ------------------------------------------

def vocabulary_analysis(df: pd.DataFrame, top_n: int = 8):
    from sklearn.feature_extraction.text import CountVectorizer

    print(f"13. VOCABULARY ANALYSIS (top {top_n} unigrams + bigrams per category, stopwords removed)")
    all_bigram_counts = {}
    for cat in sorted(df["category"].unique()):
        texts = df.loc[df["category"] == cat, "clean_text"]

        uni_vec = CountVectorizer(stop_words="english", max_features=200, ngram_range=(1, 1))
        uni_matrix = uni_vec.fit_transform(texts)
        uni_freqs = uni_matrix.sum(axis=0).A1
        uni_top = sorted(zip(uni_vec.get_feature_names_out(), uni_freqs), key=lambda x: -x[1])[:top_n]

        bi_vec = CountVectorizer(stop_words="english", max_features=200, ngram_range=(2, 2))
        bi_matrix = bi_vec.fit_transform(texts)
        bi_freqs = bi_matrix.sum(axis=0).A1
        bi_top = sorted(zip(bi_vec.get_feature_names_out(), bi_freqs), key=lambda x: -x[1])[:top_n]
        for phrase, count in zip(bi_vec.get_feature_names_out(), bi_freqs):
            all_bigram_counts[phrase] = all_bigram_counts.get(phrase, 0) + count

        print(f"\n  {cat}")
        print(f"    unigrams: {', '.join(f'{w}({c})' for w, c in uni_top)}")
        print(f"    bigrams:  {', '.join(f'{w}({c})' for w, c in bi_top)}")

    top_overall_bigrams = sorted(all_bigram_counts.items(), key=lambda x: -x[1])[:15]
    fig, ax = plt.subplots(figsize=(8, 5))
    labels, values = zip(*top_overall_bigrams)
    ax.barh(labels[::-1], values[::-1], color="#4C72B0")
    ax.set_title("Top 15 bigrams overall")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "09_top_bigrams_overall.png", dpi=120)
    plt.close(fig)
    print()


# --- 14. Date coverage --------------------------------------------------------------

def date_coverage(df: pd.DataFrame):
    print("14. DATE COVERAGE")
    print(f"Range: {df['date'].min()} to {df['date'].max()}")
    monthly = df.set_index("date").resample("ME").size()
    print("\nEmails per month:")
    print(monthly)

    fig, ax = plt.subplots(figsize=(8, 4))
    monthly.plot(kind="bar", ax=ax, color="#55A868")
    ax.set_title("Emails per month")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "10_date_coverage.png", dpi=120)
    plt.close(fig)
    print()


# --- 15. Spam/Priority correlations --------------------------------------------------

def spam_priority_correlations(df: pd.DataFrame):
    print("15. SPAM / PRIORITY CORRELATIONS")
    print("\nMean spam_score by category:")
    print(df.groupby("category")["spam_score"].mean().round(1).sort_values(ascending=False))

    print("\nPriority distribution by category (%):")
    print((pd.crosstab(df["category"], df["priority"], normalize="index") * 100).round(1))

    corr = df["spam_score"].corr(df["priority_score"])
    print(f"\nspam_score vs priority_score correlation: {corr:.3f} "
          f"(expected negative: high-spam mail is deprioritized by design)")
    print()


# --- 16. Correlation heatmap -----------------------------------------------------------

def correlation_heatmap(df: pd.DataFrame):
    print("16. CORRELATION HEATMAP (numeric features)")
    numeric_cols = ["spam_score", "priority_score", "word_count", "char_count",
                     "url_count", "num_attachments"]
    numeric_df = df[numeric_cols].copy()
    numeric_df["is_thread_reply"] = df["is_thread_reply"].astype(int)
    numeric_df["has_url"] = df["has_url"].astype(int)
    numeric_df["has_attachment"] = df["has_attachment"].astype(int)

    corr = numeric_df.corr()
    print(corr.round(2))

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                     color="black", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Correlation heatmap (numeric + boolean features)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "11_correlation_heatmap.png", dpi=120)
    plt.close(fig)
    print()


# --- Runner ------------------------------------------------------------------------------

def run_full_report(emails_dir: str, labels_csv: str):
    FIGURES_DIR.mkdir(exist_ok=True)

    df = load_emails(emails_dir, labels_csv)
    df = clean_dataframe(df)

    sections = [
        dataset_summary, missing_value_report, class_balance, text_length_analysis,
        subject_analysis, duplicate_detection, template_duplication, url_analysis,
        attachment_analysis, thread_analysis, sender_domain_analysis,
        html_fallback_analysis, vocabulary_analysis, date_coverage,
        spam_priority_correlations, correlation_heatmap,
    ]

    for section_fn in sections:
        print("=" * 70)
        section_fn(df)

    print("=" * 70)
    n_figures = len(list(FIGURES_DIR.glob("*.png")))
    print(f"\n{n_figures} figures saved to {FIGURES_DIR}/")

    return df


if __name__ == "__main__":
    emails_dir = sys.argv[1] if len(sys.argv) > 1 else "generated_emails"
    labels_csv = sys.argv[2] if len(sys.argv) > 2 else "labels.csv"
    run_full_report(emails_dir, labels_csv)