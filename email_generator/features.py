"""
features.py — Phase 3, step 1.

Builds the feature matrix a baseline classifier trains on: TF-IDF over
clean_text, combined with the engineered signals EDA already validated
as meaningful (has_url, is_thread_reply, word_count, etc.).

Critical rule: TfidfVectorizer.fit() is called on TRAIN ONLY, then used
to .transform() both train and test. Fitting on the full dataset before
splitting is a classic leakage bug -- the vectorizer would "see" test
vocabulary during fitting.

Usage:
    from split import group_aware_split
    from features import FeatureBuilder

    train_df, test_df = group_aware_split(df)
    fb = FeatureBuilder()
    X_train, y_train = fb.fit_transform(train_df)
    X_test, y_test = fb.transform(test_df)
"""

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sentence_transformers import SentenceTransformer


ENGINEERED_NUMERIC_COLS = ["word_count", "char_count", "url_count", "num_attachments"]
ENGINEERED_BOOL_COLS = ["has_url", "has_attachment", "is_thread_reply", "used_html_fallback"]


def extract_domain(df: pd.DataFrame) -> pd.Series:
    """'brightline@nimbusretail.com' -> 'nimbusretail.com'."""
    return df["from_address"].str.split("@").str[-1].fillna("unknown")


class FeatureBuilder:
    def __init__(self, max_tfidf_features: int = 5000, ngram_range=(1, 2),
                 text_fields=("subject", "clean_text"), include_engineered: bool = True,
                 include_domain: bool = True):
        """
        text_fields: which columns to concatenate as TF-IDF input, e.g.
            ("clean_text",)            -> body only
            ("subject", "clean_text")  -> subject + body (default — subjects are
                                         highly discriminative for real-world email
                                         categories and improve classifier accuracy)
        include_engineered: whether to add word_count/has_url/etc.
        include_domain: whether to add one-hot sender_domain features.
        Set both False + text_fields=("clean_text",) for a pure
        "body text only, no metadata at all" ablation.
        max_tfidf_features: vocabulary size. 5000 covers synthetic + real Gmail
            vocabulary without overfitting.
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_tfidf_features,
            ngram_range=ngram_range,
            stop_words="english",
            min_df=2,
        )
        self.domain_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True) if include_domain else None

        self.text_fields = list(text_fields)
        self.include_engineered = include_engineered
        self.include_domain = include_domain

        self._numeric_mins = None
        self._numeric_maxs = None
        self.feature_names_ = None

    def _combined_text(self, df: pd.DataFrame) -> pd.Series:
        if len(self.text_fields) == 1:
            return df[self.text_fields[0]].fillna("")
        return df[self.text_fields].fillna("").agg(" ".join, axis=1)

    def _engineered_matrix(self, df: pd.DataFrame, fit: bool):
        numeric = df[ENGINEERED_NUMERIC_COLS].astype(float).values
        boolean = df[ENGINEERED_BOOL_COLS].astype(int).values

        if fit:
            self._numeric_mins = numeric.min(axis=0)
            self._numeric_maxs = numeric.max(axis=0)

        ranges = self._numeric_maxs - self._numeric_mins
        ranges[ranges == 0] = 1.0  # avoid div-by-zero for constant columns
        # Min-max scale to [0, 1] rather than z-score standardize: keeps all
        # values non-negative, which MultinomialNB requires (it models
        # feature counts and errors on negative input).
        numeric_scaled = np.clip((numeric - self._numeric_mins) / ranges, 0, 1)
        return np.hstack([numeric_scaled, boolean])

    def fit_transform(self, df: pd.DataFrame, label_col: str = "category"):
        tfidf_matrix = self.vectorizer.fit_transform(self._combined_text(df))
        parts = [tfidf_matrix]
        self.feature_names_ = list(self.vectorizer.get_feature_names_out())

        if self.include_engineered:
            engineered = self._engineered_matrix(df, fit=True)
            parts.append(csr_matrix(engineered))
            self.feature_names_ += ENGINEERED_NUMERIC_COLS + ENGINEERED_BOOL_COLS

        if self.include_domain:
            domains = extract_domain(df).values.reshape(-1, 1)
            domain_matrix = self.domain_encoder.fit_transform(domains)
            parts.append(domain_matrix)
            self.feature_names_ += [f"sender_domain_{d}" for d in self.domain_encoder.categories_[0]]

        X = hstack(parts).tocsr()
        y = df[label_col].values
        return X, y

    def transform(self, df: pd.DataFrame, label_col: str = "category"):
        tfidf_matrix = self.vectorizer.transform(self._combined_text(df))
        parts = [tfidf_matrix]

        if self.include_engineered:
            engineered = self._engineered_matrix(df, fit=False)
            parts.append(csr_matrix(engineered))

        if self.include_domain:
            domains = extract_domain(df).values.reshape(-1, 1)
            domain_matrix = self.domain_encoder.transform(domains)
            parts.append(domain_matrix)

        X = hstack(parts).tocsr()
        y = df[label_col].values
        return X, y

    def save(self, path: str):
        """Persist the entire fitted pipeline (vectorizer + domain encoder +
        scaling params + feature names) -- not just the vectorizer alone.
        At deployment time you need all of these to reproduce the exact
        same feature space the model was trained on; saving the vectorizer
        only would silently break sender-domain and numeric-scaling
        features."""
        joblib.dump(self, path)
        print(f"FeatureBuilder saved to {path}")

    @staticmethod
    def load(path: str) -> "FeatureBuilder":
        return joblib.load(path)


class EmbeddingFeatureBuilder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2",
                 text_fields=("subject", "clean_text"), include_engineered: bool = True,
                 include_domain: bool = True):
        """
        model_name: SentenceTransformer model to use.
        text_fields: which columns to concatenate as embedding input.
        include_engineered: whether to add word_count/has_url/etc.
        include_domain: whether to add one-hot sender_domain features.
        """
        self.model_name = model_name
        self._model = None  # Lazy load

        self.domain_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False) if include_domain else None

        self.text_fields = list(text_fields)
        self.include_engineered = include_engineered
        self.include_domain = include_domain

        self._numeric_mins = None
        self._numeric_maxs = None
        self.feature_names_ = None

    @property
    def model(self):
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _combined_text(self, df: pd.DataFrame) -> list:
        if len(self.text_fields) == 1:
            return df[self.text_fields[0]].fillna("").tolist()
        return df[self.text_fields].fillna("").agg(" ".join, axis=1).tolist()

    def _engineered_matrix(self, df: pd.DataFrame, fit: bool):
        numeric = df[ENGINEERED_NUMERIC_COLS].astype(float).values
        boolean = df[ENGINEERED_BOOL_COLS].astype(int).values

        if fit:
            self._numeric_mins = numeric.min(axis=0)
            self._numeric_maxs = numeric.max(axis=0)

        ranges = self._numeric_maxs - self._numeric_mins
        ranges[ranges == 0] = 1.0
        numeric_scaled = np.clip((numeric - self._numeric_mins) / ranges, 0, 1)
        return np.hstack([numeric_scaled, boolean])

    def fit_transform(self, df: pd.DataFrame, label_col: str = "category"):
        # Embeddings
        embeddings = self.model.encode(self._combined_text(df), show_progress_bar=True)
        parts = [embeddings]
        self.feature_names_ = [f"emb_{i}" for i in range(embeddings.shape[1])]

        if self.include_engineered:
            engineered = self._engineered_matrix(df, fit=True)
            parts.append(engineered)
            self.feature_names_ += ENGINEERED_NUMERIC_COLS + ENGINEERED_BOOL_COLS

        if self.include_domain:
            domains = extract_domain(df).values.reshape(-1, 1)
            domain_matrix = self.domain_encoder.fit_transform(domains)
            parts.append(domain_matrix)
            self.feature_names_ += [f"sender_domain_{d}" for d in self.domain_encoder.categories_[0]]

        X = np.hstack(parts)
        y = df[label_col].values
        return X, y

    def transform(self, df: pd.DataFrame, label_col: str = "category"):
        # Embeddings
        embeddings = self.model.encode(self._combined_text(df), show_progress_bar=True)
        parts = [embeddings]

        if self.include_engineered:
            engineered = self._engineered_matrix(df, fit=False)
            parts.append(engineered)

        if self.include_domain:
            domains = extract_domain(df).values.reshape(-1, 1)
            domain_matrix = self.domain_encoder.transform(domains)
            parts.append(domain_matrix)

        X = np.hstack(parts)
        y = df[label_col].values if label_col in df.columns else None
        return X, y

    def save(self, path: str):
        # Don't pickling the model itself to keep file size small
        tmp_model = self._model
        self._model = None
        joblib.dump(self, path)
        self._model = tmp_model
        print(f"EmbeddingFeatureBuilder saved to {path}")

    @staticmethod
    def load(path: str) -> "EmbeddingFeatureBuilder":
        return joblib.load(path)


if __name__ == "__main__":
    import sys
    from loader import load_emails
    from cleaner import clean_dataframe
    from split import group_aware_split

    emails_dir = sys.argv[1] if len(sys.argv) > 1 else "generated_emails"
    labels_csv = sys.argv[2] if len(sys.argv) > 2 else "labels.csv"

    df = load_emails(emails_dir, labels_csv)
    df = clean_dataframe(df)
    train_df, test_df = group_aware_split(df)

    fb = FeatureBuilder()
    X_train, y_train = fb.fit_transform(train_df)
    X_test, y_test = fb.transform(test_df)

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape:  {X_test.shape}")
    print(f"Total feature count: {len(fb.feature_names_)} "
          f"({len(fb.feature_names_) - len(ENGINEERED_NUMERIC_COLS) - len(ENGINEERED_BOOL_COLS)} TF-IDF + "
          f"{len(ENGINEERED_NUMERIC_COLS) + len(ENGINEERED_BOOL_COLS)} engineered)")
    print(f"Sample feature names: {fb.feature_names_[:10]} ... {fb.feature_names_[-8:]}")