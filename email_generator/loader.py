"""
loader.py — Phase 2, step 1.

Turns a directory of .eml files + labels.csv into one structured
pandas DataFrame that every downstream module (cleaner, EDA, embeddings,
classifier...) reads from. This is the only place that should ever touch
raw .eml files — everything after this works off the DataFrame.

Usage:
    from loader import load_emails
    df = load_emails("generated_emails", "labels.csv")

Design choices worth knowing about:
- Uses email.policy.default, which gives modern, RFC-compliant parsing
  and get_body()/iter_attachments() helpers instead of manual MIME
  walking.
- A single malformed file is logged and skipped rather than crashing
  the whole load — with 10,000+ files (and eventually real Gmail data,
  which is messier than synthetic data), some failures are inevitable.
- Both body_text and body_html are kept raw and untouched here.
  Cleaning (stripping quotes/signatures/HTML tags) is cleaner.py's job,
  not this module's — loader.py's only responsibility is faithful
  extraction, not normalization.
"""

import logging
from email import policy
from email.parser import BytesParser
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _parse_single_eml(filepath: Path) -> dict:
    """Parse one .eml file into a flat dict. Raises on unrecoverable errors;
    caller is responsible for catching and logging."""
    with open(filepath, "rb") as f:
        msg = BytesParser(policy=policy.default).parse(f)

    # Prefer the plain-text part; fall back to None if the email has no
    # text/plain part at all (cleaner.py will handle HTML-only fallback).
    plain_part = msg.get_body(preferencelist=("plain",))
    html_part = msg.get_body(preferencelist=("html",))

    body_text = plain_part.get_content() if plain_part else ""
    body_html = html_part.get_content() if html_part else ""

    attachments = []
    for part in msg.iter_attachments():
        fname = part.get_filename()
        if fname:
            attachments.append(fname)

    from_header = msg.get("From", "")
    from_name, from_addr = _split_name_address(from_header)

    return {
        "filename": filepath.name,
        "message_id": msg.get("Message-ID", ""),
        "in_reply_to": msg.get("In-Reply-To", ""),
        "from_name": from_name,
        "from_address": from_addr,
        "to": msg.get("To", ""),
        "subject": msg.get("Subject", ""),
        "date_raw": msg.get("Date", ""),
        "date": _parse_date(msg),
        "body_text": body_text,
        "body_html": body_html,
        "num_attachments": len(attachments),
        "attachment_names": attachments,
    }


def _split_name_address(header_value: str):
    """'James Smith <james@x.com>' -> ('James Smith', 'james@x.com').
    Falls back gracefully if there's no display name."""
    from email.utils import parseaddr
    name, addr = parseaddr(header_value)
    return name, addr


def _parse_date(msg):
    """Return a timezone-aware datetime, or None if the Date header is
    missing/unparseable (real-world mail sometimes has junk here)."""
    from email.utils import parsedate_to_datetime
    date_header = msg.get("Date")
    if not date_header:
        return None
    try:
        return parsedate_to_datetime(date_header)
    except (TypeError, ValueError):
        return None


def load_emails(emails_dir: str, labels_csv: str = None) -> pd.DataFrame:
    """
    Parse every .eml file in emails_dir into a DataFrame, optionally
    joined against a labels CSV (on the 'filename' column).

    Returns a DataFrame with one row per successfully parsed email.
    Failed files are logged and skipped, not raised -- check the
    returned DataFrame's row count against the file count if you need
    to know how many were dropped.
    """
    emails_dir = Path(emails_dir)
    if not emails_dir.exists():
        raise FileNotFoundError(f"emails_dir not found: {emails_dir}")

    eml_files = sorted(emails_dir.glob("*.eml"))
    if not eml_files:
        logger.warning(f"No .eml files found in {emails_dir}")

    records = []
    failures = []
    for filepath in eml_files:
        try:
            records.append(_parse_single_eml(filepath))
        except Exception as e:
            failures.append((filepath.name, str(e)))

    if failures:
        logger.warning(f"Failed to parse {len(failures)} of {len(eml_files)} files:")
        for fname, err in failures[:10]:
            logger.warning(f"  {fname}: {err}")
        if len(failures) > 10:
            logger.warning(f"  ... and {len(failures) - 10} more")

    df = pd.DataFrame.from_records(records)
    logger.info(f"Loaded {len(df)} of {len(eml_files)} emails from {emails_dir}")

    if labels_csv:
        df = _join_labels(df, labels_csv)

    return df


def _join_labels(df: pd.DataFrame, labels_csv: str) -> pd.DataFrame:
    labels_path = Path(labels_csv)
    if not labels_path.exists():
        logger.warning(f"labels_csv not found: {labels_path} — returning unlabeled data")
        return df

    labels_df = pd.read_csv(labels_path)

    # Only bring in columns the loader doesn't already produce itself,
    # to avoid duplicate/conflicting columns (e.g. labels.csv also has
    # its own 'subject' and 'date' copies -- we keep the loader's parsed
    # versions as the source of truth and only pull in the label columns).
    label_only_cols = [c for c in labels_df.columns
                        if c not in df.columns or c == "filename"]
    labels_df = labels_df[label_only_cols]

    merged = df.merge(labels_df, on="filename", how="left")

    missing = merged["category"].isna().sum() if "category" in merged.columns else 0
    if missing:
        logger.warning(f"{missing} emails had no matching row in {labels_path}")

    return merged


if __name__ == "__main__":
    # Quick smoke test against the dataset generated in Phase 1.
    import sys

    emails_dir = sys.argv[1] if len(sys.argv) > 1 else "generated_emails"
    labels_csv = sys.argv[2] if len(sys.argv) > 2 else "labels.csv"

    df = load_emails(emails_dir, labels_csv)
    print(f"\nShape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nDtypes:\n{df.dtypes}")
    print(f"\nSample row:\n{df.iloc[0]}")
    print(f"\nCategory counts:\n{df['category'].value_counts()}")
