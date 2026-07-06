"""
spamassassin_loader.py — SpamAssassin corpus ingestion.

Parses the raw RFC822 email files from the Apache SpamAssassin public
corpus (https://spamassassin.apache.org/old/publiccorpus/) into a
DataFrame shaped like loader.py's output, so the SAME cleaner.py
functions work on it unchanged.

Two things this corpus is NOT good for, by design of this module:
- It is NOT merged into your synthetic 10k-email dataset. Real 2002
  corporate/personal ham has essentially nothing in common topically
  with your synthetic College/Shopping/Travel/etc. categories -- mixing
  them would just be distribution-shift noise, not useful training
  signal. This corpus is used STANDALONE, purely for a real-world
  spam-vs-ham test.
- Folder names determine the label (anything under a 'spam*' folder is
  spam, everything else is ham) -- there is no finer-grained category
  here, only a binary label.

Expected directory layout after extracting the .tar.bz2 archives:
    spamassassin_corpus/
    ├── spam/
    ├── spam_2/
    ├── easy_ham/
    ├── easy_ham_2/
    └── hard_ham/

Usage:
    python spamassassin_loader.py spamassassin_corpus
"""

import sys
import logging
from pathlib import Path
from email import policy
from email.parser import BytesParser
from email.utils import parseaddr, parsedate_to_datetime

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SPAM_FOLDER_PREFIXES = ("spam",)  # any folder starting with 'spam' -> label spam
HAM_FOLDER_PREFIXES = ("easy_ham", "hard_ham")


def _safe_get_content(part):
    """part.get_content() fails with LookupError when the email declares
    a charset that isn't a real Python codec name -- common in old mail
    from non-standard clients (e.g. 'DEFAULT_CHARSET', 'unknown-8bit',
    'CHINESEBIG5' used as a mislabeled alias). Rather than dropping the
    whole email over a bad charset LABEL, fall back to the raw decoded
    bytes with errors='replace' -- we still get usable text, just with
    a few replacement characters where the original charset was truly
    ambiguous."""
    if part is None:
        return ""
    try:
        return part.get_content()
    except (LookupError, UnicodeDecodeError):
        raw = part.get_payload(decode=True)
        if raw is None:
            return ""
        return raw.decode("utf-8", errors="replace")


def _safe_get_header(msg, header_name, default=""):
    """msg.get() can raise on malformed encoded-word headers (rare but
    present in 20+ year old mail) under policy.default's strict decoding.
    Fall back to the raw unparsed header string rather than losing the
    whole email over one bad header."""
    try:
        value = msg.get(header_name, default)
        return str(value) if value is not None else default
    except (UnicodeDecodeError, ValueError):
        # get() on the raw compat32 policy avoids the strict decode path
        try:
            return str(msg.get(header_name, default, policy=None))
        except Exception:
            return default


def _parse_single_raw_email(filepath: Path) -> dict:
    """Same parsing approach as loader.py's _parse_single_eml, adapted for
    files that aren't necessarily named .eml (SpamAssassin files have no
    extension, just numeric-ish names like '0001.7c53336b...')."""
    with open(filepath, "rb") as f:
        # SpamAssassin files occasionally have malformed headers (this is
        # 20+ year old real-world mail) -- BytesParser with policy.default
        # is reasonably tolerant, but we still wrap this whole function in
        # a try/except at the call site for the genuinely broken ones.
        msg = BytesParser(policy=policy.default).parse(f)

    plain_part = msg.get_body(preferencelist=("plain",))
    html_part = msg.get_body(preferencelist=("html",))
    body_text = _safe_get_content(plain_part)
    body_html = _safe_get_content(html_part)

    from_header = _safe_get_header(msg, "From")
    from_name, from_addr = parseaddr(from_header)

    date_header = _safe_get_header(msg, "Date")
    date = None
    if date_header:
        try:
            date = parsedate_to_datetime(date_header)
        except (TypeError, ValueError):
            date = None

    attachments = []
    try:
        for part in msg.iter_attachments():
            fname = part.get_filename()
            if fname:
                attachments.append(fname)
    except Exception:
        pass  # malformed multipart structure -- not worth failing the whole email over

    return {
        "filename": filepath.name,
        "from_name": from_name,
        "from_address": from_addr,
        "to": _safe_get_header(msg, "To"),
        "subject": _safe_get_header(msg, "Subject"),
        "date_raw": date_header,
        "date": date,
        "body_text": body_text,
        "body_html": body_html,
        "num_attachments": len(attachments),
        "attachment_names": attachments,
    }


def _label_for_folder(folder_name: str) -> str:
    if folder_name.startswith(SPAM_FOLDER_PREFIXES):
        return "spam"
    if folder_name.startswith(HAM_FOLDER_PREFIXES):
        return "ham"
    return "unknown"


def load_spamassassin_corpus(corpus_dir: str) -> pd.DataFrame:
    corpus_dir = Path(corpus_dir)
    if not corpus_dir.exists():
        raise FileNotFoundError(
            f"{corpus_dir} not found -- download and extract the SpamAssassin "
            f"public corpus first (see spamassassin_loader.py docstring for URLs)."
        )

    subfolders = [d for d in corpus_dir.iterdir() if d.is_dir()]
    if not subfolders:
        raise FileNotFoundError(
            f"No subfolders found in {corpus_dir} -- expected spam/, easy_ham/, "
            f"hard_ham/ etc. after extracting the .tar.bz2 archives."
        )

    records = []
    failures = 0
    for folder in subfolders:
        label = _label_for_folder(folder.name)
        if label == "unknown":
            logger.warning(f"Skipping folder with unrecognized name: {folder.name}")
            continue

        files = [f for f in folder.iterdir() if f.is_file() and not f.name.startswith(".")]
        # SpamAssassin folders sometimes include a 'cmds' index file -- skip non-email files
        files = [f for f in files if f.name != "cmds"]

        logger.info(f"Parsing {len(files)} files from {folder.name} (label={label})...")
        for filepath in files:
            try:
                record = _parse_single_raw_email(filepath)
                record["label"] = label
                record["source_folder"] = folder.name
                records.append(record)
            except Exception as e:
                failures += 1
                if failures <= 10:
                    logger.warning(f"Failed to parse {filepath.name}: {e}")

    if failures > 10:
        logger.warning(f"... and {failures - 10} more parse failures")

    df = pd.DataFrame.from_records(records)
    logger.info(f"Loaded {len(df)} emails ({failures} failures) from {corpus_dir}")
    logger.info(f"Label counts:\n{df['label'].value_counts().to_string()}")

    # Real corpora have some exact duplicate messages (spam campaigns
    # resend identical text) -- worth knowing before training on this.
    dup_rate = 1 - (df["body_text"].nunique() / len(df))
    logger.info(f"Exact duplicate body_text rate: {dup_rate:.1%}")

    return df


if __name__ == "__main__":
    corpus_dir = sys.argv[1] if len(sys.argv) > 1 else "spamassassin_corpus"
    df = load_spamassassin_corpus(corpus_dir)
    df.to_csv("spamassassin_emails.csv", index=False)
    print(f"\nSaved {len(df)} parsed emails to spamassassin_emails.csv")
    spam_sample = df[df["label"] == "spam"]
    ham_sample = df[df["label"] == "ham"]
    if len(spam_sample):
        print(f"\nSample spam subject: {spam_sample['subject'].iloc[0]!r}")
    if len(ham_sample):
        print(f"Sample ham subject:  {ham_sample['subject'].iloc[0]!r}")