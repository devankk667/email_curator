"""
cleaner.py — Phase 2, step 2.

Takes the raw body_text / body_html columns produced by loader.py and
produces a clean_text column (plus a few derived features) that's
actually safe to feed into embeddings, TF-IDF, or a classifier.

Usage:
    from loader import load_emails
    from cleaner import clean_dataframe

    df = load_emails("generated_emails", "labels.csv")
    df = clean_dataframe(df)   # adds clean_text, has_url, url_count, etc.

Design choices worth knowing about:
- Quoted reply chains (`> ...`) are stripped for classification purposes
  since they duplicate an earlier email's content and would bias a
  model trained on category/priority/spam labels. If you later need
  full thread context (e.g. for summarization), use the raw
  body_text/body_html columns instead -- cleaner.py never overwrites
  those, it only adds new columns.
- URLs are removed from clean_text but counted separately (has_url,
  url_count) since URL presence is itself a signal (e.g. for spam
  detection) that you don't want to lose just because the literal URL
  string isn't useful token content.
- Signature stripping is a heuristic (cuts text at the first sign-off
  phrase like "Regards," or "Thanks,"). It's deliberately generic
  rather than tuned to this project's exact synthetic signatures, so
  it'll have a fighting chance against real Gmail signatures later --
  but it WILL occasionally cut a false positive (e.g. a legitimate
  sentence that happens to start with "Thanks"). Check clean_text
  length distribution after running this; if it looks too aggressive,
  loosen SIGNATURE_MARKERS or drop this step.
- Lowercasing is a flag, not baked in. Classical ML (TF-IDF, bag-of-
  words) generally benefits from lowercasing; transformer-based
  embeddings usually don't need it and case can carry meaning (e.g.
  "US" vs "us"). Decide per downstream use, don't lowercase by default
  here.
"""

import re

import pandas as pd
from bs4 import BeautifulSoup

# --- Precompiled patterns -----------------------------------------------

URL_PATTERN = re.compile(r'(https?://\S+|www\.\S+)', re.IGNORECASE)
QUOTE_LINE_PATTERN = re.compile(r'^\s*>.*$', re.MULTILINE)
SIGNATURE_MARKERS = re.compile(
    r'(?im)^\s*(regards|thanks|thank you|best regards|warm regards|'
    r'sincerely|best|cheers)[, ]*\s*$'
)
NON_PRINTABLE = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]')
MULTI_NEWLINE = re.compile(r'\n{3,}')
MULTI_SPACE = re.compile(r'[ \t]{2,}')


# --- Individual steps (each testable/usable standalone) ------------------

def html_to_text(html: str) -> str:
    """Strip HTML tags, return visible text with line breaks preserved
    roughly where block-level tags were.

    Uses lxml as the parser backend: it's a C extension (faster than the
    stdlib html.parser) and more tolerant of malformed/broken markup,
    which matters once real-world HTML (messy newsletter templates,
    unclosed tags, etc.) enters the pipeline. Requires `pip install lxml`.
    """
    if not html:
        return ""
    soup = BeautifulSoup(html, "lxml")
    return soup.get_text(separator="\n")


def get_raw_body(body_text: str, body_html: str):
    """Prefer plain text; fall back to HTML->text if plain is empty."""
    text = body_text or ""
    used_html_fallback = False
    if not text.strip() and body_html and body_html.strip():
        text = html_to_text(body_html)
        used_html_fallback = True
    return text, used_html_fallback


def strip_quoted_lines(text: str) -> str:
    """Remove '> quoted original text' lines entirely from Re:/Fwd: threads.
    Used for clean_text (classifier training) -- quoted content duplicates
    an earlier email and would bias category learning."""
    return QUOTE_LINE_PATTERN.sub("", text)


def unquote_lines(text: str) -> str:
    """Remove ALL leading '>' markers from quoted lines (one or more,
    possibly stacked as '> > text' from a reply-to-a-reply), KEEPING the
    underlying text. Used for search_text (embeddings/search/clustering).

    Nested threads (a reply to an already-quoted reply) stack multiple
    '>' per line. A regex that strips only one leading '>' leaves a
    residual '>' and duplicated boilerplate behind on depth-2+ threads --
    e.g. 'Confirmed, thank you. Confirmed, thank you.  > Thanks for the
    update...' with a literal '>' still visible. The (?:\\s*>)+ group
    strips however many are stacked, not just one.
    """
    return re.sub(r'^(?:\s*>)+\s?', '', text, flags=re.MULTILINE)


def strip_signature(text: str) -> str:
    """Cut everything from the first sign-off phrase onward."""
    match = SIGNATURE_MARKERS.search(text)
    return text[:match.start()] if match else text


def extract_and_strip_urls(text: str):
    """Remove URLs from text, return (text_without_urls, list_of_urls)."""
    urls = URL_PATTERN.findall(text)
    return URL_PATTERN.sub(" ", text), urls


def normalize_whitespace(text: str) -> str:
    text = NON_PRINTABLE.sub("", text)
    text = MULTI_NEWLINE.sub("\n\n", text)
    text = MULTI_SPACE.sub(" ", text)
    return text.strip()


# --- Full pipeline for a single email -------------------------------------

def clean_email_body(body_text: str, body_html: str, lowercase: bool = False) -> dict:
    """Run the full cleaning pipeline on one email's body content.
    Returns a dict of derived fields -- merge these into your DataFrame,
    don't overwrite the original body_text/body_html columns.

    Produces TWO text variants:
    - clean_text: quotes fully stripped -- use for classifier training,
      where quoted duplicate content would bias category learning.
    - search_text: quote markers removed but content KEPT -- use for
      embeddings/semantic search/clustering, where a thread reply's real
      informative content lives in the quoted original message.
    """
    text, used_html_fallback = get_raw_body(body_text, body_html)

    # search_text variant: keep quoted content, just drop '>' markers
    search_text = unquote_lines(text)
    search_text = strip_signature(search_text)
    search_text, _ = extract_and_strip_urls(search_text)
    search_text = normalize_whitespace(search_text)
    if lowercase:
        search_text = search_text.lower()

    # clean_text variant: quotes fully stripped (classifier-safe)
    text = strip_quoted_lines(text)
    text = strip_signature(text)
    text, urls = extract_and_strip_urls(text)
    text = normalize_whitespace(text)

    if lowercase:
        text = text.lower()

    return {
        "clean_text": text,
        "search_text": search_text,
        "used_html_fallback": used_html_fallback,
        "has_url": len(urls) > 0,
        "url_count": len(urls),
        "char_count": len(text),
        "word_count": len(text.split()) if text else 0,
        "is_empty_after_cleaning": len(text.strip()) == 0,
    }


def clean_dataframe(df: pd.DataFrame, lowercase: bool = False) -> pd.DataFrame:
    """Apply clean_email_body to every row and merge results as new columns.
    Original body_text/body_html columns are preserved untouched."""
    results = df.apply(
        lambda row: clean_email_body(row["body_text"], row["body_html"], lowercase=lowercase),
        axis=1,
    )
    clean_df = pd.DataFrame(list(results))
    out = pd.concat([df.reset_index(drop=True), clean_df], axis=1)

    empty_count = out["is_empty_after_cleaning"].sum()
    if empty_count:
        import logging
        logging.warning(
            f"{empty_count} emails have empty clean_text after cleaning "
            f"({empty_count / len(out):.1%}) -- inspect these, the pipeline "
            f"may be over-stripping short/legitimate emails."
        )

    return out


if __name__ == "__main__":
    import sys
    from loader import load_emails

    emails_dir = sys.argv[1] if len(sys.argv) > 1 else "generated_emails"
    labels_csv = sys.argv[2] if len(sys.argv) > 2 else "labels.csv"

    df = load_emails(emails_dir, labels_csv)
    df = clean_dataframe(df)

    print(f"\nShape after cleaning: {df.shape}")
    print(f"\nEmpty-after-cleaning count: {df['is_empty_after_cleaning'].sum()}")
    print(f"HTML fallback used count: {df['used_html_fallback'].sum()}")
    print(f"Has URL count: {df['has_url'].sum()}")
    print(f"\nchar_count / word_count stats:\n{df[['char_count', 'word_count']].describe()}")

    print("\n--- before/after example: a threaded reply ---")
    thread_row = df[df["is_thread_reply"] == True].iloc[0]
    print("RAW body_text:\n", thread_row["body_text"])
    print("\nCLEAN text:\n", thread_row["clean_text"])

    print("\n--- before/after example: a signature-bearing email ---")
    row = df.iloc[2]
    print("RAW body_text:\n", row["body_text"])
    print("\nCLEAN text:\n", row["clean_text"])