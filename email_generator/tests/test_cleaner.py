"""
Unit tests for cleaner.py.

Run with:
    pytest tests/test_cleaner.py -v

Each function gets tested in isolation with known input -> expected
output, plus edge cases (empty strings, whitespace-only, unicode,
multiple matches, nested/malformed HTML). This is what actually proves
the cleaner works -- eyeballing a few sample rows (like we did earlier)
only proves it works on the rows you happened to look at.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cleaner import (
    html_to_text,
    get_raw_body,
    strip_quoted_lines,
    strip_signature,
    extract_and_strip_urls,
    normalize_whitespace,
    clean_email_body,
)


# --- html_to_text ----------------------------------------------------------

def test_html_to_text_strips_tags():
    html = "<html><body><h3>Hello</h3><p>World</p></body></html>"
    result = html_to_text(html)
    assert "<" not in result
    assert "Hello" in result
    assert "World" in result


def test_html_to_text_empty_input():
    assert html_to_text("") == ""
    assert html_to_text(None) == ""


def test_html_to_text_malformed_html_does_not_crash():
    # Unclosed tags -- BeautifulSoup should still degrade gracefully
    html = "<p>Unclosed paragraph <b>bold text"
    result = html_to_text(html)
    assert "Unclosed paragraph" in result
    assert "bold text" in result


# --- get_raw_body (plain text vs HTML fallback) -----------------------------

def test_get_raw_body_prefers_plain_text():
    text, used_fallback = get_raw_body("Plain text body", "<p>HTML body</p>")
    assert text == "Plain text body"
    assert used_fallback is False


def test_get_raw_body_falls_back_to_html_when_plain_is_empty():
    text, used_fallback = get_raw_body("", "<p>Only HTML here</p>")
    assert "Only HTML here" in text
    assert used_fallback is True


def test_get_raw_body_falls_back_when_plain_is_whitespace_only():
    text, used_fallback = get_raw_body("   \n  ", "<p>Real content</p>")
    assert "Real content" in text
    assert used_fallback is True


def test_get_raw_body_both_empty():
    text, used_fallback = get_raw_body("", "")
    assert text == ""
    assert used_fallback is False


# --- strip_quoted_lines ------------------------------------------------------

def test_strip_quoted_lines_removes_quote_marker():
    text = "Thanks for the update.\n\n> Original message here.\n> Second quoted line.\n\nSee you then."
    result = strip_quoted_lines(text)
    assert "Original message" not in result
    assert "Second quoted line" not in result
    assert "Thanks for the update." in result
    assert "See you then." in result


def test_strip_quoted_lines_no_quotes_present():
    text = "No quotes here at all."
    assert strip_quoted_lines(text) == text


def test_strip_quoted_lines_indented_quote_marker():
    # Some clients indent the '>' with leading whitespace
    text = "Reply text.\n   > indented quote line"
    result = strip_quoted_lines(text)
    assert "indented quote line" not in result


# --- strip_signature ----------------------------------------------------------

def test_strip_signature_bare_regards_line():
    text = "Your exam is on Friday.\n\nRegards,\nExamination Cell"
    result = strip_signature(text)
    assert "Regards" not in result
    assert "Examination Cell" not in result
    assert "Your exam is on Friday." in result


def test_strip_signature_bare_thanks_line():
    text = "Please review the attached document.\n\nThanks,\nHR Team"
    result = strip_signature(text)
    assert "HR Team" not in result
    assert "Please review the attached document." in result


def test_strip_signature_does_not_touch_sentence_containing_marker_word():
    # "Thanks" mid-sentence should NOT trigger stripping -- only a bare
    # marker line should. This is the known limitation documented in
    # cleaner.py: full-sentence closers are left alone on purpose.
    text = "Thanks for applying, we will review your application shortly."
    result = strip_signature(text)
    assert result == text


def test_strip_signature_no_marker_present():
    text = "This email has no sign-off at all."
    assert strip_signature(text) == text


# --- extract_and_strip_urls ----------------------------------------------------

def test_extract_and_strip_urls_finds_and_removes_http_url():
    text = "Click here: https://example.com/claim?id=123 to proceed."
    result, urls = extract_and_strip_urls(text)
    assert urls == ["https://example.com/claim?id=123"]
    assert "https://" not in result


def test_extract_and_strip_urls_finds_www_url_without_scheme():
    text = "Visit www.example.com for more info."
    result, urls = extract_and_strip_urls(text)
    assert urls == ["www.example.com"]
    assert "www.example.com" not in result


def test_extract_and_strip_urls_multiple_urls():
    text = "First https://a.com then https://b.com/path also www.c.com"
    result, urls = extract_and_strip_urls(text)
    assert len(urls) == 3


def test_extract_and_strip_urls_no_urls_present():
    text = "No links in this message."
    result, urls = extract_and_strip_urls(text)
    assert urls == []
    assert result == text


# --- normalize_whitespace -----------------------------------------------------

def test_normalize_whitespace_collapses_multiple_blank_lines():
    text = "Line one.\n\n\n\n\nLine two."
    result = normalize_whitespace(text)
    assert "\n\n\n" not in result
    assert "Line one." in result and "Line two." in result


def test_normalize_whitespace_collapses_multiple_spaces():
    text = "Too      many      spaces."
    result = normalize_whitespace(text)
    assert "  " not in result


def test_normalize_whitespace_strips_control_characters():
    text = "Text with\x00null and\x0bvertical tab chars."
    result = normalize_whitespace(text)
    assert "\x00" not in result
    assert "\x0b" not in result


def test_normalize_whitespace_trims_leading_trailing_whitespace():
    text = "   \n  Padded text.  \n   "
    result = normalize_whitespace(text)
    assert result == "Padded text."


def test_normalize_whitespace_preserves_unicode_content():
    # Cleaner should NOT mangle non-ASCII text -- important once
    # multi-language emails are added later.
    text = "Namaste, यह एक ईमेल है।"
    result = normalize_whitespace(text)
    assert "यह एक ईमेल है" in result


# --- clean_email_body (full pipeline, integration-style) ------------------------

def test_clean_email_body_full_pipeline_realistic_email():
    body_text = (
        "Thanks for the update, noted.\n\n"
        "> Your assignment is due Friday.\n"
        "> Please submit via the portal.\n\n"
        "Visit https://portal.example.com/submit to upload.\n\n"
        "Regards,\nAcademic Office"
    )
    result = clean_email_body(body_text, "")

    assert "Your assignment is due Friday" not in result["clean_text"]  # quote stripped
    assert "Academic Office" not in result["clean_text"]                 # signature stripped
    assert "https://" not in result["clean_text"]                        # URL stripped
    assert result["has_url"] is True
    assert result["url_count"] == 1
    assert result["is_empty_after_cleaning"] is False
    assert result["used_html_fallback"] is False


def test_clean_email_body_html_only_email():
    result = clean_email_body("", "<html><body><p>Order shipped today.</p></body></html>")
    assert "Order shipped today." in result["clean_text"]
    assert result["used_html_fallback"] is True


def test_clean_email_body_completely_empty_email():
    result = clean_email_body("", "")
    assert result["clean_text"] == ""
    assert result["is_empty_after_cleaning"] is True
    assert result["word_count"] == 0


def test_clean_email_body_lowercase_flag():
    result_default = clean_email_body("IMPORTANT Notice", "", lowercase=False)
    result_lower = clean_email_body("IMPORTANT Notice", "", lowercase=True)
    assert result_default["clean_text"] == "IMPORTANT Notice"
    assert result_lower["clean_text"] == "important notice"


def test_clean_email_body_whitespace_only_becomes_empty_flagged():
    result = clean_email_body("   \n\n   ", "")
    assert result["is_empty_after_cleaning"] is True
