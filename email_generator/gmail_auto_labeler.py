"""
gmail_auto_labeler.py — Heuristic category labeler for real Gmail emails.

Since real Gmail emails have no ground-truth category labels, this module
assigns them using two layers of signal:

  1. Sender domain patterns (high precision: amazon.com → Shopping)
  2. Subject + body keyword matching (broader coverage)

The output is a 'gmail_labels.csv' with columns (filename, category) that
retrain_on_real_mail.py reads to include Gmail emails in the hybrid
training corpus.

Design notes:
- Rules deliberately use the SAME category names as the synthetic dataset
  (Job, Finance, Travel, Shopping, College, Social, Government, Spam) so
  the combined training corpus has a consistent label space.
- When multiple categories score equally, priority order is applied
  (Spam > Finance > Government > Job > College > Travel > Shopping > Social)
  to prefer the more "specific" category over the generic fallback.
- Confidence is returned so callers can filter out low-confidence labels
  (those likely to be noisy training examples).

Usage:
    from gmail_auto_labeler import label_gmail_emails
    labels_df = label_gmail_emails("gmail_emails")
    labels_df.to_csv("gmail_labels.csv", index=False)
"""

import re
import logging
from pathlib import Path

import pandas as pd

from loader import load_emails
from cleaner import clean_dataframe

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain → category map (high-precision rules applied first)
# ---------------------------------------------------------------------------

DOMAIN_CATEGORY_MAP = {
    # Job / Career
    "linkedin.com": "Job",
    "naukri.com": "Job",
    "indeed.com": "Job",
    "glassdoor.com": "Job",
    "monster.com": "Job",
    "shine.com": "Job",
    "internshala.com": "Job",
    "ziprecruiter.com": "Job",
    "lever.co": "Job",
    "greenhouse.io": "Job",
    "workday.com": "Job",
    "smartrecruiters.com": "Job",

    # Shopping / E-commerce
    "amazon.com": "Shopping",
    "amazon.in": "Shopping",
    "flipkart.com": "Shopping",
    "myntra.com": "Shopping",
    "ajio.com": "Shopping",
    "snapdeal.com": "Shopping",
    "meesho.com": "Shopping",
    "ebay.com": "Shopping",
    "etsy.com": "Shopping",
    "shopify.com": "Shopping",
    "nykaa.com": "Shopping",
    "bigbasket.com": "Shopping",
    "blinkit.com": "Shopping",
    "swiggy.com": "Shopping",
    "zomato.com": "Shopping",

    # Travel
    "irctc.co.in": "Travel",
    "makemytrip.com": "Travel",
    "goibibo.com": "Travel",
    "booking.com": "Travel",
    "expedia.com": "Travel",
    "airbnb.com": "Travel",
    "hotels.com": "Travel",
    "tripadvisor.com": "Travel",
    "cleartrip.com": "Travel",
    "indigo.in": "Travel",
    "airindia.in": "Travel",
    "spicejet.com": "Travel",

    # Finance / Banking
    "hdfcbank.com": "Finance",
    "icicibank.com": "Finance",
    "sbicard.com": "Finance",
    "axisbank.com": "Finance",
    "kotak.com": "Finance",
    "paytm.com": "Finance",
    "phonepe.com": "Finance",
    "gpay.app": "Finance",
    "razorpay.com": "Finance",
    "billdesk.com": "Finance",
    "cred.club": "Finance",

    # College / Education
    "coursera.org": "College",
    "udemy.com": "College",
    "edx.org": "College",
    "khanacademy.org": "College",
    "nptel.ac.in": "College",
    "swayam.gov.in": "College",
    "duolingo.com": "College",
    "byju.com": "College",

    # Government
    "uidai.gov.in": "Government",
    "incometax.gov.in": "Government",
    "epfindia.gov.in": "Government",
    "passport.gov.in": "Government",
    "nsdl.co.in": "Government",
    "digilocker.gov.in": "Government",

    # Social
    "facebook.com": "Social",
    "instagram.com": "Social",
    "twitter.com": "Social",
    "x.com": "Social",
    "youtube.com": "Social",
    "reddit.com": "Social",
    "discord.com": "Social",
    "quora.com": "Social",
    "medium.com": "Social",
    "substack.com": "Social",
    "meetup.com": "Social",
}

# Partial domain suffix matches (applied after exact match fails)
DOMAIN_SUFFIX_MAP = [
    (".gov.in", "Government"),
    (".gov", "Government"),
    (".edu", "College"),
    (".ac.in", "College"),
    (".ac.uk", "College"),
]


# ---------------------------------------------------------------------------
# Keyword → category weights
# ---------------------------------------------------------------------------

CATEGORY_KEYWORDS = {
    "Job": [
        "interview", "hiring", "recruiter", "job offer", "job application",
        "resume", "cv", "position", "vacancy", "shortlisted", "candidature",
        "offer letter", "internship", "placement", "screening", "background check",
        "onboarding", "joining", "salary", "ctc", "role", "designation",
        "your application", "we reviewed your", "next steps", "talent",
    ],
    "Finance": [
        "invoice", "payment", "transaction", "statement", "bank", "credit card",
        "debit", "account", "balance", "bill", "receipt", "refund", "emi",
        "loan", "investment", "portfolio", "mutual fund", "subscription",
        "amount", "charged", "due date", "overdue", "tax", "gst", "pan",
    ],
    "Travel": [
        "booking confirmed", "itinerary", "flight", "hotel", "reservation",
        "check-in", "boarding pass", "pnr", "train ticket", "bus ticket",
        "departure", "arrival", "destination", "trip", "vacation",
        "holiday", "travel", "passport", "visa", "airport",
    ],
    "Shopping": [
        "order confirmed", "shipped", "out for delivery", "delivered",
        "your order", "tracking", "return", "exchange", "wishlist",
        "cart", "discount", "sale", "offer", "coupon", "promo code",
        "product", "item", "package", "dispatch",
    ],
    "College": [
        "assignment", "exam", "lecture", "course", "semester", "faculty",
        "professor", "university", "college", "campus", "grade", "marks",
        "result", "admit card", "hall ticket", "syllabus", "attendance",
        "timetable", "scholarship", "fee", "hostel", "library",
    ],
    "Government": [
        "aadhaar", "pan card", "passport", "voter id", "ration card",
        "income tax", "gst", "provident fund", "esic", "epfo",
        "government notice", "official notice", "ministry", "department",
        "police", "court", "summon", "legal notice", "tender", "e-tender",
    ],
    "Social": [
        "liked your", "commented on", "followed you", "sent you a friend",
        "invitation", "birthday", "anniversary", "event", "group",
        "newsletter", "digest", "weekly update", "unsubscribe",
        "mentioned you", "tagged you", "shared a post",
    ],
    "Spam": [
        "win", "winner", "won", "congratulations", "claim your",
        "click here", "free", "guaranteed", "limited time", "act now",
        "prize", "lottery", "airdrop", "crypto", "double your money",
        "verify your account", "suspended account", "urgent action",
        "no experience needed", "work from home earn", "gift card",
    ],
}

# Priority order for tiebreaking (higher index = higher priority)
CATEGORY_PRIORITY = ["Social", "Shopping", "Travel", "College", "Job", "Government", "Finance", "Spam"]


def _extract_domain(from_address: str) -> str:
    """Extract domain from 'user@domain.com' → 'domain.com'."""
    if not from_address or "@" not in from_address:
        return ""
    return from_address.split("@")[-1].strip().lower()


def _classify_by_domain(domain: str) -> str | None:
    """Return category if domain matches a known sender, else None."""
    if not domain:
        return None

    # Exact match
    if domain in DOMAIN_CATEGORY_MAP:
        return DOMAIN_CATEGORY_MAP[domain]

    # Suffix match (e.g. *.gov.in → Government)
    for suffix, category in DOMAIN_SUFFIX_MAP:
        if domain.endswith(suffix):
            return category

    return None


def _score_keywords(text: str) -> dict:
    """Score the text against each category's keyword list."""
    text_lower = text.lower()
    scores = {cat: 0 for cat in CATEGORY_KEYWORDS}

    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                # Multi-word keywords are more specific → higher weight
                weight = 2 if " " in kw else 1
                scores[category] += weight

    return scores


def classify_email(from_address: str, subject: str, body: str) -> tuple[str, float]:
    """
    Returns (predicted_category, confidence_0_to_1).

    confidence = 1.0  → domain match (high precision)
    confidence = 0.5–0.9 → keyword-only match (proportional to score gap)
    confidence = 0.3  → fallback Social (generic newsletter/update)
    """
    domain = _extract_domain(from_address)

    # Layer 1: domain-based classification (high confidence)
    domain_category = _classify_by_domain(domain)
    if domain_category:
        return domain_category, 1.0

    # Layer 2: keyword scoring on subject + body
    combined_text = f"{subject} {body}"
    scores = _score_keywords(combined_text)
    max_score = max(scores.values())

    if max_score == 0:
        return "Social", 0.3  # generic newsletter fallback

    # Get all categories at the maximum score (for tiebreaking)
    top_cats = [cat for cat, sc in scores.items() if sc == max_score]

    if len(top_cats) == 1:
        winner = top_cats[0]
    else:
        # Tiebreak by priority order
        winner = max(top_cats, key=lambda c: CATEGORY_PRIORITY.index(c))

    # Confidence: ratio of winning score to total (rough measure)
    total = sum(scores.values())
    confidence = round(min(0.9, max_score / max(total, 1) * 3), 2)

    return winner, confidence


def label_gmail_emails(
    gmail_dir: str = "gmail_emails",
    output_csv: str = "gmail_labels.csv",
    min_confidence: float = 0.0,
) -> pd.DataFrame:
    """
    Auto-label all Gmail emails in gmail_dir and save to output_csv.

    Parameters
    ----------
    gmail_dir : path containing .eml files
    output_csv : where to save (filename, category, confidence) CSV
    min_confidence : drop labels below this threshold (0.0 = keep all)

    Returns
    -------
    DataFrame with columns: filename, category, confidence
    """
    logger.info(f"Loading Gmail emails from {gmail_dir}...")
    df = load_emails(gmail_dir)

    if len(df) == 0:
        logger.warning("No Gmail emails found — nothing to label.")
        return pd.DataFrame(columns=["filename", "category", "confidence"])

    df = clean_dataframe(df)

    results = []
    for _, row in df.iterrows():
        category, confidence = classify_email(
            from_address=row.get("from_address", ""),
            subject=row.get("subject", ""),
            body=row.get("clean_text", ""),
        )
        results.append({
            "filename": row["filename"],
            "category": category,
            "confidence": confidence,
        })

    labels_df = pd.DataFrame(results)

    # Filter by confidence
    if min_confidence > 0:
        before = len(labels_df)
        labels_df = labels_df[labels_df["confidence"] >= min_confidence].copy()
        logger.info(f"Dropped {before - len(labels_df)} low-confidence labels (< {min_confidence})")

    logger.info(f"Labeled {len(labels_df)} emails:")
    logger.info(f"\n{labels_df['category'].value_counts().to_string()}")
    logger.info(f"Average confidence: {labels_df['confidence'].mean():.2f}")

    if output_csv:
        labels_df.to_csv(output_csv, index=False)
        logger.info(f"Saved labels to {output_csv}")

    return labels_df


if __name__ == "__main__":
    import sys
    gmail_dir = sys.argv[1] if len(sys.argv) > 1 else "gmail_emails"
    output_csv = sys.argv[2] if len(sys.argv) > 2 else "gmail_labels.csv"
    df = label_gmail_emails(gmail_dir, output_csv, min_confidence=0.0)
    print(f"\n{'='*50}")
    print("LABEL DISTRIBUTION:")
    print(df["category"].value_counts().to_string())
    print(f"\nHigh confidence (>= 0.9): {(df['confidence'] >= 0.9).sum()} emails")
    print(f"Medium confidence (0.5–0.9): {((df['confidence'] >= 0.5) & (df['confidence'] < 0.9)).sum()} emails")
    print(f"Low confidence (< 0.5): {(df['confidence'] < 0.5).sum()} emails")
