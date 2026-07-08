"""
Heuristic labeler for spam_score and priority.

These are deliberately computed from the *content* (keyword hits in
subject/body) plus a category base-rate, with random jitter -- not a
straight category lookup. That mirrors how a first-pass rule-based
labeler would work in the real world, and means a model trained on
these labels has to learn actual textual patterns (urgency language,
deadline phrasing, prize/verify-account language, etc.) rather than
just memorizing "category X = score Y".

Both values are computed at generation time and written ONLY to
labels.csv -- never into the .eml content itself (same leak-prevention
rule as the category label).
"""

import random

SPAM_KEYWORDS = {
    "free": 8, "win": 10, "winner": 12, "won": 10, "congratulations": 10,
    "claim": 8, "urgent": 6, "act now": 12, "click here": 10,
    "guaranteed": 8, "limited time": 6, "prize": 10,
    "verify your account": 14, "verify your details": 10,
    "double your money": 16, "no experience needed": 8, "airdrop": 12,
    "crypto": 6, "gift card": 6, "lottery": 14, "suspended": 8,
    "final notice": 8, "one-time opportunity": 8, "expires": 4, "$": 4,
}

CATEGORY_SPAM_BASE = {
    "Spam": 55, "Shopping": 10, "Finance": 6, "Social": 3, "Job": 4,
    "College": 2, "Travel": 4, "Government": 2,
}

PRIORITY_KEYWORDS = {
    "urgent": 22, "immediately": 18, "asap": 22, "deadline": 16,
    "due": 10, "interview": 28, "offer letter": 32, "exam": 18,
    "payment due": 18, "overdue": 20, "final notice": 20,
    "action required": 18, "important": 12, "summon": 28, "court": 22,
    "suspended": 15, "expire": 10, "renew": 8, "confirm": 6,
    "cancelled": 16, "rescheduled": 14, "delayed": 12, "visa": 16,
    "advisory": 14, "refund": 8, "boarding": 8, "check-in": 8,
    "reminder": 6, "offer": 10,
}

CATEGORY_PRIORITY_BASE = {
    "Government": 40, "Finance": 35, "College": 32, "Job": 32,
    "Travel": 26, "Shopping": 14, "Social": 6, "Spam": 2,
}


def _score(text, keyword_weights, category_base, jitter=6):
    text = text.lower()
    score = category_base
    for kw, weight in keyword_weights.items():
        if kw in text:
            score += weight
    score += random.uniform(-jitter, jitter)
    return max(0, min(100, round(score, 1)))


def compute_labels(category, subject, body):
    """Returns (spam_score: float 0-100, priority_score: float 0-100, priority: 'High'|'Medium'|'Low')."""
    text = f"{subject} {body}"

    spam_score = _score(text, SPAM_KEYWORDS, CATEGORY_SPAM_BASE.get(category, 5), jitter=6)
    priority_score = _score(text, PRIORITY_KEYWORDS, CATEGORY_PRIORITY_BASE.get(category, 10), jitter=12)

    # Very spammy mail is realistically deprioritized regardless of urgency language
    # (a "URGENT ACT NOW" spam email shouldn't count as genuinely high priority).
    if spam_score >= 60:
        priority_score = min(priority_score, 25)

    if priority_score >= 65:
        priority = "High"
    elif priority_score >= 35:
        priority = "Medium"
    else:
        priority = "Low"

    return spam_score, priority_score, priority
