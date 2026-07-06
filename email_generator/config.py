"""
Global configuration for the synthetic email generator.
Tweak these values and re-run generator.py to reshape the dataset.
"""

# Total number of emails to generate across all categories.
TOTAL_EMAILS = 10000

# Relative weights per category (need not sum to 1 -- they're normalized).
# Reflects a realistic inbox: lots of shopping/spam/social noise,
# a moderate amount of finance/college/job mail, less travel/government.
CATEGORY_WEIGHTS = {
    "Shopping": 22,
    "Spam": 18,
    "Social": 15,
    "Finance": 12,
    "College": 10,
    "Job": 10,
    "Travel": 8,
    "Government": 5,
}

# Where generated .eml files (and labels/metadata) are written.
OUTPUT_DIR = "generated_emails"

# Probability that a given email carries a small dummy attachment.
ATTACHMENT_PROBABILITY = 0.12

# Probability that a given email is a Re:/Fwd: reply to an earlier
# generated email (creates simple threads with In-Reply-To / References
# headers pointing at a prior Message-ID in the same category).
THREAD_PROBABILITY = 0.15

# Date range emails are spread across (both inclusive), format YYYY-MM-DD.
DATE_RANGE_START = "2026-01-01"
DATE_RANGE_END = "2026-07-02"

# Set to an int for reproducible output, or None for a fresh random batch
# every run.
RANDOM_SEED = 42
