import os
import sys

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler

from loader import load_emails
from cleaner import clean_dataframe
from embeddings import load_embeddings, MODEL_NAME

# 1. Reduced Category Trust Weights (Minimal nudge, content dominates)
CATEGORY_TRUST_WEIGHT = {
    "Spam": -5,
    "Government": 1,
    "Finance": 1,
    "College": 1,
    "Job": 1,
}

# 2. Four Semantic Axes Anchors (25 each for a highly representative centroid)
URGENCY_ANCHORS = [
    "This requires your immediate attention today.",
    "Deadline is in two hours.",
    "Your account will be suspended tonight.",
    "Action required before midnight.",
    "This offer expires tomorrow.",
    "Time-sensitive: please respond now.",
    "Urgent update regarding your flight.",
    "Final reminder: payment is due today.",
    "Please reply within the next hour.",
    "Immediate action needed to avoid service interruption.",
    "Your session will expire in 5 minutes.",
    "Last chance to register for the event today.",
    "Urgent: verify your identity to prevent lockout.",
    "Response required by end of business today.",
    "Your trial ends in 24 hours.",
    "Act now to secure your spot.",
    "Time is running out to claim your refund.",
    "Urgent notice: system maintenance in 1 hour.",
    "Please confirm your attendance by 5 PM today.",
    "Final call for submissions: deadline is tonight.",
    "Your reservation is pending and will expire shortly.",
    "Immediate review required for this critical issue.",
    "Hurry, this discount is only valid for the next 3 hours.",
    "Urgent: your package delivery is scheduled for today.",
    "Please finalize your decision by tomorrow morning.",
]

ACTIONABILITY_ANCHORS = [
    "Please click the link to verify your account.",
    "Sign and return the attached document.",
    "Reply to this email to confirm your attendance.",
    "Update your billing information to continue service.",
    "Reset your password using the link below.",
    "Approve this transaction request in your app.",
    "Submit your application through the portal.",
    "Confirm your shipping address before we dispatch.",
    "Complete your profile setup to unlock features.",
    "Authorize this login attempt from a new device.",
    "Please fill out the attached form and return it.",
    "Click here to download your invoice.",
    "Review and accept the updated terms of service.",
    "Schedule your appointment using the calendar link.",
    "Upload your required documents to the secure folder.",
    "Opt-in to receive important notifications.",
    "Please provide your feedback by taking this short survey.",
    "Activate your new credit card by calling this number.",
    "Claim your reward by visiting the redemption page.",
    "Please sign the contract electronically via DocuSign.",
    "Update your communication preferences in your settings.",
    "Register for the webinar using the link provided.",
    "Please acknowledge receipt of this message.",
    "Transfer the funds to the account details below.",
    "Complete the mandatory training module by Friday.",
]

CONSEQUENCE_ANCHORS = [
    "Failure to respond will result in a financial penalty.",
    "You will lose this opportunity if you do not act.",
    "Your account will be permanently locked.",
    "A late fee will be applied to your outstanding balance.",
    "Miss this deadline and your registration will be canceled.",
    "Unpaid bills will be sent to a collections agency.",
    "Your subscription will expire and you will lose access to your data.",
    "Ignoring this may result in unauthorized access to your account.",
    "You will forfeit your security deposit.",
    "Non-compliance will result in disciplinary action.",
    "Your flight will be canceled if you do not check in.",
    "Failure to verify will result in restricted account access.",
    "You will miss out on this exclusive discount.",
    "Your warranty will be voided if repairs are not authorized.",
    "Ignoring this security alert may compromise your personal data.",
    "Your application will be rejected if documents are missing.",
    "You will be removed from the waiting list.",
    "Failure to update your records may result in missed communications.",
    "Your access to the platform will be suspended.",
    "You will lose your accumulated points.",
    "Ignoring this notice may result in legal consequences.",
    "Your service will be disconnected.",
    "You will be charged the full amount if not canceled.",
    "Failure to attend will result in a no-show fee.",
    "Your eligibility for the program will be revoked.",
]

IMPORTANCE_ANCHORS = [
    "Official notice regarding your legal status.",
    "Critical update about your medical records.",
    "Important information about your tax return.",
    "Details regarding your university admission decision.",
    "Security alert for your primary bank account.",
    "Changes to your employment contract and benefits.",
    "Vital information about your upcoming international travel.",
    "Notice of property tax assessment.",
    "Important policy updates affecting your insurance coverage.",
    "Critical system maintenance affecting your workflow.",
    "Official notification of your court date.",
    "Important changes to your retirement plan.",
    "Critical security vulnerability requires your attention.",
    "Official transcript request for your academic records.",
    "Important update regarding your mortgage application.",
    "Notice of change in your utility service provider.",
    "Critical information about your child's school enrollment.",
    "Official recall notice for your vehicle.",
    "Important changes to your healthcare coverage.",
    "Critical update regarding your data privacy rights.",
    "Official notification of your visa status.",
    "Important details about your upcoming surgery.",
    "Critical alert regarding suspicious activity on your account.",
    "Official notice of lease renewal terms.",
    "Important information regarding your final exam schedule.",
]

# Cache path for the 4 centroids
CENTROID_CACHE_PATH = "semantic_axes_centroids_cache.npy"

# Weights for the 4 axes (Sum = 1.0)
AXIS_WEIGHTS = {
    "urgency": 0.40,
    "actionability": 0.30,
    "consequence": 0.20,
    "importance": 0.10,
}


def get_or_compute_centroids(model: SentenceTransformer, cache_path: str = CENTROID_CACHE_PATH):
    """Loads axis centroids from cache or computes and saves them."""
    if os.path.exists(cache_path):
        print(f"Loading cached semantic axes centroids from {cache_path}...")
        data = np.load(cache_path, allow_pickle=True).item()
        return data

    print("Computing and caching semantic axes centroids...")
    
    def make_centroid(anchors):
        vecs = model.encode(anchors, normalize_embeddings=True)
        centroid = vecs.mean(axis=0)
        return centroid / np.linalg.norm(centroid)

    centroids = {
        "urgency": make_centroid(URGENCY_ANCHORS),
        "actionability": make_centroid(ACTIONABILITY_ANCHORS),
        "consequence": make_centroid(CONSEQUENCE_ANCHORS),
        "importance": make_centroid(IMPORTANCE_ANCHORS),
    }

    np.save(cache_path, centroids)
    return centroids


def compute_semantic_scores(embeddings: np.ndarray, centroids: dict) -> np.ndarray:
    """Computes the weighted semantic score for each email across the 4 axes."""
    # Calculate similarity for each axis
    scores = {}
    for axis, centroid in centroids.items():
        scores[axis] = embeddings @ centroid

    # Combine using the defined weights
    final_score = np.zeros(embeddings.shape[0])
    for axis, weight in AXIS_WEIGHTS.items():
        final_score += weight * scores[axis]

    return final_score


def compute_ranking_score(df: pd.DataFrame, raw_semantic_score: np.ndarray) -> pd.DataFrame:
    df = df.copy()
    
    # Normalize semantic score using Z-score to avoid assuming a fixed range
    scaler = StandardScaler()
    semantic_normalized = scaler.fit_transform(raw_semantic_score.reshape(-1, 1)).flatten()
    
    # Scale to roughly match the metadata range (e.g., -30 to 30)
    # so it doesn't completely drown out the metadata heuristics.
    urgency_component = semantic_normalized * 15

    attachment_bonus = df["has_attachment"].astype(int) * 2
    thread_penalty = df["is_thread_reply"].astype(int) * -2
    trust_component = df["category"].map(CATEGORY_TRUST_WEIGHT).fillna(0)
    spam_penalty = np.clip(df["spam_score"] / 2, 0, 50) * -1

    raw_score = (
        urgency_component 
        + attachment_bonus 
        + thread_penalty 
        + trust_component 
        + spam_penalty
    )

    # Safe min-max scaling to 0-100
    score_min = raw_score.min()
    score_max = raw_score.max()
    score_range = score_max - score_min

    if score_range > 0:
        df["semantic_priority_score"] = ((raw_score - score_min) / score_range * 100).round(1)
    else:
        df["semantic_priority_score"] = 50.0

    df["semantic_priority_rank"] = df["semantic_priority_score"].rank(ascending=False, method="min").astype(int)

    return df


def main(emails_dir, labels_csv):
    df = load_emails(emails_dir, labels_csv)
    df = clean_dataframe(df)

    embeddings, ids = load_embeddings()
    id_to_row = {f: i for i, f in enumerate(ids)}
    
    valid_mask = df["filename"].isin(id_to_row)
    if not valid_mask.any():
        print("Error: No matching filenames found between dataframe and embeddings.")
        return df
        
    df = df[valid_mask].reset_index(drop=True)
    embeddings = embeddings[[id_to_row[f] for f in df["filename"]]]

    print("Loading model for anchor-phrase encoding...")
    model = SentenceTransformer(MODEL_NAME)

    centroids = get_or_compute_centroids(model)
    raw_semantic_score = compute_semantic_scores(embeddings, centroids)
    
    df = compute_ranking_score(df, raw_semantic_score)

    print("\nTop 15 highest-ranked emails (semantic_priority_score):")
    top = df.nsmallest(15, "semantic_priority_rank")[
        ["subject", "category", "priority", "spam_score", "semantic_priority_score"]
    ]
    print(top.to_string(index=False))

    print("\nBottom 10 lowest-ranked emails:")
    bottom = df.nlargest(10, "semantic_priority_rank")[
        ["subject", "category", "priority", "spam_score", "semantic_priority_score"]
    ]
    print(bottom.to_string(index=False))

    print("\nAgreement check: semantic ranking vs. existing heuristic 'priority' label")
    print("Mean semantic_priority_score by heuristic priority bucket:")
    print(df.groupby("priority")["semantic_priority_score"].mean().round(1).sort_values(ascending=False))

    if "priority_score" in df.columns:
        corr = df["semantic_priority_score"].corr(df["priority_score"])
        print(f"\nCorrelation with heuristic priority_score: {corr:.3f}")
    else:
        print("\nSkipping correlation: 'priority_score' column not found in dataframe.")

    df.to_csv("emails_with_semantic_priority.csv", index=False)
    print("\nSaved full results to emails_with_semantic_priority.csv")

    return df


if __name__ == "__main__":
    emails_dir = sys.argv[1] if len(sys.argv) > 1 else "generated_emails"
    labels_csv = sys.argv[2] if len(sys.argv) > 2 else "labels.csv"
    main(emails_dir, labels_csv)