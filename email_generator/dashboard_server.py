import os
import sys
import json
import sqlite3
import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

# Load .env file if present (optional dependency -- safe to skip if not installed)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; rely on shell environment variables

# Import local modules
from loader import load_emails
from cleaner import clean_dataframe, clean_email_body
from features import EmbeddingFeatureBuilder
from semantic_priority import get_or_compute_centroids, compute_semantic_scores, compute_ranking_score
from intelligence_engine import LocalIntelligenceEngine

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app_instance):
    """Modern FastAPI lifespan handler replacing the deprecated @app.on_event."""
    init_models()
    yield
    # Shutdown: nothing to clean up right now


app = FastAPI(title="Mail Curator.ai", lifespan=lifespan)

# Ensure templates directory exists
Path("templates").mkdir(exist_ok=True)

# Global variables/models
nlp_engine = None
embed_model = None
feature_builder = None
classifier = None
synthetic_df = None
gmail_df = None

# Cache paths for models and embeddings (from environment variables)
SYNTHETIC_DIR = os.environ.get("SYNTHETIC_DIR", "generated_emails")
SYNTHETIC_LABELS = os.environ.get("SYNTHETIC_LABELS", "labels.csv")
GMAIL_DIR = os.environ.get("GMAIL_DIR", "gmail_emails")
GMAIL_DB = os.environ.get("DB_PATH", "gmail_intelligence.db")
FEATURE_BUILDER_PATH = os.environ.get("FEATURE_BUILDER_PATH", "feature_builder.joblib")
CLASSIFIER_PATH = os.environ.get("CLASSIFIER_PATH", "classifier.joblib")

# Background retrain status tracker
retrain_status = {
    "running": False,
    "last_run": None,
    "result": None,
    "error": None,
}

class ChatRequest(BaseModel):
    message: str
    inbox_type: str = "gmail"  # "gmail" or "synthetic"

def init_models():
    """Initializes SpaCy, SentenceTransformer, and trains the baseline classifier."""
    global nlp_engine, embed_model, feature_builder, classifier, synthetic_df, gmail_df
    
    # Initialize local NLP intelligence engine
    nlp_engine = LocalIntelligenceEngine()
    embed_model = nlp_engine.embed_model
    
    # Train/Load Category Classifier
    logger.info("Setting up category classifier...")

    # Prefer the persisted hybrid-trained model (retrain_on_real_mail.py output)
    if Path(CLASSIFIER_PATH).exists() and Path(FEATURE_BUILDER_PATH).exists():
        try:
            logger.info(f"Loading pre-trained classifier from {CLASSIFIER_PATH}...")
            feature_builder = EmbeddingFeatureBuilder.load(FEATURE_BUILDER_PATH)
            classifier = joblib.load(CLASSIFIER_PATH)
            logger.info(f"Loaded classifier with classes: {list(classifier.classes_)}")
        except Exception as e:
            logger.error(f"Error loading saved classifier: {e} — falling back to training.")
            feature_builder = None
            classifier = None

    # Fallback: train on synthetic data only if no saved model exists
    if classifier is None and Path(SYNTHETIC_DIR).exists() and Path(SYNTHETIC_LABELS).exists():
        try:
            logger.info("No saved model found. Training on synthetic dataset...")
            df = load_emails(SYNTHETIC_DIR, SYNTHETIC_LABELS)
            df = clean_dataframe(df)
            synthetic_df = df

            fb = EmbeddingFeatureBuilder()
            X_train, y_train = fb.fit_transform(df)

            lr = LogisticRegression(max_iter=2000, C=2.0, class_weight="balanced")
            lr.fit(X_train, y_train)

            feature_builder = fb
            classifier = lr
            logger.info("Category classifier trained on synthetic data.")
            logger.info("TIP: Run 'python retrain_on_real_mail.py' to train on real Gmail data.")
        except Exception as e:
            logger.error(f"Error training category classifier: {e}")
    elif classifier is None:
        logger.warning("No classifier could be loaded or trained.")

    # Load Gmail emails if they exist
    load_gmail_dataframe()

def load_gmail_dataframe():
    """Load and process Gmail emails if they exist."""
    global gmail_df
    if Path(GMAIL_DIR).exists():
        try:
            # Load raw emails (unlabeled)
            df = load_emails(GMAIL_DIR)
            if len(df) > 0:
                df = clean_dataframe(df)
                
                # Merge spam score and other heuristics (required by FeatureBuilder)
                if "spam_score" not in df.columns:
                    df["spam_score"] = 0.0
                if "has_attachment" not in df.columns:
                    df["has_attachment"] = df["num_attachments"] > 0
                if "is_thread_reply" not in df.columns:
                    df["is_thread_reply"] = df["subject"].str.lower().str.startswith(("re:", "fwd:", "fw:"))
                if "priority" not in df.columns:
                    df["priority"] = "Medium"

                # Ensure category column exists
                if "category" not in df.columns:
                    df["category"] = "Unclassified"

                # Classify categories using trained model
                if classifier and feature_builder:
                    # FeatureBuilder.transform expects a category column to exist
                    X, _ = feature_builder.transform(df)
                    df["category"] = classifier.predict(X)
                else:
                    df["category"] = "Job"  # default fallback if no classifier
                    
                # Compute semantic priorities
                texts = df["search_text"].tolist()
                embeddings = embed_model.encode(texts, normalize_embeddings=True)
                centroids = get_or_compute_centroids(embed_model)
                raw_semantic_scores = compute_semantic_scores(embeddings, centroids)
                
                df = compute_ranking_score(df, raw_semantic_scores)
                
                # Cache results using intelligence engine
                for _, row in df.iterrows():
                    filepath = Path(GMAIL_DIR) / row["filename"]
                    nlp_engine.analyze_email(filepath, row["category"], row["semantic_priority_score"])
                    
                gmail_df = df
                logger.info(f"Loaded {len(df)} Gmail emails.")
            else:
                gmail_df = None
        except Exception as e:
            logger.error(f"Error loading Gmail dataframe: {e}")
            gmail_df = None
    else:
        gmail_df = None

def get_active_df(inbox_type: str):
    global gmail_df, synthetic_df
    if inbox_type == "gmail":
        if gmail_df is None or len(gmail_df) == 0:
            load_gmail_dataframe()
        return gmail_df
    return synthetic_df

# Removed: @app.on_event("startup") -- replaced by lifespan above


@app.get("/api/emails")
def get_emails(inbox_type: str = "gmail"):
    df = get_active_df(inbox_type)
    if df is None or len(df) == 0:
        return []
        
    # Return basic meta fields
    emails_list = []
    # If the df is synthetic, it might have 'priority' instead of 'semantic_priority_score'
    priority_score_col = "semantic_priority_score" if "semantic_priority_score" in df.columns else "priority_score"
    priority_level_col = "priority"
    
    for _, row in df.iterrows():
        # Get cached meta for level/score check
        cached = nlp_engine.get_cached_metadata(row["filename"])
        p_score = cached["priority_score"] if cached else float(row.get(priority_score_col, 50))
        p_level = cached["priority_level"] if cached else str(row.get(priority_level_col, "Medium"))
        
        emails_list.append({
            "filename": row["filename"],
            "subject": row["subject"],
            "from_name": row.get("from_name", "Unknown"),
            "from_address": row.get("from_address", ""),
            "date": str(row["date"]),
            "category": row["category"],
            "priority_score": p_score,
            "priority_level": p_level,
            "spam_score": float(row.get("spam_score", 0))
        })
        
    # Sort by priority score descending
    emails_list.sort(key=lambda x: x["priority_score"], reverse=True)
    return emails_list

@app.get("/api/emails/{filename}")
def get_email_detail(filename: str, inbox_type: str = "gmail"):
    df = get_active_df(inbox_type)
    if df is None:
        raise HTTPException(status_code=404, detail="Inbox not found")
        
    row = df[df["filename"] == filename]
    if len(row) == 0:
        raise HTTPException(status_code=404, detail="Email not found")
        
    row = row.iloc[0]
    
    # Run intelligence engine analysis (computes/loads cached metadata)
    emails_dir = GMAIL_DIR if inbox_type == "gmail" else SYNTHETIC_DIR
    filepath = Path(emails_dir) / filename
    
    meta = nlp_engine.analyze_email(
        filepath, 
        row["category"], 
        row.get("semantic_priority_score", row.get("priority_score", 50))
    )
    
    return {
        "filename": filename,
        "subject": row["subject"],
        "from_name": row.get("from_name", "Unknown"),
        "from_address": row.get("from_address", ""),
        "to": row.get("to", ""),
        "date": str(row["date"]),
        "body_text": row["body_text"],
        "body_html": row.get("body_html", ""),
        "intelligence": meta
    }

@app.get("/api/search")
def search_emails(query: str, inbox_type: str = "gmail"):
    df = get_active_df(inbox_type)
    if df is None or len(df) == 0:
        return []
        
    # Embed search query
    query_emb = embed_model.encode(query, normalize_embeddings=True)
    
    # Get embeddings for current dataframe
    texts = df["search_text"].tolist()
    embs = embed_model.encode(texts, normalize_embeddings=True)
    
    # Cosine similarities
    similarities = embs @ query_emb
    
    # Find matching indices
    results = []
    for idx, score in enumerate(similarities):
        if score > 0.15:  # threshold
            row = df.iloc[idx]
            cached = nlp_engine.get_cached_metadata(row["filename"])
            results.append({
                "filename": row["filename"],
                "subject": row["subject"],
                "from_name": row.get("from_name", "Unknown"),
                "date": str(row["date"]),
                "category": row["category"],
                "similarity": float(score),
                "summary": cached["summary"] if cached else row["body_text"][:100] + "..."
            })
            
    # Sort by similarity score descending
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:10]

@app.post("/api/chat")
def chat_rag(req: ChatRequest):
    df = get_active_df(req.inbox_type)
    if df is None or len(df) == 0:
        return {"answer": "No emails found to perform search.", "citations": []}
        
    # Search for top matches to serve as context
    query_emb = embed_model.encode(req.message, normalize_embeddings=True)
    texts = df["search_text"].tolist()
    embs = embed_model.encode(texts, normalize_embeddings=True)
    similarities = embs @ query_emb
    
    # Get top 3 matching emails
    top_indices = np.argsort(similarities)[-3:][::-1]
    context_emails = []
    citations = []
    
    for idx in top_indices:
        score = similarities[idx]
        if score > 0.18:
            row = df.iloc[idx]
            cached = nlp_engine.get_cached_metadata(row["filename"])
            summary = cached["summary"] if cached else row["body_text"][:100]
            context_emails.append(f"Subject: {row['subject']}\nFrom: {row.get('from_name', '')} <{row.get('from_address', '')}>\nDate: {row['date']}\nContent: {row['body_text']}")
            citations.append({
                "filename": row["filename"],
                "subject": row["subject"],
                "from_name": row.get("from_name", "Unknown"),
                "similarity": float(score)
            })

    # Groq-based RAG if API Key is set
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if groq_api_key and context_emails:
        try:
            from groq import Groq
            client = Groq(api_key=groq_api_key)
            context_text = "\n\n---\n\n".join(context_emails)
            prompt = f"""
You are an AI Email Assistant. Answer the user's question using ONLY the provided email context.
If the context doesn't contain the answer, say "I cannot find the answer in the provided emails."

Emails context:
{context_text}

Question: {req.message}
Answer:"""
            completion = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=500
            )
            return {
                "answer": completion.choices[0].message.content.strip(),
                "citations": citations
            }
        except Exception as e:
            logger.error(f"Groq RAG Error: {e}")
            
    # Local RAG fallback (extractive matching of sentences)
    if not context_emails:
        return {
            "answer": "I couldn't find any relevant emails matching your question in the inbox.",
            "citations": []
        }
        
    # Extractive answer fallback: compile summaries of the most relevant emails
    answer_parts = []
    for cit in citations[:2]:
        cached = nlp_engine.get_cached_metadata(cit["filename"])
        answer_parts.append(f"Regarding '{cit['subject']}' from {cit['from_name']}: {cached['summary']}")
        
    answer = "Based on the most relevant emails:\n- " + "\n- ".join(answer_parts)
    return {
        "answer": answer,
        "citations": citations
    }

@app.post("/api/sync")
def sync_gmail(background_tasks: BackgroundTasks):
    """Triggers the gmail_fetcher to retrieve fresh emails and reloads dataframe."""
    if not Path("credentials.json").exists():
        raise HTTPException(
            status_code=400,
            detail="credentials.json not found in email_generator directory. Please add it first."
        )

    def sync_job():
        logger.info("Starting background Gmail sync job...")
        try:
            from gmail_fetcher import fetch_and_save_emails
            fetch_and_save_emails(max_results=50)
            load_gmail_dataframe()
            logger.info("Background Gmail sync completed successfully.")
        except Exception as e:
            logger.error(f"Error in background Gmail sync: {e}")

    background_tasks.add_task(sync_job)
    return {"status": "Sync started in background."}


@app.post("/api/retrain")
def retrain_model(background_tasks: BackgroundTasks):
    """
    Triggers a full model retrain using the hybrid pipeline
    (synthetic + real Gmail + SpamAssassin).  Runs in the background
    and reloads the classifier when done.  Poll /api/training-status
    to check progress.
    """
    global retrain_status
    if retrain_status["running"]:
        return {"status": "already_running", "message": "Retraining is already in progress."}

    def retrain_job():
        global nlp_engine, feature_builder, classifier, gmail_df, retrain_status
        retrain_status["running"] = True
        retrain_status["error"] = None
        retrain_status["result"] = None
        logger.info("Starting background hybrid retrain job...")
        try:
            from retrain_on_real_mail import retrain
            result = retrain()
            retrain_status["result"] = result
            logger.info(f"Retrain complete: accuracy={result['accuracy']:.3f}, macro_f1={result['macro_f1']:.3f}")

            # Reload newly trained models into memory
            logger.info("Reloading updated models...")
            feature_builder = EmbeddingFeatureBuilder.load(FEATURE_BUILDER_PATH)
            classifier = joblib.load(CLASSIFIER_PATH)

            # Clear SQLite cache so emails get re-analyzed with new predictions
            if nlp_engine:
                nlp_engine.clear_cache()
                logger.info("NLP cache cleared.")

            # Reload Gmail emails with new classifier
            load_gmail_dataframe()
            logger.info("Gmail dataframe reloaded with new classifier.")

        except Exception as e:
            logger.error(f"Retrain job failed: {e}")
            retrain_status["error"] = str(e)
        finally:
            retrain_status["running"] = False
            import datetime
            retrain_status["last_run"] = datetime.datetime.now().isoformat()

    background_tasks.add_task(retrain_job)
    return {"status": "started", "message": "Hybrid retraining started in background. Poll /api/training-status for updates."}


@app.get("/api/training-status")
def training_status():
    """Returns current retrain job status and last result."""
    return {
        "running": retrain_status["running"],
        "last_run": retrain_status["last_run"],
        "error": retrain_status["error"],
        "result": retrain_status["result"],
        "model_loaded": classifier is not None,
        "model_path": CLASSIFIER_PATH if Path(CLASSIFIER_PATH).exists() else None,
    }

@app.get("/")
def get_dashboard():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    reload = os.environ.get("RELOAD", "true").lower() == "true"
    uvicorn.run("dashboard_server:app", host=host, port=port, reload=reload)
