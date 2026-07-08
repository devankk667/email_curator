import os
import sys
import sqlite3
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer

from cleaner import clean_email_body
from loader import load_emails
from features import FeatureBuilder

DB_PATH = os.environ.get("DB_PATH", "gmail_intelligence.db")

class LocalIntelligenceEngine:
    def __init__(self):
        print("Initializing Local NLP Intelligence Engine...")
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading 'en_core_web_sm' model...")
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
            
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS email_metadata (
                filename TEXT PRIMARY KEY,
                summary TEXT,
                action_items TEXT,
                deadlines TEXT,
                entities TEXT,
                smart_replies TEXT,
                predicted_category TEXT,
                priority_score REAL,
                priority_level TEXT
            )
        """)
        conn.commit()
        conn.close()

    def get_cached_metadata(self, filename: str) -> dict:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT * FROM email_metadata WHERE filename = ?", (filename,))
        row = c.fetchone()
        conn.close()
        
        if row:
            return {
                "filename": row[0],
                "summary": row[1],
                "action_items": json.loads(row[2]),
                "deadlines": json.loads(row[3]),
                "entities": json.loads(row[4]),
                "smart_replies": json.loads(row[5]),
                "predicted_category": row[6],
                "priority_score": row[7],
                "priority_level": row[8]
            }
        return None

    def clear_cache(self):
        """Drop and recreate the email_metadata table.

        Called after retraining the classifier so that all emails are
        re-analyzed with the new model's category predictions.  Without
        this, the old (wrong) predictions would stay cached indefinitely
        because get_cached_metadata() returns the first hit it finds.
        """
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("DROP TABLE IF EXISTS email_metadata")
        conn.commit()
        conn.close()
        self._init_db()
        print("Cache cleared -- all emails will be re-analyzed on next access.")

    def save_to_cache(self, filename: str, meta: dict):
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            INSERT OR REPLACE INTO email_metadata 
            (filename, summary, action_items, deadlines, entities, smart_replies, predicted_category, priority_score, priority_level)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            filename,
            meta["summary"],
            json.dumps(meta["action_items"]),
            json.dumps(meta["deadlines"]),
            json.dumps(meta["entities"]),
            json.dumps(meta["smart_replies"]),
            meta["predicted_category"],
            meta["priority_score"],
            meta["priority_level"]
        ))
        conn.commit()
        conn.close()

    def extract_summary(self, text: str, num_sentences: int = 3) -> str:
        """Generates a local extractive summary using sentence-to-document similarity."""
        if not text.strip():
            return "Empty email body."
            
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]
        
        if not sentences:
            return text[:200] + "..." if len(text) > 200 else text

        if len(sentences) <= num_sentences:
            return " ".join(sentences)

        # Compute document & sentence embeddings
        doc_emb = self.embed_model.encode(text, normalize_embeddings=True)
        sent_embs = self.embed_model.encode(sentences, normalize_embeddings=True)
        
        # Calculate cosine similarities (dot product since normalized)
        similarities = sent_embs @ doc_emb
        
        # Get index of most similar sentences
        top_indices = np.argsort(similarities)[-num_sentences:]
        # Sort indices to preserve original order
        top_indices.sort()
        
        summary_sents = [sentences[idx] for idx in top_indices]
        return " ".join(summary_sents)

    def extract_action_items(self, text: str) -> list:
        """Find action items/imperative sentences using SpaCy dependency parsing."""
        if not text.strip():
            return []
            
        doc = self.nlp(text)
        action_items = []
        
        action_verbs = {
            "submit", "pay", "send", "review", "attend", "confirm", "update", "call", 
            "schedule", "renew", "bring", "verify", "complete", "register", "check", "reply"
        }
        
        for sent in doc.sents:
            # Clean sentence text
            s_text = sent.text.strip()
            if len(s_text) < 8:
                continue
                
            is_action = False
            # Rule 1: Check for imperative verbs (ROOT verb in base form with no subject)
            for token in sent:
                if token.dep_ == "ROOT" and token.pos_ == "VERB":
                    # Check if there is an nominal subject (nsubj) attached to ROOT
                    has_subj = any(t.dep_ in ("nsubj", "nsubjpass") for t in token.children)
                    if not has_subj or token.lemma_.lower() in action_verbs:
                        if token.lemma_.lower() in action_verbs:
                            is_action = True
                            break
                            
            # Rule 2: Sentences starting with directive adverbs/modals
            if not is_action:
                lower_sent = s_text.lower()
                if any(lower_sent.startswith(x) for x in ["please", "kindly", "make sure", "ensure", "don't forget"]):
                    is_action = True
                elif any(x in lower_sent for x in ["need to", "have to", "require you to", "should"]):
                    is_action = True
                    
            if is_action:
                # Deduplicate and trim
                clean_item = s_text.replace("\n", " ").strip()
                if clean_item not in action_items:
                    action_items.append(clean_item)
                    
        return action_items[:5]

    def extract_deadlines(self, text: str) -> list:
        """Extract date/time entities and check if they are deadlines."""
        if not text.strip():
            return []
            
        doc = self.nlp(text)
        deadlines = []
        
        # Find dates and times using SpaCy NER
        date_ents = [ent.text.strip() for ent in doc.ents if ent.label_ in ("DATE", "TIME")]
        
        # Regex search for context matching deadline indicators
        deadline_keywords = ["due", "deadline", "by", "before", "until", "expiry", "expire", "scheduled"]
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            # If a deadline keyword is in the sentence, look for the date in this sentence
            if any(kw in sent_text.lower() for kw in deadline_keywords):
                for ent in sent.ents:
                    if ent.label_ in ("DATE", "TIME") and len(ent.text.strip()) > 3:
                        desc = f"{ent.text.strip()} (Context: {sent_text[:100]}...)"
                        if desc not in deadlines:
                            deadlines.append({
                                "date": ent.text.strip(),
                                "context": sent_text
                            })
                            
        # If no explicit keyword but dates exist, list them as potential key dates
        if not deadlines and date_ents:
            for d in date_ents[:3]:
                if len(d) > 4:
                    deadlines.append({
                        "date": d,
                        "context": "Mentioned in email"
                    })
                    
        return deadlines

    def extract_entities(self, text: str) -> dict:
        """Extract structured variables like amounts, companies, and identifiers."""
        if not text.strip():
            return {"organizations": [], "locations": [], "amounts": [], "reference_numbers": []}
            
        doc = self.nlp(text)
        orgs = list(set([ent.text.strip() for ent in doc.ents if ent.label_ == "ORG"]))
        locations = list(set([ent.text.strip() for ent in doc.ents if ent.label_ in ("GPE", "LOC")]))
        amounts = list(set([ent.text.strip() for ent in doc.ents if ent.label_ == "MONEY"]))
        
        # Regex matching for IDs/Reference numbers
        ref_patterns = [
            r"\b(?:invoice|booking|order|ticket|id|confirmation|ref)\s*(?:no|num|number)?\s*[:#-]?\s*([a-zA-Z0-9-]{4,15})\b"
        ]
        
        ref_nums = []
        for pattern in ref_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for m in matches:
                if m not in ref_nums and not m.isdigit(): # avoid plain numbers
                    ref_nums.append(m)
                    
        return {
            "organizations": orgs[:4],
            "locations": locations[:4],
            "amounts": amounts[:3],
            "reference_numbers": ref_nums[:3]
        }

    def generate_smart_replies(self, text: str, category: str) -> list:
        """Returns standard response options based on category and action cues."""
        replies = []
        text_lower = text.lower()
        
        if category == "Travel":
            replies = ["Thank you! Please add this to my calendar.", "Got it, looking forward to the trip.", "Is there a booking confirmation number?"]
        elif category == "Finance":
            replies = ["Payment has been processed.", "I will check my bank statement.", "Thank you for the invoice receipt."]
        elif category == "Job":
            replies = ["Thank you for the update. I am very interested.", "Please let me know the next steps.", "I will review the details and get back to you soon."]
        elif category == "College":
            replies = ["Thank you, I will make a note of this deadline.", "Could you clarify the assignment requirements?", "I will submit it before the deadline."]
        elif category in ("Shopping", "Social"):
            replies = ["Thanks for sharing!", "Unsubscribe from these emails.", "Looks great, thank you."]
        else:
            replies = ["Received, thank you.", "I will get back to you shortly.", "Thanks for the update."]
            
        return replies

    def analyze_email(self, filepath: Path, predicted_category="Unclassified", priority_score=50.0) -> dict:
        """Parse raw email, run all local extraction modules, and cache/return."""
        filename = filepath.name
        
        # Check cache first
        cached = self.get_cached_metadata(filename)
        if cached:
            return cached

        # Parse & clean raw email content
        from loader import _parse_single_eml
        try:
            parsed = _parse_single_eml(filepath)
            body_text = parsed.get("body_text", "")
            body_html = parsed.get("body_html", "")
            
            # Clean body for NLP parsing
            cleaned = clean_email_body(body_text, body_html)
            clean_text = cleaned["clean_text"]
            search_text = cleaned["search_text"]
        except Exception as e:
            print(f"Error parsing {filename}: {e}")
            clean_text = ""
            search_text = ""
            
        # Extractor runs
        summary = self.extract_summary(clean_text)
        action_items = self.extract_action_items(clean_text)
        deadlines = self.extract_deadlines(clean_text)
        entities = self.extract_entities(clean_text)
        smart_replies = self.generate_smart_replies(clean_text, predicted_category)
        
        priority_level = "Medium"
        if priority_score >= 70.0:
            priority_level = "High"
        elif priority_score < 30.0:
            priority_level = "Low"

        meta = {
            "filename": filename,
            "summary": summary,
            "action_items": action_items,
            "deadlines": deadlines,
            "entities": entities,
            "smart_replies": smart_replies,
            "predicted_category": predicted_category,
            "priority_score": float(priority_score),
            "priority_level": priority_level
        }
        
        # Cache results
        self.save_to_cache(filename, meta)
        return meta

if __name__ == "__main__":
    # Test execution
    engine = LocalIntelligenceEngine()
    test_text = """
    Hi Swamiprasad,
    
    Thanks for applying to Google. We reviewed your resume and would like to invite you for an interview.
    Please submit your availability for next week by Friday, July 10th. 
    The interview will be scheduled via Google Meet. Let us know if you need any accommodations.
    
    Best,
    Google Recruiting Team
    Invoice reference: GOOG-2026-992
    """
    
    print("\n--- Test Text ---")
    print(test_text)
    
    print("\n--- Extracted Summary ---")
    print(engine.extract_summary(test_text))
    
    print("\n--- Extracted Action Items ---")
    print(engine.extract_action_items(test_text))
    
    print("\n--- Extracted Deadlines ---")
    print(engine.extract_deadlines(test_text))
    
    print("\n--- Extracted Entities ---")
    print(engine.extract_entities(test_text))
    
    print("\n--- Smart Replies ---")
    print(engine.generate_smart_replies(test_text, "Job"))
