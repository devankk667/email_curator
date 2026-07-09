# Project Report: Mail Curator.ai

Date examined: 2026-07-09

## Executive Summary

This repository contains **Mail Curator.ai**, a Python/FastAPI application for email intelligence. The project combines synthetic email generation, real Gmail ingestion, SpamAssassin corpus support, machine-learning classification, semantic search, priority scoring, local NLP extraction, and a browser-based dashboard.

The application is more than a prototype: it includes a runnable FastAPI server, a substantial front-end dashboard, saved model artifacts, a retraining pipeline, generated datasets, Gmail OAuth integration files, Docker/Vercel deployment configuration, and a cached SQLite metadata store. The strongest parts are the end-to-end pipeline design, hybrid training strategy, and local intelligence features. The main risks are operational/security hygiene, large generated artifacts living in the working tree, limited tests, deployment heaviness for serverless platforms, and some visible text encoding issues.

## Repository Layout

Top-level structure:

- `email_generator/`: Main application, models, datasets, dashboard, retraining code, Gmail integration, Docker/Vercel configuration.
- `Spam_Assassin/`: Raw SpamAssassin corpus split into `easy_ham`, `easy_ham_2`, `hard_ham`, `spam`, and `spam_2`.
- `.vercel/`: Vercel project metadata/cache area.
- `.gitignore`: Top-level ignore currently only ignores `.vercel`.

Important files inside `email_generator/`:

- `run.py`: Startup script; checks environment, downloads spaCy model if needed, creates directories, launches Uvicorn.
- `dashboard_server.py`: FastAPI application and API routes.
- `templates/index.html`: Single-page dashboard UI.
- `intelligence_engine.py`: Local NLP engine for summaries, action items, deadlines, entities, smart replies, and SQLite caching.
- `features.py`: TF-IDF and SentenceTransformer feature builders.
- `retrain_on_real_mail.py`: Hybrid retraining pipeline using synthetic, Gmail, and SpamAssassin data.
- `gmail_fetcher.py`, `gmail_auto_labeler.py`: Gmail import and heuristic labeling support.
- `spamassassin_loader.py`: SpamAssassin ingestion.
- `requirements.txt`: Runtime dependencies.
- `Dockerfile`, `docker-compose.yml`, `vercel.json`, `api/index.py`: Deployment support.
- `tests/test_cleaner.py`: Existing automated tests, focused on email cleaning.

## Functional Overview

The project aims to provide an intelligent email dashboard with:

- Email classification into eight categories: `Shopping`, `Spam`, `Social`, `Finance`, `College`, `Job`, `Travel`, and `Government`.
- Semantic search using `sentence-transformers`.
- Priority scoring using semantic and heuristic features.
- Local extractive summaries.
- Action item and deadline extraction using spaCy and regex/context rules.
- Entity extraction for organizations, locations, money amounts, and reference numbers.
- Gmail sync through Google OAuth.
- RAG-style chat over relevant emails, with optional Groq API support and local fallback.
- Background model retraining from the web API.

The API routes in `dashboard_server.py` include:

- `GET /`: Serves the dashboard.
- `GET /api/emails`: Lists emails with metadata.
- `GET /api/emails/{filename}`: Returns a single email plus intelligence metadata.
- `GET /api/search`: Runs embedding similarity search.
- `POST /api/chat`: Runs email-question answering using retrieved context.
- `POST /api/sync`: Starts Gmail sync in the background.
- `POST /api/retrain`: Starts hybrid classifier retraining.
- `GET /api/training-status`: Reports retraining state and model status.

## Data and Model Assets

Observed dataset and corpus sizes:

- Synthetic emails: `10,000` files in `email_generator/generated_emails`.
- Gmail emails: `100` files in `email_generator/gmail_emails`.
- SpamAssassin corpus:
  - `easy_ham`: `2,551`
  - `easy_ham_2`: `1,401`
  - `hard_ham`: `250`
  - `spam`: `501`
  - `spam_2`: `1,398`
  - Total raw corpus files: `6,101`

Generated and saved assets include:

- `classifier.joblib`
- `feature_builder.joblib`
- `gmail_intelligence.db`
- `email_embeddings.npy`
- `semantic_axes_centroids_cache.npy`
- Multiple CSV datasets and PNG analysis charts

The latest retraining report shows:

- Train size: `9,350`
- Test size: `1,650`
- Sources: `10,000` synthetic, `900` SpamAssassin, `100` Gmail
- Accuracy: `0.969`
- Macro F1: `0.966`

These are strong hold-out metrics, but they should be interpreted carefully because most training data is synthetic and Gmail labels are auto-generated heuristically.

## Architecture

The app follows a practical pipeline:

1. Raw `.eml` files are parsed by `loader.py`.
2. Email content is cleaned and normalized by `cleaner.py`.
3. Feature builders create either TF-IDF/engineered features or embedding/engineered/domain features.
4. A Logistic Regression classifier predicts email category.
5. Semantic priority scoring ranks emails.
6. `LocalIntelligenceEngine` extracts summaries, entities, actions, deadlines, and smart replies.
7. Results are cached in SQLite to avoid recomputing metadata.
8. FastAPI exposes the data to a single-page dashboard.

This is a reasonable architecture for a local-first ML web app. The code also uses lazy/model persistence patterns to avoid retraining on every startup when saved artifacts exist.

## Strengths

- End-to-end implementation exists across ingestion, cleaning, training, inference, dashboard, and deployment.
- The retraining pipeline uses multiple data sources instead of relying only on synthetic data.
- The feature system is more mature than a basic bag-of-words classifier: it includes embeddings, engineered signals, and sender-domain features.
- SQLite caching avoids repeated expensive NLP work.
- FastAPI background tasks make Gmail sync and retraining accessible from the UI.
- The README and SETUP guide are useful and approachable.
- Docker, Docker Compose, Vercel, and Render/Railway guidance are present.
- The project includes at least one focused test suite for the cleaner.

## Risks and Issues

### Security and Privacy

Sensitive local files are present in the project directory:

- `.env`
- `credentials.json`
- `token.json`
- `gmail_intelligence.db`
- real Gmail email files

The inner `email_generator/.gitignore` correctly lists many of these, but the top-level `.gitignore` only ignores `.vercel`. The inspected `git ls-files` output did not show those sensitive files as tracked, which is good, but the root ignore policy should still be strengthened because the repository root is the Git root.

### Large Generated Artifacts

The working tree contains many large generated artifacts, including CSVs, NPY files, model files, images, a SQLite DB, and cached deployment files. This makes the project harder to clone, review, deploy, and back up. These should generally be ignored or moved to an artifact/data storage workflow unless they are intentionally versioned.

### Limited Test Coverage

Only `tests/test_cleaner.py` was found. Core behavior in classification, API routes, Gmail sync, retraining, semantic search, and RAG fallback does not appear to have automated test coverage.

### Deployment Fit

The app depends on heavy packages such as `sentence-transformers`, `spacy`, scikit-learn, and potentially PyTorch. This can be difficult on Vercel serverless due to cold starts, package size, memory use, and execution time limits. The README notes this, and a long-running server platform is likely a better production target.

### Runtime Performance

`/api/search` and `/api/chat` recompute embeddings for the active dataframe at request time. That is acceptable for small inboxes but will become slow for larger Gmail imports. Precomputed embedding caches would make search and chat much more responsive.

### Encoding Problems

Several files display mojibake characters, for example `â€”`, `âœ“`, and tree glyph corruption in README/code comments. This suggests UTF-8 content has been read or saved with the wrong encoding somewhere. It does not necessarily break runtime behavior, but it hurts documentation quality and polish.

### Evaluation Caveats

The reported metrics are high, but the combined training set is dominated by synthetic data. SpamAssassin ham is mapped into non-spam categories using keyword rules, and Gmail labels are heuristic. A manually labeled real-world validation set would provide a more reliable measure of real performance.

## Recommendations

1. Strengthen root `.gitignore` to cover `.env`, OAuth tokens, databases, generated emails, Gmail data, model binaries, embeddings, CSV datasets, generated plots, and Vercel caches.
2. Move large artifacts and private data out of the Git working tree or into a clearly documented local `data/` and `artifacts/` workflow.
3. Add tests for API endpoints, loader behavior, retraining smoke tests, model loading, semantic search, and cache behavior.
4. Add a small manually labeled real-email validation set for realistic evaluation.
5. Precompute and persist embeddings for Gmail/synthetic inboxes to avoid request-time embedding work.
6. Make Gmail sync and retraining jobs more observable, with progress, failure details, and logs exposed safely.
7. Fix UTF-8 encoding issues in documentation and comments.
8. Prefer Render/Railway/Fly.io/a VM/Docker host for production deployment over Vercel if full ML features are required.
9. Add privacy documentation explaining what data is stored locally, what is sent to Groq if enabled, and how to delete local caches/tokens.
10. Consider separating the project into clearer layers: `app/`, `ml/`, `data_pipeline/`, `scripts/`, `tests/`, and `artifacts/`.

## Overall Assessment

Mail Curator.ai is a solid applied ML application with a coherent product direction and a mostly complete local workflow. It already demonstrates meaningful functionality: Gmail ingestion, classification, semantic search, local intelligence extraction, priority scoring, and a usable dashboard. The next phase should focus less on adding features and more on hardening: data hygiene, privacy safeguards, testing, performance, deployment reliability, and real-world evaluation.

In its current state, the project is well suited for local experimentation, demos, and continued development. With cleanup and production hardening, it could become a credible personal email intelligence tool.
