# Mail Curator.ai

A comprehensive web application for email analysis, classification, and intelligence. Features include:
- **Email Classification**: Automatic categorization of emails into 8 categories (Shopping, Spam, Social, Finance, College, Job, Travel, Government)
- **Semantic Search**: Advanced search using sentence embeddings
- **Priority Scoring**: AI-powered email prioritization
- **Smart Summaries**: Extractive summarization of email content
- **Action Item Extraction**: Automatic detection of tasks and deadlines
- **Gmail Integration**: Sync with real Gmail inbox (optional)
- **RAG Chat**: AI-powered conversational email assistant

## Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd email_generator
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment**
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your configuration (optional for basic usage)
# Set GROQ_API_KEY for advanced RAG features
```

5. **Run the application**
```bash
python run.py
```

The application will start at `http://127.0.0.1:8000`

## Configuration

Edit the `.env` file to customize:

- **HOST/PORT**: Server host and port (default: 127.0.0.1:8000)
- **GROQ_API_KEY**: Optional API key for advanced AI features (get free key at https://console.groq.com)
- **Data paths**: Customize locations for email data and models
- **Gmail credentials**: For Gmail integration (see below)

## Gmail Integration (Optional)

To sync with your real Gmail inbox:

1. Create a Google Cloud Project at https://console.cloud.google.com/
2. Enable the Gmail API
3. Configure OAuth consent screen (External user type)
4. Create OAuth client ID (Desktop application type)
5. Download the credentials JSON file
6. Save it as `credentials.json` in the project directory
7. Set `CREDENTIALS_PATH=credentials.json` in `.env`

Then use the `/api/sync` endpoint in the web UI to fetch your emails.

## Project Structure

```
email_generator/
├── run.py                      # Startup script - use this to run the app
├── dashboard_server.py         # FastAPI web server
├── requirements.txt            # Python dependencies
├── .env.example               # Environment configuration template
├── config.py                  # Email generation configuration
├── generator.py               # Synthetic email generator
├── intelligence_engine.py     # NLP analysis engine
├── gmail_fetcher.py          # Gmail API integration
├── cleaner.py                # Email text cleaning
├── features.py               # Feature extraction for classification
├── semantic_priority.py      # Semantic priority scoring
├── templates/                # Web UI templates
├── generated_emails/         # Synthetic email dataset (auto-generated)
└── gmail_emails/             # Real Gmail emails (optional)
```

## Web API Endpoints

- `GET /` - Web dashboard UI
- `GET /api/emails` - List all emails with metadata
- `GET /api/emails/{filename}` - Get detailed email analysis
- `GET /api/search` - Semantic search across emails
- `POST /api/chat` - AI-powered email assistant (RAG)
- `POST /api/sync` - Sync with Gmail (requires credentials)
- `POST /api/retrain` - Retrain classification model
- `GET /api/training-status` - Check training status

## Development

### Running in development mode
The default configuration runs with auto-reload enabled. Set `RELOAD=false` in `.env` for production.

### Generating synthetic emails
```bash
python generator.py
```
This creates 10,000 synthetic emails in `generated_emails/` with labels in `labels.csv`.

### Training the classifier
```bash
python retrain_on_real_mail.py
```
Trains a hybrid model on synthetic + real Gmail + SpamAssassin data.

## Cloud Deployment

### Vercel (Serverless)

**Free tier available with automatic SSL**

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel
```

Or connect your GitHub repository to Vercel for automatic deployments.

**Environment Variables in Vercel:**
- Set `HOST=0.0.0.0`
- Set `PORT=8000`
- Set `RELOAD=false`
- Add `GROQ_API_KEY` if using AI features

**Note:** Vercel has execution time limits. For heavy ML operations, consider using a dedicated server.

### Render (Free Tier)

```bash
# Sign up at render.com
# Connect your GitHub repository
# Render will auto-detect the Python project
# Set environment variables in the Render dashboard
```

**Render Requirements:**
- Add `start.sh` script: `python run.py`
- Set environment variables in Render dashboard

### Railway (Free Tier)

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway up
```

### Docker Deployment

For containerized deployment on any platform:

```bash
# Build the image
docker build -t email-intelligence .

# Run the container
docker run -p 8000:8000 \
  -e GROQ_API_KEY=your_key_here \
  -v $(pwd)/data:/app/data \
  email-intelligence
```

**Docker Compose (recommended):**
```bash
docker-compose up --build
```

## Troubleshooting

**spaCy model not found**: The startup script will automatically download the `en_core_web_sm` model on first run.

**Import errors**: Ensure all dependencies are installed with `pip install -r requirements.txt`.

**Gmail sync fails**: Verify your `credentials.json` is properly configured and the OAuth consent screen includes your email as a test user.

**Port already in use**: Change the `PORT` in `.env` file.

## License

[Your License Here]
