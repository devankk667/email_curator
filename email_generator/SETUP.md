# Quick Setup Guide

This guide will help you get the AI Email Intelligence System running in under 5 minutes.

## Option 1: Quick Start (Recommended)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
python run.py
```

That's it! The app will automatically:
- Create necessary directories
- Download required NLP models
- Start the web server at http://127.0.0.1:8000

## Option 2: With Virtual Environment (Best Practice)

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python run.py
```

## Option 3: Docker (Easiest for Production)

```bash
# Build and run with Docker Compose
docker-compose up --build
```

## Optional: Gmail Integration

If you want to sync with your real Gmail:

1. Go to https://console.cloud.google.com/
2. Create a project and enable Gmail API
3. Create OAuth credentials (Desktop app)
4. Download credentials.json
5. Place it in the project directory
6. Add `CREDENTIALS_PATH=credentials.json` to your `.env` file

## Optional: Advanced AI Features

For better AI-powered responses:

1. Get a free API key from https://console.groq.com
2. Add `GROQ_API_KEY=your_key_here` to your `.env` file

## Troubleshooting

**Port already in use?**
Edit `.env` and change `PORT=8000` to another port.

**spaCy model error?**
Run: `python -m spacy download en_core_web_sm`

**Import errors?**
Make sure you installed dependencies: `pip install -r requirements.txt`

## Next Steps

- Open http://127.0.0.1:8000 in your browser
- Generate synthetic emails: `python generator.py`
- Train the classifier: `python retrain_on_real_mail.py`
